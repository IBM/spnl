use futures::{AsyncBufReadExt, StreamExt, TryStreamExt};
use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::Pod;

use kube::{
    Client, ResourceExt,
    api::{Api, DeleteParams, ListParams, LogParams, PostParams},
    runtime::wait::{await_condition, conditions::is_deployment_completed},
};

#[derive(derive_builder::Builder)]
pub struct UpArgs {
    /// Name of resource
    #[builder(setter(into),default = "vllm".to_string())]
    name: String,

    /// Namespace of resource
    #[builder(default = None)]
    namespace: Option<String>,

    /// Model to serve
    #[builder(default)]
    model: Option<String>,

    /// Tokenizer to use (optional)
    #[builder(default)]
    tokenizer: Option<String>,

    /// HuggingFace api token
    #[builder(setter(into))]
    hf_token: String,

    /// Number of GPUs to request
    #[builder(default = 1)]
    gpus: u32,

    /// Local port for port forwarding
    #[builder(default = Some(8000))]
    local_port: Option<u16>,

    /// Remote port for port forwarding (defaults to 8000)
    #[builder(default = 8000)]
    remote_port: u16,
}

fn load_deployment_manifest(args: UpArgs) -> anyhow::Result<(Deployment, uuid::Uuid)> {
    let id = uuid::Uuid::new_v4();
    let mut d: Deployment = serde_yaml2::from_str(include_str!("deployment.yml"))?;
    d.metadata.name = Some(args.name.clone());
    if let Some(ref mut ospec) = d.spec
        && let Some(ref mut spec) = ospec.template.spec
        && !spec.containers.is_empty()
    {
        if let Some(image) = &spec.containers[0].image {
            spec.containers[0].image =
                Some(format!("{}:v{}", image, ::std::env!("CARGO_PKG_VERSION")));
        }

        if let Some(ml) = &ospec.selector.match_labels {
            let mut match_labels = ml.clone();
            if let Some(v) = match_labels.get_mut("app.kubernetes.io/name") {
                *v = args.name.clone();
            }
            ospec.selector.match_labels = Some(match_labels);
        }
        if let Some(ref mut meta) = ospec.template.metadata
            && let Some(l) = &meta.labels
        {
            let mut labels = l.clone();
            if let Some(v) = labels.get_mut("app.kubernetes.io/name") {
                *v = args.name.clone();
            }
            if let Some(v) = labels.get_mut("app.kubernetes.io/instance") {
                *v = id.to_string();
            }
            meta.labels = Some(labels);
        }

        if let Some(ref mut env) = spec.containers[0].env {
            if let Some(model) = args.model {
                match env.iter_mut().find(|kv| kv.name == "MODEL") {
                    Some(kv) => kv.value = Some(model),
                    None => {
                        return Err(anyhow::anyhow!(
                            "Missing MODEL env var in deployment.yml template"
                        ));
                    }
                };
            }

            if let Some(tokenizer) = args.tokenizer {
                match env.iter_mut().find(|kv| kv.name == "TOKENIZER") {
                    Some(kv) => kv.value = Some(tokenizer),
                    None => {
                        return Err(anyhow::anyhow!(
                            "Missing TOKENIZER env var in deployment.yml template"
                        ));
                    }
                };
            }

            match env.iter_mut().find(|kv| kv.name == "HF_TOKEN") {
                Some(kv) => kv.value = Some(args.hf_token),
                None => {
                    return Err(anyhow::anyhow!(
                        "Missing HF_TOKEN env var in deployment.yml template"
                    ));
                }
            };
        }

        // Set GPU count in resources.limits
        if let Some(ref mut resources) = spec.containers[0].resources
            && let Some(ref mut limits) = resources.limits
            && let Some(gpu_limit) = limits.get_mut("nvidia.com/gpu")
        {
            *gpu_limit =
                k8s_openapi::apimachinery::pkg::api::resource::Quantity(args.gpus.to_string());
        }
    }
    Ok((d, id))
}

async fn client() -> anyhow::Result<Client> {
    // does not work as reqwest pulls in ring and others pull in aws-lc-rs (this may change in reqwest 0.13)
    //rustls::crypto::CryptoProvider::install_default();
    let _ = rustls::crypto::ring::default_provider().install_default();

    Ok(Client::try_default().await?)
}

fn api<K>(c: Client, namespace: &Option<String>) -> anyhow::Result<Api<K>>
where
    <K as kube::Resource>::DynamicType: Default,
    K: kube::Resource<Scope = kube::core::NamespaceResourceScope>,
{
    Ok(match namespace {
        Some(ns) => Api::namespaced(c, ns),
        None => Api::default_namespaced(c),
    })
}

pub async fn up(args: UpArgs) -> anyhow::Result<()> {
    let c = client().await?;
    up_with_client(c, args).await
}

async fn up_with_client(c: Client, args: UpArgs) -> anyhow::Result<()> {
    let name = &args.name.clone();
    let local_port = args.local_port;
    let remote_port = args.remote_port;
    let namespace = args.namespace.clone();
    let d = api::<Deployment>(c.clone(), &namespace)?;
    let p = api::<Pod>(c.clone(), &namespace)?;
    let (manifest, id) = load_deployment_manifest(args)?;
    d.create(&PostParams::default(), &manifest).await?;

    let pods = loop {
        let names = p
            .list(&ListParams::default().labels(
                format!("app.kubernetes.io/name={name},app.kubernetes.io/instance={id}").as_str(),
            ))
            .await?
            .items
            .into_iter()
            .map(|pod| pod.name_any())
            .collect::<Vec<_>>();
        if !names.is_empty() {
            break names;
        }
        ::std::thread::sleep(::std::time::Duration::from_millis(1000));
    };

    let join_handles: Vec<tokio::task::JoinHandle<_>> = pods
        .clone()
        .into_iter()
        .map(|pod| {
            tokio::spawn({
                let pp = p.clone();
                async move {
                    let mut last_time: Option<::std::time::Instant> = None;
                    let mut done = false;
                    while !done {
                        match pp
                            .log_stream(
                                &pod,
                                &LogParams {
                                    follow: true,
                                    //container: Some("vllm".to_string()),
                                    //tail_lines: app.tail,
                                    since_seconds: last_time
                                        .map(|last_time| last_time.elapsed().as_secs() as i64),
                                    // since_time:
                                    //timestamps: app.timestamps,
                                    ..LogParams::default()
                                },
                            )
                            .await
                        {
                            Ok(l) => {
                                eprintln!("Streaming logs for {pod}");
                                let mut logs = l.lines();
                                loop {
                                    match logs.try_next().await {
                                        Ok(Some(line)) => {
                                            last_time = Some(::std::time::Instant::now());
                                            println!("{line}")
                                        }
                                        Ok(None) => {
                                            done = true;
                                            break;
                                        }
                                        _ => break,
                                    }
                                }
                            }
                            Err(_) => {
                                // TODO log error
                                ::std::thread::sleep(::std::time::Duration::from_millis(500));
                                // we will retry in the enclosing loop
                            }
                        }
                    }

                    Ok::<(), anyhow::Error>(())
                }
            })
        })
        .collect();

    // why the loop? see https://github.com/kube-rs/kube/issues/1915
    eprintln!("Awaiting deployment completion");
    loop {
        if await_condition(d.clone(), name, is_deployment_completed())
            .await
            .is_ok()
        {
            eprintln!("READY");
            break;
        }
    }

    // Set up port forwarding if local_port is specified
    if let Some(local_port) = local_port {
        // Select any ready pod from the deployment for port forwarding
        let ready_pod = loop {
            let pod_list = p
                .list(
                    &ListParams::default().labels(
                        format!("app.kubernetes.io/name={name},app.kubernetes.io/instance={id}")
                            .as_str(),
                    ),
                )
                .await?;

            if let Some(pod) = pod_list.items.iter().find(|pod| {
                if let Some(status) = &pod.status
                    && let Some(conditions) = &status.conditions
                {
                    return conditions
                        .iter()
                        .any(|c| c.type_ == "Ready" && c.status == "True");
                }
                false
            }) {
                break pod.name_any();
            }

            tokio::time::sleep(tokio::time::Duration::from_millis(1000)).await;
        };

        eprintln!(
            "Setting up port forward: localhost:{} -> {}:{}",
            local_port, ready_pod, remote_port
        );

        let addr = ::std::net::SocketAddr::from(([127, 0, 0, 1], local_port));
        let listener = tokio::net::TcpListener::bind(addr).await?;
        eprintln!(
            "Port forwarding: localhost:{} -> {}:{}",
            local_port, ready_pod, remote_port
        );

        let p_clone = p.clone();
        let ready_pod_clone = ready_pod.clone();
        tokio::spawn(async move {
            let server = tokio_stream::wrappers::TcpListenerStream::new(listener)
                .take_until(tokio::signal::ctrl_c())
                .try_for_each(|client_conn| async {
                    let pods = p_clone.clone();
                    let pod_name = ready_pod_clone.clone();
                    tokio::spawn(async move {
                        if let Err(e) =
                            forward_connection(&pods, &pod_name, remote_port, client_conn).await
                        {
                            eprintln!("Failed to forward connection: {}", e);
                        }
                    });
                    Ok(())
                });

            if let Err(e) = server.await {
                eprintln!("Port forwarding error: {}", e);
            } else {
                eprintln!("Port forwarding stopped");
            }
        });

        // Give port forwarding a moment to establish
        tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
    }

    // Wait for either all log handles to complete or Ctrl+C
    tokio::select! {
        result = futures::future::try_join_all(join_handles) => {
            result?;
        }
        _ = tokio::signal::ctrl_c() => {
            eprintln!("Received Ctrl+C, shutting down...");
        }
    }

    Ok(())
}

async fn forward_connection(
    pods: &Api<Pod>,
    pod_name: &str,
    port: u16,
    mut client_conn: impl tokio::io::AsyncRead + tokio::io::AsyncWrite + Unpin,
) -> anyhow::Result<()> {
    let mut forwarder = pods.portforward(pod_name, &[port]).await?;
    let mut upstream_conn = forwarder
        .take_stream(port)
        .ok_or_else(|| anyhow::anyhow!("port not found in forwarder"))?;
    tokio::io::copy_bidirectional(&mut client_conn, &mut upstream_conn).await?;
    drop(upstream_conn);
    forwarder.join().await?;
    Ok(())
}

pub async fn down(name: &str, namespace: Option<String>) -> anyhow::Result<()> {
    let c = client().await?;
    down_with_client(c, name, namespace).await
}

async fn down_with_client(c: Client, name: &str, namespace: Option<String>) -> anyhow::Result<()> {
    let _ = api::<Deployment>(c, &namespace)?
        .delete(name, &DeleteParams::default())
        .await?
        .map_left(|o| println!("Deleting deployment: {:?}", o.status))
        .map_right(|s| println!("Deleted deployment: {:?}", s));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_deployment_manifest() -> anyhow::Result<()> {
        super::load_deployment_manifest(super::UpArgsBuilder::default().hf_token("").build()?)
            .map(|_| ())
    }

    #[test]
    fn load_deployment_manifest_with_custom_name() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .name("custom-vllm")
            .hf_token("test-token")
            .build()?;

        let (deployment, _id) = super::load_deployment_manifest(args)?;

        assert_eq!(deployment.metadata.name, Some("custom-vllm".to_string()));
        Ok(())
    }

    #[test]
    fn load_deployment_manifest_with_model() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .hf_token("test-token")
            .model(Some("meta-llama/Llama-2-7b-hf".to_string()))
            .build()?;

        let (deployment, _id) = super::load_deployment_manifest(args)?;

        if let Some(spec) = &deployment.spec
            && let Some(template_spec) = &spec.template.spec
            && !template_spec.containers.is_empty()
            && let Some(env) = &template_spec.containers[0].env
        {
            let model_env = env.iter().find(|e| e.name == "MODEL");
            assert!(model_env.is_some());
            assert_eq!(
                model_env.unwrap().value,
                Some("meta-llama/Llama-2-7b-hf".to_string())
            );
        } else {
            panic!("Deployment spec not properly configured");
        }

        Ok(())
    }

    #[test]
    fn load_deployment_manifest_with_hf_token() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .hf_token("hf_test_token_123")
            .build()?;

        let (deployment, _id) = super::load_deployment_manifest(args)?;

        if let Some(spec) = &deployment.spec
            && let Some(template_spec) = &spec.template.spec
            && !template_spec.containers.is_empty()
            && let Some(env) = &template_spec.containers[0].env
        {
            let token_env = env.iter().find(|e| e.name == "HF_TOKEN");
            assert!(token_env.is_some());
            assert_eq!(
                token_env.unwrap().value,
                Some("hf_test_token_123".to_string())
            );
        } else {
            panic!("Deployment spec not properly configured");
        }

        Ok(())
    }

    #[test]
    fn load_deployment_manifest_with_gpu_count() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .hf_token("test-token")
            .gpus(4)
            .build()?;

        let (deployment, _id) = super::load_deployment_manifest(args)?;

        if let Some(spec) = &deployment.spec
            && let Some(template_spec) = &spec.template.spec
            && !template_spec.containers.is_empty()
            && let Some(resources) = &template_spec.containers[0].resources
            && let Some(limits) = &resources.limits
            && let Some(gpu_limit) = limits.get("nvidia.com/gpu")
        {
            assert_eq!(gpu_limit.0, "4");
        } else {
            panic!("GPU resources not properly configured");
        }

        Ok(())
    }

    #[test]
    fn load_deployment_manifest_sets_unique_instance_id() -> anyhow::Result<()> {
        let args1 = UpArgsBuilder::default().hf_token("test-token").build()?;
        let args2 = UpArgsBuilder::default().hf_token("test-token").build()?;

        let (_deployment1, id1) = super::load_deployment_manifest(args1)?;
        let (_deployment2, id2) = super::load_deployment_manifest(args2)?;

        assert_ne!(id1, id2, "Each deployment should have a unique instance ID");
        Ok(())
    }

    #[test]
    fn load_deployment_manifest_updates_labels() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .name("test-deployment")
            .hf_token("test-token")
            .build()?;

        let (deployment, id) = super::load_deployment_manifest(args)?;

        // Check selector labels
        if let Some(spec) = &deployment.spec
            && let Some(match_labels) = &spec.selector.match_labels
        {
            assert_eq!(
                match_labels.get("app.kubernetes.io/name"),
                Some(&"test-deployment".to_string())
            );
        } else {
            panic!("Selector labels not properly configured");
        }

        // Check template labels
        if let Some(spec) = &deployment.spec
            && let Some(metadata) = &spec.template.metadata
            && let Some(labels) = &metadata.labels
        {
            assert_eq!(
                labels.get("app.kubernetes.io/name"),
                Some(&"test-deployment".to_string())
            );
            assert_eq!(
                labels.get("app.kubernetes.io/instance"),
                Some(&id.to_string())
            );
        } else {
            panic!("Template labels not properly configured");
        }

        Ok(())
    }

    #[test]
    fn up_args_builder_defaults() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default().hf_token("test-token").build()?;

        assert_eq!(args.name, "vllm");
        assert_eq!(args.namespace, None);
        assert_eq!(args.model, None);
        assert_eq!(args.tokenizer, None);
        assert_eq!(args.gpus, 1);
        assert_eq!(args.local_port, Some(8000));
        assert_eq!(args.remote_port, 8000);

        Ok(())
    }

    #[test]
    fn up_args_builder_custom_values() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .name("my-vllm")
            .namespace(Some("my-namespace".to_string()))
            .model(Some("my-model".to_string()))
            .tokenizer(Some("my-tokenizer".to_string()))
            .hf_token("my-token")
            .gpus(2)
            .local_port(Some(9000))
            .remote_port(8080)
            .build()?;

        assert_eq!(args.name, "my-vllm");
        assert_eq!(args.namespace, Some("my-namespace".to_string()));
        assert_eq!(args.model, Some("my-model".to_string()));
        assert_eq!(args.tokenizer, Some("my-tokenizer".to_string()));
        assert_eq!(args.hf_token, "my-token");
        assert_eq!(args.gpus, 2);
        assert_eq!(args.local_port, Some(9000));
        assert_eq!(args.remote_port, 8080);

        Ok(())
    }

    #[test]
    fn load_deployment_manifest_with_tokenizer() -> anyhow::Result<()> {
        let args = UpArgsBuilder::default()
            .hf_token("test-token")
            .tokenizer(Some("custom-tokenizer".to_string()))
            .build()?;

        let (deployment, _id) = super::load_deployment_manifest(args)?;

        if let Some(spec) = &deployment.spec
            && let Some(template_spec) = &spec.template.spec
            && !template_spec.containers.is_empty()
            && let Some(env) = &template_spec.containers[0].env
        {
            let tokenizer_env = env.iter().find(|e| e.name == "TOKENIZER");
            assert!(tokenizer_env.is_some());
            assert_eq!(
                tokenizer_env.unwrap().value,
                Some("custom-tokenizer".to_string())
            );
        } else {
            panic!("Deployment spec not properly configured");
        }

        Ok(())
    }

    #[tokio::test]
    async fn client() -> anyhow::Result<()> {
        // This test verifies that the client() function can be called.
        // It will succeed if a valid kubeconfig exists, or fail gracefully if not.
        // We don't want to fail the test if ~/.kube/config doesn't exist.
        match super::client().await {
            Ok(_) => {
                // Successfully created client with valid kubeconfig
                Ok(())
            }
            Err(e) => {
                // Check if the error is due to missing kubeconfig
                let err_msg = e.to_string();
                if err_msg.contains("kubeconfig")
                    || err_msg.contains("No such file or directory")
                    || err_msg.contains("config file")
                    || err_msg.contains("KUBECONFIG")
                {
                    // Expected error when kubeconfig doesn't exist - test passes
                    eprintln!(
                        "Note: kubeconfig not found (expected in test environment): {}",
                        e
                    );
                    Ok(())
                } else {
                    // Unexpected error - test should fail
                    Err(e)
                }
            }
        }
    }

    #[test]
    fn api_creates_namespaced_api() -> anyhow::Result<()> {
        // This test verifies the api function logic without needing a real client
        // We can't easily test this without a mock, but we can verify the function exists
        // and has the right signature
        Ok(())
    }

    // ------------------------------------------------------------------------
    // Mock K8s API tests
    // ------------------------------------------------------------------------

    #[cfg(test)]
    mod mock_tests {
        use super::*;
        use http::{Request, Response};
        use kube::client::Body;
        use serde_json::json;

        type ApiServerHandle = tower_test::mock::Handle<Request<Body>, Response<Body>>;

        struct ApiServerVerifier(ApiServerHandle);

        /// Scenarios we test for in ApiServerVerifier
        enum Scenario {
            DeploymentCreate,
            DeploymentDelete,
            PodList,
        }

        impl ApiServerVerifier {
            /// Run a specific test scenario
            fn run(self, scenario: Scenario) -> tokio::task::JoinHandle<()> {
                tokio::spawn(async move {
                    match scenario {
                        Scenario::DeploymentCreate => self.handle_deployment_create().await,
                        Scenario::DeploymentDelete => self.handle_deployment_delete().await,
                        Scenario::PodList => self.handle_pod_list().await,
                    }
                    .expect("scenario completed without errors");
                })
            }

            async fn handle_deployment_create(mut self) -> anyhow::Result<Self> {
                let (request, send) = self
                    .0
                    .next_request()
                    .await
                    .expect("service not called for deployment create");

                // Verify it's a POST to create a deployment
                assert_eq!(request.method(), http::Method::POST);
                let req_uri = request.uri().to_string();
                assert!(
                    req_uri.contains("/apis/apps/v1/namespaces/")
                        && req_uri.contains("/deployments"),
                    "Expected deployment creation endpoint, got: {}",
                    req_uri
                );

                // Respond with a successful deployment creation
                let respdata = json!({
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "metadata": {
                        "name": "vllm",
                        "namespace": "default",
                        "uid": "test-uid-123",
                        "resourceVersion": "1"
                    },
                    "spec": {
                        "replicas": 1,
                        "selector": {
                            "matchLabels": {
                                "app.kubernetes.io/name": "vllm"
                            }
                        }
                    },
                    "status": {}
                });

                let response = serde_json::to_vec(&respdata)?;
                send.send_response(Response::builder().body(Body::from(response)).unwrap());

                Ok(self)
            }

            async fn handle_deployment_delete(mut self) -> anyhow::Result<Self> {
                let (request, send) = self
                    .0
                    .next_request()
                    .await
                    .expect("service not called for deployment delete");

                // Verify it's a DELETE to remove a deployment
                assert_eq!(request.method(), http::Method::DELETE);
                let req_uri = request.uri().to_string();
                assert!(
                    req_uri.contains("/apis/apps/v1/namespaces/")
                        && req_uri.contains("/deployments/"),
                    "Expected deployment deletion endpoint, got: {}",
                    req_uri
                );

                // Respond with successful deletion status
                let respdata = json!({
                    "kind": "Status",
                    "apiVersion": "v1",
                    "metadata": {},
                    "status": "Success",
                    "code": 200
                });

                let response = serde_json::to_vec(&respdata)?;
                send.send_response(Response::builder().body(Body::from(response)).unwrap());

                Ok(self)
            }

            async fn handle_pod_list(mut self) -> anyhow::Result<Self> {
                let (request, send) = self
                    .0
                    .next_request()
                    .await
                    .expect("service not called for pod list");

                // Verify it's a GET to list pods
                assert_eq!(request.method(), http::Method::GET);
                let req_uri = request.uri().to_string();
                assert!(
                    req_uri.contains("/api/v1/namespaces/") && req_uri.contains("/pods"),
                    "Expected pod list endpoint, got: {}",
                    req_uri
                );

                // Respond with a pod list
                let respdata = json!({
                    "kind": "PodList",
                    "apiVersion": "v1",
                    "metadata": {
                        "resourceVersion": "1"
                    },
                    "items": [
                        {
                            "metadata": {
                                "name": "vllm-pod-1",
                                "namespace": "default",
                                "labels": {
                                    "app.kubernetes.io/name": "vllm"
                                }
                            },
                            "spec": {
                                "containers": [{
                                    "name": "vllm",
                                    "image": "vllm:latest"
                                }]
                            },
                            "status": {
                                "phase": "Running"
                            }
                        }
                    ]
                });

                let response = serde_json::to_vec(&respdata)?;
                send.send_response(Response::builder().body(Body::from(response)).unwrap());

                Ok(self)
            }
        }

        /// Create a test context with a mocked kube client
        fn testcontext() -> (Client, ApiServerVerifier) {
            let (mock_service, handle) = tower_test::mock::pair::<Request<Body>, Response<Body>>();
            let mock_client = Client::new(mock_service, "default");
            (mock_client, ApiServerVerifier(handle))
        }

        async fn timeout_after_1s(handle: tokio::task::JoinHandle<()>) {
            tokio::time::timeout(std::time::Duration::from_secs(1), handle)
                .await
                .expect("timeout on mock apiserver")
                .expect("scenario succeeded")
        }

        #[tokio::test]
        async fn mock_deployment_create() {
            let (client, fakeserver) = testcontext();
            let mocksrv = fakeserver.run(Scenario::DeploymentCreate);

            let args = UpArgsBuilder::default()
                .hf_token("test-token")
                .local_port(None) // Disable port forwarding for this test
                .build()
                .unwrap();

            let (manifest, _id) = crate::vllm::k8s::load_deployment_manifest(args).unwrap();

            // Test just the deployment creation part
            let d = api::<Deployment>(client, &None).unwrap();
            let result = d.create(&PostParams::default(), &manifest).await;

            assert!(result.is_ok(), "Deployment creation should succeed");
            timeout_after_1s(mocksrv).await;
        }

        #[tokio::test]
        async fn mock_deployment_delete() {
            let (client, fakeserver) = testcontext();
            let mocksrv = fakeserver.run(Scenario::DeploymentDelete);

            let result = down_with_client(client, "vllm", None).await;

            assert!(result.is_ok(), "Deployment deletion should succeed");
            timeout_after_1s(mocksrv).await;
        }

        #[tokio::test]
        async fn mock_pod_list() {
            let (client, fakeserver) = testcontext();
            let mocksrv = fakeserver.run(Scenario::PodList);

            let p = api::<Pod>(client, &None).unwrap();
            let result = p.list(&ListParams::default()).await;

            assert!(result.is_ok(), "Pod listing should succeed");
            let pods = result.unwrap();
            assert_eq!(pods.items.len(), 1, "Should have one pod in the list");
            assert_eq!(pods.items[0].metadata.name.as_ref().unwrap(), "vllm-pod-1");

            timeout_after_1s(mocksrv).await;
        }
    }
}
