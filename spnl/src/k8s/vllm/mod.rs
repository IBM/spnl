use futures::{AsyncBufReadExt, TryStreamExt};
use k8s_openapi::api::apps::v1::Deployment;
use k8s_openapi::api::core::v1::Pod;
//use tracing::*;

use kube::{
    Client, ResourceExt,
    api::{Api, DeleteParams, ListParams, LogParams, PostParams},
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

    /// HuggingFace api token
    #[builder(setter(into))]
    hf_token: String,
}

fn load_deployment_manifest(args: UpArgs) -> anyhow::Result<(Deployment, uuid::Uuid)> {
    let id = uuid::Uuid::new_v4();
    let mut d: Deployment = serde_yaml2::from_str(include_str!("deployment.yml"))?;
    d.metadata.name = Some(args.name.clone());
    if let Some(ref mut ospec) = d.spec
        && let Some(ref mut spec) = ospec.template.spec
        && let Some(ref mut env) = spec.containers[0].env
    {
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

        match env.iter_mut().find(|kv| kv.name == "HF_TOKEN") {
            Some(kv) => kv.value = Some(args.hf_token),
            None => {
                return Err(anyhow::anyhow!(
                    "Missing HF_TOKEN env var in deployment.yml template"
                ));
            }
        };
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
    let name = &args.name.clone();
    let c = client().await?;
    let d = api::<Deployment>(c.clone(), &args.namespace)?;
    let p = api::<Pod>(client().await?, &args.namespace)?;
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
        .into_iter()
        .map(|pod| {
            tokio::spawn({
                let pp = p.clone();
                async move {
                    let mut done = false;
                    while !done {
                        match pp
                            .log_stream(
                                &pod,
                                &LogParams {
                                    follow: true,
                                    //container: Some("vllm".to_string()),
                                    //tail_lines: app.tail,
                                    //since_seconds: app.since,
                                    //since_time: last_time,
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
                                            // last_time = ::std::time::Instant::now().as_secs();
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
    /* eprintln!("Awaiting deployment completion");
    loop {
        if await_condition(d.clone(), name, is_deployment_completed()).await.is_ok() {
            eprintln!("READY");
            break
        }
    }*/

    futures::future::try_join_all(join_handles).await?;

    Ok(())
}

pub async fn down(name: &str, namespace: Option<String>) -> anyhow::Result<()> {
    let _ = api::<Deployment>(client().await?, &namespace)?
        .delete(name, &DeleteParams::default())
        .await?
        .map_left(|o| println!("Deleting deployment: {:?}", o.status))
        .map_right(|s| println!("Deleted deployment: {:?}", s));
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn load_deployment_manifest() -> anyhow::Result<()> {
        super::load_deployment_manifest(super::UpArgsBuilder::default().hf_token("").build()?)
            .map(|_| ())
    }

    #[tokio::test]
    async fn client() -> anyhow::Result<()> {
        super::client().await.map(|_| ())
    }
}
