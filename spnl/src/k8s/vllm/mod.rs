use k8s_openapi::api::apps::v1::Deployment;
//use tracing::*;

use kube::{
    Client,
    api::{Api, DeleteParams, PostParams},
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

    /// HuggingFace api token
    #[builder(setter(into))]
    hf_token: String,
}

fn load_deployment_manifest(args: UpArgs) -> anyhow::Result<Deployment> {
    let mut d: Deployment = serde_yaml2::from_str(include_str!("deployment.yml"))?;
    d.metadata.name = Some(args.name);
    if let Some(ref mut spec) = d.spec
        && let Some(ref mut spec) = spec.template.spec
        && let Some(ref mut env) = spec.containers[0].env
    {
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
    Ok(d)
}

async fn client() -> anyhow::Result<Client> {
    // does not work as reqwest pulls in ring and others pull in aws-lc-rs (this may change in reqwest 0.13)
    //rustls::crypto::CryptoProvider::install_default();
    let _ = rustls::crypto::ring::default_provider().install_default();

    Ok(Client::try_default().await?)
}

async fn deployments(namespace: &Option<String>) -> anyhow::Result<Api<Deployment>> {
    let c = client().await?;
    Ok(match namespace {
        Some(ns) => Api::namespaced(c, ns),
        None => Api::default_namespaced(c),
    })
}

pub async fn up(args: UpArgs) -> anyhow::Result<()> {
    let name = &args.name.clone();
    let d = deployments(&args.namespace).await?;
    d.create(&PostParams::default(), &load_deployment_manifest(args)?)
        .await?;

    await_condition(d.clone(), name, is_deployment_completed()).await?;
    //let establish = await_condition(deployments.clone(), &args.name, is_deployment_completed());
    //let _ = tokio::time::timeout(std::time::Duration::from_secs(120), establish).await?;

    Ok(())
}

pub async fn down(name: &str, namespace: Option<String>) -> anyhow::Result<()> {
    let _ = deployments(&namespace)
        .await?
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
