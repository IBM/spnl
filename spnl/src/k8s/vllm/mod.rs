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

    /// Model to serve
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
        if let Some(model) = args.model
            && let Some(kv) = env.iter_mut().find(|kv| kv.name == "MODEL")
        {
            kv.value = Some(model);
        }
        if let Some(kv) = env.iter_mut().find(|kv| kv.name == "HF_TOKEN") {
            kv.value = Some(args.hf_token);
        }
    }
    Ok(d)
}

async fn client() -> anyhow::Result<Client> {
    // does not work as reqwest pulls in ring and others pull in aws-lc-rs (this may change in reqwest 0.13)
    //rustls::crypto::CryptoProvider::install_default();
    let _ = rustls::crypto::ring::default_provider().install_default();

    Ok(Client::try_default().await?)
}

async fn deployments() -> anyhow::Result<Api<Deployment>> {
    Ok(Api::default_namespaced(client().await?))
}

pub async fn up(args: UpArgs) -> anyhow::Result<()> {
    let d = deployments().await?;
    d.create(&PostParams::default(), &load_deployment_manifest(args)?)
        .await?;

    await_condition(d.clone(), "vllm", is_deployment_completed()).await?;
    //let establish = await_condition(deployments.clone(), "vllm", is_deployment_completed());
    //let _ = tokio::time::timeout(std::time::Duration::from_secs(120), establish).await?;

    Ok(())
}

pub async fn down(name: &str) -> anyhow::Result<()> {
    let _ = deployments()
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
        super::load_deployment_manifest(super::UpArgsBuilder::default().build()?).map(|_| ())
    }

    #[tokio::test]
    async fn client() -> anyhow::Result<()> {
        super::client().await.map(|_| ())
    }
}
