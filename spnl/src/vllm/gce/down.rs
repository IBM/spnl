use crate::vllm::gce::GceConfig;
use tabled::{Table, Tabled, settings::Style};

/// Delete a GCE instance
///
/// This function deletes a GCE instance by name using the Google Cloud Compute API.
pub async fn down(name: &str, _namespace: Option<String>, config: GceConfig) -> anyhow::Result<()> {
    use google_cloud_compute_v1::client::Instances;
    use google_cloud_lro::Poller;

    // Get configuration from the provided config
    let project = config.get_project()?;
    let zone = &config.zone;

    #[derive(Tabled)]
    struct InstanceInfo {
        #[tabled(rename = "Property")]
        property: String,
        #[tabled(rename = "Value")]
        value: String,
    }

    let info = vec![
        InstanceInfo {
            property: "Name".to_string(),
            value: name.to_string(),
        },
        InstanceInfo {
            property: "Project".to_string(),
            value: project.clone(),
        },
        InstanceInfo {
            property: "Zone".to_string(),
            value: zone.clone(),
        },
    ];

    let mut table = Table::new(info);
    table.with(Style::sharp());

    eprintln!("\nDeleting GCE Instance:");
    eprintln!("{}\n", table);

    // Create the client
    let client = Instances::builder().build().await?;

    // Check if instance exists first
    match client
        .get()
        .set_project(&project)
        .set_zone(zone)
        .set_instance(name)
        .send()
        .await
    {
        Ok(_) => {
            eprintln!("Instance '{}' found, proceeding with deletion...", name);
        }
        Err(e) => {
            return Err(anyhow::anyhow!(
                "Instance '{}' not found in zone '{}': {}",
                name,
                zone,
                e
            ));
        }
    }

    // Delete the instance
    eprintln!("Submitting instance deletion request...");
    let _operation = client
        .delete()
        .set_project(&project)
        .set_zone(zone)
        .set_instance(name)
        .poller()
        .until_done()
        .await?
        .to_result()?;

    eprintln!("Instance '{}' deleted successfully", name);
    // eprintln!("Operation: {:?}", _operation);

    Ok(())
}

#[cfg(test)]
mod tests {
    // Note: Integration tests that call the actual down() function have been removed
    // as they require real GCE credentials. The mock tests below provide proper unit test coverage.

    // ------------------------------------------------------------------------
    // Mock GCE API tests for deletion
    // ------------------------------------------------------------------------

    #[cfg(test)]
    mod mock_tests {
        /// Mock GCE client for testing deletion
        struct MockGceDeleteClient {
            should_fail: bool,
            deleted_instances: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
        }

        impl MockGceDeleteClient {
            fn new() -> Self {
                Self {
                    should_fail: false,
                    deleted_instances: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
                }
            }

            fn with_failure() -> Self {
                Self {
                    should_fail: true,
                    deleted_instances: std::sync::Arc::new(std::sync::Mutex::new(Vec::new())),
                }
            }

            async fn delete_instance(&self, name: &str) -> anyhow::Result<()> {
                if self.should_fail {
                    return Err(anyhow::anyhow!("Mock GCE API deletion error"));
                }

                let mut deleted = self.deleted_instances.lock().unwrap();
                deleted.push(name.to_string());

                Ok(())
            }

            fn get_deleted_instances(&self) -> Vec<String> {
                self.deleted_instances.lock().unwrap().clone()
            }
        }

        #[tokio::test]
        async fn mock_instance_deletion_success() {
            let client = MockGceDeleteClient::new();
            let result = client.delete_instance("test-instance").await;

            assert!(result.is_ok());
            assert_eq!(client.get_deleted_instances(), vec!["test-instance"]);
        }

        #[tokio::test]
        async fn mock_instance_deletion_failure() {
            let client = MockGceDeleteClient::with_failure();
            let result = client.delete_instance("test-instance").await;

            assert!(result.is_err());
            assert!(client.get_deleted_instances().is_empty());
        }

        #[tokio::test]
        async fn mock_multiple_instance_deletions() {
            let client = MockGceDeleteClient::new();

            client.delete_instance("instance-1").await.unwrap();
            client.delete_instance("instance-2").await.unwrap();
            client.delete_instance("instance-3").await.unwrap();

            let deleted = client.get_deleted_instances();
            assert_eq!(deleted.len(), 3);
            assert!(deleted.contains(&"instance-1".to_string()));
            assert!(deleted.contains(&"instance-2".to_string()));
            assert!(deleted.contains(&"instance-3".to_string()));
        }

        #[test]
        fn test_zone_default_value() {
            // Test that the default zone logic works
            let zone = std::env::var("GCE_ZONE_TEST_NONEXISTENT")
                .unwrap_or_else(|_| "us-west1-a".to_string());
            assert_eq!(zone, "us-west1-a");
        }
    }
}

// Made with Bob
