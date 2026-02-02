use anyhow::{Context, Result};
use russh::client::{self, Handle};
use russh::*;
use russh_keys::key;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::Mutex;

/// SSH client handler
struct Client;

#[async_trait::async_trait]
impl client::Handler for Client {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &key::PublicKey,
    ) -> Result<bool, Self::Error> {
        // Accept any server key (equivalent to StrictHostKeyChecking=no)
        Ok(true)
    }
}

/// SSH tunnel manager
pub struct SshTunnel {
    handle: Arc<Mutex<Option<Handle<Client>>>>,
    local_addr: SocketAddr,
    remote_host: String,
    remote_port: u16,
}

impl SshTunnel {
    /// Create a new SSH tunnel
    pub async fn new(
        host: &str,
        port: u16,
        username: &str,
        local_port: u16,
        remote_host: String,
        remote_port: u16,
    ) -> Result<Self> {
        // Load SSH key
        let key_path = Self::find_ssh_key()?;
        eprintln!("Using SSH key: {}", key_path.display());

        let key_pair = Self::load_private_key(&key_path)
            .await
            .context("Failed to load SSH private key")?;

        // Configure SSH client
        let config = client::Config {
            inactivity_timeout: Some(std::time::Duration::from_secs(60)),
            keepalive_interval: Some(std::time::Duration::from_secs(20)),
            keepalive_max: 3,
            ..<_>::default()
        };

        let config = Arc::new(config);

        // Connect with retries
        let mut session = None;
        let max_retries = 10;
        let mut retry_count = 0;

        while retry_count < max_retries {
            let sh = Client {};
            match client::connect(config.clone(), (host, port), sh).await {
                Ok(s) => {
                    session = Some(s);
                    break;
                }
                Err(e) => {
                    retry_count += 1;
                    if retry_count >= max_retries {
                        return Err(anyhow::anyhow!(
                            "Failed to connect after {} retries: {}",
                            max_retries,
                            e
                        ));
                    }
                    eprintln!(
                        "SSH connection attempt {}/{} failed: {}. Retrying in 5 seconds...",
                        retry_count, max_retries, e
                    );
                    tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
                }
            }
        }

        let mut session = session.unwrap();

        // Authenticate
        let auth_res = session
            .authenticate_publickey(username, Arc::new(key_pair))
            .await
            .context("SSH authentication failed")?;

        if !auth_res {
            return Err(anyhow::anyhow!("SSH authentication rejected by server"));
        }

        eprintln!("SSH connection established to {}:{}", host, port);

        // Bind local port
        let local_addr = format!("127.0.0.1:{}", local_port)
            .parse()
            .context("Invalid local address")?;

        Ok(Self {
            handle: Arc::new(Mutex::new(Some(session))),
            local_addr,
            remote_host,
            remote_port,
        })
    }

    /// Find SSH private key (try common locations)
    fn find_ssh_key() -> Result<PathBuf> {
        let home = std::env::var("HOME").context("HOME environment variable not set")?;
        let home_path = PathBuf::from(home);

        // Try common key locations in order of preference
        let key_paths = vec![
            home_path.join(".ssh/google_compute_engine"),
            home_path.join(".ssh/id_ed25519"),
            home_path.join(".ssh/id_rsa"),
            home_path.join(".ssh/id_ecdsa"),
        ];

        for path in key_paths {
            if path.exists() {
                return Ok(path);
            }
        }

        Err(anyhow::anyhow!(
            "No SSH private key found. Tried: ~/.ssh/google_compute_engine, ~/.ssh/id_ed25519, ~/.ssh/id_rsa, ~/.ssh/id_ecdsa"
        ))
    }

    /// Load private key from file
    async fn load_private_key(path: &PathBuf) -> Result<key::KeyPair> {
        let key_data = tokio::fs::read_to_string(path)
            .await
            .context("Failed to read SSH key file")?;

        // Try to decode the key (russh-keys handles various formats)
        russh_keys::decode_secret_key(&key_data, None).context("Failed to decode SSH private key")
    }

    /// Start the tunnel (listen on local port and forward connections)
    pub async fn start(self: Arc<Self>) -> Result<()> {
        let listener = TcpListener::bind(self.local_addr)
            .await
            .context("Failed to bind local port")?;

        eprintln!(
            "SSH tunnel listening on {} -> {}:{}",
            self.local_addr, self.remote_host, self.remote_port
        );

        loop {
            match listener.accept().await {
                Ok((stream, addr)) => {
                    eprintln!("Accepted connection from {}", addr);
                    let tunnel = self.clone();
                    tokio::spawn(async move {
                        if let Err(e) = tunnel.handle_connection(stream).await {
                            eprintln!("Connection error: {}", e);
                        }
                    });
                }
                Err(e) => {
                    eprintln!("Accept error: {}", e);
                }
            }
        }
    }

    /// Handle a single connection through the tunnel
    async fn handle_connection(&self, mut local_stream: TcpStream) -> Result<()> {
        // Open a direct TCP/IP channel to the remote host
        let channel = {
            let handle_guard = self.handle.lock().await;
            let session = handle_guard
                .as_ref()
                .ok_or_else(|| anyhow::anyhow!("SSH session not available"))?;

            session
                .channel_open_direct_tcpip(
                    &self.remote_host,
                    self.remote_port as u32,
                    "127.0.0.1",
                    0,
                )
                .await
                .context("Failed to open SSH channel")?
        };

        // Convert channel to a stream for bidirectional copying
        let mut channel_stream = channel.into_stream();

        // Use tokio's copy_bidirectional for efficient forwarding
        match tokio::io::copy_bidirectional(&mut local_stream, &mut channel_stream).await {
            Ok((to_remote, to_local)) => {
                eprintln!(
                    "Connection closed: {} bytes to remote, {} bytes to local",
                    to_remote, to_local
                );
            }
            Err(e) => {
                eprintln!("Connection error: {}", e);
            }
        }

        Ok(())
    }

    /// Close the tunnel
    pub async fn close(&self) -> Result<()> {
        let mut handle = self.handle.lock().await;
        if let Some(session) = handle.take() {
            session
                .disconnect(Disconnect::ByApplication, "", "en")
                .await
                .context("Failed to disconnect SSH session")?;
        }
        Ok(())
    }
}

// Made with Bob
