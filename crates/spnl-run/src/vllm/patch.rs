use std::io::{Read, Write};

const PATCH_DATA: &[u8] =
    include_bytes!("../../../../docker/vllm/llm-d/patches/0.5.0/01-spans-llmd-vllm.patch.gz");

/// Emit the vLLM patchfile to stdout
///
/// This function decompresses the embedded patch file and writes it to stdout
pub async fn patchfile() -> anyhow::Result<()> {
    // Decompress the gzipped patch data
    let mut decoder = flate2::read::GzDecoder::new(PATCH_DATA);
    let mut patch_content = Vec::new();
    decoder
        .read_to_end(&mut patch_content)
        .map_err(|e| anyhow::anyhow!("Failed to decompress patch: {}", e))?;

    // Write patch content to stdout
    std::io::stdout()
        .write_all(&patch_content)
        .map_err(|e| anyhow::anyhow!("Failed to write patch to stdout: {}", e))?;

    Ok(())
}

// Made with Bob
