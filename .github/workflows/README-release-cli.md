# Release CLI Workflow Documentation

## Overview

The `release-cli.yml` workflow automatically builds and publishes the SPNL CLI binary for multiple platforms when a GitHub release is published.

## Supported Platforms

The workflow builds binaries for **11 platform configurations**:

### Linux (7 builds)
- **x86_64 GNU** - Dynamic linking with glibc (most common)
- **ARM64 GNU** - Dynamic linking with glibc (for ARM servers)
- **x86_64 musl** - Static binary (portable, works on any Linux distro)
- **x86_64 CUDA** - GPU acceleration with CUDA (Ampere/RTX 30* series and newer)
- **x86_64 CUDA Flash-Attn** - CUDA with Flash Attention optimization (Ampere and newer)
- **x86_64 CUDA Flash-Attn-v3** - CUDA with Flash Attention v3 (Hopper/H100 only)

### macOS (2 builds)
- **x86_64** - Intel Macs
- **ARM64** - Apple Silicon (M1/M2/M3)

### Windows (2 builds)
- **x86_64** - 64-bit Windows (Intel/AMD)
- **ARM64** - 64-bit Windows on ARM (Surface Pro X, etc.)

## Trigger

The workflow runs automatically when you **publish a GitHub release**:

1. Go to your repository's Releases page
2. Click "Draft a new release"
3. Create a tag (e.g., `v0.13.0`)
4. Fill in release title and description
5. Click "Publish release"

The workflow will start immediately and build all platform binaries.

## Output Artifacts

After the workflow completes, the following files will be attached to your release with user-friendly display names:

| File | Display Name |
|------|--------------|
| `spnl-v0.13.0-linux-x86_64-gnu.tar.gz` | **Linux x86_64 (GNU)** |
| `spnl-v0.13.0-linux-x86_64-gnu-cuda.tar.gz` | **Linux x86_64 (GNU) CUDA** |
| `spnl-v0.13.0-linux-x86_64-gnu-cuda-flash-attn.tar.gz` | **Linux x86_64 (GNU) CUDA Flash-Attn** |
| `spnl-v0.13.0-linux-x86_64-gnu-cuda-flash-attn-v3.tar.gz` | **Linux x86_64 (GNU) CUDA Flash-Attn-v3** |
| `spnl-v0.13.0-linux-x86_64-musl.tar.gz` | **Linux x86_64 (musl static)** |
| `spnl-v0.13.0-linux-aarch64-gnu.tar.gz` | **Linux ARM64 (GNU)** |
| `spnl-v0.13.0-macos-x86_64.tar.gz` | **macOS x86_64** |
| `spnl-v0.13.0-macos-aarch64.tar.gz` | **macOS Apple Silicon** |
| `spnl-v0.13.0-windows-x86_64.zip` | **Windows x86_64** |
| `spnl-v0.13.0-windows-aarch64.zip` | **Windows ARM64** |
| `checksums.txt` | **SHA256 Checksums** |

The display names make it easier for users to identify the right binary for their platform in the GitHub release UI.

## Build Features

Each binary is built with the following Cargo features enabled:
- `rag` - RAG (Retrieval-Augmented Generation) support
- `vllm` - vLLM integration
- `spnl-api` - SPNL API support

**Note**: The `openssl-vendored` feature (statically linked OpenSSL) is only used for musl builds to create fully static binaries. All other platforms (GNU Linux, macOS, Windows) use system OpenSSL for faster builds and smaller binaries.

## Architecture

### Four-Job Design

1. **build-and-upload** (parallel matrix job)
   - Builds binaries for GNU libc, macOS, and Windows platforms in parallel
   - Creates compressed archives (.tar.gz or .zip)
   - Generates SHA256 checksums
   - Uploads artifacts to GitHub Actions

2. **build-musl** (parallel containerized job)
   - Builds static musl binaries using Docker containers
   - Runs in isolated Alpine Linux and cross-compilation containers
   - Creates compressed archives (.tar.gz)
   - Generates SHA256 checksums
   - Uploads artifacts to GitHub Actions

3. **build-cuda** (parallel containerized job)
   - Builds CUDA-enabled binaries using NVIDIA CUDA container
   - Runs in `nvidia/cuda:12.9.1-devel-ubuntu24.04` container
   - Includes disk space cleanup step to handle large CUDA builds (~20GB+)
   - Creates compressed archives (.tar.gz)
   - Generates SHA256 checksums
   - Uploads artifacts to GitHub Actions

4. **upload-to-release** (sequential job)
   - Downloads all build artifacts from all three build jobs
   - Creates combined checksums.txt file
   - Uploads all files to the GitHub release

### Build Methods

- **Native builds**: Used for GNU libc, macOS, and Windows platforms (faster)
- **Containerized builds**: Used for musl and CUDA builds via Docker containers
  - `rust:alpine` for x86_64-unknown-linux-musl (on ubuntu-latest)
  - `rust:alpine` for aarch64-unknown-linux-musl (on ubuntu-24.04-arm native ARM64 runner)
  - `nvidia/cuda:12.8.1-devel-ubuntu22.04` for CUDA builds (on ubuntu-latest)

## CUDA Builds (GPU Acceleration)

### Overview

The workflow includes three CUDA build variants for Linux x86_64 with GPU acceleration support. These builds enable local model inference using NVIDIA GPUs.

### Minimum GPU Requirements

**IMPORTANT**: All CUDA builds require NVIDIA GPUs with **compute capability 80 or higher** (Ampere architecture and newer).

#### Why Compute Capability 80+?

The `candle-kernels` dependency uses WMMA (Warp Matrix Multiply Accumulate) operations with bfloat16 data type, which requires compute capability 80 or higher. Attempting to use these builds on older GPUs (e.g., RTX 20* series with compute capability 75) will result in runtime errors.

### Supported GPUs

| Compute Capability | Architecture | GPUs |
|-------------------|--------------|------|
| **80** | Ampere | NVIDIA A100, A30, A10 |
| **86** | Ampere | RTX 30* series (3060, 3070, 3080, 3090, etc.) |
| **89** | Ada Lovelace | RTX 40* series (4060, 4070, 4080, 4090), L40, L40S |
| **90** | Hopper | H100, H800, RTX 50* series |

### Build Variants

1. **cuda** (Base CUDA)
   - Supports compute capabilities: 80, 86, 89, 90
   - Best for: General GPU acceleration
   - File: `spnl-v0.13.0-linux-x86_64-gnu-cuda.tar.gz`

2. **cuda-flash-attn** (Flash Attention)
   - Supports compute capabilities: 80, 86, 89
   - Best for: Optimized attention mechanism (faster inference)
   - File: `spnl-v0.13.0-linux-x86_64-gnu-cuda-flash-attn.tar.gz`

3. **cuda-flash-attn-v3** (Flash Attention v3)
   - Supports compute capability: 90 only (Hopper architecture)
   - Best for: Latest H100/H800 GPUs with maximum performance
   - File: `spnl-v0.13.0-linux-x86_64-gnu-cuda-flash-attn-v3.tar.gz`

### Build Environment

- **Container**: `nvidia/cuda:12.8.1-devel-ubuntu22.04`
- **CUDA Version**: 12.8.1 (latest stable 12.8.x release)
- **Ubuntu Version**: 22.04
- **Disk Space**: CUDA builds are very large (~20GB+)
- **Cleanup**: The workflow automatically frees up disk space before building by removing unnecessary software (Java, .NET, Android SDKs, etc.)

### Choosing the Right CUDA Build

1. **Check your GPU compute capability**:
   ```bash
   nvidia-smi --query-gpu=compute_cap --format=csv
   ```

2. **Select the appropriate build**:
   - RTX 30* series or A100: Use any variant (base CUDA, flash-attn, or flash-attn-v3 if on H100)
   - RTX 40* series or L40S: Use base CUDA or flash-attn
   - H100/H800: Use flash-attn-v3 for best performance

3. **Download and install**:
   ```bash
   wget https://github.com/IBM/spnl/releases/download/v0.13.0/spnl-v0.13.0-linux-x86_64-gnu-cuda.tar.gz
   tar xzf spnl-v0.13.0-linux-x86_64-gnu-cuda.tar.gz
   sudo mv spnl /usr/local/bin/
   ```

### Troubleshooting CUDA Builds

**Build Failures**:

1. **Out of disk space**
   - The workflow includes automatic cleanup that removes ~15GB of unnecessary software
   - Very large builds may still fail if the runner has limited space
   - The cleanup script removes: Java, .NET, Swift, Android SDKs, Chromium, etc.

2. **WMMA compilation errors** (`incomplete type "nvcuda::wmma::fragment"`)
   - **Cause**: Attempting to build for compute capability 75 or lower
   - **Solution**: Use minimum compute capability 80 (Ampere architecture)
   - **Why**: The `candle-kernels` dependency uses WMMA operations with bfloat16 data type, which requires compute capability 80+
   - **Affected GPUs**: RTX 20* series (compute capability 75) is not supported

3. **GCC compatibility errors** (`gcc: error: unrecognized debug output level 'encode'`)
   - **Cause**: CUDA 12.9+ may have GCC compatibility issues
   - **Solution**: Use CUDA 12.8.1 (latest stable 12.8.x release)
   - **Why**: CUDA 12.8.x is known to work well with standard GCC versions
   - **Result**: No special GCC configuration or compiler flags needed

4. **CUDA toolkit not found**
   - Verify the CUDA container image is accessible and properly configured
   - Check that the container image tag exists on Docker Hub

**Runtime Issues**:
- **GPU not detected**: Ensure NVIDIA drivers are installed and `nvidia-smi` works
- **Compute capability mismatch**: Use a build variant compatible with your GPU's compute capability
- **Out of memory**: Reduce batch size or use a model with fewer parameters

## GNU vs musl Builds

### GNU libc builds (dynamic)
- **Pros**: Smaller file size, faster build time
- **Cons**: Requires compatible glibc version on target system
- **Use case**: Standard Linux distributions (Ubuntu, Debian, RHEL, etc.)
- **Build method**: Native compilation on GitHub-hosted runners

### musl builds (static)
- **Pros**: Fully portable, works on any Linux distro, no dependencies
- **Cons**: Slightly larger file size, longer build time
- **Use case**: Alpine Linux, containers, embedded systems, maximum portability
- **Build method**: Containerized builds using Alpine Linux and cross-compilation images

## Debugging

To test the workflow without creating a release:

1. Uncomment the `pull_request` trigger in the workflow file:
   ```yaml
   on:
     release:
       types: [published]
     pull_request:  # Uncomment these lines
       branches: [ main ]
   ```

2. Create a pull request to trigger the workflow
3. The workflow will build all binaries but won't upload to a release

## Customization

### Adding a New Platform

To add support for a new platform, add an entry to the `matrix.platform` array:

```yaml
- runner: ubuntu-latest
  target: new-target-triple
  platform: platform-name
  arch: architecture
  libc: gnu-or-musl-or-empty
  name: Display Name
  use_cross: true-or-false
```

### Changing Build Features

Modify the `cargo build` command in the workflow:

```yaml
cargo build --manifest-path crates/spnl-cli/Cargo.toml --features your,features,here --release --target $TARGET
```

### Adjusting Compression

- Unix archives use `tar czf` (gzip compression)
- Windows archives use `7z a` (zip format)

To change compression level or format, modify the "Prepare binary" steps.

## Troubleshooting

### Build Failures

1. **Protobuf errors**: Ensure protoc is properly installed for the platform
2. **OpenSSL errors**:
   - For musl builds: The `openssl-vendored` feature builds OpenSSL from source
   - For other platforms: Ensure system OpenSSL is available (usually pre-installed)
3. **Container errors**: Check Docker container availability and compatibility
4. **musl build errors**: Verify Alpine packages (including perl and make for OpenSSL) are correctly installed in containers

### Upload Failures

1. **Permission errors**: Ensure the workflow has `contents: write` permission
2. **Release not found**: Verify the release was published (not draft)
3. **File conflicts**: The workflow uses `--clobber` to overwrite existing files

## Performance

- **Parallel builds**: All 8 platforms build simultaneously
- **Build time**: ~10-20 minutes total (depending on GitHub Actions queue)
- **Caching**: Rust toolchain and dependencies are cached between runs

## Security

- **Checksums**: SHA256 checksums are generated for all binaries
- **Verification**: Users can verify downloads using the checksums.txt file
- **Static linking**: musl builds have no external dependencies
- **macOS Code Signing**: macOS binaries are signed with Apple Developer certificate (optional)
- **macOS Notarization**: macOS binaries are notarized by Apple for Gatekeeper compatibility (optional)

### Optional: macOS Code Signing

**Note**: macOS code signing and notarization are **optional**. The workflow will build unsigned macOS binaries if the secrets are not configured. Unsigned binaries will work but may show Gatekeeper warnings to users.

To enable macOS code signing and notarization, configure these repository secrets:

| Secret Name | Description |
|-------------|-------------|
| `APPLE_CERTIFICATE` | Base64-encoded .p12 certificate file |
| `APPLE_CERTIFICATE_PASSWORD` | Password for the .p12 certificate |
| `KEYCHAIN_PASSWORD` | Password for temporary CI keychain (generate any secure random string) |
| `APPLE_ID` | Your Apple ID email |
| `APPLE_TEAM_ID` | Your Apple Developer Team ID |
| `APPLE_APP_SPECIFIC_PASSWORD` | App-specific password for notarization |

**Note**: `KEYCHAIN_PASSWORD` is NOT your Mac's keychain password. It's a password you create specifically for the temporary keychain that GitHub Actions creates during the build. You can generate any secure random string (e.g., using `openssl rand -base64 32`).

#### Setting Up Apple Secrets

1. **Export Certificate**:
   ```bash
   # Export from Keychain Access as .p12 file
   # Then encode to base64:
   base64 -i YourCertificate.p12 | pbcopy
   # Paste into APPLE_CERTIFICATE secret
   ```

2. **Get Team ID**:
   - Visit https://developer.apple.com/account
   - Find your Team ID in Membership section

3. **Create App-Specific Password**:
   - Visit https://appleid.apple.com
   - Sign in and go to Security section
   - Generate app-specific password for notarization

## Example Usage

After the workflow completes, users can download and install:

### Linux (GNU)
```bash
wget https://github.com/IBM/spnl/releases/download/v0.13.0/spnl-v0.13.0-linux-x86_64-gnu.tar.gz
tar xzf spnl-v0.13.0-linux-x86_64-gnu.tar.gz
sudo mv spnl /usr/local/bin/
```

### Linux (musl - static)
```bash
wget https://github.com/IBM/spnl/releases/download/v0.13.0/spnl-v0.13.0-linux-x86_64-musl.tar.gz
tar xzf spnl-v0.13.0-linux-x86_64-musl.tar.gz
sudo mv spnl /usr/local/bin/
```

### macOS
```bash
wget https://github.com/IBM/spnl/releases/download/v0.13.0/spnl-v0.13.0-macos-aarch64.tar.gz
tar xzf spnl-v0.13.0-macos-aarch64.tar.gz
sudo mv spnl /usr/local/bin/
```

### Windows
```powershell
# Download spnl-v0.13.0-windows-x86_64.zip
# Extract and add to PATH
```

### Verify Checksum
```bash
# Download checksums.txt
sha256sum -c checksums.txt
```

## Maintenance

- **Rust version**: Uses stable Rust via official rustup installer
- **Protobuf version**: Currently pinned to v31.1
- **Actions versions**: All actions are from GitHub's verified marketplace (checkout@v4, upload-artifact@v4, download-artifact@v4)

## Related Files

- Workflow: `.github/workflows/release-cli.yml`
- CLI Cargo.toml: `crates/spnl-cli/Cargo.toml`
- Main Cargo.toml: `Cargo.toml`