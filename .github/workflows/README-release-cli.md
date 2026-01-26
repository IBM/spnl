# Release CLI Workflow Documentation

## Overview

The `release-cli.yml` workflow automatically builds and publishes the SPNL CLI binary for multiple platforms when a GitHub release is published.

## Supported Platforms

The workflow builds binaries for **7 platform configurations**:

### Linux (4 builds)
- **x86_64 GNU** - Dynamic linking with glibc (most common)
- **ARM64 GNU** - Dynamic linking with glibc (for ARM servers)
- **x86_64 musl** - Static binary (portable, works on any Linux distro)
- **ARM64 musl** - Static binary (portable ARM build)

### macOS (2 builds)
- **x86_64** - Intel Macs
- **ARM64** - Apple Silicon (M1/M2/M3)

### Windows (1 build)
- **x86_64** - 64-bit Windows

## Trigger

The workflow runs automatically when you **publish a GitHub release**:

1. Go to your repository's Releases page
2. Click "Draft a new release"
3. Create a tag (e.g., `v0.13.0`)
4. Fill in release title and description
5. Click "Publish release"

The workflow will start immediately and build all platform binaries.

## Output Artifacts

After the workflow completes, the following files will be attached to your release:

```
spnl-v0.13.0-linux-x86_64-gnu.tar.gz       # Linux x86_64 (glibc)
spnl-v0.13.0-linux-x86_64-musl.tar.gz      # Linux x86_64 (static)
spnl-v0.13.0-linux-aarch64-gnu.tar.gz      # Linux ARM64 (glibc)
spnl-v0.13.0-linux-aarch64-musl.tar.gz     # Linux ARM64 (static)
spnl-v0.13.0-macos-x86_64.tar.gz           # macOS Intel
spnl-v0.13.0-macos-aarch64.tar.gz          # macOS Apple Silicon
spnl-v0.13.0-windows-x86_64.zip            # Windows 64-bit
checksums.txt                               # SHA256 checksums for all files
```

## Build Features

Each binary is built with the following Cargo features enabled:
- `rag` - RAG (Retrieval-Augmented Generation) support
- `vllm` - vLLM integration
- `spnl-api` - SPNL API support
- `openssl-vendored` - Statically linked OpenSSL

## Architecture

### Two-Job Design

1. **build-and-upload** (parallel matrix job)
   - Builds binaries for all 7 platforms in parallel
   - Creates compressed archives (.tar.gz or .zip)
   - Generates SHA256 checksums
   - Uploads artifacts to GitHub Actions

2. **upload-to-release** (sequential job)
   - Downloads all build artifacts
   - Creates combined checksums.txt file
   - Uploads all files to the GitHub release

### Build Methods

- **Native builds**: Used for most platforms (faster)
- **Cross-compilation**: Used for musl builds via `cross-rs/cross` tool

## GNU vs musl Builds

### GNU libc builds (dynamic)
- **Pros**: Smaller file size, faster build time
- **Cons**: Requires compatible glibc version on target system
- **Use case**: Standard Linux distributions (Ubuntu, Debian, RHEL, etc.)

### musl builds (static)
- **Pros**: Fully portable, works on any Linux distro, no dependencies
- **Cons**: Slightly larger file size, longer build time
- **Use case**: Alpine Linux, containers, embedded systems, maximum portability

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
cargo build --features your,features,here --release --package spnl-cli --target $TARGET
```

### Adjusting Compression

- Unix archives use `tar czf` (gzip compression)
- Windows archives use `7z a` (zip format)

To change compression level or format, modify the "Prepare binary" steps.

## Troubleshooting

### Build Failures

1. **Protobuf errors**: Ensure protoc is properly installed for the platform
2. **OpenSSL errors**: The `openssl-vendored` feature should handle this
3. **Cross-compilation errors**: Check the `cross-rs/cross` tool compatibility

### Upload Failures

1. **Permission errors**: Ensure the workflow has `contents: write` permission
2. **Release not found**: Verify the release was published (not draft)
3. **File conflicts**: The workflow uses `--clobber` to overwrite existing files

## Performance

- **Parallel builds**: All 7 platforms build simultaneously
- **Build time**: ~10-20 minutes total (depending on GitHub Actions queue)
- **Caching**: Rust toolchain and dependencies are cached between runs

## Security

- **Checksums**: SHA256 checksums are generated for all binaries
- **Verification**: Users can verify downloads using the checksums.txt file
- **Static linking**: musl builds have no external dependencies

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

- **Rust version**: Uses stable Rust via `dtolnay/rust-toolchain@stable`
- **Protobuf version**: Currently pinned to v31.1
- **Actions versions**: Keep actions up to date (checkout@v4, upload-artifact@v4, etc.)

## Related Files

- Workflow: `.github/workflows/release-cli.yml`
- CLI Cargo.toml: `cli/Cargo.toml`
- Main Cargo.toml: `Cargo.toml`