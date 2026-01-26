# Quick Release Guide

## Creating a New Release

### Step 1: Prepare the Release

1. Update version (run this once to get the `set-version` command): `cargo install cargo-edit`):
   ```bash
   cargo set-version 0.14.0 --verbose  && (cd web/playground/ && npm version 0.14.0 && npm i && npm i)
   ```
   Note: you may need to repeat this, as there is some bug in cargo set-version.

3. Commit and push changes:
   ```bash
   git add cli/Cargo.toml Cargo.lock
   git commit -m "Bump version to 0.14.0"
   git push origin main
   ```

### Step 2: Create and Publish Release

1. Go to: https://github.com/IBM/spnl/releases/new

2. Create a new tag:
   - Tag version: `v0.14.0` (must start with 'v')
   - Target: `main` branch

3. Click the generate release notes button

4. Click **"Publish release"**

### Step 3: Wait for Builds

The workflow will automatically:
- ✅ Build binaries for 7 platforms (takes ~10-20 minutes)
- ✅ Create compressed archives
- ✅ Generate SHA256 checksums
- ✅ Upload all files to the release

### Step 4: Verify Release

Check that all files are attached to the release:
- [ ] `spnl-v0.14.0-linux-x86_64-gnu.tar.gz`
- [ ] `spnl-v0.14.0-linux-x86_64-musl.tar.gz`
- [ ] `spnl-v0.14.0-linux-aarch64-gnu.tar.gz`
- [ ] `spnl-v0.14.0-linux-aarch64-musl.tar.gz`
- [ ] `spnl-v0.14.0-macos-x86_64.tar.gz`
- [ ] `spnl-v0.14.0-macos-aarch64.tar.gz`
- [ ] `spnl-v0.14.0-windows-x86_64.zip`
- [ ] `checksums.txt`

## Monitoring the Workflow

1. Go to: https://github.com/IBM/spnl/actions
2. Click on the "Build and Publish CLI to Release" workflow
3. Monitor the progress of all 7 build jobs

## If Something Goes Wrong

### Build Failure

1. Check the workflow logs for the failed job
2. Fix the issue in your code
3. Delete the release and tag:
   ```bash
   git tag -d v0.14.0
   git push origin :refs/tags/v0.14.0
   ```
4. Start over from Step 1

### Partial Upload

If some files are missing:
1. Re-run the failed jobs from the Actions page
2. Or delete and recreate the release

## Platform-Specific Notes

### Linux GNU vs musl

**Recommend GNU builds for:**
- Standard Linux distributions (Ubuntu, Debian, RHEL, Fedora)
- Users with glibc 2.31+ installed

**Recommend musl builds for:**
- Alpine Linux
- Docker containers (especially minimal images)
- Maximum portability across distributions
- Embedded systems

### macOS

- **x86_64**: For Intel Macs (2020 and earlier)
- **aarch64**: For Apple Silicon (M1/M2/M3, 2020+)

Users on Apple Silicon can run either version (Rosetta 2 compatibility).

### Windows

- Only 64-bit Windows is supported
- Requires Windows 10 or later

## Release Checklist

Before publishing a release:

- [ ] Version bumped in `cli/Cargo.toml`
- [ ] `Cargo.lock` updated
- [ ] Changes committed and pushed to main
- [ ] Release notes prepared
- [ ] Tag follows `vX.Y.Z` format
- [ ] All CI checks passing on main branch

After publishing:

- [ ] Workflow completed successfully
- [ ] All 7 platform binaries uploaded
- [ ] Checksums file present
- [ ] Release notes are clear and complete
- [ ] Announcement made (if applicable)

## Versioning

Follow [Semantic Versioning](https://semver.org/):

- **Major** (v1.0.0 → v2.0.0): Breaking changes
- **Minor** (v0.13.0 → v0.14.0): New features, backward compatible
- **Patch** (v0.13.0 → v0.13.1): Bug fixes, backward compatible

## Pre-releases

For beta/RC releases:

1. Use a tag like `v0.14.0-beta.1` or `v0.14.0-rc.1`
2. Check "This is a pre-release" when creating the release
3. The workflow will still build all binaries

## Hotfix Releases

For urgent fixes:

1. Create a hotfix branch from the release tag
2. Apply the fix
3. Bump patch version (e.g., v0.13.0 → v0.13.1)
4. Follow the normal release process

## Automation Tips

### GitHub CLI

Create releases from the command line:

```bash
# Create and publish release
gh release create v0.14.0 \
  --title "Release v0.14.0" \
  --notes "Release notes here"

# The workflow will automatically build and upload binaries
```

### Release Notes Template

```markdown
## What's New

- Feature 1
- Feature 2
- Feature 3

## Bug Fixes

- Fix 1
- Fix 2

## Breaking Changes

- Change 1 (if any)

## Installation

Download the appropriate binary for your platform from the assets below.

### Linux
- **GNU builds**: For standard Linux distributions
- **musl builds**: For Alpine Linux or maximum portability

### macOS
- **x86_64**: Intel Macs
- **aarch64**: Apple Silicon (M1/M2/M3)

### Windows
- **x86_64**: 64-bit Windows 10+

## Verification

Verify your download using SHA256 checksums:
\`\`\`bash
sha256sum -c checksums.txt
\`\`\`
```

## Support

For issues with the release workflow:
1. Check workflow logs in GitHub Actions
2. Review the [workflow documentation](README-release-cli.md)
3. Open an issue if you find a bug
