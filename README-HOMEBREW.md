# SPNL Homebrew Tap

This repository serves as a Homebrew tap for installing the SPNL CLI tool.

## Installation

To install SPNL using Homebrew:

```bash
# Add the tap
brew tap IBM/spnl https://github.com/IBM/spnl

# Install spnl
brew install spnl
```

Or install directly in one command:

```bash
brew install IBM/spnl/spnl
```

## Usage

After installation, you can use the `spnl` command:

```bash
spnl --version
spnl --help
```

## Updating

To update to the latest version:

```bash
brew update
brew upgrade spnl
```

## Uninstalling

To uninstall:

```bash
brew uninstall spnl
brew untap IBM/spnl
```

## Supported Platforms

- macOS ARM64 (Apple Silicon)
- Linux x86_64
- Linux ARM64

## More Information

For more information about SPNL, visit the [main repository](https://github.com/IBM/spnl).