# Building SPNL

## Building the CLI

To build the CLI for your current platform, the following will generate `./target/release/spnl`:
```bash
cargo build --release
```

Any build command accepts an optional `--features/-F` argument, which
lets you selectively add optional features to your build.[^1] Here are the
[SPNL feature flags](./feature-flags.md). The set of default features
that are always included is enumerated in the
[Cargo.toml](../spnl/Cargo.toml). 

[^1]: By Rust convention, to build with only a limited set of
    features, i.e. not including the default, run the build commands
    with `--no-default-features` and then selectively add just the
    features you need via `--feature/-F`.

### Cross-compiling the CLI

To build the CLI for a different platform, you can use
[`cross`](https://github.com/cross-rs/cross). First follow the [cross
installation
instructions](https://github.com/cross-rs/cross?tab=readme-ov-file#installation)
and then, for example, to build for Linux x86_64 with the GNU standard
library:

```bash
cross build --target x86_64-unknown-linux-gnu --release
```

## Building the Python Wheels

While SPNL is written in Rust, it is easy to build Python
wheels. First, [install
Maturin](https://www.maturin.rs/installation.html). Then, for example,
you can the following to build for a spectrum of Linux Python versions as
follows:

```bash
for v in 3.8 3.9 3.10 3.11 3.12 3.13 3.14
do
  for a in aarch64 amd64
  do podman run --arch $a --rm -v $(pwd):/io ghcr.io/pyo3/maturin build --no-default-features -F tok -i python$v
  done
done
```

There is a [GitHub Actions workflow](../.github/workflows/python.yml)
that does something similar.
