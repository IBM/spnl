#!/bin/bash
set -e

# This script verifies that the feature list in core.yml is consistent with spnl/Cargo.toml
# It ensures we don't forget to update the CI when adding new features.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CORE_YML="$REPO_ROOT/.github/workflows/core.yml"
SPNL_CARGO_TOML="$REPO_ROOT/spnl/Cargo.toml"

echo "Verifying feature consistency between core.yml and spnl/Cargo.toml..."

# Extract features from spnl/Cargo.toml
# Parse the [features] section, excluding only:
# - default = [...]
# - local (CPU inference - mistralrs unconditionally depends on CUDA via candle-core/cudarc)
# - cuda* (CUDA-specific features)
CARGO_FEATURES=$(sed -n '/^\[features\]/,/^\[/p' "$SPNL_CARGO_TOML" | \
    grep -E '^[a-z]' | \
    sed 's/ *=.*//' | \
    grep -v '^default$' | \
    grep -v '^local$' | \
    grep -v '^cuda' | \
    sort)

# Extract features from core.yml cargo test command
# Look for the feature list in the macOS test section (line ~88)
CORE_YML_FEATURES=$(grep 'cargo test -p spnl --features' "$CORE_YML" | \
    sed 's/.*--features //' | \
    sed 's/ --.*//' | \
    tr ',' '\n' | \
    sort | \
    uniq)

echo ""
echo "Features in spnl/Cargo.toml (excluding default, local, cuda*):"
echo "$CARGO_FEATURES"
echo ""
echo "Features tested in core.yml (macOS cargo test -p spnl):"
echo "$CORE_YML_FEATURES"
echo ""

# Compare the two lists
MISSING_IN_CORE=$(comm -23 <(echo "$CARGO_FEATURES") <(echo "$CORE_YML_FEATURES"))
EXTRA_IN_CORE=$(comm -13 <(echo "$CARGO_FEATURES") <(echo "$CORE_YML_FEATURES"))

if [ -n "$MISSING_IN_CORE" ]; then
    echo "❌ ERROR: Features in spnl/Cargo.toml but NOT in core.yml:"
    echo "$MISSING_IN_CORE"
    echo ""
    echo "Please update the feature list in core.yml (lines ~88 and ~104) to include these features."
    exit 1
fi

if [ -n "$EXTRA_IN_CORE" ]; then
    echo "❌ ERROR: Features in core.yml but NOT in spnl/Cargo.toml:"
    echo "$EXTRA_IN_CORE"
    echo ""
    echo "Please remove these features from core.yml or add them to spnl/Cargo.toml."
    exit 1
fi

echo "✅ Feature lists are consistent!"
echo ""
echo "Excluded features (not tested in core.yml for spnl package):"
echo "  - default (tested implicitly)"
echo "  - local (CPU inference - mistralrs unconditionally depends on CUDA via candle-core/cudarc)"
echo "  - cuda* (CUDA-specific features)"
echo ""
echo "Note: 'local' feature IS tested in spnl-cli package tests, which uses it differently."

# Made with Bob
