#!/bin/bash
set -euo pipefail

# Script to sign and notarize macOS binaries
# Usage: sign-and-notarize-macos.sh <binary_path> <apple_id> <team_id> <app_password>

BINARY_PATH="${1:?Binary path is required}"
APPLE_ID="${2:?Apple ID is required}"
APPLE_TEAM_ID="${3:?Apple Team ID is required}"
APPLE_APP_SPECIFIC_PASSWORD="${4:?App-specific password is required}"

echo "🔐 Signing and notarizing macOS binary: $BINARY_PATH"

# Sign the binary
echo "📝 Signing binary..."
codesign --force --options runtime --sign "$APPLE_TEAM_ID" --timestamp "$BINARY_PATH"

# Verify signature
echo "✅ Verifying signature..."
codesign --verify --verbose "$BINARY_PATH"

# Create a zip for notarization (required by Apple)
echo "📦 Creating notarization package..."
NOTARIZE_ZIP="$(dirname "$BINARY_PATH")/notarize-temp.zip"
ditto -c -k --keepParent "$BINARY_PATH" "$NOTARIZE_ZIP"

# Submit for notarization
echo "☁️  Submitting for notarization..."
xcrun notarytool submit "$NOTARIZE_ZIP" \
  --apple-id "$APPLE_ID" \
  --team-id "$APPLE_TEAM_ID" \
  --password "$APPLE_APP_SPECIFIC_PASSWORD" \
  --wait

# Clean up notarization zip
rm "$NOTARIZE_ZIP"

echo "✅ macOS binary signed and notarized successfully"

# Made with Bob
