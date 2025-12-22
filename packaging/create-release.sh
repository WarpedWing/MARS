#!/bin/bash
# MARS Release Creator
# Builds wheel and creates distribution packages for macOS and Windows

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Get version from pyproject.toml
VERSION=$(grep -E '^version = ' "$PROJECT_ROOT/pyproject.toml" | sed 's/version = "\(.*\)"/\1/')
echo "Creating MARS release v$VERSION"
echo ""

cd "$PROJECT_ROOT"

# Build wheel
echo "Building wheel..."
"$SCRIPT_DIR/build.sh"
echo ""

WHEEL_FILE=$(ls dist/mars-*.whl | head -1)
if [ -z "$WHEEL_FILE" ]; then
    echo "Error: Wheel not found"
    exit 1
fi

# Create release directory
RELEASE_DIR="$PROJECT_ROOT/release"
rm -rf "$RELEASE_DIR"
mkdir -p "$RELEASE_DIR"

# Wheel directories
VENDOR_WHEELS="$PROJECT_ROOT/vendor/wheels"
LICENSES_DIR="$PROJECT_ROOT/src/resources/licenses"

# Create macOS ARM64 distribution
echo "Creating macOS ARM64 distribution..."
MACOS_ARM64_DIR="$RELEASE_DIR/MARS-$VERSION-macos-arm64"
mkdir -p "$MACOS_ARM64_DIR/dist"
cp "$WHEEL_FILE" "$MACOS_ARM64_DIR/dist/"
cp "$SCRIPT_DIR/install-macos.sh" "$MACOS_ARM64_DIR/install.sh"
chmod +x "$MACOS_ARM64_DIR/install.sh"
cp "$PROJECT_ROOT/README.md" "$MACOS_ARM64_DIR/" 2>/dev/null || echo "# MARS v$VERSION" > "$MACOS_ARM64_DIR/README.md"
cp "$PROJECT_ROOT/THIRD-PARTY-NOTICES.md" "$MACOS_ARM64_DIR/" 2>/dev/null || true

# Copy macOS ARM64 wheels if available
if [ -d "$VENDOR_WHEELS/macos-arm64" ] && [ -n "$(ls -A "$VENDOR_WHEELS/macos-arm64"/*.whl 2>/dev/null)" ]; then
    echo "  Including macOS ARM64 wheels..."
    mkdir -p "$MACOS_ARM64_DIR/wheels/macos-arm64"
    cp "$VENDOR_WHEELS/macos-arm64"/*.whl "$MACOS_ARM64_DIR/wheels/macos-arm64/"
fi

# Copy license files
if [ -d "$LICENSES_DIR" ]; then
    echo "  Including license files..."
    mkdir -p "$MACOS_ARM64_DIR/licenses"
    cp "$LICENSES_DIR"/*.txt "$MACOS_ARM64_DIR/licenses/" 2>/dev/null || true
fi

# Create macOS x86_64 distribution
echo "Creating macOS x86_64 distribution..."
MACOS_X86_DIR="$RELEASE_DIR/MARS-$VERSION-macos-x86_64"
mkdir -p "$MACOS_X86_DIR/dist"
cp "$WHEEL_FILE" "$MACOS_X86_DIR/dist/"
cp "$SCRIPT_DIR/install-macos.sh" "$MACOS_X86_DIR/install.sh"
chmod +x "$MACOS_X86_DIR/install.sh"
cp "$PROJECT_ROOT/README.md" "$MACOS_X86_DIR/" 2>/dev/null || echo "# MARS v$VERSION" > "$MACOS_X86_DIR/README.md"
cp "$PROJECT_ROOT/THIRD-PARTY-NOTICES.md" "$MACOS_X86_DIR/" 2>/dev/null || true

# Copy macOS x86_64 wheels if available
if [ -d "$VENDOR_WHEELS/macos-x86_64" ] && [ -n "$(ls -A "$VENDOR_WHEELS/macos-x86_64"/*.whl 2>/dev/null)" ]; then
    echo "  Including macOS x86_64 wheels..."
    mkdir -p "$MACOS_X86_DIR/wheels/macos-x86_64"
    cp "$VENDOR_WHEELS/macos-x86_64"/*.whl "$MACOS_X86_DIR/wheels/macos-x86_64/"
fi

# Copy license files
if [ -d "$LICENSES_DIR" ]; then
    echo "  Including license files..."
    mkdir -p "$MACOS_X86_DIR/licenses"
    cp "$LICENSES_DIR"/*.txt "$MACOS_X86_DIR/licenses/" 2>/dev/null || true
fi

# Create Windows distribution
echo "Creating Windows distribution..."
WIN_DIR="$RELEASE_DIR/MARS-$VERSION-windows"
mkdir -p "$WIN_DIR/dist"
cp "$WHEEL_FILE" "$WIN_DIR/dist/"
cp "$SCRIPT_DIR/install-windows.bat" "$WIN_DIR/install.bat"
cp "$PROJECT_ROOT/README.md" "$WIN_DIR/" 2>/dev/null || echo "# MARS v$VERSION" > "$WIN_DIR/README.md"
cp "$PROJECT_ROOT/THIRD-PARTY-NOTICES.md" "$WIN_DIR/" 2>/dev/null || true

# Copy Windows wheels if available
if [ -d "$VENDOR_WHEELS/windows-x64" ] && [ -n "$(ls -A "$VENDOR_WHEELS/windows-x64"/*.whl 2>/dev/null)" ]; then
    echo "  Including Windows x64 wheels..."
    mkdir -p "$WIN_DIR/wheels/windows-x64"
    cp "$VENDOR_WHEELS/windows-x64"/*.whl "$WIN_DIR/wheels/windows-x64/"
fi

# Copy license files
if [ -d "$LICENSES_DIR" ]; then
    echo "  Including license files..."
    mkdir -p "$WIN_DIR/licenses"
    cp "$LICENSES_DIR"/*.txt "$WIN_DIR/licenses/" 2>/dev/null || true
fi

# Create ZIP archives (excluding .DS_Store files)
echo ""
echo "Creating ZIP archives..."
cd "$RELEASE_DIR"
zip -r "MARS-$VERSION-macos-arm64.zip" "MARS-$VERSION-macos-arm64" -x "*.DS_Store"
zip -r "MARS-$VERSION-macos-x86_64.zip" "MARS-$VERSION-macos-x86_64" -x "*.DS_Store"
zip -r "MARS-$VERSION-windows.zip" "MARS-$VERSION-windows" -x "*.DS_Store"

# Summary
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "                    Release Complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Distribution packages created in: $RELEASE_DIR"
echo ""
ls -la "$RELEASE_DIR"/*.zip
echo ""
echo "Contents:"
echo "  MARS-$VERSION-macos-arm64.zip  - macOS Apple Silicon distribution"
echo "  MARS-$VERSION-macos-x86_64.zip - macOS Intel distribution"
echo "  MARS-$VERSION-windows.zip      - Windows distribution"
echo ""
echo "Next steps:"
echo "  1. Test installation on target platforms"
echo "  2. Create GitHub release"
echo "  3. Upload ZIP files as release assets"
