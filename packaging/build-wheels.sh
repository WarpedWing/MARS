#!/bin/bash
# Build dfvfs dependency wheels locally
# Usage: ./build-wheels.sh [output-dir]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${1:-$PROJECT_ROOT/vendor/wheels}"

# Detect platform
PLATFORM=$(uname -s)
ARCH=$(uname -m)

if [ "$PLATFORM" = "Darwin" ]; then
    if [ "$ARCH" = "arm64" ]; then
        PLATFORM_DIR="macos-arm64"
    else
        PLATFORM_DIR="macos-x86_64"
    fi
elif [ "$PLATFORM" = "Linux" ]; then
    PLATFORM_DIR="linux-x86_64"
else
    echo "Unsupported platform: $PLATFORM"
    exit 1
fi

WHEEL_DIR="$OUTPUT_DIR/$PLATFORM_DIR"
mkdir -p "$WHEEL_DIR"

echo "Building wheels for $PLATFORM_DIR"
echo "Output directory: $WHEEL_DIR"
echo ""

# libyal packages required by dfvfs
LIBYAL_PACKAGES=(
    libbde-python
    libcaes-python
    libewf-python
    libfcrypto-python
    libfsapfs-python
    libfsext-python
    libfsfat-python
    libfshfs-python
    libfsntfs-python
    libfsxfs-python
    libfvde-python
    libfwnt-python
    libluksde-python
    libmodi-python
    libphdi-python
    libqcow-python
    libsigscan-python
    libsmdev-python
    libsmraw-python
    libvhdi-python
    libvmdk-python
    libvsapm-python
    libvsgpt-python
    libvshadow-python
    libvslvm-python
)

# Ensure pip and wheel are up to date
pip install --upgrade pip wheel build

# Build pure Python packages first (fast)
echo "Building pure Python packages..."
pip wheel dfvfs dfdatetime dtfabric PyYAML --no-deps -w "$WHEEL_DIR"

# Build xattr (macOS only)
if [ "$PLATFORM" = "Darwin" ]; then
    echo "Building xattr..."
    pip wheel xattr --no-deps -w "$WHEEL_DIR" || echo "Warning: Failed to build xattr"
fi

# Build libyal packages
echo ""
echo "Building libyal packages (this may take a while)..."
for pkg in "${LIBYAL_PACKAGES[@]}"; do
    echo "  Building $pkg..."
    pip wheel "$pkg" --no-deps -w "$WHEEL_DIR" 2>/dev/null || echo "    Warning: Failed to build $pkg"
done

# Build pytsk3
echo ""
echo "Building pytsk3..."
if [ "$PLATFORM" = "Darwin" ]; then
    # Ensure sleuthkit is installed
    if ! brew list sleuthkit &>/dev/null; then
        echo "Installing sleuthkit via Homebrew..."
        brew install sleuthkit
    fi
fi
pip wheel pytsk3 --no-deps -w "$WHEEL_DIR" || echo "Warning: Failed to build pytsk3"

echo ""
echo "Build complete!"
echo "Wheels in: $WHEEL_DIR"
ls -la "$WHEEL_DIR"
echo ""
echo "Total wheels: $(ls -1 "$WHEEL_DIR"/*.whl 2>/dev/null | wc -l | tr -d ' ')"
