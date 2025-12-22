#!/bin/bash
# MARS Build Script
# Builds the wheel for distribution

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Building MARS wheel..."

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Build wheel using uv (or pip if uv not available)
if command -v uv &> /dev/null; then
    uv build --wheel
else
    python -m build --wheel
fi

echo ""
echo "Build complete!"
echo "Wheel location: $PROJECT_ROOT/dist/"
ls -la dist/*.whl
