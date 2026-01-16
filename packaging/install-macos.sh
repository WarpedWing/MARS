#!/bin/bash
# MARS Installer for macOS
# Creates isolated virtual environment and installs MARS

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$SCRIPT_DIR"
VENV_DIR="$INSTALL_DIR/.venv"

echo -e "${CYAN}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           MARS - macOS Artifact Recovery Suite            ║"
echo "║                      Installer                            ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python version
echo -e "${CYAN}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    echo "Please install Python 3.13 or higher from https://python.org"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 13 ]); then
    echo -e "${RED}Error: Python 3.13 or higher is required. Found: $PYTHON_VERSION${NC}"
    exit 1
fi
echo -e "${GREEN}Found Python $PYTHON_VERSION${NC}"

# Find wheel file
echo -e "${CYAN}Looking for MARS wheel...${NC}"
WHEEL_FILE=$(find "$SCRIPT_DIR" -maxdepth 2 -name "mars-*.whl" | head -1)
if [ -z "$WHEEL_FILE" ]; then
    WHEEL_FILE=$(find "$SCRIPT_DIR/dist" -maxdepth 1 -name "mars-*.whl" 2>/dev/null | head -1)
fi

if [ -z "$WHEEL_FILE" ]; then
    echo -e "${RED}Error: No MARS wheel file found.${NC}"
    echo "Expected location: $SCRIPT_DIR/dist/mars-*.whl"
    exit 1
fi
echo -e "${GREEN}Found: $(basename "$WHEEL_FILE")${NC}"

# Create virtual environment
echo ""
echo -e "${CYAN}Creating virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    read -p "Virtual environment already exists. Reinstall? [y/N] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Keeping existing installation."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate and install
echo ""
source "$VENV_DIR/bin/activate"
pip install --upgrade pip wheel > /dev/null 2>&1

# Detect architecture and check for pre-built wheels
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    WHEEL_DIR="$SCRIPT_DIR/wheels/macos-arm64"
elif [ "$ARCH" = "x86_64" ]; then
    WHEEL_DIR="$SCRIPT_DIR/wheels/macos-x86_64"
else
    WHEEL_DIR=""
fi

echo -e "${CYAN}Installing MARS...${NC}"
if [ -d "$WHEEL_DIR" ] && [ -n "$(ls -A "$WHEEL_DIR"/*.whl 2>/dev/null)" ]; then
    echo -e "${GREEN}Using pre-built wheels (fast install)${NC}"
    pip install --find-links "$WHEEL_DIR" "$WHEEL_FILE"
else
    echo -e "${YELLOW}Pre-built wheels not found, building from source...${NC}"
    echo "This may take 10+ minutes for dfvfs dependencies."
    pip install "$WHEEL_FILE"
fi
echo -e "${GREEN}MARS installed successfully${NC}"

# Create symlink to bundled tools for easy user access (LGPL compliance)
echo ""
echo -e "${CYAN}Creating tools symlink...${NC}"
MARS_TOOLS_DIR="$INSTALL_DIR/tools"
SITE_PACKAGES=$("$VENV_DIR/bin/python3" -c "import site; print(site.getsitepackages()[0])")
RESOURCES_DIR="$SITE_PACKAGES/resources/macos"

if [ -d "$RESOURCES_DIR" ]; then
    # Remove existing symlink if present
    [ -L "$MARS_TOOLS_DIR" ] && rm "$MARS_TOOLS_DIR"
    ln -s "$RESOURCES_DIR" "$MARS_TOOLS_DIR"
    # Fix execute permissions on binaries
    chmod +x "$RESOURCES_DIR/bin/"* 2>/dev/null || true
    echo -e "${GREEN}Tools accessible at: $MARS_TOOLS_DIR${NC}"
    echo "  Bundled binaries and libraries are symlinked here."
    echo "  See THIRD-PARTY-NOTICES.md for licenses and source code."
else
    echo -e "${YELLOW}Note: Resources directory not found in site-packages${NC}"
fi

# Create symlink to mars package for easy source code access
echo ""
echo -e "${CYAN}Creating mars source symlink...${NC}"
MARS_SRC_DIR="$INSTALL_DIR/mars_src"
MARS_PKG_DIR="$SITE_PACKAGES/mars"

if [ -d "$MARS_PKG_DIR" ]; then
    # Remove existing symlink if present
    [ -L "$MARS_SRC_DIR" ] && rm "$MARS_SRC_DIR"
    ln -s "$MARS_PKG_DIR" "$MARS_SRC_DIR"
    echo -e "${GREEN}MARS source accessible at: $MARS_SRC_DIR${NC}"
else
    echo -e "${YELLOW}Note: MARS package directory not found in site-packages${NC}"
fi

# Optional dependencies
echo ""
echo -e "${YELLOW}Optional Dependencies${NC}"
echo "MARS can export timeline visualizations as PDF/PNG images."
echo "This requires Kaleido."
echo ""
read -p "Install Kaleido for PDF/PNG export? [y/N] " response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo -e "${CYAN}Installing Kaleido...${NC}"
    pip install kaleido
    echo -e "${GREEN}Kaleido installed${NC}"
else
    echo "Skipping Kaleido. You can install it later with:"
    echo "  $VENV_DIR/bin/pip install kaleido"
fi

# Optional: fuse-t for mounting forensic images
echo ""
echo -e "${YELLOW}Optional: fuse-t${NC}"
echo "fuse-t allows MARS to mount forensic disk images (E01, raw)"
echo "as virtual file systems for direct analysis."
echo ""

# Check if fuse-t is already installed
if command -v brew &> /dev/null && brew list --cask fuse-t &> /dev/null; then
    echo -e "${GREEN}fuse-t is already installed${NC}"
else
    read -p "Install fuse-t via Homebrew? [y/N] " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        if ! command -v brew &> /dev/null; then
            echo -e "${YELLOW}Homebrew not found. Install from https://brew.sh first.${NC}"
            echo "Then run:"
            echo "  brew tap macos-fuse-t/homebrew-cask"
            echo "  brew install fuse-t fuse-t-sshfs"
        else
            echo -e "${CYAN}Installing fuse-t...${NC}"
            brew tap macos-fuse-t/homebrew-cask
            brew install fuse-t
            brew install fuse-t-sshfs
            echo -e "${GREEN}fuse-t installed successfully${NC}"
        fi
    else
        echo "Skipping fuse-t. You can install it later with:"
        echo "  brew tap macos-fuse-t/homebrew-cask"
        echo "  brew install fuse-t fuse-t-sshfs"
    fi
fi

# Create launcher scripts
echo ""
echo -e "${CYAN}Creating launcher scripts...${NC}"
cat > "$INSTALL_DIR/mars" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/.venv/bin/activate"
exec mars "$@"
EOF
chmod +x "$INSTALL_DIR/mars"
echo -e "${GREEN}Launcher created: $INSTALL_DIR/mars${NC}"

# Create mars-sudo launcher for live system scans with elevated privileges
cat > "$INSTALL_DIR/mars-sudo" << 'EOF'
#!/bin/bash
# mars-sudo - Run MARS with elevated privileges
# Redirects uv cache to avoid permission issues after sudo exits
#
# Usage: mars-sudo [args...]
#
# This script is called automatically when user selects "Live System" scan
# and agrees to run with administrator privileges.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Use temp cache to avoid polluting user's cache with root-owned files
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-mars-sudo-$$}"

# Activate venv and run mars with sudo
exec sudo -E "$SCRIPT_DIR/.venv/bin/mars" "$@"
EOF
chmod +x "$INSTALL_DIR/mars-sudo"
echo -e "${GREEN}Sudo launcher created: $INSTALL_DIR/mars-sudo${NC}"

# Installation complete
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Installation Complete!                       ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "To run MARS:"
echo -e "  ${CYAN}$INSTALL_DIR/mars${NC}"
echo ""
echo "Or add to your PATH:"
echo -e "  ${CYAN}export PATH=\"\$PATH:$INSTALL_DIR\"${NC}"
echo ""
echo "Then run with just:"
echo -e "  ${CYAN}mars${NC}"
echo ""

# Optional: Add to PATH in shell config
echo "Would you like to add MARS to your PATH automatically?"
read -p "This will modify your shell config. [y/N] " response
if [[ "$response" =~ ^[Yy]$ ]]; then
    SHELL_CONFIG=""
    if [ -f "$HOME/.zshrc" ]; then
        SHELL_CONFIG="$HOME/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    fi

    if [ -n "$SHELL_CONFIG" ]; then
        echo "" >> "$SHELL_CONFIG"
        echo "# MARS - macOS Artifact Recovery Suite" >> "$SHELL_CONFIG"
        echo "export PATH=\"\$PATH:$INSTALL_DIR\"" >> "$SHELL_CONFIG"
        echo -e "${GREEN}Added to $SHELL_CONFIG${NC}"
        echo "Run 'source $SHELL_CONFIG' or restart your terminal."
    else
        echo -e "${YELLOW}Could not find shell config file. Add manually:${NC}"
        echo "  export PATH=\"\$PATH:$INSTALL_DIR\""
    fi
fi

echo ""
echo -e "${GREEN}Done!${NC}"
