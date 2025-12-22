# MARS Packaging and Distribution

This directory contains scripts for building and distributing MARS.

## For Developers: Building a Release

### Prerequisites

- Python 3.13+
- UV (recommended) or pip with build module

### Building the Wheel

```bash
# From project root
./packaging/build.sh

# Or manually:
uv build --wheel
```

The wheel will be created in `dist/mars-X.X.X-py3-none-any.whl`

### Creating a Distribution Package

1. Build the wheel:

   ```bash
   ./packaging/build.sh
   ```

2. Create distribution folder:

   ```bash
   mkdir -p MARS-1.0.0-macos
   cp dist/mars-*.whl MARS-1.0.0-macos/
   cp packaging/install-macos.sh MARS-1.0.0-macos/install.sh
   cp README.md MARS-1.0.0-macos/
   ```

3. For Windows:

   ```bash
   mkdir -p MARS-1.0.0-windows
   cp dist/mars-*.whl MARS-1.0.0-windows/
   cp packaging/install-windows.bat MARS-1.0.0-windows/install.bat
   cp README.md MARS-1.0.0-windows/
   ```

4. Create ZIP archives:

   ```bash
   zip -r MARS-1.0.0-macos.zip MARS-1.0.0-macos/
   zip -r MARS-1.0.0-windows.zip MARS-1.0.0-windows/
   ```

---

## For End Users: Installation

### System Requirements

- **macOS**: macOS 11+ (Big Sur or later), Python 3.13+
- **Windows**: Windows 10/11, Python 3.13+

### macOS Installation

1. Extract the distribution archive
2. Open Terminal in the extracted folder
3. Run the installer:

   ```bash
   chmod +x install.sh
   ./install.sh
   ```

4. Follow the prompts

### Windows Installation

1. Extract the distribution archive
2. Double-click `install.bat` (or run from Command Prompt)
3. Follow the prompts

### What Gets Installed

- MARS application in an isolated virtual environment
- `mars` launcher script in the installation directory
- Optional: Kaleido for PDF/PNG export (prompted during install)
- Optional: fuse-t for mounting forensic images (prompted during install, macOS only)

### Optional Dependencies

| Package | Purpose                                                        | Platform      |
|---------|----------------------------------------------------------------|---------------|
| kaleido | PDF/PNG export for timeline visualizations                     | macOS/Windows |
| fuse-t  | Mount forensic disk images (E01, raw) as virtual file systems  | macOS only    |

#### Installing Kaleido Later

```bash
# macOS
.venv/bin/pip install kaleido

# Windows
.venv\Scripts\pip install kaleido
```

#### Installing fuse-t Later (macOS only)

fuse-t allows MARS to mount forensic disk images directly for analysis without extracting them first. Install via Homebrew:

```bash
brew tap macos-fuse-t/homebrew-cask
brew install fuse-t fuse-t-sshfs
```

Note: Homebrew must be installed first. Get it from <https://brew.sh>

## For Developers: Installation

```bash
git clone https://github.com/WarpedWing/mars.git
cd mars
uv sync        # or: pip install -e .
uv run mars    # launch MARS
```

## Running MARS

After installation:

```bash
# From installation directory
./mars           # macOS
mars.bat         # Windows

# Or if added to PATH
mars
```

---

## Uninstallation

Simply delete the installation folder. MARS is fully self-contained.

```bash
# macOS
rm -rf /path/to/mars-installation

# Windows
rmdir /s /q C:\path\to\mars-installation
```

---

## Troubleshooting

### Python Not Found

Make sure Python 3.13+ is installed and in your PATH:

- macOS: Install from <https://python.org> or via Homebrew (`brew install python@3.13`)
- Windows: Install from <https://python.org> (check "Add Python to PATH")

### Permission Denied (macOS)

If you see permission errors when running bundled binaries:

```bash
chmod +x .venv/lib/python3.*/site-packages/resources/macos/bin/*
```

### Gatekeeper Warning (macOS)

First time running MARS or its bundled tools may trigger Gatekeeper.
Allow in System Preferences > Security & Privacy.

### Windows Defender Warning

First run may trigger Windows Defender SmartScreen.
Click "More info" > "Run anyway" if you trust the source.
