# ![MARS](https://github.com/WarpedWing/WarpedWing/blob/main/_mars_transparent.png?raw=true)<br />macOS Artifact Recovery Suite

**MARS** is a data extraction and recovery toolkit for macOS that salvages and recovers SQLite,
plist, log, and cache data from a set of raw, carved files and matches them with artifacts of
forensic interest from a reference system.

In some cases, MARS can recover **thousands more database rows and hundreds of extra days of data**
beyond what exists in the original reference files alone.

<img src="https://raw.githubusercontent.com/WarpedWing/WarpedWing/refs/heads/main/mars_ss/mars_main.png"
alt="MARS main window" />

> <sub>MARS Main Menu</sub>

<br />

## Core MARS Definitions
>
### Exemplar
>
> The set of baseline target artifacts (databases, logs, etc.) from a reference system that forms
> the "ground truth" for data recovery.
>
### Candidate
>
> An unclassified file MARS will recover then attempt to match to exemplar artifacts.
>
### Rubric
>
> A JSON schema that contains per-column metadata used for matching candidate databases.

## How It Works

> [!TIP]
> Check out the [**online help documentation**](https://warpedwing.github.io/MARS/help/mars_help.html)
>and the [**introductory video.**](https://www.youtube.com/watch?v=YKRyHVraYgI)

### Exemplar Scan

 MARS uses a catalog of known artifacts to collect target files from an [exemplar
 system](https://warpedwing.github.io/MARS/help/mars_help.html#exemplar-scan).

It can scan most disk image formats (EWF, etc.), folders, archives, and live macOS
 systems.

 <img src="https://raw.githubusercontent.com/WarpedWing/WarpedWing/refs/heads/main/mars_ss/arc.png" alt="MARS catalog" />

> <sub>ARC Management page showing groups (and number of targets)</sub>

<br />

 Artifacts with associated archives - like Powerlog's .gz backups - are automatically
 decompressed, deduplicated, and combined.

Databases are then ["fingerprinted"](https://warpedwing.github.io/MARS/help/mars_architecture.html)
column-by-column to create rubrics for matching candidates
against.

### Candidates Scan

<img src="https://raw.githubusercontent.com/WarpedWing/WarpedWing/refs/heads/main/mars_ss/l_and_f.png"
alt="MARS candidate scan" />

> <sub>MARS catalogs a powerlog database during a Candidate Scan</sub>

<br />

The [recovery and vetting process](https://warpedwing.github.io/MARS/help/mars_help.html#candidates-scan)
ensures that **all candidate data that *can be* recovered *is* recovered**.

MARS assesses and classifies the recovered data - including from within
corrupt SQLite databases - then matches it against exemplar rubrics.

Truly unrepairable databases are [byte-carved](https://warpedwing.github.io/MARS/help/pages/carver.html) with protobuf extraction
and timestamp detection for manual analysis.

### Reports

Both Exemplar and Candidate reports provide quick links to artifact folders and [module reports](https://warpedwing.github.io/MARS/help/mars_help.html#report_modules),
such as WiFi history and Biome parsing.

Data Comparison Reports show exactly how much data you've gained beyond baseline, measured in rows
and days, and include a comprehensive zoomable timeline.

<img src="https://raw.githubusercontent.com/WarpedWing/WarpedWing/refs/heads/main/mars_ss/timeline_crop.png"
alt="MARS comparison timeline" />

> <sub>Exemplar vs Candidate data timeline, based on tables with timestamp data.</sub>

<br />

### Export Options

 Export original Exemplar files, matched Candidates, or Both. The full-path option recreates the
 original file and folder structure, making the data easily parsable by external tools such as
 [mac_apt](https://github.com/ydkhatri/mac_apt), [APOLLO](https://github.com/mac4n6/APOLLO),
 [plaso](https://github.com/log2timeline/plaso), and others.

<img src="https://raw.githubusercontent.com/WarpedWing/WarpedWing/refs/heads/main/mars_ss/export_full.png"
alt="Export full path" />

> <sub>Full path exports maintain the original macOS file structure of the exemplar</sub>

<br />

The combined export deduplicates and merges data while maintaining its integrity. Discrete user and
profile account data is never mixed. An optional database source column marks each row's origin -
so you can always trace the information back to its source.

<img src="https://raw.githubusercontent.com/WarpedWing/WarpedWing/refs/heads/main/mars_ss/data_source.png"
alt="Data source column" />

> <sub>The data source column tracks exemplar, found, and carved data (and original filenames).</sub>

<br />

### Free Scan

If you just want to salvage corrupt SQLite databases, MARS can do that, too. Run the recovery
pipeline on any set of files to automatically recover as much data as possible. Try it on the
[SQLite Forensic Corpus](https://digitalcorpora.s3.amazonaws.com/s3_browser.html#corpora/sql/) to see how it works.

## Additional Features

- Easily mount EWF images directly in macOS via [FUSE-T](https://www.fuse-t.org/)
  - *(Completely avoids kernel-level FUSE install)*
- Automatic pseudo-logarchive creation, ready for [Unified Logs](https://github.com/mandiant/macos-UnifiedLogs) parsing
- [Database timeline plotter](src/mars/plotter/README.md) for SQLite with [Plotly](https://plotly.com/)
- Add and edit targets using the [Artifact Recovery Catalog (ARC) Manager](src/mars/catalog/README.md)
- Export and import anonymized exemplar catalog packages to share with other MARS users
- In-depth HTML documentation (no internet required)

<img src="https://raw.githubusercontent.com/WarpedWing/WarpedWing/refs/heads/main/mars_ss/mnt_ewf.png"
alt="EWF mounted in macOS" />

> <sub>macOS EWF image shown mounted in macOS</sub>

<br />

### v1.0 Report Modules

- [WiFi activity](src/mars/report_modules/wifi_report_generator/README.md) and location mapping
- [Biome parsing](src/mars/report_modules/biome_parser/README.md) with CCL Group's [SEGB parser](https://github.com/cclgroupltd/ccl-segb)
- [Firefox JSONLZ4](src/mars/report_modules/firefox_jsonlz4_parser/README.md) parsing
- [Firefox cache](src/mars/report_modules/firefox_cache_parser/README.md) parsing (extract images, HTML, etc.)

<img src="https://raw.githubusercontent.com/WarpedWing/WarpedWing/refs/heads/main/mars_ss/wifi_known_overlay_crop.png"
alt="Wifi known networks" />

> <sub>WiFi Report - Known Networks</sub>

<br />

<img src="https://raw.githubusercontent.com/WarpedWing/WarpedWing/refs/heads/main/mars_ss/network_loc_map.png"
alt="Wifi location map" />

> <sub>WiFi Report - interactive location map with accuracy radius</sub>

<br />

## System Requirements

- **macOS**: macOS 11+ (Big Sur or later)
- **Windows**: Windows 10/11
- **Python**: 3.13+

## Quick Start

### Installation

1. Download the latest release for your platform
2. Extract the archive to an external drive and enter the directory
3. Run the installer:

**macOS:**

```bash
chmod +x install.sh
```

```bash
./install.sh
```

**Windows:**

```bash
install.bat
```

Or double-click `install.bat`.

### What Gets Installed

- MARS application in an isolated virtual environment
- `mars` launcher script in the installation directory
- Optional: Kaleido for PDF/PNG export (prompted during install)
- Optional: fuse-t for mounting forensic images (prompted during install, macOS only)

#### Installing Kaleido Later

**macOS:**

```bash
.venv/bin/pip install kaleido
```

**Windows:**

```bash
.venv\Scripts\pip install kaleido
```

#### Installing fuse-t Later (macOS only)

fuse-t allows MARS to mount forensic disk images directly for analysis without extracting them first. Install via Homebrew:

```bash
brew tap macos-fuse-t/homebrew-cask
```

```bash
brew install fuse-t fuse-t-sshfs
```

> [!NOTE]
> Homebrew must be installed first. Get it from <https://brew.sh>

### Developer Installation

See [packaging/README.md](packaging/README.md)

## Running MARS

### From installation directory

**macOS:**

```bash
./mars
```

**Windows:**

```bash
mars.bat
```

### Or if added to PATH, from anywhere

```bash
mars
```

---

MARS uses a three-stage workflow:

1. **Create Project** – Set up a new case
2. **Exemplar Scan** – Collect baseline artifacts from known locations to establish "ground truth"
3. **Candidates Scan** – Process carved/recovered files against the exemplar to recover additional data

The TUI guides you through each step.

**Free Match Mode**: Process any set of SQLite databases through the recovery pipeline without an exemplar baseline.

> [!TIP]
> **Press `h` in the MARS console for a comprehensive help guide.**

## Uninstallation

Simply delete the installation folder. MARS is fully self-contained.

```bash
# macOS
rm -rf /path/to/mars-installation

# Windows
rmdir /s /q C:\path\to\mars-installation
```

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

## License

MARS is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

### Third-Party Licenses

This project includes bundled binaries and references third-party software.
For complete license details, see [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

Key components:

- **libewf** - Expert Witness Format library (LGPL v3)
- **gzrecover** - Corrupted gzip recovery (GPL v2)
- **sqlite_dissect** - SQLite forensic parser (DC3 License)
- **dfVFS** - Virtual file system layer (Apache 2.0)
- **zlib** - Compression library (zlib License)

Individual license files are in [src/resources/licenses/](src/resources/licenses/).

## Acknowledgments

MARS builds upon the work of several open-source projects:

- **[blackboxprotobuf](https://github.com/nccgroup/blackboxprotobuf)** (bbpb) - Protobuf decoding without .proto files
- **[unfurl](https://github.com/obsidianforensics/unfurl)** - URL and timestamp parsing
- **[libewf](https://github.com/libyal/libewf)** - Expert Witness Format library
([MARS fork](https://github.com/WarpedWing/libewf) with macOS FUSE-T support)
- **[sqlite_dissect](https://github.com/dod-cyber-crime-center/sqlite-dissect)** -
SQLite forensic parser by DC3 ([MARS fork](https://github.com/WarpedWing/sqlite-dissect))
- [SEGB Parser](https://github.com/cclgroupltd/ccl-segb) - by CCL Group

## Disclaimer

**MARS is for informational purposes only.** The data presented should be
independently verified before being relied upon for any legal, regulatory,
or evidentiary purpose.
