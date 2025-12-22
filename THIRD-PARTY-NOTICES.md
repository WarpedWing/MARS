# Third-Party Notices

MARS (the macOS Artifact Recovery Suite) is licensed under the Apache License 2.0.

This document lists third-party software components that may be included in MARS,
whether as bundled binaries, dynamically linked libraries, or build-time tools.
Each component remains licensed under its own terms. The corresponding license
texts are provided under `src/resources/licenses/`.

**Path Conventions:**

- **Source**: Paths relative to the source repository (for developers)
- **Installed**: Paths after installation (for end users)
  - The `tools/` folder in your MARS installation directory (symlink/junction to site-packages resources)
  - Example: `MARS-1.0.0-macos-arm64/tools/bin/ewfmount`

Users can replace bundled binaries with their own builds by modifying files in the `tools/` directory.

Thank you to all the developers whose hard work brought us the incredible tools below.

---

## libewf and ewfmount (modified)

**Components**

- Source: `src/resources/macos/lib/libewf.3.dylib`, `src/resources/macos/bin/ewfmount`
- Installed: `tools/lib/libewf.3.dylib`, `tools/bin/ewfmount`

**Purpose**

Support for Expert Witness Format (EWF) disk images and mounting them as
virtual file systems.

**License**

- GNU Lesser General Public License (LGPL); see:
  - `src/resources/licenses/libewf-COPYING.LESSER.txt`
  - `src/resources/licenses/libewf-COPYING.txt`

**Notes**

MARS includes a modified build of `ewfmount` with fuse-t support on macOS.
The fuse-t–related notes are documented in:

- `src/resources/licenses/libewf-fuse-t-README.txt`

The corresponding modified source code is available in the public fork of
libewf here: <https://github.com/WarpedWing/libewf>.

---

## gzrecover

**Components**

- Source: `src/resources/macos/bin/gzrecover`, `src/resources/windows/bin/gzrecover.exe`
- Installed: `tools/bin/gzrecover` (macOS), `tools\bin\gzrecover.exe` (Windows)

**Purpose**

Recovery of data from corrupted gzip files.

**License**

- GNU General Public License v2; see:
  - `src/resources/licenses/gzrecover-LICENSE-GPLv2.txt`

Additional project information:

- `src/resources/licenses/gzrecover-README.txt`

---

## sqlite_dissect / sql-dissect

**Components**

- Source: `src/resources/macos/bin/sqlite_dissect`, `src/resources/windows/bin/sqlite_dissect.exe`
- Installed: `tools/bin/sqlite_dissect` (macOS), `tools\bin\sqlite_dissect.exe` (Windows)

**Purpose**

Analysis and reconstruction of SQLite databases, including forensic recovery
of deleted or partially damaged data.

**License**

- As provided by the sqlite-dissect / sql-dissect project; see:
  - `src/resources/licenses/sqlite-dissect-LICENSE.txt`

Additional project information:

- `src/resources/licenses/sql-dissect-README.md`

**Note**

MARS uses a modified version of sqlite_dissect.
Please find the forked source code here: <https://github.com/WarpedWing/sqlite-dissect>

---

## ccl_segb (used by Biome parser)

**Files**

- `src/resources/licenses/ccl_segb_LICENSE.txt`
- `src/resources/licenses/ccl_segb-README.md`

**Purpose**

The ccl_segb library is used by the Biome parsing module
as part of MARS’s report pipeline.

**License**

- See `ccl_segb_LICENSE.txt` and `ccl_segb-README.md` for details.

Note: ccl_segb may be built from source during the MARS build process rather
than shipped as a standalone binary. The license information is included here
for clarity when the resulting binaries are distributed with MARS builds.

Source: <https://github.com/cclgroupltd/ccl-segb>

---

## LZ4

**Components**

- Source: `src/resources/windows/lib/lz4.dll`
- Installed: `tools\lib\lz4.dll` (Windows only)

**Purpose**

Fast compression/decompression for components that require LZ4.

**License**

- See `src/resources/licenses/LZ4-LICENSE.txt`.

---

## zlib

**Files**

- `src/resources/licenses/zlib-LICENSE.txt`

**Purpose**

zlib is used by bundled tools and libraries that perform gzip/DEFLATE
compression and decompression.

**License**

- See `src/resources/licenses/zlib-LICENSE.txt`.

---

## OpenSSL (libcrypto, libssl)

**Components**

- Source: `src/resources/macos/lib/libcrypto.3.dylib`, `src/resources/macos/lib/libssl.3.dylib`
- Installed: `tools/lib/libcrypto.3.dylib`, `tools/lib/libssl.3.dylib` (macOS only)

**Purpose**

Cryptographic and TLS functionality used by bundled tools and libraries.

**License**

- OpenSSL / Apache-style license; see:
  - `src/resources/licenses/openssl-LICENSE.txt`

---

## SQLite

**Components**

- Source: `src/resources/windows/bin/sqlite3.exe`
- Installed: `tools\bin\sqlite3.exe` (Windows only)

**Purpose**

Command-line SQLite client used by MARS utilities and tooling.

**License**

SQLite is in the public domain. See:

- `src/resources/licenses/sqlite-LICENSE.md`

for the SQLite project’s license notice.

---

## dfVFS (Digital Forensics Virtual File System)

MARS bundles pre-built wheels for dfVFS and its dependencies for faster
installation.

**Location**

- `wheels/macos-arm64/*.whl`
- `wheels/macos-x86_64/*.whl`
- `wheels/windows-x64/*.whl`

**Purpose**

Unified virtual file system layer for forensic access to a wide variety of
disk images and file systems.

- Source code: <https://github.com/log2timeline/dfvfs>
- Docs: <https://dfvfs.readthedocs.io/en/latest/>

**License**

- Apache License 2.0; See:
  - `src/resources/licenses/Apache-2.0.txt`

dfVFS – Digital Forensics Virtual File System
Copyright © dfVFS authors and contributors
Licensed under the Apache License, Version 2.0.

---

## dfVFS Dependencies (libyal libraries)

MARS bundles pre-built wheels for the following dfVFS dependencies.

### Apache License 2.0

- **dfvfs** - <https://github.com/log2timeline/dfvfs>
- **dfdatetime** - <https://github.com/log2timeline/dfdatetime>
- **dtfabric** - <https://github.com/libyal/dtfabric>
- **pytsk3** - <https://github.com/py4n6/pytsk>

### LGPL v3.0+ (libyal libraries)

The following libyal libraries are licensed under the GNU Lesser General
Public License version 3.0 or later. Source code is available at
<https://github.com/libyal/>

- **libbde-python** - BitLocker Drive Encryption
- **libcaes-python** - AES encryption
- **libewf-python** - Expert Witness Format (EWF/E01)
- **libfcrypto-python** - Encryption formats
- **libfsapfs-python** - Apple File System (APFS)
- **libfsext-python** - Extended File System (ext2/3/4)
- **libfsfat-python** - FAT file system
- **libfshfs-python** - HFS+ file system
- **libfsntfs-python** - NTFS file system
- **libfsxfs-python** - XFS file system
- **libfvde-python** - FileVault Drive Encryption
- **libfwnt-python** - Windows NT data types
- **libluksde-python** - LUKS Disk Encryption
- **libmodi-python** - Mac OS disk images
- **libphdi-python** - Parallels Hard Disk images
- **libqcow-python** - QCOW image format
- **libsigscan-python** - Signature scanning
- **libsmdev-python** - Storage media devices
- **libsmraw-python** - Split RAW images
- **libvhdi-python** - Virtual Hard Disk (VHD)
- **libvmdk-python** - VMware Virtual Disk (VMDK)
- **libvsapm-python** - Apple Partition Map
- **libvsgpt-python** - GUID Partition Table (GPT)
- **libvshadow-python** - Volume Shadow Copies
- **libvslvm-python** - Logical Volume Manager (LVM)

**License texts:**

- `src/resources/licenses/Apache-2.0.txt`
- `src/resources/licenses/LGPL-3.0.txt`
