EWF Mount Helper (fuse-t)

Requirements:
- macOS with Homebrew-installed fuse-t (brew install fuse-t)
- mount helper go-nfsv4 in PATH (provided by fuse-t)

Contents:
- bin/ewfmount            — ewf mount tool linked for fuse-t
- lib/libewf.3.dylib     — bundled libewf
- lib/libssl.3.dylib     — bundled OpenSSL (if present)
- lib/libcrypto.3.dylib  — bundled OpenSSL (if present)

Usage:
1) Ensure fuse-t is installed and go-nfsv4 runs (go-nfsv4 --version).
2) From your app, run:
   bin/ewfmount [-v] [-X "volname=My EWF"] /path/to/image.E01 /path/to/mount_dir
   The EWF appears under the "fuse-t" volume; inside your mount_dir is file "ewf1" (raw device).
3) To attach the raw view as disk devices:
   hdiutil attach -nomount -imagekey diskimage-class=CRawDiskImage \
     "/Volumes/fuse-t/<mount_dir>/ewf1"
   Then use diskutil list/mount to mount partitions read-only.

Notes:
- This bundle relies on system/lib paths for libfuse-t.dylib; it does not ship fuse-t.
- RPATHs include @loader_path/../lib (this bundle) and common brew paths.
