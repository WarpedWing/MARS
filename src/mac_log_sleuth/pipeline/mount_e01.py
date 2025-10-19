#!/usr/bin/env python3

"""
E01 Image Mounter for Mac Log Sleuth
by WarpedWing Labs

Mounts EnCase Evidence File (E01) images for forensic analysis.
Supports ewfmount (libewf) for mounting E01 images as raw disk images,
then uses hdiutil to mount the filesystem.

Dependencies:
  - libewf (ewfmount) - Bundled in resources/macos/bin/
  - hdiutil (built-in macOS)

Usage:
    python mount_e01.py --image /path/to/evidence.E01
    python mount_e01.py --image /path/to/evidence.E01 --mount-point /Volumes/Evidence
    python mount_e01.py --unmount /Volumes/Evidence
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


class E01Mounter:
    """Handles mounting and unmounting of E01 forensic images."""

    def __init__(self):
        """Initialize the E01 mounter."""
        self.ewfmount_path = self._find_ewfmount()
        self.lib_path = self._get_lib_path()
        self.check_dependencies()

    def _find_ewfmount(self) -> Path:
        """
        Find ewfmount binary.

        Checks in order:
        1. System PATH
        2. Bundled binary in resources/macos/bin/

        Returns:
            Path to ewfmount executable
        """
        # Check PATH first
        system_ewfmount = shutil.which("ewfmount")
        if system_ewfmount:
            return Path(system_ewfmount)

        # Check bundled binary
        system_name = platform.system().lower()
        if system_name == "darwin":
            bundled_path = (
                Path(__file__).parent.parent.parent
                / "resources"
                / "macos"
                / "bin"
                / "ewfmount"
            )
            if bundled_path.exists():
                return bundled_path

        # Not found
        print("[ERROR] Error: ewfmount not found")
        print("\nInstall libewf:")
        print("  brew install libewf")
        print("\nOr download from: https://github.com/libyal/libewf")
        sys.exit(1)

    def _get_lib_path(self) -> Path | None:
        """
        Get path to bundled libraries if using bundled ewfmount.

        Returns:
            Path to lib directory or None if using system ewfmount
        """
        # Only needed if using bundled binary
        bundled_bin = (
            Path(__file__).parent.parent.parent
            / "resources"
            / "macos"
            / "bin"
            / "ewfmount"
        )

        if self.ewfmount_path == bundled_bin:
            lib_path = (
                Path(__file__).parent.parent.parent
                / "resources"
                / "macos"
                / "lib"
            )
            return lib_path if lib_path.exists() else None

        return None

    def check_dependencies(self):
        """Check if required tools are installed."""
        # Check for hdiutil (should always be present on macOS)
        if not shutil.which("hdiutil"):
            print("[ERROR] Error: hdiutil not found (required macOS tool)")
            sys.exit(1)

        # Report which ewfmount we're using
        if self.lib_path:
            print(f"[OK] Using bundled ewfmount: {self.ewfmount_path}")
            print(f"[OK] Using bundled libraries: {self.lib_path}")
        else:
            print(f"[OK] Using system ewfmount: {self.ewfmount_path}")
        print("[OK] Dependencies OK (ewfmount, hdiutil)")

    def mount_e01(
        self, e01_path: Path, raw_mount_point: Path | None = None, partition_num: int | None = None
    ) -> tuple[Path, Path | None, str]:
        """
        Mount an E01 image.

        Args:
            e01_path: Path to .E01 file (or first segment if split)
            raw_mount_point: Optional directory to mount raw image (default: temp dir)
            partition_num: Optional partition number to mount (default: auto-select)

        Returns:
            Tuple of (raw_mount_point, filesystem_mount_point, disk_device)
        """
        if not e01_path.exists():
            print(f"[ERROR] Error: E01 file not found: {e01_path}")
            sys.exit(1)

        # Step 1: Create raw mount point
        if raw_mount_point is None:
            raw_mount_point = Path(f"/tmp/ewfmount_{e01_path.stem}")

        if raw_mount_point.exists():
            print(f"[WARNING]  Warning: Raw mount point already exists: {raw_mount_point}")
            print("   If this is from a previous mount, unmount it first:")
            print(f"   umount {raw_mount_point}")
            response = input("\nContinue anyway? (y/n): ").strip().lower()
            if response != "y":
                sys.exit(1)
        else:
            raw_mount_point.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Mounting E01 Image")
        print(f"{'='*80}")
        print(f"E01 file: {e01_path}")
        print(f"Raw mount: {raw_mount_point}")

        # Step 2: Mount E01 as raw image using ewfmount
        print(f"\n[STEP] Step 1: Mounting E01 to raw image...")
        cmd = [str(self.ewfmount_path), str(e01_path), str(raw_mount_point)]
        print(f"   Command: {' '.join(cmd)}")

        # Set up environment for bundled libraries
        env = os.environ.copy()
        if self.lib_path:
            # Add bundled lib path to DYLD_LIBRARY_PATH for macOS
            if "DYLD_LIBRARY_PATH" in env:
                env["DYLD_LIBRARY_PATH"] = f"{self.lib_path}:{env['DYLD_LIBRARY_PATH']}"
            else:
                env["DYLD_LIBRARY_PATH"] = str(self.lib_path)
            print(f"   Using library path: {self.lib_path}")

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=30, env=env
            )
            if result.stdout:
                print(f"   {result.stdout}")
            print(f"   [OK] E01 mounted as raw image")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error mounting E01: {e}")
            if e.stderr:
                print(f"   {e.stderr}")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print("[ERROR] Error: ewfmount timed out (image may be very large)")
            sys.exit(1)

        # Wait for ewf1 file to appear
        time.sleep(1)

        # Step 3: Find the raw disk image (ewf1)
        raw_image = raw_mount_point / "ewf1"
        if not raw_image.exists():
            print(f"[ERROR] Error: Raw image not found at {raw_image}")
            print("   ewfmount may have failed silently")
            # Try to unmount
            subprocess.run(["umount", str(raw_mount_point)], capture_output=True)
            sys.exit(1)

        print(f"   Raw image: {raw_image}")

        # Step 4: Attach disk image without auto-mounting
        print(f"\n[STEP] Step 2: Attaching disk image...")

        cmd = [
            "hdiutil",
            "attach",
            "-nomount",
            "-readonly",
            "-imagekey",
            "diskimage-class=CRawDiskImage",
            str(raw_image),
        ]
        print(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=60
            )

            # Parse device node (e.g., /dev/disk4)
            disk_device = None
            for line in result.stdout.strip().split("\n"):
                if line.strip().startswith("/dev/disk") and "GUID_partition_scheme" in line:
                    disk_device = line.split()[0]
                    break

            if not disk_device:
                # Fallback: get first /dev/disk entry
                for line in result.stdout.strip().split("\n"):
                    if line.strip().startswith("/dev/disk"):
                        disk_device = line.split()[0]
                        break

            if not disk_device:
                print(f"[ERROR] Error: Could not determine disk device")
                print(f"   hdiutil output:\n{result.stdout}")
                subprocess.run(["umount", str(raw_mount_point)], capture_output=True)
                sys.exit(1)

            print(f"   [OK] Disk attached as: {disk_device}")
            print(f"\n   Partition layout:")

            # Show partition layout
            diskutil_result = subprocess.run(
                ["diskutil", "list", disk_device],
                capture_output=True,
                text=True,
            )
            for line in diskutil_result.stdout.strip().split("\n"):
                print(f"      {line}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error attaching disk: {e}")
            if e.stderr:
                print(f"   {e.stderr}")
            print("\n   Cleaning up raw mount...")
            subprocess.run(["umount", str(raw_mount_point)], capture_output=True)
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print("[ERROR] Error: hdiutil timed out")
            print("\n   Cleaning up raw mount...")
            subprocess.run(["umount", str(raw_mount_point)], capture_output=True)
            sys.exit(1)

        # Step 5: Mount the main HFS+ partition
        print(f"\n[STEP] Step 3: Mounting HFS+ partition...")

        # Determine which partition to mount
        hfs_partition = None

        # If user specified a partition number, use that
        if partition_num is not None:
            hfs_partition = f"{disk_device}s{partition_num}"
            print(f"   Using user-specified partition: {hfs_partition}")
        else:
            # Auto-select: Find the largest HFS+/APFS partition that's NOT a recovery partition
            partition_candidates = []

            for line in diskutil_result.stdout.strip().split("\n"):
                # Skip header lines and non-partition lines
                if not line.strip() or "TYPE NAME" in line or "#:" in line:
                    continue

                # Look for HFS+ or APFS partitions
                if "Apple_HFS" in line or "Apple_APFS" in line:
                    # Skip Recovery partitions
                    if "Recovery" in line or "Apple_Boot" in line:
                        print(f"   Skipping recovery partition: {line.strip()}")
                        continue

                    parts = line.split()
                    if len(parts) >= 4:
                        # Parse: partition_type, name, size, identifier
                        # Example: "2: Apple_HFS Macintosh HD 499.4 GB disk4s2"
                        identifier = parts[-1]  # disk4s2
                        size_str = parts[-2]    # GB/MB/KB
                        size_unit = parts[-3]   # 499.4

                        # Convert size to bytes for comparison
                        try:
                            size_value = float(size_unit)
                            if "GB" in size_str or "gb" in size_str.lower():
                                size_bytes = size_value * 1024 * 1024 * 1024
                            elif "MB" in size_str or "mb" in size_str.lower():
                                size_bytes = size_value * 1024 * 1024
                            elif "KB" in size_str or "kb" in size_str.lower():
                                size_bytes = size_value * 1024
                            else:
                                size_bytes = size_value

                            partition_candidates.append({
                                'identifier': identifier,
                                'size': size_bytes,
                                'line': line.strip()
                            })
                        except (ValueError, IndexError):
                            # If we can't parse size, add it anyway with size 0
                            partition_candidates.append({
                                'identifier': identifier,
                                'size': 0,
                                'line': line.strip()
                            })

            # Select the largest partition
            if partition_candidates:
                partition_candidates.sort(key=lambda x: x['size'], reverse=True)
                hfs_partition = partition_candidates[0]['identifier']
                print(f"   Found {len(partition_candidates)} candidate partition(s)")
                print(f"   Auto-selected largest: {partition_candidates[0]['line']}")

        if not hfs_partition:
            print(f"[WARNING]  Warning: Could not find HFS+ or APFS partition")
            print(f"   Available partitions shown above")
            print(f"   Disk device: {disk_device}")
            mount_point = None
        else:
            # Create mount point in /private/tmp to avoid /Volumes name conflicts
            fs_mount_point = Path(f"/private/tmp/{e01_path.stem}_filesystem")
            fs_mount_point.mkdir(parents=True, exist_ok=True)

            print(f"   HFS+ partition: {hfs_partition}")
            print(f"   Mount point: {fs_mount_point}")

            # Use mount_hfs with -j flag (journal) which allows mounting
            # even when there's a volume name conflict
            cmd = [
                "mount_hfs",
                "-j",
                "-o",
                "rdonly",
                f"/dev/{hfs_partition}",
                str(fs_mount_point),
            ]
            print(f"   Command: {' '.join(cmd)}")

            try:
                result = subprocess.run(
                    cmd, check=True, capture_output=True, text=True, timeout=30
                )
                mount_point = fs_mount_point
                print(f"   [OK] Filesystem mounted successfully")

            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Error mounting HFS+ partition: {e}")
                if e.stderr:
                    print(f"   {e.stderr}")
                print(f"\n   Attempting diskutil mount as fallback...")

                # Fallback: try diskutil mount
                fallback_result = subprocess.run(
                    ["diskutil", "mount", f"/dev/{hfs_partition}"],
                    capture_output=True,
                    text=True,
                )

                if fallback_result.returncode == 0:
                    # Check where it mounted
                    info_result = subprocess.run(
                        ["diskutil", "info", f"/dev/{hfs_partition}"],
                        capture_output=True,
                        text=True,
                    )
                    for line in info_result.stdout.split("\n"):
                        if "Mount Point:" in line:
                            mount_point = Path(line.split(":", 1)[1].strip())
                            print(f"   [OK] Mounted via diskutil at: {mount_point}")
                            break
                else:
                    print(f"   [WARNING]  Could not mount filesystem")
                    print(f"   You may need to mount {hfs_partition} manually")
                    mount_point = None
                    # Don't exit - we can still provide the disk device info

        # Step 6: Success summary
        print(f"\n{'='*80}")
        print(f"[OK] E01 Image Mounted Successfully")
        print(f"{'='*80}")
        print(f"Raw EWF mount: {raw_mount_point}")
        print(f"Disk device: {disk_device}")
        if mount_point:
            print(f"Filesystem mount: {mount_point}")
            print(f"\nYou can now run the exemplar scanner:")
            print(f"  python pipeline/exemplar_scanner.py --source {mount_point}")
        else:
            print(f"Filesystem: Not mounted (manual mount required)")
            print(f"\nTo manually mount partition {hfs_partition}:")
            print(f"  mkdir -p /tmp/my_mount")
            print(f"  mount_hfs -j -o rdonly /dev/{hfs_partition} /tmp/my_mount")

        print(f"\nTo unmount when done:")
        if mount_point:
            print(f"  umount {mount_point}")
        print(f"  hdiutil detach {disk_device}")
        print(f"  umount {raw_mount_point}")
        print(f"{'='*80}\n")

        return raw_mount_point, mount_point, disk_device

    def unmount_raw(self, raw_mount_point: Path):
        """
        Unmount a raw E01 mount.

        Args:
            raw_mount_point: Path to the raw mount point
        """
        if not raw_mount_point.exists():
            print(f"[WARNING]  Mount point does not exist: {raw_mount_point}")
            return

        print(f"\n[STEP] Unmounting raw E01 mount: {raw_mount_point}")

        try:
            result = subprocess.run(
                ["umount", str(raw_mount_point)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            print(f"   [OK] Raw mount unmounted")

            # Try to remove the directory
            try:
                raw_mount_point.rmdir()
                print(f"   [OK] Mount point directory removed")
            except OSError:
                print(f"   [WARNING]  Could not remove directory (may not be empty)")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error unmounting: {e}")
            if e.stderr:
                print(f"   {e.stderr}")
            sys.exit(1)

    def unmount_filesystem(self, mount_point: Path):
        """
        Unmount a filesystem mounted via hdiutil.

        Args:
            mount_point: Path to the filesystem mount point
        """
        if not mount_point.exists():
            print(f"[WARNING]  Mount point does not exist: {mount_point}")
            return

        print(f"\n[STEP] Unmounting filesystem: {mount_point}")

        try:
            result = subprocess.run(
                ["hdiutil", "detach", str(mount_point)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            print(f"   [OK] Filesystem unmounted")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error unmounting: {e}")
            if e.stderr:
                print(f"   {e.stderr}")
            sys.exit(1)

    def list_mounts(self):
        """List current ewfmount and hdiutil mounts."""
        print(f"\n{'='*80}")
        print(f"Current Mounts")
        print(f"{'='*80}\n")

        # Check for ewfmount processes
        print("[SCAN] EWF Raw Mounts:")
        try:
            result = subprocess.run(
                ["mount"], check=True, capture_output=True, text=True
            )
            ewf_mounts = [
                line for line in result.stdout.split("\n") if "ewfmount" in line.lower()
            ]

            if ewf_mounts:
                for mount in ewf_mounts:
                    print(f"   {mount}")
            else:
                print("   (none)")
        except subprocess.CalledProcessError:
            print("   Error checking mounts")

        # Check for disk images
        print("\n[SCAN] Disk Image Mounts:")
        try:
            result = subprocess.run(
                ["hdiutil", "info"], check=True, capture_output=True, text=True
            )

            # Parse for mounted images
            in_images_section = False
            for line in result.stdout.split("\n"):
                if "image-path" in line.lower():
                    in_images_section = True
                    print(f"   {line.strip()}")
                elif in_images_section and "mount-point" in line.lower():
                    print(f"   {line.strip()}")
                elif in_images_section and line.strip() == "":
                    in_images_section = False

        except subprocess.CalledProcessError:
            print("   Error checking disk images")

        print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Mount E01 forensic images for analysis"
    )
    parser.add_argument(
        "--image", type=Path, help="Path to E01 image file (or first segment)"
    )
    parser.add_argument(
        "--raw-mount-point",
        type=Path,
        help="Optional: Directory for raw mount (default: /tmp/ewfmount_*)",
    )
    parser.add_argument(
        "--partition",
        type=int,
        help="Optional: Partition number to mount (default: auto-select largest HFS+/APFS)",
    )
    parser.add_argument(
        "--unmount-raw", type=Path, help="Unmount a raw EWF mount point"
    )
    parser.add_argument(
        "--unmount-filesystem", type=Path, help="Unmount a filesystem mount point"
    )
    parser.add_argument(
        "--list", action="store_true", help="List current mounts"
    )

    args = parser.parse_args()

    mounter = E01Mounter()

    if args.list:
        mounter.list_mounts()
    elif args.unmount_raw:
        mounter.unmount_raw(args.unmount_raw)
    elif args.unmount_filesystem:
        mounter.unmount_filesystem(args.unmount_filesystem)
    elif args.image:
        mounter.mount_e01(args.image, args.raw_mount_point, args.partition)
    else:
        parser.print_help()
        print("\n[TIP] Quick start:")
        print("  1. Mount E01:")
        print("     python mount_e01.py --image /path/to/evidence.E01")
        print("\n  2. Scan the mounted image:")
        print("     python exemplar_scanner.py --source /Volumes/MountedImage")
        print("\n  3. Check current mounts:")
        print("     python mount_e01.py --list")


if __name__ == "__main__":
    main()
