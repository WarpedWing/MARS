#!/usr/bin/env python3

"""
E01 Image Mounter for Mac Log Sleuth
by WarpedWing Labs

Mounts EnCase Evidence File (E01) images for forensic analysis.
Supports ewfmount (libewf) for mounting E01 images as raw disk images,
then uses hdiutil to mount the filesystem.

Dependencies:
  - libewf (ewfmount) - Install via: brew install libewf
  - hdiutil (built-in macOS)

Usage:
    python mount_e01.py --image /path/to/evidence.E01
    python mount_e01.py --image /path/to/evidence.E01 --mount-point /Volumes/Evidence
    python mount_e01.py --unmount /Volumes/Evidence
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path


class E01Mounter:
    """Handles mounting and unmounting of E01 forensic images."""

    def __init__(self):
        """Initialize the E01 mounter."""
        self.check_dependencies()

    def check_dependencies(self):
        """Check if required tools are installed."""
        # Check for ewfmount
        if not shutil.which("ewfmount"):
            print("[ERROR] Error: ewfmount not found")
            print("\nInstall libewf:")
            print("  brew install libewf")
            print("\nOr download from: https://github.com/libyal/libewf")
            sys.exit(1)

        # Check for hdiutil (should always be present on macOS)
        if not shutil.which("hdiutil"):
            print("[ERROR] Error: hdiutil not found (required macOS tool)")
            sys.exit(1)

        print("[OK] Dependencies OK (ewfmount, hdiutil)")

    def mount_e01(
        self, e01_path: Path, raw_mount_point: Path | None = None
    ) -> tuple[Path, Path]:
        """
        Mount an E01 image.

        Args:
            e01_path: Path to .E01 file (or first segment if split)
            raw_mount_point: Optional directory to mount raw image (default: temp dir)

        Returns:
            Tuple of (raw_mount_point, filesystem_mount_point)
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
        cmd = ["ewfmount", str(e01_path), str(raw_mount_point)]
        print(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=30
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

        # Step 4: Mount the filesystem using hdiutil
        print(f"\n[STEP] Step 2: Mounting filesystem from raw image...")

        # First, get info about the image
        try:
            info_result = subprocess.run(
                ["hdiutil", "imageinfo", str(raw_image)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            print(f"   Image info retrieved")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING]  Warning: Could not get image info: {e}")

        # Attempt to mount read-only
        cmd = [
            "hdiutil",
            "attach",
            str(raw_image),
            "-readonly",
            "-nobrowse",
            "-noverify",
        ]
        print(f"   Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd, check=True, capture_output=True, text=True, timeout=60
            )

            # Parse output to find mount point
            mount_point = None
            for line in result.stdout.strip().split("\n"):
                if "/Volumes/" in line or "/dev/" in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith("/Volumes/"):
                            mount_point = Path(part)
                            break
                if mount_point:
                    break

            if mount_point is None:
                # Try to extract from last column
                lines = result.stdout.strip().split("\n")
                if lines:
                    last_line = lines[-1]
                    parts = last_line.split()
                    if parts:
                        mount_point = Path(parts[-1])

            if mount_point and mount_point.exists():
                print(f"   [OK] Filesystem mounted at: {mount_point}")
            else:
                print(f"   [WARNING]  Warning: Could not determine mount point")
                print(f"   hdiutil output:\n{result.stdout}")
                mount_point = None

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Error mounting filesystem: {e}")
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

        # Step 5: Success summary
        print(f"\n{'='*80}")
        print(f"[OK] E01 Image Mounted Successfully")
        print(f"{'='*80}")
        print(f"Raw mount point: {raw_mount_point}")
        print(f"Filesystem mount: {mount_point}")
        print(f"\nYou can now run the exemplar scanner:")
        print(f"  python pipeline/exemplar_scanner.py --source {mount_point}")
        print(f"\nTo unmount when done:")
        print(f"  python pipeline/mount_e01.py --unmount-raw {raw_mount_point}")
        print(f"  hdiutil detach {mount_point}")
        print(f"{'='*80}\n")

        return raw_mount_point, mount_point

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
        mounter.mount_e01(args.image, args.raw_mount_point)
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
