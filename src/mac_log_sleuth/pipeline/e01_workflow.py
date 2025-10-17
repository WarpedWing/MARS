#!/usr/bin/env python3

"""
E01 Workflow for Mac Log Sleuth (FUSE-free)
by WarpedWing Labs

Alternative workflows for working with E01 images without FUSE/ewfmount:
  1. Convert E01 to raw DD image (then mount the DD)
  2. Extract specific files from E01 directly

Dependencies:
  - libewf (ewfexport, ewfinfo) - Install via: brew install libewf
  - hdiutil (built-in macOS)
  - sleuthkit (optional, for file extraction) - brew install sleuthkit

Usage:
    # Convert E01 to raw DD image
    python e01_workflow.py --convert /path/to/image.E01 --output /path/to/image.dd

    # Get info about E01
    python e01_workflow.py --info /path/to/image.E01

    # Mount a raw DD image
    python e01_workflow.py --mount-dd /path/to/image.dd
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


class E01Workflow:
    """Handle E01 forensic images without requiring FUSE."""

    def __init__(self):
        """Initialize E01 workflow handler."""
        self.check_dependencies()

    def check_dependencies(self):
        """Check for available tools."""
        tools = {
            "ewfexport": "libewf (brew install libewf)",
            "ewfinfo": "libewf (brew install libewf)",
            "hdiutil": "macOS built-in",
        }

        missing = []
        for tool, install_info in tools.items():
            if not shutil.which(tool):
                missing.append(f"  - {tool}: {install_info}")

        if missing:
            print("[WARNING] Some tools are missing:")
            for m in missing:
                print(m)
            print()
        else:
            print("[OK] All required tools found")

    def get_e01_info(self, e01_path: Path):
        """
        Get information about an E01 image.

        Args:
            e01_path: Path to E01 file
        """
        if not e01_path.exists():
            print(f"[ERROR] E01 file not found: {e01_path}")
            sys.exit(1)

        print(f"\n{'='*80}")
        print(f"E01 Image Information")
        print(f"{'='*80}")
        print(f"File: {e01_path}\n")

        try:
            result = subprocess.run(
                ["ewfinfo", str(e01_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to get E01 info: {e}")
            if e.stderr:
                print(e.stderr)
            sys.exit(1)
        except FileNotFoundError:
            print("[ERROR] ewfinfo not found. Install with: brew install libewf")
            sys.exit(1)

    def convert_e01_to_raw(self, e01_path: Path, output_path: Path):
        """
        Convert E01 to raw DD image using ewfexport.

        Args:
            e01_path: Path to E01 file
            output_path: Path for output DD file
        """
        if not e01_path.exists():
            print(f"[ERROR] E01 file not found: {e01_path}")
            sys.exit(1)

        if output_path.exists():
            print(f"[WARNING] Output file already exists: {output_path}")
            response = input("Overwrite? (y/n): ").strip().lower()
            if response != "y":
                print("Cancelled.")
                sys.exit(0)

        print(f"\n{'='*80}")
        print(f"Converting E01 to Raw DD Image")
        print(f"{'='*80}")
        print(f"Source: {e01_path}")
        print(f"Output: {output_path}")
        print(f"\nThis may take a while depending on image size...")
        print(f"{'='*80}\n")

        # Create output directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use ewfexport to convert to raw
        # Format: ewfexport -t target -f raw input.E01
        cmd = [
            "ewfexport",
            "-t", str(output_path),
            "-f", "raw",
            "-u",  # unattended mode
            str(e01_path)
        ]

        print(f"Running: {' '.join(cmd)}\n")

        try:
            result = subprocess.run(
                cmd,
                check=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # ewfexport adds .raw extension, rename if needed
            raw_file = Path(str(output_path) + ".raw")
            if raw_file.exists() and raw_file != output_path:
                raw_file.rename(output_path)

            print(f"\n[OK] Conversion complete: {output_path}")
            print(f"\nYou can now mount this DD image:")
            print(f"  python e01_workflow.py --mount-dd {output_path}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Conversion failed: {e}")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            print("[ERROR] Conversion timed out (image too large or system slow)")
            sys.exit(1)
        except FileNotFoundError:
            print("[ERROR] ewfexport not found. Install with: brew install libewf")
            sys.exit(1)

    def diagnose_dd(self, dd_path: Path):
        """
        Diagnose a DD image to determine its structure.

        Args:
            dd_path: Path to raw DD image
        """
        if not dd_path.exists():
            print(f"[ERROR] DD file not found: {dd_path}")
            return

        print(f"\n{'='*80}")
        print(f"Diagnosing DD Image")
        print(f"{'='*80}")
        print(f"Image: {dd_path}\n")

        # Check file type
        try:
            result = subprocess.run(
                ["file", str(dd_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=10
            )
            print(f"File type: {result.stdout.strip()}\n")
        except Exception:
            pass

        # Try to read partition table with fdisk
        print("Checking for partition table (MBR)...")
        try:
            result = subprocess.run(
                ["fdisk", str(dd_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.stdout:
                print(result.stdout)
        except Exception as e:
            print(f"Could not read with fdisk: {e}\n")

        # Try diskutil
        print("\nTrying diskutil...")
        try:
            # First attach without mounting to see partitions
            result = subprocess.run(
                ["hdiutil", "attach", "-nomount", str(dd_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                print(f"Attached as:\n{result.stdout}")
                dev_node = result.stdout.strip().split()[0] if result.stdout.strip() else None

                if dev_node:
                    # Try to list partitions
                    list_result = subprocess.run(
                        ["diskutil", "list", dev_node],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    if list_result.returncode == 0:
                        print(f"\nPartition layout:\n{list_result.stdout}")

                    # Detach
                    subprocess.run(
                        ["hdiutil", "detach", dev_node],
                        capture_output=True,
                        timeout=10
                    )
            else:
                print(f"Could not attach: {result.stderr}")
        except Exception as e:
            print(f"diskutil check failed: {e}")

        print(f"\n{'='*80}\n")

    def mount_raw_dd(self, dd_path: Path):
        """
        Mount a raw DD image using hdiutil.

        Args:
            dd_path: Path to raw DD image
        """
        if not dd_path.exists():
            print(f"[ERROR] DD file not found: {dd_path}")
            sys.exit(1)

        print(f"\n{'='*80}")
        print(f"Mounting Raw DD Image")
        print(f"{'='*80}")
        print(f"Image: {dd_path}\n")

        # First try standard mount
        cmd = [
            "hdiutil", "attach",
            str(dd_path),
            "-readonly",
            "-nobrowse"
        ]

        print(f"Attempting: {' '.join(cmd)}\n")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            # Parse mount point from output
            mount_point = None
            for line in result.stdout.strip().split("\n"):
                if "/Volumes/" in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith("/Volumes/"):
                            mount_point = Path(part)
                            break
                if mount_point:
                    break

            if mount_point and mount_point.exists():
                print(f"[OK] Mounted successfully at: {mount_point}")
                print(f"\n{'='*80}")
                print(f"You can now run the exemplar scanner:")
                print(f"  uv run src/mac_log_sleuth/pipeline/exemplar_scanner.py --source {mount_point}")
                print(f"\nTo unmount when done:")
                print(f"  hdiutil detach {mount_point}")
                print(f"{'='*80}\n")
            else:
                print("[WARNING] Mounted, but could not determine mount point")
                print("Check 'mount' or 'diskutil list' to find it\n")
                print("hdiutil output:")
                print(result.stdout)
        else:
            # Standard mount failed - try diagnostics
            print(f"[ERROR] Standard mount failed")
            if result.stderr:
                print(f"Error: {result.stderr}\n")

            print("Running diagnostics to determine image structure...\n")
            self.diagnose_dd(dd_path)

            print("\n[TIP] Alternative approaches:")
            print("  1. If this is a partition image (not full disk), try:")
            print(f"     hdiutil attach -nomount {dd_path}")
            print("     Then manually mount the partition with diskutil")
            print("\n  2. Use The Sleuth Kit to browse/extract files:")
            print("     brew install sleuthkit")
            print(f"     fls -r {dd_path}")
            print(f"     icat {dd_path} [inode] > outputfile")
            print("\n  3. Run the exemplar scanner on the DD directly (if ext4/APFS):")
            print("     (requires additional tools for non-HFS+ filesystems)")
            sys.exit(1)

    def unmount_volume(self, mount_point: Path):
        """
        Unmount a volume.

        Args:
            mount_point: Path to mounted volume
        """
        if not mount_point.exists():
            print(f"[WARNING] Mount point does not exist: {mount_point}")
            return

        print(f"\nUnmounting: {mount_point}")

        try:
            subprocess.run(
                ["hdiutil", "detach", str(mount_point)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            print("[OK] Unmounted successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Unmount failed: {e}")
            if e.stderr:
                print(e.stderr)
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="E01 workflow without FUSE mounting"
    )
    parser.add_argument(
        "--info",
        type=Path,
        help="Show information about an E01 image"
    )
    parser.add_argument(
        "--convert",
        type=Path,
        help="Convert E01 to raw DD image"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path for converted DD image"
    )
    parser.add_argument(
        "--mount-dd",
        type=Path,
        help="Mount a raw DD image"
    )
    parser.add_argument(
        "--unmount",
        type=Path,
        help="Unmount a volume"
    )
    parser.add_argument(
        "--diagnose",
        type=Path,
        help="Diagnose a DD image to determine structure"
    )

    args = parser.parse_args()

    workflow = E01Workflow()

    if args.info:
        workflow.get_e01_info(args.info)
    elif args.convert:
        if not args.output:
            print("[ERROR] --output required when using --convert")
            sys.exit(1)
        workflow.convert_e01_to_raw(args.convert, args.output)
    elif args.diagnose:
        workflow.diagnose_dd(args.diagnose)
    elif args.mount_dd:
        workflow.mount_raw_dd(args.mount_dd)
    elif args.unmount:
        workflow.unmount_volume(args.unmount)
    else:
        parser.print_help()
        print("\n" + "="*80)
        print("WORKFLOW OPTIONS (No FUSE required)")
        print("="*80)
        print("\nOption 1: Convert E01 to raw DD, then mount")
        print("  Step 1: Get E01 info")
        print("    python e01_workflow.py --info /path/to/image.E01")
        print("\n  Step 2: Convert to raw DD")
        print("    python e01_workflow.py --convert /path/to/image.E01 --output /path/to/image.dd")
        print("\n  Step 3: Mount the DD image")
        print("    python e01_workflow.py --mount-dd /path/to/image.dd")
        print("\n  Step 4: Run exemplar scanner")
        print("    uv run src/mac_log_sleuth/pipeline/exemplar_scanner.py --source /Volumes/MountedImage")
        print("\n  Step 5: Unmount when done")
        print("    python e01_workflow.py --unmount /Volumes/MountedImage")
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
