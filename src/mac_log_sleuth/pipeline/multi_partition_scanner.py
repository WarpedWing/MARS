#!/usr/bin/env python3

"""
Multi-Partition Forensic Scanner
by WarpedWing Labs

Scans all partitions in a forensic disk image (both macOS and Windows).
Uses The Sleuth Kit (TSK) to browse filesystems without mounting.

Handles:
  - HFS+ (macOS)
  - APFS (macOS)
  - NTFS (Windows/Boot Camp)
  - FAT32 (EFI, Boot Camp)

Dependencies:
  - sleuthkit (fls, icat, mmls, fsstat) - brew install sleuthkit
  - ntfs-3g (optional, for NTFS mounting) - brew install ntfs-3g-mac

Usage:
    # Scan a DD image (auto-detects partitions)
    python multi_partition_scanner.py --image /path/to/image.dd --output-dir ./output

    # Scan specific partition
    python multi_partition_scanner.py --device /dev/disk5s2 --output-dir ./output
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# from mac_log_sleuth.pipeline.output_structure import OutputStructure


class PartitionInfo:
    """Information about a disk partition."""

    def __init__(
        self,
        number: int,
        fs_type: str,
        label: str,
        size: int,
        start_sector: int,
        device_path: str | None = None,
    ):
        self.number = number
        self.fs_type = fs_type
        self.label = label
        self.size = size
        self.start_sector = start_sector
        self.device_path = device_path

    def __repr__(self):
        return f"Partition {self.number}: {self.fs_type} '{self.label}' ({self.size / 1e9:.1f} GB)"


class MultiPartitionScanner:
    """Scan multiple partitions from a forensic image."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.check_dependencies()

    def check_dependencies(self):
        """Check for required tools."""
        required = ["fls", "icat", "mmls", "fsstat"]
        missing = [tool for tool in required if not shutil.which(tool)]

        if missing:
            print("[ERROR] Missing required tools from The Sleuth Kit:")
            for tool in missing:
                print(f"  - {tool}")
            print("\nInstall with: brew install sleuthkit")
            sys.exit(1)

        print("[OK] Sleuth Kit tools found")

    def attach_image(self, image_path: Path) -> str | None:
        """
        Attach a DD image and return device path.

        Args:
            image_path: Path to DD image

        Returns:
            Device path (e.g., /dev/disk5) or None
        """
        print(f"\n[ATTACH] Attaching image: {image_path}")

        try:
            result = subprocess.run(
                [
                    "hdiutil",
                    "attach",
                    "-nomount",
                    "-imagekey",
                    "diskimage-class=CRawDiskImage",
                    str(image_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Extract device path
            for line in result.stdout.strip().split("\n"):
                if line.startswith("/dev/disk"):
                    device = line.split()[0]
                    print(f"[OK] Attached as {device}")
                    return device

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to attach image: {e}")
            if e.stderr:
                print(e.stderr)

        return None

    def list_partitions(self, device_or_image: str) -> list[PartitionInfo]:
        """
        List partitions using diskutil (if device) or mmls (if file).

        Args:
            device_or_image: Device path (/dev/diskX) or image file path

        Returns:
            List of PartitionInfo objects
        """
        partitions = []
        device_or_path = Path(device_or_image)

        # If it's a device, use diskutil
        if device_or_image.startswith("/dev/"):
            print(f"\n[SCAN] Listing partitions on {device_or_image}")

            try:
                result = subprocess.run(
                    ["diskutil", "list", device_or_image],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                print(result.stdout)

                # Parse diskutil output
                #    #:                       TYPE NAME                    SIZE       IDENTIFIER
                #    1:                        EFI EFI                     209.7 MB   disk5s1
                for line in result.stdout.split("\n"):
                    line_stripped = line.strip()
                    if not line_stripped:
                        continue
                    if "TYPE NAME" in line_stripped or "GUID_partition_scheme" in line_stripped:
                        continue
                    if "/dev/disk" in line_stripped and "disk image" in line_stripped:
                        continue

                    # Check if line starts with partition number
                    if ":" in line_stripped:
                        parts = line_stripped.split(maxsplit=1)
                        if len(parts) >= 2 and parts[0].rstrip(":").isdigit():
                            partition_num = int(parts[0].rstrip(":"))

                            # Parse rest of line
                            rest = parts[1].strip().split()
                            if len(rest) >= 3:
                                fs_type = rest[0]
                                identifier = rest[-1]  # Last item is identifier
                                size_str = rest[-2]  # Second to last is size

                                # Name is everything between fs_type and size
                                name = " ".join(rest[1:-2]) if len(rest) > 3 else rest[1]

                                # Convert size to bytes
                                size_bytes = self._parse_size(size_str)

                                device_path = f"{device_or_image}s{partition_num}"

                                partitions.append(
                            PartitionInfo(
                                number=partition_num,
                                fs_type=fs_type,
                                label=name,
                                size=size_bytes,
                                start_sector=0,  # Not available from diskutil
                                device_path=device_path,
                            )
                        )

            except subprocess.CalledProcessError as e:
                print(f"[WARNING] diskutil failed: {e}")

        # If it's a file, use mmls (Sleuth Kit)
        elif device_or_path.exists():
            print(f"\n[SCAN] Listing partitions in {device_or_image}")

            try:
                result = subprocess.run(
                    ["mmls", str(device_or_image)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

                print(result.stdout)

                # Parse mmls output
                for line in result.stdout.split("\n"):
                    line = line.strip()
                    if not line or line.startswith("DOS") or line.startswith("Units"):
                        continue

                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            slot = parts[0]
                            start = int(parts[2])
                            end = int(parts[3])
                            length = int(parts[4])
                            description = " ".join(parts[5:])

                            # Skip meta partitions
                            if "Meta" in slot or "Table" in description:
                                continue

                            partition_num = int(slot.rstrip(":"))

                            partitions.append(
                                PartitionInfo(
                                    number=partition_num,
                                    fs_type=self._guess_fs_type(description),
                                    label=description,
                                    size=length * 512,  # Assuming 512-byte sectors
                                    start_sector=start,
                                    device_path=None,
                                )
                            )
                        except (ValueError, IndexError):
                            continue

            except subprocess.CalledProcessError as e:
                print(f"[WARNING] mmls failed: {e}")

        return partitions

    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '499.4 GB') to bytes."""
        size_str = size_str.upper().strip()

        multipliers = {
            "B": 1,
            "KB": 1000,
            "MB": 1000**2,
            "GB": 1000**3,
            "TB": 1000**4,
            "KIB": 1024,
            "MIB": 1024**2,
            "GIB": 1024**3,
            "TIB": 1024**4,
        }

        for unit, mult in multipliers.items():
            if size_str.endswith(unit):
                try:
                    number = float(size_str[: -len(unit)].strip())
                    return int(number * mult)
                except ValueError:
                    pass

        return 0

    def _guess_fs_type(self, description: str) -> str:
        """Guess filesystem type from partition description."""
        desc_lower = description.lower()

        if "ntfs" in desc_lower or "microsoft basic data" in desc_lower:
            return "NTFS"
        elif "hfs" in desc_lower or "apple_hfs" in desc_lower:
            return "HFS+"
        elif "apfs" in desc_lower:
            return "APFS"
        elif "fat32" in desc_lower or "fat" in desc_lower:
            return "FAT32"
        elif "efi" in desc_lower:
            return "EFI"
        elif "boot" in desc_lower:
            return "Boot"
        else:
            return "Unknown"

    def scan_partition_with_tsk(
        self, device_or_image: str, partition: PartitionInfo, output_dir: Path
    ):
        """
        Scan a partition using Sleuth Kit tools.

        Args:
            device_or_image: Device path or image file
            partition: Partition information
            output_dir: Output directory for extracted files
        """
        print(f"\n[SCAN] Scanning {partition}")

        # Determine what to scan
        if partition.device_path:
            target = partition.device_path
        else:
            # Use image file with offset
            target = device_or_image

        # Get filesystem info
        print(f"[INFO] Getting filesystem information...")
        try:
            fsstat_result = subprocess.run(
                ["fsstat", target],
                capture_output=True,
                text=True,
                timeout=30,
            )

            if fsstat_result.returncode == 0:
                print(f"Filesystem type confirmed: {partition.fs_type}")
                # Save fsstat output
                fsstat_file = output_dir / f"partition_{partition.number}_fsstat.txt"
                fsstat_file.write_text(fsstat_result.stdout)
                print(f"[SAVED] {fsstat_file}")

        except subprocess.TimeoutExpired:
            print("[WARNING] fsstat timed out")

        # List interesting directories based on OS
        fs_type_lower = partition.fs_type.lower()
        if any(x in fs_type_lower for x in ["hfs", "apfs", "apple"]):
            self._scan_macos_partition(target, partition, output_dir)
        elif any(x in fs_type_lower for x in ["ntfs", "microsoft", "basic data"]):
            self._scan_windows_partition(target, partition, output_dir)

    def _scan_macos_partition(
        self, target: str, partition: PartitionInfo, output_dir: Path
    ):
        """Scan macOS partition for databases."""
        print(f"[macOS] Scanning for user databases...")

        # Common paths to check
        paths_to_check = [
            "/Users",
            "/Library/Application Support/com.apple.TCC",
            "/private/var/db/diagnostics",
        ]

        file_list_path = output_dir / f"partition_{partition.number}_macos_files.txt"

        try:
            # Get recursive file listing
            result = subprocess.run(
                ["fls", "-r", "-p", target],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for large partitions
            )

            if result.returncode == 0:
                file_list_path.write_text(result.stdout)
                print(f"[SAVED] File listing: {file_list_path}")

                # Filter for database files
                db_files = [
                    line
                    for line in result.stdout.split("\n")
                    if line.endswith(".db")
                    or line.endswith(".sqlite")
                    or "History" in line
                    or "chat.db" in line
                ]

                if db_files:
                    db_list_path = output_dir / f"partition_{partition.number}_databases.txt"
                    db_list_path.write_text("\n".join(db_files))
                    print(f"[FOUND] {len(db_files)} potential database files")
                    print(f"[SAVED] Database list: {db_list_path}")

        except subprocess.TimeoutExpired:
            print("[WARNING] File listing timed out (partition too large)")

    def _scan_windows_partition(
        self, target: str, partition: PartitionInfo, output_dir: Path
    ):
        """Scan Windows/NTFS partition for databases."""
        print(f"[Windows] Scanning for browser databases...")

        file_list_path = output_dir / f"partition_{partition.number}_windows_files.txt"

        try:
            # Get recursive file listing
            result = subprocess.run(
                ["fls", "-r", "-p", target],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                file_list_path.write_text(result.stdout)
                print(f"[SAVED] File listing: {file_list_path}")

                # Filter for interesting files
                interesting = [
                    line
                    for line in result.stdout.split("\n")
                    if any(
                        keyword in line.lower()
                        for keyword in [
                            "appdata",
                            "chrome",
                            "firefox",
                            "edge",
                            ".db",
                            ".sqlite",
                            "ntuser.dat",
                        ]
                    )
                ]

                if interesting:
                    interesting_path = (
                        output_dir
                        / f"partition_{partition.number}_interesting_files.txt"
                    )
                    interesting_path.write_text("\n".join(interesting))
                    print(f"[FOUND] {len(interesting)} interesting files")
                    print(f"[SAVED] {interesting_path}")

        except subprocess.TimeoutExpired:
            print("[WARNING] File listing timed out")

    def detach_image(self, device: str):
        """Detach an attached disk image."""
        print(f"\n[DETACH] Detaching {device}")

        try:
            subprocess.run(
                ["hdiutil", "detach", device], check=True, timeout=30
            )
            print(f"[OK] Detached {device}")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Failed to detach: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Scan all partitions in a forensic disk image"
    )
    parser.add_argument(
        "--image", type=Path, help="Path to DD image file"
    )
    parser.add_argument(
        "--device", type=str, help="Already-attached device (e.g., /dev/disk5)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./partition_scan_output"),
        help="Output directory for scan results",
    )
    parser.add_argument(
        "--partition",
        type=int,
        help="Scan only this partition number (default: scan all)",
    )

    args = parser.parse_args()

    if not args.image and not args.device:
        parser.print_help()
        print("\n[ERROR] Must specify either --image or --device")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scanner = MultiPartitionScanner(args.output_dir)

    device_to_scan = None
    attached_device = None

    try:
        # Attach image if needed
        if args.image:
            if not args.image.exists():
                print(f"[ERROR] Image not found: {args.image}")
                sys.exit(1)

            attached_device = scanner.attach_image(args.image)
            if not attached_device:
                print("[ERROR] Failed to attach image")
                sys.exit(1)

            device_to_scan = attached_device
        else:
            device_to_scan = args.device

        # List partitions
        partitions = scanner.list_partitions(device_to_scan)

        if not partitions:
            print("[WARNING] No partitions found")
            return

        print(f"\n[INFO] Found {len(partitions)} partition(s)")

        # Filter by partition number if specified
        if args.partition:
            partitions = [p for p in partitions if p.number == args.partition]
            if not partitions:
                print(f"[ERROR] Partition {args.partition} not found")
                sys.exit(1)

        # Scan each partition
        for partition in partitions:
            part_output_dir = args.output_dir / f"partition_{partition.number}"
            part_output_dir.mkdir(exist_ok=True)

            scanner.scan_partition_with_tsk(
                device_to_scan, partition, part_output_dir
            )

        # Save summary
        summary = {
            "image": str(args.image) if args.image else None,
            "device": device_to_scan,
            "partitions": [
                {
                    "number": p.number,
                    "type": p.fs_type,
                    "label": p.label,
                    "size_bytes": p.size,
                }
                for p in partitions
            ],
        }

        summary_path = args.output_dir / "scan_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n[OK] Scan complete!")
        print(f"[REPORT] Output directory: {args.output_dir}")
        print(f"[REPORT] Summary: {summary_path}")

    finally:
        # Detach if we attached
        if attached_device:
            scanner.detach_image(attached_device)


if __name__ == "__main__":
    main()
