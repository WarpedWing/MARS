#!/usr/bin/env python3

"""
E01 Image Mounter for MARS
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

from mars.utils.debug_logger import logger


class E01Mounter:
    """Handles mounting and unmounting of E01 forensic images."""

    def __init__(self):
        """Initialize the E01 mounter."""
        self.ewfmount_path = self._find_ewfmount()
        self.lib_path = self._get_lib_path()
        self.use_sudo = False  # Track if sudo is being used for mount operations
        self.check_dependencies()

    def _find_ewfmount(self) -> Path:
        """
        Find ewfmount binary.

        Returns:
            Path to bundled ewfmount executable (built with fuse-t support)

        Raises:
            SystemExit if bundled ewfmount not found
        """
        # Only use bundled binary (built with fuse-t support for macOS)
        system_name = platform.system().lower()
        if system_name == "darwin":
            # Path: src/mars/pipeline/mount_utils/e01_mounter.py
            # Need to go up to src/, then to resources/
            bundled_path = Path(__file__).parent.parent.parent.parent / "resources" / "macos" / "bin" / "ewfmount"
            if bundled_path.exists():
                return bundled_path

        # Not found - exit with clear error
        logger.error("Error: Bundled ewfmount not found")
        logger.info("\nThe bundled ewfmount binary (built with fuse-t support) is required.")
        logger.info("Expected location: resources/macos/bin/ewfmount")
        logger.info("\nNote: System ewfmount (e.g., from homebrew) is NOT compatible.")
        logger.info("The bundled version is specifically built for fuse-t on macOS.")
        sys.exit(1)

    def _get_lib_path(self) -> Path | None:
        """
        Get path to bundled libraries if using bundled ewfmount.

        Returns:
            Path to lib directory or None if using system ewfmount
        """
        # Only needed if using bundled binary
        bundled_bin = Path(__file__).parent.parent.parent.parent / "resources" / "macos" / "bin" / "ewfmount"

        if self.ewfmount_path == bundled_bin:
            lib_path = Path(__file__).parent.parent.parent.parent / "resources" / "macos" / "lib"
            return lib_path if lib_path.exists() else None

        return None

    def check_dependencies(self):
        """Check if required tools are installed."""
        # Check for hdiutil (should always be present on macOS)
        if not shutil.which("hdiutil"):
            logger.error("Error: hdiutil not found (required macOS tool)")
            sys.exit(1)

        # Report ewfmount location
        logger.debug(f"Using bundled ewfmount: {self.ewfmount_path}")
        if self.lib_path:
            logger.debug(f"Using bundled libraries: {self.lib_path}")
        logger.debug("Dependencies OK (ewfmount with fuse-t, hdiutil)")

    def _cleanup_orphaned_ewfmount(self, image_stem: str):
        """
        Kill any orphaned ewfmount processes for this specific image.

        Args:
            image_stem: Stem of the E01 filename (e.g., "01955-1_MBP_SATA")
        """
        try:
            # Find ewfmount processes
            result = subprocess.run(
                ["pgrep", "-f", f"ewfmount.*{image_stem}"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                pids = result.stdout.strip().split("\n")
                logger.debug(f"Found {len(pids)} orphaned ewfmount process(es)")
                for pid in pids:
                    try:  # noqa: SIM105
                        subprocess.run(
                            ["kill", pid],
                            capture_output=True,
                            timeout=5,
                        )
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        pass
                # Wait for processes to die
                time.sleep(0.5)

        except Exception:
            # pgrep not found or other error - not critical
            pass

    def mount_e01(
        self,
        e01_path: Path,
        raw_mount_point: Path | None = None,
        partition_num: int | None = None,
        use_sudo: bool = False,
    ) -> tuple[Path, Path | None, str]:
        """
        Mount an E01 image.

        Args:
            e01_path: Path to .E01 file (or first segment if split)
            raw_mount_point: Optional directory to mount raw image (default: temp dir)
            partition_num: Optional partition number to mount (default: auto-select)
            use_sudo: Use sudo for mounting (required for system-protected files)

        Returns:
            Tuple of (raw_mount_point, filesystem_mount_point, disk_device)
        """
        if not e01_path.exists():
            logger.error(f"Error: E01 file not found: {e01_path}")
            sys.exit(1)

        # Store sudo flag for later use in unmount operations
        self.use_sudo = use_sudo

        # Step 1: Create raw mount point
        if raw_mount_point is None:
            raw_mount_point = Path(f"/tmp/ewfmount_{e01_path.stem}")

        # Kill any orphaned ewfmount processes for this image
        self._cleanup_orphaned_ewfmount(e01_path.stem)

        # Clean up any existing mount point automatically
        if raw_mount_point.exists():
            logger.debug(f"Found existing mount point: {raw_mount_point}")
            logger.debug("Attempting automatic cleanup...")

            # Try to unmount if it's already mounted
            try:
                subprocess.run(
                    ["umount", str(raw_mount_point)],
                    capture_output=True,
                    timeout=10,
                )
                logger.debug("[OK] Unmounted successfully")
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                # Try force unmount
                try:
                    subprocess.run(
                        ["umount", "-f", str(raw_mount_point)],
                        capture_output=True,
                        timeout=10,
                    )
                    logger.debug("[OK] Force unmounted")
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                    # Can't unmount - try to remove directory if empty
                    pass

            # Try to remove the directory
            try:
                raw_mount_point.rmdir()
                logger.debug("[OK] Removed directory")
            except OSError:
                # Directory not empty or still mounted - create with timestamp suffix
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                raw_mount_point = Path(f"/tmp/ewfmount_{e01_path.stem}_{timestamp}")
                logger.info(f"   [INFO] Using alternate mount point: {raw_mount_point}")

        # Create mount point directory
        raw_mount_point.mkdir(parents=True, exist_ok=True)

        logger.info(f"\n{'=' * 80}")
        logger.info("Mounting E01 Image")
        logger.info(f"{'=' * 80}")
        logger.info(f"E01 file: {e01_path}")
        logger.info(f"Raw mount: {raw_mount_point}")

        # Step 2: Mount E01 as raw image using ewfmount
        logger.info("\n[STEP] Step 1: Mounting E01 to raw image...")
        cmd = [str(self.ewfmount_path), str(e01_path), str(raw_mount_point)]
        logger.debug(f"Command: {' '.join(cmd)}")

        # Set up environment for bundled libraries
        env = os.environ.copy()
        if self.lib_path:
            # Add bundled lib path to DYLD_LIBRARY_PATH for macOS
            if "DYLD_LIBRARY_PATH" in env:
                env["DYLD_LIBRARY_PATH"] = f"{self.lib_path}:{env['DYLD_LIBRARY_PATH']}"
            else:
                env["DYLD_LIBRARY_PATH"] = str(self.lib_path)
            logger.debug(f"Using library path: {self.lib_path}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30, env=env)
            if result.stdout:
                logger.debug(f"{result.stdout}")
            logger.debug("[OK] E01 mounted as raw image")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error mounting E01: {e}")
            if e.stderr:
                logger.debug(f"{e.stderr}")
            sys.exit(1)
        except subprocess.TimeoutExpired:
            logger.error("Error: ewfmount timed out (image may be very large)")
            sys.exit(1)

        # Wait for ewf1 file to appear
        time.sleep(1)

        # Step 3: Find the raw disk image (ewf1)
        # IMPORTANT: With fuse-t, the ewf1 file appears under /Volumes/fuse-t/<mount_point>/ewf1
        # not directly at <mount_point>/ewf1
        fuse_t_path = Path("/Volumes/fuse-t") / raw_mount_point.relative_to("/") / "ewf1"
        raw_image = raw_mount_point / "ewf1"

        # Check fuse-t path first
        if fuse_t_path.exists():
            raw_image = fuse_t_path
            logger.info(f"   Raw image (via fuse-t): {raw_image}")
        elif raw_image.exists():
            logger.info(f"   Raw image: {raw_image}")
        else:
            logger.error("Error: Raw image not found")
            logger.info(f"   Checked: {raw_image}")
            logger.info(f"   Checked: {fuse_t_path}")
            logger.debug("ewfmount may have failed silently")
            # Try to unmount
            subprocess.run(["umount", str(raw_mount_point)], capture_output=True)
            sys.exit(1)

        # Step 4: Attach disk image without auto-mounting
        logger.info("\n[STEP] Step 2: Attaching disk image...")

        cmd = [
            "hdiutil",
            "attach",
            "-nomount",
            "-readonly",
            "-imagekey",
            "diskimage-class=CRawDiskImage",
            str(raw_image),
        ]
        logger.debug(f"Command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)

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
                logger.error("Error: Could not determine disk device")
                logger.info(f"   hdiutil output:\n{result.stdout}")
                subprocess.run(["umount", str(raw_mount_point)], capture_output=True)
                sys.exit(1)

            logger.info(f"   [OK] Disk attached as: {disk_device}")
            logger.info("\n   Partition layout:")

            # Show partition layout
            diskutil_result = subprocess.run(
                ["diskutil", "list", disk_device],
                capture_output=True,
                text=True,
            )
            for line in diskutil_result.stdout.strip().split("\n"):
                logger.info(f"      {line}")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error attaching disk: {e}")
            if e.stderr:
                logger.debug(f"{e.stderr}")
            logger.info("\n   Cleaning up raw mount...")
            subprocess.run(["umount", str(raw_mount_point)], capture_output=True)
            sys.exit(1)
        except subprocess.TimeoutExpired:
            logger.error("Error: hdiutil timed out")
            logger.info("\n   Cleaning up raw mount...")
            subprocess.run(["umount", str(raw_mount_point)], capture_output=True)
            sys.exit(1)

        # Step 5: Mount the filesystem partition
        logger.info("\n[STEP] Step 3: Mounting filesystem...")

        # Determine which partition to mount
        target_partition = None
        is_apfs_container = False
        container_disk = None

        # If user specified a partition number, use that
        if partition_num is not None:
            target_partition = f"{disk_device}s{partition_num}"
            logger.info(f"   Using partition: {target_partition}")

            # Check if this is an APFS container
            for line in diskutil_result.stdout.strip().split("\n"):
                if target_partition in line and "Apple_APFS" in line and "Container" in line:
                    is_apfs_container = True
                    # Extract container disk (e.g., "Container disk5")
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.lower() == "container" and i + 1 < len(parts):
                            container_disk = parts[i + 1]
                            break
                    break
        else:
            # Find candidate partitions for user selection
            partition_candidates = []

            for line in diskutil_result.stdout.strip().split("\n"):
                # Skip header lines and non-partition lines
                if not line.strip() or "TYPE NAME" in line or "#:" in line:
                    continue

                # Look for HFS+ or APFS partitions
                if "Apple_HFS" in line or "Apple_APFS" in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        identifier = parts[-1]  # disk4s2
                        # Extract partition number (e.g., "disk4s2" -> 2)
                        part_num = identifier.split("s")[-1]

                        is_recovery = "Recovery" in line or "Apple_Boot" in line

                        # Check if this is an APFS container and extract container disk
                        candidate_container_disk = None
                        candidate_is_apfs = False
                        if "Apple_APFS" in line and "Container" in line:
                            candidate_is_apfs = True
                            # Extract container disk (e.g., "Container disk5")
                            for i, part in enumerate(parts):
                                if part.lower() == "container" and i + 1 < len(parts):
                                    candidate_container_disk = parts[i + 1]
                                    break

                        partition_candidates.append(
                            {
                                "number": part_num,
                                "identifier": identifier,
                                "line": line.strip(),
                                "is_recovery": is_recovery,
                                "is_apfs_container": candidate_is_apfs,
                                "container_disk": candidate_container_disk,
                            }
                        )

            if not partition_candidates:
                logger.debug("[WARNING] No HFS+/APFS partitions found")
            elif len(partition_candidates) == 1:
                # Only one option - auto-select
                candidate = partition_candidates[0]
                target_partition = candidate["identifier"]
                is_apfs_container = candidate["is_apfs_container"]
                container_disk = candidate["container_disk"]
                logger.info(f"   Selected partition: {candidate['line']}")
            else:
                # Multiple partitions - user must choose via --partition flag
                logger.debug("[WARNING] Multiple partitions found:")
                for candidate in partition_candidates:
                    recovery_tag = " [RECOVERY]" if candidate["is_recovery"] else ""
                    logger.info(f"      Partition {candidate['number']}: {candidate['line']}{recovery_tag}")
                logger.info("\n   Please re-run with --partition <number> to select a partition")
                logger.info(f"   Example: --partition {partition_candidates[0]['number']}")
                target_partition = None

        # Initialize mount_point before conditional assignment
        mount_point = None

        if not target_partition:
            logger.info("[WARNING]  Warning: Could not find HFS+ or APFS partition")
            logger.debug("Available partitions shown above")
            logger.info(f"   Disk device: {disk_device}")
        else:
            # Handle APFS containers vs HFS+ partitions differently
            if is_apfs_container and container_disk:
                # APFS container - mount volumes using diskutil
                logger.info(f"   APFS container: {target_partition} ({container_disk})")
                logger.debug("Mounting APFS volumes...")

                # Mount all volumes in the APFS container
                mount_result = subprocess.run(
                    ["diskutil", "mountDisk", container_disk],
                    capture_output=True,
                    text=True,
                )

                if mount_result.returncode == 0:
                    logger.debug("[OK] APFS volumes mounted")

                    # Try to get the mount point of the main volume
                    # Usually the first non-recovery, non-preboot volume
                    volumes_result = subprocess.run(
                        ["diskutil", "apfs", "list", container_disk],
                        capture_output=True,
                        text=True,
                    )

                    # Parse for volume mount points
                    mount_points = []
                    for line in volumes_result.stdout.split("\n"):
                        if "Mount Point:" in line:
                            mp = line.split(":", 1)[1].strip()
                            if mp and mp != "Not Mounted" and "/System/Volumes" not in mp:
                                mount_points.append(Path(mp))

                    if mount_points:
                        mount_point = mount_points[0]
                        logger.info(f"   Primary volume mounted at: {mount_point}")
                        if len(mount_points) > 1:
                            logger.debug("Additional volumes:")
                            for mp in mount_points[1:]:
                                logger.info(f"      {mp}")
                    else:
                        # Fallback: check /Volumes for newly mounted volumes
                        volumes_dir = Path("/Volumes")
                        existing_volumes = set(volumes_dir.iterdir()) if volumes_dir.exists() else set()
                        # The volume should have been auto-mounted to /Volumes
                        for vol in existing_volumes:
                            if vol.is_dir() and vol.name not in [
                                "Macintosh HD",
                                "Data",
                            ]:
                                mount_point = vol
                                logger.info(f"   Volume mounted at: {mount_point}")
                                break
                        else:
                            logger.warning("APFS volumes mounted but mount point not detected")
                            logger.debug("Check /Volumes for mounted volumes")
                            mount_point = None
                else:
                    logger.error(f"Failed to mount APFS volumes: {mount_result.stderr}")
                    mount_point = None

            else:
                # HFS+ partition - use mount_hfs
                # Create mount point in /private/tmp to avoid /Volumes name conflicts
                fs_mount_point = Path(f"/private/tmp/{e01_path.stem}_filesystem")

                # Clean up if mount point already exists
                if fs_mount_point.exists():
                    # Try to unmount if it's already mounted
                    try:
                        subprocess.run(
                            ["umount", str(fs_mount_point)],
                            capture_output=True,
                            timeout=10,
                        )
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                        # Try force unmount
                        try:  # noqa: SIM105
                            subprocess.run(
                                ["umount", "-f", str(fs_mount_point)],
                                capture_output=True,
                                timeout=10,
                            )
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                        ):
                            pass

                    # Remove directory if empty
                    try:
                        fs_mount_point.rmdir()
                    except OSError:
                        # Still in use - use alternate with timestamp
                        from datetime import datetime

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        fs_mount_point = Path(f"/private/tmp/{e01_path.stem}_filesystem_{timestamp}")

                fs_mount_point.mkdir(parents=True, exist_ok=True)

                logger.info(f"   HFS+ partition: {target_partition}")
                logger.debug(f"   Mount point: {fs_mount_point}")

                # Use mount_hfs with -j flag (journal) which allows mounting
                # even when there's a volume name conflict
                if use_sudo:
                    cmd = [
                        "sudo",
                        "mount_hfs",
                        "-j",
                        "-o",
                        "rdonly",
                        f"/dev/{target_partition}",
                        str(fs_mount_point),
                    ]
                else:
                    cmd = [
                        "mount_hfs",
                        "-j",
                        "-o",
                        "rdonly",
                        f"/dev/{target_partition}",
                        str(fs_mount_point),
                    ]
                logger.debug(f"Command: {' '.join(cmd)}")

                try:
                    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=30)
                    mount_point = fs_mount_point
                    logger.debug("[OK] Filesystem mounted successfully")

                except subprocess.CalledProcessError as e:
                    logger.error(f"Error mounting HFS+ partition: {e}")
                    if e.stderr:
                        logger.debug(f"{e.stderr}")
                    logger.info("\n   Attempting diskutil mount as fallback...")

                    # Fallback: try diskutil mount
                    fallback_result = subprocess.run(
                        ["diskutil", "mount", f"/dev/{target_partition}"],
                        capture_output=True,
                        text=True,
                    )

                    if fallback_result.returncode == 0:
                        # Check where it mounted
                        info_result = subprocess.run(
                            ["diskutil", "info", f"/dev/{target_partition}"],
                            capture_output=True,
                            text=True,
                        )
                        for line in info_result.stdout.split("\n"):
                            if "Mount Point:" in line:
                                mount_point = Path(line.split(":", 1)[1].strip())
                                logger.info(f"   [OK] Mounted via diskutil at: {mount_point}")
                                break
                    else:
                        logger.debug("[WARNING]  Could not mount filesystem")
                        logger.info(f"   You may need to mount {target_partition} manually")
                        mount_point = None
                        # Don't exit - we can still provide the disk device info

        # Step 6: Success summary
        logger.debug(f"\n{'=' * 80}")
        logger.debug("E01 Image Mounted Successfully")
        logger.debug(f"{'=' * 80}")
        # logger.info(f"Raw EWF mount: {raw_mount_point}")
        # logger.info(f"Disk device: {disk_device}")
        if not mount_point:
            logger.info("Filesystem: Not mounted (manual mount required)")
            if is_apfs_container and container_disk:
                logger.info(f"\nTo manually mount APFS container {target_partition} ({container_disk}):")
                logger.info(f"  diskutil mountDisk {container_disk}")
            elif target_partition:
                logger.info(f"\nTo manually mount HFS+ partition {target_partition}:")
                logger.info("  mkdir -p /tmp/my_mount")
                logger.info(f"  mount_hfs -j -o rdonly /dev/{target_partition} /tmp/my_mount")

        return raw_mount_point, mount_point, disk_device

    def unmount_raw(self, raw_mount_point: Path):
        """
        Unmount a raw E01 mount.

        Args:
            raw_mount_point: Path to the raw mount point
        """
        if not raw_mount_point.exists():
            logger.info(f"[WARNING]  Mount point does not exist: {raw_mount_point}")
            return

        logger.info(f"\n[STEP] Unmounting raw E01 mount: {raw_mount_point}")

        try:
            subprocess.run(
                ["umount", str(raw_mount_point)],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.debug("[OK] Raw mount unmounted")

            # Try to remove the directory
            try:
                raw_mount_point.rmdir()
                logger.debug("[OK] Mount point directory removed")
            except OSError:
                logger.debug("[WARNING]  Could not remove directory (may not be empty)")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error unmounting: {e}")
            if e.stderr:
                logger.debug(f"{e.stderr}")
            sys.exit(1)

    def unmount_filesystem(self, mount_point: Path):
        """
        Unmount a filesystem mounted via hdiutil.

        Args:
            mount_point: Path to the filesystem mount point
        """
        if not mount_point.exists():
            logger.info(f"[WARNING]  Mount point does not exist: {mount_point}")
            return

        logger.info(f"\n[STEP] Unmounting filesystem: {mount_point}")

        try:
            # Use umount with sudo if we mounted with sudo
            cmd = ["sudo", "umount", str(mount_point)] if self.use_sudo else ["hdiutil", "detach", str(mount_point)]

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.debug("[OK] Filesystem unmounted")

        except subprocess.CalledProcessError as e:
            logger.error(f"Error unmounting: {e}")
            if e.stderr:
                logger.debug(f"{e.stderr}")
            sys.exit(1)

    def invalidate_sudo(self):
        """Invalidate sudo credentials if they were used."""
        if self.use_sudo:
            try:
                subprocess.run(
                    ["sudo", "-k"],
                    capture_output=True,
                    timeout=5,
                    check=False,
                )
                logger.debug("[OK] Administrator privileges cleared")
            except Exception:
                # Not critical if this fails
                pass
            finally:
                self.use_sudo = False

    def detach_disk(self, disk_device: str, graceful: bool = True):
        """
        Detach a disk device using hdiutil.

        Args:
            disk_device: Disk device identifier (e.g., /dev/disk2)
            graceful: If True, print warnings on errors; if False, exit on error
        """
        if not disk_device:
            return

        logger.info(f"  [STEP] Detaching disk: {disk_device}")

        try:
            subprocess.run(
                ["hdiutil", "detach", disk_device],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            logger.debug(" [OK] Disk detached")

        except subprocess.CalledProcessError as e:
            if graceful:
                logger.info(f"    [WARNING] Failed to detach disk: {e}")
                if e.stderr:
                    logger.info(f"    {e.stderr}")
            else:
                logger.error(f"Error detaching disk: {e}")
                if e.stderr:
                    logger.debug(f"{e.stderr}")
                sys.exit(1)

        except subprocess.TimeoutExpired:
            if graceful:
                logger.debug(" [WARNING] Detach timed out")
            else:
                logger.error("Detach timed out")
                sys.exit(1)

    def cleanup_mount(
        self,
        fs_mount_point: Path | None = None,
        disk_device: str | None = None,
        raw_mount_point: Path | None = None,
        graceful: bool = True,
    ):
        """
        Complete cleanup of E01 mount (filesystem + disk + raw mount).

        Performs all necessary unmounting steps in the correct order:
        1. Unmount filesystem (if mounted)
        2. Detach disk device (if attached)
        3. Unmount raw EWF mount (if mounted)
        4. Remove mount directories

        Args:
            fs_mount_point: Filesystem mount point to unmount
            disk_device: Disk device to detach (e.g., /dev/disk2)
            raw_mount_point: Raw EWF mount point to unmount
            graceful: If True, continue with warnings on errors;
                     if False, exit on first error

        Example:
            mounter.cleanup_mount(
                fs_mount_point=Path("/tmp/fs_mount"),
                disk_device="/dev/disk2",
                raw_mount_point=Path("/tmp/ewfmount"),
                graceful=True
            )
        """
        # Step 1: Unmount filesystem
        if fs_mount_point and fs_mount_point.exists():
            logger.info(f"  [1/4] Unmounting filesystem: {fs_mount_point}")
            try:
                subprocess.run(
                    ["umount", str(fs_mount_point)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                logger.debug(" [OK] Filesystem unmounted")
            except subprocess.CalledProcessError as e:
                if graceful:
                    logger.info(f"    [WARNING] Failed to unmount filesystem: {e}")
                else:
                    logger.error(f"Error unmounting filesystem: {e}")
                    if e.stderr:
                        logger.debug(f"{e.stderr}")
                    sys.exit(1)
            except subprocess.TimeoutExpired:
                if graceful:
                    logger.debug(" [WARNING] Unmount timed out")
                else:
                    logger.error("Unmount timed out")
                    sys.exit(1)

            # Remove filesystem mount directory
            try:
                fs_mount_point.rmdir()
                logger.debug(" [OK] Mount directory removed")
            except OSError:
                if graceful:
                    logger.debug(" [WARNING] Could not remove mount directory")
                else:
                    logger.error("Could not remove mount directory")
                    sys.exit(1)

        # Step 2: Detach disk device
        if disk_device:
            logger.info(f"  [2/4] Detaching disk: {disk_device}")
            self.detach_disk(disk_device, graceful=graceful)

        # Step 3: Unmount raw EWF mount
        if raw_mount_point and raw_mount_point.exists():
            logger.info(f"  [3/4] Unmounting raw EWF: {raw_mount_point}")
            try:
                subprocess.run(
                    ["umount", str(raw_mount_point)],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                logger.debug(" [OK] Raw mount unmounted")
            except subprocess.CalledProcessError as e:
                if graceful:
                    logger.info(f"    [WARNING] Failed to unmount raw mount: {e}")
                else:
                    logger.error(f"Error unmounting raw mount: {e}")
                    if e.stderr:
                        logger.debug(f"{e.stderr}")
                    sys.exit(1)
            except subprocess.TimeoutExpired:
                if graceful:
                    logger.debug(" [WARNING] Unmount timed out")
                else:
                    logger.error("Unmount timed out")
                    sys.exit(1)

            # Remove raw mount directory
            try:
                raw_mount_point.rmdir()
                logger.debug(" [OK] Raw mount directory removed")
            except OSError:
                if graceful:
                    logger.debug(" [WARNING] Could not remove raw mount directory")
                else:
                    logger.error("Could not remove raw mount directory")
                    sys.exit(1)

        logger.info("  [4/4] Cleanup complete")

    def list_mounts(self):
        """List current ewfmount and hdiutil mounts."""
        logger.info(f"\n{'=' * 80}")
        logger.info("Current Mounts")
        logger.info(f"{'=' * 80}\n")

        # Check for ewfmount processes
        logger.info("[SCAN] EWF Raw Mounts:")
        try:
            result = subprocess.run(["mount"], check=True, capture_output=True, text=True)
            ewf_mounts = [line for line in result.stdout.split("\n") if "ewfmount" in line.lower()]

            if ewf_mounts:
                for mount in ewf_mounts:
                    logger.debug(f"{mount}")
            else:
                logger.debug("(none)")
        except subprocess.CalledProcessError:
            logger.debug("Error checking mounts")

        # Check for disk images
        logger.info("\n[SCAN] Disk Image Mounts:")
        try:
            result = subprocess.run(["hdiutil", "info"], check=True, capture_output=True, text=True)

            # Parse for mounted images
            in_images_section = False
            for line in result.stdout.split("\n"):
                if "image-path" in line.lower():
                    in_images_section = True
                    logger.debug(f"{line.strip()}")
                elif in_images_section and "mount-point" in line.lower():
                    logger.debug(f"{line.strip()}")
                elif in_images_section and line.strip() == "":
                    in_images_section = False

        except subprocess.CalledProcessError:
            logger.debug("Error checking disk images")

        logger.info(f"\n{'=' * 80}\n")


def main():
    parser = argparse.ArgumentParser(description="Mount E01 forensic images for analysis")
    parser.add_argument("--image", type=Path, help="Path to E01 image file (or first segment)")
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
    parser.add_argument("--unmount-raw", type=Path, help="Unmount a raw EWF mount point")
    parser.add_argument("--unmount-filesystem", type=Path, help="Unmount a filesystem mount point")
    parser.add_argument("--list", action="store_true", help="List current mounts")

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
        logger.info("\n[TIP] Quick start:")
        logger.info("  1. Mount E01:")
        logger.debug("  python mount_e01.py --image /path/to/evidence.E01")
        logger.info("\n  2. Scan the mounted image:")
        logger.debug("  python exemplar_scanner.py --source /Volumes/MountedImage")
        logger.info("\n  3. Check current mounts:")
        logger.debug("  python mount_e01.py --list")


if __name__ == "__main__":
    main()
