#!/usr/bin/env python3

"""
Cleanup Orphaned Mounts Utility
by WarpedWing Labs

Cleans up orphaned E01 mounts left behind from interrupted scans.
Unmounts all ewfmount instances and removes temporary mount directories.

Usage:
    python cleanup_orphaned_mounts.py
    python cleanup_orphaned_mounts.py --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from mars.utils.debug_logger import logger


def find_ewfmount_points() -> list[str]:
    """Find all active ewfmount mount points."""
    try:
        result = subprocess.run(["mount"], check=True, capture_output=True, text=True)

        ewf_mounts = []
        seen = set()

        for line in result.stdout.split("\n"):
            if "ewfmount" in line.lower():
                # Parse: "fuse-t:/ewfmount_XXX on /private/tmp/ewfmount_XXX (nfs, ...)"
                parts = line.split(" on ")
                if len(parts) >= 2:
                    mount_point = parts[1].split(" (")[0].strip()
                    if mount_point not in seen:
                        ewf_mounts.append(mount_point)
                        seen.add(mount_point)

        return ewf_mounts

    except subprocess.CalledProcessError:
        return []


def find_temp_mount_dirs() -> list[Path]:
    """Find temporary mount directories in /tmp."""
    tmp_dir = Path("/tmp")
    temp_mounts = []

    # Find ewfmount_ directories
    for item in tmp_dir.glob("ewfmount_*"):
        if item.is_dir():
            temp_mounts.append(item)

    # Find _filesystem directories
    for item in tmp_dir.glob("*_filesystem"):
        if item.is_dir():
            temp_mounts.append(item)

    return temp_mounts


def kill_ewfmount_processes(dry_run: bool = False):
    """Kill all ewfmount processes."""
    try:
        # Find ewfmount processes
        result = subprocess.run(
            ["pgrep", "ewfmount"],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            pids = result.stdout.strip().split("\n")
            logger.info(f"\n  Found {len(pids)} ewfmount process(es)")

            for pid in pids:
                logger.debug(f" Killing PID {pid}")
                if not dry_run:
                    try:
                        subprocess.run(["kill", pid], check=True, timeout=5)
                        logger.debug("   [OK] Killed")
                    except subprocess.CalledProcessError:
                        logger.debug("   [WARNING] Could not kill")
                else:
                    logger.debug("   [DRY RUN] Would kill")

            # Wait a moment for processes to die
            if not dry_run:
                import time

                time.sleep(1)

    except Exception as e:
        logger.debug(f" [WARNING] Error finding processes: {e}")


def unmount_ewfmount(mount_point: str, dry_run: bool = False) -> bool:
    """Unmount an ewfmount mount point."""
    logger.info(f"  Unmounting: {mount_point}")

    if dry_run:
        logger.debug(" [DRY RUN] Would unmount")
        return True

    try:
        subprocess.run(
            ["umount", mount_point],
            check=True,
            capture_output=True,
            timeout=30,
        )
        logger.debug(" [OK] Unmounted")
        return True
    except subprocess.CalledProcessError as e:
        # Try force unmount
        logger.debug(" [WARNING] Normal unmount failed, trying force...")
        try:
            subprocess.run(
                ["umount", "-f", mount_point],
                check=True,
                capture_output=True,
                timeout=30,
            )
            logger.debug(" [OK] Force unmounted")
            return True
        except subprocess.CalledProcessError:
            logger.debug(f" [ERROR] Failed: {e}")
            return False
    except subprocess.TimeoutExpired:
        logger.debug(" [ERROR] Timeout")
        return False


def remove_directory(dir_path: Path, dry_run: bool = False) -> bool:
    """Remove a directory if empty."""
    logger.info(f"  Removing: {dir_path}")

    if dry_run:
        logger.debug(" [DRY RUN] Would remove")
        return True

    try:
        dir_path.rmdir()
        logger.debug(" [OK] Removed")
        return True
    except OSError as e:
        logger.debug(f" [WARNING] Could not remove: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Clean up orphaned E01 mounts and temporary directories")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be cleaned up without actually doing it",
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("Orphaned Mount Cleanup Utility")
    logger.info("=" * 80)

    if args.dry_run:
        logger.info("\n[MODE] DRY RUN - No changes will be made\n")

    # Step 1: Kill ewfmount processes
    logger.info("\n[STEP 1] Killing ewfmount processes...")
    kill_ewfmount_processes(args.dry_run)

    # Step 2: Find and unmount ewfmount instances
    logger.info("\n[STEP 2] Finding ewfmount instances...")
    ewf_mounts = find_ewfmount_points()

    if ewf_mounts:
        logger.info(f"\nFound {len(ewf_mounts)} ewfmount instance(s):")
        for mount in ewf_mounts:
            logger.info(f"  - {mount}")

        logger.info(f"\nUnmounting {len(ewf_mounts)} ewfmount instance(s)...")
        success_count = 0
        for mount in ewf_mounts:
            if unmount_ewfmount(mount, args.dry_run):
                success_count += 1

        logger.info(f"\n  Result: {success_count}/{len(ewf_mounts)} unmounted successfully")
    else:
        logger.info("  No ewfmount instances found")

    # Step 3: Find and remove temp directories
    logger.info("\n[STEP 3] Finding temporary mount directories...")
    temp_dirs = find_temp_mount_dirs()

    if temp_dirs:
        logger.info(f"\nFound {len(temp_dirs)} temporary directory(ies):")
        for dir_path in temp_dirs:
            logger.info(f"  - {dir_path}")

        logger.info(f"\nRemoving {len(temp_dirs)} temporary directory(ies)...")
        success_count = 0
        for dir_path in temp_dirs:
            if remove_directory(dir_path, args.dry_run):
                success_count += 1

        logger.info(f"\n  Result: {success_count}/{len(temp_dirs)} removed successfully")
    else:
        logger.info("  No temporary directories found")

    # Summary
    logger.info("\n" + "=" * 80)
    if args.dry_run:
        logger.info("[DRY RUN] No changes were made")
        logger.info("\nRun without --dry-run to perform cleanup:")
        logger.info("  python cleanup_orphaned_mounts.py")
    else:
        logger.info("[COMPLETE] Cleanup finished")
        logger.info("\nTo verify all mounts are cleaned up:")
        logger.info("  mount | grep ewfmount")
        logger.info("  ls -la /tmp/ | grep -E 'ewfmount|filesystem'")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    main()
