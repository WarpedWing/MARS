#!/usr/bin/env python3
"""
Time Machine Utilities for MARS

Functions for parsing Time Machine backup manifests and enumerating
available backups for forensic analysis.
"""

from __future__ import annotations

import plistlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING

from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass
class TimeMachineBackup:
    """
    Represents a single Time Machine backup snapshot.

    Attributes:
        backup_path: Path to the backup directory (e.g., /Volumes/TM/2026-01-19-134628.previous)
        backup_date: Completion timestamp of the backup (UTC)
        start_date: Start timestamp of the backup (UTC)
        machine_name: Name of the backed-up volume (e.g., "Data")
        data_root: Path to the Data/ directory containing the filesystem
        file_count: Number of files in this backup (changed + propagated)
        logical_size: Logical size in bytes
        volume_uuids: List of volume UUIDs included in this backup
    """

    backup_path: Path
    backup_date: datetime
    start_date: datetime
    machine_name: str | None
    data_root: Path
    file_count: int
    logical_size: int
    volume_uuids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure data_root is set correctly."""
        if self.data_root is None:
            self.data_root = self.backup_path / "Data"

    @property
    def display_name(self) -> str:
        """Human-readable name for UI display."""
        return get_backup_display_name(self)

    @property
    def backup_id(self) -> str:
        """
        Unique backup identifier extracted from directory name.

        Returns the timestamp portion of the backup directory name
        (e.g., "2026-01-19-131444" from "2026-01-19-131444.previous").
        """
        name = self.backup_path.name
        # Remove common suffixes
        for suffix in (".previous", ".inprogress", ".inProgress", ".backup"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name


def find_time_machine_volume(path: Path) -> Path | None:
    """
    Detect if a path is or contains a Time Machine backup volume.

    Checks for the presence of backup_manifest.plist which is the
    definitive indicator of a Time Machine backup volume root.

    Args:
        path: Path to check (can be volume root or any subdirectory)

    Returns:
        Path to the Time Machine volume root if found, None otherwise

    Example:
        >>> from pathlib import Path
        >>> tm_root = find_time_machine_volume(Path("/Volumes/Simone 1"))
        >>> if tm_root:
        ...     print(f"Found TM volume at {tm_root}")
    """
    # Check if this is the TM root
    manifest_path = path / "backup_manifest.plist"
    if manifest_path.exists():
        return path

    # Check if any parent is the TM root (if user selected a subdirectory)
    for parent in path.parents:
        manifest_path = parent / "backup_manifest.plist"
        if manifest_path.exists():
            return parent

    return None


def parse_backup_manifest(manifest_path: Path) -> list[TimeMachineBackup]:
    """
    Parse a Time Machine backup_manifest.plist and return available backups.

    The manifest is structured as an array where:
    - Even indices (0, 2, 4...) are completion timestamps
    - Odd indices (1, 3, 5...) are metadata dictionaries

    Args:
        manifest_path: Path to backup_manifest.plist file

    Returns:
        List of TimeMachineBackup objects sorted by date (newest first)

    Raises:
        FileNotFoundError: If manifest file doesn't exist
        plistlib.InvalidFileException: If manifest is not a valid plist

    Example:
        >>> from pathlib import Path
        >>> backups = parse_backup_manifest(Path("/Volumes/TM/backup_manifest.plist"))
        >>> for b in backups:
        ...     print(f"{b.backup_date}: {b.file_count} files")
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    tm_volume = manifest_path.parent
    backup_dirs = _find_backup_directories(tm_volume)

    with manifest_path.open("rb") as f:
        manifest = plistlib.load(f)

    if not isinstance(manifest, list):
        logger.warning(f"Unexpected manifest format: {type(manifest)}")
        return []

    backups: list[TimeMachineBackup] = []

    # Process paired entries (timestamp at even, metadata at odd)
    for i in range(0, len(manifest) - 1, 2):
        completion_time = manifest[i]
        metadata = manifest[i + 1]

        if not isinstance(completion_time, datetime):
            logger.debug(f"Skipping non-datetime at index {i}: {type(completion_time)}")
            continue

        if not isinstance(metadata, dict):
            logger.debug(f"Skipping non-dict at index {i + 1}: {type(metadata)}")
            continue

        backup = _parse_backup_entry(completion_time, metadata, tm_volume, backup_dirs)
        if backup:
            backups.append(backup)

    # Sort by backup date, newest first
    backups.sort(key=lambda b: b.backup_date, reverse=True)

    return backups


def _find_backup_directories(tm_volume: Path) -> dict[str, Path]:
    """
    Find all backup directories in a Time Machine volume.

    Returns:
        Dict mapping directory stem to full path
        e.g., {"2026-01-19-134628": Path("/Volumes/TM/2026-01-19-134628.previous")}
    """
    backup_dirs: dict[str, Path] = {}

    for item in tm_volume.iterdir():
        if item.is_dir() and item.name.endswith((".previous", ".inProgress", ".backup")):
            # Extract date portion without suffix
            stem = item.name.rsplit(".", 1)[0]
            backup_dirs[stem] = item

    return backup_dirs


def _parse_backup_entry(
    completion_time: datetime,
    metadata: dict,
    tm_volume: Path,
    backup_dirs: dict[str, Path],
) -> TimeMachineBackup | None:
    """
    Parse a single backup entry from the manifest.

    Args:
        completion_time: Backup completion timestamp
        metadata: Metadata dictionary from manifest
        tm_volume: Path to Time Machine volume root
        backup_dirs: Dict of available backup directories

    Returns:
        TimeMachineBackup object or None if no matching directory found
    """
    start_date = metadata.get("startDate")
    if not isinstance(start_date, datetime):
        start_date = completion_time

    # Ensure UTC timezone
    if completion_time.tzinfo is None:
        completion_time = completion_time.replace(tzinfo=UTC)
    if start_date.tzinfo is None:
        start_date = start_date.replace(tzinfo=UTC)

    # Extract stats
    stats = metadata.get("stats", {})
    changed = stats.get("changed", {})
    propagated = stats.get("propagated", {})

    file_count = changed.get("count", 0) + propagated.get("count", 0)
    logical_size = changed.get("logicalSize", 0) + propagated.get("logicalSize", 0)

    # Extract volume info
    volume_store_info = metadata.get("volumeStoreInfo", {})
    volume_uuids = list(volume_store_info.keys())
    machine_name = _extract_machine_name(volume_store_info)

    # Find matching backup directory
    backup_path = _find_matching_backup_dir(start_date, backup_dirs, tm_volume)

    if backup_path is None:
        # No matching directory found - this may be a snapshot entry
        # without a physical directory (common in APFS TM)
        logger.debug(f"No backup directory found for {start_date}")
        return None

    data_root = backup_path / "Data"
    if not data_root.exists():
        logger.warning(f"Data directory not found in {backup_path}")
        return None

    return TimeMachineBackup(
        backup_path=backup_path,
        backup_date=completion_time,
        start_date=start_date,
        machine_name=machine_name,
        data_root=data_root,
        file_count=file_count,
        logical_size=logical_size,
        volume_uuids=volume_uuids,
    )


def _extract_machine_name(volume_store_info: dict) -> str | None:
    """Extract the primary volume name from volumeStoreInfo."""
    for _uuid, info in volume_store_info.items():
        if isinstance(info, dict):
            name = info.get("name")
            if name:
                return str(name)
    return None


def _find_matching_backup_dir(
    start_date: datetime,
    backup_dirs: dict[str, Path],
    tm_volume: Path,
) -> Path | None:
    """
    Find the backup directory matching a given start date.

    Time Machine directory names use the format: YYYY-MM-DD-HHMMSS
    (in local time, derived from the backup start time).
    """
    # Try exact match first using the start date
    # TM directories use local time, so we need to convert
    # Try both UTC and local representations
    date_formats = [
        start_date.strftime("%Y-%m-%d-%H%M%S"),  # UTC
    ]

    # Also try converting to local time (naive approach - just try common offsets)
    # This handles the UTC vs local time ambiguity in TM directory naming
    for offset_hours in range(-12, 13):
        local_time = start_date + timedelta(hours=offset_hours)
        date_formats.append(local_time.strftime("%Y-%m-%d-%H%M%S"))

    for date_str in date_formats:
        if date_str in backup_dirs:
            return backup_dirs[date_str]

    # If no exact match, return the first available directory
    # (common when TM uses snapshots within a single directory)
    if backup_dirs:
        return next(iter(backup_dirs.values()))

    return None


def get_backup_display_name(backup: TimeMachineBackup) -> str:
    """
    Generate a human-readable display name for a backup.

    Args:
        backup: TimeMachineBackup object

    Returns:
        Formatted string like "2026-01-19 13:46 (1.4M files, 297 GB)"

    Example:
        >>> backup = TimeMachineBackup(...)
        >>> print(get_backup_display_name(backup))
        '2026-01-19 13:46 (1,456,498 files, 297.0 GB)'
    """
    date_str = backup.backup_date.strftime("%Y-%m-%d %H:%M")
    files_str = f"{backup.file_count:,}" if backup.file_count else "?"
    size_str = _format_bytes(backup.logical_size) if backup.logical_size else "?"

    return f"{date_str} ({files_str} files, {size_str})"


def _format_bytes(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024  # type: ignore[assignment]
    return f"{size_bytes:.1f} PB"


def iter_backup_files(
    backup: TimeMachineBackup,
    glob_pattern: str,
) -> Iterator[Path]:
    """
    Iterate over files in a backup matching a glob pattern.

    Args:
        backup: TimeMachineBackup to search
        glob_pattern: Glob pattern relative to filesystem root
                     (e.g., "Users/*/Library/Safari/History.db")

    Yields:
        Path objects for matching files within the backup

    Example:
        >>> backup = backups[0]
        >>> for path in iter_backup_files(backup, "Users/*/Library/Safari/History.db"):
        ...     print(path)
    """
    # Glob patterns in the catalog don't have leading slash
    # They're relative to the filesystem root (Data/ in TM backup)
    pattern = glob_pattern.lstrip("/")

    try:
        yield from backup.data_root.glob(pattern)
    except (OSError, PermissionError) as e:
        logger.warning(f"Error globbing {pattern} in {backup.data_root}: {e}")


def get_relative_path(backup: TimeMachineBackup, file_path: Path) -> str:
    """
    Get the path of a file relative to the backup's filesystem root.

    Args:
        backup: TimeMachineBackup containing the file
        file_path: Absolute path to the file

    Returns:
        Path relative to the Data/ directory (e.g., "Users/john/Library/Safari/History.db")

    Example:
        >>> rel_path = get_relative_path(backup, Path("/Volumes/TM/.../Data/Users/john/file.db"))
        >>> print(rel_path)
        'Users/john/file.db'
    """
    try:
        return str(file_path.relative_to(backup.data_root))
    except ValueError:
        # File is not within the backup's data root
        return str(file_path)
