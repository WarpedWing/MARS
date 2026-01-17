#!/usr/bin/env python3
"""
File Utilities for MARS

Common file operations used across the codebase.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path


# Standard kwargs for creating directories with parent creation and no errors if exists
MKDIR_KWARGS = {"parents": True, "exist_ok": True}


def compute_md5_hash(file_path: Path) -> str:
    """
    Compute MD5 hash of a file.

    Uses chunked reading to handle large files efficiently without
    loading entire file into memory.

    Args:
        file_path: Path to file to hash

    Returns:
        MD5 hash as hexadecimal string

    Example:
        >>> from pathlib import Path
        >>> hash_value = compute_md5_hash(Path("/path/to/file.db"))
        >>> logger.info(hash_value)
        'a1b2c3d4e5f6...'
    """
    md5 = hashlib.md5()
    with file_path.open("rb") as f:
        # Read in 8KB chunks to handle large files
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


# ============================================================================
# JSONL (JSON Lines) Utilities
# ============================================================================


def iter_jsonl(path: Path, skip_errors: bool = True) -> Iterator[dict]:
    """
    Iterate over records in a JSONL file (memory-efficient for large files).

    Args:
        path: Path to JSONL file
        skip_errors: If True, skip malformed lines with a warning. If False, raise exception.

    Yields:
        Parsed JSON records as dictionaries

    Example:
        >>> from pathlib import Path
        >>> for record in iter_jsonl(Path("data.jsonl")):
        ...     logger.info(record["id"])
    """

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                if skip_errors:
                    import sys

                    logger.debug(
                        f"Warning: Failed to parse line {line_num} in {path.name}: {e}",
                        file=sys.stderr,
                    )
                    continue
                raise


def read_jsonl(path: Path, filter_type: str | None = None, skip_errors: bool = True) -> list[dict]:
    """
    Read all records from a JSONL file into a list.

    Args:
        path: Path to JSONL file
        filter_type: Optional type filter - only return records where record["type"] == filter_type
        skip_errors: If True, skip malformed lines with a warning. If False, raise exception.

    Returns:
        List of parsed JSON records

    Example:
        >>> from pathlib import Path
        >>> records = read_jsonl(Path("results.jsonl"), filter_type="case")
        >>> logger.info(f"Found {len(records)} case records")
    """
    records = []
    for record in iter_jsonl(path, skip_errors=skip_errors):
        if filter_type is None or record.get("type") == filter_type:
            records.append(record)
    return records


def write_jsonl(path: Path, records: Iterable[dict], append: bool = False) -> None:
    """
    Write records to a JSONL file.

    Args:
        path: Path to JSONL file
        records: Iterable of dictionaries to write
        append: If True, append to existing file. If False, overwrite.

    Example:
        >>> from pathlib import Path
        >>> records = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
        >>> write_jsonl(Path("output.jsonl"), records)
    """
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    """
    Append a single record to a JSONL file.

    Args:
        path: Path to JSONL file
        record: Dictionary to append

    Example:
        >>> from pathlib import Path
        >>> append_jsonl(Path("log.jsonl"), {"timestamp": "2025-01-01", "event": "scan_complete"})
    """
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ============================================================================
# File Timestamp Utilities
# ============================================================================


@dataclass
class FileTimestamps:
    """
    Original file timestamps with forensic significance.

    These timestamps come from the filesystem metadata of the original file,
    NOT when MARS processed the file. Important for forensic traceability.

    Attributes:
        file_created: File creation time (st_birthtime on macOS, st_ctime fallback)
        file_modified: File modification time (st_mtime)
        file_accessed: File access time (st_atime)
    """

    file_created: datetime | None
    file_modified: datetime | None
    file_accessed: datetime | None

    def to_dict(self) -> dict[str, str | None]:
        """Convert to dictionary with ISO format strings for JSON serialization."""
        return {
            "file_created": self.file_created.isoformat() if self.file_created else None,
            "file_modified": self.file_modified.isoformat() if self.file_modified else None,
            "file_accessed": self.file_accessed.isoformat() if self.file_accessed else None,
        }


def get_file_timestamps(path: Path) -> FileTimestamps:
    """
    Extract original file timestamps from a file path.

    Args:
        path: Path to the file

    Returns:
        FileTimestamps with created, modified, and accessed times (UTC).
        Returns None for individual timestamps if file is inaccessible.

    Note:
        - file_created uses st_birthtime on macOS (true creation time),
          falls back to st_ctime on Linux (inode change time)
        - All times are converted to UTC for consistency
        - If file cannot be accessed, returns FileTimestamps with all None values

    Example:
        >>> from pathlib import Path
        >>> timestamps = get_file_timestamps(Path("/path/to/file.db"))
        >>> print(timestamps.file_modified.isoformat())
        '2025-01-15T10:30:00+00:00'
    """
    try:
        stat = path.stat()

        # st_birthtime is macOS-specific (true file creation time)
        # Falls back to st_ctime on Linux (inode change time, not ideal but best available)
        birth_ts = getattr(stat, "st_birthtime", stat.st_ctime)
        file_created = datetime.fromtimestamp(birth_ts, UTC)

        file_modified = datetime.fromtimestamp(stat.st_mtime, UTC)
        file_accessed = datetime.fromtimestamp(stat.st_atime, UTC)

        return FileTimestamps(
            file_created=file_created,
            file_modified=file_modified,
            file_accessed=file_accessed,
        )
    except (OSError, PermissionError):
        return FileTimestamps(
            file_created=None,
            file_modified=None,
            file_accessed=None,
        )


def migrate_provenance_fields(provenance: dict) -> dict:
    """
    Migrate old provenance field names to new naming convention.

    This provides backward compatibility when loading provenance files
    created before the timestamp field renaming.

    Old → New field mappings:
        - created_time → file_created
        - modified_time → file_modified
        - timestamp → processed_at (MARS processing time)

    Args:
        provenance: Provenance dictionary (modified in place)

    Returns:
        The same dictionary with migrated field names

    Example:
        >>> from pathlib import Path
        >>> import json
        >>> with Path("file.provenance.json").open() as f:
        ...     prov = json.load(f)
        >>> prov = migrate_provenance_fields(prov)
    """
    # Migrate created_time → file_created
    if "created_time" in provenance and "file_created" not in provenance:
        provenance["file_created"] = provenance.pop("created_time")

    # Migrate modified_time → file_modified
    if "modified_time" in provenance and "file_modified" not in provenance:
        provenance["file_modified"] = provenance.pop("modified_time")

    # Migrate timestamp → processed_at (only if it represents MARS processing time)
    # Note: Some old files may have "timestamp" as the MARS processing time
    if "timestamp" in provenance and "processed_at" not in provenance:
        provenance["processed_at"] = provenance.pop("timestamp")

    return provenance
