#!/usr/bin/env python3
"""
Time Machine Artifact Extractor for MARS

Extracts artifacts from Time Machine backups using the Artifact Recovery Catalog (ARC)
to find known file paths, then copies them to a working directory for candidate
processing.
"""

from __future__ import annotations

import shutil
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - used at runtime
from typing import TYPE_CHECKING

import yaml

from mars.pipeline.raw_scanner.arc_artifact_mapper import ARCArtifactMapper
from mars.utils.debug_logger import logger
from mars.utils.file_utils import FileTimestamps, compute_md5_hash, get_file_timestamps

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from datetime import datetime

    from mars.utils.time_machine_utils import TimeMachineBackup


# SQLite magic bytes for identification
SQLITE_MAGIC = b"SQLite format 3\x00"

# Default catalog path
DEFAULT_CATALOG_PATH = Path(__file__).parent.parent.parent / "catalog" / "artifact_recovery_catalog.yaml"


@dataclass
class ExtractedArtifact:
    """
    Represents an artifact extracted from a Time Machine backup.

    Attributes:
        source_path: Original path in TM backup
        working_path: Extracted copy in working directory
        artifact_name: From ARC catalog (e.g., "Safari History")
        artifact_type: "database", "cache", "log", etc.
        backup_date: When this backup was created
        backup_id: Unique backup identifier (e.g., "2026-01-19-131444")
        file_timestamps: Original file metadata from TM backup
        content_hash: MD5 hash of file content (for deduplication)
        relative_path: Path relative to filesystem root
    """

    source_path: Path
    working_path: Path
    artifact_name: str
    artifact_type: str
    backup_date: datetime
    backup_id: str  # Full timestamp like "2026-01-19-131444"
    file_timestamps: FileTimestamps
    content_hash: str
    relative_path: str


@dataclass
class ExtractionResult:
    """
    Result of extracting artifacts from one or more backups.

    Attributes:
        output_dir: Working directory containing extracted files
        artifacts: List of extracted artifacts
        skipped_duplicates: Number of files skipped due to identical content
        extraction_errors: List of (path, error) tuples for failed extractions
    """

    output_dir: Path
    artifacts: list[ExtractedArtifact] = field(default_factory=list)
    skipped_duplicates: int = 0
    extraction_errors: list[tuple[Path, str]] = field(default_factory=list)


def load_artifact_catalog(catalog_path: Path | None = None) -> dict:
    """
    Load the Artifact Recovery Catalog.

    Args:
        catalog_path: Optional path to catalog YAML. Defaults to built-in catalog.

    Returns:
        Catalog dictionary

    Raises:
        FileNotFoundError: If catalog file doesn't exist
    """
    path = catalog_path or DEFAULT_CATALOG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Catalog not found: {path}")

    with path.open() as f:
        return yaml.safe_load(f)


def iter_catalog_entries(catalog: dict) -> Iterator[tuple[str, dict, str]]:
    """
    Iterate over all entries in the artifact catalog.

    Yields:
        Tuples of (artifact_name, entry_dict, category_name)

    Example:
        >>> catalog = load_artifact_catalog()
        >>> for name, entry, category in iter_catalog_entries(catalog):
        ...     print(f"{category}/{name}: {entry.get('glob_pattern')}")
    """
    for category_name, entries in catalog.items():
        # Skip metadata and special entries
        if category_name in ("catalog_metadata", "skip_databases"):
            continue

        if not isinstance(entries, list):
            continue

        for entry in entries:
            if isinstance(entry, dict) and "name" in entry:
                yield entry["name"], entry, category_name


def get_file_type_from_entry(entry: dict) -> str:
    """
    Determine the artifact type from a catalog entry.

    Returns "database", "cache", "log", or "keychain" based on the file_type field.
    Defaults to "cache" if not specified.
    """
    file_type = entry.get("file_type", "").lower()

    if file_type == "database":
        return "database"
    if file_type == "log":
        return "log"
    if file_type == "keychain":
        return "keychain"
    # Plists, biome, cache, and other types go to caches
    return "cache"


def get_output_subdir(artifact_type: str) -> str:
    """Map artifact type to output subdirectory name."""
    if artifact_type == "database":
        return "databases"
    if artifact_type == "log":
        return "logs"
    if artifact_type == "keychain":
        return "keychains"
    return "caches"


def extract_artifacts_from_backup(
    backup: TimeMachineBackup,
    output_dir: Path,
    catalog: dict | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    seen_hashes: dict[str, Path] | None = None,
) -> ExtractionResult:
    """
    Extract artifacts from a single Time Machine backup.

    Uses the Artifact Recovery Catalog to find known file paths in the backup,
    then copies them to the output directory with proper organization.

    Args:
        backup: TimeMachineBackup to extract from
        output_dir: Root directory for extracted files
        catalog: Optional pre-loaded catalog (loads default if None)
        progress_callback: Optional callback(current, total, message)
        seen_hashes: Dict of content_hash -> first_path for deduplication.
                    Pass same dict across multiple backups to dedupe.

    Returns:
        ExtractionResult with extracted artifacts and statistics

    Example:
        >>> from mars.utils.time_machine_utils import parse_backup_manifest
        >>> backups = parse_backup_manifest(Path("/Volumes/TM/backup_manifest.plist"))
        >>> result = extract_artifacts_from_backup(backups[0], Path("/tmp/extracted"))
        >>> print(f"Extracted {len(result.artifacts)} artifacts")
    """
    if catalog is None:
        catalog = load_artifact_catalog()

    if seen_hashes is None:
        seen_hashes = {}

    # Create ARC mapper for semantic folder names
    arc_mapper = ARCArtifactMapper()

    result = ExtractionResult(output_dir=output_dir)

    # Create output subdirectories
    (output_dir / "databases").mkdir(parents=True, exist_ok=True)
    (output_dir / "caches").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / "keychains").mkdir(parents=True, exist_ok=True)

    # Collect all catalog entries for progress tracking
    entries = list(iter_catalog_entries(catalog))
    total_entries = len(entries)

    for idx, (artifact_name, entry, _category) in enumerate(entries):
        if progress_callback:
            progress_callback(idx, total_entries, f"Scanning: {artifact_name}")

        glob_pattern = entry.get("glob_pattern", "")
        if not glob_pattern:
            # Check for "primary" format used in some catalog entries
            primary = entry.get("primary", {})
            glob_pattern = primary.get("glob_pattern", "")

        if not glob_pattern:
            logger.debug(f"Skipping {artifact_name}: no glob_pattern")
            continue

        artifact_type = get_file_type_from_entry(entry)
        preserve_structure = entry.get("preserve_structure", False)

        # Glob for matching files in the backup
        try:
            matches = list(backup.data_root.glob(glob_pattern.lstrip("/")))
        except (OSError, PermissionError) as e:
            logger.warning(f"Error globbing {glob_pattern}: {e}")
            result.extraction_errors.append((backup.data_root / glob_pattern, str(e)))
            continue

        for source_path in matches:
            if not source_path.is_file():
                continue

            extracted = _extract_single_file(
                source_path=source_path,
                backup=backup,
                output_dir=output_dir,
                artifact_name=artifact_name,
                artifact_type=artifact_type,
                seen_hashes=seen_hashes,
                arc_mapper=arc_mapper,
                preserve_structure=preserve_structure,
            )

            if extracted is None:
                result.skipped_duplicates += 1
            elif isinstance(extracted, str):
                # Error message
                result.extraction_errors.append((source_path, extracted))
            else:
                result.artifacts.append(extracted)

    if progress_callback:
        progress_callback(total_entries, total_entries, "Extraction complete")

    return result


def _extract_single_file(
    source_path: Path,
    backup: TimeMachineBackup,
    output_dir: Path,
    artifact_name: str,
    artifact_type: str,
    seen_hashes: dict[str, Path],
    arc_mapper: ARCArtifactMapper,
    preserve_structure: bool = False,
) -> ExtractedArtifact | str | None:
    """
    Extract a single file from a backup.

    Args:
        source_path: Path to source file in TM backup
        backup: TimeMachineBackup object
        output_dir: Root output directory
        artifact_name: Name of artifact from ARC catalog
        artifact_type: Type (database, cache, log)
        seen_hashes: Dict for deduplication tracking
        arc_mapper: ARC artifact mapper for semantic folder names
        preserve_structure: If True, maintain full directory structure (for unified logs, etc.)

    Returns:
        - ExtractedArtifact if successful
        - None if skipped (duplicate)
        - str with error message if failed
    """
    try:
        # Get file timestamps from backup
        file_timestamps = get_file_timestamps(source_path)

        # Compute content hash for deduplication
        content_hash = compute_md5_hash(source_path)

        # Check for duplicate
        if content_hash in seen_hashes:
            logger.debug(f"Skipping duplicate: {source_path.name} (same as {seen_hashes[content_hash].name})")
            return None

        # Determine output path using ARC mapper for semantic folder names
        relative_path = str(source_path.relative_to(backup.data_root))
        backup_id = backup.backup_id
        suffix = source_path.suffix

        # Try to get semantic folder info from ARC mapper
        folder_info = arc_mapper.get_folder_info(artifact_name)

        if folder_info and not preserve_structure:
            # Use semantic folder structure from ARC catalog
            # Output: databases/Safari History/History_2026-01-19-131444.db
            type_prefix, folder_name, base_filename = folder_info
            output_subdir = output_dir / type_prefix / folder_name
            output_filename = f"{base_filename}_{backup_id}{suffix}"
            output_path = output_subdir / output_filename
            # Handle naming conflicts by adding a counter suffix
            counter = 1
            while output_path.exists():
                output_filename = f"{base_filename}_{backup_id}_{counter}{suffix}"
                output_path = output_subdir / output_filename
                counter += 1
        elif preserve_structure:
            # Maintain full directory structure (for unified logs, uuid text, etc.)
            # DON'T include backup_id in path - it breaks logarchive compatibility
            # Handle conflicts by appending backup_id to filename if needed
            if folder_info:
                type_prefix, folder_name, _ = folder_info
                # Output: logs/Unified Log (All Diagnostics)/private/var/db/diagnostics/...
                output_path = output_dir / type_prefix / folder_name / relative_path
            else:
                subdir = get_output_subdir(artifact_type)
                output_path = output_dir / subdir / artifact_name / relative_path
            # Handle conflicts by appending backup_id to filename
            if output_path.exists():
                stem = output_path.stem
                suffix = output_path.suffix
                output_path = output_path.with_name(f"{stem}_{backup_id}{suffix}")
                # If still conflicts, add counter
                counter = 1
                base_output_path = output_path
                while output_path.exists():
                    output_path = base_output_path.with_stem(f"{base_output_path.stem}_{counter}")
                    counter += 1
        else:
            # Fallback: use artifact_name as folder (no ARC mapping available)
            # Output: databases/Unknown Artifact/filename_2026-01-19-131444.db
            subdir = get_output_subdir(artifact_type)
            output_subdir = output_dir / subdir / artifact_name
            stem = source_path.stem
            output_filename = f"{stem}_{backup_id}{suffix}"
            output_path = output_subdir / output_filename
            # Handle naming conflicts by adding a counter suffix
            counter = 1
            while output_path.exists():
                output_filename = f"{stem}_{backup_id}_{counter}{suffix}"
                output_path = output_subdir / output_filename
                counter += 1

        # Create directory and copy file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, output_path)

        # Also copy WAL and SHM files for SQLite databases
        if artifact_type == "database":
            _copy_sqlite_companions(source_path, output_path)

        # Track hash for deduplication
        seen_hashes[content_hash] = output_path

        return ExtractedArtifact(
            source_path=source_path,
            working_path=output_path,
            artifact_name=artifact_name,
            artifact_type=artifact_type,
            backup_date=backup.backup_date,
            backup_id=backup_id,
            file_timestamps=file_timestamps,
            content_hash=content_hash,
            relative_path=relative_path,
        )

    except (OSError, PermissionError) as e:
        return f"Failed to extract: {e}"


def _copy_sqlite_companions(source_db: Path, dest_db: Path) -> None:
    """
    Copy SQLite companion files (WAL, SHM) if they exist.

    SQLite databases may have -wal and -shm files that contain
    uncommitted transactions. Copying them allows SQLite to merge
    the data on first open.
    """
    for suffix in ("-wal", "-shm"):
        companion = source_db.parent / f"{source_db.name}{suffix}"
        if companion.exists():
            dest_companion = dest_db.parent / f"{dest_db.name}{suffix}"
            try:
                shutil.copy2(companion, dest_companion)
                logger.debug(f"Copied companion file: {companion.name}")
            except (OSError, PermissionError) as e:
                logger.warning(f"Failed to copy {companion.name}: {e}")


def extract_from_multiple_backups(
    backups: list[TimeMachineBackup],
    output_dir: Path,
    catalog: dict | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ExtractionResult:
    """
    Extract artifacts from multiple backups with deduplication.

    Files with identical content across backups are only extracted once.
    The first occurrence (from the newest backup by default) is kept.

    Args:
        backups: List of backups to extract from (should be sorted newest first)
        output_dir: Root directory for extracted files
        catalog: Optional pre-loaded catalog
        progress_callback: Optional progress callback

    Returns:
        Combined ExtractionResult

    Example:
        >>> backups = parse_backup_manifest(manifest_path)
        >>> result = extract_from_multiple_backups(backups[:3], output_dir)
        >>> print(f"Total: {len(result.artifacts)} unique artifacts")
    """
    if catalog is None:
        catalog = load_artifact_catalog()

    # Shared hash dict for cross-backup deduplication
    seen_hashes: dict[str, Path] = {}

    combined = ExtractionResult(output_dir=output_dir)
    total_backups = len(backups)

    for idx, backup in enumerate(backups):
        backup_label = backup.backup_date.strftime("%Y-%m-%d %H:%M")

        if progress_callback:
            progress_callback(
                idx,
                total_backups,
                f"Extracting backup {idx + 1}/{total_backups}: {backup_label}",
            )

        result = extract_artifacts_from_backup(
            backup=backup,
            output_dir=output_dir,
            catalog=catalog,
            seen_hashes=seen_hashes,
        )

        combined.artifacts.extend(result.artifacts)
        combined.skipped_duplicates += result.skipped_duplicates
        combined.extraction_errors.extend(result.extraction_errors)

    if progress_callback:
        progress_callback(total_backups, total_backups, "All backups extracted")

    return combined


def get_extraction_summary(result: ExtractionResult) -> dict:
    """
    Generate a summary of extraction results.

    Returns:
        Dictionary with extraction statistics
    """
    # Count by type
    by_type: dict[str, int] = {}
    for artifact in result.artifacts:
        by_type[artifact.artifact_type] = by_type.get(artifact.artifact_type, 0) + 1

    # Count by backup date
    by_date: dict[str, int] = {}
    for artifact in result.artifacts:
        date_str = artifact.backup_date.strftime("%Y-%m-%d")
        by_date[date_str] = by_date.get(date_str, 0) + 1

    return {
        "total_extracted": len(result.artifacts),
        "skipped_duplicates": result.skipped_duplicates,
        "extraction_errors": len(result.extraction_errors),
        "by_type": by_type,
        "by_date": by_date,
        "output_dir": str(result.output_dir),
    }


def _get_database_row_count(db_path: Path) -> int:
    """
    Get total row count across all tables in a database.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Total row count across all tables (excluding sqlite_* tables),
        or 0 if database cannot be read
    """
    try:
        # Use read-only immutable mode to avoid recreating WAL/SHM files
        with sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            total = 0
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                    total += cursor.fetchone()[0]
                except Exception:  # noqa: S110, BLE001
                    pass
            return total
    except Exception:  # noqa: S110, BLE001
        return 0


def write_extraction_manifest(result: ExtractionResult, manifest_path: Path | None = None) -> Path:
    """
    Write an extraction manifest file mapping each extracted file to its metadata.

    This manifest allows the candidate pipeline to trust the ARC-based classification
    rather than re-fingerprinting files.

    Args:
        result: ExtractionResult from extraction
        manifest_path: Optional path for manifest. Defaults to output_dir/extraction_manifest.json

    Returns:
        Path to the written manifest file
    """
    import json

    if manifest_path is None:
        manifest_path = result.output_dir / "extraction_manifest.json"

    # Build manifest entries keyed by relative path from output_dir
    manifest: dict[str, dict] = {}
    for artifact in result.artifacts:
        try:
            rel_path = str(artifact.working_path.relative_to(result.output_dir))
        except ValueError:
            rel_path = str(artifact.working_path)

        entry = {
            "artifact_name": artifact.artifact_name,
            "artifact_type": artifact.artifact_type,
            "backup_id": artifact.backup_id,
            "backup_date": artifact.backup_date.isoformat(),
            "content_hash": artifact.content_hash,
            "original_path": artifact.relative_path,
            "source_path": str(artifact.source_path),
            "file_timestamps": {
                "created": artifact.file_timestamps.file_created.isoformat()
                if artifact.file_timestamps.file_created
                else None,
                "modified": artifact.file_timestamps.file_modified.isoformat()
                if artifact.file_timestamps.file_modified
                else None,
                "accessed": artifact.file_timestamps.file_accessed.isoformat()
                if artifact.file_timestamps.file_accessed
                else None,
            },
        }

        # Add row count for database artifacts
        if artifact.artifact_type == "database" and artifact.working_path.exists():
            entry["row_count"] = _get_database_row_count(artifact.working_path)

        manifest[rel_path] = entry

    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)

    logger.debug(f"Wrote extraction manifest with {len(manifest)} entries to {manifest_path}")
    return manifest_path


def load_extraction_manifest(manifest_path: Path) -> dict[str, dict] | None:
    """
    Load an extraction manifest file.

    Args:
        manifest_path: Path to manifest JSON file

    Returns:
        Dict mapping relative paths to artifact metadata, or None if not found
    """
    import json

    if not manifest_path.exists():
        return None

    with manifest_path.open() as f:
        return json.load(f)
