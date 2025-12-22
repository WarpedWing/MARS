#!/usr/bin/env python3
"""Residue Processor for db_variant_selector output.

Processes variant selector results to:
1. Extract lost_and_found tables (in some recovered variants) from matched databases into separate DBs
2. Clean up unnecessary variant files for empty/unmatched databases
3. Generate cleanup reports

Usage:
    python residue_processor.py results.jsonl --output-dir residue/
"""

from __future__ import annotations

import gc
import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mars.pipeline.lf_processor.db_reconstructor import cleanup_wal_files
from mars.utils.debug_logger import logger
from mars.utils.file_utils import read_jsonl, write_jsonl
from mars.utils.platform_utils import is_windows

from .db_variant_selector_helpers import quote_identifier

# Mapping from single-letter variant tags to full variant names
# Note: Both O and X map to "original" (same file), but X is a semantic marker
#       X = "empty database, keep original for byte carving"
#       O = "original has good data"
TAG_TO_VARIANT_NAME = {
    "O": "original",
    "C": "clone",
    "R": "recover",
    "D": "dissect_rebuilt",
    "X": "original",  # Same file as O, but signals empty/carving intent
}


def drop_lost_and_found_tables(source_db: Path, tables: list[str]) -> dict[str, Any]:
    """Drop lost_and_found tables from source database after extraction.

    Args:
        source_db: Path to source database
        tables: List of table names to drop

    Returns:
        Dictionary with drop metadata
    """
    if not tables:
        return {"dropped": False, "reason": "no_tables", "tables": []}

    try:
        with sqlite3.connect(source_db) as conn:
            cursor = conn.cursor()

            dropped_tables = []
            for table in tables:
                try:
                    quoted_table = quote_identifier(table)
                    cursor.execute(f"DROP TABLE IF EXISTS {quoted_table}")
                    dropped_tables.append(table)
                except sqlite3.Error as e:
                    logger.warning(f"Failed to drop table {table}: {e}")
                    continue

            conn.commit()
            # WAL cleanup before closing to prevent Windows file locks
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.execute("PRAGMA journal_mode=DELETE;")

        # Note: gc.collect() moved to batch cleanup in process_results()
        # Manual WAL file deletion only needed on Windows
        if is_windows():
            cleanup_wal_files(source_db)

        return {
            "dropped": True,
            "tables": dropped_tables,
            "count": len(dropped_tables),
        }

    except Exception as e:
        return {"dropped": False, "reason": f"error: {e}", "tables": []}


def extract_lost_and_found_tables(
    source_db: Path,
    output_dir: Path,
    tables: list[str],
    case_name: str,
) -> dict[str, Any]:
    """Extract lost_and_found tables from a database into separate DB.

    Args:
        source_db: Path to source database
        output_dir: Directory to save extracted database
        tables: List of table names to extract
        case_name: Name for the extracted database

    Returns:
        Dictionary with extraction metadata
    """
    if not tables:
        return {
            "extracted": False,
            "reason": "no_tables",
            "tables": [],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_db = output_dir / f"{case_name}_lost_and_found.sqlite"

    try:
        # Connect to both databases
        with sqlite3.connect(source_db) as source_conn, sqlite3.connect(output_db) as dest_conn:
            source_cur = source_conn.cursor()
            dest_cur = dest_conn.cursor()

            extracted_tables = []
            total_rows = 0

            for table in tables:
                try:
                    # Get table schema
                    source_cur.execute(
                        "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                        (table,),
                    )
                    schema = source_cur.fetchone()

                    if not schema or not schema[0]:
                        continue

                    # Create table in destination
                    dest_cur.execute(schema[0])

                    # Copy data
                    quoted_table = quote_identifier(table)
                    source_cur.execute(f"SELECT * FROM {quoted_table}")
                    rows = source_cur.fetchall()

                    if rows:
                        # Get column count
                        column_count = len(rows[0])
                        placeholders = ",".join(["?"] * column_count)
                        dest_cur.executemany(f"INSERT INTO {quoted_table} VALUES ({placeholders})", rows)

                        extracted_tables.append(table)
                        total_rows += len(rows)

                except sqlite3.Error as e:
                    logger.warning(f"Failed to extract table {table}: {e}")
                    continue

            dest_conn.commit()
            # WAL cleanup for destination database to prevent Windows file locks
            dest_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            dest_conn.execute("PRAGMA journal_mode=DELETE;")

        # Note: gc.collect() moved to batch cleanup in process_results()
        # Manual WAL file deletion only needed on Windows
        if is_windows():
            cleanup_wal_files(output_db)

        if extracted_tables:
            return {
                "extracted": True,
                "output_path": str(output_db),
                "tables": extracted_tables,
                "total_rows": total_rows,
                "size_bytes": output_db.stat().st_size,
            }
        # No tables were extracted, remove empty DB
        output_db.unlink()
        return {
            "extracted": False,
            "reason": "no_data",
            "tables": [],
        }

    except Exception as e:
        return {
            "extracted": False,
            "reason": f"error: {e}",
            "tables": [],
        }


def cleanup_variant_files(
    variant_outputs: dict[str, str],
    chosen_variant: str,
    aggressive: bool = False,
) -> dict[str, Any]:
    """Clean up unnecessary variant files.

    Args:
        variant_outputs: Dictionary mapping variant names to file paths
        chosen_variant: Single-letter tag of the chosen variant (O/C/R/D)
        aggressive: If True, also delete chosen variant for empty/unmatched

    Returns:
        Dictionary with cleanup metadata
    """
    deleted_files = []
    deleted_bytes = 0
    kept_files = []
    failed_deletions = []  # Track files that fail to delete

    # Convert single-letter tag to full variant name
    chosen_variant_name = TAG_TO_VARIANT_NAME.get(chosen_variant, chosen_variant)

    # Special handling for .gz files: if chosen is "original" and there's a "decompressed",
    # we want to keep the decompressed database and delete the .gz file
    if chosen_variant_name == "original" and "decompressed" in variant_outputs:
        chosen_variant_name = "decompressed"

    for variant_name, path_str in variant_outputs.items():
        path = Path(path_str)

        # Keep chosen variant unless aggressive mode
        if variant_name == chosen_variant_name and not aggressive:
            if path.exists():
                kept_files.append(
                    {
                        "variant": variant_name,
                        "path": str(path),
                        "size_bytes": path.stat().st_size,
                    }
                )
            continue

        # Delete this variant - try once without gc overhead
        if path.exists():
            size = path.stat().st_size
            try:
                path.unlink()
                deleted_files.append(
                    {
                        "variant": variant_name,
                        "path": str(path),
                        "size_bytes": size,
                    }
                )
                deleted_bytes += size
            except PermissionError:
                # Track for retry after gc.collect()
                failed_deletions.append((variant_name, path, size))
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")

    # Retry failed deletions after a single gc.collect() (Windows file handle release)
    if failed_deletions:
        if is_windows():
            gc.collect()
            time.sleep(0.1)

        for variant_name, path, size in failed_deletions:
            if not path.exists():
                continue  # Already deleted somehow

            try:
                path.unlink()
                deleted_files.append(
                    {
                        "variant": variant_name,
                        "path": str(path),
                        "size_bytes": size,
                    }
                )
                deleted_bytes += size
            except Exception as e:
                logger.warning(f"Failed to delete {path}: {e}")

    # Clean up orphaned files (temp files, .sql files not in variant_outputs)
    # These are files created during processing but not tracked in variant_outputs
    orphaned_deleted = []
    if variant_outputs:
        first_path = Path(next(iter(variant_outputs.values())))
        folder = first_path.parent

        # Get list of all tracked file paths (basenames)
        tracked_basenames = {Path(p).name for p in variant_outputs.values()}

        # Get the kept file basename (if any)
        kept_basenames = {Path(kf["path"]).name for kf in kept_files}

        for file in folder.glob("*"):
            if not file.is_file():
                continue

            # Skip tracked files, kept files, and lost_and_found files
            if file.name in tracked_basenames or file.name in kept_basenames:
                continue
            if "_lost_and_found.sqlite" in file.name:
                continue
            if file.name == ".DS_Store":
                continue

            # Delete orphaned temp files and .sql files
            if file.name.startswith("tmp") or file.suffix in [".sql"]:
                try:
                    size = file.stat().st_size
                    file.unlink()
                    orphaned_deleted.append(str(file))
                    deleted_bytes += size
                except Exception as e:
                    logger.warning(f"Failed to delete orphaned file {file.name}: {e}")

    result = {
        "deleted_files": deleted_files,
        "deleted_count": len(deleted_files),
        "deleted_bytes": deleted_bytes,
        "kept_files": kept_files,
        "kept_count": len(kept_files),
    }

    if orphaned_deleted:
        result["orphaned_deleted"] = orphaned_deleted
        result["orphaned_deleted_count"] = len(orphaned_deleted)

    return result


def process_case_record(
    record: dict[str, Any],
    output_root: Path,
    aggressive_cleanup: bool = False,
) -> dict[str, Any]:
    """Process a single case record from results.jsonl.

    Args:
        record: Case record from variant selector output
        output_root: Root directory for residue processing output
        aggressive_cleanup: If True, delete even chosen variants for unmatched

    Returns:
        Processing result metadata
    """
    if record.get("type") != "case":
        return {"skipped": True, "reason": "not_case_record"}

    # Handle skipped records (e.g., GeoServices databases that were filtered out)
    if record.get("skipped"):
        # For skipped databases, try to find and remove any empty folders
        # Skipped DBs may have folders created during scanning but no files kept
        # The folder should be in the same databases dir with pattern f{name}_{hash}
        # But we don't have the hash, so we can't reliably find it
        # This is best effort - the folder cleanup will happen for non-skipped DBs

        return {
            "skipped": True,
            "reason": record.get("skip_reason", "unknown"),
            "path": record.get("path", "unknown"),
        }

    case_path = Path(record["case_path"])
    case_name = case_path.stem
    decision = record["decision"]
    matched = decision.get("matched", False)
    empty = decision.get("empty", False)

    result = {
        "case_path": str(case_path),
        "case_name": case_name,
        "matched": matched,
        "empty": empty,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # Extract lost_and_found tables from ANY recovered variant that has them
    # (regardless of whether it was chosen, or whether the database is otherwise empty)
    meta_snapshot = record.get("meta_snapshot", {})
    has_lf = meta_snapshot.get("has_lost_and_found", False)
    lf_tables = meta_snapshot.get("lost_and_found_tables", [])

    if has_lf and lf_tables:
        variant_outputs = record.get("variant_outputs", {})

        # Try to find ANY recovered variant that exists
        # Preference order: recover > dissect_rebuilt > dissect
        recover_path = None
        for variant_key in ["recover", "dissect_rebuilt", "dissect"]:
            path_str = variant_outputs.get(variant_key)
            if path_str and Path(path_str).exists():
                recover_path = path_str
                break

        if recover_path:
            # Extract to the SAME folder as the database, not to residue/
            lf_dir = Path(recover_path).parent
            extraction = extract_lost_and_found_tables(
                source_db=Path(recover_path),
                output_dir=lf_dir,
                tables=lf_tables,
                case_name=case_name,
            )
            result["lost_and_found"] = extraction

            # Drop lost_and_found tables from source DB after successful extraction
            if extraction.get("extracted"):
                drop_result = drop_lost_and_found_tables(
                    source_db=Path(recover_path),
                    tables=lf_tables,
                )
                result["lost_and_found"]["dropped_from_source"] = drop_result
        else:
            result["lost_and_found"] = {
                "extracted": False,
                "reason": "no_recovered_variant_found",
                "tables": lf_tables,
            }
    else:
        result["lost_and_found"] = {
            "extracted": False,
            "reason": "no_lost_and_found",
            "tables": [],
        }

    # Clean up variant files based on outcome
    variant_outputs = record.get("variant_outputs", {})
    if variant_outputs:
        chosen_variant = record["variant_chosen"]

        # Aggressive cleanup for empty/unmatched databases
        # EXCEPT for variant X (preserve original file for byte carving)
        aggressive = aggressive_cleanup and (empty or not matched) and chosen_variant != "X"

        cleanup = cleanup_variant_files(
            variant_outputs=variant_outputs,
            chosen_variant=chosen_variant,
            aggressive=aggressive,
        )
        result["cleanup"] = cleanup

        # Remove folder if it's now empty (after cleanup)
        # Get the folder from any variant path
        if variant_outputs:
            first_path = Path(next(iter(variant_outputs.values())))
            folder = first_path.parent

            # Check if folder is empty (no files, only .DS_Store allowed)
            if folder.exists():
                remaining_files = [f for f in folder.iterdir() if f.is_file() and f.name != ".DS_Store"]
                if not remaining_files:
                    try:
                        # Remove any .DS_Store first
                        for f in folder.iterdir():
                            if f.name == ".DS_Store":
                                f.unlink()
                        # Remove the empty folder
                        folder.rmdir()
                        result["cleanup"]["folder_removed"] = str(folder)
                    except Exception as e:
                        logger.warning(f"Failed to remove empty folder {folder}: {e}")
    else:
        result["cleanup"] = {
            "deleted_files": [],
            "deleted_count": 0,
            "deleted_bytes": 0,
            "kept_files": [],
            "kept_count": 0,
        }

    return result


def process_results(
    results_path: Path,
    output_root: Path,
    aggressive_cleanup: bool = False,
) -> dict[str, Any]:
    """Process variant selector results and clean up residue.

    Args:
        results_path: Path to sqlite_scan_results.jsonl from variant selector
        output_root: Root directory for residue processing output
        aggressive_cleanup: If True, delete even chosen variants for unmatched

    Returns:
        Summary statistics
    """

    records = read_jsonl(results_path)
    case_records = [r for r in records if r.get("type") == "case"]

    results = []
    stats: dict[str, Any] = {
        "total_cases": len(case_records),
        "matched_cases": 0,
        "empty_cases": 0,
        "unmatched_cases": 0,
        "lost_and_found_extracted": 0,
        "total_deleted_bytes": 0,
        "total_deleted_files": 0,
        "total_kept_files": 0,
    }

    # Process records in batches, with periodic gc.collect() for Windows handle cleanup
    # On macOS/Linux, skip gc.collect() since file handles are released immediately
    batch_size = 100  # Increased from 50 for better performance
    for i, record in enumerate(case_records, 1):
        result = process_case_record(
            record=record,
            output_root=output_root,
            aggressive_cleanup=aggressive_cleanup,
        )
        results.append(result)

        # Update stats
        if result.get("matched"):
            stats["matched_cases"] += 1
        if result.get("empty"):
            stats["empty_cases"] += 1
        if not result.get("matched") and not result.get("empty"):
            stats["unmatched_cases"] += 1

        lf = result.get("lost_and_found", {})
        if lf.get("extracted"):
            stats["lost_and_found_extracted"] += 1

        # Periodic gc.collect() every batch_size records (Windows handle cleanup only)
        if is_windows() and i % batch_size == 0:
            gc.collect()

        cleanup = result.get("cleanup", {})
        if cleanup.get("deleted_count", 0) > 0:
            stats["total_deleted_files"] += cleanup["deleted_count"]
            stats["total_deleted_bytes"] += cleanup["deleted_bytes"]

        if cleanup.get("kept_count", 0) > 0:
            stats["total_kept_files"] += cleanup["kept_count"]

    # Final cleanup: Remove any empty folders left in databases directory
    # This catches skipped databases and any other orphaned folders
    databases_dir = results_path.parent  # Same directory as sqlite_scan_results.jsonl
    empty_folders_removed = 0

    for folder in databases_dir.iterdir():
        if folder.is_dir() and folder.name.startswith("f") and "_" in folder.name:
            # Check if folder is empty (no files except .DS_Store)
            remaining_files = [f for f in folder.iterdir() if f.is_file() and f.name != ".DS_Store"]
            if not remaining_files:
                try:
                    # Remove any .DS_Store first
                    for f in folder.iterdir():
                        if f.name == ".DS_Store":
                            f.unlink()
                    # Remove the empty folder
                    folder.rmdir()
                    empty_folders_removed += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {folder.name}: {e}")

    stats["empty_folders_removed"] = empty_folders_removed

    # Save detailed results
    results_output = output_root / "residue_processing.jsonl"
    write_jsonl(results_output, results)

    # Save summary
    stats["processing_date"] = datetime.now(UTC).isoformat()
    stats["results_file"] = str(results_output)
    stats["storage_saved_mb"] = stats["total_deleted_bytes"] / (1024 * 1024)

    summary_output = output_root / "residue_summary.json"
    with summary_output.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    return stats
