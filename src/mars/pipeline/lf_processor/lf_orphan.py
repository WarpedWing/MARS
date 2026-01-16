#!/usr/bin/env python3
"""
ORPHAN: Unmatched Tables Processor

Handles Lost & Found tables that don't match any exemplar (orphan/remnant tables).

Process:
1. Identify LF tables that weren't matched in MERGE/CATALOG/NEAREST
2. Create orphan databases for each source containing unmatched tables
3. Name output databases with match hints (if available) for context

Output: databases/found_data/{db_name}_orphans_{hash}/
"""

from __future__ import annotations

import gc
import sqlite3
from typing import TYPE_CHECKING, Any

from mars.pipeline.lf_processor.db_reconstructor import cleanup_wal_files
from mars.pipeline.lf_processor.lf_combiner import group_lf_tables_by_match
from mars.pipeline.lf_processor.lf_matcher import get_lf_tables
from mars.pipeline.lf_processor.uc_helpers import (
    create_shortened_name_hash,
    determine_match_label,
    sanitize_filename,
)
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from pathlib import Path

    from rich.progress import TaskID

    from mars.utils.progress_utils import ProgressContextType


def process_use_case_4_orphans(
    split_db: Path,
    remnant_lf_tables: list[str],
    output_found_data_dir: Path,
    source_db_name: str | None = None,
    exact_matches: list[dict] | None = None,
    metamatch: dict | None = None,
) -> dict[str, Any]:
    """
    Use Case 4: Preserve orphan LF tables that don't match anything.

    Copies all remnant LF tables into a single database in found_data/ for manual review.

    Args:
        split_db: Path to split database with LF tables
        remnant_lf_tables: List of LF table names that weren't matched
        output_found_data_dir: Base found_data output directory
        source_db_name: Original database name (for folder naming)
        exact_matches: List of exact match objects from database record (optional)
        metamatch: Metamatch object from database record (optional)

    Returns:
        {
            "orphan_database": Path or None,
            "tables": [
                {"lf_table": str, "row_count": int},
                ...
            ]
        }
    """
    if not remnant_lf_tables:
        return {
            "orphan_database": None,
            "tables": [],
        }

    # Create folder name using match hints when available
    match_label = determine_match_label(exact_matches or [], metamatch or {})

    # Create folder name - ALWAYS include filename for forensic tracking
    if match_label and source_db_name:
        # Use meaningful name WITH filename: Safari_History_f12345678_orphans
        folder_name = f"{sanitize_filename(match_label)}_{source_db_name}_orphans"
    elif source_db_name:
        # Fall back to hash-based name with source db
        folder_name = create_shortened_name_hash(f"{source_db_name}_orphans")
    else:
        # Fall back to generic hash name
        folder_name = create_shortened_name_hash("orphans")

    output_dir = output_found_data_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create single database for all orphan tables
    # Use match label for filename WITH filename for forensic tracking
    if match_label and source_db_name:
        db_filename = f"{sanitize_filename(match_label)}_{source_db_name}_orphans.sqlite"
    elif source_db_name:
        db_filename = f"{source_db_name}_orphans.sqlite"
    else:
        db_filename = "orphans.sqlite"

    output_db_path = output_dir / db_filename

    # Remove existing database if present
    if output_db_path.exists():
        output_db_path.unlink()

    with sqlite3.connect(split_db) as con_source, sqlite3.connect(output_db_path) as con_target:
        # Attach source database
        con_target.execute("ATTACH DATABASE ? AS source", (str(split_db),))

        orphan_tables = []

        for lf_table in remnant_lf_tables:
            # Get row count
            cursor = con_source.execute(f"SELECT COUNT(*) FROM {lf_table}")
            row_count = cursor.fetchone()[0]

            # Copy table structure and data
            con_target.execute(f"CREATE TABLE {lf_table} AS SELECT * FROM source.{lf_table}")

            orphan_tables.append(
                {
                    "lf_table": lf_table,
                    "row_count": row_count,
                }
            )

        con_target.execute("DETACH DATABASE source")
        con_target.commit()

        # Switch to DELETE journal mode to avoid WAL file locks on Windows
        con_target.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        con_target.execute("PRAGMA journal_mode=DELETE;")

    # Explicitly delete WAL files (Windows compatibility)
    cleanup_wal_files(output_db_path)

    return {
        "orphan_database": output_db_path,
        "tables": orphan_tables,
        "total_rows": sum(t["row_count"] for t in orphan_tables),
    }


def process_orphan_tables(
    prep_lookup: dict,
    matched_lf_by_db: dict[str, set],
    found_data_dir: Path,
    progress_context: ProgressContextType | None = None,
    phase_sub_task: TaskID | None = None,
) -> dict:
    """
    Process orphan LF tables that don't match any exemplar.

    Args:
        prep_lookup: Dict mapping db_name -> prepared database info
        matched_lf_by_db: Dict mapping db_name -> set of matched LF table names
        found_data_dir: Output directory for orphan databases
        progress_context: Optional LFProgressContext for hierarchical progress tracking
        phase_sub_task: Optional TaskID for the phase-level sub-task (managed by orchestrator)

    Returns:
        Dict with processing results:
            - total_orphan_databases: Number of orphan databases created
            - total_orphan_tables: Total number of orphan tables
            - orphan_results: Dict mapping db_name -> orphan processing result
    """
    orphan_results = {}
    total_orphan_databases = 0
    total_orphan_tables = 0

    # Count databases with potential orphans for progress tracking
    total_dbs = len(prep_lookup)
    current_idx = 0

    for db_name, db_info in prep_lookup.items():
        current_idx += 1
        if progress_context and phase_sub_task:
            progress_context.update_sub(
                phase_sub_task,
                current_idx,
                description=f"Orphan check {current_idx}/{total_dbs}: {db_name}",
            )
        split_db = db_info.get("split_db")
        if not split_db or not split_db.exists():
            continue

        # Get all LF tables in this database
        all_lf_tables = get_lf_tables(split_db)
        if not all_lf_tables:
            continue

        # Get matched LF tables (if any)
        matched_tables = matched_lf_by_db.get(db_name, set())

        # Calculate orphan tables (unmatched)
        orphan_tables = [t for t in all_lf_tables if t not in matched_tables]

        if not orphan_tables:
            continue

        # Extract match hints from record for better naming
        record = db_info.get("record", {})
        exact_matches = record.get("exact_matches", [])
        metamatch = record.get("metamatch", {})

        orphan_result = process_use_case_4_orphans(
            split_db=split_db,
            remnant_lf_tables=orphan_tables,
            output_found_data_dir=found_data_dir,
            source_db_name=db_name,
            exact_matches=exact_matches,
            metamatch=metamatch,
        )

        if orphan_result.get("orphan_database"):
            orphan_results[db_name] = orphan_result
            total_orphan_databases += 1
            total_orphan_tables += len(orphan_result["tables"])

        # Force garbage collection after each database to release SQLite connections
        gc.collect()

    logger.debug(
        f"Orphan: {total_orphan_databases} database(s) created, {total_orphan_tables} orphan table(s) preserved"
    )

    return {
        "total_orphan_databases": total_orphan_databases,
        "total_orphan_tables": total_orphan_tables,
        "orphan_results": orphan_results,
    }


def collect_matched_lf_tables(
    metamatch_results: dict,
    catalog_tracking: dict,
    nearest_tracking: dict,
) -> dict[str, set]:
    """
    Collect all matched LF tables from MERGE/CATALOG/NEAREST for orphan detection.

    Args:
        metamatch_results: Results from MERGE (metamatch) processing
        catalog_tracking: Results from CATALOG (exact match) processing
        nearest_tracking: Results from NEAREST (best-fit) processing

    Returns:
        Dict mapping db_name -> set of matched LF table names
    """
    matched_lf_by_db = {}  # {db_name: set(lf_table_names)}

    # Phase 3: Metamatch groups (MERGE)
    for group_id, result in metamatch_results.items():
        if result.get("success") and "lf_matches" in result:
            for db_name, match_info in result["lf_matches"].items():
                match_results = match_info.get("match_results", {})
                if match_results:
                    # Group by match with medium confidence threshold
                    grouped = group_lf_tables_by_match(match_results, confidence_threshold="medium")
                    # All LF tables in grouped are considered matched
                    matched_tables = set()
                    for (_, _), lf_tables in grouped.items():
                        matched_tables.update(lf_tables)

                    if db_name not in matched_lf_by_db:
                        matched_lf_by_db[db_name] = set()
                    matched_lf_by_db[db_name].update(matched_tables)

    # Phase 4: Catalog groups (CATALOG)
    for exemplar_name, db_entries in catalog_tracking.items():
        for entry in db_entries:
            db_name = entry["db_name"]
            match_results = entry.get("match_results")
            if match_results:
                grouped = group_lf_tables_by_match(match_results, confidence_threshold="medium")
                matched_tables = set()
                for (_, _), lf_tables in grouped.items():
                    matched_tables.update(lf_tables)

                if db_name not in matched_lf_by_db:
                    matched_lf_by_db[db_name] = set()
                matched_lf_by_db[db_name].update(matched_tables)

    # Phase 5: Individual/nearest exemplar (NEAREST)
    for exemplar_name, db_entries in nearest_tracking.items():
        for entry in db_entries:
            db_name = entry["db_name"]
            match_results = entry.get("match_results")
            if match_results:
                grouped = group_lf_tables_by_match(match_results, confidence_threshold="medium")
                matched_tables = set()
                for (_, _), lf_tables in grouped.items():
                    matched_tables.update(lf_tables)

                if db_name not in matched_lf_by_db:
                    matched_lf_by_db[db_name] = set()
                matched_lf_by_db[db_name].update(matched_tables)

    return matched_lf_by_db
