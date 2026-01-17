#!/usr/bin/env python3
"""
MERGE: Metamatch Group Processor

Handles Lost & Found reconstruction for metamatch groups (databases with similar schemas).

Process:
1. Combine all databases in the metamatch group
2. Generate superrubric from combined database
3. Match LF tables from each source database against superrubric
4. Reconstruct output database with combined intact + LF data
5. Deduplicate tables
6. Generate manifest file

Note: Remnant/orphan handling is centralized in lf_orphan.py after all phases complete.

Output: databases/metamatches/{group_label}/{group_label}.sqlite
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING

from mars.pipeline.lf_processor.db_reconstructor import (
    copy_table_data_with_provenance,
    create_manifest_file,
    deduplicate_table,
    get_table_schema,
    insert_lf_data_into_table,
    quote_identifier,
    reconstruct_database_with_schema,
)
from mars.pipeline.lf_processor.lf_combiner import (
    group_lf_tables_by_match,
    prepare_combined_tables,
)
from mars.pipeline.lf_processor.lf_matcher import match_lf_tables_to_exemplars
from mars.pipeline.lf_processor.uc_helpers import is_fts_table
from mars.pipeline.matcher.generate_sqlite_schema_rubric import (
    fetch_tables,
    generate_rubric,
)
from mars.pipeline.output.database_combiner import merge_sqlite_databases
from mars.utils.database_utils import readonly_connection
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from mars.utils.progress_utils import ProgressContextType


def _count_database_rows(db_path: Path) -> int:
    """
    Count total rows across all non-system tables in a database.

    Args:
        db_path: Path to SQLite database

    Returns:
        Total row count across all tables, or 0 on error
    """
    try:
        with readonly_connection(db_path) as con:
            cursor = con.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]
            total_rows = 0
            for table in tables:
                try:
                    cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                    total_rows += cursor.fetchone()[0]
                except Exception:
                    pass
            return total_rows
    except Exception:
        return 0


def _copy_fts_table_data(source_db: Path, target_db: Path, table_name: str) -> int:
    """
    Copy data from an FTS virtual table to output database.

    FTS tables don't support adding columns like data_source, so we copy
    the data directly using INSERT.

    Args:
        source_db: Path to source database (combined_db)
        target_db: Path to target database (output_db)
        table_name: Name of the FTS table

    Returns:
        Number of rows copied
    """
    import sqlite3

    rows_copied = 0
    q_table = f'"{table_name}"'

    try:
        with sqlite3.connect(f"file:{source_db}?mode=ro", uri=True) as src_conn:
            src_conn.text_factory = lambda x: x.decode("utf-8", "replace") if isinstance(x, bytes) else x
            src_cur = src_conn.cursor()

            # Get column names from FTS table
            cols_info = src_cur.execute(f"PRAGMA table_info({q_table})").fetchall()
            if not cols_info:
                logger.debug(f"      FTS table {table_name}: no column info from PRAGMA table_info")
                return 0

            col_names = [c[1] for c in cols_info]
            col_list = ", ".join(f'"{c}"' for c in col_names)
            placeholders = ", ".join("?" for _ in col_names)

            # Fetch all rows from source
            rows = src_cur.execute(f"SELECT {col_list} FROM {q_table}").fetchall()

            if not rows:
                logger.debug(f"      FTS table {table_name}: no rows to copy from source")
                return 0

            with sqlite3.connect(target_db) as dst_conn:
                dst_cur = dst_conn.cursor()
                for row in rows:
                    try:
                        dst_cur.execute(
                            f"INSERT OR IGNORE INTO {q_table} ({col_list}) VALUES ({placeholders})",
                            row,
                        )
                        rows_copied += 1
                    except sqlite3.Error:
                        pass  # Skip problematic rows
                dst_conn.commit()

    except Exception as e:
        logger.debug(f"      Could not copy FTS data for {table_name}: {e}")

    return rows_copied


def process_metamatch_group(
    group_id: str,
    group_info: dict,
    prep_lookup: dict,
    databases_dir: Path,
    lf_temp_dir: Path,
    exemplar_db_dir: Path | None = None,
    progress_context: ProgressContextType | None = None,  # Hierarchical progress context
    min_timestamp_rows: int = 1,
    min_role_sample_size: int = 5,
    min_year: int = 2000,
    max_year: int = 2038,
) -> dict:
    """
    Process a metamatch group: combine DBs, generate superrubric, match LF.

    Args:
        group_id: Metamatch group ID (schema hash)
        group_info: Dict with group_label, first_table, databases list
        prep_lookup: Dict mapping db_name -> prepared database info
        databases_dir: Output base directory
        lf_temp_dir: Temp directory for intermediate databases
        exemplar_db_dir: Optional exemplar directory for rubric metadata
        progress_context: Optional LFProgressContext for hierarchical progress tracking.
            Parent caller (orchestrator) manages the group-level progress bar.
        min_timestamp_rows: Minimum timestamp values to assign role (default: 1)
        min_role_sample_size: Minimum samples for UUID/programming_case detection (default: 5)
        min_year: Minimum year for timestamp validation (default: 2000)
        max_year: Maximum year for timestamp validation (default: 2038)

    Returns:
        Dict with processing results and statistics
    """
    group_label = group_info["group_label"]
    databases = group_info["databases"]

    # Progress tracking is handled by parent caller (lf_orchestrator)
    # Internal steps are logged but not reported via callback

    source_dbs = []
    db_names = []
    for record in databases:
        db_name = Path(record["case_path"]).stem
        db_info = prep_lookup.get(db_name)
        if db_info and db_info["source_db"]:
            source_dbs.append(db_info["source_db"])
            db_names.append(db_name)

    if not source_dbs:
        logger.warning("    No openable databases in group, skipping")
        return {"success": False, "error": "No openable databases in group"}

    # Count intact rows per source BEFORE merging
    # Uses optimized single-query counting (O(2) queries instead of O(n_tables + 1))
    intact_rows_per_source = {db_name: _count_database_rows(db_path) for db_path, db_name in zip(source_dbs, db_names)}

    # Step 2: Combine all databases in the group

    try:
        combined_db = lf_temp_dir / f"{group_label}_combined.sqlite"

        merge_sqlite_databases(
            source_dbs=source_dbs,
            output_db=combined_db,
            combine_strategy="insert_or_ignore",
        )

    except Exception as e:
        logger.error(f"    Failed to combine databases: {e}")
        return {"success": False, "error": f"Failed to combine databases: {e}"}

    # Step 3: Generate superrubric from combined database

    try:
        with readonly_connection(combined_db) as con:
            all_tables = fetch_tables(con)
            superrubric = generate_rubric(
                con,
                all_tables,
                rubric_name=group_label,
                min_timestamp_rows=min_timestamp_rows,
                min_role_sample_size=min_role_sample_size,
                min_year=min_year,
                max_year=max_year,
            )

        # Save superrubric to databases/schemas/{group_label}/
        schemas_dir = databases_dir / "schemas" / group_label
        schemas_dir.mkdir(parents=True, exist_ok=True)
        superrubric_path = schemas_dir / f"{group_label}.superrubric.json"

        with superrubric_path.open("w") as f:
            json.dump(superrubric, f, indent=2)

    except Exception as e:
        logger.error(f"    Failed to generate superrubric: {e}")
        return {"success": False, "error": f"Failed to generate superrubric: {e}"}

    # Prepare rubric for matching
    exemplar_rubrics = [{"name": group_label, "rubric": superrubric}]

    # Step 4: Match LF tables from all group members

    all_matches = {}
    for record in databases:
        db_name = Path(record["case_path"]).stem
        db_info = prep_lookup.get(db_name)
        if not db_info or not db_info["split_db"]:
            continue

        split_db = db_info["split_db"]

        # Match this database's LF tables against superrubric
        # Note: per_db_ignorable_tables is None for metamatch groups since superrubrics
        # are dynamically generated and don't have catalog entries. Global ignorable
        # tables (Z_PRIMARYKEY, etc.) are still filtered in lf_matcher.py.
        match_results = match_lf_tables_to_exemplars(
            split_db,
            exemplar_rubrics,
            exact_match_name=None,
            nearest_exemplar_names=None,
            per_db_ignorable_tables=None,
            min_timestamp_rows=min_timestamp_rows,
            min_role_sample_size=min_role_sample_size,
            min_year=min_year,
            max_year=max_year,
        )

        # Store matches keyed by source database
        all_matches[db_name] = {
            "split_db": split_db,
            "match_results": match_results,
            "record": record,  # Store record for access to exact_matches/metamatch
        }

    # Step 5: Create output directory and database schema

    metamatches_dir = databases_dir / "metamatches" / group_label
    metamatches_dir.mkdir(parents=True, exist_ok=True)
    output_db = metamatches_dir / f"{group_label}.sqlite"

    # Remove existing output if present
    if output_db.exists():
        output_db.unlink()

    # Get schemas from combined database
    # Also identify FTS tables directly from sqlite_master (more reliable than parsing CREATE SQL)
    with readonly_connection(combined_db) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        all_table_names = [row[0] for row in cursor.fetchall()]

        # Directly identify FTS virtual tables by checking CREATE SQL in sqlite_master
        cursor = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%USING fts%' AND name NOT LIKE 'sqlite_%'"
        )
        fts_table_names = {row[0] for row in cursor.fetchall()}

    # Build set of FTS shadow table names (auto-created by SQLite for FTS tables)
    # Shadow tables follow pattern: {fts_table}_suffix where suffix is one of:
    # _data, _idx, _content, _config, _docsize (FTS5 uses these)
    fts_shadow_suffixes = ("_data", "_idx", "_content", "_config", "_docsize")
    fts_shadow_tables = set()
    for fts_name in fts_table_names:
        for suffix in fts_shadow_suffixes:
            shadow_name = f"{fts_name}{suffix}"
            if shadow_name in all_table_names:
                fts_shadow_tables.add(shadow_name)

    # Get schemas for all tables
    table_schemas = {}
    for table_name in all_table_names:
        schema = get_table_schema(combined_db, table_name)
        table_schemas[table_name] = schema

    # Check if this is an FTS-only database (all tables are FTS or shadow tables)
    # FTS tables can't have data_source column added, so we handle them specially
    # Filter out both FTS virtual tables AND their shadow tables
    non_fts_tables = [t for t in all_table_names if t not in fts_table_names and t not in fts_shadow_tables]

    # Log what we found for debugging
    if fts_table_names:
        logger.debug(f"    Found {len(fts_table_names)} FTS tables: {sorted(fts_table_names)}")
    if fts_shadow_tables:
        logger.debug(f"    Found {len(fts_shadow_tables)} FTS shadow tables (excluded)")
    if non_fts_tables:
        logger.debug(f"    Found {len(non_fts_tables)} non-FTS tables: {sorted(non_fts_tables)}")

    if not non_fts_tables:
        # FTS-only database: copy combined_db directly instead of reconstruction
        # FTS tables don't support adding data_source column, so just preserve as-is
        logger.debug(f"    FTS-only database detected, copying directly: {group_label}")
        import shutil

        shutil.copy2(combined_db, output_db)

        # Count rows in the copied database for statistics
        total_fts_rows = _count_database_rows(output_db)

        # Create manifest for FTS-only database
        source_db_list = []
        for db_name in db_names:
            source_info = {
                "db_name": db_name,
                "intact_rows": intact_rows_per_source.get(db_name, 0),
                "lf_rows": 0,  # No LF processing for FTS-only
            }
            source_db_list.append(source_info)

        combined_stats = {
            "total_intact_rows": 0,
            "total_lf_rows": 0,
            "total_fts_rows": total_fts_rows,
            "duplicates_removed": 0,
            "table_stats": [{"name": t, "rows": "FTS"} for t in all_table_names],
            "note": "FTS-only database copied directly (no reconstruction)",
        }

        manifest_path = metamatches_dir / f"{group_label}_manifest.json"
        manifest_created = create_manifest_file(
            output_path=manifest_path,
            output_type="metamatch",
            output_name=group_label,
            source_databases=source_db_list,
            combined_stats=combined_stats,
        )

        return {
            "success": True,
            "group_label": group_label,
            "combined_db": combined_db,
            "output_db": output_db,
            "superrubric": superrubric,
            "superrubric_path": superrubric_path,
            "source_dbs": db_names,
            "db_count": len(source_dbs),
            "lf_matches": {},
            "lf_rows_added": 0,
            "fts_rows_copied": total_fts_rows,
            "duplicates_removed": 0,
            "tables_with_lf": [],
            "manifest_created": manifest_created,
            "fts_only": True,
        }

    # Create output database with data_source column
    reconstruct_database_with_schema(
        output_db,
        table_schemas,
        add_data_source_column=True,
    )

    # Step 6: Copy all data from combined database with provenance

    total_intact_rows = 0

    # Load rubric once for validation
    rubric_metadata = None
    if exemplar_db_dir:
        exemplar_schemas_dir = exemplar_db_dir / "schemas" / group_label
        for suffix in ["", "_combined"]:
            rubric_path = exemplar_schemas_dir / f"{group_label}{suffix}.rubric.json"
            if rubric_path.exists():
                try:
                    with rubric_path.open() as f:
                        rubric_metadata = json.load(f)
                    break
                except Exception as e:
                    logger.warning(f"      ⚠ Failed to load rubric {rubric_path}: {e}")

    # Track FTS rows separately (they don't have data_source column)
    total_fts_rows = 0

    for table_name in all_table_names:
        # Skip FTS shadow tables - they're auto-managed by SQLite
        if table_name in fts_shadow_tables:
            continue

        # Handle FTS virtual tables separately - copy data directly without data_source column
        if table_name in fts_table_names:
            try:
                fts_rows = _copy_fts_table_data(combined_db, output_db, table_name)
                total_fts_rows += fts_rows
                if fts_rows > 0:
                    logger.debug(f"      Copied {fts_rows} rows to FTS table: {table_name}")
            except Exception as e:
                logger.warning(f"      Error copying FTS table {table_name}: {e}")
            continue

        try:
            rows = copy_table_data_with_provenance(
                target_db=output_db,
                target_table=table_name,
                source_db=combined_db,
                source_table=table_name,
                data_source_value="carved",
                exemplar_schemas_dir=(exemplar_db_dir / "schemas" / group_label if exemplar_db_dir else None),
                rubric_metadata=rubric_metadata,  # Pass pre-loaded rubric
            )
            total_intact_rows += rows
        except Exception as e:
            logger.warning(f"      Error copying {table_name}: {e}")

    # Force garbage collection after copying all tables to release SQLite connections
    gc.collect()

    # Step 7: Process matched LF data from each source database

    total_lf_rows = 0
    tables_with_lf = set()
    lf_rows_per_source = {}  # Track LF rows per source for manifest

    for db_name, match_info in all_matches.items():
        match_results = match_info["match_results"]
        split_db = match_info["split_db"]

        # Group matches by table at MEDIUM confidence
        grouped = group_lf_tables_by_match(match_results, confidence_threshold="medium")

        # Prepare combined tables info
        rubric_dict = {group_label: superrubric}
        prepared = prepare_combined_tables(grouped, match_results, rubric_dict)

        # Track LF rows for this source
        source_lf_rows = 0

        # Insert LF data into each matched table
        for (exemplar_name, table_name), table_info in prepared.items():
            # Skip FTS auxiliary tables - they don't have data_source column
            if is_fts_table(table_name, table_schemas):
                continue

            try:
                rows_inserted = insert_lf_data_into_table(
                    target_db=output_db,
                    target_table=table_name,
                    source_lf_db=split_db,
                    source_lf_tables=table_info["lf_tables"],
                    column_mapping=table_info["column_mapping"],
                    data_source_value=f"found_{db_name}",
                    rubric_metadata=superrubric,  # Pass superrubric for validation
                )

                if rows_inserted > 0:
                    total_lf_rows += rows_inserted
                    source_lf_rows += rows_inserted
                    tables_with_lf.add(table_name)

            except Exception as e:
                logger.warning(f"      Error inserting LF data into {table_name}: {e}")

        lf_rows_per_source[db_name] = source_lf_rows

        # Force garbage collection after each source database to release connections
        gc.collect()

    # Step 8: Deduplicate all tables that received LF data

    total_dupes_removed = 0
    for table_name in tables_with_lf:
        # Skip FTS auxiliary tables (shouldn't be in this set, but be safe)
        if is_fts_table(table_name, table_schemas):
            continue

        try:
            dupes = deduplicate_table(output_db, table_name)
            total_dupes_removed += dupes
        except Exception as e:
            logger.warning(f"      Error deduplicating {table_name}: {e}")

    # Force garbage collection after deduplication to release SQLite connections
    gc.collect()

    # Check if we have any data - skip empty groups
    # Include FTS rows in the check since FTS-only databases are valid
    if total_intact_rows == 0 and total_lf_rows == 0 and total_fts_rows == 0:
        # Clean up empty output (Windows-compatible)
        from mars.utils.cleanup_utilities import cleanup_sqlite_directory

        if metamatches_dir.exists():
            cleanup_sqlite_directory(metamatches_dir)
        logger.debug(f"    Skipping empty metamatch group: {group_label}")
        return {
            "success": False,
            "reason": "empty_group",
            "error": "Empty group (no data)",
        }

    # NOTE: Remnant/orphan databases are created centrally in lf_orphan.py
    # after all phases complete. This ensures each LF table appears in exactly
    # one location (either matched to an exemplar or in found_data/ as orphan).

    # Step 9: Create manifest file
    source_db_list = []
    for db_name in db_names:
        source_info = {
            "db_name": db_name,
            "intact_rows": intact_rows_per_source.get(db_name, 0),
            "lf_rows": lf_rows_per_source.get(db_name, 0),
        }
        source_db_list.append(source_info)

    # Get table-level statistics from output database
    with readonly_connection(output_db) as con:
        table_stats = []
        for table_name in all_table_names:
            if is_fts_table(table_name, table_schemas):
                continue
            try:
                cursor = con.execute(f"SELECT COUNT(*) FROM {quote_identifier(table_name)}")
                row_count = cursor.fetchone()[0]
                table_stats.append({"name": table_name, "rows": row_count})
            except Exception:
                pass

    combined_stats = {
        "total_intact_rows": total_intact_rows,
        "total_lf_rows": total_lf_rows,
        "total_fts_rows": total_fts_rows,
        "duplicates_removed": total_dupes_removed,
        "table_stats": table_stats,
    }

    manifest_path = metamatches_dir / f"{group_label}_manifest.json"
    manifest_created = create_manifest_file(
        output_path=manifest_path,
        output_type="metamatch",
        output_name=group_label,
        source_databases=source_db_list,
        combined_stats=combined_stats,
    )

    if not manifest_created:
        logger.warning("    ⚠ Failed to create manifest")

    return {
        "success": True,
        "group_label": group_label,
        "combined_db": combined_db,
        "output_db": output_db,
        "superrubric": superrubric,
        "superrubric_path": superrubric_path,
        "source_dbs": db_names,
        "db_count": len(source_dbs),
        "lf_matches": all_matches,
        "lf_rows_added": total_lf_rows,
        "fts_rows_copied": total_fts_rows,
        "duplicates_removed": total_dupes_removed,
        "tables_with_lf": list(tables_with_lf),
        "manifest_created": manifest_created,
    }
