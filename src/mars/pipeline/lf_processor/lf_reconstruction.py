#!/usr/bin/env python3
"""
Shared Database Reconstruction Logic for CATALOG/NEAREST

Common functions for reconstructing databases from intact + LF data.
Used by both CATALOG (exact match) and NEAREST (best-fit) processors.
"""

from __future__ import annotations

import gc
import json
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from mars.pipeline.lf_processor.db_reconstructor import (
    cleanup_wal_files,
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
from mars.pipeline.lf_processor.uc_helpers import is_fts_table
from mars.utils.database_utils import readonly_connection
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from mars.utils.progress_utils import ProgressContextType


def reconstruct_exemplar_database(
    exemplar_name: str,
    db_entries: list[dict],
    exemplar_rubric: dict,
    output_dir: Path,
    output_type: str,  # "candidate" or "found_data"
    exemplar_db_dir: Path | None,
    progress_context: ProgressContextType | None = None,  # Hierarchical progress context
    skip_shared_tables: bool = True,
    contributing_exemplar_names: list[str] | None = None,
    base_name: str | None = None,
    username: str | None = None,
    profiles: list[str] | None = None,
    consumed_lf_tables: dict[str, set[str]] | None = None,
) -> dict:
    """
    Reconstruct a database from multiple sources using an exemplar rubric.

    This handles both CATALOG (exact match) and NEAREST (best-fit) reconstruction.

    Args:
        exemplar_name: Name of the exemplar (used for output filename)
        db_entries: List of database entries with split_db, source_db, match_results
        exemplar_rubric: Exemplar rubric dict with table schemas
        output_dir: Output directory for reconstructed database
        output_type: "candidate" for CATALOG, "found_data" for NEAREST
        exemplar_db_dir: Optional exemplar directory for rubric metadata
        progress_context: Optional LFProgressContext for hierarchical progress tracking.
            Parent caller handles sub-task management; this function does not update progress.
        skip_shared_tables: If True, skip tables marked with "shared_with" in rubric
        contributing_exemplar_names: For _multi catalogs, list of user exemplar names
            whose LF match results should be accepted (e.g., ["Mail Envelope Index_user",
            "Mail Envelope Index_username"] for Mail Envelope Index_multi)
            (they will be processed by the _multi catalog instead)
        base_name: Catalog base name (e.g., "Chrome Cookies" for "Chrome Cookies_user").
            Used by comparison calculator to group variants by base name.
        username: Username for export path resolution (e.g., "username" for user-scoped
            databases, "_multi" for shared tables).
        profiles: List of Chrome/browser profiles this database matches (e.g., ["Default"],
            ["Profile 1"], ["Guest Profile", "System Profile"]). Used for export path resolution.
        consumed_lf_tables: Optional dict tracking which LF tables have been consumed.
            Keys are db_name, values are sets of lf_table names. If provided, this function
            will skip LF tables that have already been consumed by a prior phase/exemplar
            and will mark newly consumed tables. Modified in place.

    Returns:
        Dict with reconstruction statistics
    """
    # Get table schemas from rubric
    tables = exemplar_rubric.get("tables", {})
    table_names = [t.get("name") for t in tables] if isinstance(tables, list) else list(tables.keys())

    # Build schemas from exemplar database (canonical schema)
    table_schemas = {}
    source_db_for_schema = None

    # Try to get schema from exemplar database path
    if db_entries:
        first_record = db_entries[0].get("record", {})

        # Try exact_matches first (CATALOG)
        exact_matches = first_record.get("exact_matches", [])
        if exact_matches:
            # Find the actual exact match (tables_equal+columns_equal or hash)
            # Not just the first entry which may be a partial (tables-only) match
            exact_match_entry = None
            for match in exact_matches:
                match_type = match.get("match", "")
                if match_type in ("tables_equal+columns_equal", "hash"):
                    exact_match_entry = match
                    break

            # Fall back to first entry if no full exact match found
            if not exact_match_entry:
                exact_match_entry = exact_matches[0]

            exemplar_path = exact_match_entry.get("sample_path")
            if exemplar_path:
                exemplar_db = Path(exemplar_path)
                if exemplar_db.exists():
                    try:
                        with readonly_connection(exemplar_db) as con:
                            con.execute("SELECT name FROM sqlite_master LIMIT 1")
                        source_db_for_schema = exemplar_db
                    except Exception:
                        pass

        # Try nearest_exemplars (NEAREST) if exact_matches didn't work
        if not source_db_for_schema and "exemplar_info" in db_entries[0]:
            exemplar_info = db_entries[0].get("exemplar_info", {})
            exemplar_path = exemplar_info.get("exemplar")
            if exemplar_path:
                exemplar_db = Path(exemplar_path)
                if exemplar_db.exists():
                    try:
                        with readonly_connection(exemplar_db) as con:
                            con.execute("SELECT name FROM sqlite_master LIMIT 1")
                        source_db_for_schema = exemplar_db
                    except Exception:
                        pass

    # Fallback: Try source databases only if exemplar unavailable
    if not source_db_for_schema:
        for entry in db_entries:
            source_db = entry["source_db"]
            if not source_db or not source_db.exists():
                continue

            try:
                with readonly_connection(source_db) as con:
                    con.execute("SELECT name FROM sqlite_master LIMIT 1")
                source_db_for_schema = source_db
                logger.warning("      Using source database for schema (exemplar unavailable)")
                break
            except Exception:
                continue

    if not source_db_for_schema:
        logger.warning("      ⚠ No openable source database, skipping")
        return {"success": False, "error": "No openable source database"}

    # Get schemas from source database
    for table_name in table_names:
        try:
            schema = get_table_schema(source_db_for_schema, table_name)
            table_schemas[table_name] = schema
        except Exception:
            continue

    if not table_schemas:
        logger.warning("      ⚠ No table schemas found, skipping")
        return {"success": False, "error": "No table schemas found"}

    # Track progress within this function (internal step counting)
    sources_with_lf = [e for e in db_entries if e.get("split_db") and e.get("match_results")]
    current_step = 0

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    output_db = output_dir / f"{exemplar_name}.sqlite"

    # Create output database with data_source column
    try:
        reconstruct_database_with_schema(
            output_db,
            table_schemas,
            add_data_source_column=True,
        )
    except Exception as e:
        logger.error(f"      ⚠ Error creating output database: {e}")
        import traceback

        logger.debug(f"      Traceback: {traceback.format_exc()}")
        return {"success": False, "error": f"Error creating output database: {e}"}

    # Copy data from all source databases
    total_intact_rows = 0
    intact_rows_per_source = {}

    # Load rubric once for validation
    rubric_metadata = None
    if exemplar_db_dir:
        exemplar_schemas_dir = exemplar_db_dir / "schemas" / exemplar_name
        for suffix in ["", "_combined"]:
            rubric_path = exemplar_schemas_dir / f"{exemplar_name}{suffix}.rubric.json"
            if rubric_path.exists():
                try:
                    with rubric_path.open() as f:
                        rubric_metadata = json.load(f)
                    break
                except Exception as e:
                    logger.warning(f"      ⚠ Failed to load rubric {rubric_path}: {e}")

    # Detect shared tables (marked with "shared_with" by multi_user_splitter)
    # These tables should be routed to the _multi catalog instead
    shared_tables: set[str] = set()
    if skip_shared_tables and rubric_metadata:
        rubric_tables = rubric_metadata.get("tables", {})
        if isinstance(rubric_tables, dict):
            for table_name, table_info in rubric_tables.items():
                if isinstance(table_info, dict) and table_info.get("shared_with"):
                    shared_tables.add(table_name)

        if shared_tables:
            logger.debug(
                f"      Skipping {len(shared_tables)} shared table(s) "
                f"(routed to _multi): {', '.join(sorted(shared_tables)[:3])}..."
            )

    total_sources = len(db_entries)

    # Create sub-task for large reconstructions (multiple sources or many tables)
    # This provides feedback during long operations like Powerlog processing
    reconstruction_sub = None
    sources_with_lf_count = len([e for e in db_entries if e.get("split_db") and e.get("match_results")])
    total_operations = total_sources + sources_with_lf_count + 1  # +1 for dedup/vacuum
    if progress_context and (total_sources > 1 or len(table_schemas) > 5):
        reconstruction_sub = progress_context.create_sub_task(
            f"Processing data (0/{total_operations - 1} steps)",
            total=total_operations,
        )

    # Insert matched LF data FIRST (so found_* rows survive deduplication)
    # When same data exists in both carved and L&F sources, we prefer L&F provenance
    # because it proves the L&F recovery process successfully matched the fragment
    total_lf_rows = 0
    tables_with_lf: set[str] = set()
    lf_rows_per_source: dict[str, int] = {}

    # Count sources with LF data for progress
    sources_with_lf = [e for e in db_entries if e.get("split_db") and e.get("match_results")]
    lf_source_idx = 0

    for entry in db_entries:
        match_results = entry.get("match_results")
        split_db = entry.get("split_db")
        db_name = entry["db_name"]
        lf_rows_per_source[db_name] = 0

        # Skip LF processing if database has no LF tables
        if not split_db or not match_results:
            continue

        # Filter out already-consumed LF tables to prevent duplicates across exemplars
        # This ensures each LF table only contributes data to ONE output database
        if consumed_lf_tables is not None:
            already_consumed = consumed_lf_tables.get(db_name, set())
            if already_consumed:
                match_results = {
                    lf_table: results for lf_table, results in match_results.items() if lf_table not in already_consumed
                }
                if not match_results:
                    continue  # All LF tables from this source already consumed

        lf_source_idx += 1
        # Log progress for LF insertion
        if len(sources_with_lf) > 1 or len(table_schemas) > 10:
            logger.debug(f"      [{lf_source_idx}/{len(sources_with_lf)}] Inserting LF data from {db_name}")

        # Update sub-progress for LF insertion
        if reconstruction_sub and progress_context:
            progress_context.update_sub(
                reconstruction_sub,
                lf_source_idx,
                description=f"Processing LF data from {db_name}",
            )

        current_step += 1

        # Group matches by table
        grouped = group_lf_tables_by_match(match_results, confidence_threshold="medium")
        # Build rubric_dict with all valid exemplar names
        # For _multi catalogs, include user exemplar names that contribute LF data
        rubric_dict = {exemplar_name: exemplar_rubric}
        if contributing_exemplar_names:
            for contrib_name in contributing_exemplar_names:
                rubric_dict[contrib_name] = exemplar_rubric
        prepared = prepare_combined_tables(grouped, match_results, rubric_dict)

        # Insert LF data and track consumed tables
        for (_, table_name), table_info in prepared.items():
            if is_fts_table(table_name, table_schemas):
                continue

            # Skip shared tables - they're routed to _multi catalog
            if table_name in shared_tables:
                continue

            try:
                rows_inserted = insert_lf_data_into_table(
                    target_db=output_db,
                    target_table=table_name,
                    source_lf_db=split_db,
                    source_lf_tables=table_info["lf_tables"],
                    column_mapping=table_info["column_mapping"],
                    data_source_value=f"found_{db_name}",
                    rubric_metadata=exemplar_rubric,
                )
                if rows_inserted > 0:
                    total_lf_rows += rows_inserted
                    lf_rows_per_source[db_name] += rows_inserted
                    tables_with_lf.add(table_name)

                    # Mark LF tables as consumed (even if 0 rows - they were processed)
                    if consumed_lf_tables is not None:
                        if db_name not in consumed_lf_tables:
                            consumed_lf_tables[db_name] = set()
                        consumed_lf_tables[db_name].update(table_info["lf_tables"])
            except Exception as e:
                logger.debug(f"[DEBUG] insert_lf_data_into_table failed for {table_name}: {e}")

        # Periodic gc.collect() after each source database (not per-table)
        # Prevents file handle exhaustion without per-table overhead
        gc.collect()

    # Copy intact data from all source databases SECOND
    # This ensures that when deduplication runs, L&F rows (inserted first) have lower
    # ROWIDs and survive, preserving their found_* provenance
    for entry_idx, entry in enumerate(db_entries, 1):
        db_name = entry["db_name"]
        source_db = entry["source_db"]
        intact_rows_per_source[db_name] = 0

        if not source_db or not source_db.exists():
            continue

        # Test if database can actually be opened
        try:
            with readonly_connection(source_db) as con:
                con.execute("SELECT name FROM sqlite_master LIMIT 1")
        except Exception:
            continue

        # Log progress for large reconstructions (e.g., Powerlog)
        if total_sources > 1 or len(table_schemas) > 10:
            logger.debug(
                f"      [{entry_idx}/{total_sources}] Copying intact data from {db_name} ({len(table_schemas)} tables)"
            )

        # Update sub-progress for large reconstructions
        if reconstruction_sub and progress_context:
            progress_context.update_sub(
                reconstruction_sub,
                sources_with_lf_count + entry_idx,
                description=f"Copying from {db_name} ({len(table_schemas)} tables)",
            )

        current_step += 1

        for table_name in table_schemas:
            if is_fts_table(table_name, table_schemas):
                continue

            # Skip shared tables - they're routed to _multi catalog
            if table_name in shared_tables:
                continue

            try:
                rows = copy_table_data_with_provenance(
                    target_db=output_db,
                    target_table=table_name,
                    source_db=source_db,
                    source_table=table_name,
                    data_source_value=(f"candidate_{db_name}" if output_type == "candidate" else f"found_{db_name}"),
                    exemplar_schemas_dir=(exemplar_db_dir / "schemas" / exemplar_name if exemplar_db_dir else None),
                    rubric_metadata=rubric_metadata,
                )
                total_intact_rows += rows
                intact_rows_per_source[db_name] += rows
            except sqlite3.OperationalError as e:
                # Table might not exist in this source - expected for carved DBs
                # But log resource errors (ERRNO 24) which indicate handle exhaustion
                err_str = str(e).lower()
                if "errno 24" in err_str or "too many open files" in err_str:
                    logger.error(f"[RESOURCE] Handle exhaustion copying {table_name}: {e}")
                # Other operational errors (no such table) are expected - don't log
            except Exception as e:
                # Unexpected errors should be logged for debugging
                logger.debug(f"[DEBUG] Error copying {table_name} from {db_name}: {e}")

        # Periodic gc.collect() after each source database (not per-table)
        # Prevents file handle exhaustion on large databases like Powerlog
        gc.collect()

    # Deduplicate ALL tables (both intact and LF data)
    total_dupes = 0
    tables_to_dedup = (set(table_schemas) | tables_with_lf) - shared_tables

    # Update sub-progress for deduplication phase
    if reconstruction_sub and progress_context:
        progress_context.update_sub(
            reconstruction_sub,
            total_operations - 1,  # Before final vacuum step
            description=f"Deduplicating {len(tables_to_dedup)} tables",
        )

    for idx, table_name in enumerate(tables_to_dedup):
        if is_fts_table(table_name, table_schemas):
            continue

        current_step += 1

        try:
            dupes = deduplicate_table(output_db, table_name)
            if dupes > 0:
                total_dupes += dupes
                logger.debug(f"      Removed {dupes:,} duplicate rows from {table_name}")
        except sqlite3.OperationalError as e:
            err_str = str(e).lower()
            if "errno 24" in err_str or "too many open files" in err_str:
                logger.error(f"[RESOURCE] Handle exhaustion deduplicating {table_name}: {e}")
        except Exception as e:
            logger.debug(f"[DEBUG] Error deduplicating {table_name}: {e}")

    # Reclaim space from deleted duplicate rows
    # Without VACUUM, deleted pages remain in the freelist causing database bloat
    if total_dupes > 0:
        # Update sub-progress for vacuum phase
        if reconstruction_sub and progress_context:
            progress_context.update_sub(
                reconstruction_sub,
                total_operations,
                description=f"Vacuuming ({total_dupes:,} duplicates removed)",
            )
        try:
            with sqlite3.connect(output_db) as conn:
                conn.execute("VACUUM")
                # Switch to DELETE journal mode to avoid WAL file locks on Windows
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                conn.execute("PRAGMA journal_mode=DELETE;")
            # Explicitly delete WAL files (Windows compatibility)
            cleanup_wal_files(output_db)
            logger.debug(f"      Vacuumed database after removing {total_dupes:,} duplicates")
        except Exception as e:
            logger.debug(f"      Failed to vacuum database: {e}")

    # NOTE: Remnant/orphan databases are created centrally in lf_orphan.py
    # after all phases complete. This ensures each LF table appears in exactly
    # one location (either matched to an exemplar or in found_data/ as orphan).

    # Create manifest file
    source_db_list = []
    for entry in db_entries:
        db_name = entry["db_name"]
        source_info = {
            "db_name": db_name,
            "intact_rows": intact_rows_per_source.get(db_name, 0),
            "lf_rows": lf_rows_per_source.get(db_name, 0),
        }
        source_db_list.append(source_info)

    # Force garbage collection to release any unreferenced connections
    # before opening final readonly connection (prevents FD exhaustion)
    gc.collect()

    # Get table-level statistics
    with readonly_connection(output_db) as con:
        table_stats = []
        for table_name in table_schemas:
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
        "duplicates_removed": total_dupes,
        "table_stats": table_stats,
    }

    manifest_path = output_dir / f"{exemplar_name}_manifest.json"
    manifest_created = create_manifest_file(
        output_path=manifest_path,
        output_type=output_type,
        output_name=exemplar_name,
        source_databases=source_db_list,
        combined_stats=combined_stats,
        base_name=base_name,
        username=username,
        profiles=profiles,
    )

    if not manifest_created:
        logger.warning("      ⚠ Failed to create manifest")

    # Cleanup sub-task
    if reconstruction_sub and progress_context:
        progress_context.remove_sub(reconstruction_sub)

    return {
        "success": True,
        "output_db": output_db,
        "total_intact_rows": total_intact_rows,
        "total_lf_rows": total_lf_rows,
        "duplicates_removed": total_dupes,
        "manifest_created": manifest_created,
        "shared_tables_skipped": list(shared_tables),  # Tables routed to _multi
    }
