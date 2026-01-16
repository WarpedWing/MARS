#!/usr/bin/env python3
"""
Selects the most data-rich version of a SQLite DB.

First, it tries the original (O). If it passes checks and isn't empty, original is selected.
If corrupted or empty, it tries clone (C), then recovered (R). If the DB also matches an
exemplar, it also tries sqlite_dissect (D) to attempt a deeper data extraction.
If none provides usable data, the DB is marked (X) for byte carving to extract any embedded data.

If non-corrupt variants are tied in score (same amount of data), the variant closest to
the original is chosen. E.g., if R and C have the same amount of data, C will be selected.

Data similarity scores are also calculated for exemplar matches on a table-by-table basis
and incorporated into the final scoring matrix. This way, DB variants that may have more recovered
rows of data - but that data is determined to be too unlike the exemplar - will be penalized,
or even thrown out of the running.

Creates:
    Folders of chosen variants
    JSONL chronicling each processed variant
"""
# Template matcher for carved SQLite DBs (generic; signature-free)
# - Default: STRICT mode. Only exact schema matches are accepted (0% otherwise).
# - Default: require exact column sets (can opt-out with --no-require-columns).
# - If DB malformed; falls back to bytes scan of CREATE TABLE statements.
# - Streams JSONL: header + one record per case.

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path so template_selector package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

import contextlib
import gc
import hashlib
import json
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Lock, Thread

from mars.pipeline.raw_scanner.db_variant_selector.db_variant_selector_helpers import (
    _which_sqlite_dissect,
    build_ignore_sets_from_catalog,
    byte_scan_tables,
    discover_sqlites,
    discovery_report,
    find_sqlite_dissect_db,
    is_effectively_empty,
    is_ignorable_table,
    load_skip_databases_from_catalog,
    md5_bytes,
    nearest_by_tables,
    normalize_columns,
    parse_sqlite_master,
    rebuild_sqlite_dissect_to_exemplar_shape,
    run_pragma,
    run_sqlite_dissect,
    safe_connect,
    table_has_nonnull,
    table_rowcount,
    write_jsonl,
)
from mars.utils.debug_logger import logger

# Import centralized compression utilities for .gz handling
try:
    from mars.utils.compression_utils import recover_gzip  # type: ignore

    COMPRESSION_UTILS_AVAILABLE = True
except ImportError:
    COMPRESSION_UTILS_AVAILABLE = False

    # Define stub function to satisfy type checker
    def recover_gzip(*args, **kwargs) -> Path | None:  # type: ignore
        return None


from mars.pipeline.raw_scanner.db_variant_selector.models import DBMeta, Variant
from mars.pipeline.raw_scanner.db_variant_selector.schema_matcher import (
    compute_exact_matches,
    load_hash_lookup,
)
from mars.pipeline.raw_scanner.db_variant_selector.selector_output import build_case_record
from mars.pipeline.raw_scanner.db_variant_selector.selector_profiles import (
    compute_variant_profiles,
    select_profile_tables,
)
from mars.pipeline.raw_scanner.db_variant_selector.table_profiler import compare_table_profiles, profile_table
from mars.pipeline.raw_scanner.db_variant_selector.variant_operations import (
    SALVAGE_TABLE_CANON,
    _variant_score,
    choose_best_variant,
    introspect_with_repairs,
    is_salvage_name,
    resolve_variant_priority,
)


# ==========================================================================
# ====================== THREAD-SAFE STATISTICS ============================
# ==========================================================================
class SafeStats:
    """Thread-safe statistics tracking for multithreaded database processing."""

    def __init__(self):
        self._lock = Lock()
        self.dissect_stats = {
            "total_matched": 0,
            "has_recovered": 0,
            "has_integrity": 0,
            "binary_found": 0,
            "dissect_attempted": 0,
            "dissect_succeeded": 0,
            "dissect_rebuilt": 0,
        }
        self.skip_stats = {
            "skipped_count": 0,
            "skip_reasons": {},
        }

    def increment_dissect(self, key: str) -> None:
        """Atomically increment a dissect statistics counter."""
        with self._lock:
            self.dissect_stats[key] += 1

    def increment_skip(self, reason: str) -> None:
        """Atomically increment skip statistics."""
        with self._lock:
            self.skip_stats["skipped_count"] += 1
            self.skip_stats["skip_reasons"][reason] = self.skip_stats["skip_reasons"].get(reason, 0) + 1

    def get_dissect_stats(self) -> dict:
        """Get a snapshot of dissect statistics."""
        with self._lock:
            return dict(self.dissect_stats)

    def get_skip_stats(self) -> dict:
        """Get a snapshot of skip statistics."""
        with self._lock:
            return {
                "skipped_count": self.skip_stats["skipped_count"],
                "skip_reasons": dict(self.skip_stats["skip_reasons"]),
            }


def _jsonl_writer_thread(write_queue: Queue, results_path: Path) -> None:
    """
    Queue-based JSONL writer thread.

    Consumes records from the queue and writes them to the JSONL file.
    Stops when it receives None (sentinel value).

    Args:
        write_queue: Queue containing records to write (dict objects)
        results_path: Path to JSONL results file
    """
    # Keep file handle open for duration of processing to avoid FD exhaustion
    with results_path.open("a", encoding="utf-8") as f:
        while True:
            record = write_queue.get()
            if record is None:  # Sentinel value signals shutdown
                write_queue.task_done()
                break
            try:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()  # Ensure data is written promptly
            finally:
                write_queue.task_done()


# ==========================================================================
# ============================ INTROSPECTION ===============================
# ==========================================================================
def introspect_db(
    path: Path,
    ignore_tables: set[str],
    ignore_prefixes: set[str],
    ignore_suffixes: set[str],
    detect_salvage: bool = False,
) -> DBMeta:
    meta = DBMeta(path=path, opens=False)
    try:
        meta.file_hash = md5_bytes(path)
    except Exception as e:
        meta.notes.append(f"hash_fail: {e}")

    con = safe_connect(path)
    if not con:
        # Fallback: byte scan
        td = byte_scan_tables(path)
        meta.tables = td
        meta.table_names = set(td.keys())
        meta.notes.append("schema_fallback: byte_scan_tables used")
        meta.byte_scanned = True  # Mark as unreliable for exact matching

        # Salvage detection only if requested
        lower_all = {t.lower() for t in meta.table_names}
        if detect_salvage:
            lf_found = {t for t in lower_all if t in SALVAGE_TABLE_CANON}
            meta.has_lost_and_found = bool(lf_found)
            meta.lost_and_found_tables = lf_found

        # Build effective set excluding ignorable + salvage
        meta.effective_table_names = {
            t for t in meta.table_names if not is_ignorable_table(t, ignore_tables, ignore_prefixes, ignore_suffixes)
        }

        # columns_by_table for effective tables
        meta.columns_by_table = {}
        for t, tdef in td.items():
            if t in meta.effective_table_names:
                meta.columns_by_table[t] = normalize_columns(tdef.columns)

        meta.notes.append("open_fail: connection could not be established")
        return meta

    # Connected path
    meta.opens = True
    try:
        # Pragmas
        aid = run_pragma(con, "application_id")
        uv = run_pragma(con, "user_version")
        ps = run_pragma(con, "page_size")
        enc = run_pragma(con, "encoding")
        with contextlib.suppress(Exception):
            meta.application_id = int(aid) if aid and aid.isdigit() else None
        with contextlib.suppress(Exception):
            meta.user_version = int(uv) if uv and uv.isdigit() else None
        with contextlib.suppress(Exception):
            meta.page_size = int(ps) if ps and ps.isdigit() else None
        meta.encoding = enc

        tables, indices, rate = parse_sqlite_master(con, meta)
        if not tables:
            # fallback to byte scan
            td = byte_scan_tables(path)
            tables, indices, rate = td, set(), 0.0
            meta.notes.append("schema_fallback: byte_scan_tables used")
            meta.byte_scanned = True  # Mark as unreliable for exact matching
            meta.opens = False  # Schema not readable through normal SQLite operations

        meta.tables = tables
        meta.indices = indices
        meta.table_names = set(tables.keys())
        meta.ddl_parse_rate = rate
        meta.schema_ok = bool(tables)

        # Salvage detection only if requested
        lower_all = {t.lower() for t in meta.table_names}
        if detect_salvage:
            lf_found = {t for t in lower_all if t in SALVAGE_TABLE_CANON}
            meta.has_lost_and_found = bool(lf_found)
            meta.lost_and_found_tables = lf_found

        # Effective tables (exclude ignorable + salvage)
        meta.effective_table_names = {
            t for t in meta.table_names if not is_ignorable_table(t, ignore_tables, ignore_prefixes, ignore_suffixes)
        }

        # Non-empty / non-null probes (skip salvage)
        # Use original_name for queries (handles weird table names like \t, \n, etc.)
        probe_list = [t for t in sorted(meta.effective_table_names) if not is_salvage_name(t)][:50]
        for t in probe_list:
            try:
                # Use original name for querying if available
                query_name = tables[t].original_name or t
                has_row, has_nonnull = table_has_nonnull(con, query_name, sample_rows=50)
                if has_row:
                    meta.nonempty_tables.add(t)
                if has_nonnull:
                    meta.nonnull_tables.add(t)
            except sqlite3.Error:
                continue

        # columns_by_table for effective tables
        meta.columns_by_table = {}
        for t in meta.effective_table_names:
            meta.columns_by_table[t] = normalize_columns(tables[t].columns)

        # rows_by_table and total_rows for effective tables
        meta.rows_by_table = {}
        total = 0
        for t in sorted(meta.effective_table_names):
            try:
                # Use original name for querying if available
                query_name = tables[t].original_name or t
                rc = table_rowcount(con, query_name)
            except Exception:
                rc = 0
            meta.rows_by_table[t] = rc
            total += rc
        meta.total_rows = total

        # integrity_check
        ic = run_pragma(con, "integrity_check")
        meta.integrity_ok = (ic or "").lower() == "ok"

    except Exception as e:
        meta.notes.append(f"introspect_error: {e}")
    finally:
        with contextlib.suppress(Exception):
            con.close()

    return meta


# ===================================================================
# =================== SINGLE DATABASE PROCESSING ====================
# ===================================================================
def _process_single_database(
    case_path: Path,
    ignore_tables: set[str],
    ignore_prefixes: set[str],
    ignore_suffixes: set[str],
    exemplars: list[DBMeta],
    exemplars_root: Path,
    variants_root: Path,
    skip_patterns: dict[str, list[str]],
    hash_lookup: dict[str, list[str]],
    emit_nearest: bool,
    dissect_all: bool,
    script_dir: Path,
    schemas_dir: Path | None,
    safe_stats: SafeStats,
    error_log: Path,
) -> dict | None:
    """
    Process a single database case and return the record to write to JSONL.

    Returns None if the database was skipped (but skip is recorded in safe_stats).
    Returns a record dict to be written to JSONL otherwise.

    This function is thread-safe and can be called concurrently.
    """
    try:
        best, variants, saved_paths = introspect_with_repairs(
            case_path,
            ignore_tables,
            ignore_prefixes,
            ignore_suffixes,
            try_clone=True,
            try_recover=True,
            sqlite3_path=None,
            save_dir=variants_root,
        )
        case = best.meta

        # Check if this database should be skipped
        if skip_patterns:
            db_tables = set(case.table_names or [])
            should_skip = False
            skip_reason = None

            for skip_name, patterns in skip_patterns.items():
                matched = 0
                matched_tables = []
                for pattern in patterns:
                    pattern_lower = pattern.lower()
                    if "*" in pattern_lower:
                        import fnmatch

                        for table in db_tables:
                            if fnmatch.fnmatch(table, pattern_lower):
                                matched += 1
                                matched_tables.append(f"{table}(matches:{pattern})")
                                break
                    else:
                        if pattern_lower in db_tables:
                            matched += 1
                            matched_tables.append(pattern_lower)

                if matched >= len(patterns) * 0.7:
                    should_skip = True
                    skip_reason = skip_name
                    break

            if should_skip and skip_reason:
                safe_stats.increment_skip(skip_reason)

                from contextlib import suppress

                for saved_path in saved_paths.values():
                    if saved_path and Path(saved_path).exists():
                        with suppress(Exception):
                            Path(saved_path).unlink()

                return {
                    "type": "case",
                    "path": str(case_path),
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "tables": sorted(db_tables),
                }

        lf_tables_from_recover: set[str] = set()
        for v in variants:
            if v.tag == "R":
                lf_tables_from_recover = set(v.meta.lost_and_found_tables or set())
                break

        exact_matches = compute_exact_matches(case, exemplars, exemplars_root, hash_lookup)

        if exact_matches:
            safe_stats.increment_dissect("total_matched")

        recovered_variant = next(
            (v for v in variants if v.tag == "R" and v.meta.integrity_ok),
            None,
        )

        if recovered_variant:
            safe_stats.increment_dissect("has_recovered")
            if recovered_variant.meta.integrity_ok:
                safe_stats.increment_dissect("has_integrity")

        dissect_attempted = False
        dissect_pass = False
        dissect_rebuilt = False
        dissect_rebuilt_path: str | None = None
        should_attempt_dissect = recovered_variant is not None and (exact_matches or dissect_all)
        if should_attempt_dissect and recovered_variant:
            bin_path = _which_sqlite_dissect(script_dir)
            if bin_path:
                safe_stats.increment_dissect("binary_found")
                dissect_attempted = True
                safe_stats.increment_dissect("dissect_attempted")
                recovered_db = recovered_variant.path
                case_dir = recovered_db.parent
                debug_mode = False

                dissect_pass, _ = run_sqlite_dissect(bin_path, recovered_db, case_dir, debug=debug_mode)
                if dissect_pass:
                    safe_stats.increment_dissect("dissect_succeeded")

                if dissect_pass:
                    dissect_db = find_sqlite_dissect_db(recovered_db)
                    ex_path = Path(exact_matches[0]["sample_path"]) if exact_matches else None
                    if dissect_db:
                        saved_paths["dissect"] = dissect_db.as_posix()
                    if dissect_db and ex_path and ex_path.exists():
                        out_dissect = dissect_db.parent / f"{dissect_db.stem}.rebuilt.sqlite"
                        try:
                            rebuild_sqlite_dissect_to_exemplar_shape(
                                dissect_db,
                                ex_path,
                                out_dissect,
                                schemas_dir,
                            )
                            dissect_rebuilt = True
                            safe_stats.increment_dissect("dissect_rebuilt")
                            dissect_rebuilt_path = out_dissect.as_posix()
                            saved_paths["dissect_rebuilt"] = dissect_rebuilt_path
                            try:
                                dissect_meta = introspect_db(
                                    out_dissect,
                                    ignore_tables,
                                    ignore_prefixes,
                                    ignore_suffixes,
                                )
                                variants.append(
                                    Variant(
                                        "D",
                                        out_dissect,
                                        dissect_meta,
                                    )
                                )
                            except Exception:
                                pass  # Introspect error - not critical for rebuild
                        except Exception:
                            pass

        profile_tables = select_profile_tables(variants)
        exemplar_path = Path(exact_matches[0]["sample_path"]) if exact_matches and profile_tables else None
        profile_result = compute_variant_profiles(
            variants,
            exemplar_path,
            profile_tables,
            safe_connect=safe_connect,
            profile_table_fn=profile_table,
            compare_profiles_fn=compare_table_profiles,
        )

        best = choose_best_variant(variants, profile_result.profile_score_map)
        case = best.meta
        empty_bool, empty_reason = is_effectively_empty(case)

        if empty_bool:
            for v in variants:
                if v.tag == "O":
                    best = v
                    case = best.meta
                    break

        exact_matches = compute_exact_matches(case, exemplars, exemplars_root, hash_lookup)

        best_priority, best_closeness = resolve_variant_priority(best.tag)

        variant_score_map = {v.path.as_posix(): _variant_score(v.meta) for v in variants}
        variant_priorities = {v.tag: resolve_variant_priority(v.tag) for v in variants}

        # Compute nearest exemplars for:
        # 1. All unmatched databases (for metamatch/NEAREST routing decisions)
        # 2. Databases with L&F tables (for reconstruction matching)
        should_compute_nearest = emit_nearest and (not exact_matches or bool(lf_tables_from_recover))
        nearest = nearest_by_tables(case, exemplars, k=5) if should_compute_nearest else None

        record = build_case_record(
            case_path=case_path,
            best=best,
            best_priority=best_priority,
            best_closeness=best_closeness,
            variants=variants,
            variant_score_map=variant_score_map,
            variant_priorities=variant_priorities,
            lf_tables_from_recover=lf_tables_from_recover,
            saved_paths=saved_paths,
            exact_matches=exact_matches,
            empty_bool=empty_bool,
            empty_reason=empty_reason,
            dissect_attempted=dissect_attempted,
            dissect_pass=dissect_pass,
            dissect_rebuilt=dissect_rebuilt,
            dissect_rebuilt_path=dissect_rebuilt_path,
            nearest=nearest,
        )

        if empty_bool and best.tag == "O":
            record["variant_chosen"] = "X"
            if "chosen_variant" in record and record["chosen_variant"]:
                record["chosen_variant"]["tag"] = "X"

        return record

    except Exception as e:
        with error_log.open("a", encoding="utf-8") as ef:
            ef.write(f"[ERROR case] {case_path} : {e!r}\n")
        return None


# ===================================================================
# ============================= MAIN ================================
# ===================================================================
def main(
    cases_dir: Path,
    exemplars_dir: Path,
    variants_dir: Path,
    results_path: Path,
    config,  # MARSConfig (avoid circular import)
    schemas_dir: Path | None = None,
    use_catalog: bool = True,
    use_rubrics: bool = True,
    emit_nearest: bool = True,
    richConsole=None,  # rich.console.Console | None (avoid import at module level)
    sqlite_paths: list[Path] | None = None,
):
    """
    Main entry point for db_variant_selector.

    Args:
        cases_dir: Directory containing carved SQLite files (for discovery if sqlite_paths not provided)
        exemplars_dir: Directory containing exemplar databases (catalog or full exemplar set)
        variants_dir: Directory to save variant databases
        results_path: Path to output JSONL results file
        config: MARSConfig instance with all settings
        schemas_dir: Optional path to schemas directory with rubrics (for hash lookup)
        use_catalog: Use catalog for ignorable tables/skip patterns (default: True)
        use_rubrics: Use rubric-first matching for performance (default: True)
        emit_nearest: Include top-5 nearest exemplars in output (default: True)
        richConsole: Optional Rich console for progress display
        sqlite_paths: Optional pre-scanned list of SQLite paths (skips filesystem scan)
    """

    # Resolve paths
    exemplars_root = exemplars_dir.resolve()
    cases_dir = cases_dir.resolve()
    variants_root = variants_dir.resolve()
    results_path = results_path.resolve()

    # Point to src/ directory for resource lookup
    script_dir = (
        Path(__file__).resolve().parent.parent.parent.parent.parent
    )  # Always define for _which_sqlite_dissect()

    # Build a unique run tag for this processing run
    run_seed = f"{time.time_ns()}|{os.getpid()}|{cases_dir}|{exemplars_root}"
    run_tag = hashlib.md5(run_seed.encode("utf-8")).hexdigest()[:8]

    # Create directories if needed
    variants_root.mkdir(parents=True, exist_ok=True)
    error_log = variants_root.parent / "errors.log"

    logger.debug(f"Variants directory: {variants_root}")
    logger.debug(f"Results: {results_path}")

    # Reset output files
    for p in [results_path, error_log]:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    # Build ignore sets - use catalog if requested, otherwise use config
    if use_catalog:
        ignore_tables, ignore_prefixes, ignore_suffixes = build_ignore_sets_from_catalog()
        # Load skip_databases patterns from catalog (silent)
        skip_patterns = load_skip_databases_from_catalog()
    else:
        # Use values from config
        ignore_tables = config.schema_comparison.ignorable_tables.copy()
        ignore_prefixes = config.schema_comparison.ignorable_prefixes.copy()
        ignore_suffixes = config.schema_comparison.ignorable_suffixes.copy()
        skip_patterns = {}

    # Load hash lookup table for fast matching (if available)
    hash_lookup: dict[str, list[str]] = {}
    if use_rubrics and schemas_dir:
        if schemas_dir.exists():
            logger.debug(f"Loading schema index from {schemas_dir}")
            hash_lookup = load_hash_lookup(schemas_dir)
            if hash_lookup and len(hash_lookup) > 0:
                logger.debug(f"Loaded {len(hash_lookup)} hash entries")
            else:
                logger.debug(f"Hash lookup not found or empty at {schemas_dir}")
        else:
            logger.debug(f"Schemas directory not found: {schemas_dir}")

    # Use pre-scanned SQLite paths or discover them
    from mars.utils.progress_utils import (
        create_multithreaded_progress,
        create_standard_progress,
    )

    if sqlite_paths is not None:
        # Use pre-scanned list from file_categorizer
        cases = sqlite_paths
        if richConsole:
            logger.debug(f"[green]Received {len(cases)} pre-scanned SQLite databases[/green]")
    elif richConsole:
        # Discover SQLite files with progress bar
        logger.debug("[cyan]Discovering candidate files...[/cyan]")

        with create_standard_progress(
            "Scanning files",
            console=richConsole,
            show_count=True,
            show_percentage=True,
            config=config,
        ) as progress:
            task = progress.add_task("Discovering...", total=None)

            def update_progress(current: int, total: int):
                if progress.tasks[task].total != total:
                    progress.update(task, total=total)
                progress.update(task, completed=current)

            cases = discover_sqlites(cases_dir, progress_callback=update_progress)

        logger.debug(f"[green]Found {len(cases)} SQLite databases[/green]")
    else:
        cases = discover_sqlites(cases_dir)

    # Discover exemplar SQLite files (fast - no progress bar needed)
    if richConsole:
        logger.debug("[cyan]Finding exemplar files...[/cyan]")
    ex_paths = discover_sqlites(exemplars_dir)
    if richConsole:
        logger.debug(f"[green]Found {len(ex_paths)} exemplar databases[/green]")

    if richConsole:
        logger.debug("[cyan]Processing report...[/cyan]")

    discovery_report(cases_dir, cases, results_path)
    discovery_report(exemplars_dir, ex_paths, results_path)

    # Pre-introspect exemplars once (this is the slow part - show progress)
    # Use parallel loading for ~4x speedup (I/O bound database introspection)
    exemplars: list[DBMeta] = []
    exemplar_errors: list[tuple[Path, Exception]] = []
    max_exemplar_workers = 4

    def _introspect_single(path: Path) -> DBMeta | None:
        """Introspect a single exemplar database (worker function)."""
        try:
            return introspect_db(path, ignore_tables, ignore_prefixes, ignore_suffixes)
        except Exception as e:
            exemplar_errors.append((path, e))
            return None

    if richConsole:
        with create_standard_progress(
            "Loading exemplar schemas",
            console=richConsole,
            show_count=True,
            show_percentage=True,
            config=config,
        ) as progress:
            task = progress.add_task("Loading...", total=len(ex_paths))
            completed = 0

            # Parallel introspection with progress updates
            with ThreadPoolExecutor(max_workers=max_exemplar_workers) as executor:
                futures = {executor.submit(_introspect_single, p): p for p in ex_paths}
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        exemplars.append(result)
                    completed += 1
                    progress.update(task, completed=completed)
    else:
        # Non-interactive mode: still use parallel loading
        with ThreadPoolExecutor(max_workers=max_exemplar_workers) as executor:
            futures = {executor.submit(_introspect_single, p): p for p in ex_paths}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    exemplars.append(result)

    # Write any collected errors to the error log
    for path, error in exemplar_errors:
        with error_log.open("a", encoding="utf-8") as ef:
            ef.write(f"[ERROR exemplar] {path} : {error!r}\n")

    write_jsonl(
        results_path,
        {
            "type": "header",
            "run_id": run_tag,
            "version": 6,
            "mode": "strict",
            "exemplars_scanned": len(exemplars),
            "cases_scanned": len(cases),
            "results_file": str(results_path),
            "errors_file": str(error_log),
            "ignorable": {
                "tables": sorted(ignore_tables),
                "prefixes": sorted(ignore_prefixes),
                "suffixes": sorted(ignore_suffixes),
            },
            "skip_patterns": dict(skip_patterns) if skip_patterns else {},
        },
    )

    # =====================================================================
    # ======================== PROCESS CASES ==============================
    # =====================================================================
    # Thread-safe statistics tracking
    safe_stats = SafeStats()

    # Queue-based JSONL writer for thread safety
    write_queue: Queue = Queue()
    writer_thread = Thread(target=_jsonl_writer_thread, args=(write_queue, results_path), daemon=True)
    writer_thread.start()

    # Single-threaded processing to prevent FD exhaustion
    # Each database can consume 6+ file descriptors during variant creation
    # (subprocess pipes, SQLite connections, temp files)
    max_workers = 1

    # Process databases with Rich progress bar (or fallback to simple iteration)
    if richConsole:
        logger.debug(f"[cyan]Extracting data from {len(cases)} databases[/cyan]")

        with create_multithreaded_progress(
            "Processing",
            console=richConsole,
            config=config,
            header_title="Selecting Optimum DB Variants",
            show_time="elapsed",
        ) as progress:
            task = progress.add_task("Loading...", total=len(cases))

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all cases for processing
                futures = {
                    executor.submit(
                        _process_single_database,
                        case_path,
                        ignore_tables,
                        ignore_prefixes,
                        ignore_suffixes,
                        exemplars,
                        exemplars_root,
                        variants_root,
                        skip_patterns,
                        hash_lookup,
                        emit_nearest,
                        config.variant_selector.dissect_all,
                        script_dir,
                        schemas_dir,
                        safe_stats,
                        error_log,
                    ): case_path
                    for case_path in cases
                }

                # Process completed futures as they finish
                for future in as_completed(futures):
                    case_path = futures[future]
                    db_name = case_path.name
                    try:
                        # Add timeout to prevent hanging on problematic databases
                        # 120 seconds should be enough for variant processing
                        record = future.result(timeout=120)

                        # Put record in queue for writer thread (if not None/skipped)
                        if record:
                            write_queue.put(record)

                        # Update progress with last processed database
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]({db_name})",
                        )
                    except TimeoutError:
                        if progress.console:
                            progress.console.print(f"[yellow]Timeout processing {db_name} (>120s)[/yellow]")
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]({db_name})",
                        )
                    except Exception as e:
                        # Log any unexpected errors from worker
                        if progress.console:
                            progress.console.print(f"[yellow]Error processing {db_name}: {e}[/yellow]")
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]({db_name})",
                        )

                    # Force garbage collection after each database to prevent FD buildup
                    # Without this, subprocess calls (attempt_recover) can fail from FD exhaustion
                    gc.collect()
    else:
        # Fallback: process without Rich progress bar
        logger.info(f"Processing {len(cases)} databases with {max_workers} workers...")

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all cases for processing
            futures = {
                executor.submit(
                    _process_single_database,
                    case_path,
                    ignore_tables,
                    ignore_prefixes,
                    ignore_suffixes,
                    exemplars,
                    exemplars_root,
                    variants_root,
                    skip_patterns,
                    hash_lookup,
                    emit_nearest,
                    config.variant_selector.dissect_all,
                    script_dir,
                    schemas_dir,
                    safe_stats,
                    error_log,
                ): case_path
                for case_path in cases
            }

            # Process completed futures as they finish
            for future in as_completed(futures):
                case_path = futures[future]
                try:
                    # Get result from worker thread
                    record = future.result()

                    # Put record in queue for writer thread (if not None/skipped)
                    if record:
                        write_queue.put(record)
                except Exception as e:
                    # Log any unexpected errors from worker
                    logger.warning(f"Error processing {case_path.name}: {e}")

                # Force garbage collection after each database to prevent FD buildup
                # Without this, subprocess calls (attempt_recover) can fail from FD exhaustion
                gc.collect()

    # Shutdown writer thread cleanly
    write_queue.put(None)  # Sentinel to stop writer
    writer_thread.join()

    # Force garbage collection to release any lingering file handles
    # This helps prevent FD exhaustion (ERRNO 24) on systems with low limits
    gc.collect()

    # Get final statistics
    dissect_stats = safe_stats.get_dissect_stats()

    # Critical warnings only (not verbose progress)
    if (
        dissect_stats["total_matched"] > 0
        and dissect_stats["dissect_attempted"] == 0
        and dissect_stats["binary_found"] == 0
    ):
        logger.warning("sqlite_dissect binary not found - cannot recover corrupted databases")

    write_jsonl(results_path, {"type": "done"})


if __name__ == "__main__":
    logger.error("db_variant_selector is not meant to be run directly")
    import sys

    sys.exit(1)
