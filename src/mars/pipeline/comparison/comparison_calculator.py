"""
Comparison Calculator for MARS.

Compares exemplar and candidate databases to calculate:
- Row deduplication (what's unique to candidate vs overlap)
- Timeline coverage extension (date ranges filled)
- Recovery metrics per database and aggregate

Uses existing rubric metadata for timestamp column detection.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mars.pipeline.comparison.types import (
    ComparisonResult,
    DatabaseComparison,
    LostAndFoundStats,
    TableComparison,
    _is_sentinel_date,
)
from mars.pipeline.matcher.rubric_utils import TimestampFormat
from mars.utils.database_utils import quote_ident, readonly_connection
from mars.utils.debug_logger import logger


class ComparisonCalculator:
    """
    Calculate comparison metrics between exemplar and candidate databases.

    Uses rubric metadata for timestamp column detection and format conversion.
    """

    def __init__(
        self,
        exemplar_db_dir: Path,
        candidate_matched_dir: Path,
        candidate_results_jsonl: Path,
        rubrics_dir: Path | None = None,
    ):
        """
        Initialize comparison calculator.

        Args:
            exemplar_db_dir: Path to exemplar/databases/ directory
            candidate_matched_dir: Path to candidates/run/matched/ directory
            candidate_results_jsonl: Path to sqlite_scan_results.jsonl
            rubrics_dir: Path to rubrics directory (auto-detected if None)
        """
        self.exemplar_db_dir = exemplar_db_dir
        self.candidate_matched_dir = candidate_matched_dir
        self.candidate_results_jsonl = candidate_results_jsonl

        # Auto-detect rubrics directory
        # Rubrics are in exemplar/databases/schemas/ (same level as catalog/)
        if rubrics_dir is None:
            self.rubrics_dir = exemplar_db_dir / "schemas"
        else:
            self.rubrics_dir = rubrics_dir

    def calculate(self) -> ComparisonResult:
        """
        Calculate comparison between exemplar and candidate databases.

        Returns:
            ComparisonResult with all metrics
        """
        # Build mapping of exemplar databases
        exemplar_dbs = self._discover_exemplar_databases()

        # Discover candidate databases from catalog (preferred) or scan results
        candidate_catalog = self._discover_candidate_catalog()

        # Collect all comparisons
        raw_comparisons: list[DatabaseComparison] = []

        if candidate_catalog:
            # Use catalog-based comparison (more reliable)
            # candidate_catalog is keyed by folder name (e.g., "DB Name_username")
            # exemplar_dbs is keyed by base name (e.g., "DB Name")
            # We need to group candidates by their matching exemplar base name

            # First pass: group candidates by exemplar match
            candidates_by_exemplar: dict[str, dict] = {}

            for cand_folder_name, cand_info in candidate_catalog.items():
                exemplar_match = self._find_exemplar_match(cand_folder_name, exemplar_dbs)

                if not exemplar_match:
                    continue

                # Aggregate candidate info under the exemplar base name
                if exemplar_match not in candidates_by_exemplar:
                    candidates_by_exemplar[exemplar_match] = {
                        "paths": [],
                        "manifests": [],
                        "lf_rows": 0,
                        "lf_dirs": [],
                        "labels": [],
                    }

                # Merge this candidate's info into the aggregated entry
                agg = candidates_by_exemplar[exemplar_match]
                agg["paths"].extend(cand_info.get("paths", []))
                agg["manifests"].extend(cand_info.get("manifests", []))
                agg["lf_rows"] += cand_info.get("lf_rows", 0)
                agg["lf_dirs"].extend(cand_info.get("lf_dirs", []))
                agg["labels"].extend(cand_info.get("labels", []))

            # Second pass: compare each aggregated candidate group to exemplar
            for exemplar_name, aggregated_cand_info in candidates_by_exemplar.items():
                exemplar_info = exemplar_dbs[exemplar_name]
                category = exemplar_info.get("category", "unknown")

                db_comparison = self._compare_databases_from_catalog(
                    name=exemplar_name,
                    category=category,
                    exemplar_info=exemplar_info,
                    candidate_info=aggregated_cand_info,
                )

                raw_comparisons.append(db_comparison)
        else:
            # Fallback to scan results (legacy approach)
            raw_comparisons = self._compare_from_scan_results(exemplar_dbs)

        # Aggregate by database name (multiple candidates may match same exemplar)
        databases = self._aggregate_by_database(raw_comparisons)

        # Calculate totals from aggregated results
        total_unique = 0
        total_overlap = 0
        new_data_count = 0
        total_lf_rows = 0
        dbs_with_lf = 0
        rebuilt_count = 0
        error_count = 0
        path_failures = 0

        for db in databases:
            if db.error:
                error_count += 1
                if "not found" in db.error.lower():
                    path_failures += 1
                logger.debug(f"Comparison error for {db.name}: {db.error}")
            if db.rebuilt_from_lf:
                rebuilt_count += 1
            if db.total_unique_recovered > 0:
                new_data_count += 1
                total_unique += db.total_unique_recovered
            total_overlap += db.total_overlap
            if db.lost_and_found.total_rows > 0:
                total_lf_rows += db.lost_and_found.total_rows
                dbs_with_lf += 1

        if error_count > 0:
            logger.warning(f"Comparison had {error_count} errors ({path_failures} path resolution failures)")

        # Calculate category breakdown
        by_category = self._calculate_category_breakdown(databases)

        return ComparisonResult(
            exemplar_database_count=len(exemplar_dbs),
            candidate_matched_count=len(databases),
            candidate_with_new_data_count=new_data_count,
            total_unique_rows_recovered=total_unique,
            total_overlap_rows=total_overlap,
            total_lf_rows_recovered=total_lf_rows,
            databases_with_lf=dbs_with_lf,
            rebuilt_databases_count=rebuilt_count,
            databases=databases,
            by_category=by_category,
            exemplar_scan_dir=str(self.exemplar_db_dir.parent),
            candidate_run_dir=str(self.candidate_matched_dir.parent),
            generated_at=datetime.now(UTC).isoformat(),
            databases_with_errors=error_count,
            path_resolution_failures=path_failures,
        )

    def _load_candidate_results(self) -> list[dict]:
        """Load sqlite_scan_results.jsonl."""
        results = []
        if not self.candidate_results_jsonl.exists():
            logger.warning(f"Candidate results not found: {self.candidate_results_jsonl}")
            return results

        with Path.open(self.candidate_results_jsonl) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return results

    def _discover_candidate_catalog(self) -> dict[str, dict]:
        """
        Discover candidate databases from catalog directory structure, grouped by base name.

        The catalog contains combined databases with manifests showing
        aggregated stats including L&F row counts. Groups by catalog base name
        to enable conglomeration of user variants and _multi databases.

        Returns:
            Dict mapping catalog base name to {paths[], manifests[], lf_rows, lf_dirs[], labels[]}
        """
        catalog: dict[str, dict] = {}

        # Catalog is in run_dir/databases/catalog/
        run_dir = self.candidate_matched_dir.parent
        catalog_dir = run_dir / "databases" / "catalog"

        if not catalog_dir.exists():
            logger.debug(f"Candidate catalog not found: {catalog_dir}")
            return catalog

        def get_base_name(entry_dir: Path, folder_name: str) -> str:
            """Get catalog base name from provenance or manifest, or extract from folder."""
            # Try provenance file first
            for prov_file in entry_dir.glob("*.provenance.json"):
                try:
                    with prov_file.open() as f:
                        prov = json.load(f)
                    if "base_name" in prov:
                        return prov["base_name"]
                except Exception:
                    pass

            # Try manifest file
            for manifest_file in entry_dir.glob("*_manifest.json"):
                try:
                    with manifest_file.open() as f:
                        manifest = json.load(f)
                    if "base_name" in manifest:
                        return manifest["base_name"]
                except Exception:
                    pass

            # No base_name found - return folder name as-is
            # Matching will be done later against exemplar base names
            return folder_name

        for entry_dir in catalog_dir.iterdir():
            if not entry_dir.is_dir() or entry_dir.name.startswith("."):
                continue

            folder_name = entry_dir.name
            base_name = get_base_name(entry_dir, folder_name)

            # Find the combined database file
            db_files = list(entry_dir.glob("*.sqlite")) + list(entry_dir.glob("*.db"))
            db_path = db_files[0] if db_files else None

            # Find and load manifest
            manifest_files = list(entry_dir.glob("*_manifest.json"))
            manifest = None
            total_lf_rows = 0

            if manifest_files:
                try:
                    with Path.open(manifest_files[0]) as f:
                        manifest = json.load(f)
                    # Sum up L&F rows from all source databases
                    for src in manifest.get("source_databases", []):
                        total_lf_rows += src.get("lf_rows", 0)
                except (json.JSONDecodeError, OSError):
                    pass

            # Check for rejected subfolder (contains rows that failed validation)
            lf_dir = entry_dir / "rejected"

            # Group by base_name for conglomeration
            if base_name not in catalog:
                catalog[base_name] = {
                    "paths": [],
                    "manifests": [],
                    "lf_rows": 0,
                    "lf_dirs": [],
                    "labels": [],  # Track original folder names
                }

            if db_path:
                catalog[base_name]["paths"].append(db_path)
            if manifest:
                catalog[base_name]["manifests"].append(manifest)
            catalog[base_name]["lf_rows"] += total_lf_rows
            if lf_dir and lf_dir.exists():
                catalog[base_name]["lf_dirs"].append(lf_dir)
            catalog[base_name]["labels"].append(folder_name)

        logger.debug(f"Discovered {len(catalog)} candidate catalog groups")
        return catalog

    def _find_exemplar_match(self, candidate_folder_name: str, exemplar_dbs: dict[str, dict]) -> str | None:
        """
        Find matching exemplar for a candidate database folder name.

        Exemplar databases are grouped by their true catalog base name (e.g., "Chrome Cookies").
        Candidate folder names may have user suffixes (e.g., "Chrome Cookies_someuser").

        Matching strategy:
        1. Direct match (candidate folder name equals exemplar base name)
        2. Prefix match (candidate folder name starts with exemplar base name + "_")

        Args:
            candidate_folder_name: Folder name from candidate catalog (may have suffixes)
            exemplar_dbs: Dict mapping exemplar base names to their info

        Returns:
            Matching exemplar base name, or None if no match
        """
        # Direct match (no suffix case, e.g., "Powerlog")
        if candidate_folder_name in exemplar_dbs:
            return candidate_folder_name

        # Prefix match: "Chrome Cookies_someuser" should match "Chrome Cookies"
        # Sort by length (longest first) to match most specific base name
        for exemplar_base_name in sorted(exemplar_dbs.keys(), key=len, reverse=True):
            if candidate_folder_name.startswith(exemplar_base_name + "_"):
                return exemplar_base_name

        return None

    def _compare_databases_from_catalog(
        self,
        name: str,
        category: str,
        exemplar_info: dict,
        candidate_info: dict,
    ) -> DatabaseComparison:
        """
        Compare databases using catalog structure, aggregating across all paths.

        Handles conglomerated databases where multiple user variants are grouped
        by catalog base name (e.g., Chrome Cookies_user1 + Chrome Cookies_user2
        + Chrome Cookies_multi are all compared as "Chrome Cookies").

        Args:
            name: Catalog base name (e.g., "Chrome Cookies")
            category: Database category
            exemplar_info: Dict with paths[], rubrics[], labels[]
            candidate_info: Dict with paths[], manifests[], lf_rows, lf_dirs[], labels[]
        """
        exemplar_paths = exemplar_info.get("paths", [])
        candidate_paths = candidate_info.get("paths", [])
        lf_rows = candidate_info.get("lf_rows", 0)
        manifests = candidate_info.get("manifests", [])

        # Use first available paths for the comparison record
        exemplar_path = exemplar_paths[0] if exemplar_paths else None
        candidate_path = candidate_paths[0] if candidate_paths else None

        # Check if this is a rebuilt database (L&F only, no intact data)
        is_rebuilt = False
        for manifest in manifests:
            combined_output = manifest.get("combined_output", {}) if manifest else {}
            total_intact_rows = combined_output.get("total_intact_rows", -1)
            if total_intact_rows == 0 and lf_rows > 0:
                is_rebuilt = True
                break

        comparison = DatabaseComparison(
            name=name,
            category=category,
            exemplar_path=exemplar_path,
            candidate_path=candidate_path,
            matched=bool(exemplar_paths) and bool(candidate_paths),
            rebuilt_from_lf=is_rebuilt,
        )

        # Set L&F stats from manifest
        comparison.lost_and_found = LostAndFoundStats(
            total_rows=lf_rows,
            table_count=1 if lf_rows > 0 else 0,
            tables=["from_manifest"] if lf_rows > 0 else [],
        )

        if not exemplar_paths:
            comparison.error = "Exemplar database not found"
            return comparison

        if not candidate_paths:
            # If we have L&F rows but no main DB, it's a rebuilt database
            if lf_rows > 0:
                comparison.rebuilt_from_lf = True
                return comparison
            comparison.error = "Candidate database not found"
            return comparison

        # Use union-then-compare approach to avoid over-counting when multiple
        # user variants exist. Instead of comparing each exemplar×candidate pair
        # and summing (which counts the same candidate row multiple times),
        # we union all exemplar hashes and all candidate hashes, then compare once.
        #
        # IMPORTANT: We must use consistent column sets and ordering when hashing
        # rows from different databases. Otherwise, the same row content hashed
        # with different column orders produces different hashes.
        rubrics = exemplar_info.get("rubrics", [])
        rubric = self._select_best_rubric(rubrics)

        try:
            # Phase 1: Discover tables and columns from all paths
            # We need to find common columns for each table BEFORE hashing
            exemplar_table_columns: dict[str, set[str]] = {}  # table -> set of column sets
            candidate_table_columns: dict[str, set[str]] = {}
            exemplar_date_data: dict[str, dict] = {}  # table_name -> date info
            candidate_date_data: dict[str, dict] = {}  # table_name -> date info
            seen_lf_tables: set[str] = set()

            # Collect column info from exemplar paths
            for ex_path in exemplar_paths:
                if not ex_path or not ex_path.exists():
                    continue
                with readonly_connection(ex_path) as ex_conn:
                    ex_tables = self._get_tables(ex_conn)
                    for table_name in ex_tables:
                        cols = self._get_hashable_columns(ex_conn, table_name)
                        if table_name not in exemplar_table_columns:
                            exemplar_table_columns[table_name] = set(cols)
                        else:
                            # Intersect to find common columns across variants
                            exemplar_table_columns[table_name] &= set(cols)

                        # Collect date info for timeline analysis
                        ts_col, ts_format = self._get_timestamp_column(table_name, rubric)
                        if ts_col:
                            qname = quote_ident(table_name)
                            date_range = self._get_date_range(ex_conn, qname, ts_col, ts_format)
                            dates = self._get_unique_dates(ex_conn, qname, ts_col, ts_format)

                            if table_name not in exemplar_date_data:
                                exemplar_date_data[table_name] = {
                                    "ts_col": ts_col,
                                    "ts_format": ts_format,
                                    "date_range": date_range,
                                    "dates": set(dates),
                                }
                            else:
                                # Merge date ranges
                                existing = exemplar_date_data[table_name]
                                if date_range:
                                    if existing["date_range"]:
                                        existing["date_range"] = (
                                            min(existing["date_range"][0], date_range[0]),
                                            max(existing["date_range"][1], date_range[1]),
                                        )
                                    else:
                                        existing["date_range"] = date_range
                                existing["dates"].update(dates)

            # Collect column info from candidate paths
            for cand_path in candidate_paths:
                if not cand_path or not cand_path.exists():
                    continue
                with readonly_connection(cand_path) as cand_conn:
                    cand_tables = self._get_tables(cand_conn)

                    # Check for L&F tables (only count once)
                    cand_lf_stats = self._get_lost_and_found_stats(cand_conn)
                    for lf_table in cand_lf_stats.tables:
                        if lf_table not in seen_lf_tables:
                            seen_lf_tables.add(lf_table)
                            try:
                                cur = cand_conn.cursor()
                                cur.execute(f"SELECT COUNT(*) FROM {quote_ident(lf_table)}")
                                count = cur.fetchone()[0]
                                comparison.lost_and_found.total_rows += count
                            except sqlite3.Error:
                                pass
                    comparison.lost_and_found.table_count = len(seen_lf_tables)
                    comparison.lost_and_found.tables = list(seen_lf_tables)

                    for table_name in cand_tables:
                        cols = self._get_hashable_columns(cand_conn, table_name)
                        if table_name not in candidate_table_columns:
                            candidate_table_columns[table_name] = set(cols)
                        else:
                            # Intersect to find common columns across variants
                            candidate_table_columns[table_name] &= set(cols)

                        # Collect date info for timeline analysis
                        ts_col, ts_format = self._get_timestamp_column(table_name, rubric)
                        if ts_col:
                            qname = quote_ident(table_name)
                            date_range = self._get_date_range(cand_conn, qname, ts_col, ts_format)
                            dates = self._get_unique_dates(cand_conn, qname, ts_col, ts_format)

                            if table_name not in candidate_date_data:
                                candidate_date_data[table_name] = {
                                    "ts_col": ts_col,
                                    "ts_format": ts_format,
                                    "date_range": date_range,
                                    "dates": set(dates),
                                }
                            else:
                                # Merge date ranges
                                existing = candidate_date_data[table_name]
                                if date_range:
                                    if existing["date_range"]:
                                        existing["date_range"] = (
                                            min(existing["date_range"][0], date_range[0]),
                                            max(existing["date_range"][1], date_range[1]),
                                        )
                                    else:
                                        existing["date_range"] = date_range
                                existing["dates"].update(dates)

            # Determine common tables and their common columns (sorted for consistency)
            common_tables = set(exemplar_table_columns.keys()) & set(candidate_table_columns.keys())
            table_hash_columns: dict[str, list[str]] = {}
            for table_name in common_tables:
                ex_cols = exemplar_table_columns.get(table_name, set())
                cand_cols = candidate_table_columns.get(table_name, set())
                common_cols = ex_cols & cand_cols
                if common_cols:
                    # Sort alphabetically for consistent ordering
                    table_hash_columns[table_name] = sorted(common_cols)

            # Phase 2: Collect row hashes using consistent column sets
            all_exemplar_hashes: dict[str, set[int]] = {}
            all_candidate_hashes: dict[str, set[int]] = {}
            all_lf_hashes: dict[str, set[int]] = {}  # L&F-only hashes for separate tracking
            carved_row_counts: dict[str, int] = {}  # Carved row counts per table
            lf_row_counts: dict[str, int] = {}  # L&F row counts per table

            for ex_path in exemplar_paths:
                if not ex_path or not ex_path.exists():
                    continue
                with readonly_connection(ex_path) as ex_conn:
                    for table_name, columns in table_hash_columns.items():
                        hashes = self._get_row_hashes(ex_conn, table_name, columns)
                        if table_name not in all_exemplar_hashes:
                            all_exemplar_hashes[table_name] = set()
                        all_exemplar_hashes[table_name].update(hashes)

            for cand_path in candidate_paths:
                if not cand_path or not cand_path.exists():
                    continue
                with readonly_connection(cand_path) as cand_conn:
                    for table_name, columns in table_hash_columns.items():
                        # Get all hashes (for total candidate dedup)
                        hashes = self._get_row_hashes(cand_conn, table_name, columns)
                        if table_name not in all_candidate_hashes:
                            all_candidate_hashes[table_name] = set()
                        all_candidate_hashes[table_name].update(hashes)

                        # Get L&F-only hashes for separate tracking
                        lf_hashes = self._get_row_hashes_by_source(cand_conn, table_name, columns, lf_only=True)
                        if table_name not in all_lf_hashes:
                            all_lf_hashes[table_name] = set()
                        all_lf_hashes[table_name].update(lf_hashes)

                        # Count rows by source type for accurate row counts
                        carved, lf = self._count_rows_by_source(cand_conn, table_name)
                        carved_row_counts[table_name] = carved_row_counts.get(table_name, 0) + carved
                        lf_row_counts[table_name] = lf_row_counts.get(table_name, 0) + lf

            # Phase 3: Compare ONCE per table using unioned hash sets

            for table_name in table_hash_columns:
                ex_hashes = all_exemplar_hashes.get(table_name, set())
                cand_hashes = all_candidate_hashes.get(table_name, set())

                # Calculate set differences (correct - each row counted once)
                unique_to_candidate = len(cand_hashes - ex_hashes)
                overlap = len(cand_hashes & ex_hashes)
                unique_to_exemplar = len(ex_hashes - cand_hashes)

                # Get timestamp info
                ts_col = None
                ts_format = None
                ex_date_range = None
                cand_date_range = None
                ex_dates: list[str] = []
                cand_dates: list[str] = []

                if table_name in exemplar_date_data:
                    ed = exemplar_date_data[table_name]
                    ts_col = ed["ts_col"]
                    ts_format = ed["ts_format"]
                    ex_date_range = ed["date_range"]
                    ex_dates = sorted(ed["dates"])

                if table_name in candidate_date_data:
                    cd = candidate_date_data[table_name]
                    if not ts_col:
                        ts_col = cd["ts_col"]
                        ts_format = cd["ts_format"]
                    cand_date_range = cd["date_range"]
                    cand_dates = sorted(cd["dates"])

                # Calculate timeline extension
                extended_days = 0
                extended_after_days = 0
                if ex_date_range and cand_date_range:
                    if cand_date_range[0] < ex_date_range[0]:
                        extended_days = (ex_date_range[0] - cand_date_range[0]).days
                    if cand_date_range[1] > ex_date_range[1]:
                        extended_after_days = (cand_date_range[1] - ex_date_range[1]).days

                unique_cand_days = len(set(cand_dates) - set(ex_dates))

                # Get row counts by source (carved vs L&F)
                carved_count = carved_row_counts.get(table_name, 0)
                lf_count = lf_row_counts.get(table_name, 0)

                table_comparison = TableComparison(
                    name=table_name,
                    exemplar_rows=len(ex_hashes),  # Deduplicated count
                    candidate_rows=carved_count,  # Carved/intact rows only (via data_source)
                    lf_rows=lf_count,  # L&F rows (data_source LIKE 'found_%')
                    unique_to_candidate=unique_to_candidate,  # Total unique recovered (carved + L&F)
                    overlap=overlap,
                    unique_to_exemplar=unique_to_exemplar,
                    timestamp_column=ts_col,
                    timestamp_format=ts_format,
                    exemplar_date_range=ex_date_range,
                    candidate_date_range=cand_date_range,
                    timeline_extended_days=extended_days,
                    timeline_extended_after_days=extended_after_days,
                    exemplar_dates=ex_dates,
                    candidate_dates=cand_dates,
                    unique_candidate_days=unique_cand_days,
                )

                comparison.tables.append(table_comparison)

            if not comparison.tables:
                if comparison.lost_and_found.total_rows > 0:
                    comparison.rebuilt_from_lf = True
                    return comparison
                comparison.error = "No common tables found in any path combination"
                return comparison

        except sqlite3.Error as e:
            comparison.error = f"SQLite error: {e}"
        except Exception as e:
            comparison.error = f"Error: {e}"

        return comparison

    def _compare_from_scan_results(self, exemplar_dbs: dict[str, dict]) -> list[DatabaseComparison]:
        """
        Backup comparison using sqlite_scan_results.jsonl.

        Fallback when catalog is not available.
        """
        raw_comparisons: list[DatabaseComparison] = []
        candidate_results = self._load_candidate_results()

        for result in candidate_results:
            if result.get("type") != "case":
                continue

            exact_matches = result.get("exact_matches", [])
            if not exact_matches:
                continue

            chosen_variant = result.get("chosen_variant", {})
            variant_path = chosen_variant.get("path", "")

            if not variant_path:
                continue

            candidate_path = self._resolve_candidate_path(variant_path)

            for match_info in exact_matches:
                match_name = match_info.get("label") if isinstance(match_info, dict) else match_info
                if not match_name:
                    continue

                exemplar_info = exemplar_dbs.get(match_name)
                if not exemplar_info:
                    continue

                exemplar_path = exemplar_info.get("path")
                category = exemplar_info.get("category", "unknown")

                db_comparison = self._compare_databases(
                    name=match_name,
                    category=category,
                    exemplar_path=exemplar_path,
                    candidate_path=candidate_path,
                )

                raw_comparisons.append(db_comparison)

        return raw_comparisons

    def _aggregate_by_database(self, comparisons: list[DatabaseComparison]) -> list[DatabaseComparison]:
        """
        Aggregate multiple comparisons for the same database into one.

        When multiple candidate files match the same exemplar database,
        combine their metrics into a single entry per database type.
        """
        if not comparisons:
            return []

        # Group by database name
        by_name: dict[str, list[DatabaseComparison]] = {}
        for comp in comparisons:
            if comp.name not in by_name:
                by_name[comp.name] = []
            by_name[comp.name].append(comp)

        # Aggregate each group
        aggregated: list[DatabaseComparison] = []
        for name, group in by_name.items():
            if len(group) == 1:
                # No aggregation needed
                aggregated.append(group[0])
                continue

            # Merge multiple comparisons into one
            merged = self._merge_comparisons(name, group)
            aggregated.append(merged)

        return aggregated

    def _merge_comparisons(self, name: str, group: list[DatabaseComparison]) -> DatabaseComparison:
        """Merge multiple comparisons for the same database."""
        # Use first as base (all should have same category)
        base = group[0]

        # Merge tables: combine all tables, aggregate by table name
        merged_tables: dict[str, TableComparison] = {}
        for comp in group:
            for table in comp.tables:
                if table.name not in merged_tables:
                    merged_tables[table.name] = table
                else:
                    # Aggregate table stats
                    existing = merged_tables[table.name]
                    merged_tables[table.name] = TableComparison(
                        name=table.name,
                        exemplar_rows=existing.exemplar_rows,  # Same exemplar
                        candidate_rows=existing.candidate_rows + table.candidate_rows,
                        lf_rows=existing.lf_rows + table.lf_rows,
                        unique_to_candidate=existing.unique_to_candidate + table.unique_to_candidate,
                        overlap=existing.overlap + table.overlap,
                        unique_to_exemplar=existing.unique_to_exemplar,  # Same exemplar
                        timestamp_column=existing.timestamp_column or table.timestamp_column,
                        timestamp_format=existing.timestamp_format or table.timestamp_format,
                        # Use widest date ranges
                        exemplar_date_range=existing.exemplar_date_range or table.exemplar_date_range,
                        candidate_date_range=self._merge_date_ranges(
                            existing.candidate_date_range, table.candidate_date_range
                        ),
                        timeline_extended_days=max(
                            existing.timeline_extended_days,
                            table.timeline_extended_days,
                        ),
                        timeline_extended_after_days=max(
                            existing.timeline_extended_after_days,
                            table.timeline_extended_after_days,
                        ),
                    )

        # Merge lost_and_found stats
        merged_lf = LostAndFoundStats(
            table_count=sum(c.lost_and_found.table_count for c in group),
            total_rows=sum(c.lost_and_found.total_rows for c in group),
            tables=list({t for c in group for t in c.lost_and_found.tables}),
        )

        # Only mark as rebuilt if ALL entries are L&F only
        # If any entry has carved data, the merged result is Carved + L&F
        all_rebuilt = all(c.rebuilt_from_lf for c in group)

        return DatabaseComparison(
            name=name,
            category=base.category,
            exemplar_path=base.exemplar_path,
            candidate_path=None,  # Multiple candidates
            matched=True,
            rebuilt_from_lf=all_rebuilt,
            tables=list(merged_tables.values()),
            lost_and_found=merged_lf,
        )

    def _merge_date_ranges(
        self,
        range1: tuple[datetime, datetime] | None,
        range2: tuple[datetime, datetime] | None,
    ) -> tuple[datetime, datetime] | None:
        """Merge two date ranges to get the widest coverage."""
        if range1 is None:
            return range2
        if range2 is None:
            return range1
        return (min(range1[0], range2[0]), max(range1[1], range2[1]))

    def _discover_exemplar_databases(self) -> dict[str, dict]:
        """
        Discover exemplar databases and their metadata, grouped by catalog base name.

        Groups databases by their catalog base name (e.g., "Chrome Cookies") instead of
        folder name (e.g., "Chrome Cookies_username"). This enables conglomeration of
        data from different users/variants into a single comparison entry.

        Returns:
            Dict mapping catalog base name to {paths[], category, rubrics[], labels[]}
        """
        exemplar_dbs: dict[str, dict] = {}

        if not self.exemplar_db_dir.exists():
            logger.warning(f"Exemplar DB dir not found: {self.exemplar_db_dir}")
            return exemplar_dbs

        def get_base_name(db_path: Path, folder_label: str) -> str:
            """Get catalog base name from provenance file, fallback to folder label."""
            # Try meta-provenance first (at folder level for multi-profile dbs)
            folder_path = db_path.parent
            if folder_path.name != folder_label:
                # Nested profile structure: catalog/Chrome Cookies_user/Default/Cookies
                folder_path = folder_path.parent

            meta_prov = folder_path / f"{folder_label}.provenance.json"
            if meta_prov.exists():
                try:
                    with meta_prov.open() as f:
                        prov = json.load(f)
                    return prov.get("base_name", prov.get("name", folder_label))
                except Exception:
                    pass

            # Try database-level provenance
            db_prov = db_path.parent / f"{db_path.stem}.provenance.json"
            if db_prov.exists():
                try:
                    with db_prov.open() as f:
                        prov = json.load(f)
                    return prov.get("base_name", prov.get("name", folder_label))
                except Exception:
                    pass

            return folder_label

        # Walk through database files
        for db_path in self.exemplar_db_dir.rglob("*.db"):
            folder_label = self._exemplar_label(db_path)
            base_name = get_base_name(db_path, folder_label)

            category = self._extract_category(db_path)
            rubric = self._load_rubric(folder_label)

            # Group by base_name for conglomeration
            if base_name not in exemplar_dbs:
                exemplar_dbs[base_name] = {
                    "paths": [],
                    "category": category,
                    "rubrics": [],
                    "labels": [],  # Track original folder labels for rubric lookup
                }

            exemplar_dbs[base_name]["paths"].append(db_path)
            exemplar_dbs[base_name]["labels"].append(folder_label)
            if rubric:
                exemplar_dbs[base_name]["rubrics"].append(rubric)

        # Also check for .sqlite files
        for db_path in self.exemplar_db_dir.rglob("*.sqlite*"):
            folder_label = self._exemplar_label(db_path)
            base_name = get_base_name(db_path, folder_label)

            if base_name in exemplar_dbs and db_path in exemplar_dbs[base_name]["paths"]:
                continue  # Skip duplicates

            category = self._extract_category(db_path)
            rubric = self._load_rubric(folder_label)

            if base_name not in exemplar_dbs:
                exemplar_dbs[base_name] = {
                    "paths": [],
                    "category": category,
                    "rubrics": [],
                    "labels": [],
                }

            exemplar_dbs[base_name]["paths"].append(db_path)
            exemplar_dbs[base_name]["labels"].append(folder_label)
            if rubric:
                exemplar_dbs[base_name]["rubrics"].append(rubric)

        logger.debug(f"Discovered {len(exemplar_dbs)} exemplar database groups")
        return exemplar_dbs

    def _exemplar_label(self, path: Path) -> str:
        """
        Extract label from exemplar database path.

        Mirrors the logic in db_variant_selector_helpers.exemplar_label():
        - catalog/{db_name}/{file} → db_name
        - originals/{file} → file stem
        """
        try:
            rel = path.relative_to(self.exemplar_db_dir)
            parts = rel.parts
            if not parts:
                return path.stem

            # Handle catalog/ and originals/ subdirectories
            first = parts[0] if len(parts) > 0 else None
            second = parts[1] if len(parts) > 1 else None

            if first == "catalog" and second:
                # catalog/{db_name}/{file.db} → use db_name (folder)
                return second
            if first == "originals" and second:
                # originals/{file.db} → use filename stem
                filename = second
                return filename.replace(".sqlite", "").replace(".db", "")

            # Fallback: use first directory or filename stem
            return first if first else path.stem
        except ValueError:
            # Path not under exemplar_db_dir
            return path.stem

    def _extract_category(self, db_path: Path) -> str:
        """Extract category from database path."""
        try:
            rel = db_path.relative_to(self.exemplar_db_dir)
            parts = rel.parts

            # For catalog/{db_name}/... the category is db_name (or parent if nested)
            if len(parts) >= 2 and parts[0] == "catalog":
                return parts[1]
            if len(parts) >= 1 and parts[0] == "originals":
                return "originals"

            return db_path.parent.name if db_path.parent != self.exemplar_db_dir else "unknown"
        except ValueError:
            return "unknown"

    def _load_rubric(self, db_name: str) -> dict | None:
        """Load rubric for a database if available."""
        if not self.rubrics_dir or not self.rubrics_dir.exists():
            return None

        # Rubric structure: schemas/{db_name}/{db_name}.rubric.json
        # or schemas/{db_name}/{db_name}_combined.rubric.json
        rubric_candidates = [
            self.rubrics_dir / db_name / f"{db_name}_combined.rubric.json",
            self.rubrics_dir / db_name / f"{db_name}.rubric.json",
        ]

        for rubric_path in rubric_candidates:
            if rubric_path.exists():
                try:
                    with Path.open(rubric_path) as f:
                        return json.load(f)
                except (json.JSONDecodeError, OSError):
                    continue

        return None

    def _select_best_rubric(self, rubrics: list[dict]) -> dict | None:
        """
        Select the best rubric from a list, preferring those with known timestamp formats.

        When multiple user variants are conglomerated (e.g., _username, _user_empty),
        we may have multiple rubrics. Prefer rubrics with defined timestamp formats
        over those with "unknown" (which happens when exemplar was empty).

        Args:
            rubrics: List of rubric dicts

        Returns:
            Best rubric, or None if empty list
        """
        if not rubrics:
            return None

        if len(rubrics) == 1:
            return rubrics[0]

        # Score rubrics: prefer those with known timestamp formats
        def rubric_score(rubric: dict) -> int:
            """Higher score = better rubric."""
            score = 0
            tables = rubric.get("tables", {})
            for table_meta in tables.values():
                if not isinstance(table_meta, dict):
                    continue
                columns = table_meta.get("columns", [])
                if isinstance(columns, list):
                    for col in columns:
                        if isinstance(col, dict) and col.get("role") == "timestamp":
                            ts_format = col.get("timestamp_format", "unknown")
                            # Known formats get +10, unknown gets 0
                            if ts_format and ts_format != "unknown":
                                score += 10
                elif isinstance(columns, dict):
                    for col_meta in columns.values():
                        if isinstance(col_meta, dict) and col_meta.get("role") == "timestamp":
                            ts_format = col_meta.get("timestamp_format", "unknown")
                            if ts_format and ts_format != "unknown":
                                score += 10
            return score

        # Return rubric with highest score
        return max(rubrics, key=rubric_score)

    def _resolve_candidate_path(self, variant_path: str) -> Path | None:
        """
        Resolve candidate database path.

        The variant_path from sqlite_scan_results.jsonl can be:
        - Absolute path to the scanned file
        - Relative path from the run directory
        """
        path = Path(variant_path)

        # Try as-is (absolute path)
        if path.exists():
            return path

        # Get run directory (parent of candidate_matched_dir)
        run_dir = self.candidate_matched_dir.parent

        # Try relative to run directory
        candidate_path = run_dir / variant_path
        if candidate_path.exists():
            return candidate_path

        # Try just the filename in various locations
        filename = path.name

        # Try in databases/selected_variants/
        selected_variants_dir = run_dir / "databases" / "selected_variants"
        if selected_variants_dir.exists():
            # Search recursively for the file
            matches = list(selected_variants_dir.rglob(filename))
            if matches:
                return matches[0]

        # Try in databases/catalog/
        catalog_dir = run_dir / "databases" / "catalog"
        if catalog_dir.exists():
            matches = list(catalog_dir.rglob(filename))
            if matches:
                return matches[0]

        # Try in matched directory
        matched_path = self.candidate_matched_dir / filename
        if matched_path.exists():
            return matched_path

        logger.debug(f"Could not resolve candidate path: {variant_path}")
        return None

    def _compare_databases(
        self,
        name: str,
        category: str,
        exemplar_path: Path | None,
        candidate_path: Path | None,
    ) -> DatabaseComparison:
        """
        Compare a single database pair.

        Args:
            name: Database name
            category: Database category
            exemplar_path: Path to exemplar database
            candidate_path: Path to candidate database

        Returns:
            DatabaseComparison with per-table metrics
        """
        comparison = DatabaseComparison(
            name=name,
            category=category,
            exemplar_path=exemplar_path,
            candidate_path=candidate_path,
            matched=exemplar_path is not None and candidate_path is not None,
        )

        if not exemplar_path or not exemplar_path.exists():
            comparison.error = "Exemplar database not found"
            return comparison

        if not candidate_path or not candidate_path.exists():
            comparison.error = "Candidate database not found"
            return comparison

        try:
            # Get common tables
            with (
                readonly_connection(exemplar_path) as ex_conn,
                readonly_connection(candidate_path) as cand_conn,
            ):
                ex_tables = self._get_tables(ex_conn)
                cand_tables = self._get_tables(cand_conn)
                common_tables = set(ex_tables) & set(cand_tables)

                # Collect lost_and_found stats from main DB
                lf_stats = self._get_lost_and_found_stats(cand_conn)

                # Also check for companion L&F file (separate *_lost_and_found.sqlite)
                companion_lf_stats = self._get_companion_lf_stats(candidate_path)
                if companion_lf_stats.total_rows > 0:
                    # Merge companion L&F stats
                    lf_stats.total_rows += companion_lf_stats.total_rows
                    lf_stats.table_count += companion_lf_stats.table_count
                    lf_stats.tables.extend(companion_lf_stats.tables)

                comparison.lost_and_found = lf_stats

                if not common_tables:
                    # Check if candidate only has lost_and_found tables (rebuilt DB)
                    if lf_stats.total_rows > 0:
                        # Rebuilt database - L&F data IS the recovery
                        comparison.rebuilt_from_lf = True
                        logger.debug(
                            f"Rebuilt DB {name}: {lf_stats.total_rows} L&F rows from {lf_stats.table_count} tables"
                        )
                        # Return successfully - L&F rows are the recovered data
                        return comparison

                    # No common tables AND no L&F data - actual mismatch
                    logger.warning(
                        f"No common tables for {name}: exemplar={ex_tables[:5]}, candidate={cand_tables[:5]}"
                    )
                    comparison.error = (
                        f"No common tables: exemplar has {len(ex_tables)}, candidate has {len(cand_tables)}"
                    )
                    return comparison

                # Load rubric for timestamp info
                rubric = self._load_rubric(name)

                # Compare each common table
                for table_name in sorted(common_tables):
                    table_comparison = self._compare_table(
                        ex_conn,
                        cand_conn,
                        table_name,
                        rubric,
                    )
                    comparison.tables.append(table_comparison)

        except sqlite3.Error as e:
            comparison.error = f"SQLite error: {e}"
            logger.debug(f"SQLite error comparing {name}: {e}")
        except Exception as e:
            comparison.error = f"Error: {e}"
            logger.debug(f"Error comparing {name}: {e}")

        return comparison

    def _get_tables(self, conn: sqlite3.Connection) -> list[str]:
        """Get list of user tables in database."""
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' AND name NOT LIKE 'lost_and_found%'"
        )
        return [row[0] for row in cur.fetchall()]

    def _get_lost_and_found_stats(self, conn: sqlite3.Connection) -> LostAndFoundStats:
        """
        Get statistics for lost_and_found tables in database.

        These tables contain rows recovered/reconstituted from deleted data.
        """
        stats = LostAndFoundStats()

        try:
            cur = conn.cursor()
            # Find all lost_and_found tables
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'lost_and_found%'")
            lf_tables = [row[0] for row in cur.fetchall()]

            stats.tables = lf_tables
            stats.table_count = len(lf_tables)

            # Count rows in each table
            for table in lf_tables:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {quote_ident(table)}")
                    count = cur.fetchone()[0]
                    stats.total_rows += count
                except sqlite3.Error:
                    continue

        except sqlite3.Error:
            pass

        return stats

    def _get_companion_lf_stats(self, candidate_path: Path | None) -> LostAndFoundStats:
        """
        Find and read companion lost_and_found file.

        Many candidate databases have a separate *_lost_and_found.sqlite file
        containing recovered/reconstituted rows.

        Args:
            candidate_path: Path to main candidate database

        Returns:
            LostAndFoundStats from companion file (empty if not found)
        """
        stats = LostAndFoundStats()

        if not candidate_path:
            return stats

        # Try to find companion L&F file
        # Pattern: same stem + "_lost_and_found.sqlite" in same or parent folder
        stem = candidate_path.stem  # e.g., "f431521576"
        parent = candidate_path.parent

        # Search patterns for L&F companion files
        lf_candidates = [
            parent / f"{stem}_lost_and_found.sqlite",  # Same folder
            parent / f"{stem}_lost_and_found.db",
        ]

        # Also search in parent folder (common in variant structure)
        if parent.parent:
            lf_candidates.extend(
                [
                    parent.parent / f"{stem}_lost_and_found.sqlite",
                    parent.parent / f"{stem}_lost_and_found.db",
                ]
            )

        # Also search recursively in the selected_variants folder
        run_dir = self.candidate_matched_dir.parent
        selected_variants = run_dir / "databases" / "selected_variants"
        if selected_variants.exists():
            matches = list(selected_variants.rglob(f"{stem}_lost_and_found.*"))
            lf_candidates.extend(matches)

        for lf_path in lf_candidates:
            if lf_path.exists():
                try:
                    with readonly_connection(lf_path) as conn:
                        return self._get_lost_and_found_stats(conn)
                except Exception as e:
                    logger.debug(f"Error reading L&F file {lf_path}: {e}")
                    continue

        return stats

    def _compare_table(
        self,
        ex_conn: sqlite3.Connection,
        cand_conn: sqlite3.Connection,
        table_name: str,
        rubric: dict | None,
    ) -> TableComparison:
        """
        Compare a single table between exemplar and candidate.

        Uses primary key or rowid for deduplication.
        """
        qname = quote_ident(table_name)

        # Get row counts
        ex_count = self._count_rows(ex_conn, qname)
        # Get candidate row counts split by source (carved vs L&F)
        carved_count, lf_count = self._count_rows_by_source(cand_conn, table_name)

        # Always use content hash comparison, excluding PK/ID columns
        # PK comparison is unreliable because recovered data gets new PKs
        unique_to_candidate, overlap, unique_to_exemplar = self._compare_by_hash(ex_conn, cand_conn, table_name)

        # Get timestamp info from rubric
        ts_col, ts_format = self._get_timestamp_column(table_name, rubric)

        # Calculate date ranges if timestamp column exists
        ex_range = None
        cand_range = None
        extended_days = 0
        extended_after_days = 0
        unique_cand_days = 0
        ex_dates: list[str] = []
        cand_dates: list[str] = []

        if ts_col:
            ex_range = self._get_date_range(ex_conn, qname, ts_col, ts_format)
            cand_range = self._get_date_range(cand_conn, qname, ts_col, ts_format)

            # Get unique dates for gap visualization
            ex_dates = self._get_unique_dates(ex_conn, qname, ts_col, ts_format)
            cand_dates = self._get_unique_dates(cand_conn, qname, ts_col, ts_format)

            # Calculate unique candidate days (dates with data not in exemplar)
            # This includes both range extension AND gap-filling
            unique_cand_days = len(set(cand_dates) - set(ex_dates))

            # Calculate timeline extension (before exemplar start) - legacy
            if ex_range and cand_range and cand_range[0] < ex_range[0]:
                extended_days = (ex_range[0] - cand_range[0]).days

            # Calculate timeline extension (after exemplar end) - legacy
            if ex_range and cand_range and cand_range[1] > ex_range[1]:
                extended_after_days = (cand_range[1] - ex_range[1]).days

        return TableComparison(
            name=table_name,
            exemplar_rows=ex_count,
            candidate_rows=carved_count,  # Carved/intact rows only (via data_source)
            lf_rows=lf_count,  # L&F rows (data_source LIKE 'found_%')
            unique_to_candidate=unique_to_candidate,  # Total unique recovered (carved + L&F)
            overlap=overlap,
            unique_to_exemplar=unique_to_exemplar,
            timestamp_column=ts_col,
            timestamp_format=ts_format,
            exemplar_date_range=ex_range,
            candidate_date_range=cand_range,
            timeline_extended_days=extended_days,
            timeline_extended_after_days=extended_after_days,
            exemplar_dates=ex_dates,
            candidate_dates=cand_dates,
            unique_candidate_days=unique_cand_days,
        )

    def _count_rows(self, conn: sqlite3.Connection, qname: str) -> int:
        """Count rows in a table."""
        try:
            cur = conn.cursor()
            cur.execute(f"SELECT COUNT(*) FROM {qname}")
            return cur.fetchone()[0]
        except sqlite3.Error:
            return 0

    def _get_primary_key(self, conn: sqlite3.Connection, table_name: str) -> str | None:
        """Get primary key column name for a table."""
        try:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({quote_ident(table_name)})")
            for row in cur.fetchall():
                # row: (cid, name, type, notnull, dflt_value, pk)
                if row[5]:  # pk flag is non-zero
                    return row[1]
        except sqlite3.Error:
            pass
        return None

    def _get_hashable_columns(self, conn: sqlite3.Connection, table_name: str) -> list[str]:
        """
        Get list of columns suitable for content hashing (excludes PK/ID columns).

        Args:
            conn: Database connection
            table_name: Name of table

        Returns:
            List of column names (excluding PK and common ID columns)
        """
        qname = quote_ident(table_name)
        try:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({qname})")
            cols_info = cur.fetchall()

            id_column_names = {"id", "rowid", "z_pk", "z_ent", "z_opt"}
            result = []

            for col in cols_info:
                _, name, _, _, _, pk = col[0], col[1], col[2], col[3], col[4], col[5]
                if pk == 0 and name.lower() not in id_column_names:
                    result.append(name)

            return result
        except sqlite3.Error:
            return []

    def _get_row_hashes(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        columns: list[str] | None = None,
    ) -> set[int]:
        """
        Get set of content hashes for all rows in a table.

        Args:
            conn: Database connection
            table_name: Name of table to hash
            columns: Explicit list of columns to hash (in order). If None,
                    auto-detects columns excluding PK/ID columns.

        Returns:
            Set of integer hashes representing unique row content
        """
        qname = quote_ident(table_name)

        try:
            cur = conn.cursor()

            if columns is not None:
                # Use explicitly provided columns (already sorted/ordered)
                hash_columns = [quote_ident(c) for c in columns]
            else:
                # Auto-detect columns, excluding PK/ID columns
                cur.execute(f"PRAGMA table_info({qname})")
                cols_info = cur.fetchall()

                # Identify columns to exclude from hash:
                # - Primary key columns (pk > 0)
                # - Common ID column names that are auto-generated
                id_column_names = {"id", "rowid", "z_pk", "z_ent", "z_opt"}
                exclude_indices = set()
                all_columns = []

                for col in cols_info:
                    cid, name, _, _, _, pk = col[0], col[1], col[2], col[3], col[4], col[5]
                    all_columns.append(name)
                    if pk > 0 or name.lower() in id_column_names:
                        exclude_indices.add(cid)

                # Build column list excluding PK/ID columns
                hash_columns = [quote_ident(name) for i, name in enumerate(all_columns) if i not in exclude_indices]

            if not hash_columns:
                # Table only has PK/ID columns
                return set()

            col_list = ", ".join(hash_columns)

            cur.execute(f"SELECT {col_list} FROM {qname}")
            return {hash(tuple(row)) for row in cur.fetchall()}

        except sqlite3.Error:
            return set()

    def _count_rows_by_source(
        self,
        conn: sqlite3.Connection,
        table_name: str,
    ) -> tuple[int, int]:
        """Count rows by data_source column (carved/intact vs L&F).

        Args:
            conn: Database connection
            table_name: Table to count

        Returns:
            (carved_count, lf_count) - carved includes 'carved_*' and 'intact',
            lf includes 'found_*' sources
        """
        qname = quote_ident(table_name)
        try:
            cur = conn.cursor()

            # Check if data_source column exists
            cur.execute(f"PRAGMA table_info({qname})")
            columns = [row[1].lower() for row in cur.fetchall()]

            if "data_source" not in columns:
                # No provenance tracking - all rows are "carved"
                cur.execute(f"SELECT COUNT(*) FROM {qname}")
                return cur.fetchone()[0], 0

            # Count by source type
            cur.execute(f"""
                SELECT
                    SUM(CASE WHEN data_source LIKE 'found_%' THEN 1 ELSE 0 END) as lf_count,
                    SUM(CASE WHEN data_source NOT LIKE 'found_%' OR data_source IS NULL THEN 1 ELSE 0 END) as carved_count
                FROM {qname}
            """)
            row = cur.fetchone()
            return row[1] or 0, row[0] or 0

        except sqlite3.Error:
            return 0, 0

    def _get_row_hashes_by_source(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        columns: list[str] | None = None,
        lf_only: bool = False,
    ) -> set[int]:
        """Get row hashes filtered by data_source.

        Args:
            conn: Database connection
            table_name: Table to hash
            columns: Columns to include in hash (None = auto-detect)
            lf_only: If True, only return hashes for 'found_*' rows

        Returns:
            Set of row hashes for the specified source type
        """
        qname = quote_ident(table_name)
        try:
            cur = conn.cursor()

            # Check if data_source column exists
            cur.execute(f"PRAGMA table_info({qname})")
            col_info = cur.fetchall()
            col_names = [row[1].lower() for row in col_info]

            if "data_source" not in col_names:
                # No provenance - return empty for lf_only, full set otherwise
                if lf_only:
                    return set()
                return self._get_row_hashes(conn, table_name, columns)

            # Build column list for hashing (same logic as _get_row_hashes)
            if columns is not None:
                hash_columns = [quote_ident(c) for c in columns]
            else:
                id_column_names = {"id", "rowid", "z_pk", "z_ent", "z_opt", "data_source"}
                exclude_indices = set()
                all_columns = []

                for col in col_info:
                    cid, name, _, _, _, pk = col[0], col[1], col[2], col[3], col[4], col[5]
                    all_columns.append(name)
                    if pk > 0 or name.lower() in id_column_names:
                        exclude_indices.add(cid)

                hash_columns = [quote_ident(name) for i, name in enumerate(all_columns) if i not in exclude_indices]

            if not hash_columns:
                return set()

            col_list = ", ".join(hash_columns)

            # Filter by source type
            if lf_only:
                cur.execute(f"SELECT {col_list} FROM {qname} WHERE data_source LIKE 'found_%'")
            else:
                cur.execute(
                    f"SELECT {col_list} FROM {qname} WHERE data_source NOT LIKE 'found_%' OR data_source IS NULL"
                )

            return {hash(tuple(row)) for row in cur.fetchall()}

        except sqlite3.Error:
            return set()

    def _compare_by_hash(
        self,
        ex_conn: sqlite3.Connection,
        cand_conn: sqlite3.Connection,
        table_name: str,
    ) -> tuple[int, int, int]:
        """
        Compare tables by hashing row content, excluding PK/ID columns.

        Excludes primary key and common ID columns from hash because recovered
        data often has different IDs for the same content.

        Returns:
            (unique_to_candidate, overlap, unique_to_exemplar)
        """
        ex_hashes = self._get_row_hashes(ex_conn, table_name)
        cand_hashes = self._get_row_hashes(cand_conn, table_name)

        if not ex_hashes and not cand_hashes:
            return 0, 0, 0

        # Calculate set differences
        unique_to_candidate = len(cand_hashes - ex_hashes)
        overlap = len(cand_hashes & ex_hashes)
        unique_to_exemplar = len(ex_hashes - cand_hashes)

        return unique_to_candidate, overlap, unique_to_exemplar

    def _get_timestamp_column(self, table_name: str, rubric: dict | None) -> tuple[str | None, str | None]:
        """
        Get timestamp column and format from rubric metadata.

        Returns:
            (column_name, timestamp_format) or (None, None)
        """
        if not rubric:
            return None, None

        tables = rubric.get("tables", {})
        table_meta = tables.get(table_name, {})
        columns = table_meta.get("columns", {})

        # Handle columns as either dict or list format
        if isinstance(columns, dict):
            col_items = columns.items()
        elif isinstance(columns, list):
            # List format: [{"name": "col1", "role": "..."}, ...]
            col_items = [(c.get("name"), c) for c in columns if isinstance(c, dict)]
        else:
            return None, None

        for col_name, col_meta in col_items:
            if not col_name or not isinstance(col_meta, dict):
                continue
            role = col_meta.get("role")
            # Handle both string and list roles
            if isinstance(role, str):
                roles = [role]
            elif isinstance(role, list):
                roles = role
            else:
                roles = []

            if "timestamp" in roles:
                ts_format = col_meta.get("timestamp_format", "unix_seconds")
                return col_name, ts_format

        return None, None

    def _get_date_range(
        self,
        conn: sqlite3.Connection,
        qname: str,
        ts_col: str,
        ts_format: str | None,
    ) -> tuple[datetime, datetime] | None:
        """
        Get min/max dates from a timestamp column.

        Returns:
            (min_date, max_date) or None if no valid timestamps
        """
        qts = quote_ident(ts_col)

        try:
            cur = conn.cursor()
            cur.execute(f"SELECT MIN({qts}), MAX({qts}) FROM {qname} WHERE {qts} IS NOT NULL")
            row = cur.fetchone()

            if not row or row[0] is None:
                return None

            min_val, max_val = row

            # Convert to datetime using rubric format
            min_dt = self._timestamp_to_datetime(min_val, ts_format)
            max_dt = self._timestamp_to_datetime(max_val, ts_format)

            # Filter out sentinel dates (epoch zeros, distantPast/Future)
            if min_dt and _is_sentinel_date(min_dt.strftime("%Y-%m-%d")):
                min_dt = None
            if max_dt and _is_sentinel_date(max_dt.strftime("%Y-%m-%d")):
                max_dt = None

            if min_dt and max_dt:
                return (min_dt, max_dt)

        except sqlite3.Error:
            pass

        return None

    def _get_unique_dates(
        self,
        conn: sqlite3.Connection,
        qname: str,
        ts_col: str,
        ts_format: str | None,
    ) -> list[str]:
        """
        Get unique dates (YYYY-MM-DD) from a timestamp column for gap visualization.

        Returns:
            List of date strings in YYYY-MM-DD format, sorted ascending
        """
        qts = quote_ident(ts_col)

        try:
            cur = conn.cursor()
            # Get distinct timestamp values
            cur.execute(f"SELECT DISTINCT {qts} FROM {qname} WHERE {qts} IS NOT NULL ORDER BY {qts}")

            unique_dates: set[str] = set()
            for (ts_val,) in cur.fetchall():
                dt = self._timestamp_to_datetime(ts_val, ts_format)
                if dt:
                    # Format as YYYY-MM-DD
                    date_str = dt.strftime("%Y-%m-%d")
                    # Filter out sentinel dates (epoch zeros, distantPast/Future)
                    if not _is_sentinel_date(date_str):
                        unique_dates.add(date_str)

            return sorted(unique_dates)

        except sqlite3.Error:
            pass

        return []

    def _timestamp_to_datetime(self, value: Any, ts_format: str | None) -> datetime | None:
        """Convert timestamp value to datetime using format hint."""
        try:
            val = float(value)
            unix_seconds = TimestampFormat._to_unix_seconds(val, ts_format)
            if unix_seconds is None:
                return None
            return datetime.fromtimestamp(unix_seconds, tz=UTC)
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    def _calculate_category_breakdown(self, databases: list[DatabaseComparison]) -> dict[str, dict[str, Any]]:
        """Calculate statistics by category."""
        by_category: dict[str, dict[str, Any]] = {}

        for db in databases:
            cat = db.category
            if cat not in by_category:
                by_category[cat] = {
                    "matched": 0,
                    "with_new_data": 0,
                    "unique_rows": 0,
                    "overlap_rows": 0,
                    "timeline_extensions": 0,
                }

            by_category[cat]["matched"] += 1
            if db.total_unique_recovered > 0:
                by_category[cat]["with_new_data"] += 1
            by_category[cat]["unique_rows"] += db.total_unique_recovered
            by_category[cat]["overlap_rows"] += db.total_overlap
            if db.has_timeline_extension:
                by_category[cat]["timeline_extensions"] += 1

        return by_category
