#!/usr/bin/env python3

"""
Exemplar Scanner - Phase 1 of Mac Log Sleuth Pipeline
by WarpedWing Labs

Scans a reference macOS system (exemplar) to locate and catalog databases.
Generates schemas and rubrics from found databases to create "ground truth"
fingerprints for later forensic recovery operations.

This integrates with existing tools:
  - generate_powerlog_schema_rubric.py: Schema/rubric generation
  - rubric_loader.py: Rubric loading and token indexing
  - schema_manager.py: Schema management and comparison
  - output_structure.py: Organized output directory structure

Usage:
    python exemplar_scanner.py --source /Volumes/MacintoshHD --case-name "MacBookPro_Exemplar"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

# Import existing tools
try:
    from mac_log_sleuth.generate_powerlog_schema_rubric import (
        fetch_tables,
        fetch_columns,
        sample_table_info,
        infer_examples,
        is_epoch_column,
        EPOCH_MIN_2010,
        EPOCH_MAX_2100,
    )
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from mac_log_sleuth.generate_powerlog_schema_rubric import (
        fetch_tables,
        fetch_columns,
        sample_table_info,
        infer_examples,
        is_epoch_column,
        EPOCH_MIN_2010,
        EPOCH_MAX_2100,
    )

try:
    from mac_log_sleuth.pipeline.output_structure import OutputStructure
except ImportError:
    from output_structure import OutputStructure


DEFAULT_CATALOG_PATH = Path(__file__).parent.parent / "catalog" / "database_catalog.yaml"


class ExemplarScanner:
    """
    Scans a reference macOS system to locate databases and generate schemas/rubrics.
    """

    def __init__(
        self,
        source_path: Path,
        output_structure: OutputStructure,
        catalog_path: Path | None = None,
    ):
        """
        Initialize the exemplar scanner.

        Args:
            source_path: Root path of the exemplar system (e.g., /Volumes/MacintoshHD)
            output_structure: OutputStructure instance for organized output
            catalog_path: Path to database catalog YAML (defaults to built-in catalog)
        """
        self.source_path = source_path
        self.output = output_structure
        self.catalog_path = catalog_path or DEFAULT_CATALOG_PATH

        # Load database catalog
        self.catalog = self._load_catalog()

        # Tracking
        self.found_databases: list[dict[str, Any]] = []
        self.failed_databases: list[dict[str, Any]] = []
        self.generated_schemas: list[Path] = []
        self.generated_rubrics: list[Path] = []

    def _load_catalog(self) -> dict[str, Any]:
        """Load the database catalog YAML file."""
        if not self.catalog_path.exists():
            print(
                f"Warning: Catalog not found at {self.catalog_path}, using empty catalog"
            )
            return {}

        with open(self.catalog_path) as f:
            return yaml.safe_load(f)

    def scan(
        self,
        categories: list[str] | None = None,
        priorities: list[str] | None = None,
        custom_paths: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Scan the exemplar system for databases.

        Args:
            categories: Filter by categories (e.g., ['browser', 'security'])
            priorities: Filter by priority levels (e.g., ['critical', 'high'])
            custom_paths: Additional custom paths to scan (glob patterns)

        Returns:
            Summary dictionary with scan results
        """
        print(f"\n{'='*80}")
        print(f"Mac Log Sleuth - Exemplar Scanner")
        print(f"{'='*80}")
        print(f"Source: {self.source_path}")
        print(f"Output: {self.output.root}")
        print(f"Catalog: {self.catalog_path}")
        print(f"{'='*80}\n")

        # Scan catalog databases
        for category_name, databases in self.catalog.items():
            if category_name == "catalog_metadata":
                continue

            if not isinstance(databases, list):
                continue

            for db_def in databases:
                # Apply filters
                if categories and db_def.get("category") not in categories:
                    continue

                if priorities and db_def.get("priority") not in priorities:
                    continue

                self._scan_database_definition(db_def, category_name)

        # Scan custom paths
        if custom_paths:
            for custom_glob in custom_paths:
                self._scan_custom_path(custom_glob)

        # Generate summary
        summary = {
            "scan_completed": datetime.now(UTC).isoformat(),
            "source_path": str(self.source_path),
            "total_found": len(self.found_databases),
            "total_failed": len(self.failed_databases),
            "schemas_generated": len(self.generated_schemas),
            "rubrics_generated": len(self.generated_rubrics),
            "found_databases": self.found_databases,
            "failed_databases": self.failed_databases,
        }

        # Save summary
        summary_path = self.output.get_report_path("exemplar_scan_summary")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Scan Summary:")
        print(f"  Databases found: {len(self.found_databases)}")
        print(f"  Databases failed: {len(self.failed_databases)}")
        print(f"  Schemas generated: {len(self.generated_schemas)}")
        print(f"  Rubrics generated: {len(self.generated_rubrics)}")
        print(f"{'='*80}\n")

        return summary

    def _scan_database_definition(self, db_def: dict[str, Any], category: str):
        """Scan for a database based on catalog definition."""
        db_name = db_def.get("name", "Unknown")
        glob_pattern = db_def.get("glob_pattern")

        if not glob_pattern:
            # No glob pattern, try direct path
            direct_path = db_def.get("path")
            if direct_path:
                # Expand wildcards in direct path
                glob_pattern = direct_path

        if not glob_pattern:
            print(f"[WARNING]  Skipping {db_name}: No path or glob pattern defined")
            return

        print(f"[SCAN] Scanning for: {db_name}")
        print(f"   Pattern: {glob_pattern}")

        # Search for databases matching the pattern
        found_paths = list(self.source_path.glob(glob_pattern))

        if not found_paths:
            print(f"   [ERROR] Not found")
            return

        for db_path in found_paths:
            if not db_path.is_file():
                continue

            print(f"   [OK] Found: {db_path.relative_to(self.source_path)}")

            try:
                self._process_database(db_path, db_def, category)
            except Exception as e:
                print(f"   [WARNING]  Failed to process: {e}")
                self.failed_databases.append(
                    {
                        "name": db_name,
                        "path": str(db_path),
                        "category": category,
                        "error": str(e),
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

    def _scan_custom_path(self, glob_pattern: str):
        """Scan a custom path pattern not in the catalog."""
        print(f"[SCAN] Scanning custom pattern: {glob_pattern}")

        found_paths = list(self.source_path.glob(glob_pattern))

        if not found_paths:
            print(f"   [ERROR] Not found")
            return

        for db_path in found_paths:
            if not db_path.is_file():
                continue

            print(f"   [OK] Found: {db_path.relative_to(self.source_path)}")

            db_def = {
                "name": db_path.name,
                "path": str(db_path.relative_to(self.source_path)),
                "description": f"Custom database: {db_path.name}",
                "priority": "custom",
                "category": "custom",
            }

            try:
                self._process_database(db_path, db_def, "custom")
            except Exception as e:
                print(f"   [WARNING]  Failed to process: {e}")
                self.failed_databases.append(
                    {
                        "name": db_path.name,
                        "path": str(db_path),
                        "category": "custom",
                        "error": str(e),
                        "timestamp": datetime.now(UTC).isoformat(),
                    }
                )

    def _process_database(self, db_path: Path, db_def: dict[str, Any], category: str):
        """
        Process a found database: copy, hash, generate schema/rubric.

        Args:
            db_path: Path to the database file
            db_def: Database definition from catalog
            category: Category name
        """
        db_name = db_def.get("name", db_path.name)
        original_filename = db_path.name

        # Compute hash
        md5_hash = self._compute_md5(db_path)

        # Copy to originals directory
        original_copy = self.output.copy_to_originals(db_path, original_filename)

        # Create provenance file
        provenance_data = {
            "name": db_name,
            "category": category,
            "source_path": str(db_path),
            "relative_path": str(db_path.relative_to(self.source_path)),
            "md5": md5_hash,
            "file_size": db_path.stat().st_size,
            "modified_time": datetime.fromtimestamp(
                db_path.stat().st_mtime, UTC
            ).isoformat(),
            "description": db_def.get("description", ""),
            "priority": db_def.get("priority", "unknown"),
            "application": db_def.get("application"),
            "versions": db_def.get("versions"),
            "notes": db_def.get("notes"),
        }

        self.output.create_provenance_file(
            original_copy, original_filename, provenance_data
        )

        # Check if it's a SQLite database
        if not self._is_sqlite_database(db_path):
            print(f"   [INFO]  Not a SQLite database (or unsupported format)")
            self.found_databases.append(
                {
                    **provenance_data,
                    "is_sqlite": False,
                    "schema_generated": False,
                    "rubric_generated": False,
                }
            )
            return

        # Generate schema and rubric
        schema_dir = self.output.get_schema_dir(original_filename)

        try:
            schema_path, rubric_path = self._generate_schema_and_rubric(
                original_copy, schema_dir, original_filename
            )

            self.generated_schemas.append(schema_path)
            self.generated_rubrics.append(rubric_path)

            print(f"   [SCHEMA] Schema: {schema_path.name}")
            print(f"   [REPORT] Rubric: {rubric_path.name}")

            self.found_databases.append(
                {
                    **provenance_data,
                    "is_sqlite": True,
                    "schema_generated": True,
                    "rubric_generated": True,
                    "schema_path": str(schema_path),
                    "rubric_path": str(rubric_path),
                }
            )

        except Exception as e:
            print(f"   [WARNING]  Schema/rubric generation failed: {e}")
            self.found_databases.append(
                {
                    **provenance_data,
                    "is_sqlite": True,
                    "schema_generated": False,
                    "rubric_generated": False,
                    "generation_error": str(e),
                }
            )

    def _compute_md5(self, file_path: Path) -> str:
        """Compute MD5 hash of a file."""
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _is_sqlite_database(self, db_path: Path) -> bool:
        """Check if a file is a SQLite database."""
        try:
            # Check magic bytes
            with open(db_path, "rb") as f:
                header = f.read(16)
                if not header.startswith(b"SQLite format 3\x00"):
                    return False

            # Try to open it
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            conn.close()
            return True
        except Exception:
            return False

    def _generate_schema_and_rubric(
        self, db_path: Path, output_dir: Path, base_name: str
    ) -> tuple[Path, Path]:
        """
        Generate schema CSV and rubric JSON for a database.

        Integrates with generate_powerlog_schema_rubric.py logic.

        Args:
            db_path: Path to database file
            output_dir: Output directory for schema/rubric
            base_name: Base name for output files

        Returns:
            Tuple of (schema_path, rubric_path)
        """
        # Use timestamp for uniqueness
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        schema_name = Path(base_name).stem

        schema_path = output_dir / f"{schema_name}.schema.csv"
        rubric_path = output_dir / f"{schema_name}.rubric.json"

        # Open database read-only
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

        try:
            # Fetch tables
            tables = fetch_tables(conn)

            # Generate schema CSV
            import csv

            with schema_path.open("w", newline="", encoding="utf-8") as csvf:
                writer = csv.writer(csvf)
                writer.writerow(["Table", "Column", "Type"])

                for tname, _sql in tables:
                    cols = fetch_columns(conn, tname)
                    if not cols:
                        continue
                    for _cid, col_name, col_type, _notnull, _dflt, _pk in cols:
                        writer.writerow([tname, col_name, col_type or ""])

            # Generate rubric JSON
            rubric = {
                "generated_at_utc": datetime.now(UTC).isoformat() + "Z",
                "source_db": str(db_path),
                "epoch_bounds": {"min": EPOCH_MIN_2010, "max": EPOCH_MAX_2100},
                "notes": [
                    "Schema and rubric generated from exemplar system",
                    "Portable across hosts; ranges are hints, not constraints",
                    "Columns with 'timestamp' or 'date' are assumed to be Unix epoch",
                ],
                "tables": {},
            }

            for tname, _sql in tables:
                cols = fetch_columns(conn, tname)
                if not cols:
                    continue

                # Sample data for examples
                sample_rows = sample_table_info(conn, tname, limit=200)
                sample_count = len(sample_rows)

                # Count non-nulls
                non_null_counts = [0] * len(cols)
                if sample_count:
                    for row in sample_rows:
                        for idx in range(min(len(row), len(cols))):
                            if row[idx] is not None:
                                non_null_counts[idx] += 1

                columns_list = []
                types_list = []
                columns_dict = {}

                for idx, (cid, col_name, col_type, notnull, dflt, pk) in enumerate(
                    cols
                ):
                    columns_list.append(col_name)
                    types_list.append(col_type or "TEXT")

                    declared_notnull = bool(notnull)
                    observed_non_null = non_null_counts[idx] if sample_count else 0
                    observed_fill_ratio = (
                        observed_non_null / sample_count if sample_count else None
                    )
                    observed_notnull = bool(
                        sample_count and observed_non_null == sample_count
                    )
                    effective_notnull = declared_notnull or observed_notnull

                    cinfo = {
                        "type": col_type or "TEXT",
                        "notnull": effective_notnull,
                        "declared_notnull": declared_notnull,
                        "observed_non_null": observed_non_null if sample_count else 0,
                        "observed_total": sample_count,
                        "primary_key": bool(pk),
                    }

                    if observed_fill_ratio is not None:
                        cinfo["observed_fill_ratio"] = round(observed_fill_ratio, 5)

                    # Detect roles
                    if col_name == "ID":
                        cinfo["role"] = "id"

                    if is_epoch_column(col_name):
                        cinfo["role"] = "epoch_timestamp"
                        cinfo["epoch_min"] = EPOCH_MIN_2010
                        cinfo["epoch_max"] = EPOCH_MAX_2100

                    # Examples
                    examples = infer_examples(sample_rows, idx)
                    if examples:
                        clean = []
                        for v in examples:
                            if isinstance(v, bytes):
                                try:
                                    clean.append(v.decode("utf-8", errors="replace"))
                                except Exception:
                                    clean.append(repr(v))
                            else:
                                clean.append(v)
                        cinfo["examples"] = clean
                    else:
                        cinfo["note"] = cinfo.get("note", "no sample data")

                    columns_dict[col_name] = cinfo

                rubric["tables"][tname] = {
                    "columns": columns_dict,
                    "column_list": columns_list,
                    "types": types_list,
                }

            # Write rubric JSON
            with rubric_path.open("w", encoding="utf-8") as jf:
                json.dump(rubric, jf, indent=2, ensure_ascii=False)

        finally:
            conn.close()

        return schema_path, rubric_path


def main():
    """Main entry point for exemplar scanner."""
    parser = argparse.ArgumentParser(
        description="Scan exemplar macOS system for databases and generate schemas/rubrics"
    )
    parser.add_argument(
        "--source",
        required=True,
        type=Path,
        help="Path to exemplar macOS system (e.g., /Volumes/MacintoshHD)",
    )
    parser.add_argument(
        "--case-name",
        default="Exemplar",
        help="Name for this exemplar case (default: Exemplar)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: ./MacLogSleuth_Output)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        help=f"Path to database catalog YAML (default: {DEFAULT_CATALOG_PATH})",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        help="Filter by categories (e.g., browser security system)",
    )
    parser.add_argument(
        "--priorities",
        nargs="+",
        help="Filter by priorities (e.g., critical high)",
    )
    parser.add_argument(
        "--custom-paths",
        nargs="+",
        help="Additional custom glob patterns to scan",
    )

    args = parser.parse_args()

    # Validate source path
    if not args.source.exists():
        print(f"Error: Source path does not exist: {args.source}", file=sys.stderr)
        sys.exit(1)

    if not args.source.is_dir():
        print(f"Error: Source path is not a directory: {args.source}", file=sys.stderr)
        sys.exit(1)

    # Create output structure
    output = OutputStructure(base_output_dir=args.output_dir, case_name=args.case_name)
    output.create()

    # Create scanner
    scanner = ExemplarScanner(
        source_path=args.source, output_structure=output, catalog_path=args.catalog
    )

    # Run scan
    try:
        summary = scanner.scan(
            categories=args.categories,
            priorities=args.priorities,
            custom_paths=args.custom_paths,
        )

        print(f"\n[OK] Exemplar scan completed successfully")
        print(f"[DIR] Output directory: {output.root}")
        print(
            f"[REPORT] Summary: {len(summary['found_databases'])} databases, "
            f"{len(summary['schemas_generated'])} schemas, "
            f"{len(summary['rubrics_generated'])} rubrics"
        )

    except KeyboardInterrupt:
        print("\n\n[WARNING]  Scan interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n[ERROR] Scan failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
