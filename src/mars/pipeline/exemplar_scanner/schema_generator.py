#!/usr/bin/env python3
"""Schema and rubric generation utilities for exemplar scanner.

Extracts schema fingerprinting, rubric generation, and hash lookup creation
from the main scanner to improve code organization.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from mars.config.schema import GLOBAL_IGNORABLE_TABLES
from mars.pipeline.matcher.generate_sqlite_schema_rubric import (
    fetch_columns,
    fetch_tables,
    generate_rubric,
)
from mars.utils.database_utils import readonly_connection
from mars.utils.debug_logger import logger


def get_table_names(db_path: Path) -> list[str]:
    """Get list of table names from a SQLite database.

    Args:
        db_path: Path to database file

    Returns:
        List of table names (excluding sqlite_ system tables)
    """
    with readonly_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;")
        tables = [row[0] for row in cursor.fetchall()]
        return tables


def get_schema_fingerprint(db_path: Path) -> str | None:
    """Generate schema fingerprint for database.

    Creates MD5 hash of (table_name, column_name, column_type) tuples.
    This detects schema differences even when table names match.

    Args:
        db_path: Path to database file

    Returns:
        MD5 hash of schema structure, or None if error
    """
    try:
        with readonly_connection(db_path) as conn:
            cursor = conn.cursor()

            # Get all regular tables (exclude sqlite_ system tables)
            cursor.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
            table_names = [row[0] for row in cursor.fetchall()]

            # Build schema structure
            tables = {}
            for table_name in table_names:
                # Use parameter binding for table name via string formatting with quotes
                # PRAGMA statements don't support parameter binding, so quote the identifier
                quoted_table = f'"{table_name}"'
                try:
                    cursor.execute(f"PRAGMA table_info({quoted_table})")
                    columns = [
                        (row[1], row[2].upper())  # (name, type)
                        for row in cursor.fetchall()
                    ]
                    tables[table_name] = columns
                except Exception:
                    # Skip virtual tables with missing modules (e.g., echo_document_map)
                    # These are custom SQLite extensions not available in standard SQLite
                    continue

            # Generate fingerprint hash
            schema_tuple = tuple((table_name, tuple(columns)) for table_name, columns in sorted(tables.items()))
            schema_json = json.dumps(schema_tuple, sort_keys=True)
            fingerprint = hashlib.md5(schema_json.encode()).hexdigest()

            return fingerprint

    except Exception:
        return None


def generate_schema_and_rubric(
    db_path: Path,
    output_dir: Path,
    base_name: str,
    min_timestamp_rows: int = 1,
) -> tuple[Path, Path]:
    """Generate schema CSV and rubric JSON for a database.

    Args:
        db_path: Path to database file
        output_dir: Output directory for schema/rubric
        base_name: Base name for output files
        min_timestamp_rows: Minimum timestamp values to assign role (default: 1)

    Returns:
        Tuple of (schema_path, rubric_path)
    """
    schema_name = Path(base_name).stem

    schema_path = output_dir / f"{schema_name}.schema.csv"
    rubric_path = output_dir / f"{schema_name}.rubric.json"

    # Open database read-only with immutable flag for mounted forensic images
    with readonly_connection(db_path) as conn:
        # Fetch tables
        tables = fetch_tables(conn)

        # Generate schema CSV
        with schema_path.open("w", newline="", encoding="utf-8") as csvf:
            writer = csv.writer(csvf)
            writer.writerow(["Table", "Column", "Type"])

            for tname, _sql in tables:
                cols = fetch_columns(conn, tname)
                if not cols:
                    continue
                for _cid, col_name, col_type, _notnull, _dflt, _pk in cols:
                    writer.writerow([tname, col_name, col_type or ""])

        # Generate rubric JSON using unified rubric generator
        # This includes all improvements: UUID detection, timestamp format,
        # signature patterns, FK inference, example confidence, etc.
        rubric = generate_rubric(conn, tables, rubric_name=schema_name, min_timestamp_rows=min_timestamp_rows)

        # Add source_db to metadata
        rubric["source_db"] = str(db_path)

        # Write rubric JSON
        with rubric_path.open("w", encoding="utf-8") as jf:
            json.dump(rubric, jf, indent=2, ensure_ascii=False)

    return schema_path, rubric_path


def generate_hash_lookup(
    schemas_dir: Path,
) -> None:
    """Generate hash lookup table from rubrics for fast matching.

    Creates a lightweight JSON file mapping table name hashes to database names.
    This enables O(1) matching during variant selection instead of loading all rubrics.

    Args:
        schemas_dir: Directory containing schema files
    """
    try:
        # Import hash lookup generator
        from mars.pipeline.raw_scanner.db_variant_selector.generate_hash_lookup import (
            generate_hash_lookup as _generate_hash_lookup,
        )

        if not schemas_dir.exists():
            logger.debug("  [SKIP] No schemas directory found")
            return

        # Load ignorable tables/prefixes/suffixes from config (single source of truth)
        try:
            from mars.config.schema import SchemaComparisonConfig

            _config = SchemaComparisonConfig()
            ignore_tables = _config.ignorable_tables
            ignore_prefixes = _config.ignorable_prefixes
            ignore_suffixes = _config.ignorable_suffixes
        except ImportError:
            # Fallback if config not available
            ignore_tables = GLOBAL_IGNORABLE_TABLES
            ignore_prefixes = {"sqlite_", "sqlean_"}
            ignore_suffixes = {
                "_content",
                "_segments",
                "_segdir",
                "_docsize",
                "_stat",
                "_SqliteDatabaseProperties",
            }

        # Generate hash lookup (suppress verbose output)
        import sys
        from io import StringIO

        # Redirect stdout to suppress verbose print statements
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        try:
            hash_lookup = _generate_hash_lookup(schemas_dir, ignore_tables, ignore_prefixes, ignore_suffixes)
        finally:
            # Restore stdout
            sys.stdout = old_stdout

        if hash_lookup:
            lookup_path = schemas_dir / "exemplar_hash_lookup.json"
            logger.debug(f"  [SUCCESS] Generated {len(hash_lookup)} hash entries")
            logger.debug(f"  [SAVED] {lookup_path}")
        else:
            logger.warning("  [WARNING] No hash entries generated")

    except Exception as e:
        logger.error(f"  [ERROR] Failed to generate hash lookup: {e}")
        import traceback

        traceback.print_exc()
        # Non-fatal - continue with scan
