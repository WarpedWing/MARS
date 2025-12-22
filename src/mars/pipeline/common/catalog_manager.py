"""Database catalog management module.

Handles loading and querying the database catalog for metadata,
ignorable tables, and skip patterns.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import yaml

from mars.config.schema import GLOBAL_IGNORABLE_TABLES
from mars.utils.debug_logger import logger

# Note: GLOBAL_IGNORABLE_TABLES is now imported from config.schema
# Schema-specific ignorable tables (like 'meta' for Chrome) are in artifact_recovery_catalog.yaml


class CatalogManager:
    """Manage database catalog and metadata lookups."""

    def __init__(self, catalog_path: Path | None = None):
        """
        Initialize catalog manager.

        Args:
            catalog_path: Optional path to catalog YAML file.
                          If provided but doesn't exist, falls back to default.
        """
        # If a custom path was provided but doesn't exist, fall back to default
        if catalog_path is not None and not catalog_path.exists():
            catalog_path = None
        self.catalog_path = catalog_path or self._find_default_catalog()
        self._catalog_cache = None

    def _find_default_catalog(self) -> Path:
        """Find default catalog location."""
        # Assume catalog is in ../../catalog/artifact_recovery_catalog.yaml relative to this file
        return Path(__file__).parent.parent.parent / "catalog" / "artifact_recovery_catalog.yaml"

    def get_catalog(self) -> dict | None:
        """
        Get or load database catalog (lazy loading).

        Returns:
            Dictionary with catalog data or None if not available
        """
        if self._catalog_cache is not None:
            return self._catalog_cache

        try:
            if not self.catalog_path.exists():
                logger.debug("Database catalog not found")
                return None

            with Path.open(self.catalog_path) as f:
                self._catalog_cache = yaml.safe_load(f)

            return self._catalog_cache
        except Exception as e:
            logger.debug(f"Failed to load database catalog: {e}")
            return None

    def get_ignorable_tables_for_rubric(self, rubric_name: str) -> list[str]:
        """
        Look up ignorable tables for a given rubric from the catalog.

        Args:
            rubric_name: Name of the rubric (e.g., "Chrome History_admin")
                         User suffixes (e.g., "_admin") are stripped automatically

        Returns:
            List of table names to ignore, or empty list if not found
        """
        catalog = self.get_catalog()
        if not catalog:
            return []

        # Strip user suffix if present (e.g., "Chrome History_admin" -> "Chrome History")
        clean_rubric_name = rubric_name
        if "_" in rubric_name:
            parts = rubric_name.rsplit("_", 1)
            clean_rubric_name = parts[0]

        # Search through all categories in catalog
        for category_name, entries in catalog.items():
            if category_name == "catalog_metadata":
                continue

            if not isinstance(entries, list):
                continue

            for entry in entries:
                if isinstance(entry, dict):
                    catalog_name = entry.get("name")
                    # Try both with and without user suffix
                    if catalog_name in (rubric_name, clean_rubric_name):
                        return entry.get("ignorable_tables", [])

        return []

    def get_skip_databases(self) -> dict:
        """
        Get list of database schemas to skip from catalog.

        Returns:
            Dictionary of skip patterns by category, or empty dict if not available
        """
        catalog = self.get_catalog()
        if not catalog:
            return {}

        return catalog.get("skip_databases", {})

    def get_all_group_names(self) -> list[str]:
        """
        Get all database group names from the catalog.

        Returns the top-level keys (group names like 'safari', 'chrome', 'messages')
        sorted alphabetically, excluding special entries like 'skip_databases'
        and 'catalog_metadata'.

        Returns:
            Sorted list of group names, or empty list if catalog not available
        """
        catalog = self.get_catalog()
        if not catalog:
            return []

        # Exclude special entries that aren't database groups
        excluded_keys = {"skip_databases", "catalog_metadata"}

        return sorted(key for key in catalog if key not in excluded_keys)

    def should_skip_database(self, db_tables: list[str]) -> tuple[bool, str, str]:
        """
        Check if database should be skipped based on its table schema.

        Args:
            db_tables: List of table names in the database

        Returns:
            Tuple of (should_skip, skip_category, reason)
            e.g., (True, "geoservices", "Read-only Apple system cache")
        """
        skip_dbs = self.get_skip_databases()
        if not skip_dbs:
            return (False, "", "")

        # Check each skip pattern
        for skip_category, skip_info in skip_dbs.items():
            table_patterns = skip_info.get("table_patterns", [])

            # Count how many tables match the patterns
            matches = 0
            for pattern in table_patterns:
                if "*" in pattern:
                    # Wildcard pattern (e.g., "GeoLookup_*")
                    prefix = pattern.replace("*", "")
                    matches += sum(1 for table in db_tables if table.startswith(prefix))
                else:
                    # Exact match
                    if pattern in db_tables:
                        matches += 1

            # Require at least 3 matching tables to avoid false positives
            if matches >= 3:
                reason = skip_info.get("reason", "System database")
                return (True, skip_category, reason)

        return (False, "", "")

    def is_database_empty(
        self,
        db_path: Path,
        schema_ignorable_tables: list[str] | None = None,
    ) -> bool:
        """
        Check if a database is "empty" (has tables but no meaningful data).

        A database is considered empty if:
        1. All meaningful tables have 0 rows, OR
        2. All rows in meaningful tables contain only NULL values

        Ignores metadata/system tables (global + schema-specific) when
        determining if database is empty.

        Args:
            db_path: Path to database to check
            schema_ignorable_tables: Optional schema-specific tables to ignore

        Returns:
            True if database has no meaningful data, False otherwise
        """
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0) as conn:
                cursor = conn.cursor()

                # Get all user tables (exclude sqlite_* tables)
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                all_tables = [row[0] for row in cursor.fetchall()]

                # Combine global and schema-specific ignorable tables
                ignorable_tables = GLOBAL_IGNORABLE_TABLES.copy()
                if schema_ignorable_tables:
                    ignorable_tables.update(schema_ignorable_tables)

                # Filter out ignorable tables
                tables = [t for t in all_tables if t not in ignorable_tables]

                if not tables:
                    return True  # No meaningful tables = empty

                # Check if any table has meaningful data (non-NULL values)
                has_real_data = False
                for table in tables:
                    try:
                        # First check if table has rows
                        cursor.execute(f"SELECT COUNT(*) FROM '{table}'")
                        count = cursor.fetchone()[0]

                        if count == 0:
                            continue  # Empty table

                        # Table has rows - check if they contain any non-NULL data
                        # Get column names
                        cursor.execute(f"PRAGMA table_info('{table}')")
                        columns = [row[1] for row in cursor.fetchall()]

                        if not columns:
                            continue

                        # Build query to check if ANY column in ANY row has
                        # non-NULL data. Use OR to check each column:
                        # (col1 IS NOT NULL OR col2 IS NOT NULL ...)
                        null_check = " OR ".join(f'"{col}" IS NOT NULL' for col in columns)
                        query = f"SELECT 1 FROM '{table}' WHERE {null_check} LIMIT 1"

                        result = cursor.execute(query).fetchone()
                        if result:
                            # Found at least one row with non-NULL data
                            has_real_data = True
                            break

                    except Exception:
                        # Skip tables that can't be queried (corrupted, etc.)
                        # If we can't check, assume it has data (conservative)
                        has_real_data = True
                        break

                # Empty if no tables have real data
                return not has_real_data

        except Exception:
            # If we can't read it, assume not empty (to avoid false positives)
            return False
