#!/usr/bin/env python3
"""
Self-Rubric Generation for CATALOG Lost & Found Processing

Generates schema rubrics from recovered databases to enable matching of
lost_and_found tables when catalog rubrics are unavailable.

Extracted from processor.py to improve modularity.
"""

from __future__ import annotations

import json
import traceback
from typing import TYPE_CHECKING

from mars.utils.database_utils import readonly_connection
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from pathlib import Path


def generate_self_rubric_for_catalog(
    db_path: Path,
    db_name: str,
    databases_dir: Path,
    min_tables: int = 3,
) -> dict | None:
    """
    Generate a self-rubric from the chosen variant database for CATALOG processing.

    This creates a rubric and schema from the database's existing tables,
    saves them to databases/schemas/{db_name}/, and returns rubric info for catalog.

    Args:
        db_path: Path to chosen variant database
        db_name: Database name
        databases_dir: Base directory (typically paths.db_selected_variants.parent)
        min_tables: Minimum tables required (default: 3)

    Returns:
        Dict with "rubric", "rubric_path", "db_path" or None if not qualified
    """

    try:
        # Get all tables
        from mars.pipeline.matcher.generate_sqlite_schema_rubric import (
            fetch_tables,
            generate_rubric,
        )

        with readonly_connection(db_path) as con:
            all_tables = fetch_tables(con)

            # Filter out system and FTS tables
            filtered_tables = []
            for name, sql in all_tables:
                if name.startswith("sqlite_"):
                    continue
                if any(
                    name.endswith(suffix)
                    for suffix in [
                        "_content",
                        "_segdir",
                        "_segments",
                        "_docsize",
                        "_stat",
                    ]
                ):
                    continue
                filtered_tables.append((name, sql))

            # Check if qualifies (≥min_tables tables)
            if len(filtered_tables) < min_tables:
                return None

            # Generate rubric
            rubric = generate_rubric(con, filtered_tables, rubric_name=db_name)

        # Validate rubric is a dict with tables
        if not isinstance(rubric, dict):
            return None

        if "tables" not in rubric:
            return None

        if not isinstance(rubric["tables"], (dict, list)):
            return None

        if isinstance(rubric["tables"], dict) and len(rubric["tables"]) == 0:
            return None

        # Create descriptive folder name: tablename_hashid

        # Get first table name for descriptive label
        tables = rubric.get("tables", {})
        if isinstance(tables, dict):
            first_table = next(iter(tables.keys())) if tables else "unknown"
        else:
            first_table = tables[0].get("name", "unknown") if tables else "unknown"

        # Create hash from db_name (last 8 chars)
        db_hash = db_name[-8:] if len(db_name) >= 8 else db_name

        # Format: tablename_hashid (similar to metamatch groups)
        schema_label = f"{first_table}_{db_hash}"
        schemas_dir = databases_dir / "schemas" / schema_label

        schemas_dir.mkdir(parents=True, exist_ok=True)

        rubric_path = schemas_dir / f"{schema_label}.rubric.json"
        with rubric_path.open("w") as f:
            json.dump(rubric, f, indent=2)

        return {
            "rubric": rubric,
            "rubric_path": rubric_path,
            "db_path": db_path,
            "name": db_name,
            "schema_label": schema_label,
        }

    except Exception as e:
        # Check if this is an expected error for malformed databases (variant X)
        error_msg = str(e).lower()
        is_malformed_error = any(
            keyword in error_msg
            for keyword in [
                "database disk image is malformed",
                "malformed database",
                "file is not a database",
                "file is encrypted",
            ]
        )

        if not is_malformed_error:
            # Unexpected error - log at ERROR level with traceback
            logger.error(f"    ✗ Failed to generate self-rubric: {e}")
            logger.error(f"    Traceback: {traceback.format_exc()}")

        return None
