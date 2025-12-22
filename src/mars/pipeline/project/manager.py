#!/usr/bin/env python3
"""
MARS Project Management
by WarpedWing Labs

Handles project creation, loading, and persistence.
Each project consists of:
  - .marsproj file (JSON config with metadata)
  - project.db (SQLite database with analysis data)
  - output/ directory (organized forensic artifacts)
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mars.utils.debug_logger import logger


class MARSProject:
    """MARS forensic analysis project."""

    VERSION = "1.0.0"

    def __init__(self, project_path: Path):
        """
        Initialize a project.

        Args:
            project_path: Path to .marsproj file
        """
        self.project_path = project_path.resolve()
        self.project_dir = self.project_path.parent
        self.config: dict[str, Any] = {}
        self.db_path = self.project_dir / "project.db"

    @classmethod
    def create(
        cls,
        directory: Path,
        project_name: str,
        examiner_name: str | None = None,
        case_number: str | None = None,
        description: str | None = None,
    ) -> MARSProject:
        """
        Create a new project.

        Args:
            directory: Parent directory to create project in
            project_name: Name of the project
            examiner_name: Name of forensic examiner
            case_number: Case/evidence number
            description: Project description

        Returns:
            MARSProject instance
        """
        # Create project folder based on project name
        # Sanitize project name for filesystem use
        safe_project_name = "".join(c if c.isalnum() or c in ("-", "_", " ") else "_" for c in project_name)
        project_dir = (directory / safe_project_name).resolve()
        project_dir.mkdir(parents=True, exist_ok=True)

        # Create .marsproj file in the project directory
        project_file = project_dir / f"{safe_project_name}.marsproj"

        config = {
            "version": cls.VERSION,
            "project_name": project_name,
            "examiner_name": examiner_name,
            "case_number": case_number,
            "description": description,
            "created_at": datetime.now(UTC).isoformat(),
        }

        # Write config
        with Path.open(project_file, "w") as f:
            json.dump(config, f, indent=2)

        # Create project instance
        project = cls(project_file)
        project.config = config

        # Initialize database
        project._init_database()

        # Create output directory structure
        project._create_directory_structure()

        return project

    @classmethod
    def load(cls, project_path: Path) -> MARSProject:
        """
        Load an existing project.

        Args:
            project_path: Path to .marsproj file

        Returns:
            MARSProject instance

        Raises:
            FileNotFoundError: If project file doesn't exist
            ValueError: If project file is invalid
        """
        project_path = project_path.resolve()

        if not project_path.exists():
            raise FileNotFoundError(f"Project file not found: {project_path}")

        if project_path.suffix != ".marsproj":
            raise ValueError(f"Invalid project file (must be .marsproj): {project_path}")
        project = cls(project_path)
        project._load_config()

        # Auto-migrate existing scans from filesystem
        with contextlib.suppress(Exception):
            project.migrate_existing_scans()

        return project

    def _load_config(self):
        """Load project configuration from .marsproj file."""
        with Path.open(self.project_path) as f:
            self.config = json.load(f)

    def _init_database(self):
        """Initialize project SQLite database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Create tables
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                )
            """
            )

            # Track exemplar analysis scans
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS exemplar_scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    databases_found INTEGER DEFAULT 0,
                    schemas_generated INTEGER DEFAULT 0,
                    rubrics_generated INTEGER DEFAULT 0,
                    hash_entries INTEGER DEFAULT 0,
                    scan_duration_seconds REAL,
                    is_active INTEGER DEFAULT 1,
                    is_last_used INTEGER DEFAULT 0,
                    notes TEXT
                )
            """
            )

            # Track recovery/carved file scans
            # Note: exemplar_scan_id = -1 indicates imported exemplar (no project scan)
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS recovery_scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    exemplar_scan_id INTEGER NOT NULL,
                    target_path TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    status TEXT NOT NULL,
                    databases_found INTEGER DEFAULT 0,
                    databases_matched INTEGER DEFAULT 0,
                    databases_unmatched INTEGER DEFAULT 0,
                    scan_duration_seconds REAL,
                    is_active INTEGER DEFAULT 1,
                    notes TEXT,
                    FOREIGN KEY (exemplar_scan_id) REFERENCES exemplar_scans(id)
                )
            """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS processing_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    status TEXT,
                    details TEXT,
                    error TEXT
                )
            """
            )

            # Migration: Add description column to exemplar_scans if not exists
            cursor.execute("PRAGMA table_info(exemplar_scans)")
            columns = [row[1] for row in cursor.fetchall()]
            if "description" not in columns:
                cursor.execute("ALTER TABLE exemplar_scans ADD COLUMN description TEXT")

            # Migration: Add description column to recovery_scans if not exists
            cursor.execute("PRAGMA table_info(recovery_scans)")
            columns = [row[1] for row in cursor.fetchall()]
            if "description" not in columns:
                cursor.execute("ALTER TABLE recovery_scans ADD COLUMN description TEXT")

            conn.commit()

    def _create_directory_structure(self):
        """Create standard project directory structure."""
        directories = [
            "output",
            "reports",
        ]

        for dir_name in directories:
            (self.project_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def log_operation(
        self,
        operation: str,
        status: str = "success",
        details: str | None = None,
        error: str | None = None,
    ):
        """
        Log an operation to the processing log.

        Args:
            operation: Description of operation
            status: success, error, warning, info
            details: Additional details
            error: Error message if applicable
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO processing_log (timestamp, operation, status, details, error)
                VALUES (?, ?, ?, ?, ?)
            """,
                (datetime.now(UTC).isoformat(), operation, status, details, error),
            )

            conn.commit()

    def get_processing_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get recent processing log entries.

        Args:
            limit: Maximum number of entries to return

        Returns:
            List of log entry dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM processing_log
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )

            results = [dict(row) for row in cursor.fetchall()]

        return results

    # Exemplar Scan Management

    def start_exemplar_scan(
        self,
        source_path: Path,
        output_dir: Path,
        description: str | None = None,
    ) -> int:
        """
        Start a new exemplar scan.

        Args:
            source_path: Path to source system
            output_dir: Output directory name (relative to project output/)
            description: Optional user description for this scan

        Returns:
            Scan ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            timestamp = datetime.now(UTC).isoformat()
            output_dir_rel = str(output_dir.relative_to(self.project_dir / "output"))

            cursor.execute(
                """
                INSERT INTO exemplar_scans
                (timestamp, source_path, output_dir, status, description)
                VALUES (?, ?, ?, 'in_progress', ?)
                """,
                (timestamp, str(source_path), output_dir_rel, description),
            )

            scan_id = cursor.lastrowid
            conn.commit()

        if scan_id is None:
            raise RuntimeError("Failed to create exemplar scan: no ID returned")
        return scan_id

    def complete_exemplar_scan(
        self,
        scan_id: int,
        databases_found: int = 0,
        schemas_generated: int = 0,
        rubrics_generated: int = 0,
        hash_entries: int = 0,
        duration_seconds: float | None = None,
    ):
        """Mark exemplar scan as completed with results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE exemplar_scans
                SET status = 'completed',
                    databases_found = ?,
                    schemas_generated = ?,
                    rubrics_generated = ?,
                    hash_entries = ?,
                    scan_duration_seconds = ?
                WHERE id = ?
                """,
                (
                    databases_found,
                    schemas_generated,
                    rubrics_generated,
                    hash_entries,
                    duration_seconds,
                    scan_id,
                ),
            )

            conn.commit()

    def fail_exemplar_scan(self, scan_id: int, error: str):
        """Mark exemplar scan as failed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE exemplar_scans
                SET status = 'failed', notes = ?
                WHERE id = ?
                """,
                (error, scan_id),
            )

            conn.commit()

    def get_exemplar_scans(self, active_only: bool = True) -> list[dict[str, Any]]:
        """
        Get all exemplar scans.

        Args:
            active_only: If True, only return active (non-deleted) scans

        Returns:
            List of scan dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if active_only:
                cursor.execute(
                    """
                    SELECT * FROM exemplar_scans
                    WHERE is_active = 1 AND status = 'completed'
                    ORDER BY timestamp DESC
                    """
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM exemplar_scans
                    ORDER BY timestamp DESC
                    """
                )

            results = [dict(row) for row in cursor.fetchall()]

        return results

    def get_last_used_exemplar_scan(self) -> dict[str, Any] | None:
        """Get the last used exemplar scan."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT * FROM exemplar_scans
                WHERE is_active = 1 AND status = 'completed' AND is_last_used = 1
                ORDER BY timestamp DESC
                LIMIT 1
                """
            )

            row = cursor.fetchone()

        return dict(row) if row else None

    def set_last_used_exemplar_scan(self, scan_id: int):
        """Mark an exemplar scan as last used."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Clear all last_used flags
            cursor.execute("UPDATE exemplar_scans SET is_last_used = 0")

            # Set this one as last used
            cursor.execute(
                "UPDATE exemplar_scans SET is_last_used = 1 WHERE id = ?",
                (scan_id,),
            )

            conn.commit()

    def mark_exemplar_scan_inactive(self, scan_id: int):
        """Mark an exemplar scan as inactive (output directory no longer exists)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                "UPDATE exemplar_scans SET is_active = 0 WHERE id = ?",
                (scan_id,),
            )

            conn.commit()

    # Recovery Scan Management

    def start_recovery_scan(
        self,
        target_path: Path,
        exemplar_scan_id: int,
        output_dir: Path,
        description: str | None = None,
    ) -> int:
        """
        Start a new recovery scan.

        Args:
            target_path: Path to target directory with carved files
            exemplar_scan_id: ID of exemplar scan to match against, or -1 for imported exemplars
            output_dir: Output directory name (relative to exemplar's candidates/ or project output/)
            description: Optional user description for this scan

        Returns:
            Scan ID
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            timestamp = datetime.now(UTC).isoformat()

            # Determine output_dir_rel based on whether we have an exemplar scan
            # Note: -1 is sentinel for imported exemplar (no project exemplar scan)
            if exemplar_scan_id is not None and exemplar_scan_id > 0:
                # Get exemplar output dir to construct full path
                cursor.execute(
                    "SELECT output_dir FROM exemplar_scans WHERE id = ?",
                    (exemplar_scan_id,),
                )
                result = cursor.fetchone()
                if not result:
                    raise ValueError(f"Exemplar scan {exemplar_scan_id} not found")

                exemplar_output = result[0]
                output_dir_rel = str(output_dir.relative_to(self.project_dir / "output" / exemplar_output))
            else:
                # For imported exemplars (-1) or None, store path relative to project output
                output_dir_rel = str(output_dir.relative_to(self.project_dir / "output"))

            cursor.execute(
                """
                INSERT INTO recovery_scans
                (timestamp, exemplar_scan_id, target_path, output_dir, status, description)
                VALUES (?, ?, ?, ?, 'in_progress', ?)
                """,
                (
                    timestamp,
                    exemplar_scan_id,
                    str(target_path),
                    output_dir_rel,
                    description,
                ),
            )

            scan_id = cursor.lastrowid
            conn.commit()

        if scan_id is None:
            raise RuntimeError("Failed to create recovery scan: no ID returned")
        return scan_id

    def complete_recovery_scan(
        self,
        scan_id: int,
        databases_found: int = 0,
        databases_matched: int = 0,
        databases_unmatched: int = 0,
        duration_seconds: float | None = None,
    ):
        """Mark recovery scan as completed with results."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE recovery_scans
                SET status = 'completed',
                    databases_found = ?,
                    databases_matched = ?,
                    databases_unmatched = ?,
                    scan_duration_seconds = ?
                WHERE id = ?
                """,
                (
                    databases_found,
                    databases_matched,
                    databases_unmatched,
                    duration_seconds,
                    scan_id,
                ),
            )

            conn.commit()

    def fail_recovery_scan(self, scan_id: int, error: str):
        """Mark recovery scan as failed."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE recovery_scans
                SET status = 'failed', notes = ?
                WHERE id = ?
                """,
                (error, scan_id),
            )

            conn.commit()

    def get_recovery_scans(self, exemplar_scan_id: int | None = None, active_only: bool = True) -> list[dict[str, Any]]:
        """
        Get recovery scans.

        Args:
            exemplar_scan_id: If specified, only return scans for this exemplar
            active_only: If True, only return active scans

        Returns:
            List of scan dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            if exemplar_scan_id:
                if active_only:
                    cursor.execute(
                        """
                        SELECT * FROM recovery_scans
                        WHERE exemplar_scan_id = ? AND is_active = 1
                        ORDER BY timestamp DESC
                        """,
                        (exemplar_scan_id,),
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM recovery_scans
                        WHERE exemplar_scan_id = ?
                        ORDER BY timestamp DESC
                        """,
                        (exemplar_scan_id,),
                    )
            else:
                if active_only:
                    cursor.execute(
                        """
                        SELECT * FROM recovery_scans
                        WHERE is_active = 1
                        ORDER BY timestamp DESC
                        """
                    )
                else:
                    cursor.execute(
                        """
                        SELECT * FROM recovery_scans
                        ORDER BY timestamp DESC
                        """
                    )

            return [dict(row) for row in cursor.fetchall()]

    def migrate_existing_scans(self):
        """
        Auto-detect and import existing exemplar scans from filesystem.

        Scans the output directory for existing exemplar folders and imports them
        into the database if they're not already tracked.
        """
        output_dir = self.project_dir / "output"
        if not output_dir.exists():
            return

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get existing tracked output dirs
            cursor.execute("SELECT output_dir FROM exemplar_scans")
            tracked_dirs = {row[0] for row in cursor.fetchall()}

            # Scan for exemplar folders
            for item in output_dir.iterdir():
                if not item.is_dir():
                    continue

                # Check if this looks like an exemplar scan output
                exemplar_dir = item / "exemplar"
                if not exemplar_dir.exists():
                    continue

                # Check if already tracked
                rel_path = str(item.relative_to(output_dir))
                if rel_path in tracked_dirs:
                    continue

                # Check if it has the required structure
                catalog_dir = exemplar_dir / "databases" / "catalog"
                schemas_dir = exemplar_dir / "databases" / "schemas"

                if not (catalog_dir.exists() and schemas_dir.exists()):
                    continue

                # Skip salvage scans - they create empty exemplar dirs but aren't real exemplars
                manifest_path = item / "_manifest.json"
                if manifest_path.exists():
                    try:
                        import json

                        with manifest_path.open() as f:
                            manifest = json.load(f)
                            if manifest.get("source_type") == "salvage":
                                continue
                    except Exception:
                        pass

                # Try to extract metadata from the scan
                # Look for a scan summary or metadata file
                databases_found = 0
                schemas_generated = 0
                rubrics_generated = 0
                hash_entries = 0

                # Count databases in catalog
                if catalog_dir.exists():
                    databases_found = sum(1 for _ in catalog_dir.iterdir() if _.is_dir())

                # Count schemas
                if schemas_dir.exists():
                    schemas_generated = sum(1 for _ in schemas_dir.iterdir() if _.is_dir())
                    rubrics_generated = schemas_generated

                # Check for hash lookup file
                hash_file = schemas_dir / "exemplar_hash_lookup.json"
                if hash_file.exists():
                    try:
                        import json

                        with hash_file.open() as f:
                            hash_data = json.load(f)
                            hash_entries = len(hash_data)
                    except Exception:
                        pass

                # Extract timestamp from folder name if possible
                # Format: MARS_ProjectName_20251101_185012
                parts = item.name.split("_")
                timestamp = datetime.now(UTC).isoformat()
                if len(parts) >= 2:
                    # Try to parse date/time from last parts
                    try:
                        date_str = parts[-2]  # YYYYMMDD
                        time_str = parts[-1]  # HHMMSS
                        datetime_str = f"{date_str}_{time_str}"
                        dt = datetime.strptime(datetime_str, "%Y%m%d_%H%M%S")
                        timestamp = dt.replace(tzinfo=UTC).isoformat()
                    except (ValueError, IndexError):
                        pass

                # Insert as completed scan with unknown source
                cursor.execute(
                    """
                    INSERT INTO exemplar_scans
                    (timestamp, source_path, output_dir, status, databases_found,
                     schemas_generated, rubrics_generated, hash_entries, notes)
                    VALUES (?, ?, ?, 'completed', ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        "unknown (imported)",
                        rel_path,
                        databases_found,
                        schemas_generated,
                        rubrics_generated,
                        hash_entries,
                        "Auto-imported from existing scan",
                    ),
                )

                logger.info(f"Imported existing scan: {rel_path}")

            conn.commit()

    def __repr__(self) -> str:
        """String representation of project."""
        return f"MARSProject(name='{self.config.get('project_name')}', path='{self.project_path}')"
