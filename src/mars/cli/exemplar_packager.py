"""Exemplar Package Export/Import.

Create sanitized exemplar packages for sharing (rubrics + schemas + empty DB shells)
without private data. Import external exemplar packages for candidate scanning.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import sqlite3
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from mars.utils.cleanup_utilities import cleanup_sqlite_directory
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from collections.abc import Callable

# Format version for package compatibility checking
PACKAGE_FORMAT_VERSION = "1.0"

# Pattern to extract username from exemplar names like "Safari History_john_doe"
# or "CFURL Cache Database_admin_v1"
_USERNAME_PATTERN = re.compile(r"^(.+?)_([a-zA-Z][a-zA-Z0-9_]*)(?:_v\d+)?$")


@dataclass
class PackageManifest:
    """Metadata for an exemplar package."""

    format_version: str = PACKAGE_FORMAT_VERSION
    exemplar_name: str = ""
    description: str = ""
    created_date: str = ""
    created_by_mars_version: str = "1.0"
    database_count: int = 0
    os_info: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "format_version": self.format_version,
            "exemplar_name": self.exemplar_name,
            "description": self.description,
            "created_date": self.created_date,
            "created_by_mars_version": self.created_by_mars_version,
            "database_count": self.database_count,
            "os_info": self.os_info,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PackageManifest:
        """Create from dictionary."""
        return cls(
            format_version=data.get("format_version", "1.0"),
            exemplar_name=data.get("exemplar_name", ""),
            description=data.get("description", ""),
            created_date=data.get("created_date", ""),
            created_by_mars_version=data.get("created_by_mars_version", "1.0"),
            database_count=data.get("database_count", 0),
            os_info=data.get("os_info", {}),
        )


@dataclass
class ExportResult:
    """Result of an exemplar export operation."""

    success: bool
    output_path: Path | None = None
    database_count: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class ExemplarPackager:
    """Export sanitized exemplar packages for sharing."""

    def __init__(self, exemplar_dir: Path):
        """Initialize packager.

        Args:
            exemplar_dir: Path to project's exemplar directory
                          (e.g., project/output/MARS_*/exemplar)
        """
        self.exemplar_dir = exemplar_dir
        self.schemas_dir = exemplar_dir / "databases" / "schemas"
        self.catalog_dir = exemplar_dir / "databases" / "catalog"

    def export(
        self,
        output_path: Path,
        exemplar_name: str,
        description: str = "",
        os_info: dict | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> ExportResult:
        """Create shareable exemplar package.

        Args:
            output_path: Directory where package folder will be created
            exemplar_name: Name for the package (e.g., "macOS_13.6_Reference")
            description: Optional description of the exemplar
            os_info: Optional OS information dict with keys: name, version, build
            progress_callback: Optional callback(current, total, message) for progress

        Returns:
            ExportResult with success status and details
        """
        result = ExportResult(success=False)

        # Validate source directories exist
        if not self.exemplar_dir.exists():
            result.errors.append(f"Exemplar directory not found: {self.exemplar_dir}")
            return result

        if not self.schemas_dir.exists():
            result.errors.append("No schemas directory found. Run exemplar scan first.")
            return result

        # Create package folder
        package_name = f"Exemplar_{exemplar_name}"
        package_dir = output_path / package_name

        if package_dir.exists():
            result.errors.append(f"Package directory already exists: {package_dir}")
            return result

        try:
            package_dir.mkdir(parents=True)

            # Build user mapping for anonymization
            user_mapping = self._build_user_mapping()

            # Count total items for progress
            schema_dirs = [d for d in self.schemas_dir.iterdir() if d.is_dir()]
            catalog_dirs = [d for d in self.catalog_dir.iterdir() if d.is_dir()] if self.catalog_dir.exists() else []
            total_items = len(schema_dirs) + len(catalog_dirs) + 2  # +2 for manifest & catalog.yaml
            current_item = 0

            # Create package structure
            pkg_schemas_dir = package_dir / "schemas"
            pkg_catalog_dir = package_dir / "catalog"
            pkg_schemas_dir.mkdir()
            pkg_catalog_dir.mkdir()

            # Copy and sanitize schemas/rubrics
            for schema_dir in schema_dirs:
                current_item += 1
                if progress_callback:
                    progress_callback(
                        current_item,
                        total_items,
                        f"Processing schema: {schema_dir.name}",
                    )

                sanitized_name = self._sanitize_name(schema_dir.name, user_mapping)
                dest_dir = pkg_schemas_dir / sanitized_name
                dest_dir.mkdir()

                # Copy and sanitize rubric
                for rubric_file in schema_dir.glob("*.rubric.json"):
                    sanitized_rubric = self._sanitize_rubric(rubric_file, user_mapping)
                    dest_rubric = dest_dir / self._sanitize_name(rubric_file.name, user_mapping)
                    with dest_rubric.open("w") as f:
                        json.dump(sanitized_rubric, f, indent=2)

                # Copy schema CSV files (no sanitization needed - just structure)
                for schema_file in schema_dir.glob("*.schema.csv"):
                    dest_schema = dest_dir / self._sanitize_name(schema_file.name, user_mapping)
                    shutil.copy2(schema_file, dest_schema)

                result.database_count += 1

            # Create empty database shells from catalog
            if self.catalog_dir.exists():
                for catalog_entry in catalog_dirs:
                    current_item += 1
                    if progress_callback:
                        progress_callback(
                            current_item,
                            total_items,
                            f"Creating shell: {catalog_entry.name}",
                        )

                    sanitized_name = self._sanitize_name(catalog_entry.name, user_mapping)
                    dest_dir = pkg_catalog_dir / sanitized_name
                    dest_dir.mkdir()

                    # Process database files in this catalog entry
                    self._process_catalog_entry(catalog_entry, dest_dir, user_mapping, result)

            # Copy relevant artifact_recovery_catalog.yaml entries
            current_item += 1
            if progress_callback:
                progress_callback(current_item, total_items, "Copying catalog.yaml")

            self._copy_catalog_yaml(package_dir)

            # Generate manifest.json
            current_item += 1
            if progress_callback:
                progress_callback(current_item, total_items, "Creating manifest")

            manifest = PackageManifest(
                exemplar_name=exemplar_name,
                description=description,
                created_date=datetime.now(UTC).isoformat(),
                database_count=result.database_count,
                os_info=os_info or {},
            )

            manifest_path = package_dir / "manifest.json"
            with manifest_path.open("w") as f:
                json.dump(manifest.to_dict(), f, indent=2)

            result.success = True
            result.output_path = package_dir

        except Exception as e:
            result.errors.append(f"Export failed: {e}")
            # Clean up partial export
            if package_dir.exists():
                shutil.rmtree(package_dir, ignore_errors=True)

        return result

    def _build_user_mapping(self) -> dict[str, str]:
        """Build mapping of actual usernames to generic user1, user2, etc.

        Scans schema directory names to find unique usernames.

        Returns:
            Dict mapping actual_username -> "user1", "user2", etc.
        """
        usernames: set[str] = set()

        # Scan schema directories for username patterns
        if self.schemas_dir.exists():
            for schema_dir in self.schemas_dir.iterdir():
                if not schema_dir.is_dir():
                    continue

                match = _USERNAME_PATTERN.match(schema_dir.name)
                if match:
                    username = match.group(2)
                    # Exclude system-like names
                    if username.lower() not in ("multi", "system", "root", "empty"):
                        usernames.add(username)

        # Create mapping
        mapping = {}
        for i, username in enumerate(sorted(usernames), start=1):
            mapping[username] = f"user{i}"

        return mapping

    def _sanitize_name(self, name: str, user_mapping: dict[str, str]) -> str:
        """Sanitize a name by replacing usernames with generic placeholders.

        Args:
            name: Original name (e.g., "Safari History_john_doe.rubric.json")
            user_mapping: Dict of username -> placeholder

        Returns:
            Sanitized name (e.g., "Safari History_user1.rubric.json")
        """
        result = name
        for username, placeholder in user_mapping.items():
            # Replace _username with _placeholder (case-sensitive)
            result = result.replace(f"_{username}", f"_{placeholder}")
        return result

    def _sanitize_rubric(self, rubric_path: Path, user_mapping: dict[str, str]) -> dict:
        """Load and sanitize a rubric file.

        Removes/anonymizes:
        - source_db field (contains file paths with usernames)
        - Any path-like values that might contain usernames

        Args:
            rubric_path: Path to rubric JSON file
            user_mapping: Dict of username -> placeholder

        Returns:
            Sanitized rubric dict
        """
        with rubric_path.open() as f:
            rubric = json.load(f)

        # Rubric files are always dict at top level
        sanitized = self._sanitize_rubric_dict(rubric, user_mapping)
        assert isinstance(sanitized, dict)
        return sanitized

    def _sanitize_rubric_dict(self, data: dict | list, user_mapping: dict) -> dict | list:
        """Recursively sanitize a rubric dictionary.

        Args:
            data: Dict or list to sanitize
            user_mapping: Dict of username -> placeholder

        Returns:
            Sanitized data structure
        """
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                # Remove source_db entirely (contains full paths)
                if key == "source_db":
                    continue

                # Sanitize rubric_name field
                if key == "rubric_name" and isinstance(value, str):
                    result[key] = self._sanitize_name(value, user_mapping)
                elif isinstance(value, (dict, list)):
                    result[key] = self._sanitize_rubric_dict(value, user_mapping)
                elif isinstance(value, str):
                    # Check if value looks like a path and sanitize if so
                    if "/" in value or "\\" in value:
                        sanitized = value
                        for username, placeholder in user_mapping.items():
                            sanitized = sanitized.replace(f"/Users/{username}/", f"/Users/{placeholder}/")
                            sanitized = sanitized.replace(f"\\Users\\{username}\\", f"\\Users\\{placeholder}\\")
                        result[key] = sanitized
                    else:
                        result[key] = value
                else:
                    result[key] = value
            return result
        if isinstance(data, list):
            return [
                self._sanitize_rubric_dict(item, user_mapping) if isinstance(item, (dict, list)) else item
                for item in data
            ]
        return data

    def _create_empty_shell(self, source_db: Path, dest_db: Path, db_name: str = "") -> None:
        """Create empty shell of database (schema only, no data).

        1. Copy database to output
        2. DELETE FROM all tables (with fallback to DROP+CREATE if triggers fail)
        3. VACUUM to remove free page data
        4. Verify all tables have 0 rows - FAIL if not

        Args:
            source_db: Path to source database
            dest_db: Path for output empty shell
            db_name: Database name for logging context

        Raises:
            RuntimeError: If any table cannot be cleared of data
        """
        # Copy database first
        shutil.copy2(source_db, dest_db)

        # Connect and clear data
        con = sqlite3.connect(dest_db)
        cur = con.cursor()

        try:
            # Get all user tables with their CREATE statements
            cur.execute("""
                SELECT name, sql FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL
            """)
            tables = [(row[0], row[1]) for row in cur.fetchall()]

            failed_tables: list[str] = []

            # Track tables that were already deleted (e.g., FTS shadow tables)
            already_deleted: set[str] = set()

            # Delete all data from each table
            for table, create_sql in tables:
                # Skip tables that were already deleted (e.g., FTS shadow tables)
                if table in already_deleted:
                    continue

                # Quote table name to handle special characters
                quoted = f'"{table}"' if '"' not in table else f"[{table}]"

                # Try DELETE first
                try:
                    cur.execute(f"DELETE FROM {quoted}")
                except sqlite3.Error as e:
                    error_msg = str(e).lower()

                    # If table doesn't exist, it was already deleted (e.g., FTS shadow table)
                    # This is expected and safe - no data to leak
                    if "no such table" in error_msg:
                        logger.debug(f"[{db_name}] Table {table} already deleted (FTS shadow table)")
                        already_deleted.add(table)
                        continue

                    # DELETE failed (likely trigger issue) - try DROP+CREATE
                    logger.debug(f"[{db_name}] DELETE failed for {table}: {e}, trying DROP+CREATE")

                    # Step 1: Try to drop the table
                    drop_succeeded = False
                    try:
                        cur.execute(f"DROP TABLE {quoted}")
                        drop_succeeded = True
                    except sqlite3.Error as drop_err:
                        drop_error_msg = str(drop_err).lower()
                        # Table doesn't exist - already deleted
                        if "no such table" in drop_error_msg:
                            logger.debug(f"[{db_name}] Table {table} already deleted")
                            already_deleted.add(table)
                            continue
                        logger.debug(f"[{db_name}] DROP failed for {table}: {drop_err}")

                    # Step 2: Try to recreate with original SQL
                    if drop_succeeded:
                        try:
                            cur.execute(create_sql)
                            continue  # Success - move to next table
                        except sqlite3.Error as create_err:
                            logger.debug(f"[{db_name}] CREATE failed for {table}: {create_err}, trying stub")

                    # Step 3: If we get here, try stub table (drop first if needed)
                    stub_sql = self._create_stub_table_sql(table, create_sql)
                    if stub_sql:
                        try:
                            # Make sure table is dropped before creating stub
                            if not drop_succeeded:
                                try:
                                    cur.execute(f"DROP TABLE IF EXISTS {quoted}")
                                except sqlite3.Error:
                                    # DROP failed (custom tokenizer prevents even DROP)
                                    # Use writable_schema to forcibly remove the table entry
                                    logger.debug(f"[{db_name}] Using writable_schema to remove {table}")
                                    cur.execute("PRAGMA writable_schema = ON")
                                    cur.execute(
                                        "DELETE FROM sqlite_master WHERE type='table' AND name=?",
                                        (table,),
                                    )
                                    # Also remove any associated shadow tables AND their indexes (FTS)
                                    # FTS shadow tables: entity_fts_content, entity_fts_segments, etc.
                                    cur.execute(
                                        "DELETE FROM sqlite_master WHERE name LIKE ?",
                                        (f"{table}_%",),
                                    )
                                    # Remove orphan indexes that reference deleted shadow tables
                                    # These are named like: sqlite_autoindex_entity_fts_segdir_1
                                    cur.execute(
                                        "DELETE FROM sqlite_master WHERE type='index' AND name LIKE ?",
                                        (f"sqlite_autoindex_{table}_%",),
                                    )
                                    cur.execute("PRAGMA writable_schema = OFF")
                                    # Must commit after schema changes, then reopen connection
                                    # to clear SQLite's internal cache of the deleted table
                                    con.commit()
                                    con.close()
                                    con = sqlite3.connect(dest_db)
                                    cur = con.cursor()
                            cur.execute(stub_sql)
                            logger.debug(f"[{db_name}] Created stub table for {table}")
                            continue  # Success
                        except sqlite3.Error as e3:
                            logger.error(f"[{db_name}] Stub creation failed for {table}: {e3}")
                            failed_tables.append(table)
                    else:
                        logger.error(f"[{db_name}] Could not create stub for {table}")
                        failed_tables.append(table)

            con.commit()

            # Clear SQLite system tables that contain stale metadata
            # Keep table/index names but reset data values (now meaningless for empty DBs)
            with contextlib.suppress(sqlite3.Error):
                # Reset autoincrement counters to 0 (preserves table names)
                cur.execute("UPDATE sqlite_sequence SET seq = 0")

            with contextlib.suppress(sqlite3.Error):
                # Clear statistics but keep table/index names (stat format: "nrow ncol...")
                cur.execute("UPDATE sqlite_stat1 SET stat = '0'")

            with contextlib.suppress(sqlite3.Error):
                # Clear detailed statistics (sample values no longer valid)
                cur.execute("UPDATE sqlite_stat4 SET stat = '0', sample = NULL")

            con.commit()

            # VACUUM to remove free page data (where deleted data might linger)
            cur.execute("VACUUM")

            # Verify ALL tables are empty - this is critical for privacy
            # Track tables that already failed clearing (we know they have issues)
            already_failed = set(failed_tables)

            for table, _ in tables:
                # Skip tables that were already deleted (FTS shadow tables)
                if table in already_deleted:
                    continue

                quoted = f'"{table}"' if '"' not in table else f"[{table}]"
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {quoted}")
                    count = cur.fetchone()[0]
                    if count > 0:
                        failed_tables.append(f"{table} ({count} rows)")
                except sqlite3.Error:
                    # If we couldn't clear this table AND can't query it,
                    # it's unsafe - data may still exist
                    if table in already_failed:
                        failed_tables.append(f"{table} (unverifiable)")
                    # Otherwise table was successfully dropped/recreated, skip

            if failed_tables:
                raise RuntimeError(f"Failed to clear user data from tables: {', '.join(failed_tables)}")

        finally:
            con.close()

    def _create_stub_table_sql(self, table_name: str, original_sql: str) -> str | None:
        """Create SQL for a stub table when original (e.g., FTS) can't be recreated.

        Parses FTS virtual table definitions and creates equivalent regular tables.
        This preserves column structure for schema matching while avoiding
        custom tokenizers that aren't available in standard SQLite.

        Args:
            table_name: Name of the table
            original_sql: Original CREATE statement that failed

        Returns:
            CREATE TABLE SQL for stub, or None if parsing fails
        """
        # Handle FTS tables: CREATE VIRTUAL TABLE x USING fts4(col1, col2, ...)
        original_lower = original_sql.lower()
        if "virtual table" in original_lower and "using fts" in original_lower:
            # Match: USING fts3/fts4/fts5(columns...)
            match = re.search(r"using\s+fts\d\s*\((.+)\)\s*$", original_sql, re.IGNORECASE)
            if match:
                content = match.group(1)
                # Parse FTS column definitions - filter out FTS options like tokenize=
                columns = []
                for part in content.split(","):
                    part = part.strip()
                    # Skip FTS options (tokenize=, content=, etc.)
                    if "=" in part:
                        continue
                    # Column name (may have type like "col TEXT")
                    col_name = part.split()[0].strip('"[]')
                    if col_name:
                        columns.append(f'"{col_name}" TEXT')

                if columns:
                    quoted_name = f'"{table_name}"' if '"' not in table_name else f"[{table_name}]"
                    return f"CREATE TABLE {quoted_name} ({', '.join(columns)})"

        return None

    def _process_catalog_entry(
        self,
        source_dir: Path,
        dest_dir: Path,
        user_mapping: dict[str, str],
        result: ExportResult,
    ) -> None:
        """Process a catalog entry directory, handling nested structures.

        Handles both flat structures (DB file directly in entry dir) and
        multi-profile structures (Chrome's Default/, Profile 1/, etc.).

        Args:
            source_dir: Source catalog entry directory
            dest_dir: Destination directory for sanitized files
            user_mapping: Username mapping for sanitization
            result: ExportResult to accumulate warnings
        """
        # Find all potential database files in this directory
        for item in source_dir.iterdir():
            # Skip non-relevant files
            if item.name.startswith("."):
                continue
            if item.suffix in ("-shm", "-wal", "-journal"):
                continue
            if item.name.endswith(".provenance.json"):
                continue

            if item.is_dir():
                # Recurse into subdirectories (for multi-profile Chrome DBs)
                sub_dest = dest_dir / item.name
                sub_dest.mkdir(exist_ok=True)
                self._process_catalog_entry(item, sub_dest, user_mapping, result)
            elif item.is_file() and self._is_sqlite_file(item):
                # Process SQLite database file
                dest_db = dest_dir / self._sanitize_name(item.name, user_mapping)
                try:
                    self._create_empty_shell(item, dest_db, db_name=item.name)
                except RuntimeError as e:
                    # Data clearing failed - do NOT include this database (privacy risk)
                    result.errors.append(f"SKIPPED {item.name}: {e}")
                    if dest_db.exists():
                        dest_db.unlink()  # Remove partially-cleared database
                except Exception as e:
                    result.warnings.append(f"Could not create shell for {item.name}: {e}")
                    if dest_db.exists():
                        dest_db.unlink()  # Remove failed database

    def _is_sqlite_file(self, file_path: Path) -> bool:
        """Check if a file is a SQLite database by reading its header.

        Args:
            file_path: Path to file to check

        Returns:
            True if file has SQLite magic header
        """
        # SQLite files start with "SQLite format 3\x00"
        sqlite_magic = b"SQLite format 3\x00"

        try:
            with file_path.open("rb") as f:
                header = f.read(16)
                return header.startswith(sqlite_magic)
        except Exception:
            return False

    def _copy_catalog_yaml(self, package_dir: Path) -> None:
        """Copy artifact_recovery_catalog.yaml to package.

        The catalog YAML contains database definitions (paths, scopes, etc.)
        that are needed for matching. Paths are template paths, not actual
        user paths, so they don't need sanitization.

        Args:
            package_dir: Package directory to copy to
        """
        # Find catalog YAML - could be in project root or catalog directory
        catalog_yaml = self.exemplar_dir.parent.parent / "artifact_recovery_catalog.yaml"

        if not catalog_yaml.exists():
            # Try alternative location
            catalog_yaml = Path(__file__).parent.parent / "catalog" / "artifact_recovery_catalog.yaml"

        if catalog_yaml.exists():
            shutil.copy2(catalog_yaml, package_dir / "artifact_recovery_catalog.yaml")


class ExemplarImporter:
    """Import external exemplar packages for candidate scanning."""

    def __init__(self, imports_dir: Path):
        """Initialize importer.

        Args:
            imports_dir: Directory where imported packages are stored
                         (e.g., project/imports/)
        """
        self.imports_dir = imports_dir

    def import_package(self, package_path: Path) -> dict:
        """Extract and register exemplar package.

        Args:
            package_path: Path to exemplar package folder

        Returns:
            Dict with import result:
                success: bool
                manifest: PackageManifest (if successful)
                error: str (if failed)
        """
        result = {"success": False, "manifest": None, "error": None}

        # Validate package structure
        if not package_path.is_dir():
            result["error"] = "Package path is not a directory"
            return result

        manifest_path = package_path / "manifest.json"
        if not manifest_path.exists():
            result["error"] = "Invalid package: manifest.json not found"
            return result

        # Read and validate manifest
        try:
            with manifest_path.open() as f:
                manifest_data = json.load(f)
            manifest = PackageManifest.from_dict(manifest_data)
        except Exception as e:
            result["error"] = f"Could not read manifest: {e}"
            return result

        # Check format version compatibility
        if manifest.format_version != PACKAGE_FORMAT_VERSION:
            result["error"] = (
                f"Incompatible format version: {manifest.format_version} (expected {PACKAGE_FORMAT_VERSION})"
            )
            return result

        # Validate required directories
        schemas_dir = package_path / "schemas"
        if not schemas_dir.exists():
            result["error"] = "Invalid package: schemas/ directory not found"
            return result

        # Create imports directory if needed
        self.imports_dir.mkdir(parents=True, exist_ok=True)

        # Copy package to imports directory
        dest_dir = self.imports_dir / package_path.name

        if dest_dir.exists():
            result["error"] = f"Package already imported: {package_path.name}"
            return result

        try:
            shutil.copytree(package_path, dest_dir)
            result["success"] = True
            result["manifest"] = manifest
        except Exception as e:
            result["error"] = f"Could not import package: {e}"
            # Clean up partial import
            if dest_dir.exists():
                shutil.rmtree(dest_dir, ignore_errors=True)

        return result

    def list_imported(self) -> list[dict]:
        """List all imported exemplar packages.

        Returns:
            List of dicts with package info:
                name: str (folder name)
                manifest: PackageManifest
                path: Path
        """
        packages = []

        if not self.imports_dir.exists():
            return packages

        for package_dir in self.imports_dir.iterdir():
            if not package_dir.is_dir():
                continue

            manifest_path = package_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                with manifest_path.open() as f:
                    manifest_data = json.load(f)
                manifest = PackageManifest.from_dict(manifest_data)

                packages.append(
                    {
                        "name": package_dir.name,
                        "manifest": manifest,
                        "path": package_dir,
                    }
                )
            except Exception:
                # Skip invalid packages
                continue

        return packages

    def get_schemas_dir(self, package_name: str) -> Path | None:
        """Get schemas directory for an imported exemplar.

        Args:
            package_name: Name of imported package folder

        Returns:
            Path to schemas directory, or None if not found
        """
        schemas_dir = self.imports_dir / package_name / "schemas"
        return schemas_dir if schemas_dir.exists() else None

    def get_catalog_dir(self, package_name: str) -> Path | None:
        """Get catalog directory (empty DB shells) for an imported exemplar.

        Args:
            package_name: Name of imported package folder

        Returns:
            Path to catalog directory, or None if not found
        """
        catalog_dir = self.imports_dir / package_name / "catalog"
        return catalog_dir if catalog_dir.exists() else None

    def delete_imported(self, package_name: str) -> bool:
        """Delete an imported exemplar package.

        Args:
            package_name: Name of imported package folder

        Returns:
            True if deleted, False if not found
        """
        package_dir = self.imports_dir / package_name
        if package_dir.exists():
            cleanup_sqlite_directory(package_dir)
            return True
        return False
