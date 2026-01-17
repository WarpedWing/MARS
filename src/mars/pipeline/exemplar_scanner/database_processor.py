#!/usr/bin/env python3
"""
Database Processor for Exemplar Scanner

Handles individual database processing operations including file routing,
metadata collection, and provenance tracking.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from mars.pipeline.exemplar_scanner.database_scanner import (
    scan_custom_path,
    scan_database_definition,
)
from mars.utils.database_utils import (
    is_encrypted_database,
    is_sqlite_database,
)
from mars.utils.file_utils import compute_md5_hash

if TYPE_CHECKING:
    from pathlib import Path
    from threading import Lock

    from mars.pipeline.mount_utils.dfvfs_exporter import ExportRecord
    from mars.pipeline.output.structure import OutputStructure


class DatabaseProcessor:
    """Processes individual database files during exemplar scanning."""

    def __init__(
        self,
        output: OutputStructure,
        source_path: Path,
        dfvfs_manifest: dict[Path, ExportRecord],
        metadata_lock: Lock,
        found_databases: list[dict],
        failed_databases: list[dict],
    ):
        """
        Initialize the database processor.

        Args:
            output: OutputStructure instance for file operations
            source_path: Path to source filesystem/image
            dfvfs_manifest: Mapping of exported paths to dfVFS records
            metadata_lock: Thread lock for metadata updates
            found_databases: List to track successfully processed databases
            failed_databases: List to track processing failures
        """
        self.output = output
        self.source_path = source_path
        self._dfvfs_manifest = dfvfs_manifest
        self._metadata_lock = metadata_lock
        self.found_databases = found_databases
        self.failed_databases = failed_databases

    def process_database_safe(self, db_path: Path, db_def: dict[str, Any], category: str):
        """
        Thread-safe wrapper for processing a database.

        Args:
            db_path: Path to database file
            db_def: Database definition from catalog
            category: Category name
        """
        db_name = db_def.get("name", db_path.name)

        try:
            self.process_database(db_path, db_def, category)
        except Exception as e:
            # Thread-safe append to failed list
            self.failed_databases.append(
                {
                    "name": db_name,
                    "path": str(db_path),
                    "category": category,
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )
            raise

    def scan_database_definition(self, db_def: dict[str, Any], category: str):
        """Scan for a database based on catalog definition.

        Delegated to database_scanner.scan_database_definition().
        """
        scan_database_definition(
            db_def,
            category,
            self.source_path,
            self.process_database,
            self.failed_databases,
        )

    def scan_custom_path(self, glob_pattern: str):
        """Scan a custom path pattern not in the catalog.

        Delegated to database_scanner.scan_custom_path().
        """
        scan_custom_path(
            glob_pattern,
            self.source_path,
            self.process_database,
            self.failed_databases,
        )

    def process_database(self, db_path: Path, db_def: dict[str, Any], category: str):
        """
        Process a found database: copy, hash, generate schema/rubric.

        Args:
            db_path: Path to the database file
            db_def: Database definition from catalog
            category: Category name
        """
        db_name = db_def.get("name", db_path.name)
        export_meta = self._dfvfs_manifest.get(db_path)
        original_filename = db_path.name

        # Determine folder name for storage (use catalog name to avoid collisions)
        storage_folder_name = db_name  # e.g., "Safari History", "Chrome History"

        # Extract username and browser profile information from path
        # IMPORTANT: Use virtual_path from export metadata when available (e.g., from dfVFS)
        # This preserves the original path from the disk image, not the local workspace path
        username = None
        browser_profile = None
        if export_meta and export_meta.virtual_path:
            # Use the original path from the image (forward-slash separated)
            path_parts = tuple(export_meta.virtual_path.strip("/").split("/"))
        else:
            # Fallback to local path for non-image scans (e.g., live system, directory)
            path_parts = db_path.parts

        # Extract username for user-scope databases
        if db_def.get("scope") == "user" and "Users" in path_parts:
            user_idx = path_parts.index("Users")
            if user_idx + 1 < len(path_parts):
                username = path_parts[user_idx + 1]
                storage_folder_name = f"{db_name}_{username}"

        # Extract browser profile name (Default, Profile 1, Profile 2, etc.)
        # Look for Chrome, Edge, Firefox profile indicators in path
        for i, part in enumerate(path_parts):
            if part in ("Chrome", "Edge"):
                # Next directory after browser name is usually the profile
                # Chrome/Edge profiles: Default, Profile 1, Profile 2, Guest Profile, System Profile
                if i + 1 < len(path_parts):
                    potential_profile = path_parts[i + 1]
                    # Check if it looks like a profile name (not a subfolder like "databases")
                    if potential_profile.startswith(("Default", "Profile", "Guest", "System")):
                        browser_profile = potential_profile
                        break
            # Firefox uses Profiles subdirectory with random profile names
            elif part == "Profiles" and i + 1 < len(path_parts):
                browser_profile = path_parts[i + 1]
                break

        # For multi_profile databases (browsers with multiple profiles),
        # use nested folder structure: {db_name}_{username}/{profile}/
        # This keeps profiles separate while allowing meta-rubric generation
        profile_subfolder = None
        if db_def.get("multi_profile") and browser_profile:
            # Sanitize profile name for filesystem safety
            profile_subfolder = browser_profile.replace("/", "_").replace("\\", "_")

        # Sanitize folder name for filesystem safety
        storage_folder_name = storage_folder_name.replace("/", "_").replace("\\", "_")

        # If this is an archive file, put it in Archive subfolder
        archive_subfolder = None
        if db_def.get("is_archive_file"):
            archive_type = db_def.get("archive_type", "Archive")
            archive_subfolder = archive_type  # e.g., "Archives", "Quarantine"

        # Handle permission errors gracefully
        try:
            # Compute hash
            md5_hash = export_meta.md5 if export_meta else compute_md5_hash(db_path)

            # Check if database is encrypted FIRST (before routing)
            is_encrypted = is_encrypted_database(db_path)

            # Route files based on encryption status and file_type:
            file_type = db_def.get("file_type", "database")

            # Common arguments for all copy_to_* calls
            copy_base_kwargs = {
                "folder_name": storage_folder_name,
                "subfolder": archive_subfolder,
                "profile_subfolder": profile_subfolder,
            }

            if is_encrypted:
                # Encrypted database - route to encrypted/ folder
                original_copy = self.output.copy_to_encrypted(
                    db_path,
                    original_filename,
                    **copy_base_kwargs,
                )
            elif file_type == "log":
                # Text log file - route to logs/
                # Check if we should preserve directory structure
                preserve_structure = db_def.get("preserve_structure", False)
                base_dir_str = None
                virtual_path_str = None

                if preserve_structure:
                    # Calculate base directory by counting segments before recursive glob.
                    # Pattern like 'Users/*/Library/.../Profiles/**/*' contains wildcards.
                    # We find where ** starts, then extract that many segments from the
                    # actual virtual_path (which has resolved usernames, not wildcards).
                    catalog_path = db_def.get("glob_pattern", "")
                    if not catalog_path:
                        # Try glob_patterns (list) if glob_pattern not found
                        patterns = db_def.get("glob_patterns", [])
                        if patterns:
                            catalog_path = patterns[0] if isinstance(patterns, list) else patterns

                    # Use virtual path from dfVFS if available (for forensic images)
                    # Otherwise use the actual file path (for directory scans)
                    if export_meta:
                        virtual_path_str = export_meta.virtual_path

                    if catalog_path:
                        # Count path segments before the recursive glob (**)
                        pattern_parts = catalog_path.split("/")
                        base_segment_count = len(pattern_parts)

                        for i, part in enumerate(pattern_parts):
                            if "**" in part:
                                # Found recursive glob - base is everything before
                                base_segment_count = i
                                break

                        # Also handle trailing single * (e.g., 'foo/bar/*')
                        if base_segment_count == len(pattern_parts) and pattern_parts and pattern_parts[-1] == "*":
                            base_segment_count -= 1

                        # Extract that many segments from the actual virtual path.
                        # This resolves wildcards to concrete values.
                        path_to_use = virtual_path_str or str(db_path)

                        # Normalize firmlink paths for base_dir calculation.
                        # macOS firmlinks create duplicate paths like:
                        #   /System/Volumes/Data/private/var/... and /private/var/...
                        # The catalog pattern uses the canonical form (without firmlink prefix),
                        # so we need to strip the prefix to get correct segment counts.
                        firmlink_prefix = "/System/Volumes/Data"
                        if path_to_use.startswith(firmlink_prefix):
                            path_to_use = path_to_use[len(firmlink_prefix) :]

                        if base_segment_count > 0 and path_to_use:
                            path_parts = path_to_use.split("/")
                            # Handle leading empty part from absolute paths
                            if path_parts and path_parts[0] == "":
                                path_parts = path_parts[1:]
                            base_dir_str = "/".join(path_parts[:base_segment_count])

                # Extend base kwargs with log-specific arguments
                log_kwargs = {
                    **copy_base_kwargs,
                    "preserve_structure": preserve_structure,
                    "base_dir": base_dir_str,
                    "virtual_path": virtual_path_str,
                }
                original_copy = self.output.copy_to_logs(
                    db_path,
                    original_filename,
                    **log_kwargs,
                )
            elif file_type == "keychain":
                # Keychain file - route to keychains/
                original_copy = self.output.copy_to_keychains(
                    db_path,
                    original_filename,
                    **copy_base_kwargs,
                )
            elif file_type == "cache":
                # Binary cache file - route to caches/ (non-database artifacts with potential forensic value)
                original_copy = self.output.copy_to_caches(
                    db_path,
                    original_filename,
                    **copy_base_kwargs,
                )
            else:
                # Database file (default) - route to originals/
                original_copy = self.output.copy_to_originals(
                    db_path,
                    original_filename,
                    **copy_base_kwargs,
                )

            # Thread-safe metadata update
            if export_meta:
                with self._metadata_lock:
                    if self.output.metadata.get("databases_processed"):
                        processed_entry = self.output.metadata["databases_processed"][-1]
                        processed_entry["source_path"] = export_meta.virtual_path
                        processed_entry["virtual_path"] = export_meta.virtual_path
                # Note: Metadata saved at end of scan to avoid race conditions
        except PermissionError as e:
            raise PermissionError("Permission denied accessing file") from e

        # Check if it's a SQLite database AFTER copying
        # This determines whether to generate schema, not where to copy
        is_sqlite = is_sqlite_database(db_path)

        file_stat = db_path.stat()
        file_size = export_meta.size if export_meta else file_stat.st_size

        # Extract ORIGINAL file timestamps (not MARS processing time)
        # Use dfVFS metadata when available, fallback to os.stat
        if export_meta and export_meta.file_created:
            file_created = export_meta.file_created
        else:
            birth_ts = getattr(file_stat, "st_birthtime", file_stat.st_ctime)
            file_created = datetime.fromtimestamp(birth_ts, UTC)

        if export_meta and export_meta.file_modified:
            file_modified = export_meta.file_modified
        else:
            file_modified = datetime.fromtimestamp(file_stat.st_mtime, UTC)

        if export_meta and export_meta.file_accessed:
            file_accessed = export_meta.file_accessed
        else:
            file_accessed = datetime.fromtimestamp(file_stat.st_atime, UTC)

        source_virtual_path = export_meta.virtual_path if export_meta else str(db_path)
        relative_virtual_path = (
            export_meta.virtual_path.lstrip("/") if export_meta else str(db_path.relative_to(self.source_path))
        )

        # Create provenance data
        provenance_data = {
            "name": db_name,
            "category": category,
            "source_path": source_virtual_path,
            "relative_path": relative_virtual_path,
            "md5": md5_hash,
            "file_size": file_size,
            # Original file timestamps (from source, not MARS processing time)
            "file_created": file_created.isoformat(),
            "file_modified": file_modified.isoformat(),
            "file_accessed": file_accessed.isoformat(),
            # MARS processing timestamp
            "processed_at": datetime.now(UTC).isoformat(),
            "description": db_def.get("description", ""),
            "username": username,  # macOS username (if user-scope)
            "browser_profile": browser_profile,  # Browser profile name (Default, Profile 1, etc.)
            "is_sqlite": is_sqlite,  # Track file type in provenance
        }

        # If not SQLite, skip schema/rubric generation and individual provenance file
        # (logs/caches/keychains get consolidated provenance files instead)
        if not is_sqlite:
            # Determine actual location based on encryption status and file_type
            if is_encrypted:
                location = "encrypted"
            elif file_type == "log":
                location = "logs"
            elif file_type == "cache":
                location = "caches"
            elif file_type == "keychain":
                location = "keychains"
            else:
                location = "originals"

            # Check if it's encrypted (use previously checked value)
            if is_encrypted:
                # Database is encrypted - skip schema generation
                self.found_databases.append(
                    {
                        **provenance_data,
                        "is_sqlite": False,
                        "is_encrypted": True,
                        "schema_generated": False,
                        "rubric_generated": False,
                        "skip_reason": "encrypted",
                        "location": location,
                        "file_type": file_type,
                    }
                )
            else:
                # Log files, caches, and keychains are expected to be non-SQLite
                # Non-SQLite file processed and routed appropriately
                self.found_databases.append(
                    {
                        **provenance_data,
                        "is_sqlite": False,
                        "is_encrypted": False,
                        "schema_generated": False,
                        "rubric_generated": False,
                        "skip_reason": (
                            "log_file"
                            if file_type == "log"
                            else (
                                "cache_file"
                                if file_type == "cache"
                                else ("keychain_file" if file_type == "keychain" else "not_sqlite")
                            )
                        ),
                        "location": location,
                        "file_type": file_type,
                    }
                )
            return

        # Skip schema generation in Phase 1-2 for ALL databases
        # Schemas will be generated in Phase 3 or Phase 4 with proper organization:
        # - Phase 3: System-scope databases with archives (combined)
        # - Phase 4: All other databases (auto-catalog with version detection)
        #
        # This ensures:
        # 1. No duplicate schemas (Phase 1-2 + Phase 4)
        # 2. Proper versioning for multi-schema databases
        # 3. Schema count matches catalog count

        # Determine which phase will process this database
        if db_def.get("has_archives", False) and db_def.get("scope", "system") == "system":
            phase = "Phase 3 (archive combining)"
        else:
            phase = "Phase 4 (cataloging)"

        # Create individual provenance file for SQLite databases
        # (logs/caches/keychains get consolidated provenance files instead)
        self.output.create_provenance_file(original_copy, original_copy.name, provenance_data)

        # Determine location for SQLite databases
        location = "encrypted" if is_encrypted else "originals"

        # Schema will be generated later in Phase 3 or Phase 4
        self.found_databases.append(
            {
                **provenance_data,
                "is_sqlite": True,
                "is_encrypted": is_encrypted,
                "schema_generated": False,
                "rubric_generated": False,
                "location": location,
                "note": f"Schema/rubric will be generated in {phase}",
            }
        )
