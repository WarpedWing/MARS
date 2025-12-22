#!/usr/bin/env python3
"""
Exemplar Cataloger

Catalogs databases into the exemplar catalog structure during exemplar scanning.

This module manages:
- Phase 3: Combining databases with archives (has_archives=True)
- Phase 4: Auto-detecting and cataloging SQLite files

Note: Low-level database merging is handled by mars.pipeline.output.database_combiner.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mars.pipeline.exemplar_scanner.schema_generator import (
    generate_schema_and_rubric,
    get_schema_fingerprint,
)
from mars.pipeline.output.database_combiner import (
    combine_database_group,
    merge_sqlite_databases,
)
from mars.utils.database_utils import (
    copy_database_with_auxiliary_files,
    has_effective_tables,
    is_database_empty_or_null,
    is_sqlite_database,
)
from mars.utils.debug_logger import logger
from mars.utils.file_utils import MKDIR_KWARGS

if TYPE_CHECKING:
    from collections.abc import Callable

    from rich.console import Group

    from mars.config import MARSConfig
    from mars.pipeline.exemplar_scanner.cleanup_manager import (
        CleanupManager,
    )
    from mars.pipeline.output.structure import OutputStructure


class ExemplarCataloger:
    """Catalogs databases into the exemplar catalog structure during exemplar scanning."""

    def __init__(
        self,
        output: OutputStructure,
        config: MARSConfig,
        cleanup_manager: CleanupManager,
        unique_schema_dirs: set[Path],
        generated_schemas: list[Path],
        generated_rubrics: list[Path],
        combined_databases: list[dict],
    ):
        """
        Initialize the database combiner.

        Args:
            output: OutputStructure instance for paths and directory operations
            config: MARSConfig instance (provides should_ignore_file)
            cleanup_manager: Cleanup manager for post-processing cleanup
            unique_schema_dirs: Set to track unique schema directories
            generated_schemas: List to track generated schema files
            generated_rubrics: List to track generated rubric files
            combined_databases: List to track combined database records
        """
        self.output = output
        self.config = config
        self.cleanup_manager = cleanup_manager
        self.unique_schema_dirs = unique_schema_dirs
        self.generated_schemas = generated_schemas
        self.generated_rubrics = generated_rubrics
        self.combined_databases = combined_databases

    def _get_database_row_count(self, db_path: Path) -> int:
        """
        Get total row count across all tables in database.

        Args:
            db_path: Path to SQLite database file

        Returns:
            Total row count across all tables (excluding sqlite_* tables),
            or 0 if database cannot be read
        """
        try:
            # Use read-only immutable mode to avoid recreating WAL/SHM files
            with sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = [row[0] for row in cursor.fetchall()]
                total = 0
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM [{table}]")
                        total += cursor.fetchone()[0]
                    except Exception:  # noqa: S110
                        pass
                return total
        except Exception:  # noqa: S110
            return 0

    def _get_catalog_base_name(self, db_folder: Path) -> str | None:
        """
        Get the true catalog base name from any provenance file in the folder.

        Individual provenance files (created by database_processor.py) contain
        the correct catalog name in their "name" field. This is the authoritative
        source for the base name, as it comes directly from artifact_recovery_catalog.yaml.

        Args:
            db_folder: Database folder in originals/ or catalog/

        Returns:
            Catalog base name (e.g., "Chrome Cookies") or None if not found
        """
        # Search for any .provenance.json file in the folder tree
        for prov_file in db_folder.rglob("*.provenance.json"):
            try:
                with prov_file.open() as f:
                    prov_data = json.load(f)
                # Individual provenance files have the correct "name" from catalog
                if "name" in prov_data:
                    return prov_data["name"]
            except Exception:  # noqa: S110
                continue
        return None

    def _deduplicate_by_md5(self, db_files: list[Path]) -> list[Path]:
        """
        Deduplicate database files by MD5 hash from provenance.

        macOS firmlinks can cause the same file to be discovered via multiple
        paths (e.g., /private/var/... and /System/Volumes/Data/private/var/...).
        This filters duplicates by checking MD5 hashes in provenance files.

        When duplicates are found, keeps the file with the shorter source path
        (typically the /private/var/... variant) as it's the canonical path.

        Args:
            db_files: List of database file paths

        Returns:
            Deduplicated list of database files
        """
        if len(db_files) <= 1:
            return db_files

        # Map MD5 -> (db_file, source_path_length)
        md5_to_file: dict[str, tuple[Path, int]] = {}

        for db_file in db_files:
            # Find matching provenance file
            prov_file = db_file.parent / f"{db_file.stem}.provenance.json"
            if not prov_file.exists():
                # No provenance - keep the file (can't deduplicate)
                # Use empty string as unique key
                md5_to_file[f"_no_prov_{db_file}"] = (db_file, 0)
                continue

            try:
                with prov_file.open() as f:
                    prov_data = json.load(f)
                md5 = prov_data.get("md5")
                source_path = prov_data.get("source_path", "")

                if not md5:
                    # No MD5 in provenance - keep the file
                    md5_to_file[f"_no_md5_{db_file}"] = (db_file, 0)
                    continue

                # Check if we've seen this MD5 before
                if md5 in md5_to_file:
                    existing_file, existing_len = md5_to_file[md5]
                    # Keep the one with shorter source path (canonical path)
                    if len(source_path) < existing_len:
                        # This one has shorter path - replace
                        logger.debug(f"Dedup: keeping {db_file.name} (shorter path) over {existing_file.name}")
                        md5_to_file[md5] = (db_file, len(source_path))
                    else:
                        logger.debug(f"Dedup: skipping {db_file.name} (duplicate of {existing_file.name})")
                else:
                    md5_to_file[md5] = (db_file, len(source_path))

            except Exception:  # noqa: S110
                # Error reading provenance - keep the file
                md5_to_file[f"_error_{db_file}"] = (db_file, 0)

        # Return deduplicated list, preserving original order for determinism
        deduped_files = [f for f, _ in md5_to_file.values()]
        original_order = {f: i for i, f in enumerate(db_files)}
        return sorted(deduped_files, key=lambda f: original_order.get(f, len(db_files)))

    def _collect_source_provenances(
        self,
        source_dbs: list[Path],
        extra_fields_fn: Callable[[Path], dict] | None = None,
    ) -> list[dict]:
        """
        Collect provenance information from source database files.

        For each source database, attempts to read its .provenance.json file
        and extract metadata. Handles missing or unreadable provenance gracefully.

        Args:
            source_dbs: List of source database paths
            extra_fields_fn: Optional callable that takes a Path and returns dict
                           of additional fields to include in each provenance entry

        Returns:
            List of provenance dictionaries
        """
        source_provenances = []
        for source_db in source_dbs:
            source_prov_file = source_db.parent / f"{source_db.stem}.provenance.json"

            # Base fields present in all cases
            base_fields = {
                "filename": source_db.name,
                "location": str(source_db.relative_to(self.output.root)),
            }

            # Add extra fields if callback provided
            if extra_fields_fn:
                base_fields.update(extra_fields_fn(source_db))

            if source_prov_file.exists():
                try:
                    with source_prov_file.open("r") as f:
                        source_prov = json.load(f)
                    source_provenances.append(
                        {
                            **base_fields,
                            "original_source": source_prov.get("source_path"),
                            "md5": source_prov.get("md5"),
                            "file_size": source_prov.get("file_size"),
                        }
                    )
                except Exception:
                    # If we can't read provenance, use base fields only
                    source_provenances.append(base_fields)
            else:
                # No provenance file, use base fields only
                source_provenances.append(base_fields)

        return source_provenances

    def _build_combined_provenance(
        self,
        output_db: Path,
        source_provenances: list[dict],
        stats: dict,
        operation: str,
        name: str,
        source_count: int,
        combine_strategy: str,
        extra_fields: dict | None = None,
    ) -> dict:
        """
        Build combined provenance dictionary.

        Args:
            output_db: Path to combined output database
            source_provenances: List of source database provenances
            stats: Combination statistics dict
            operation: Operation type (e.g., "combine_archives", "combine_databases")
            name: Display name for the combined database
            source_count: Number of source databases
            combine_strategy: Strategy used for combining
            extra_fields: Optional additional fields to include

        Returns:
            Combined provenance dictionary
        """
        base_provenance = {
            "name": name,
            "operation": operation,
            "timestamp": datetime.now(UTC).isoformat(),
            "combined_filename": output_db.name,
            "combined_location": str(output_db.relative_to(self.output.root)),
            "source_count": source_count,
            "source_databases": source_provenances,
            "combine_strategy": combine_strategy,
            "total_rows_merged": stats.get("total_rows_merged", 0),
            "tables_merged": stats.get("tables_merged", 0),
            "case_name": self.output.case_name,
        }

        if extra_fields:
            base_provenance.update(extra_fields)

        return base_provenance

    def combine_databases(
        self,
        db_jobs: list[tuple[dict[str, Any], str]],
        panel_group: Group | None = None,
    ):
        """
        Combine databases with archives (Phase 3).

        For databases with has_archives=True and scope=system, combine all
        instances (primary + archives) into a single merged database.

        Args:
            db_jobs: List of (normalized_db_def, category_name) tuples
            panel_group: Optional Rich Group for progress updates in Live context
        """
        # Group databases by name and check if they need combining
        databases_to_combine = {}

        for db_def, category in db_jobs:
            # Combine if has_archives=True (databases with archive subfolders)
            has_archives = db_def.get("has_archives", False)

            if not has_archives:
                continue

            db_name = db_def.get("name", "Unknown")
            combine_strategy = db_def.get("combine_strategy", "decompress_and_merge")

            # decompress_only: skip combining, files are already decompressed in place
            # by workspace_manager.py
            if combine_strategy == "decompress_only":
                continue

            # Check if we have this database in our found_databases
            # Find all instances in originals/ directory
            db_folder = self.output.get_original_db_dir(db_name)

            if not db_folder.exists():
                # No originals folder
                continue

            # Collect all database files (primary + archives)
            all_db_files = []
            primary_file = None
            archive_files = []

            # Get primary file (not in Archives/ subfolder)
            # For has_archives, we take the first file as primary
            # Sort for deterministic order
            for file_path in sorted(db_folder.iterdir()):
                if file_path.is_file() and not self.config.should_ignore_file(file_path):
                    primary_file = file_path
                    all_db_files.append(file_path)
                    break

            # Get archive files from Archive subfolders
            archives_config = db_def.get("archives", [])
            for archive_def in archives_config:
                archive_type = archive_def.get("name", "Archive")
                archive_subfolder = db_folder / archive_type

                if archive_subfolder.exists() and archive_subfolder.is_dir():
                    # Sort for deterministic order
                    for archive_file in sorted(archive_subfolder.iterdir()):
                        if archive_file.is_file() and not self.config.should_ignore_file(archive_file):
                            archive_files.append(archive_file)
                            all_db_files.append(archive_file)

            if not all_db_files:
                # No database files found
                continue

            if len(all_db_files) == 1 and not archive_files:
                # Only primary file, no archives
                continue

            # Check if any database has effective tables
            has_effective = any(has_effective_tables(f) for f in all_db_files)
            if not has_effective:
                # No effective tables
                continue

            # Store for combining
            databases_to_combine[db_name] = {
                "db_def": db_def,
                "category": category,
                "primary": primary_file,
                "archives": archive_files,
                "all_files": all_db_files,
                "combine_strategy": combine_strategy,
            }

        if not databases_to_combine:
            # No databases require combining
            return

        # Process each database - sort for deterministic order
        total_dbs = len(databases_to_combine)
        for db_idx, (db_name, combine_info) in enumerate(sorted(databases_to_combine.items()), 1):
            # Update progress panel if available
            if panel_group:
                archive_count = len(combine_info["archives"])
                panel_group.renderables.pop()
                panel_group.renderables.append(
                    f" • Combining [bold]{db_name}[/bold] ({archive_count} archives) [{db_idx}/{total_dbs}]..."
                )

            catalog_dir = self.output.paths.exemplar_catalog / db_name
            output_db = catalog_dir / f"{db_name}.combined.db"

            try:
                # Create parent directory only (databases/catalog/)
                catalog_dir.parent.mkdir(**MKDIR_KWARGS)

                # Create directory for output database
                catalog_dir.mkdir(**MKDIR_KWARGS)

                # Combine databases
                stats = combine_database_group(
                    primary_db=(combine_info["primary"] if combine_info["primary"] else combine_info["all_files"][0]),
                    archive_dbs=combine_info["archives"],
                    output_db=output_db,
                    combine_strategy=combine_info["combine_strategy"],
                )

                if stats.get("status") == "success":
                    # Successfully combined primary and archive databases

                    # Create provenance file for archive-combined database (Phase 3)
                    try:
                        # Collect info from all source files (primary + archives)
                        source_provenances = self._collect_source_provenances(
                            combine_info["all_files"],
                            extra_fields_fn=lambda db: {
                                "is_primary": db == combine_info["primary"],
                                "is_archive": db in combine_info["archives"],
                            },
                        )

                        # Create combined provenance
                        combined_provenance = self._build_combined_provenance(
                            output_db=output_db,
                            source_provenances=source_provenances,
                            stats=stats,
                            operation="combine_archives",
                            name=db_name,
                            source_count=len(combine_info["all_files"]),
                            combine_strategy=combine_info["combine_strategy"],
                            extra_fields={
                                "primary_count": 1 if combine_info["primary"] else 0,
                                "archive_count": len(combine_info["archives"]),
                            },
                        )

                        # Write combined provenance file
                        combined_prov_path = catalog_dir / f"{output_db.stem}.provenance.json"
                        with combined_prov_path.open("w") as f:
                            json.dump(combined_provenance, f, indent=2)

                        # Provenance file created for combined database
                    except Exception:
                        # Could not create provenance file
                        pass

                    # Generate schema and rubric from combined database
                    schema_dir = self.output.get_schema_dir(db_name)

                    try:
                        schema_path, rubric_path = generate_schema_and_rubric(
                            output_db,
                            schema_dir,
                            f"{db_name}_combined",
                            min_timestamp_rows=self.config.exemplar.min_timestamp_rows,
                        )

                        # Track unique schema directory
                        self.unique_schema_dirs.add(schema_path.parent)
                        self.generated_schemas.append(schema_path)
                        self.generated_rubrics.append(rubric_path)

                        # Schema and rubric generated for combined database

                        # Track combined database
                        self.combined_databases.append(
                            {
                                "name": db_name,
                                "output_db": str(output_db),
                                "total_rows": stats["total_rows_merged"],
                                "total_tables": stats["tables_merged"],
                                "source_files": len(combine_info["all_files"]),
                                "schema_path": str(schema_path),
                                "rubric_path": str(rubric_path),
                                "combined_at": datetime.now(UTC).isoformat(),
                            }
                        )

                    except Exception as e:
                        # Schema/rubric generation failed

                        self.combined_databases.append(
                            {
                                "name": db_name,
                                "output_db": str(output_db),
                                "total_rows": stats["total_rows_merged"],
                                "total_tables": stats["tables_merged"],
                                "source_files": len(combine_info["all_files"]),
                                "schema_generated": False,
                                "generation_error": str(e),
                                "combined_at": datetime.now(UTC).isoformat(),
                            }
                        )
                else:
                    # Database combination failed
                    # Remove empty directory if combine failed
                    # Check if catalog_dir was successfully created and output_db does not exist
                    if catalog_dir.exists() and not output_db.exists():
                        with contextlib.suppress(OSError):
                            catalog_dir.rmdir()

            except Exception:
                # Failed to combine databases
                # Remove empty directory if combine failed and output DB was not created
                if catalog_dir.exists() and not output_db.exists():
                    with contextlib.suppress(OSError):
                        catalog_dir.rmdir()

        # Phase 3 combining complete

    def _detect_profile_subfolders(self, db_folder: Path) -> list[Path]:
        """
        Detect browser profile subfolders within a database folder.

        Profile subfolders are created for multi_profile databases (Chrome, Firefox, Edge)
        and contain the actual database files.

        Args:
            db_folder: Database folder in originals/

        Returns:
            List of profile subfolder paths, or empty list if no profiles detected
        """
        profile_subfolders = []

        # Check for direct subdirectories that look like browser profiles
        # Profile names: Default, Profile 1, Profile 2, Guest Profile, System Profile
        # Firefox: random profile names like abc123.default, xyz789.default-release
        for item in sorted(db_folder.iterdir()):
            if not item.is_dir():
                continue

            # Skip known non-profile subfolders
            if item.name in (
                "Archives",
                "Quarantine",
                "empty",
                "lost_and_found",
                "remnants",
            ):
                continue

            # Check if this subdirectory contains database files directly
            # (not another level of subdirectories)
            has_db_files = any(
                f.is_file() and not self.config.should_ignore_file(f) and is_sqlite_database(f)
                for f in item.iterdir()
                if f.is_file()
            )

            if has_db_files:
                profile_subfolders.append(item)

        return profile_subfolders

    def _process_multi_profile_folder(
        self,
        db_name: str,
        db_folder: Path,
        profile_subfolders: list[Path],
        panel_group: Group | None = None,
        catalog_base_name: str | None = None,
    ) -> bool:
        """
        Process a multi-profile database folder with nested structure.

        Creates:
        - Nested catalog structure: catalog/{db_name}/{profile}/ (or {db_name}_v1/{profile}/ for multiple schemas)
        - ONE schema/rubric per schema version from temporarily combined data (meta-rubric)
        - Discards combined DB after rubric generation

        Args:
            db_name: Name of the database folder (e.g., "Chrome Cookies_username")
            db_folder: Path to database folder in originals/
            profile_subfolders: List of profile subfolder paths
            panel_group: Optional Rich Group for progress updates
            catalog_base_name: True catalog name (e.g., "Chrome Cookies"). If None, uses db_name.

        Returns:
            True if successfully processed, False otherwise
        """
        # Use catalog_base_name if provided, otherwise fall back to db_name
        base_name_for_provenance = catalog_base_name or db_name
        if panel_group:
            panel_group.renderables.pop()
            panel_group.renderables.append(
                f" • Cataloging multi-profile: [bold]{db_name}[/bold] ({len(profile_subfolders)} profiles)"
            )

        # Collect all databases from all profiles, grouped by schema fingerprint
        # schema_groups: fingerprint -> list of (profile_folder, db_file)
        schema_groups: dict[str, list[tuple[Path, Path]]] = {}

        for profile_folder in profile_subfolders:
            # Collect SQLite files from this profile
            for item in sorted(profile_folder.iterdir()):
                if item.is_file() and not self.config.should_ignore_file(item) and is_sqlite_database(item):
                    fingerprint = get_schema_fingerprint(item)
                    if fingerprint:
                        if fingerprint not in schema_groups:
                            schema_groups[fingerprint] = []
                        schema_groups[fingerprint].append((profile_folder, item))

        if not schema_groups:
            return False

        # Process each schema version separately
        # If only one schema version, no version suffix needed
        # If multiple, add _v1, _v2, etc.
        num_versions = len(schema_groups)

        # Sort groups by size (descending) for version numbering
        sorted_groups = sorted(
            schema_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True,
        )

        for version_idx, (fingerprint, profile_dbs) in enumerate(sorted_groups, 1):
            # Check if all databases in this version are empty
            version_all_empty = all(is_database_empty_or_null(db_file)[0] for _, db_file in profile_dbs)

            # Build output name with version suffix if multiple versions
            if num_versions > 1:
                version_name = f"{db_name}_v{version_idx}"
                if version_all_empty:
                    version_name = f"{version_name}_empty"
            else:
                version_name = f"{db_name}_empty" if version_all_empty else db_name

            # Process this schema version
            success = self._process_multi_profile_version(
                version_name,
                base_name_for_provenance,  # Use true catalog name for provenance
                profile_dbs,
                panel_group,
            )

            if not success:
                # If any version fails, return False to fall back to standard processing
                return False

        return True

    def _process_multi_profile_version(
        self,
        version_name: str,
        base_name: str,
        profile_dbs: list[tuple[Path, Path]],
        panel_group: Group | None = None,
    ) -> bool:
        """
        Process a single schema version of a multi-profile database.

        Args:
            version_name: Name for this version (e.g., "Chrome Cookies_admin" or "Chrome Cookies_admin_v1")
            base_name: Base database name without version
            profile_dbs: List of (profile_folder, db_file) tuples with matching schema
            panel_group: Optional Rich Group for progress updates

        Returns:
            True if successfully processed, False otherwise
        """
        # Create nested catalog structure: catalog/{version_name}/{profile}/
        catalog_dir = self.output.paths.exemplar_catalog / version_name
        catalog_dir.mkdir(**MKDIR_KWARGS)

        # Create schema directory (ONE schema for this version's profiles)
        schema_dir = self.output.get_schema_dir(version_name)

        # Move each profile database to its nested folder in catalog
        profile_db_paths: list[Path] = []  # Track moved DB paths for combining
        source_provenances: list[dict] = []
        profile_names: list[str] = []  # Track profile names for provenance

        for profile_folder, db_file in profile_dbs:
            profile_names.append(profile_folder.name)
            profile_name = profile_folder.name

            # Create nested profile folder in catalog
            profile_catalog_dir = catalog_dir / profile_name
            profile_catalog_dir.mkdir(**MKDIR_KWARGS)

            # Move database to catalog with profile-specific name
            output_db = profile_catalog_dir / db_file.name
            copy_database_with_auxiliary_files(db_file, output_db)
            profile_db_paths.append(output_db)

            # Handle provenance file
            source_provenance = db_file.parent / f"{db_file.stem}.provenance.json"
            if source_provenance.exists():
                try:
                    with source_provenance.open("r") as f:
                        provenance_data = json.load(f)

                    # Create history entry
                    history_entry = {
                        "operation": "move_to_catalog_profile",
                        "timestamp": datetime.now(UTC).isoformat(),
                        "original_filename": db_file.name,
                        "new_filename": output_db.name,
                        "original_location": str(db_file.relative_to(self.output.root)),
                        "new_location": str(output_db.relative_to(self.output.root)),
                        "profile": profile_name,
                        "reason": "Multi-profile database cataloged with nested structure",
                    }

                    if "history" not in provenance_data:
                        provenance_data["history"] = []
                    provenance_data["history"].append(history_entry)

                    provenance_data["catalog_filename"] = output_db.name
                    provenance_data["catalog_location"] = str(output_db.relative_to(self.output.root))
                    provenance_data["browser_profile"] = profile_name

                    # Write updated provenance to catalog profile folder
                    catalog_provenance = profile_catalog_dir / f"{output_db.stem}.provenance.json"
                    with catalog_provenance.open("w") as f:
                        json.dump(provenance_data, f, indent=2)

                    # Collect for meta-provenance
                    source_provenances.append(
                        {
                            "filename": db_file.name,
                            "profile": profile_name,
                            "location": str(output_db.relative_to(self.output.root)),
                            "original_source": provenance_data.get("source_path"),
                            "md5": provenance_data.get("md5"),
                            "file_size": provenance_data.get("file_size"),
                        }
                    )

                    # Remove original provenance from originals/
                    source_provenance.unlink()
                except Exception:
                    # Collect basic info without provenance details
                    source_provenances.append(
                        {
                            "filename": db_file.name,
                            "profile": profile_name,
                            "location": str(output_db.relative_to(self.output.root)),
                        }
                    )

            # Remove original database and auxiliary files
            try:
                db_file.unlink()
                for suffix in ["-shm", "-wal", "-journal"]:
                    aux_file = db_file.parent / f"{db_file.name}{suffix}"
                    if aux_file.exists():
                        aux_file.unlink()
            except Exception:
                pass

        # Now generate meta-rubric from temporarily combined data
        # The combined database is TEMPORARY - only used for richer statistics
        import tempfile

        temp_combined_path = None
        try:
            # Create temporary file for combined database
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as temp_combined:
                temp_combined_path = Path(temp_combined.name)

            # Sort profile DBs by row count to ensure fullest is primary
            sorted_profile_dbs = sorted(
                profile_db_paths,
                key=lambda f: (self._get_database_row_count(f), f.name),
                reverse=True,
            )

            # Combine all profile databases temporarily
            stats = merge_sqlite_databases(
                source_dbs=sorted_profile_dbs,
                output_db=temp_combined_path,
                combine_strategy="insert_or_ignore",
            )

            # Generate schema and rubric from combined data
            schema_path, rubric_path = generate_schema_and_rubric(
                temp_combined_path,
                schema_dir,
                f"{version_name}_combined",
                min_timestamp_rows=self.config.exemplar.min_timestamp_rows,
            )

            # Track unique schema directory
            self.unique_schema_dirs.add(schema_path.parent)
            self.generated_schemas.append(schema_path)
            self.generated_rubrics.append(rubric_path)

            # Create meta-provenance file in catalog root
            meta_provenance = {
                "name": version_name,
                "base_name": base_name,
                "operation": "multi_profile_catalog",
                "timestamp": datetime.now(UTC).isoformat(),
                "profile_count": len(profile_dbs),
                "profiles": profile_names,
                "source_databases": source_provenances,
                "schema_path": str(schema_path.relative_to(self.output.root)),
                "rubric_path": str(rubric_path.relative_to(self.output.root)),
                "rubric_generation": "meta-rubric from temporarily combined profile data",
                "total_rows_in_combined": stats.get("total_rows_merged", 0),
                "tables_merged": stats.get("tables_merged", 0),
                "case_name": self.output.case_name,
            }

            meta_prov_path = catalog_dir / f"{version_name}.provenance.json"
            with meta_prov_path.open("w") as f:
                json.dump(meta_provenance, f, indent=2)

            # Track cataloged database
            self.combined_databases.append(
                {
                    "name": version_name,
                    "base_name": base_name,
                    "output_dir": str(catalog_dir),
                    "is_multi_profile": True,
                    "profile_count": len(profile_dbs),
                    "profiles": profile_names,
                    "total_rows": stats.get("total_rows_merged", 0),
                    "total_tables": stats.get("tables_merged", 0),
                    "source_files": len(profile_dbs),
                    "schema_path": str(schema_path),
                    "rubric_path": str(rubric_path),
                    "cataloged_at": datetime.now(UTC).isoformat(),
                    "auto_detected": True,
                }
            )

            return True

        except Exception as e:
            # Schema/rubric generation failed - still record the attempt
            self.combined_databases.append(
                {
                    "name": version_name,
                    "base_name": base_name,
                    "output_dir": str(catalog_dir),
                    "is_multi_profile": True,
                    "profile_count": len(profile_dbs),
                    "profiles": profile_names,
                    "source_files": len(profile_dbs),
                    "schema_generated": False,
                    "generation_error": str(e),
                    "cataloged_at": datetime.now(UTC).isoformat(),
                    "auto_detected": True,
                }
            )
            return False

        finally:
            # Clean up temporary combined database (IMPORTANT: always discard)
            if temp_combined_path and temp_combined_path.exists():
                try:
                    temp_combined_path.unlink()
                    # Also clean up any auxiliary files
                    for suffix in ["-shm", "-wal", "-journal"]:
                        aux = temp_combined_path.parent / f"{temp_combined_path.name}{suffix}"
                        if aux.exists():
                            aux.unlink()
                except Exception:
                    pass

    def auto_combine_multiple_files(self, panel_group: Group | None = None):
        """
        Auto-detect and catalog SQLite files (Phase 4).

        All working databases are cataloged to databases/catalog/
        - Multi-profile databases → catalog/{name}/{profile}/ with meta-rubric
        - Combined databases (2+ with matching schema) → catalog/{name}_v1/
        - Single databases (1 with unique schema) → catalog/{name}_v2/
        - Empty databases moved to originals/{name}/empty/
        """

        # Scan all folders in originals/
        if not self.output.paths.exemplar_originals.exists():
            # No originals directory found
            return

        folders_to_combine = []

        # Sort for deterministic order
        for db_folder in sorted(self.output.paths.exemplar_originals.iterdir()):
            if not db_folder.is_dir():
                continue

            folder_name = db_folder.name  # Folder name (e.g., "Chrome Cookies_username")
            # Get catalog base name from provenance files (e.g., "Chrome Cookies")
            catalog_base_name = self._get_catalog_base_name(db_folder) or folder_name

            # Update progress panel
            if panel_group:
                panel_group.renderables.pop()
                panel_group.renderables.append(f" • Cataloging: [bold]{folder_name}[/bold]")

            # Skip if already cataloged in Phase 3 (check if catalog output exists)
            catalog_output = self.output.paths.exemplar_catalog / folder_name / f"{folder_name}.combined.db"
            if catalog_output.exists():
                continue

            # Check for multi-profile structure (nested browser profile folders)
            # This handles Chrome, Firefox, Edge databases with multiple profiles
            profile_subfolders = self._detect_profile_subfolders(db_folder)
            if profile_subfolders and self._process_multi_profile_folder(
                folder_name,
                db_folder,
                profile_subfolders,
                panel_group,
                catalog_base_name=catalog_base_name,
            ):
                # Successfully processed, continue to next folder
                continue
            # If no profile subfolders or processing failed, fall through to standard processing

            # Collect all SQLite files in this folder (and subfolders)
            # IMPORTANT: Sort for deterministic iteration order
            sqlite_files = [
                item
                for item in sorted(db_folder.rglob("*"))
                if item.is_file() and not self.config.should_ignore_file(item) and is_sqlite_database(item)
            ]

            # Skip if no SQLite files found
            if not sqlite_files:
                continue

            # Deduplicate files by MD5 hash (handles macOS firmlink duplicates)
            sqlite_files = self._deduplicate_by_md5(sqlite_files)

            # NEW: Check for empty databases but DON'T skip them
            # Empty databases still have valid schemas that are useful for raw file matching!
            try:
                # Track which files are empty (but still process them)
                empty_status = {}  # db_file -> (is_empty, reason)
                all_empty = True

                for db_file in sqlite_files:
                    is_empty, reason = is_database_empty_or_null(db_file)
                    empty_status[db_file] = (is_empty, reason)
                    if not is_empty:
                        all_empty = False

                # Skip if no SQLite files at all
                if len(sqlite_files) == 0:
                    continue

                # Handle single database (empty or not)
                if len(sqlite_files) == 1:
                    db_file = sqlite_files[0]

                    # Check if database has effective tables
                    if not has_effective_tables(db_file):
                        # Database has no effective tables, skip
                        continue

                    fingerprint = get_schema_fingerprint(db_file)
                    if fingerprint:
                        folders_to_combine.append(
                            {
                                "name": folder_name,
                                "catalog_base_name": catalog_base_name,
                                "files": [db_file],
                                "schema_version": None,
                                "is_single": True,
                                "is_empty": all_empty,  # Mark if empty
                                "empty_status": empty_status,
                            }
                        )
                    continue

                # Group databases by schema fingerprint (process all DBs, empty or not)
                schema_groups = {}  # fingerprint -> [db_files]

                for db_file in sqlite_files:
                    fingerprint = get_schema_fingerprint(db_file)
                    if fingerprint:
                        if fingerprint not in schema_groups:
                            schema_groups[fingerprint] = []
                        schema_groups[fingerprint].append(db_file)

                # Check if all databases have the same schema
                if len(schema_groups) == 0:
                    # Could not fingerprint schemas
                    continue

                if len(schema_groups) == 1:
                    # All databases have matching schema - combine them
                    matched_files = list(schema_groups.values())[0]

                    # Check if any database has effective tables
                    has_effective = any(has_effective_tables(f) for f in matched_files)
                    if not has_effective:
                        # No effective tables in any matching database
                        continue

                    folders_to_combine.append(
                        {
                            "name": folder_name,
                            "catalog_base_name": catalog_base_name,
                            "files": matched_files,
                            "schema_version": None,  # No version suffix needed
                            "is_empty": all_empty,  # Mark if all files are empty
                            "empty_status": empty_status,
                        }
                    )
                else:
                    # Multiple schema versions detected - separate them
                    # Sort by group size (largest first) for v1, v2, etc. assignment
                    sorted_groups = sorted(
                        schema_groups.items(),
                        key=lambda x: len(x[1]),
                        reverse=True,
                    )

                    # First pass: filter out versions with no effective tables
                    valid_versions = []
                    for fingerprint, db_files in sorted_groups:
                        # Check if any database in this version has effective tables
                        has_effective = any(has_effective_tables(f) for f in db_files)
                        if not has_effective:
                            # Schema version has no effective tables, skip
                            continue

                        # Check if all files in this version group are empty
                        version_all_empty = all(empty_status.get(f, (False, ""))[0] for f in db_files)

                        valid_versions.append(
                            {
                                "fingerprint": fingerprint,
                                "files": db_files,
                                "is_empty": version_all_empty,
                            }
                        )

                    # Second pass: assign version numbers only to valid versions
                    # If only one version remains, don't add version suffix
                    for idx, version_info in enumerate(valid_versions):
                        fingerprint = version_info["fingerprint"]
                        db_files = version_info["files"]
                        version_all_empty = version_info["is_empty"]

                        # Only add version suffix if multiple valid versions
                        schema_version = f"v{idx + 1}" if len(valid_versions) > 1 else None

                        # Schema version identified with fingerprint

                        # Add to processing
                        folders_to_combine.append(
                            {
                                "name": folder_name,
                                "catalog_base_name": catalog_base_name,
                                "files": db_files,
                                "schema_version": schema_version,
                                "is_single": len(db_files) == 1,  # Mark singles
                                "is_empty": version_all_empty,  # Mark if all in this version are empty
                                "empty_status": empty_status,
                            }
                        )

            except Exception:
                # Error checking schema, skip this database
                continue

        if not folders_to_combine:
            # No databases to catalog
            return

        # Process each database/group
        for combine_info in folders_to_combine:
            folder_name = combine_info["name"]  # Folder name (e.g., "Chrome Cookies_username")
            catalog_base_name = combine_info.get(
                "catalog_base_name", folder_name
            )  # Catalog name (e.g., "Chrome Cookies")
            sqlite_files = combine_info["files"]
            schema_version = combine_info.get("schema_version")
            is_single = combine_info.get("is_single", False)
            is_empty = combine_info.get("is_empty", False)
            empty_status = combine_info.get("empty_status", {})

            # Build output name with schema version and empty suffix if needed
            output_base_name = f"{folder_name}_{schema_version}" if schema_version else folder_name

            # Add _empty suffix if all databases in this group are empty
            output_name = f"{output_base_name}_empty" if is_empty else output_base_name

            display_name = output_name

            # Create catalog/ output directory
            catalog_dir = self.output.paths.exemplar_catalog / output_name
            catalog_dir.mkdir(**MKDIR_KWARGS)

            output_db = catalog_dir / f"{output_name}.db"

            try:
                if is_single:
                    # Single database - move to catalog (with auxiliary files)
                    source_db = sqlite_files[0]
                    output_db = catalog_dir / f"{output_name}.db"

                    # Move database with auxiliary files (.db-shm, .db-wal, .db-journal)
                    # First copy with auxiliary files to catalog
                    copy_database_with_auxiliary_files(source_db, output_db)

                    # Handle provenance file (the "passport")
                    # Find original provenance file and update it for the move
                    source_provenance = source_db.parent / f"{source_db.stem}.provenance.json"
                    if source_provenance.exists():
                        try:
                            # Read original provenance
                            with source_provenance.open("r") as f:
                                provenance_data = json.load(f)

                            # Create history entry to track the rename/move
                            history_entry = {
                                "operation": "move_to_catalog",
                                "timestamp": datetime.now(UTC).isoformat(),
                                "original_filename": source_db.name,
                                "new_filename": output_db.name,
                                "original_location": str(source_db.relative_to(self.output.root)),
                                "new_location": str(output_db.relative_to(self.output.root)),
                                "reason": "Single database cataloged (not combined)",
                            }

                            # Add history tracking (like passport stamps)
                            if "history" not in provenance_data:
                                provenance_data["history"] = []
                            provenance_data["history"].append(history_entry)

                            # Update current filename to reflect catalog location
                            provenance_data["catalog_filename"] = output_db.name
                            provenance_data["catalog_location"] = str(output_db.relative_to(self.output.root))

                            # Write updated provenance to catalog
                            catalog_provenance = catalog_dir / f"{output_db.stem}.provenance.json"
                            with catalog_provenance.open("w") as f:
                                json.dump(provenance_data, f, indent=2)

                            # Remove original provenance from originals/
                            source_provenance.unlink()

                            # Provenance file moved and updated
                        except Exception:
                            # Could not update provenance file
                            pass

                    # Then remove original and auxiliary files from originals/
                    # (since this is a single DB not being combined, we don't need originals copy)
                    try:
                        source_db.unlink()  # Remove main DB
                        # Remove auxiliary files if they exist
                        for suffix in ["-shm", "-wal", "-journal"]:
                            aux_file = source_db.parent / f"{source_db.name}{suffix}"
                            if aux_file.exists():
                                aux_file.unlink()
                    except Exception:
                        pass  # If removal fails, leave files in place

                    # Database moved to catalog with auxiliary files

                    # Track as "cataloged single database"
                    stats = {
                        "total_rows_merged": 0,  # Not applicable for single DB
                        "tables_merged": 0,  # Not applicable for single DB
                        "is_single": True,
                    }

                else:
                    # Multiple databases - combine them
                    output_db = catalog_dir / f"{output_name}.combined.db"

                    # Combine databases using merge_sqlite_databases
                    # Sort files by row count (descending) to ensure fullest database is first
                    # This prevents starting with an empty variant and losing data
                    # Sort by row count (descending), then by filename for determinism
                    sorted_files = sorted(
                        sqlite_files,
                        key=lambda f: (self._get_database_row_count(f), f.name),
                        reverse=True,
                    )

                    stats = merge_sqlite_databases(
                        source_dbs=sorted_files,
                        output_db=output_db,
                        combine_strategy="insert_or_ignore",
                    )
                    stats["is_single"] = False

                    # Databases combined successfully

                    # Create provenance file for combined database
                    # This documents which source files were combined to create this database
                    try:
                        # Collect provenance info from source databases
                        source_provenances = self._collect_source_provenances(sqlite_files)

                        # Create combined provenance
                        combined_provenance = self._build_combined_provenance(
                            output_db=output_db,
                            source_provenances=source_provenances,
                            stats=stats,
                            operation="combine_databases",
                            name=display_name,
                            source_count=len(sqlite_files),
                            combine_strategy="insert_or_ignore",
                            extra_fields={"schema_version": schema_version},
                        )

                        # Write combined provenance file
                        combined_prov_path = catalog_dir / f"{output_db.stem}.provenance.json"
                        with combined_prov_path.open("w") as f:
                            json.dump(combined_provenance, f, indent=2)

                        # Provenance file created for combined database
                    except Exception:
                        # Could not create provenance file
                        pass

                # Generate schema and rubric
                schema_dir = self.output.get_schema_dir(output_name)

                try:
                    # Use appropriate base name for schema files
                    schema_base = output_name if is_single else f"{output_name}_combined"

                    schema_path, rubric_path = generate_schema_and_rubric(
                        output_db,
                        schema_dir,
                        schema_base,
                        min_timestamp_rows=self.config.exemplar.min_timestamp_rows,
                    )

                    # Track unique schema directory
                    self.unique_schema_dirs.add(schema_path.parent)
                    self.generated_schemas.append(schema_path)
                    self.generated_rubrics.append(rubric_path)

                    # Schema and rubric generated successfully

                    # Track cataloged database - success case
                    catalog_record = {
                        "name": output_name,
                        "base_name": catalog_base_name,  # True catalog name (e.g., "Chrome Cookies")
                        "schema_version": schema_version,
                        "output_db": str(output_db),
                        "is_single": stats.get("is_single", False),
                        "is_empty": is_empty,
                        "total_rows": stats.get("total_rows_merged", 0),
                        "total_tables": stats.get("tables_merged", 0),
                        "source_files": len(sqlite_files),
                        "schema_path": str(schema_path),
                        "rubric_path": str(rubric_path),
                        "cataloged_at": datetime.now(UTC).isoformat(),
                        "auto_detected": True,
                    }

                except Exception as e:
                    # Schema/rubric generation failed

                    # Track cataloged database - error case
                    catalog_record = {
                        "name": output_name,
                        "base_name": catalog_base_name,  # True catalog name (e.g., "Chrome Cookies")
                        "schema_version": schema_version,
                        "output_db": str(output_db),
                        "is_single": stats.get("is_single", False),
                        "is_empty": is_empty,
                        "total_rows": stats.get("total_rows_merged", 0),
                        "total_tables": stats.get("tables_merged", 0),
                        "source_files": len(sqlite_files),
                        "schema_generated": False,
                        "generation_error": str(e),
                        "cataloged_at": datetime.now(UTC).isoformat(),
                        "auto_detected": True,
                    }

                # Update progress panel
                if panel_group:
                    panel_group.renderables.pop()
                    panel_group.renderables.append(f" • Adding to Catalog: [bold]{output_name}[/bold]")

                # Add catalog record (created in either success or error case)
                self.combined_databases.append(catalog_record)

            except Exception as e:
                # Failed to catalog database - log the error for debugging
                logger.debug(f"Failed to catalog database '{folder_name}': {e}")
                # Remove empty directory if operation failed and no DB was created
                if catalog_dir.exists() and not output_db.exists():
                    with contextlib.suppress(OSError):
                        catalog_dir.rmdir()

                    # Update progress panel
        if panel_group:
            panel_group.renderables.append("[bold] • Cleanup empty files[/bold]")

        # Cleanup: Remove empty folders from originals/
        self.cleanup_manager.cleanup_empty_originals_folders()

        # Phase 4 cataloging complete
