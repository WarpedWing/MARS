"""External Tool Export Packager.

Package MARS output (exemplar, candidate, or combined/deduped) into a format
compatible with external forensic tools (mac_apt, APOLLO, plaso) by renaming
databases to their canonical macOS filenames.
"""

from __future__ import annotations

import contextlib
import json
import re
import shutil
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

import yaml

from mars.cli.export.types import (
    LOG_FOLDER_TO_PATH,
    LOG_FOLDERS_TO_CONCATENATE,
    LOG_FOLDERS_TO_SKIP,
    ExportedFile,
    ExportedLog,
    ExportMethod,
    ExportResult,
    ExportSource,
    ExportStructure,
)
from mars.utils.cleanup_utilities import cleanup_sqlite_directory
from mars.utils.database_utils import quote_ident, readonly_connection
from mars.utils.debug_logger import logger
from mars.utils.file_utils import compute_md5_hash

# Re-export for backward compatibility
__all__ = [
    "ExportedFile",
    "ExportedLog",
    "ExportMethod",
    "ExportPackager",
    "ExportResult",
    "ExportSource",
    "ExportStructure",
]


def _robust_text_factory(data: bytes) -> str | bytes:
    """Text factory that handles BLOB data stored in TEXT columns.

    SQLite's dynamic typing allows BLOB data to be stored in TEXT/VARCHAR columns.
    Python's sqlite3 module by default tries to decode TEXT columns as UTF-8,
    which fails for binary data. This factory tries UTF-8 first and falls back
    to returning raw bytes for non-decodable content.

    Args:
        data: Raw bytes from SQLite

    Returns:
        Decoded string if valid UTF-8, otherwise raw bytes
    """
    try:
        return data.decode("utf-8")
    except (UnicodeDecodeError, AttributeError):
        return data


def _safe_relative_path(path: Path, base: Path) -> str:
    """Get relative path that works across drives on Windows.

    Both Path.relative_to() and os.path.relpath() fail when paths are on
    different drives (e.g., C: vs E:). This function falls back to returning
    the absolute path when a relative path cannot be computed.

    Args:
        path: The path to make relative
        base: The base path to compute relative to

    Returns:
        Relative path string, or absolute path if cross-drive on Windows
    """
    import os

    try:
        return os.path.relpath(path, base)
    except ValueError:
        # Cross-drive on Windows - return absolute path
        return str(path)


class ExportPackager:
    """Package MARS output for external forensic tools."""

    def __init__(
        self,
        project_output_dir: Path,
        catalog_yaml: dict | None = None,
        case_info: dict | None = None,
    ):
        """Initialize export packager.

        Args:
            project_output_dir: Path to MARS project output directory
            catalog_yaml: Optional pre-loaded catalog dict (loads from file if None)
            case_info: Optional case information dict with keys:
                - case_name: Name of the case
                - case_number: Case/incident number
                - examiner_name: Name of the examiner
                - exemplar_description: Description of the exemplar scan
                - candidate_description: Description of the candidate scan
        """
        self.output_dir = project_output_dir
        self.catalog = catalog_yaml or self._load_catalog()
        self.name_to_entry = self._build_lookup()
        self._combined_catalog_temp: Path | None = None  # Temp dir for combined export
        self.case_info = case_info or {}

    def _load_catalog(self) -> dict:
        """Load database catalog from YAML file."""
        catalog_path = Path(__file__).parent.parent / "catalog" / "artifact_recovery_catalog.yaml"

        if not catalog_path.exists():
            logger.warning(f"Database catalog not found at {catalog_path}")
            return {}

        with Path.open(catalog_path) as f:
            return yaml.safe_load(f) or {}

    def _build_lookup(self) -> dict[str, dict]:
        """Build name -> catalog entry lookup from yaml.

        Flattens all entries from all categories into a single lookup dict.
        """
        lookup = {}

        for category_name, entries in self.catalog.items():
            # Skip non-list entries (metadata, skip_databases, etc.)
            if not isinstance(entries, list):
                continue

            for entry in entries:
                if isinstance(entry, dict) and "name" in entry:
                    name = entry["name"]
                    # Store entry with category for reference
                    lookup[name] = {**entry, "_category": category_name}

        return lookup

    def _lookup_catalog_entry(self, mars_name: str) -> dict | None:
        """Look up catalog entry with flexible name matching.

        MARS folder names follow patterns like:
        - "Safari History" (direct match)
        - "Safari History_admin" (with user suffix)
        - "Chrome History_admin_v2" (user + version)
        - "Chrome Web Data_admin_v3_empty" (user + version + empty flag)

        Progressively strips suffixes from right until a match is found.

        Args:
            mars_name: MARS catalog folder name

        Returns:
            Catalog entry dict or None if not found
        """
        # Try direct match
        entry = self.name_to_entry.get(mars_name)
        if entry:
            return entry

        # Pattern for suffixes we want to strip (from right to left)
        # Matches: _empty, _v1, _v2, _admin, _root, _user, etc.
        suffix_pattern = re.compile(r"_(empty|v\d+|[a-z][a-z0-9_]*)$", re.IGNORECASE)

        # Progressively strip suffixes and try to match
        current = mars_name
        for _ in range(5):  # Max 5 levels of stripping
            match = suffix_pattern.search(current)
            if not match:
                break

            current = current[: match.start()]
            entry = self.name_to_entry.get(current)
            if entry:
                return entry

        return None

    def _extract_version_suffix(self, mars_name: str) -> str | None:
        """Extract version suffix (e.g., _v1, _v2) from MARS folder name.

        MARS naming: {base_name}_{username}_v{N} or {base_name}_v{N}

        Args:
            mars_name: MARS catalog folder name

        Returns:
            Version suffix like "_v1" or None if no version
        """
        # Match _v1, _v2, etc. (must be followed by end or _empty)
        match = re.search(r"(_v\d+)(?:_empty)?$", mars_name)
        if match:
            return match.group(1)
        return None

    def _get_canonical_path(self, entry: dict) -> str | None:
        """Get canonical macOS path from catalog entry.

        Uses glob_pattern field which contains the file path pattern.
        For databases with archives (like Powerlog), uses primary.glob_pattern.

        Args:
            entry: Catalog entry dict

        Returns:
            Canonical path string or None if not found
        """
        # For combined databases with primary/archives structure
        if "primary" in entry and isinstance(entry["primary"], dict):
            return entry["primary"].get("glob_pattern")

        # Use glob_pattern field
        return entry.get("glob_pattern")

    def _get_canonical_filename(
        self,
        canonical_path: str,
        original_filename: str | None = None,
        actual_db_name: str | None = None,
    ) -> str:
        """Extract canonical filename from path.

        For paths without wildcards, returns the filename from the path.
        For paths with glob patterns in the filename (e.g., Accounts*.sqlite,
        [0-9][0-9]*.db), uses the original filename from provenance,
        or falls back to actual.

        Args:
            canonical_path: Canonical path from catalog (may contain globs)
            original_filename: Original macOS filename from provenance
            actual_db_name: Actual MARS database filename (fallback)

        Returns:
            Canonical filename to use for export
        """
        # Use PurePosixPath to preserve forward slashes for macOS paths
        filename = PurePosixPath(canonical_path).name

        # If filename contains glob patterns, use original from provenance or actual
        # Check for: * wildcard, [x] bracket patterns, ? single char wildcard
        has_glob = "*" in filename or "[" in filename or "?" in filename
        if has_glob:
            if original_filename:
                return original_filename
            if actual_db_name:
                return actual_db_name

        return filename

    def _disambiguate_multi_filename(
        self,
        filename: str,
        profiles: list[str],
        resolved_dir: str,
    ) -> str:
        """Add profile names to filename when exporting to _multi folder.

        For databases that match multiple Chrome profiles, appends profile names
        to create unique filenames in the _multi folder.

        Args:
            filename: Original canonical filename (e.g., "History")
            profiles: List of profiles this database matches
            resolved_dir: Resolved directory path

        Returns:
            Disambiguated filename (e.g., "History_Default_Profile_1") or original
        """
        # Only disambiguate if going to _multi folder and have multiple profiles
        if "_multi" not in resolved_dir or len(profiles) <= 1:
            return filename

        # Build profile suffix: replace spaces with underscores, join with underscores
        profile_suffix = "_".join(p.replace(" ", "_") for p in sorted(profiles))

        # Split filename into name and extension
        path = PurePosixPath(filename)
        stem = path.stem
        suffix = path.suffix  # e.g., ".sqlite" or ""

        return f"{stem}_{profile_suffix}{suffix}"

    def _get_canonical_dir(self, canonical_path: str) -> str:
        """Extract canonical directory from path.

        Returns directory portion of the path.
        e.g., "private/var/db/CoreDuet/Knowledge/knowledgeC.db"
              -> "private/var/db/CoreDuet/Knowledge"
        """
        # Use PurePosixPath to preserve forward slashes for macOS paths
        return str(PurePosixPath(canonical_path).parent)

    def _extract_username(self, source_path: str) -> str | None:
        """Extract username from source path.

        Looks for pattern like /Users/username/ or Users/username/

        Args:
            source_path: Original source path string

        Returns:
            Username string or None if not found
        """
        # Match Users/<username>/ pattern (case insensitive)
        match = re.search(r"[/\\]Users[/\\]([^/\\]+)[/\\]", source_path, re.IGNORECASE)
        if match:
            username = match.group(1)
            # Filter out wildcards and system accounts
            if username not in ("*", "Shared", ".localized"):
                return username
        return None

    def _read_provenance(self, catalog_folder: Path) -> dict | None:
        """Read provenance.json or manifest.json from catalog folder.

        Tries multiple naming patterns for provenance and manifest files.
        Candidate catalogs use _manifest.json, exemplar catalogs use .provenance.json.

        Args:
            catalog_folder: Path to catalog folder

        Returns:
            Provenance/manifest dict or None if not found
        """
        # Try standard naming patterns - provenance first, then manifest
        patterns = [
            catalog_folder / f"{catalog_folder.name}.provenance.json",
            catalog_folder / "provenance.json",
            catalog_folder / f"{catalog_folder.name}_manifest.json",
        ]

        # Also try any file ending in .provenance.json or _manifest.json
        if catalog_folder.exists():
            patterns.extend(catalog_folder.glob("*.provenance.json"))
            patterns.extend(catalog_folder.glob("*_manifest.json"))

        for prov_path in patterns:
            if isinstance(prov_path, Path) and prov_path.exists():
                try:
                    with prov_path.open() as f:
                        return json.load(f)
                except Exception as e:
                    logger.debug(f"Failed to read provenance/manifest {prov_path}: {e}")

        return None

    def _get_original_filename_from_provenance(self, provenance: dict) -> str | None:
        """Extract original macOS filename from provenance.

        Args:
            provenance: Provenance dict

        Returns:
            Original filename or None
        """
        # Direct original_filename field
        if "original_filename" in provenance:
            return provenance["original_filename"]

        # Check history entries
        if "history" in provenance:
            for entry in provenance["history"]:
                if isinstance(entry, dict) and "original_filename" in entry:
                    return entry["original_filename"]

        # For combined databases, check first source
        if "source_databases" in provenance:
            for source in provenance["source_databases"]:
                if isinstance(source, dict) and "original_source" in source:
                    # Extract filename from path
                    return Path(source["original_source"]).name

        return None

    def _get_original_path_from_provenance(self, provenance: dict) -> str | None:
        """Extract original macOS path from provenance.

        For single-source databases, returns the original macOS path.
        For combined databases with multiple sources, returns None.

        Args:
            provenance: Provenance dict

        Returns:
            Original path string or None
        """
        # Check if this is a combined database (has multiple sources)
        if "source_databases" in provenance:
            sources = provenance["source_databases"]
            if isinstance(sources, list) and len(sources) > 1:
                # Combined database - no single source path
                return None

        # Best option: relative_path (clean macOS path without temp prefix)
        if "relative_path" in provenance:
            return provenance["relative_path"]

        # Direct source_path field (may have temp folder prefix)
        if "source_path" in provenance:
            return provenance["source_path"]

        # Single source database from source_databases list
        if "source_databases" in provenance:
            sources = provenance["source_databases"]
            if isinstance(sources, list) and len(sources) == 1:
                source = sources[0]
                if isinstance(source, dict) and "original_source" in source:
                    return source["original_source"]

        return None

    def _get_username_from_provenance(self, provenance: dict) -> str | None:
        """Extract username from provenance data.

        Handles both single DB and combined DB provenance formats,
        as well as candidate manifests with explicit username field.

        Args:
            provenance: Provenance dict (or manifest for candidate DBs)

        Returns:
            Username string or None
        """
        # Check for direct username field first (candidate manifests)
        if "username" in provenance and provenance["username"]:
            return provenance["username"]

        # Single DB: check source_path directly
        if "source_path" in provenance:
            username = self._extract_username(provenance["source_path"])
            if username:
                return username

        # Combined DB: check source_databases list
        if "source_databases" in provenance:
            for source in provenance["source_databases"]:
                if isinstance(source, dict):
                    # Check original_source field
                    orig = source.get("original_source") or source.get("source_path")
                    if orig:
                        username = self._extract_username(orig)
                        if username:
                            return username

        # Check history entries (for moved files)
        if "history" in provenance:
            for entry in provenance["history"]:
                if isinstance(entry, dict):
                    orig = entry.get("original_location") or entry.get("source_path")
                    if orig:
                        username = self._extract_username(orig)
                        if username:
                            return username

        return None

    def _get_profiles_from_provenance(self, provenance: dict) -> list[str]:
        """Extract profiles list from provenance data.

        For multi-profile databases (e.g., Chrome with Default, Profile 1),
        returns the list of profiles this database matches.

        Args:
            provenance: Provenance dict (or manifest for candidate DBs)

        Returns:
            List of profile names (e.g., ["Default"], ["Profile 1"]) or empty list
        """
        # Check for direct profiles field (candidate manifests and provenance)
        if "profiles" in provenance:
            profiles = provenance["profiles"]
            if isinstance(profiles, list):
                return profiles

        # For combined DBs, collect profiles from source_databases
        if "source_databases" in provenance:
            sources = provenance["source_databases"]
            if isinstance(sources, list):
                profiles = set()
                for source in sources:
                    if isinstance(source, dict) and "profile" in source:
                        profiles.add(source["profile"])
                if profiles:
                    return sorted(profiles)

        return []

    def _find_database_file(self, catalog_folder: Path, prefer_combined: bool = False) -> tuple[Path | None, bool]:
        """Find the main database file in a catalog folder.

        Args:
            catalog_folder: Path to catalog folder
            prefer_combined: If True, prefer .combined.db files when available

        Returns:
            Tuple of (database file path or None, is_combined_file)
        """
        # Common database extensions
        db_extensions = [".db", ".sqlite", ".sqlite3", ".PLSQL", ".storedata"]
        folder_name = catalog_folder.name

        # If prefer_combined, look for .combined.db files first
        if prefer_combined:
            combined_files = list(catalog_folder.glob("*.combined.db"))
            if combined_files:
                # Prefer non-auxiliary files
                for match in combined_files:
                    if not any(match.name.endswith(aux) for aux in ["-shm", "-wal", "-journal"]):
                        return match, True
                return combined_files[0], True

        # First priority: look for a file matching the folder name
        # Candidate catalogs often have {folder_name}.sqlite with combined data
        for ext in db_extensions:
            folder_match = catalog_folder / f"{folder_name}{ext}"
            if folder_match.exists() and folder_match.stat().st_size > 0:
                return folder_match, False

        # Look for files with database extensions
        for ext in db_extensions:
            matches = list(catalog_folder.glob(f"*{ext}"))
            if matches:
                # Prefer non-auxiliary files (not -shm, -wal, -journal)
                # Also skip .combined.db files when not preferring combined
                # And skip empty files (0 bytes)
                for match in matches:
                    if any(match.name.endswith(aux) for aux in ["-shm", "-wal", "-journal", ".combined.db"]):
                        continue
                    # Skip empty files
                    if match.stat().st_size == 0:
                        continue
                    return match, False
                # If only auxiliary, combined, or empty files, try again without size check
                for match in matches:
                    if match.name.endswith(".combined.db"):
                        continue
                    return match, False

        # Fallback: look for any file that's not provenance/metadata/hidden
        for f in catalog_folder.iterdir():
            if (
                f.is_file()
                and not f.name.startswith(".")
                and not f.name.endswith((".json", ".md", ".txt", ".combined.db"))
            ):
                return f, False

        # Final fallback: use .combined.db if that's all we have
        # (e.g., Powerlog with archives combined in Phase 3)
        combined_files = list(catalog_folder.glob("*.combined.db"))
        if combined_files:
            return combined_files[0], True

        return None, False

    def _detect_profile_subfolders(self, catalog_folder: Path) -> list[Path]:
        """Detect browser profile subfolders in a multi-profile catalog folder.

        Multi-profile folders contain nested profile subdirectories like:
        - Default/
        - Profile 1/
        - abc123.default/  (Firefox)

        Args:
            catalog_folder: Catalog folder to check

        Returns:
            List of profile subfolder paths, or empty list if not multi-profile
        """
        profile_subfolders = []

        # Folders to skip - not browser profiles
        skip_folders = {"Archives", "Quarantine", "empty", "rejected", "remnants"}

        for item in sorted(catalog_folder.iterdir()):
            if not item.is_dir():
                continue

            # Skip known non-profile subfolders
            if item.name in skip_folders:
                continue

            # Check if this subdirectory contains a database file directly
            db_file, _ = self._find_database_file(item, prefer_combined=False)
            if db_file:
                profile_subfolders.append(item)

        return profile_subfolders

    def _resolve_canonical_dir(
        self,
        canonical_dir: str,
        username: str | None,
        original_path: str | None,
        profiles: list[str] | None = None,
    ) -> str:
        """Resolve wildcards in canonical directory path.

        Uses original_path from provenance to resolve wildcards when available.
        For multi-profile databases (e.g., Chrome), uses profiles list to resolve
        profile wildcards. For combined databases without a single source,
        sanitizes wildcards.

        Args:
            canonical_dir: Canonical directory path (may contain wildcards)
            username: Actual username to substitute for Users/*
            original_path: Original macOS path from provenance (for single-source DBs)
            profiles: List of profile names (e.g., ["Default"], ["Profile 1"]) for
                multi-profile databases like Chrome

        Returns:
            Path with wildcards resolved or sanitized
        """
        result = canonical_dir

        # First, resolve Users/* with username
        if username:
            result = result.replace("Users/*", f"Users/{username}")
            result = result.replace("Users\\*", f"Users\\{username}")

        # Check if wildcards remain
        has_wildcards = "*" in result or "[" in result or "?" in result

        if has_wildcards and original_path:
            # Try to extract the actual directory from original_path
            # First try to strip common temp folder prefixes
            extracted = self._extract_macos_path(original_path)
            if extracted:
                # Use the directory portion of the extracted path
                # Use PurePosixPath to preserve forward slashes for macOS paths
                extracted_dir = str(PurePosixPath(extracted).parent)
                # Strip leading slash if present
                extracted_dir = extracted_dir.lstrip("/")
                if extracted_dir:
                    return extracted_dir
            # If extraction didn't work, original_path might already be a clean relative path
            elif original_path.startswith("Users/") or original_path.startswith("private/"):
                # Already a clean macOS relative path
                # Use PurePosixPath to preserve forward slashes for macOS paths
                extracted_dir = str(PurePosixPath(original_path).parent)
                if extracted_dir and extracted_dir != ".":
                    return extracted_dir

        # Check if wildcards remain after original_path resolution
        has_wildcards = "*" in result or "[" in result or "?" in result

        # If we have profiles info and still have wildcards, try to resolve profile wildcard
        # This handles paths like "Users/admin/Library/.../Chrome/*/History"
        # where the second * is for the Chrome profile (Default, Profile 1, etc.)
        if has_wildcards and profiles:
            if len(profiles) == 1:
                # Single profile - substitute directly
                # Find the remaining * (should be the profile wildcard after Users/* resolved)
                profile_name = profiles[0]
                result = result.replace("*", profile_name, 1)  # Replace first remaining *
            elif len(profiles) > 1:
                # Multiple profiles - use _multi with profile names in filename
                # The folder goes to _multi, filename disambiguation happens elsewhere
                result = result.replace("*", "_multi", 1)

        # If we still have wildcards, sanitize them for filesystem safety
        if "*" in result or "[" in result or "?" in result:
            # Replace remaining wildcards with descriptive placeholders
            result = result.replace("*", "_multi")
            result = re.sub(r"\[.*?\]", "_var", result)
            result = result.replace("?", "_")

        return result

    def _extract_macos_path(self, original_path: str) -> str | None:
        """Extract macOS path from original_source path.

        Strips temp folder prefixes like /var/folders/.../T/mars_dfvfs_export_xxx/
        to get the actual macOS path.

        Args:
            original_path: Full original_source path from provenance

        Returns:
            Extracted macOS path or None
        """
        # Common patterns for temp folder prefixes
        patterns = [
            # macOS dfvfs export temp folders
            r"/var/folders/[^/]+/[^/]+/T/mars_dfvfs_export_[^/]+/",
            # macOS generic temp folders
            r"/var/folders/[^/]+/[^/]+/T/[^/]+/",
            r"/tmp/[^/]+/",
            r"/private/var/folders/[^/]+/[^/]+/T/[^/]+/",
            # macOS direct volume mounts
            r"/Volumes/[^/]+/",
            # Windows temp folders
            r"[A-Za-z]:\\Users\\[^\\]+\\AppData\\Local\\Temp\\[^\\]+\\",
            r"[A-Za-z]:\\Temp\\[^\\]+\\",
            r"[A-Za-z]:\\Windows\\Temp\\[^\\]+\\",
        ]

        for pattern in patterns:
            match = re.search(pattern, original_path)
            if match:
                # Return everything after the matched prefix
                return original_path[match.end() :]

        # If no pattern matched, check if it starts with standard paths
        # macOS paths
        for prefix in ["Users/", "private/", "Library/", "var/"]:
            idx = original_path.find(prefix)
            if idx >= 0:
                return original_path[idx:]
        # Windows paths (using backslash)
        for prefix in ["Users\\", "ProgramData\\", "Windows\\"]:
            idx = original_path.find(prefix)
            if idx >= 0:
                return original_path[idx:]

        return None

    def _get_flat_export_path(
        self,
        export_dir: Path,
        canonical_filename: str,
        base_name: str,
        profile: str | None = None,
        username: str | None = None,
        version: str | None = None,
    ) -> Path:
        """Get export path for flat structure.

        Uses catalog base_name as root folder, with optional username suffix
        for user-scoped databases and profile/version subfolders.

        Creates paths like:
        - Chrome History_admin/Default/History
        - Chrome History_admin/Profile 1/History
        - Chrome History_admin/_multi/History_Default_Profile_1
        - CFURL Cache Database_admin/_v1/Cache.db    (versioned user-scoped)
        - CFURL Cache Database_admin/_v2/Cache.db
        - CFURL Cache Database_admin/Cache.db        (non-versioned user-scoped)
        - Knowledge Store (CoreDuet System)/knowledgeC.db  (system-scoped)
        - Notification Center Database (Temp Folder)/_v1/db  (versioned system-scoped)

        Args:
            export_dir: Base export directory
            canonical_filename: Canonical macOS filename
            base_name: Catalog base name (e.g., "Chrome History", "CFURL Cache Database")
            profile: Profile subfolder name (e.g., "Default", "_multi") or None for non-profile DBs
            username: Username to append to base_name for user-scoped DBs (e.g., "admin")
            version: Version suffix for multi-schema DBs (e.g., "_v1", "_v2") or None

        Returns:
            Export path
        """
        # Build root folder name: base_name or base_name_username
        root_name = f"{base_name}_{username}" if username else base_name

        # Build subfolder: profile takes precedence over version
        # Profile DBs: root/profile/file (versions handled via profile structure)
        # Non-profile versioned DBs: root/_v1/file
        # Non-profile non-versioned DBs: root/file
        if profile:
            folder = export_dir / root_name / profile
        elif version:
            folder = export_dir / root_name / version
        else:
            folder = export_dir / root_name

        folder.mkdir(parents=True, exist_ok=True)
        return folder / canonical_filename

    def _get_full_path_export_path(self, export_dir: Path, canonical_dir: str, canonical_filename: str) -> Path:
        """Get export path for full path structure.

        Creates paths like:
        - private/var/db/CoreDuet/Knowledge/knowledgeC.db
        - Users/john/Library/Safari/History.db

        Args:
            export_dir: Base export directory
            canonical_dir: Canonical directory path (must be relative)
            canonical_filename: Canonical filename

        Returns:
            Export path
        """
        # Ensure canonical_dir is relative - absolute paths would replace export_dir
        canonical_dir = canonical_dir.lstrip("/")
        dest_dir = export_dir / canonical_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        return dest_dir / canonical_filename

    def _export_file(
        self,
        source: Path,
        dest: Path,
        method: ExportMethod,
    ) -> int:
        """Export a single file.

        Args:
            source: Source file path
            dest: Destination file path
            method: Export method (copy or symlink)

        Returns:
            Size of exported file in bytes
        """
        if method == ExportMethod.SYMLINK:
            # Create symlink (fallback to copy on Windows if symlink fails)
            if dest.exists() or dest.is_symlink():
                dest.unlink()
            try:
                dest.symlink_to(source.resolve())
                return source.stat().st_size
            except OSError:
                # Windows requires admin privileges or Developer Mode for symlinks
                # Fall back to copying the file
                pass  # Fall through to copy below

        # Copy file
        shutil.copy2(source, dest)
        return dest.stat().st_size

    def _get_catalog_dir(
        self,
        source: ExportSource,
        candidate_run: Path | None = None,
        track_data_source: bool = False,
    ) -> Path | None:
        """Get the catalog directory for the given source type.

        Args:
            source: Export source type
            candidate_run: Optional specific candidate run directory
            track_data_source: If True, add data_source column to combined DBs

        Returns:
            Path to catalog directory or None if not found
        """
        if source == ExportSource.EXEMPLAR:
            catalog_dir = self.output_dir / "exemplar" / "databases" / "catalog"
        elif source == ExportSource.CANDIDATE:
            if candidate_run:
                # Use specified candidate run
                catalog_dir = candidate_run / "databases" / "catalog"
            else:
                # Default to latest run directory
                candidates_dir = self.output_dir / "candidates"
                if candidates_dir.exists():
                    run_dirs = sorted(
                        [d for d in candidates_dir.iterdir() if d.is_dir()],
                        key=lambda d: d.name,
                        reverse=True,
                    )
                    if run_dirs:
                        catalog_dir = run_dirs[0] / "databases" / "catalog"
                    else:
                        return None
                else:
                    return None
        else:
            # Combined - build merged catalog with exemplar + unique candidate rows
            combined_catalog = self._build_combined_catalog(candidate_run, track_data_source)
            if combined_catalog:
                # Store for cleanup later
                self._combined_catalog_temp = combined_catalog
                return combined_catalog
            return None

        return catalog_dir if catalog_dir.exists() else None

    def export_databases(
        self,
        export_dir: Path,
        source: ExportSource = ExportSource.EXEMPLAR,
        structure: ExportStructure = ExportStructure.FLAT,
        method: ExportMethod = ExportMethod.COPY,
        candidate_run: Path | None = None,
        track_data_source: bool = False,
        include_logs: bool = False,
    ) -> ExportResult:
        """Export databases with canonical names/paths.

        Args:
            export_dir: Directory to export to
            source: Source type (exemplar, candidate, combined)
            structure: Directory structure (flat or full_path)
            method: Export method (copy or symlink)
            candidate_run: Optional specific candidate run directory
            track_data_source: If True, add data_source column to combined DBs
                to track row provenance (exemplar, recovered)
            include_logs: If True, also export logs and keychains

        Returns:
            ExportResult with details of exported files
        """
        result = ExportResult(success=False, export_dir=export_dir)

        # Create export directory
        export_dir.mkdir(parents=True, exist_ok=True)

        # Get catalog directory
        catalog_dir = self._get_catalog_dir(source, candidate_run, track_data_source)
        if not catalog_dir:
            result.errors.append(("", f"Catalog directory not found for {source.value}"))
            return result

        # Process each catalog folder
        for folder in sorted(catalog_dir.iterdir()):
            if not folder.is_dir():
                continue

            mars_name = folder.name

            # Look up in catalog with flexible name matching
            entry = self._lookup_catalog_entry(mars_name)

            if not entry:
                result.skipped_files.append((mars_name, "Not in catalog"))
                continue

            # Get canonical path info
            canonical_path = self._get_canonical_path(entry)
            if not canonical_path:
                result.skipped_files.append((mars_name, "No canonical path in catalog"))
                continue

            # Check for multi-profile structure (nested profile folders)
            profile_subfolders = self._detect_profile_subfolders(folder)

            if profile_subfolders:
                # Multi-profile folder - export each profile's database separately
                for profile_folder in profile_subfolders:
                    profile_name = profile_folder.name
                    profile_mars_name = f"{mars_name}/{profile_name}"

                    # Find database file in profile folder
                    prefer_combined = source == ExportSource.COMBINED
                    db_file, is_combined = self._find_database_file(profile_folder, prefer_combined)
                    if not db_file:
                        result.skipped_files.append((profile_mars_name, "No database file found"))
                        continue

                    # Read provenance from profile folder
                    provenance = self._read_provenance(profile_folder)
                    original_filename = None
                    original_path = None
                    sub_profiles = self._get_profiles_from_provenance(provenance) if provenance else []
                    if provenance:
                        original_filename = self._get_original_filename_from_provenance(provenance)
                        original_path = self._get_original_path_from_provenance(provenance)

                    # Get canonical filename
                    canonical_filename = self._get_canonical_filename(canonical_path, original_filename, db_file.name)
                    canonical_dir_str = self._get_canonical_dir(canonical_path)
                    scope = entry.get("scope", "system")

                    # Get username from provenance
                    username = None
                    if scope == "user" and provenance:
                        username = self._get_username_from_provenance(provenance)

                    # Resolve wildcards in canonical dir using provenance data
                    resolved_dir = self._resolve_canonical_dir(canonical_dir_str, username, original_path, sub_profiles)

                    # Disambiguate filename if exporting to _multi folder
                    export_filename = self._disambiguate_multi_filename(canonical_filename, sub_profiles, resolved_dir)

                    # Determine export path based on structure
                    if structure == ExportStructure.FLAT:
                        # Use catalog base_name with profile subfolder and username
                        base_name = entry.get("name", mars_name)
                        flat_username = username if scope == "user" else None
                        dest_path = self._get_flat_export_path(
                            export_dir, export_filename, base_name, profile_name, flat_username
                        )
                    else:
                        dest_path = self._get_full_path_export_path(export_dir, resolved_dir, export_filename)

                    # Export the file
                    try:
                        size = self._export_file(db_file, dest_path, method)
                        result.total_size += size

                        exported = ExportedFile(
                            mars_name=profile_mars_name,
                            canonical_name=export_filename,
                            canonical_path=f"{resolved_dir}/{export_filename}",
                            source_path=db_file,
                            export_path=dest_path,
                            username=username,
                            scope=scope,
                            is_combined=is_combined,
                            method=method.value,
                        )
                        result.exported_files.append(exported)

                    except Exception as e:
                        result.errors.append((profile_mars_name, str(e)))

                # Done with multi-profile folder
                continue

            # Standard (non-profile) folder processing
            # Find database file first (need actual filename for glob resolution)
            # For combined exports, prefer .combined.db files when available
            prefer_combined = source == ExportSource.COMBINED
            db_file, is_combined = self._find_database_file(folder, prefer_combined)
            if not db_file:
                result.skipped_files.append((mars_name, "No database file found"))
                continue

            # Read provenance for original filename, path, username, and profiles
            provenance = self._read_provenance(folder)
            original_filename = None
            original_path = None
            profile_list: list[str] = []
            if provenance:
                original_filename = self._get_original_filename_from_provenance(provenance)
                original_path = self._get_original_path_from_provenance(provenance)
                profile_list = self._get_profiles_from_provenance(provenance)

            # Get canonical filename (use original from provenance for glob patterns)
            canonical_filename = self._get_canonical_filename(canonical_path, original_filename, db_file.name)
            canonical_dir_str = self._get_canonical_dir(canonical_path)
            scope = entry.get("scope", "system")

            # Get username from provenance for user-scoped databases
            username = None
            if scope == "user" and provenance:
                username = self._get_username_from_provenance(provenance)

            # Resolve wildcards in canonical dir using provenance data
            resolved_dir = self._resolve_canonical_dir(canonical_dir_str, username, original_path, profile_list)

            # Disambiguate filename if exporting to _multi folder
            export_filename = self._disambiguate_multi_filename(canonical_filename, profile_list, resolved_dir)

            # Determine export path based on structure
            if structure == ExportStructure.FLAT:
                # Use catalog base_name with profile subfolder (if applicable)
                base_name = entry.get("name", mars_name)
                # Determine profile for flat export path
                if len(profile_list) == 1:
                    flat_profile: str | None = profile_list[0]
                elif len(profile_list) > 1:
                    flat_profile = "_multi"
                else:
                    flat_profile = None
                # Extract version suffix and username for flat export
                flat_username = username if scope == "user" else None
                flat_version = self._extract_version_suffix(mars_name) if not flat_profile else None
                dest_path = self._get_flat_export_path(
                    export_dir, export_filename, base_name, flat_profile, flat_username, flat_version
                )
            else:
                dest_path = self._get_full_path_export_path(export_dir, resolved_dir, export_filename)

            # Export the file
            try:
                size = self._export_file(db_file, dest_path, method)
                result.total_size += size

                exported = ExportedFile(
                    mars_name=mars_name,
                    canonical_name=export_filename,
                    canonical_path=f"{resolved_dir}/{export_filename}",
                    source_path=db_file,
                    export_path=dest_path,
                    username=username,
                    scope=scope,
                    is_combined=is_combined,
                    method=method.value,
                )
                result.exported_files.append(exported)

            except Exception as e:
                result.errors.append((mars_name, str(e)))

        # Export logs if requested
        if include_logs:
            logs, keychains, skipped = self.export_logs(
                export_dir=export_dir,
                source=source,
                structure=structure,
                method=method,
                candidate_run=candidate_run,
            )
            result.exported_logs = logs
            result.exported_keychains = keychains
            result.skipped_logs = skipped

        # Write manifest
        self._write_manifest(export_dir, result, source, structure, method)

        # Clean up temporary combined catalog if created
        if self._combined_catalog_temp and self._combined_catalog_temp.exists():
            try:
                cleanup_sqlite_directory(self._combined_catalog_temp)
                logger.debug(f"Cleaned up temp combined catalog: {self._combined_catalog_temp}")
            except Exception as e:
                logger.debug(f"Failed to clean up temp dir: {e}")
            finally:
                self._combined_catalog_temp = None

        result.success = len(result.exported_files) > 0
        return result

    def _write_manifest(
        self,
        export_dir: Path,
        result: ExportResult,
        source: ExportSource,
        structure: ExportStructure,
        method: ExportMethod,
    ) -> None:
        """Write export manifest JSON file.

        Args:
            export_dir: Export directory
            result: Export result
            source: Export source type
            structure: Export structure type
            method: Export method
        """
        # Build case_info section based on export source
        case_info_section = {
            "case_name": self.case_info.get("case_name"),
            "case_number": self.case_info.get("case_number"),
            "examiner_name": self.case_info.get("examiner_name"),
        }

        # Add descriptions based on export source
        if source == ExportSource.EXEMPLAR:
            case_info_section["exemplar_description"] = self.case_info.get("exemplar_description")
        elif source == ExportSource.CANDIDATE:
            case_info_section["candidate_description"] = self.case_info.get("candidate_description")
        elif source == ExportSource.COMBINED:
            case_info_section["exemplar_description"] = self.case_info.get("exemplar_description")
            case_info_section["candidate_description"] = self.case_info.get("candidate_description")

        manifest = {
            "case_info": case_info_section,
            "export_info": {
                "created": datetime.now(UTC).isoformat(),
                "source": source.value,
                "structure": structure.value,
                "method": method.value,
                "total_files": len(result.exported_files),
                "total_logs": len(result.exported_logs),
                "total_keychains": len(result.exported_keychains),
                "total_size_bytes": result.total_size,
                "mars_output_dir": str(self.output_dir),
            },
            "exported_files": [
                {
                    "mars_name": f.mars_name,
                    "canonical_name": f.canonical_name,
                    "canonical_path": f.canonical_path,
                    "export_path": _safe_relative_path(f.export_path, export_dir),
                    "username": f.username,
                    "scope": f.scope,
                    "is_combined": f.is_combined,
                }
                for f in result.exported_files
            ],
            "exported_logs": [
                {
                    "folder_name": log.folder_name,
                    "original_filename": log.original_filename,
                    "relative_path": log.relative_path,
                    "export_path": _safe_relative_path(log.export_path, export_dir),
                    "md5_hash": log.md5_hash,
                }
                for log in result.exported_logs
            ],
            "exported_keychains": [
                {
                    "folder_name": kc.folder_name,
                    "original_filename": kc.original_filename,
                    "relative_path": kc.relative_path,
                    "export_path": _safe_relative_path(kc.export_path, export_dir),
                    "md5_hash": kc.md5_hash,
                }
                for kc in result.exported_keychains
            ],
            "skipped": [{"name": name, "reason": reason} for name, reason in result.skipped_files],
            "skipped_logs": [{"name": name, "reason": reason} for name, reason in result.skipped_logs],
            "errors": [{"name": name, "error": error} for name, error in result.errors],
        }

        manifest_path = export_dir / "_manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

    # ========== Log Export Methods ==========

    def export_logs(
        self,
        export_dir: Path,
        source: ExportSource,
        structure: ExportStructure,
        method: ExportMethod,
        candidate_run: Path | None = None,
    ) -> tuple[list[ExportedLog], list[ExportedLog], list[tuple[str, str]]]:
        """Export log and keychain files.

        Args:
            export_dir: Directory to export to
            source: Source type (exemplar, candidate, combined)
            structure: Directory structure (flat or full_path)
            method: Export method (copy or symlink)
            candidate_run: Optional specific candidate run directory

        Returns:
            Tuple of (exported_logs, exported_keychains, skipped_logs)
        """
        exported_logs: list[ExportedLog] = []
        exported_keychains: list[ExportedLog] = []
        skipped_logs: list[tuple[str, str]] = []

        # Get source directories based on source type
        exemplar_logs = self.output_dir / "exemplar" / "logs"
        exemplar_keychains = self.output_dir / "exemplar" / "keychains"

        if candidate_run:
            candidate_logs = candidate_run / "logs"
            candidate_keychains = candidate_run / "keychains"
        else:
            # Find latest candidate run
            candidates_dir = self.output_dir / "candidates"
            if candidates_dir.exists():
                run_dirs = sorted(
                    [d for d in candidates_dir.iterdir() if d.is_dir()],
                    key=lambda x: x.name,
                    reverse=True,
                )
                if run_dirs:
                    candidate_run = run_dirs[0]
                    candidate_logs = candidate_run / "logs"
                    candidate_keychains = candidate_run / "keychains"
                else:
                    candidate_logs = None
                    candidate_keychains = None
            else:
                candidate_logs = None
                candidate_keychains = None

        # Build exemplar hash set for combined deduplication
        exemplar_hashes: dict[str, set[str]] = {}
        if source == ExportSource.COMBINED and exemplar_logs and exemplar_logs.exists():
            exemplar_hashes = self._build_exemplar_log_hashes(exemplar_logs)
            if exemplar_keychains and exemplar_keychains.exists():
                kc_hashes = self._build_exemplar_log_hashes(exemplar_keychains)
                exemplar_hashes.update(kc_hashes)

        # Export based on source type
        if source == ExportSource.EXEMPLAR:
            if exemplar_logs and exemplar_logs.exists():
                logs, skipped = self._export_log_directory(exemplar_logs, export_dir, structure, method, "log")
                exported_logs.extend(logs)
                skipped_logs.extend(skipped)
            if exemplar_keychains and exemplar_keychains.exists():
                keychains, skipped = self._export_log_directory(
                    exemplar_keychains, export_dir, structure, method, "keychain"
                )
                exported_keychains.extend(keychains)
                skipped_logs.extend(skipped)

        elif source == ExportSource.CANDIDATE:
            if candidate_logs and candidate_logs.exists():
                logs, skipped = self._export_log_directory(candidate_logs, export_dir, structure, method, "log")
                exported_logs.extend(logs)
                skipped_logs.extend(skipped)
            if candidate_keychains and candidate_keychains.exists():
                keychains, skipped = self._export_log_directory(
                    candidate_keychains, export_dir, structure, method, "keychain"
                )
                exported_keychains.extend(keychains)
                skipped_logs.extend(skipped)

        elif source == ExportSource.COMBINED:
            # First, handle text log folders that need concatenation (not deduplication)
            for folder_name in LOG_FOLDERS_TO_CONCATENATE:
                exemplar_folder = exemplar_logs / folder_name if exemplar_logs else None
                candidate_folder = candidate_logs / folder_name if candidate_logs else None

                # Check if either folder exists
                exemplar_exists = exemplar_folder and exemplar_folder.exists()
                candidate_exists = candidate_folder and candidate_folder.exists()

                if exemplar_exists or candidate_exists:
                    result = self._concatenate_text_logs(
                        folder_name,
                        exemplar_folder if exemplar_exists else None,
                        candidate_folder if candidate_exists else None,
                        export_dir,
                        structure,
                    )
                    if result:
                        exported_logs.append(result)

            # Export exemplar logs (excluding concatenated folders)
            if exemplar_logs and exemplar_logs.exists():
                logs, skipped = self._export_log_directory(
                    exemplar_logs,
                    export_dir,
                    structure,
                    method,
                    "log",
                    skip_folders=LOG_FOLDERS_TO_CONCATENATE,
                )
                exported_logs.extend(logs)
                skipped_logs.extend(skipped)
            if exemplar_keychains and exemplar_keychains.exists():
                keychains, skipped = self._export_log_directory(
                    exemplar_keychains, export_dir, structure, method, "keychain"
                )
                exported_keychains.extend(keychains)
                skipped_logs.extend(skipped)

            # Export candidate logs with deduplication (excluding concatenated folders)
            if candidate_logs and candidate_logs.exists():
                logs, skipped = self._export_log_directory(
                    candidate_logs,
                    export_dir,
                    structure,
                    method,
                    "log",
                    exemplar_hashes=exemplar_hashes,
                    skip_folders=LOG_FOLDERS_TO_CONCATENATE,
                )
                exported_logs.extend(logs)
                skipped_logs.extend(skipped)
            if candidate_keychains and candidate_keychains.exists():
                keychains, skipped = self._export_log_directory(
                    candidate_keychains,
                    export_dir,
                    structure,
                    method,
                    "keychain",
                    exemplar_hashes=exemplar_hashes,
                )
                exported_keychains.extend(keychains)
                skipped_logs.extend(skipped)

        return exported_logs, exported_keychains, skipped_logs

    def _build_exemplar_log_hashes(self, logs_dir: Path) -> dict[str, set[str]]:
        """Build MD5 hash lookup from exemplar logs.

        Args:
            logs_dir: Directory containing log folders

        Returns:
            Dict mapping folder_name to set of MD5 hashes (or names for logarchives)
        """
        hashes: dict[str, set[str]] = {}

        if not logs_dir.exists():
            return hashes

        for folder in logs_dir.iterdir():
            # Skip .logarchive bundles - they only exist in exemplar, never in candidate
            if folder.suffix == ".logarchive":
                continue

            if not folder.is_dir():
                continue

            if folder.name in LOG_FOLDERS_TO_SKIP:
                continue

            folder_hashes: set[str] = set()
            for file_path in folder.rglob("*"):
                if file_path.is_file():
                    try:
                        md5 = compute_md5_hash(file_path)
                        folder_hashes.add(md5)
                    except Exception:
                        pass
            hashes[folder.name] = folder_hashes

        return hashes

    def _concatenate_text_logs(
        self,
        folder_name: str,
        exemplar_folder: Path | None,
        candidate_folder: Path | None,
        export_dir: Path,
        structure: ExportStructure,
    ) -> ExportedLog | None:
        """Concatenate text log files from exemplar and candidate into single output.

        For text logs (System Log, WiFi Log, Install Log), we want to combine
        all fragments into a single file rather than deduplicating by hash.

        Args:
            folder_name: MARS folder name (e.g., "System Log")
            exemplar_folder: Path to exemplar log folder (may be None)
            candidate_folder: Path to candidate log folder (may be None)
            export_dir: Export destination directory
            structure: Directory structure (flat or full_path)

        Returns:
            ExportedLog if successful, None if no files to concatenate
        """
        # Determine output path based on folder name
        output_filename_map = {
            "System Log": "system.log",
            "WiFi Log": "wifi.log",
            "Install Log": "install.log",
        }
        output_filename = output_filename_map.get(folder_name)
        if not output_filename:
            return None

        # Determine destination path
        if structure == ExportStructure.FLAT:
            dest_path = export_dir / "logs" / folder_name / output_filename
        else:
            # Use the canonical macOS path from mapping
            relative_path = LOG_FOLDER_TO_PATH.get(folder_name)
            if relative_path:
                dest_path = export_dir / relative_path
            else:
                dest_path = export_dir / "logs" / folder_name / output_filename

        # Collect all files to concatenate (exemplar first, then candidate)
        files_to_concat: list[Path] = []

        if exemplar_folder and exemplar_folder.exists():
            for f in sorted(exemplar_folder.iterdir()):
                if f.is_file() and not f.name.endswith("_provenance.json"):
                    files_to_concat.append(f)

        if candidate_folder and candidate_folder.exists():
            for f in sorted(candidate_folder.iterdir()):
                if f.is_file() and not f.name.endswith("_provenance.json"):
                    files_to_concat.append(f)

        if not files_to_concat:
            return None

        # Concatenate all files
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with dest_path.open("wb") as outfile:
            for src_file in files_to_concat:
                try:
                    with src_file.open("rb") as infile:
                        content = infile.read()
                        outfile.write(content)
                        # Add newline separator if content doesn't end with one
                        if content and not content.endswith(b"\n"):
                            outfile.write(b"\n")
                except Exception as e:
                    logger.warning(f"Failed to concatenate {src_file}: {e}")

        # Compute hash of combined file
        combined_hash = compute_md5_hash(dest_path) if dest_path.exists() else ""

        return ExportedLog(
            folder_name=folder_name,
            original_filename=output_filename,
            relative_path=_safe_relative_path(dest_path, export_dir),
            source_path=exemplar_folder or candidate_folder or dest_path,
            export_path=dest_path,
            md5_hash=combined_hash,
            artifact_type="log",
            method="concatenate",
        )

    def _export_log_directory(
        self,
        source_dir: Path,
        export_dir: Path,
        structure: ExportStructure,
        method: ExportMethod,
        artifact_type: str,
        exemplar_hashes: dict[str, set[str]] | None = None,
        skip_folders: set[str] | None = None,
    ) -> tuple[list[ExportedLog], list[tuple[str, str]]]:
        """Export all files from a log/keychain directory.

        Args:
            source_dir: Source directory (logs/ or keychains/)
            export_dir: Export destination directory
            structure: Directory structure (flat or full_path)
            method: Export method (copy or symlink)
            artifact_type: "log" or "keychain"
            exemplar_hashes: Optional hash set for deduplication
            skip_folders: Optional set of folder names to skip (handled separately)

        Returns:
            Tuple of (exported_logs, skipped_logs)
        """
        exported: list[ExportedLog] = []
        skipped: list[tuple[str, str]] = []

        if not source_dir.exists():
            return exported, skipped

        # Handle .logarchive bundles in the logs directory
        # (only exist in exemplar, never in candidate, so no deduplication needed)
        for item in source_dir.iterdir():
            if item.suffix == ".logarchive" and item.is_dir():
                result = self._export_logarchive(item, export_dir, structure, method)
                if result:
                    exported.append(result)

        # Process log folders
        for folder in source_dir.iterdir():
            if not folder.is_dir() or folder.suffix == ".logarchive":
                continue

            # Skip folders that are handled separately (e.g., concatenated text logs)
            if skip_folders and folder.name in skip_folders:
                continue

            if folder.name in LOG_FOLDERS_TO_SKIP:
                skipped.append((folder.name, "Skipped folder"))
                continue

            # Read provenance for this folder
            provenance = self._read_log_provenance(folder)

            # Export all files in folder (recursively for nested structures)
            for file_path in folder.rglob("*"):
                if not file_path.is_file():
                    continue

                # Skip provenance files (both *_provenance.json and *.provenance.json patterns)
                if file_path.name.endswith("_provenance.json") or file_path.name.endswith(".provenance.json"):
                    continue

                try:
                    # Compute MD5 for deduplication
                    md5 = compute_md5_hash(file_path)

                    # Check for duplicates in combined export
                    if exemplar_hashes:
                        folder_hashes = exemplar_hashes.get(folder.name, set())
                        if md5 in folder_hashes:
                            skipped.append((f"{folder.name}/{file_path.name}", "Duplicate of exemplar"))
                            continue

                    # Get relative path from provenance or folder mapping
                    relative_path = self._get_log_relative_path(file_path, folder, provenance)

                    # If no mapping exists (unknown folder), skip file
                    if relative_path is None:
                        skipped.append((f"{folder.name}/{file_path.name}", "Unknown folder, no path mapping"))
                        continue

                    # Determine export path
                    if structure == ExportStructure.FLAT:
                        dest_path = self._get_log_export_path_flat(
                            export_dir, folder.name, file_path, folder, artifact_type
                        )
                    else:
                        dest_path = self._get_log_export_path_full(export_dir, relative_path)

                    # Export file
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    self._export_file(file_path, dest_path, method)

                    exported.append(
                        ExportedLog(
                            folder_name=folder.name,
                            original_filename=file_path.name,
                            relative_path=relative_path,
                            source_path=file_path,
                            export_path=dest_path,
                            md5_hash=md5,
                            artifact_type=artifact_type,
                            method=method.value,
                        )
                    )
                except Exception as e:
                    skipped.append((f"{folder.name}/{file_path.name}", str(e)))

        return exported, skipped

    def _export_logarchive(
        self,
        logarchive_path: Path,
        export_dir: Path,
        structure: ExportStructure,
        method: ExportMethod,
    ) -> ExportedLog | None:
        """Export a .logarchive bundle.

        Logarchives only exist in exemplar scans (created by combining Unified Log
        and UUID Text folders). Candidates never have logarchives, so no deduplication
        is needed.

        Args:
            logarchive_path: Path to .logarchive directory
            export_dir: Export destination directory
            structure: Directory structure (flat or full_path)
            method: Export method (copy or symlink)

        Returns:
            ExportedLog if exported, None on error
        """
        try:
            # Determine export path
            if structure == ExportStructure.FLAT:
                dest_path = export_dir / "logs" / logarchive_path.name
            else:
                # Full path: private/var/db/diagnostics/{name}.logarchive
                dest_path = export_dir / "private" / "var" / "db" / "diagnostics" / logarchive_path.name

            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # Export the entire directory bundle
            if method == ExportMethod.SYMLINK:
                try:
                    dest_path.symlink_to(logarchive_path)
                except OSError:
                    # Fallback to copy
                    shutil.copytree(logarchive_path, dest_path, dirs_exist_ok=True)
            else:
                shutil.copytree(logarchive_path, dest_path, dirs_exist_ok=True)

            return ExportedLog(
                folder_name="_logarchive",
                original_filename=logarchive_path.name,
                relative_path=f"private/var/db/diagnostics/{logarchive_path.name}",
                source_path=logarchive_path,
                export_path=dest_path,
                md5_hash="",  # Not computed for directory bundles
                artifact_type="log",
                method=method.value,
            )
        except Exception as e:
            logger.warning(f"Failed to export logarchive {logarchive_path}: {e}")
            return None

    def _read_log_provenance(self, folder: Path) -> dict | None:
        """Read provenance file for a log folder.

        Args:
            folder: Log folder path

        Returns:
            Provenance dict or None if not found
        """
        # Look for {folder_name}_provenance.json
        provenance_file = folder / f"{folder.name}_provenance.json"
        if provenance_file.exists():
            try:
                with provenance_file.open() as f:
                    return json.load(f)
            except Exception:
                pass

        # Also check for *_provenance.json pattern
        for prov_file in folder.glob("*_provenance.json"):
            try:
                with prov_file.open() as f:
                    return json.load(f)
            except Exception:
                pass

        return None

    def _get_log_relative_path(self, file_path: Path, folder: Path, provenance: dict | None) -> str | None:
        """Get the original macOS relative path for a log file.

        Args:
            file_path: Current file path
            folder: Parent log folder
            provenance: Provenance dict if available

        Returns:
            Relative macOS path string, or None if folder not in mapping (should skip)
        """
        # Try to find in provenance (exemplar files have this)
        if provenance and "files" in provenance:
            for file_info in provenance["files"]:
                if file_info.get("original_filename") == file_path.name:
                    # Check for relative_path first (clean path)
                    if "relative_path" in file_info:
                        return file_info["relative_path"]
                    if "source_path" in file_info:
                        source = file_info["source_path"]
                        # Extract path after dfvfs temp folder
                        # Pattern: /var/folders/.../mars_dfvfs_export_.../actual/path
                        if "mars_dfvfs_export_" in source:
                            # Find the temp folder marker and take path after it
                            parts = source.split("/")
                            for i, part in enumerate(parts):
                                if part.startswith("mars_dfvfs_export_"):
                                    # Everything after this is the real path
                                    real_path = "/".join(parts[i + 1 :])
                                    return real_path.lstrip("/")
                        # Fallback: strip leading / and return
                        return source.lstrip("/")

        # No provenance - use folder name mapping (candidate carved files)
        folder_name = folder.name
        username = None

        # Handle user-scoped folders (e.g., "AirDrop Hash_username1" → base="AirDrop Hash", username="username1")
        if "_" in folder_name:
            parts = folder_name.rsplit("_", 1)
            base_name = parts[0]
            potential_username = parts[1]
            # Check if base name exists in mapping (indicates user-scoped folder)
            if base_name in LOG_FOLDER_TO_PATH:
                folder_name = base_name
                username = potential_username

        if folder_name in LOG_FOLDER_TO_PATH:
            base_path = LOG_FOLDER_TO_PATH[folder_name]
            # Replace _user placeholder with actual username
            if username and "_user" in base_path:
                base_path = base_path.replace("_user", username)
            # For file-type paths (wifi.log, system.log), use as-is
            if base_path.endswith((".log", ".plist")):
                return base_path
            # For directory-type paths (ASL, DHCP), append filename
            return f"{base_path}/{file_path.name}"

        # Unknown folder - not in mapping, caller should skip
        return None

    def _get_log_export_path_flat(
        self,
        export_dir: Path,
        folder_name: str,
        file_path: Path,
        folder: Path,
        artifact_type: str,
    ) -> Path:
        """Get flat export path for a log file.

        Args:
            export_dir: Export directory
            folder_name: Log folder name
            file_path: Source file path
            folder: Parent folder path
            artifact_type: "log" or "keychain"

        Returns:
            Destination path
        """
        # Determine base directory
        base = "logs" if artifact_type == "log" else "keychains"

        # Preserve subfolder structure within the log folder
        rel_to_folder = file_path.relative_to(folder)
        return export_dir / base / folder_name / rel_to_folder

    def _get_log_export_path_full(self, export_dir: Path, relative_path: str) -> Path:
        """Get full macOS path export location.

        Args:
            export_dir: Export directory
            relative_path: Original macOS relative path

        Returns:
            Destination path
        """
        # Clean up path
        clean_path = relative_path.lstrip("/")
        return export_dir / clean_path

    # ========== Combined Export Methods ==========

    def _build_combined_catalog(
        self, candidate_run: Path | None = None, track_data_source: bool = False
    ) -> Path | None:
        """Build a combined catalog by merging exemplar + unique candidate rows.

        Creates a temporary directory with combined databases that have:
        - All rows from exemplar (clean intact data)
        - Plus unique rows from candidate (recovered data)
        - Without duplicates

        Args:
            candidate_run: Optional specific candidate run directory
            track_data_source: If True, add data_source column to track row provenance

        Returns:
            Path to temporary combined catalog directory, or None on error
        """
        # Get exemplar and candidate catalog directories
        exemplar_catalog = self.output_dir / "exemplar" / "databases" / "catalog"
        if not exemplar_catalog.exists():
            logger.warning("Exemplar catalog not found")
            return None

        # Get candidate catalog
        if candidate_run:
            candidate_catalog = candidate_run / "databases" / "catalog"
        else:
            candidates_dir = self.output_dir / "candidates"
            if candidates_dir.exists():
                run_dirs = sorted(
                    [d for d in candidates_dir.iterdir() if d.is_dir()],
                    key=lambda d: d.name,
                    reverse=True,
                )
                if run_dirs:
                    candidate_catalog = run_dirs[0] / "databases" / "catalog"
                else:
                    # No candidate runs - just use exemplar
                    return exemplar_catalog
            else:
                return exemplar_catalog

        if not candidate_catalog.exists():
            # No candidate catalog - just use exemplar
            return exemplar_catalog

        # Create temporary combined catalog
        combined_dir = Path(tempfile.mkdtemp(prefix="mars_combined_"))
        logger.debug(f"Creating combined catalog in {combined_dir}")

        # Build mapping of candidate databases by name
        candidate_map = self._build_candidate_map(candidate_catalog)

        # Process each exemplar database
        for exemplar_folder in sorted(exemplar_catalog.iterdir()):
            if not exemplar_folder.is_dir():
                continue

            db_name = exemplar_folder.name

            # Check for nested profile structure
            profile_subfolders = self._detect_profile_subfolders(exemplar_folder)

            if profile_subfolders:
                # Multi-profile: process each profile separately
                combined_folder = combined_dir / db_name
                combined_folder.mkdir(parents=True, exist_ok=True)

                # Check if candidate has nested profiles or flat structure
                candidate_folder = candidate_map.get(db_name)
                candidate_has_profiles = False
                if candidate_folder:
                    cand_profiles = self._detect_profile_subfolders(candidate_folder)
                    candidate_has_profiles = len(cand_profiles) > 0

                for profile_folder in profile_subfolders:
                    profile_name = profile_folder.name
                    profile_key = f"{db_name}/{profile_name}"

                    if candidate_has_profiles:
                        # Candidate has matching profile structure - merge per-profile
                        self._combine_profile_folder(
                            profile_folder,
                            combined_folder / profile_name,
                            candidate_map.get(profile_key),
                            track_data_source,
                        )
                    else:
                        # Candidate has flat structure - copy exemplar profiles pristine
                        self._combine_profile_folder(
                            profile_folder,
                            combined_folder / profile_name,
                            None,  # Don't merge - keep pristine
                            track_data_source=False,  # No mixed sources here
                        )

                # If candidate has flat structure, merge based on its profile(s)
                if candidate_folder and not candidate_has_profiles:
                    # Read candidate's manifest to get profiles list
                    candidate_manifest = self._read_provenance(candidate_folder)
                    candidate_profiles = (
                        self._get_profiles_from_provenance(candidate_manifest) if candidate_manifest else []
                    )

                    if len(candidate_profiles) == 1:
                        # Single profile - merge candidate into that profile folder
                        target_profile = candidate_profiles[0]
                        target_folder = combined_folder / target_profile
                        if target_folder.exists():
                            self._merge_candidate_into_profile(candidate_folder, target_folder, track_data_source)
                        else:
                            # Profile folder doesn't exist yet - just copy candidate
                            self._copy_folder_contents(candidate_folder, target_folder)
                    elif len(candidate_profiles) > 1:
                        # Multiple profiles - merge into _multi, dedupe against all profiles
                        self._merge_candidate_into_multi(
                            candidate_folder,
                            combined_folder,
                            profile_subfolders,
                            track_data_source,
                        )
                    else:
                        # No profile info - merge into _multi by default
                        self._merge_candidate_into_multi(
                            candidate_folder,
                            combined_folder,
                            profile_subfolders,
                            track_data_source,
                        )
            else:
                # Standard folder: combine directly
                self._combine_single_folder(
                    exemplar_folder,
                    combined_dir / db_name,
                    candidate_map.get(db_name),
                    track_data_source,
                )

        # Process _multi candidate folders (no exemplar, dedupe against user exemplars)
        self._process_multi_candidates(exemplar_catalog, candidate_map, combined_dir, track_data_source)

        return combined_dir

    def _build_candidate_map(self, candidate_catalog: Path) -> dict[str, Path]:
        """Build mapping of candidate database names to their folders.

        Args:
            candidate_catalog: Path to candidate catalog directory

        Returns:
            Dict mapping db_name (and db_name/profile) to folder paths
        """
        candidate_map: dict[str, Path] = {}

        for folder in candidate_catalog.iterdir():
            if not folder.is_dir():
                continue

            db_name = folder.name
            candidate_map[db_name] = folder

            # Also check for profile subfolders
            profile_subfolders = self._detect_profile_subfolders(folder)
            for profile_folder in profile_subfolders:
                profile_key = f"{db_name}/{profile_folder.name}"
                candidate_map[profile_key] = profile_folder

        return candidate_map

    def _process_multi_candidates(
        self,
        exemplar_catalog: Path,
        candidate_map: dict[str, Path],
        combined_dir: Path,
        track_data_source: bool = False,
    ) -> None:
        """Process _multi candidate folders, deduplicating against user exemplars.

        _multi folders contain carved data that matched shared tables across multiple
        users. These rows should be deduplicated against ALL related user exemplars
        to avoid duplicating data that also exists in user-specific catalogs.

        Args:
            exemplar_catalog: Path to exemplar catalog directory
            candidate_map: Mapping of candidate names to folder paths
            combined_dir: Path to combined output directory
            track_data_source: If True, add data_source column
        """
        # Find _multi candidate folders
        multi_candidates = {name: path for name, path in candidate_map.items() if name.endswith("_multi")}

        if not multi_candidates:
            return

        # Get all exemplar folder names for grouping
        exemplar_names = {f.name for f in exemplar_catalog.iterdir() if f.is_dir()}

        for multi_name, multi_folder in multi_candidates.items():
            # Extract base name (e.g., "Mail Envelope Index" from "Mail Envelope Index_multi")
            base_name = multi_name.rsplit("_multi", 1)[0]

            # Find all user exemplars with same base name
            related_exemplars = [
                exemplar_catalog / name
                for name in exemplar_names
                if name.startswith(base_name + "_") and not name.endswith("_multi")
            ]

            if not related_exemplars:
                logger.debug(f"No related exemplars found for {multi_name}, skipping")
                continue

            # Get the candidate database file
            cand_db, _ = self._find_database_file(multi_folder, prefer_combined=False)
            if not cand_db:
                logger.debug(f"No database found in {multi_name}, skipping")
                continue

            # Create combined _multi folder
            multi_combined = combined_dir / multi_name
            multi_combined.mkdir(parents=True, exist_ok=True)

            # Copy candidate as base (since no _multi exemplar exists)
            combined_db = multi_combined / cand_db.name
            shutil.copy2(cand_db, combined_db)

            # Deduplicate against each related user exemplar
            total_removed = 0
            for exemplar_folder in related_exemplars:
                # Check if this is a multi-profile exemplar (e.g., Chrome with Guest Profile/, System Profile/)
                profile_subfolders = self._detect_profile_subfolders(exemplar_folder)
                if profile_subfolders:
                    # Dedupe against each profile's database
                    for profile_folder in profile_subfolders:
                        exemplar_db, _ = self._find_database_file(profile_folder, prefer_combined=False)
                        if exemplar_db:
                            removed = self._dedupe_against_exemplar(combined_db, exemplar_db)
                            total_removed += removed
                else:
                    # Standard single-database exemplar
                    exemplar_db, _ = self._find_database_file(exemplar_folder, prefer_combined=False)
                    if exemplar_db:
                        removed = self._dedupe_against_exemplar(combined_db, exemplar_db)
                        total_removed += removed

            if total_removed > 0:
                logger.debug(f"Deduplicated {multi_name}: removed {total_removed} rows that existed in user exemplars")

            # Copy metadata if exists
            self._copy_metadata(multi_folder, multi_combined)

    def _dedupe_against_exemplar(self, target_db: Path, exemplar_db: Path) -> int:
        """Remove rows from target that exist in exemplar.

        Args:
            target_db: Database to deduplicate (modified in place)
            exemplar_db: Reference database to check against

        Returns:
            Number of rows removed
        """
        import sqlite3

        total_removed = 0

        try:
            with sqlite3.connect(target_db) as target_con, sqlite3.connect(exemplar_db) as exemplar_con:
                # Get tables that exist in both databases
                target_cur = target_con.cursor()
                exemplar_cur = exemplar_con.cursor()

                target_tables = {
                    row[0] for row in target_cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }
                exemplar_tables = {
                    row[0]
                    for row in exemplar_cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                }

                common_tables = target_tables & exemplar_tables

                for table in common_tables:
                    if table.startswith("sqlite_"):
                        continue

                    removed = self._dedupe_table(target_con, exemplar_con, table)
                    total_removed += removed

                target_con.commit()

        except Exception as e:
            logger.warning(f"Error deduplicating {target_db.name}: {e}")

        return total_removed

    def _dedupe_table(
        self,
        target_con,
        exemplar_con,
        table_name: str,
    ) -> int:
        """Remove rows from target table that exist in exemplar table.

        Args:
            target_con: Connection to target database
            exemplar_con: Connection to exemplar database
            table_name: Name of table to deduplicate

        Returns:
            Number of rows removed
        """
        qname = quote_ident(table_name)

        # Get column info
        target_cur = target_con.cursor()
        target_cur.execute(f"PRAGMA table_info({qname})")
        cols_info = target_cur.fetchall()

        if not cols_info:
            return 0

        # Common ID column names to exclude from hash
        id_column_names = {"id", "rowid", "z_pk", "z_ent", "z_opt", "data_source"}
        pk_columns = {row[1] for row in cols_info if row[5] > 0}

        # Columns to use for comparison (exclude PK and ID columns)
        all_cols = [row[1] for row in cols_info]
        hash_cols = [c for c in all_cols if c not in pk_columns and c.lower() not in id_column_names]

        if not hash_cols:
            return 0

        # Get all exemplar row hashes
        exemplar_cur = exemplar_con.cursor()
        quoted_hash_cols = ", ".join(quote_ident(c) for c in hash_cols)

        try:
            exemplar_cur.execute(f"SELECT {quoted_hash_cols} FROM {qname}")
            exemplar_hashes = {hash(tuple(row)) for row in exemplar_cur.fetchall()}
        except Exception:
            return 0  # Table doesn't exist or columns don't match

        if not exemplar_hashes:
            return 0

        # Find and delete matching rows in target
        quoted_all_cols = ", ".join(quote_ident(c) for c in all_cols)
        target_cur.execute(f"SELECT rowid, {quoted_all_cols} FROM {qname}")
        rows_to_delete = []

        for row in target_cur.fetchall():
            rowid = row[0]
            # Build hash from only hash_cols
            hash_values = tuple(row[1 + all_cols.index(c)] for c in hash_cols)
            if hash(hash_values) in exemplar_hashes:
                rows_to_delete.append(rowid)

        if rows_to_delete:
            placeholders = ",".join("?" * len(rows_to_delete))
            target_cur.execute(
                f"DELETE FROM {qname} WHERE rowid IN ({placeholders})",
                rows_to_delete,
            )

        return len(rows_to_delete)

    def _combine_profile_folder(
        self,
        exemplar_folder: Path,
        combined_folder: Path,
        candidate_folder: Path | None,
        track_data_source: bool = False,
    ) -> None:
        """Combine an exemplar profile folder with candidate data.

        Args:
            exemplar_folder: Path to exemplar profile folder
            combined_folder: Path to output combined folder
            candidate_folder: Path to matching candidate folder (if any)
            track_data_source: If True, add data_source column to track row provenance
        """
        combined_folder.mkdir(parents=True, exist_ok=True)

        # Find exemplar database
        exemplar_db, _ = self._find_database_file(exemplar_folder, prefer_combined=False)
        if not exemplar_db:
            # Just copy metadata
            self._copy_metadata(exemplar_folder, combined_folder)
            return

        combined_db = combined_folder / exemplar_db.name

        # Create combined database with proper schema
        # Use _create_combined_database to avoid ALTER TABLE corruption issues
        if track_data_source:
            self._create_combined_database(combined_db, exemplar_db, add_data_source=True, source_tag="exemplar")
        else:
            shutil.copy2(exemplar_db, combined_db)

        # If we have a candidate, merge unique rows
        if candidate_folder:
            candidate_db, _ = self._find_database_file(candidate_folder, prefer_combined=False)
            if candidate_db:
                rows_added = self._merge_unique_rows(combined_db, candidate_db, track_data_source)
                if rows_added > 0:
                    logger.debug(f"Merged {rows_added} unique rows into {combined_db.name}")

        # Copy metadata files
        self._copy_metadata(exemplar_folder, combined_folder)

    def _combine_single_folder(
        self,
        exemplar_folder: Path,
        combined_folder: Path,
        candidate_folder: Path | None,
        track_data_source: bool = False,
    ) -> None:
        """Combine an exemplar folder with candidate data.

        Args:
            exemplar_folder: Path to exemplar folder
            combined_folder: Path to output combined folder
            candidate_folder: Path to matching candidate folder (if any)
            track_data_source: If True, add data_source column to track row provenance
        """
        combined_folder.mkdir(parents=True, exist_ok=True)

        # Find exemplar database
        exemplar_db, _ = self._find_database_file(exemplar_folder, prefer_combined=False)
        if not exemplar_db:
            # Just copy metadata
            self._copy_metadata(exemplar_folder, combined_folder)
            return

        combined_db = combined_folder / exemplar_db.name

        # Create combined database with proper schema
        # Use _create_combined_database to avoid ALTER TABLE corruption issues
        if track_data_source:
            self._create_combined_database(combined_db, exemplar_db, add_data_source=True, source_tag="exemplar")
        else:
            shutil.copy2(exemplar_db, combined_db)

        # If we have a candidate, merge unique rows
        if candidate_folder:
            candidate_db, _ = self._find_database_file(candidate_folder, prefer_combined=False)
            if candidate_db:
                rows_added = self._merge_unique_rows(combined_db, candidate_db, track_data_source)
                if rows_added > 0:
                    logger.debug(f"Merged {rows_added} unique rows into {combined_folder.name}")

        # Copy metadata files (provenance, etc.)
        self._copy_metadata(exemplar_folder, combined_folder)

    def _copy_metadata(self, source_folder: Path, dest_folder: Path) -> None:
        """Copy metadata files (provenance, manifest) from source to dest.

        Args:
            source_folder: Source folder with metadata
            dest_folder: Destination folder
        """
        for pattern in ["*.provenance.json", "*_manifest.json", "*.rubric.json"]:
            for meta_file in source_folder.glob(pattern):
                shutil.copy2(meta_file, dest_folder / meta_file.name)

    def _merge_candidate_into_profile(
        self,
        candidate_folder: Path,
        target_profile_folder: Path,
        track_data_source: bool,
    ) -> None:
        """Merge candidate database into an existing profile folder.

        When candidate matches a single profile, merge its data into the
        corresponding exemplar profile folder with deduplication.

        Args:
            candidate_folder: Path to candidate catalog folder (flat structure)
            target_profile_folder: Path to target profile folder to merge into
            track_data_source: If True, add data_source column and tag rows
        """
        # Find database files
        candidate_db, _ = self._find_database_file(candidate_folder, prefer_combined=False)
        profile_db, _ = self._find_database_file(target_profile_folder, prefer_combined=False)

        if not candidate_db:
            logger.debug(f"No candidate database found in {candidate_folder}")
            return

        if not profile_db:
            # No profile database - copy candidate as the profile database
            target_profile_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_db, target_profile_folder / candidate_db.name)
            self._copy_metadata(candidate_folder, target_profile_folder)
            logger.debug(f"Copied candidate to profile folder: {candidate_db.name}")
            return

        # Merge unique rows from candidate into profile
        rows_added = self._merge_unique_rows(profile_db, candidate_db, track_data_source)
        logger.debug(f"Merged {rows_added} unique rows from candidate into {target_profile_folder.name}")

    def _merge_candidate_into_multi(
        self,
        candidate_folder: Path,
        combined_folder: Path,
        profile_folders: list[Path],
        track_data_source: bool,
    ) -> None:
        """Merge flat candidate into _multi folder, deduplicating against all profiles.

        When candidate matches multiple profiles, put it in _multi folder and
        remove any rows that already exist in any of the exemplar profile folders.

        Args:
            candidate_folder: Path to candidate catalog folder (flat structure)
            combined_folder: Path to combined output folder (contains profile subfolders)
            profile_folders: List of exemplar profile folders to dedupe against
            track_data_source: If True, add data_source column and tag rows
        """
        # Find candidate database
        candidate_db, _ = self._find_database_file(candidate_folder, prefer_combined=False)
        if not candidate_db:
            logger.debug(f"No candidate database found in {candidate_folder}")
            return

        # Create _multi folder
        multi_folder = combined_folder / "_multi"
        multi_folder.mkdir(parents=True, exist_ok=True)

        # Copy candidate database to _multi
        dest_db = multi_folder / candidate_db.name
        shutil.copy2(candidate_db, dest_db)

        # Copy metadata
        self._copy_metadata(candidate_folder, multi_folder)

        # Deduplicate against each profile database
        for profile_folder in profile_folders:
            profile_db, _ = self._find_database_file(profile_folder, prefer_combined=False)
            if profile_db and profile_db.exists():
                self._deduplicate_against(dest_db, profile_db)
                logger.debug(f"Deduplicated _multi against {profile_folder.name}")

    def _deduplicate_against(self, target_db: Path, source_db: Path) -> int:
        """Remove rows from target that already exist in source.

        Compares tables by row content (excluding data_source column) and
        deletes matching rows from target.

        Args:
            target_db: Path to target database (will be modified)
            source_db: Path to source database (read only)

        Returns:
            Number of rows removed
        """
        total_removed = 0

        try:
            with sqlite3.connect(target_db) as target_conn, readonly_connection(source_db) as source_conn:
                # Get common tables
                target_tables = self._get_user_tables(target_conn)
                source_tables = self._get_user_tables(source_conn)
                common_tables = set(target_tables) & set(source_tables)

                for table_name in common_tables:
                    try:
                        # Get columns (excluding data_source for comparison)
                        target_conn.row_factory = sqlite3.Row
                        cur = target_conn.execute(f"PRAGMA table_info({quote_ident(table_name)})")
                        columns = [row["name"] for row in cur.fetchall() if row["name"] != "data_source"]

                        if not columns:
                            continue

                        # Create comparison query
                        col_list = ", ".join(quote_ident(c) for c in columns)

                        # Get row hashes from source
                        source_cur = source_conn.execute(f"SELECT {col_list} FROM {quote_ident(table_name)}")
                        source_rows = set()
                        for row in source_cur:
                            row_hash = hash(tuple(row))
                            source_rows.add(row_hash)

                        if not source_rows:
                            continue

                        # Find and delete matching rows in target
                        target_cur = target_conn.execute(f"SELECT rowid, {col_list} FROM {quote_ident(table_name)}")
                        rows_to_delete = []
                        for row in target_cur:
                            rowid = row[0]
                            row_hash = hash(tuple(row[1:]))
                            if row_hash in source_rows:
                                rows_to_delete.append(rowid)

                        if rows_to_delete:
                            placeholders = ",".join("?" * len(rows_to_delete))
                            target_conn.execute(
                                f"DELETE FROM {quote_ident(table_name)} WHERE rowid IN ({placeholders})",
                                rows_to_delete,
                            )
                            total_removed += len(rows_to_delete)

                    except sqlite3.Error as e:
                        logger.debug(f"Error deduplicating table {table_name}: {e}")
                        continue

                target_conn.commit()

        except sqlite3.Error as e:
            logger.warning(f"Error deduplicating databases: {e}")

        return total_removed

    def _copy_folder_contents(self, source_folder: Path, dest_folder: Path) -> None:
        """Copy all contents from source folder to destination.

        Args:
            source_folder: Path to source folder
            dest_folder: Path to destination folder
        """
        dest_folder.mkdir(parents=True, exist_ok=True)

        for item in source_folder.iterdir():
            if item.name.startswith("."):
                continue
            dest_path = dest_folder / item.name
            if item.is_file():
                shutil.copy2(item, dest_path)
            elif item.is_dir():
                shutil.copytree(item, dest_path)

    def _merge_unique_rows(self, base_db: Path, candidate_db: Path, track_data_source: bool = False) -> int:
        """Merge unique rows from candidate into base database.

        Compares tables by primary key or row hash and inserts
        rows that exist only in candidate into base.

        Args:
            base_db: Path to base database (will be modified)
            candidate_db: Path to candidate database (read only)
            track_data_source: If True, add data_source column and tag rows

        Returns:
            Total number of rows added
        """
        # Backup before ALTER TABLE to recover from schema corruption
        backup_path = None
        if track_data_source:
            backup_path = base_db.with_suffix(".sqlite.backup")
            try:
                shutil.copy2(base_db, backup_path)
            except OSError:
                backup_path = None

        total_added = 0
        schema_error = False

        try:
            with (
                sqlite3.connect(base_db) as base_conn,
                readonly_connection(candidate_db) as cand_conn,
            ):
                # Handle BLOB data stored in TEXT columns (SQLite dynamic typing)
                base_conn.text_factory = _robust_text_factory
                cand_conn.text_factory = _robust_text_factory

                # Get common tables (skip system and L&F tables)
                base_tables = self._get_user_tables(base_conn)
                cand_tables = self._get_user_tables(cand_conn)
                common_tables = set(base_tables) & set(cand_tables)

                for table_name in common_tables:
                    try:
                        rows_added = self._merge_table_rows(base_conn, cand_conn, table_name, track_data_source)
                        total_added += rows_added
                    except sqlite3.Error as e:
                        err = str(e).lower()
                        if "malformed database schema" in err or "after add column" in err:
                            logger.debug(f"Schema error in {table_name}, will retry: {e}")
                            schema_error = True
                            break
                        logger.debug(f"Error merging table {table_name}: {e}")
                        continue

                if not schema_error:
                    base_conn.commit()

        except sqlite3.Error as e:
            if "malformed database schema" in str(e).lower():
                schema_error = True
            else:
                logger.warning(f"Error merging databases: {e}")

        # Restore from backup and retry without data_source tracking
        if schema_error and backup_path and backup_path.exists():
            logger.debug(f"Restoring {base_db.name} from backup, retrying without data_source")
            try:
                shutil.copy2(backup_path, base_db)
                total_added = self._merge_unique_rows(base_db, candidate_db, track_data_source=False)
            except OSError as e:
                logger.warning(f"Could not restore backup: {e}")

        # Clean up backup
        if backup_path and backup_path.exists():
            with contextlib.suppress(OSError):
                backup_path.unlink()

        return total_added

    def _get_user_tables(self, conn: sqlite3.Connection) -> list[str]:
        """Get list of user tables (excluding system, L&F, and FTS tables).

        FTS virtual tables are excluded because:
        1. They may use unknown tokenizers (e.g., Apple's ab_cf_tokenizer)
        2. Their data is derived from content tables, not stored directly
        3. They're handled separately in _create_combined_database()

        Args:
            conn: Database connection

        Returns:
            List of table names
        """
        cur = conn.cursor()
        # Get tables with their CREATE statements to identify FTS tables
        cur.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' "
            "AND name NOT LIKE 'sqlite_%' "
            "AND name NOT LIKE 'lost_and_found%' "
            "AND name NOT LIKE 'lf_%'"
        )
        tables = []
        for name, sql in cur.fetchall():
            # Skip FTS virtual tables (they may have unknown tokenizers)
            if sql and re.search(r"USING\s+fts[345]", sql, re.IGNORECASE):
                continue
            # Also skip FTS shadow tables (e.g., entity_fts_content, _docsize, _segdir)
            if re.match(r".*_(?:content|docsize|segdir|segments|stat|config|data|idx)$", name, re.IGNORECASE):
                continue
            tables.append(name)
        return tables

    def _get_table_primary_key(self, conn: sqlite3.Connection, table_name: str) -> str | None:
        """Get primary key column for a table.

        Args:
            conn: Database connection
            table_name: Table name

        Returns:
            Primary key column name or None
        """
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

    def _get_table_columns(self, conn: sqlite3.Connection, table_name: str) -> list[str]:
        """Get column names for a table.

        Args:
            conn: Database connection
            table_name: Table name

        Returns:
            List of column names
        """
        try:
            cur = conn.cursor()
            cur.execute(f"PRAGMA table_info({quote_ident(table_name)})")
            return [row[1] for row in cur.fetchall()]
        except sqlite3.Error:
            return []

    def _substitute_fts_tokenizer(self, create_sql: str) -> str:
        """Substitute unknown FTS tokenizers with unicode61.

        Apple databases use custom tokenizers (like ab_cf_tokenizer based on
        CFStringTokenizer) that aren't available in standard SQLite. This
        substitutes them with unicode61 which provides similar Unicode-aware
        word boundary detection.

        Args:
            create_sql: Original CREATE VIRTUAL TABLE statement

        Returns:
            Modified statement with standard tokenizer, or original if no change needed
        """
        # Only process FTS tables
        if not re.search(r"USING\s+fts[345]", create_sql, re.IGNORECASE):
            return create_sql

        # Built-in tokenizers that don't need substitution
        builtin_tokenizers = {"unicode61", "ascii", "porter", "trigram"}

        # Try to extract tokenizer specification
        # Format 1: tokenize='...' or tokenize="..." (quoted)
        tokenize_match = re.search(r"tokenize\s*=\s*['\"]([^'\"]+)['\"]", create_sql, re.IGNORECASE)
        if tokenize_match:
            tokenizer_spec = tokenize_match.group(1)
            tokenizer_name = tokenizer_spec.split()[0].lower()

            if tokenizer_name in builtin_tokenizers:
                return create_sql

            # Substitute with unicode61
            logger.debug(f"Substituting unknown tokenizer '{tokenizer_name}' with 'unicode61'")
            new_sql = re.sub(
                r"(tokenize\s*=\s*['\"])[^'\"]+(['\"])",
                r"\1unicode61\2",
                create_sql,
                flags=re.IGNORECASE,
            )
            return new_sql

        # Format 2: tokenize=name or tokenize=name arg1 arg2 (unquoted)
        # Match tokenize= followed by everything up to the next , or )
        # This captures the full tokenize spec including any arguments
        tokenize_match = re.search(r"tokenize\s*=\s*([^,)]+)", create_sql, re.IGNORECASE)
        if tokenize_match:
            tokenizer_spec = tokenize_match.group(1).strip()
            # Skip if this looks quoted (handled above)
            if tokenizer_spec.startswith(("'", '"')):
                return create_sql

            tokenizer_name = tokenizer_spec.split()[0].lower()

            if tokenizer_name in builtin_tokenizers:
                return create_sql

            # Substitute entire tokenize spec with unicode61 (quoted for safety)
            logger.debug(f"Substituting unknown unquoted tokenizer '{tokenizer_name}' with 'unicode61'")
            logger.debug(f"  Original SQL: {create_sql[:200]}...")
            new_sql = re.sub(
                r"tokenize\s*=\s*[^,)]+",
                "tokenize='unicode61'",
                create_sql,
                flags=re.IGNORECASE,
            )
            logger.debug(f"  Modified SQL: {new_sql[:200]}...")
            return new_sql

        # No tokenizer specified, use default
        return create_sql

    def _add_data_source_to_schema(self, create_sql: str) -> str:
        """Modify a CREATE TABLE statement to include data_source column.

        This avoids using ALTER TABLE which can corrupt databases with
        Apple-specific schema extensions that Python's SQLite can't parse.

        Args:
            create_sql: Original CREATE TABLE statement

        Returns:
            Modified CREATE TABLE statement with data_source TEXT column
        """
        # Skip if data_source already present
        if "data_source" in create_sql.lower():
            return create_sql

        # Skip FTS virtual tables - don't modify them
        if re.search(r"USING\s+fts[345]", create_sql, re.IGNORECASE):
            return create_sql

        # Remove STRICT keyword to allow BLOB data in TEXT columns
        create_sql = re.sub(r"\s+STRICT\b", "", create_sql, flags=re.IGNORECASE)

        # Remove trailing ) WITHOUT ROWID or ); or )
        stripped_sql = create_sql.rstrip()
        has_semicolon = stripped_sql.endswith(");")
        has_without_rowid = bool(re.search(r"\)\s*WITHOUT\s+ROWID", stripped_sql, re.IGNORECASE))

        # Remove trailing bits
        if has_without_rowid:
            stripped_sql = re.sub(
                r"\)\s*WITHOUT\s+ROWID\s*;?\s*$",
                "",
                stripped_sql,
                flags=re.IGNORECASE,
            )
        elif has_semicolon:
            stripped_sql = stripped_sql[:-2]
        elif stripped_sql.endswith(")"):
            stripped_sql = stripped_sql[:-1]

        # Look for table-level constraints (must come AFTER all column definitions)
        constraint_pattern = (
            r",\s*("
            r"PRIMARY\s+KEY\s*\(|"
            r"FOREIGN\s+KEY\s*\(|"
            r"UNIQUE\s*\(|"
            r"CHECK\s*\(|"
            r"CONSTRAINT\s+\w+"
            r")"
        )
        match = re.search(constraint_pattern, stripped_sql, re.IGNORECASE)

        if match:
            # Insert data_source BEFORE the first table-level constraint
            insertion_point = match.start()
            create_sql = stripped_sql[:insertion_point] + ",\n    data_source TEXT" + stripped_sql[insertion_point:]
        else:
            # No table-level constraints, add at the end
            create_sql = stripped_sql + ",\n    data_source TEXT"

        # Re-add closing paren and optional clauses
        create_sql += "\n)"
        if has_without_rowid:
            create_sql += " WITHOUT ROWID"
        if has_semicolon:
            create_sql += ";"

        return create_sql

    def _create_combined_database(
        self,
        output_db: Path,
        source_db: Path,
        add_data_source: bool = True,
        source_tag: str = "exemplar",
    ) -> None:
        """Create a new database with schema from source, optionally adding data_source column.

        This creates a fresh database instead of copying and using ALTER TABLE,
        which avoids schema corruption issues with Apple-specific SQL extensions.

        Args:
            output_db: Path for new database
            source_db: Path to source database (exemplar)
            add_data_source: If True, add data_source column to all tables
            source_tag: Value to use for data_source column (e.g., 'exemplar')
        """
        # Remove existing file if present
        if output_db.exists():
            output_db.unlink()

        with (
            sqlite3.connect(output_db) as dst_conn,
            readonly_connection(source_db) as src_conn,
        ):
            # Handle BLOB data stored in TEXT columns
            dst_conn.text_factory = _robust_text_factory
            src_conn.text_factory = _robust_text_factory

            dst_conn.execute("PRAGMA journal_mode=WAL")
            dst_conn.execute("PRAGMA synchronous=NORMAL")

            src_cur = src_conn.cursor()
            dst_cur = dst_conn.cursor()

            # Get all tables and their schemas
            tables = src_cur.execute(
                "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()

            # Track FTS virtual tables for later rebuild
            fts_tables: list[tuple[str, str]] = []

            for table_name, create_sql in tables:
                if not create_sql:
                    continue

                # Track FTS virtual tables for later rebuild
                is_fts = bool(re.search(r"USING\s+fts[345]", create_sql, re.IGNORECASE))
                if is_fts:
                    fts_tables.append((table_name, create_sql))
                    # Substitute unknown tokenizers (like Apple's ab_cf_tokenizer) with unicode61
                    modified_sql = self._substitute_fts_tokenizer(create_sql)
                else:
                    # Modify schema to include data_source if requested
                    modified_sql = self._add_data_source_to_schema(create_sql) if add_data_source else create_sql

                # Create the table
                try:
                    dst_cur.execute(modified_sql)
                except sqlite3.Error as e:
                    err_msg = str(e).lower()
                    if "already exists" in err_msg:
                        # Table exists (e.g., FTS shadow table auto-created) - still copy data
                        logger.debug(f"Table {table_name} already exists, will copy data")
                    else:
                        logger.debug(f"Could not create table {table_name}: {e}")
                        continue

                # Skip data copy for FTS virtual tables - they either:
                # 1. Derive data from content tables (external content) - rebuilt later
                # 2. Have data in shadow tables (internal content) - copied separately
                # Attempting to SELECT from source FTS would fail if it uses unknown tokenizers
                # Note: FTS shadow tables (like messages_content) are NOT virtual tables
                # and should have their data copied normally
                if is_fts:
                    continue

                # Get column names from source table
                src_cols = [
                    row[1] for row in src_cur.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
                ]
                if not src_cols:
                    continue

                # Copy data from source to destination
                quoted_cols = ", ".join(quote_ident(c) for c in src_cols)
                qname = quote_ident(table_name)

                # Check if destination table actually has data_source column
                # (FTS virtual tables and others may not have it even if we requested it)
                dst_cols = [
                    row[1] for row in dst_cur.execute(f"PRAGMA table_info({quote_ident(table_name)})").fetchall()
                ]
                dst_has_data_source = "data_source" in [c.lower() for c in dst_cols]

                # If tracking data_source and destination has the column, add the tag to each row
                # Use INSERT OR IGNORE to handle any edge cases with duplicate PKs
                if add_data_source and dst_has_data_source and "data_source" not in [c.lower() for c in src_cols]:
                    # Source doesn't have data_source but destination does, add it
                    insert_cols = quoted_cols + ", data_source"
                    placeholders = ", ".join(["?"] * len(src_cols)) + ", ?"

                    rows = src_cur.execute(f"SELECT {quoted_cols} FROM {qname}").fetchall()
                    for row in rows:
                        dst_cur.execute(
                            f"INSERT OR IGNORE INTO {qname} ({insert_cols}) VALUES ({placeholders})",
                            (*row, source_tag),
                        )
                else:
                    # Just copy as-is (FTS tables, tables that already have data_source, etc.)
                    placeholders = ", ".join(["?"] * len(src_cols))
                    rows = src_cur.execute(f"SELECT {quoted_cols} FROM {qname}").fetchall()
                    for row in rows:
                        dst_cur.execute(
                            f"INSERT OR IGNORE INTO {qname} ({quoted_cols}) VALUES ({placeholders})",
                            row,
                        )

            # Copy indexes (excluding auto-generated ones)
            indexes = src_cur.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
            for (index_sql,) in indexes:
                with contextlib.suppress(sqlite3.Error):
                    dst_cur.execute(index_sql)

            # Rebuild external content FTS tables
            # These tables reference separate content tables and need to be re-indexed
            # after the content is copied. See: https://www.sqlite.org/fts5.html
            for fts_name, fts_sql in fts_tables:
                # Check if this is an external content table (content='table_name')
                if "content=" in fts_sql.lower() or "content =" in fts_sql.lower():
                    try:
                        q_fts = quote_ident(fts_name)
                        dst_cur.execute(f"INSERT INTO {q_fts}({q_fts}) VALUES('rebuild');")
                        logger.debug(f"Rebuilt FTS index for: {fts_name}")
                    except sqlite3.OperationalError as e:
                        logger.debug(f"Could not rebuild FTS index for {fts_name}: {e}")

            dst_conn.commit()

    def _add_data_source_column(self, conn: sqlite3.Connection, table_name: str) -> bool:
        """Add data_source column to a table if not present.

        NOTE: This uses ALTER TABLE which can fail on databases with Apple-specific
        schema extensions. Prefer _create_combined_database() for new databases.

        Args:
            conn: Database connection
            table_name: Table name

        Returns:
            True if column was added or already exists
        """
        try:
            # Check if column already exists
            cols = self._get_table_columns(conn, table_name)
            if "data_source" in cols:
                return True

            # Add the column
            qname = quote_ident(table_name)
            cur = conn.cursor()
            cur.execute(f"ALTER TABLE {qname} ADD COLUMN data_source TEXT")
            return True
        except sqlite3.Error as e:
            logger.debug(f"Could not add data_source column to {table_name}: {e}")
            return False

    def _tag_existing_rows(self, conn: sqlite3.Connection, table_name: str, source_tag: str) -> int:
        """Tag existing rows with a data_source value.

        Args:
            conn: Database connection
            table_name: Table name
            source_tag: Value to set for data_source (e.g., 'exemplar')

        Returns:
            Number of rows tagged
        """
        try:
            qname = quote_ident(table_name)
            cur = conn.cursor()
            # Only tag rows where data_source is NULL (not already tagged)
            cur.execute(
                f"UPDATE {qname} SET data_source = ? WHERE data_source IS NULL",
                (source_tag,),
            )
            return cur.rowcount
        except sqlite3.Error as e:
            logger.debug(f"Could not tag rows in {table_name}: {e}")
            return 0

    def _merge_table_rows(
        self,
        base_conn: sqlite3.Connection,
        cand_conn: sqlite3.Connection,
        table_name: str,
        track_data_source: bool = False,
    ) -> int:
        """Merge unique rows from candidate table into base table.

        Args:
            base_conn: Base database connection (writable)
            cand_conn: Candidate database connection (read only)
            table_name: Table to merge
            track_data_source: If True, add data_source column and tag rows

        Returns:
            Number of rows added
        """
        # Get columns from both tables
        base_cols = self._get_table_columns(base_conn, table_name)
        cand_cols = self._get_table_columns(cand_conn, table_name)

        # Check if candidate has data_source column (preserve original values)
        cand_has_data_source = "data_source" in cand_cols

        # Find common columns (exclude data_source - we handle it separately)
        common_cols = [c for c in base_cols if c in cand_cols and c != "data_source"]
        if not common_cols:
            logger.debug(f"No common columns for {table_name}, skipping merge")
            return 0

        # Check if base already has data_source column (should be there from _create_combined_database)
        # Do NOT use ALTER TABLE to add it - that can corrupt databases with Apple-specific schemas
        base_has_data_source = "data_source" in base_cols
        if track_data_source and not base_has_data_source:
            # Column wasn't created during database setup - skip data_source tracking for this table
            # but continue with the merge
            logger.debug(f"Table {table_name} missing data_source column, skipping provenance tracking")
            track_data_source = False

        # Always use content hash comparison, excluding PK/ID columns
        # PK comparison is unreliable because recovered data gets new PKs
        return self._merge_by_hash(
            base_conn,
            cand_conn,
            table_name,
            common_cols,
            track_data_source,
            cand_has_data_source,
        )

    def _merge_by_hash(
        self,
        base_conn: sqlite3.Connection,
        cand_conn: sqlite3.Connection,
        table_name: str,
        common_cols: list[str],
        track_data_source: bool = False,
        cand_has_data_source: bool = False,
    ) -> int:
        """Merge rows using row hash comparison, excluding PK/ID columns.

        Excludes primary key and common ID columns from hash because recovered
        data often has different IDs for the same content.

        Args:
            base_conn: Base database connection
            cand_conn: Candidate database connection
            table_name: Table name
            common_cols: List of columns that exist in both databases
            track_data_source: If True, tag inserted rows with data_source
            cand_has_data_source: If True, preserve candidate's data_source values

        Returns:
            Number of rows added
        """
        qname = quote_ident(table_name)

        # Common ID column names that are auto-generated and should be excluded from hash
        id_column_names = {"id", "rowid", "z_pk", "z_ent", "z_opt"}

        # Get PK columns from base table
        base_cur = base_conn.cursor()
        base_cur.execute(f"PRAGMA table_info({qname})")
        cols_info = base_cur.fetchall()
        pk_columns = {row[1] for row in cols_info if row[5] > 0}

        # Columns to use for hashing (exclude PK and common ID columns)
        hash_cols = [c for c in common_cols if c not in pk_columns and c.lower() not in id_column_names]

        if not hash_cols:
            # All columns are PK/ID - can't deduplicate meaningfully
            return 0

        # Build column indices for hash extraction
        hash_indices = [common_cols.index(c) for c in hash_cols]

        quoted_common = ", ".join(quote_ident(c) for c in common_cols)

        # Get all rows from base and hash them (using only hash_cols)
        base_cur.execute(f"SELECT {quoted_common} FROM {qname}")
        base_hashes = {hash(tuple(row[i] for i in hash_indices)) for row in base_cur.fetchall()}

        # Build select columns - include data_source if candidate has it and we're tracking
        select_cols = common_cols[:]
        if track_data_source and cand_has_data_source:
            select_cols.append("data_source")

        quoted_select = ", ".join(quote_ident(c) for c in select_cols)
        cand_cur = cand_conn.cursor()
        cand_cur.execute(f"SELECT {quoted_select} FROM {qname}")
        cand_rows = cand_cur.fetchall()

        # Find unique rows (hash not in base) - hash only non-PK/ID columns
        unique_rows = []
        for row in cand_rows:
            # Hash only the hash_cols portion (exclude PK, ID, and data_source)
            row_hash = hash(tuple(row[i] for i in hash_indices))
            if row_hash not in base_hashes:
                unique_rows.append(row)

        if not unique_rows:
            return 0

        # Build insert statement - include data_source if tracking
        if track_data_source:
            if cand_has_data_source:
                # Candidate has data_source - preserve it (already in unique_rows)
                insert_cols = common_cols + ["data_source"]
            else:
                # Candidate doesn't have data_source - add "recovered" tag
                insert_cols = common_cols + ["data_source"]
                unique_rows = [tuple(row) + ("recovered",) for row in unique_rows]

            quoted_insert = ", ".join(quote_ident(c) for c in insert_cols)
            placeholders = ", ".join(["?"] * len(insert_cols))
            insert_sql = f"INSERT OR IGNORE INTO {qname} ({quoted_insert}) VALUES ({placeholders})"
        else:
            placeholders = ", ".join(["?"] * len(common_cols))
            insert_sql = f"INSERT OR IGNORE INTO {qname} ({quoted_common}) VALUES ({placeholders})"

        base_cur.executemany(insert_sql, unique_rows)
        return len(unique_rows)
