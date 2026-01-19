#!/usr/bin/env python3
"""ARC Artifact Mapper for Time Machine extraction.

This module provides an ARCArtifactMapper class that extracts artifact folder
mappings from the artifact_recovery_catalog.yaml file. It maps artifact names
to their semantic folder destinations based on exemplar_pattern fields.

Example:
    >>> mapper = ARCArtifactMapper()
    >>> mapper.get_folder_info("Safari History")
    ("Safari History", "History")
    >>> mapper.get_folder_info("TCC Database (System)")
    ("TCC Database (System)", "TCC")
"""

from __future__ import annotations

from pathlib import Path

import yaml

from mars.utils.debug_logger import logger

# Default catalog path
DEFAULT_CATALOG_PATH = Path(__file__).parent.parent.parent / "catalog" / "artifact_recovery_catalog.yaml"


class ARCArtifactMapper:
    """Maps artifact names to exemplar folder structure from ARC catalog.

    Extracts folder mapping information from the artifact_recovery_catalog.yaml file.
    Each artifact entry's exemplar_pattern field specifies the semantic folder name
    (e.g., "databases/catalog/Safari History*" → folder "Safari History").

    Attributes:
        catalog_path: Path to artifact_recovery_catalog.yaml
        _artifact_map: Cached mapping dictionary (loaded on first access)
    """

    def __init__(self, catalog_path: Path | None = None):
        """Initialize mapper with catalog path.

        Args:
            catalog_path: Path to artifact_recovery_catalog.yaml file.
                         Defaults to built-in catalog location.
        """
        self.catalog_path = catalog_path or DEFAULT_CATALOG_PATH
        self._artifact_map: dict[str, tuple[str, str, str]] | None = None

    def _load_catalog(self) -> dict:
        """Load the artifact recovery catalog YAML.

        Returns:
            Parsed catalog dictionary

        Raises:
            FileNotFoundError: If catalog file doesn't exist
        """
        if not self.catalog_path.exists():
            raise FileNotFoundError(f"Catalog not found: {self.catalog_path}")

        with self.catalog_path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _parse_exemplar_pattern(self, exemplar_pattern: str) -> tuple[str, str]:
        """Parse exemplar_pattern to extract folder name and type prefix.

        Args:
            exemplar_pattern: Pattern like "databases/catalog/Safari History*"
                            or "logs/Wi-Fi Analytics (message tracer 1)*"

        Returns:
            Tuple of (type_prefix, folder_name)
            - type_prefix: "databases", "logs", or "caches"
            - folder_name: Semantic folder name (e.g., "Safari History")

        Example:
            >>> _parse_exemplar_pattern("databases/catalog/Safari History*")
            ("databases", "Safari History")
            >>> _parse_exemplar_pattern("logs/Wi-Fi Analytics (message tracer 1)*")
            ("logs", "Wi-Fi Analytics (message tracer 1)")
        """
        # Remove trailing wildcard and whitespace
        pattern = exemplar_pattern.rstrip("*").strip()

        # Split into parts
        parts = pattern.split("/")

        # First part is type (databases, logs, caches)
        type_prefix = parts[0] if parts else "caches"

        # For databases, may have subdir (catalog, found_data, metamatches)
        # The folder name is the last component
        folder_name = parts[-1].strip() if len(parts) >= 2 else pattern

        return type_prefix, folder_name

    def _extract_base_filename(self, glob_pattern: str) -> str:
        """Extract base filename from glob pattern.

        Args:
            glob_pattern: Path pattern like "Library/Application Support/com.apple.TCC/TCC.db"

        Returns:
            Base filename without extension (e.g., "TCC")
        """
        if not glob_pattern:
            return ""

        # Get the filename part
        path = Path(glob_pattern)
        base_filename = path.stem

        # Clean up glob characters and invalid filename chars
        if base_filename:
            # Remove glob characters (* and ?)
            base_filename = base_filename.replace("*", "").replace("?", "")
            # Remove Windows-invalid characters
            for char in '<>:"/\\|':
                base_filename = base_filename.replace(char, "")
            base_filename = base_filename.strip()

        return base_filename

    def get_artifact_map(self) -> dict[str, tuple[str, str, str]]:
        """Build artifact map from catalog (cached after first call).

        Returns:
            Dictionary mapping artifact_name to (type_prefix, folder_name, base_filename) tuples

        Example:
            {
                "Safari History": ("databases", "Safari History", "History"),
                "TCC Database (System)": ("databases", "TCC Database (System)", "TCC"),
                "Wi-Fi Analytics (message tracer 1)": ("logs", "Wi-Fi Analytics (message tracer 1)", "com.apple.wifi"),
            }
        """
        if self._artifact_map is not None:
            return self._artifact_map

        catalog = self._load_catalog()
        artifact_map: dict[str, tuple[str, str, str]] = {}

        # Iterate over all categories in catalog
        for category_name, entries in catalog.items():
            # Skip metadata and special entries
            if category_name in ("catalog_metadata", "skip_databases"):
                continue

            if not isinstance(entries, list):
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                artifact_name = entry.get("name", "")
                if not artifact_name:
                    continue

                # Get exemplar_pattern - may be at top level or nested in 'primary'
                exemplar_pattern = entry.get("exemplar_pattern", "")
                if not exemplar_pattern:
                    primary = entry.get("primary", {})
                    if isinstance(primary, dict):
                        exemplar_pattern = primary.get("exemplar_pattern", "")

                # Get glob_pattern for base filename
                glob_pattern = entry.get("glob_pattern", "")
                if not glob_pattern:
                    primary = entry.get("primary", {})
                    if isinstance(primary, dict):
                        glob_pattern = primary.get("glob_pattern", "")

                # Parse exemplar pattern for type and folder
                if exemplar_pattern:
                    type_prefix, folder_name = self._parse_exemplar_pattern(exemplar_pattern)
                else:
                    # Fallback: determine type from file_type field
                    file_type = entry.get("file_type", "").lower()
                    if file_type == "database":
                        type_prefix = "databases"
                    elif file_type == "log":
                        type_prefix = "logs"
                    else:
                        type_prefix = "caches"
                    # Use artifact name as folder name
                    folder_name = artifact_name

                # Extract base filename
                base_filename = self._extract_base_filename(glob_pattern)
                if not base_filename:
                    # Fallback: sanitize artifact name
                    base_filename = artifact_name.replace(" ", "_").replace("(", "").replace(")", "")

                artifact_map[artifact_name] = (type_prefix, folder_name, base_filename)
                logger.debug(f"Mapped artifact '{artifact_name}' -> {type_prefix}/{folder_name}")

        self._artifact_map = artifact_map
        logger.debug(f"Built ARC artifact map with {len(artifact_map)} entries")
        return artifact_map

    def get_folder_info(self, artifact_name: str) -> tuple[str, str, str] | None:
        """Get folder info for a specific artifact.

        Args:
            artifact_name: The artifact name from ARC catalog (e.g., "Safari History")

        Returns:
            Tuple of (type_prefix, folder_name, base_filename) if found, None otherwise
            - type_prefix: "databases", "logs", or "caches"
            - folder_name: Semantic folder name
            - base_filename: Base filename for the artifact

        Example:
            >>> mapper.get_folder_info("Safari History")
            ("databases", "Safari History", "History")
        """
        return self.get_artifact_map().get(artifact_name)

    def get_output_path(
        self,
        artifact_name: str,
        backup_id: str,
        original_suffix: str = "",
    ) -> tuple[str, str] | None:
        """Get the output path components for an extracted artifact.

        Args:
            artifact_name: The artifact name from ARC catalog
            backup_id: The backup identifier (e.g., "2026-01-19-131444")
            original_suffix: Original file suffix (e.g., ".db", ".plist")

        Returns:
            Tuple of (relative_dir, filename) if artifact is mapped, None otherwise
            - relative_dir: e.g., "databases/Safari History"
            - filename: e.g., "History_2026-01-19-131444.db"

        Example:
            >>> mapper.get_output_path("Safari History", "2026-01-19-131444", ".db")
            ("databases/Safari History", "History_2026-01-19-131444.db")
        """
        info = self.get_folder_info(artifact_name)
        if info is None:
            return None

        type_prefix, folder_name, base_filename = info
        relative_dir = f"{type_prefix}/{folder_name}"
        filename = f"{base_filename}_{backup_id}{original_suffix}"

        return relative_dir, filename

    def has_artifact(self, artifact_name: str) -> bool:
        """Check if an artifact name exists in the map.

        Args:
            artifact_name: The artifact name to check

        Returns:
            True if the artifact exists, False otherwise
        """
        return artifact_name in self.get_artifact_map()

    def get_all_artifact_names(self) -> list[str]:
        """Get list of all mapped artifact names.

        Returns:
            List of artifact name strings from the catalog
        """
        return list(self.get_artifact_map().keys())

    def get_artifacts_by_type(self, type_prefix: str) -> dict[str, tuple[str, str]]:
        """Get all artifacts of a specific type.

        Args:
            type_prefix: "databases", "logs", or "caches"

        Returns:
            Dictionary mapping artifact_name to (folder_name, base_filename) for matching type
        """
        result: dict[str, tuple[str, str]] = {}
        for name, (t, folder, base) in self.get_artifact_map().items():
            if t == type_prefix:
                result[name] = (folder, base)
        return result
