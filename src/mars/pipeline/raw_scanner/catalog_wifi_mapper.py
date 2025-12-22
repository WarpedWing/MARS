#!/usr/bin/env python3
"""Dynamic WiFi plist folder mapping from database catalog.

This module provides a CatalogWifiMapper class that extracts WiFi plist
categorization mappings from the artifact_recovery_catalog.yaml file.
WiFi-related entries are organized under group names (airport_preferences,
dhcp_lease, eapol, message_tracer, network_interfaces, network_preferences)
and the group name is used as the category identifier.
"""

from __future__ import annotations

from pathlib import Path


class CatalogWifiMapper:
    """Maps WiFi plist types to folder destinations using catalog data.

    Extracts folder mapping information from the artifact_recovery_catalog.yaml file.
    WiFi-related entries are organized under their group names (e.g., airport_preferences,
    dhcp_lease, etc.) and the group name is used as the category identifier.

    Attributes:
        catalog_path: Path to artifact_recovery_catalog.yaml
        _folder_map: Cached mapping dictionary (loaded on first access)
    """

    # WiFi-related group names in the catalog
    WIFI_GROUPS = frozenset(
        {
            "airport_preferences",
            "dhcp_lease",
            "eapol",
            "message_tracer",
            "network_interfaces",
            "network_preferences",
        }
    )

    def __init__(self, catalog_path: Path):
        """Initialize mapper with catalog path.

        Args:
            catalog_path: Path to artifact_recovery_catalog.yaml file
        """
        self.catalog_path = catalog_path
        self._folder_map: dict[str, tuple[str, str]] | None = None

    def get_folder_map(self) -> dict[str, tuple[str, str]]:
        """Build folder map from catalog (cached after first call).

        Returns:
            Dictionary mapping group name to (folder_name, base_filename) tuples

        Example:
            {
                "airport_preferences": ("Wi-Fi Known Networks (new)", "com.apple.airport.preferences"),
                "message_tracer": ("Wi-Fi Analytics (message tracer 1)", "com.apple.wifi.message-tracer"),
                ...
            }
        """
        if self._folder_map is not None:
            return self._folder_map

        # Import yaml here to avoid module-level dependency
        import yaml

        # Load catalog YAML
        with Path.open(self.catalog_path, encoding="utf-8") as f:
            catalog = yaml.safe_load(f)

        folder_map: dict[str, tuple[str, str]] = {}

        # Iterate over WiFi-related groups by their group name
        for group_name in self.WIFI_GROUPS:
            entries = catalog.get(group_name, [])
            if not isinstance(entries, list):
                continue

            # Use the first entry's info for the folder mapping
            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                # Extract folder name from exemplar_pattern
                # Example: "logs/Wi-Fi Known Networks (new)*" -> "Wi-Fi Known Networks (new)"
                exemplar = entry.get("exemplar_pattern", "")
                folder_name = exemplar.replace("logs/", "").rstrip("*").strip()

                if not folder_name:
                    # Fallback: use name field if exemplar_pattern is missing/malformed
                    folder_name = entry.get("name", "WiFi (Unknown)")

                # Extract base filename from glob_pattern
                # Example: "Library/Preferences/com.apple.airport.preferences.plist" -> "com.apple.airport.preferences"
                path = entry.get("glob_pattern", "")
                base_filename = Path(path).stem

                # Sanitize base_filename: remove glob characters and Windows-invalid chars
                # Paths may contain glob patterns like "preferences*.plist" or "leases/*"
                if base_filename:
                    # Remove glob characters (* and ?) that are invalid in Windows filenames
                    base_filename = base_filename.replace("*", "").replace("?", "")
                    # Also remove other Windows-invalid characters: < > : " / \ |
                    for char in '<>:"/\\|':
                        base_filename = base_filename.replace(char, "")
                    base_filename = base_filename.strip()

                if not base_filename:
                    # Fallback: use sanitized group_name if path is missing or became empty
                    base_filename = group_name.replace("_", "-")

                # Use group name as the category identifier
                folder_map[group_name] = (folder_name, base_filename)
                break  # Only use first entry per group

        self._folder_map = folder_map
        return folder_map

    def get_folder_info(self, group_name: str) -> tuple[str, str] | None:
        """Get folder info for a specific WiFi group.

        Args:
            group_name: The group name (e.g., "airport_preferences")

        Returns:
            Tuple of (folder_name, base_filename) if found, None otherwise

        Example:
            >>> mapper.get_folder_info("airport_preferences")
            ("Wi-Fi Known Networks (new)", "com.apple.airport.preferences")
        """
        return self.get_folder_map().get(group_name)

    def has_category(self, group_name: str) -> bool:
        """Check if a group name exists in the WiFi groups.

        Args:
            group_name: The group name to check

        Returns:
            True if the group exists, False otherwise
        """
        return group_name in self.get_folder_map()

    def get_all_categories(self) -> list[str]:
        """Get list of all available WiFi group names.

        Returns:
            List of group name strings from the catalog

        Example:
            ["airport_preferences", "message_tracer", "eapol", ...]
        """
        return list(self.get_folder_map().keys())
