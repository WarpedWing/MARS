"""Catalog editor module for modifying artifact_recovery_catalog.yaml.

Uses ruamel.yaml to preserve comments and formatting when editing.
"""

from __future__ import annotations

import shutil
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ruamel.yaml import YAML

if TYPE_CHECKING:
    from ruamel.yaml.comments import CommentedMap


class CatalogEditor:
    """Editor for artifact_recovery_catalog.yaml that preserves formatting."""

    # Keys that are not artifact groups
    EXCLUDED_KEYS = frozenset({"skip_databases", "catalog_metadata"})

    def __init__(self, catalog_path: Path | None = None):
        """Initialize the catalog editor.

        Args:
            catalog_path: Path to the catalog YAML file. If None, uses default location.
        """
        # Use round-trip mode to preserve formatting
        self._yaml = YAML(typ="rt")
        self._yaml.preserve_quotes = True
        self._yaml.default_flow_style = False
        # Prevent line wrapping
        self._yaml.width = 4096
        # Set indentation: mapping=2, sequence=4, offset=2
        # This matches the catalog format where list items are indented 2 from parent
        # and content after '-' is indented 2 more (total 4 from parent key)
        self._yaml.indent(mapping=2, sequence=4, offset=2)

        self.catalog_path = catalog_path or self._find_default_catalog()
        self._data: CommentedMap | None = None
        self._original_data: dict | None = None  # Store original for change comparison
        self._has_unsaved_changes = False

    def _find_default_catalog(self) -> Path:
        """Find default catalog location."""
        return Path(__file__).parent / "artifact_recovery_catalog.yaml"

    def _to_plain_dict(self, obj: Any) -> Any:
        """Recursively convert ruamel.yaml objects to plain Python types for comparison."""
        if isinstance(obj, dict):
            return {k: self._to_plain_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_plain_dict(item) for item in obj]
        return obj

    def _deep_copy_data(self) -> dict | None:
        """Create a plain dict deep copy of current data for comparison."""
        if self._data is None:
            return None
        return self._to_plain_dict(self._data)

    @property
    def has_unsaved_changes(self) -> bool:
        """Check if there are actual changes compared to original loaded data.

        Uses a two-step check: first the quick flag, then actual data comparison.
        This allows create-then-delete sequences to correctly report no changes.
        """
        if not self._has_unsaved_changes:
            return False
        # Quick flag check passed, now verify actual changes exist
        if self._original_data is None:
            return True
        current = self._deep_copy_data()
        return current != self._original_data

    def load(self) -> bool:
        """Load the catalog from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            if not self.catalog_path.exists():
                return False
            with self.catalog_path.open(encoding="utf-8") as f:
                self._data = self._yaml.load(f)
            # Store deep copy of original for change comparison
            self._original_data = self._deep_copy_data()
            self._has_unsaved_changes = False
            return True
        except Exception:
            return False

    def save(self) -> bool:
        """Save the catalog to disk.

        Creates a backup before saving and updates catalog_metadata.updated.
        Keys are reordered: catalog_metadata first, skip_databases second,
        then all artifact groups alphabetically.

        Returns:
            True if saved successfully, False otherwise.
        """
        if self._data is None:
            return False
        try:
            # Create backup before saving (keep last 3)
            self._create_backup()

            # Update catalog_metadata.updated to current date
            if "catalog_metadata" in self._data:
                self._data["catalog_metadata"]["updated"] = date.today().isoformat()

            # Reorder keys: catalog_metadata first, skip_databases second, then alphabetically
            from ruamel.yaml.comments import CommentedMap

            ordered = CommentedMap()

            # catalog_metadata first
            if "catalog_metadata" in self._data:
                ordered["catalog_metadata"] = self._data["catalog_metadata"]

            # skip_databases second (if exists)
            if "skip_databases" in self._data:
                ordered["skip_databases"] = self._data["skip_databases"]

            # Then all other groups alphabetically
            for key in sorted(self._data.keys()):
                if key not in ("catalog_metadata", "skip_databases"):
                    ordered[key] = self._data[key]

            with self.catalog_path.open("w", encoding="utf-8") as f:
                # Write document start marker
                f.write("---\n")
                self._yaml.dump(ordered, f)
            # Update original data to current state after successful save
            self._original_data = self._deep_copy_data()
            self._has_unsaved_changes = False
            return True
        except Exception:
            return False

    def _create_backup(self) -> None:
        """Create a timestamped backup, keeping only the last 3."""
        if not self.catalog_path.exists():
            return

        backup_dir = self.catalog_path.parent / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create timestamped backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"artifact_recovery_catalog_{timestamp}.yaml"
        shutil.copy(self.catalog_path, backup_path)

        # Remove old backups (keep 3)
        backups = sorted(backup_dir.glob("artifact_recovery_catalog_*.yaml"))
        for old_backup in backups[:-3]:
            old_backup.unlink()

    def get_groups(self) -> list[str]:
        """Get all artifact group names.

        Returns:
            Sorted list of group names (excluding skip_databases, catalog_metadata).
        """
        if self._data is None:
            return []
        return sorted(k for k in self._data if k not in self.EXCLUDED_KEYS)

    def get_targets(self, group: str) -> list[dict[str, Any]]:
        """Get all targets (entries) for a group.

        Args:
            group: The group name.

        Returns:
            List of target dictionaries, or empty list if group not found.
        """
        if self._data is None or group not in self._data:
            return []
        entries = self._data[group]
        if not isinstance(entries, list):
            return []
        return list(entries)

    def get_target(self, group: str, index: int) -> dict[str, Any] | None:
        """Get a specific target by group and index.

        Args:
            group: The group name.
            index: The target index (0-based).

        Returns:
            The target dictionary, or None if not found.
        """
        targets = self.get_targets(group)
        if 0 <= index < len(targets):
            return targets[index]
        return None

    def add_group(self, group_name: str) -> bool:
        """Add a new empty group.

        Args:
            group_name: Name of the new group.

        Returns:
            True if added, False if group already exists or data not loaded.
        """
        if self._data is None:
            return False
        if group_name in self._data:
            return False

        # Create an empty list for the new group
        from ruamel.yaml.comments import CommentedSeq

        self._data[group_name] = CommentedSeq()

        # Add blank line before new group for readability
        self._data.yaml_set_comment_before_after_key(group_name, before="\n")

        self._has_unsaved_changes = True
        return True

    def delete_group(self, group_name: str) -> bool:
        """Delete a group and all its targets.

        Args:
            group_name: Name of the group to delete.

        Returns:
            True if deleted, False if group not found or protected.
        """
        if self._data is None:
            return False
        if group_name in self.EXCLUDED_KEYS:
            return False
        if group_name not in self._data:
            return False

        del self._data[group_name]
        self._has_unsaved_changes = True
        return True

    def add_target(self, group: str, target: dict[str, Any]) -> bool:
        """Add a new target to a group.

        Args:
            group: The group name.
            target: The target dictionary.

        Returns:
            True if added, False if group not found.
        """
        if self._data is None or group not in self._data:
            return False

        entries = self._data[group]
        if not isinstance(entries, list):
            return False

        # Convert dict to CommentedMap to preserve YAML structure
        from ruamel.yaml.comments import CommentedMap

        commented_target = CommentedMap(target)

        entries.append(commented_target)

        # Add blank line after the new target (creates gap before next group)
        new_index = len(entries) - 1
        entries.yaml_set_comment_before_after_key(new_index, after="\n")  # type: ignore[union-attr]

        self._has_unsaved_changes = True
        return True

    def update_target(self, group: str, index: int, target: dict[str, Any]) -> bool:
        """Update an existing target.

        Args:
            group: The group name.
            index: The target index (0-based).
            target: The new target dictionary.

        Returns:
            True if updated, False if not found.
        """
        if self._data is None or group not in self._data:
            return False

        entries = self._data[group]
        if not isinstance(entries, list) or not (0 <= index < len(entries)):
            return False

        # Update in place to preserve any comments
        existing = entries[index]
        existing.clear()
        existing.update(target)
        self._has_unsaved_changes = True
        return True

    def update_target_field(self, group: str, index: int, field: str, value: Any) -> bool:
        """Update a single field of a target.

        Args:
            group: The group name.
            index: The target index (0-based).
            field: The field name to update.
            value: The new value.

        Returns:
            True if updated, False if not found.
        """
        if self._data is None or group not in self._data:
            return False

        entries = self._data[group]
        if not isinstance(entries, list) or not (0 <= index < len(entries)):
            return False

        entries[index][field] = value
        self._has_unsaved_changes = True
        return True

    def delete_target_field(self, group: str, index: int, field: str) -> bool:
        """Delete a field from a target.

        Args:
            group: The group name.
            index: The target index (0-based).
            field: The field name to delete.

        Returns:
            True if deleted, False if not found.
        """
        if self._data is None or group not in self._data:
            return False

        entries = self._data[group]
        if not isinstance(entries, list) or not (0 <= index < len(entries)):
            return False

        target = entries[index]
        if field in target:
            del target[field]
            self._has_unsaved_changes = True
            return True
        return False

    def delete_target(self, group: str, index: int) -> bool:
        """Delete a target from a group.

        Args:
            group: The group name.
            index: The target index (0-based).

        Returns:
            True if deleted, False if not found.
        """
        if self._data is None or group not in self._data:
            return False

        entries = self._data[group]
        if not isinstance(entries, list) or not (0 <= index < len(entries)):
            return False

        del entries[index]
        self._has_unsaved_changes = True
        return True

    def get_target_count(self, group: str) -> int:
        """Get the number of targets in a group.

        Args:
            group: The group name.

        Returns:
            Number of targets, or 0 if group not found.
        """
        return len(self.get_targets(group))

    def rename_group(self, old_name: str, new_name: str) -> bool:
        """Rename a group.

        Args:
            old_name: Current group name.
            new_name: New group name.

        Returns:
            True if renamed, False if old not found or new already exists.
        """
        if self._data is None:
            return False
        if old_name not in self._data or old_name in self.EXCLUDED_KEYS:
            return False
        if new_name in self._data:
            return False

        # ruamel.yaml preserves insertion order, so we need to recreate
        # the data structure to maintain position
        from ruamel.yaml.comments import CommentedMap

        new_data = CommentedMap()
        for key in self._data:
            if key == old_name:
                new_data[new_name] = self._data[old_name]
            else:
                new_data[key] = self._data[key]

        self._data = new_data
        self._has_unsaved_changes = True
        return True

    def move_target(self, group: str, from_index: int, to_index: int) -> bool:
        """Move a target within a group.

        Args:
            group: The group name.
            from_index: Current position (0-based).
            to_index: New position (0-based).

        Returns:
            True if moved, False if invalid indices.
        """
        if self._data is None or group not in self._data:
            return False

        entries = self._data[group]
        if not isinstance(entries, list):
            return False

        if not (0 <= from_index < len(entries) and 0 <= to_index < len(entries)):
            return False

        target = entries.pop(from_index)
        entries.insert(to_index, target)
        self._has_unsaved_changes = True
        return True

    def duplicate_target(self, group: str, index: int) -> bool:
        """Duplicate a target within a group.

        Args:
            group: The group name.
            index: The target index to duplicate (0-based).

        Returns:
            True if duplicated, False if not found.
        """
        target = self.get_target(group, index)
        if target is None:
            return False

        # Create a deep copy
        import copy

        new_target = copy.deepcopy(dict(target))

        # Modify name to indicate it's a copy
        if "name" in new_target:
            new_target["name"] = f"{new_target['name']} (copy)"

        return self.add_target(group, new_target)
