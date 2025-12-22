"""
Target resolver for MARS report modules.

Resolves module targets to actual filesystem paths using the database catalog.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class TargetResolver:
    """Resolves module target strings to filesystem paths."""

    def __init__(self, catalog: dict, source_root: Path):
        """Initialize target resolver.

        Args:
            catalog: Database catalog dict loaded from artifact_recovery_catalog.yaml
            source_root: Root path of the scan (exemplar or candidate root)
        """
        self.catalog = catalog
        self.source_root = source_root

    def resolve(self, target: str) -> list[Path]:
        """Resolve target string to list of filesystem paths.

        Args:
            target: Target specification:
                - "root": Return scan root (module handles its own file discovery)
                - Catalog name (e.g., "Firefox Cache"): Look up and glob
                - Custom path: Use as literal glob pattern

        Returns:
            List of matching paths (may be empty if no matches)

        Examples:
            >>> resolver = TargetResolver(catalog, Path("/exemplar"))
            >>> resolver.resolve("root")
            [Path("/exemplar")]

            >>> resolver.resolve("Firefox Cache")
            [
                Path("/exemplar/caches/Firefox Cache_alice"),
                Path("/exemplar/caches/Firefox Cache_bob")
            ]
        """
        # Special target: return root path
        if target.lower() == "root":
            return [self.source_root]

        # Try to find target in catalog
        glob_pattern = self._find_catalog_pattern(target)

        if glob_pattern:
            # Use catalog pattern
            return self._glob_paths(glob_pattern)

        # If not in catalog, treat as literal glob pattern
        return self._glob_paths(target)

    def _find_catalog_pattern(self, target_name: str) -> str | None:
        """Find glob pattern for target name in catalog.

        Args:
            target_name: Name to search for in catalog

        Returns:
            Glob pattern string or None if not found

        Notes:
            Searches all catalog sections (macos, third_party, cloud, etc.)
            for an entry with matching 'name' field.
            Prefers exemplar_pattern (for processed exemplar scans),
            then glob_pattern, then path.
        """
        # Catalog structure: {section: [{name, glob_pattern, ...}, ...]}
        for section_name, entries in self.catalog.items():
            if not isinstance(entries, list):
                continue

            for entry in entries:
                if not isinstance(entry, dict):
                    continue

                # Check if name matches (case-insensitive)
                entry_name = entry.get("name", "")
                if entry_name.lower() == target_name.lower():
                    # Prefer exemplar_pattern (for exemplar scans), fall back to glob_pattern
                    return entry.get("exemplar_pattern") or entry.get("glob_pattern")

        return None

    def _glob_paths(self, pattern: str) -> list[Path]:
        """Execute glob pattern from source root.

        Args:
            pattern: Glob pattern (e.g., "Users/*/Library/Caches/Firefox")

        Returns:
            List of matching paths, sorted by name

        Notes:
            - Filters out non-directories
            - Returns empty list if no matches
            - Sorts results for consistent ordering
        """
        try:
            matches = list(self.source_root.glob(pattern))
            # Filter to directories only (most targets are directories)
            # If pattern explicitly targets files (*.db), keep files too
            if "*.*" in pattern or pattern.endswith((".db", ".sqlite", ".json")):
                # Pattern targets files - keep all matches
                result = [p for p in matches if p.exists()]
            else:
                # Pattern targets directories - filter to dirs only
                result = [p for p in matches if p.is_dir()]

            return sorted(result, key=lambda p: p.name)
        except (OSError, ValueError):
            # Invalid pattern or permission error
            # Return empty list rather than crashing
            return []

    def resolve_with_usernames(self, target: str) -> list[tuple[Path, str]]:
        """Resolve target and extract usernames from paths.

        For user-scoped artifacts, extract the username from the path
        to generate unique output directory names.

        Args:
            target: Target specification (same as resolve())

        Returns:
            List of (path, username) tuples where username is extracted
            from path or empty string if not user-scoped

        Examples:
            >>> resolver.resolve_with_usernames("Firefox Cache")
            [
                (Path("...Firefox Cache_alice"), "alice"),
                (Path("...Firefox Cache_bob"), "bob")
            ]

            >>> resolver.resolve_with_usernames("root")
            [(Path("/exemplar"), "")]
        """
        paths = self.resolve(target)
        results = []

        for path in paths:
            # Try to extract username from path
            username = self._extract_username(path)
            results.append((path, username))

        return results

    def _extract_username(self, path: Path) -> str:
        """Extract username from path if present.

        Args:
            path: Path potentially containing username

        Returns:
            Username string or empty string if not found

        Notes:
            Looks for patterns like:
            - /Users/{username}/...
            - /.../{artifact}_{username}/...
        """
        # Check if path contains /Users/{username}/
        parts = path.parts
        try:
            users_idx = parts.index("Users")
            if users_idx + 1 < len(parts):
                return parts[users_idx + 1]
        except ValueError:
            pass

        # Check for artifact_username pattern in final component
        # E.g., "Firefox Cache_alice" -> "alice"
        name = path.name
        if "_" in name:
            # Split on underscore and take last part as potential username
            potential_username = name.split("_")[-1]
            # Basic validation: usernames are typically alphanumeric
            if potential_username.replace("-", "").replace(".", "").isalnum():
                return potential_username

        return ""
