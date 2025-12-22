"""Shared helpers for dfVFS-based glob expansion."""

from __future__ import annotations

import copy
import fnmatch
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import yaml
from dfvfs.helpers import source_scanner
from dfvfs.lib import definitions as dfvfs_definitions
from dfvfs.path import factory as path_factory
from dfvfs.resolver import resolver
from dfvfs.volume import factory as volume_system_factory

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from pathlib import Path

# Import the GPT name enumeration utility
try:
    from mars.pipeline.mount_utils.get_gpt_names import enumerate_volume_names
except ImportError:
    enumerate_volume_names = None


@dataclass
class TargetFS:
    label: str
    type_indicator: str
    fs_spec: object
    volume_name: str | None = None
    size_bytes: int | None = None


@dataclass
class PatternEntry:
    pattern: str
    target: str | None
    dir_prefixes: tuple[str, ...]
    relative_segments: tuple[str, ...]
    base_start: str
    compiled_regex: re.Pattern | None = None  # Pre-compiled for ** patterns


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


def load_catalog_patterns(
    catalog_path: Path,
    excluded_file_types: set[str] | None = None,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load catalog patterns, optionally filtering by file_type.

    Args:
        catalog_path: Path to the catalog YAML file
        excluded_file_types: Optional set of file_types to exclude (e.g., {'cache', 'log'})

    Returns:
        Tuple of (patterns_by_target, pattern_to_target)
    """
    with catalog_path.open("r", encoding="utf-8") as fh:
        catalog = yaml.safe_load(fh)

    keys_of_interest = {
        "pattern",
        "patterns",
        "globs",
        "glob",
        "glob_pattern",
        "glob_patterns",
        "direct",
        "direct_paths",
        "fallback_globs",
        "fallback_paths",
    }

    pattern_to_target: dict[str, str] = {}
    generated_patterns: list[str] = []
    excluded_file_types = excluded_file_types or set()

    def to_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [str(v) for v in value]
        return [str(value)]

    def collect(node):
        if isinstance(node, dict):
            # Check if this entry should be excluded based on file_type
            file_type = node.get("file_type")
            if file_type and file_type in excluded_file_types:
                # Skip this entry entirely - don't collect its patterns
                # But still recurse into nested structures (other entries in same group)
                for value in sorted(node.values(), key=lambda v: str(v)):
                    collect(value)
                return

            name = node.get("name")
            local_patterns: list[str] = []
            for key in keys_of_interest:
                if key in node:
                    local_patterns.extend(to_list(node.get(key)))

            # Extract patterns from nested "primary" key
            if "primary" in node and isinstance(node["primary"], dict):
                primary = node["primary"]
                for key in keys_of_interest:
                    if key in primary:
                        local_patterns.extend(to_list(primary.get(key)))

            # Also collect archive patterns (for powerlog, etc.)
            if "archives" in node and isinstance(node["archives"], list):
                for archive_def in node["archives"]:
                    if isinstance(archive_def, dict):
                        # Get parent path from primary
                        parent_path = None
                        if "primary" in node:
                            primary = node["primary"]
                            if isinstance(primary, dict):
                                primary_path = primary.get("glob_pattern", "")
                            else:
                                primary_path = str(primary)
                            # Get parent directory
                            if primary_path:
                                parts = primary_path.rstrip("/").split("/")
                                parent_path = "/".join(parts[:-1])

                        archive_subpath = archive_def.get("subpath", "")
                        # Check for glob_pattern or pattern (catalog uses both)
                        archive_glob = archive_def.get("glob_pattern") or archive_def.get("pattern", "*")
                        if parent_path is not None:
                            # If subpath is empty, archives are in same directory as primary
                            if archive_subpath:
                                archive_pattern = f"{parent_path}/{archive_subpath}/{archive_glob}"
                            else:
                                archive_pattern = f"{parent_path}/{archive_glob}"
                            local_patterns.append(archive_pattern)

            if name and local_patterns:
                for pat in local_patterns:
                    normalized = _normalize(pat)
                    pattern_to_target.setdefault(normalized, name)
                    generated_patterns.append(normalized)
            # Sort dictionary values for deterministic iteration order
            for value in sorted(node.values(), key=lambda v: str(v)):
                collect(value)
        elif isinstance(node, list):
            for item in node:
                collect(item)

    collect(catalog)

    for pat in list(generated_patterns):
        target = pattern_to_target.get(pat)
        for variant in _data_variants(pat, target):
            if variant not in pattern_to_target:
                pattern_to_target[variant] = target or "unknown"
                generated_patterns.append(variant)

    grouped = _group_patterns_by_start(generated_patterns, pattern_to_target)

    patterns_by_target: dict[str, list[str]] = defaultdict(list)
    # Sort grouped items for deterministic iteration order
    for key in sorted(grouped.keys()):
        entries = grouped[key]
        for entry in entries:
            patterns_by_target[entry.target or "unknown"].append(entry.pattern)
    return patterns_by_target, pattern_to_target


# ---------------------------------------------------------------------------
# dfVFS enumeration helpers
# ---------------------------------------------------------------------------


def scan_sources(image_path: str, include: Iterable[str]) -> list[TargetFS]:
    include_set = {s.lower() for s in include}
    scanner = source_scanner.SourceScanner()
    ctx = source_scanner.SourceScannerContext()
    ctx.OpenSourcePath(image_path)
    scanner.Scan(ctx)

    # Try to get detailed partition names using GPT parsing
    partition_names_map = {}
    apfs_volumes_map = {}  # Initialize here so it's always defined
    if enumerate_volume_names:
        try:
            partition_info = enumerate_volume_names(image_path)
            # Build maps: partition_id -> name and partition_id -> apfs_volumes
            for info in partition_info:
                pid = info.get("id")  # e.g., "p1", "p2"
                # Prefer filesystem label (more specific), then GPT entry name, then just the ID
                best_name = info.get("label") or info.get("gpt_entry_name") or pid
                if best_name and pid:
                    partition_names_map[pid] = best_name
                # Store APFS volumes list if present
                apfs_list = info.get("apfs")
                if apfs_list:
                    apfs_volumes_map[pid] = apfs_list
        except Exception:
            # Silently fail - we'll fall back to simple partition IDs
            pass

    targets: list[TargetFS] = []

    def walk(node):
        if node is None:
            return
        ti = getattr(node, "type_indicator", None)
        ps = getattr(node, "path_spec", None)
        # Use partition identifier as label for uniqueness, not filesystem location
        # Will be updated below with proper partition ID
        label = None

        if ti and ps and ps.IsFileSystem() and ti.lower() in include_set:
            # Try to extract volume metadata from parent volume system
            volume_name = None
            size_bytes = None

            if ps.HasParent():
                parent_ps = ps.parent
                parent_ti = getattr(parent_ps, "type_indicator", None)

                # Extract unique partition label from parent PathSpec
                # For non-APFS: use parent location (e.g., "/p1", "/p2")
                # For APFS: use grandparent location + volume index (e.g., "p2v0", "p2v1")
                if parent_ti == dfvfs_definitions.TYPE_INDICATOR_APFS_CONTAINER:
                    # APFS: need grandparent (GPT partition) + volume index
                    if parent_ps.HasParent():
                        grandparent_ps = parent_ps.parent
                        gpt_location = getattr(grandparent_ps, "location", None)
                        if gpt_location:
                            partition_id = gpt_location.lstrip("/").lstrip("\\")
                            # Get volume index from parent PathSpec
                            try:
                                parent_dict = parent_ps.CopyToDict()
                                if "volume_index" in parent_dict:
                                    vol_idx = parent_dict["volume_index"]
                                    label = f"{partition_id}v{vol_idx}"  # e.g., "p2v0", "p2v1"
                                else:
                                    label = partition_id  # Fallback to just partition ID
                            except Exception:
                                label = partition_id
                else:
                    # Non-APFS: use parent partition location
                    parent_location = getattr(parent_ps, "location", None)
                    if parent_location:
                        label = parent_location.lstrip("/").lstrip("\\")  # e.g., "p1", "p2"

                # If we still don't have a label, fall back to filesystem location
                if not label:
                    label = getattr(ps, "location", None) or ti

                # For APFS, try to get volume-specific metadata from apfs_volumes_map first
                if parent_ti == dfvfs_definitions.TYPE_INDICATOR_APFS_CONTAINER:
                    # Try to use the pre-built apfs_volumes_map if available
                    if apfs_volumes_map and parent_ps.HasParent():
                        try:
                            # Get the GPT partition ID from grandparent
                            grandparent_ps = parent_ps.parent
                            gpt_location = getattr(grandparent_ps, "location", None)
                            if gpt_location:
                                # Strip leading slash (e.g., "/p2" -> "p2")
                                partition_id = gpt_location.lstrip("/").lstrip("\\")

                                # Try to get the APFS volume index from the parent PathSpec
                                # For APFS: current PathSpec is the filesystem (/), parent is the APFS container with volume_index
                                volume_idx = None
                                try:
                                    # Check parent's dictionary for volume_index
                                    parent_dict = parent_ps.CopyToDict()
                                    if "volume_index" in parent_dict:
                                        volume_idx = parent_dict["volume_index"]

                                except Exception:
                                    pass

                                # Look up the volume name from apfs_volumes_map
                                if partition_id in apfs_volumes_map and volume_idx is not None:
                                    volume_names_list = apfs_volumes_map[partition_id]
                                    if isinstance(volume_names_list, list) and 0 <= volume_idx < len(volume_names_list):
                                        apfs_vol_name = volume_names_list[volume_idx]
                                        if apfs_vol_name:
                                            volume_name = apfs_vol_name
                        except Exception:
                            pass

                    # If we didn't get a name from apfs_volumes_map, try extracting from container
                    if not volume_name:
                        try:
                            apfs_volume_system = volume_system_factory.Factory.NewVolumeSystem(parent_ti)
                            if apfs_volume_system:
                                apfs_volume_system.Open(parent_ps)

                                # Get volume index from the APFS filesystem PathSpec
                                volume_idx = getattr(ps, "volume_index", None)

                                # Try to get the APFS volume by index
                                if (
                                    volume_idx is not None
                                    and hasattr(apfs_volume_system, "volumes")
                                    and apfs_volume_system.volumes
                                ):
                                    volumes_list = list(apfs_volume_system.volumes)
                                    if 0 <= volume_idx < len(volumes_list):
                                        apfs_volume = volumes_list[volume_idx]

                                        # Get APFS volume name from attributes
                                        if hasattr(apfs_volume, "attributes"):
                                            for attr in apfs_volume.attributes:
                                                if (
                                                    hasattr(attr, "identifier")
                                                    and hasattr(attr, "value")
                                                    and attr.identifier in ["name", "volume_name"]
                                                    and attr.value
                                                ):
                                                    volume_name = str(attr.value)
                                                    break

                                        # Get APFS volume size from extents
                                        if hasattr(apfs_volume, "extents") and apfs_volume.extents:
                                            extents_list = list(apfs_volume.extents)
                                            total_size = sum(extent.size for extent in extents_list)
                                            if total_size > 0:
                                                size_bytes = total_size
                        except Exception:
                            pass  # Fall back to GPT partition metadata

                # For APFS, also walk up to find the GPT partition for fallback
                # APFS structure: APFS filesystem -> APFS container -> GPT partition
                gpt_parent_ps = None
                gpt_parent_ti = None
                if parent_ti == dfvfs_definitions.TYPE_INDICATOR_APFS_CONTAINER and parent_ps.HasParent():
                    grandparent_ps = parent_ps.parent
                    grandparent_ti = getattr(grandparent_ps, "type_indicator", None)
                    if grandparent_ti in [
                        dfvfs_definitions.TYPE_INDICATOR_TSK_PARTITION,
                        dfvfs_definitions.TYPE_INDICATOR_GPT,
                        dfvfs_definitions.TYPE_INDICATOR_APM,
                    ]:
                        gpt_parent_ps = grandparent_ps
                        gpt_parent_ti = grandparent_ti

                # Determine which parent to use for metadata extraction
                # For APFS: use grandparent (GPT partition) if we didn't get volume-specific data
                # For others: use parent directly
                metadata_parent_ps = gpt_parent_ps if gpt_parent_ps else parent_ps
                metadata_parent_ti = gpt_parent_ti if gpt_parent_ti else parent_ti

                # Check if parent is a volume system layer (GPT, TSK partition, APM)
                if metadata_parent_ti in [
                    dfvfs_definitions.TYPE_INDICATOR_TSK_PARTITION,
                    dfvfs_definitions.TYPE_INDICATOR_GPT,
                    dfvfs_definitions.TYPE_INDICATOR_APM,
                ]:
                    try:
                        # Open the volume system using the factory
                        volume_system = volume_system_factory.Factory.NewVolumeSystem(metadata_parent_ti)
                        if volume_system:
                            volume_system.Open(metadata_parent_ps)

                            # Get volume identifier from parent PathSpec
                            volume_identifier = getattr(metadata_parent_ps, "location", None)
                            if not volume_identifier:
                                volume_identifier = getattr(metadata_parent_ps, "volume_index", None)
                            if not volume_identifier:
                                volume_identifier = getattr(metadata_parent_ps, "part_index", None)

                            # Strip leading slash if present (e.g., "/p1" -> "p1")
                            if volume_identifier and isinstance(volume_identifier, str):
                                volume_identifier = volume_identifier.lstrip("/").lstrip("\\")

                            if volume_identifier and hasattr(volume_system, "volume_identifiers"):
                                # Try to get volume by identifier
                                volume = None
                                try:
                                    volume = volume_system.GetVolumeByIdentifier(volume_identifier)
                                except Exception:
                                    # If that fails, iterate through volumes to find a match
                                    for vol_id in volume_system.volume_identifiers:
                                        if vol_id == volume_identifier:
                                            try:
                                                volume = volume_system.GetVolumeByIdentifier(vol_id)
                                                break
                                            except Exception:
                                                continue

                                if volume:
                                    # Get volume size from extents
                                    if hasattr(volume, "extents") and volume.extents:
                                        extents_list = list(volume.extents)
                                        total_size = sum(extent.size for extent in extents_list)
                                        if total_size > 0:
                                            size_bytes = total_size

                                    # Try to get volume name from attributes
                                    if hasattr(volume, "attributes"):
                                        for attr in volume.attributes:
                                            if (
                                                hasattr(attr, "identifier")
                                                and hasattr(attr, "value")
                                                and attr.identifier
                                                in [
                                                    "description",
                                                    "name",
                                                    "type",
                                                    "partition_name",
                                                ]
                                                and attr.value
                                                and not volume_name
                                            ):
                                                volume_name = str(attr.value)

                                    # If still no name, try getting it from the partition entry
                                    if not volume_name:
                                        try:
                                            file_entry = resolver.Resolver.OpenFileEntry(metadata_parent_ps)
                                            if file_entry and hasattr(file_entry, "name") and file_entry.name:
                                                volume_name = file_entry.name
                                        except Exception:
                                            pass
                                    # Finally, check if we got a better name from GPT parsing
                                    # Try both: the current volume_name AND the volume_identifier
                                    # BUT: For APFS, if we already have a volume-specific name, don't overwrite with GPT partition name
                                    if parent_ti != dfvfs_definitions.TYPE_INDICATOR_APFS_CONTAINER or not volume_name:
                                        if volume_identifier and volume_identifier in partition_names_map:
                                            better_name = partition_names_map.get(volume_identifier)
                                            if better_name:
                                                volume_name = better_name
                                        elif volume_name and volume_name in partition_names_map:
                                            better_name = partition_names_map.get(volume_name)
                                            if better_name and better_name != volume_name:
                                                volume_name = better_name
                    except Exception:
                        # Silently fail - volume metadata is optional
                        pass

            targets.append(
                TargetFS(
                    label=label or ti,
                    type_indicator=ti,
                    fs_spec=ps,
                    volume_name=volume_name,
                    size_bytes=size_bytes,
                )
            )
        # Sort sub_nodes for deterministic traversal order
        sub_nodes = getattr(node, "sub_nodes", []) or []
        sorted_nodes = sorted(sub_nodes, key=lambda n: str(getattr(n, "type_indicator", "") or ""))
        for child in sorted_nodes:
            walk(child)

    root = getattr(ctx, "root_scan_node", None)
    if hasattr(ctx, "GetRootScanNode"):
        root = ctx.GetRootScanNode()
    walk(root)

    # Sort targets by label for consistent ordering across runs
    return sorted(targets, key=lambda t: t.label or "")


# Directories to skip during iteration (cause hangs due to huge file counts)
# These are non-forensic cache directories that slow down scans significantly
# Note: Use */*/ prefix to match /Users/username/ (two levels)
SKIP_DIRECTORIES: set[str] = {
    "*/*/Library/Caches/Google/Chrome/*/Code Cache",
    "*/*/Library/Caches/Google/Chrome/*/Code Cache/*",
    "*/*/Library/Caches/com.apple.Safari/fsCachedData",
    "*/*/Library/Caches/*/com.apple.metal",
    "*/*/Library/Caches/*/GPUCache",
}


def iter_entries(
    fs_type: str,
    fs_spec,
    start_location: str,
    dir_filters: set[str] | None = None,
    dir_excludes: set[str] | None = None,
    status_callback: Callable[[str], None] | None = None,
    skip_nested_archive_detection: bool = False,  # Required for ZIP directory detection workaround
) -> Iterator[tuple[str, object]]:
    entry, location = _open_entry(fs_type, fs_spec, start_location)
    if entry is None:
        return

    # Combine default skip directories with any passed excludes
    all_excludes = SKIP_DIRECTORIES.copy()
    if dir_excludes:
        all_excludes.update(dir_excludes)

    stack = [(location, entry)]

    # Match result constants for pattern matching
    _NO_MATCH = 0  # Path doesn't match and can't lead to match
    _PREFIX = 1  # Path is a prefix that could lead to a match
    _EXACT = 2  # Path matches pattern exactly
    _PAST = 3  # Path matched pattern but extended past it

    def _classify_match(
        path_parts: list[str],
        pattern_parts: list[str],
        pi: int,
        pati: int,
    ) -> int:
        """Classify how a path relates to a pattern.

        Supports globstar (**) anywhere in the pattern:
        - '*' matches exactly one directory level
        - '**' matches zero or more directory levels

        Returns one of: _NO_MATCH, _PREFIX, _EXACT, _PAST
        """
        while pati < len(pattern_parts):
            pat = pattern_parts[pati]

            if pat == "**":
                remaining = pattern_parts[pati + 1 :]
                if not remaining:
                    # ** at end of pattern - matches everything from here
                    return _EXACT

                # Try each position where ** could stop consuming segments
                best = _NO_MATCH
                for try_pi in range(pi, len(path_parts) + 1):
                    sub_result = _classify_match(path_parts, remaining, try_pi, 0)
                    if sub_result == _EXACT:
                        return _EXACT
                    if sub_result == _PREFIX and best < _PREFIX:
                        best = _PREFIX
                    if sub_result == _PAST and best < _PAST:
                        best = _PAST

                if best != _NO_MATCH:
                    return best

                # No match found at any position
                # Is path short enough to be a valid prefix?
                remaining_path_len = len(path_parts) - pi
                if remaining_path_len < len(remaining):
                    return _PREFIX

                return _NO_MATCH

            if pi >= len(path_parts):
                # Path exhausted, pattern remains - valid prefix
                return _PREFIX

            # Match single segment: * matches any single segment name
            if not fnmatch.fnmatch(path_parts[pi], pat):
                return _NO_MATCH

            pi += 1
            pati += 1

        # Pattern exhausted
        if pi >= len(path_parts):
            return _EXACT
        return _PAST

    def _path_match_inclusion(path: str, pattern: str) -> bool:
        """Match path against INCLUSION filter pattern.

        Returns True if path matches pattern exactly OR is a prefix of where
        pattern could match (i.e., we need to descend further).

        Returns False if path has gone PAST the pattern (matched then extended).

        Supports:
        - '*' matches any single directory level
        - '**' matches zero or more directory levels (can appear anywhere)

        Examples with pattern '**/Library/Caches':
        - /Users → True (prefix, need to descend)
        - /Users/admin/Library/Caches → True (exact match)
        - /Users/admin/Library/Caches/Chrome → False (past the pattern)
        - /Library/Caches → True (** matches zero segments)
        """
        path_parts = path.strip("/").split("/") if path.strip("/") else []
        pattern_parts = pattern.strip("/").split("/") if pattern.strip("/") else []

        result = _classify_match(path_parts, pattern_parts, 0, 0)
        return result in (_PREFIX, _EXACT)

    def _path_match_exclusion(path: str, pattern: str) -> bool:
        """Match path against EXCLUSION (skip) pattern.

        Returns True ONLY if path is AT or INSIDE the excluded directory.
        Does NOT match if path is an ancestor of the pattern.

        Supports:
        - '*' matches any single directory level
        - '**' matches zero or more directory levels (can appear anywhere)

        Examples with pattern '**/node_modules':
        - /Users → False (prefix/ancestor, don't exclude)
        - /Users/project/node_modules → True (exact match)
        - /Users/project/node_modules/lodash → True (inside excluded dir)
        - /node_modules → True (** matches zero segments)

        Examples with pattern 'Users/**/Caches':
        - /Users → False (prefix/ancestor)
        - /Users/admin/Library/Caches → True (** matches 'admin/Library')
        - /Users/admin/Library/Caches/Chrome → True (inside)
        """
        path_parts = path.strip("/").split("/") if path.strip("/") else []
        pattern_parts = pattern.strip("/").split("/") if pattern.strip("/") else []

        result = _classify_match(path_parts, pattern_parts, 0, 0)
        return result in (_EXACT, _PAST)

    def should_descend(path: str) -> bool:
        # First check exclusions - skip if path matches any exclude pattern
        # Use _path_match_exclusion which only matches paths AT or INSIDE the pattern
        for excl in all_excludes:
            if _path_match_exclusion(path, excl):
                return False
        # Then check inclusion filters
        # Use _path_match_inclusion which matches prefixes too (need to descend)
        if not dir_filters:
            return True
        return any(_path_match_inclusion(path, flt) for flt in dir_filters)

    archive_indicators = {
        dfvfs_definitions.TYPE_INDICATOR_ZIP,
        dfvfs_definitions.TYPE_INDICATOR_TAR,
        dfvfs_definitions.TYPE_INDICATOR_GZIP,
        dfvfs_definitions.TYPE_INDICATOR_BZIP2,
    }
    # IMPORTANT: reopen_cache is a workaround for a dfVFS quirk where ZIP directories
    # may initially report as files during enumeration. By reopening each entry, we can
    # correctly detect directories and traverse into them. Without this, user directories
    # like /Users/ would be skipped, missing all user-scoped databases.
    # DO NOT disable this for archives unless you verify directory detection works.
    reopen_cache: dict[str, object] | None = (
        {} if fs_type in archive_indicators and not skip_nested_archive_detection else None
    )

    # Threshold for skipping sort (large dirs don't need deterministic order)
    LARGE_DIR_THRESHOLD = 5000

    while stack:
        path, node = stack.pop()
        try:
            # Collect entries - for large directories, skip sorting
            # Use incremental collection with progress logging for very slow dirs
            try:
                entries = []
                for entry_count, entry in enumerate(node.sub_file_entries, 1):
                    entries.append(entry)
                    # Update UI status for large directories (every 1000 entries)
                    if entry_count % 1000 == 0 and status_callback is not None:
                        display_path = path
                        if len(display_path) > 80:
                            display_path = "..." + display_path[-77:]
                        status_callback(f"{display_path} ({entry_count:,} entries...)")

                if len(entries) <= LARGE_DIR_THRESHOLD:
                    entries.sort(key=lambda e: getattr(e, "name", ""))
            except Exception:
                # If listing fails, skip this directory
                continue

            for child in entries:
                name = getattr(child, "name", None)
                if not name or name in {".", ".."}:
                    continue
                child_path = path.rstrip("/") + "/" + name if path != "/" else "/" + name
                try:
                    is_directory = child.IsDirectory()
                    directory_entry = child
                    if not is_directory and reopen_cache is not None:
                        cached_entry = reopen_cache.get(child_path)
                        if cached_entry is None:
                            reopened_entry, _ = _open_entry(
                                fs_type,
                                fs_spec,
                                child_path,
                            )
                            if reopened_entry is not None:
                                reopen_cache[child_path] = reopened_entry
                                cached_entry = reopened_entry
                        if cached_entry is not None:
                            directory_entry = cached_entry
                            is_directory = True

                    if is_directory:
                        if should_descend(child_path):
                            stack.append((child_path, directory_entry))
                            yield child_path, directory_entry
                    elif child.IsFile():
                        yield child_path, child
                except Exception:
                    continue
        except Exception:
            continue


# ---------------------------------------------------------------------------
# Pattern expansion
# ---------------------------------------------------------------------------


def _group_patterns_by_start(
    patterns: list[str], pattern_targets: Mapping[str, str | None]
) -> dict[str, list[PatternEntry]]:
    grouped: dict[str, list[PatternEntry]] = defaultdict(list)
    for pattern in patterns:
        norm = _normalize(pattern)
        parts = [part for part in norm.strip("/").split("/") if part]
        base: list[str] = []
        for part in parts:
            if _segment_has_wildcard(part):
                break
            base.append(part)
        start = "/" + "/".join(base) if base else "/"
        entry = _make_pattern_entry(norm, pattern_targets.get(pattern), start)
        grouped[entry.base_start].append(entry)
    return grouped


def _expand_entries_for_start(
    fs_type: str,
    fs_spec,
    start_location: str,
    entries: list[PatternEntry],
) -> dict[str, list[PatternEntry]]:
    expanded: dict[str, list[PatternEntry]] = defaultdict(list)
    cache: dict[str, list[str]] = {}
    queue: deque[tuple[str, PatternEntry]] = deque((start_location, entry) for entry in entries)
    visited: set[tuple[str, str, tuple[str, ...]]] = set()

    while queue:
        current_start, entry = queue.popleft()
        state = (current_start, entry.pattern, entry.relative_segments)
        if state in visited:
            continue
        visited.add(state)

        rel_segments = entry.relative_segments
        next_segment = rel_segments[0] if rel_segments else None

        # Don't expand the final wildcard segment (let pattern matching handle files)
        # Also don't expand ** - let it pass through to be matched via regex
        is_last_segment = len(rel_segments) == 1

        # ** should not be expanded - it's handled by regex matching at file-match time
        if next_segment == "**":
            expanded[current_start].append(entry)
            continue

        should_expand = next_segment and _segment_has_wildcard(next_segment) and not is_last_segment

        if should_expand and next_segment:  # Add next_segment check for type checker
            names = _list_directory_names(fs_type, fs_spec, current_start, cache)
            for name in names:
                if fnmatch.fnmatch(name, next_segment):
                    new_start = (current_start.rstrip("/") + "/" + name).replace("//", "/")
                    remaining = entry.relative_segments[1:]
                    new_pattern = new_start.rstrip("/") + "/" + "/".join(remaining) if remaining else new_start
                    new_entry = _make_pattern_entry(new_pattern, entry.target, new_start)
                    queue.append((new_start, new_entry))
            continue

        expanded[current_start].append(entry)
    return expanded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(path: str) -> str:
    path = path.replace("\\", "/")
    if not path.startswith("/"):
        path = "/" + path
    return path.rstrip("/") or "/"


def _segment_has_wildcard(segment: str) -> bool:
    return any(ch in segment for ch in "*?[]") or segment == "**"


def _compile_globstar_regex(pattern: str) -> re.Pattern | None:
    """Pre-compile a glob pattern containing ** to a regex.

    Returns None if pattern doesn't contain ** (use fnmatch instead).

    Handles ** matching zero or more path segments:
    - /**/foo matches /foo (zero segments) and /bar/foo (one segment)
    - **/ at start matches from root
    - /** at end matches everything below
    """
    if "**" not in pattern:
        return None

    try:
        regex_pattern = pattern
        # Escape regex special chars (except * and ? and [ ])
        # Note: escape backslash FIRST, then other chars, to avoid double-escaping
        regex_pattern = regex_pattern.replace("\\", "\\\\")
        for char in [".", "^", "$", "+", "{", "}", "(", ")", "|"]:
            regex_pattern = regex_pattern.replace(char, "\\" + char)

        # Handle ** with surrounding slashes specially to allow zero-segment match
        # /**/ in middle: keep leading /, make the **/ part optional
        # /foo/**/bar -> /foo/(?:.*/)?bar matches /foo/bar or /foo/x/bar or /foo/x/y/bar
        regex_pattern = regex_pattern.replace("/**/", "/\x00GLOBSTAR_MID\x00")
        # **/ at start: match from root with optional prefix
        regex_pattern = regex_pattern.replace("**/", "\x00GLOBSTAR_START\x00")
        # /** at end: match everything below
        regex_pattern = regex_pattern.replace("/**", "\x00GLOBSTAR_END\x00")
        # Standalone ** (entire pattern or unusual usage)
        regex_pattern = regex_pattern.replace("**", "\x00GLOBSTAR\x00")

        # Convert single * and ? (must happen after ** is protected)
        regex_pattern = regex_pattern.replace("*", "[^/]*")  # * = any chars except /
        regex_pattern = regex_pattern.replace("?", "[^/]")  # ? = single char except /

        # Now convert globstar placeholders to regex
        # /**/ -> /(?:.*/)? which matches / then optionally (anything ending in /)
        regex_pattern = regex_pattern.replace("\x00GLOBSTAR_MID\x00", "(?:.*/)?")
        # **/  at start -> match empty OR anything/ (zero or more segments from root)
        regex_pattern = regex_pattern.replace("\x00GLOBSTAR_START\x00", "(?:.*/)?")
        # /** at end -> match empty OR /anything (zero or more segments to end)
        regex_pattern = regex_pattern.replace("\x00GLOBSTAR_END\x00", "(?:/.*)?")
        # Standalone ** -> match anything
        regex_pattern = regex_pattern.replace("\x00GLOBSTAR\x00", ".*")

        # Ensure we match the full path
        regex_pattern = "^" + regex_pattern + "$"

        return re.compile(regex_pattern)
    except Exception:
        return None


def _make_pattern_entry(pattern: str, target: str | None, start: str) -> PatternEntry:
    norm_pattern = _normalize(pattern)
    norm_start = _normalize(start)

    relative = norm_pattern[len(norm_start) :] if norm_pattern.startswith(norm_start) else norm_pattern
    relative_segments = tuple(seg for seg in relative.strip("/").split("/") if seg)

    prefixes: list[str] = []
    base = norm_start.rstrip("/") or "/"
    prefixes.append(base)

    current = base
    for seg in relative_segments:
        if seg == "**":
            wild = (current if current != "/" else "") + "/**"
            prefixes.append(wild.replace("//", "/"))
            current = wild.replace("//", "/")  # Update current to include **
            continue
        if _segment_has_wildcard(seg):
            wild = (current if current != "/" else "") + "/" + seg
            prefixes.append(wild.replace("//", "/"))
            current = wild.replace("//", "/")  # Update current to include wildcard
            continue
        current = "/" + seg if current == "/" else current.rstrip("/") + "/" + seg
        current = current.replace("//", "/")
        prefixes.append(current)

    unique_prefixes: list[str] = []
    seen = set()
    for pref in prefixes:
        pref = pref.replace("//", "/")
        if pref not in seen:
            seen.add(pref)
            unique_prefixes.append(pref)

    base_parts: list[str] = []
    segments = norm_pattern.strip("/").split("/")
    for seg in segments[:-1]:
        if seg == "**" or _segment_has_wildcard(seg):
            break
        base_parts.append(seg)
    base_start = "/" + "/".join(base_parts) if base_parts else "/"

    # Pre-compile regex for patterns containing ** (globstar)
    compiled_regex = _compile_globstar_regex(norm_pattern)

    return PatternEntry(
        pattern=norm_pattern,
        target=target,
        dir_prefixes=tuple(unique_prefixes),
        relative_segments=relative_segments,
        base_start=_normalize(base_start),
        compiled_regex=compiled_regex,
    )


def _open_entry(fs_type: str, fs_spec, location: str):
    target_location = _normalize(location)
    # Create new path spec reusing parent reference instead of expensive deep copy
    try:
        start_spec = path_factory.Factory.NewPathSpec(
            fs_spec.type_indicator,
            location=target_location,
            parent=fs_spec.parent,
        )
    except Exception:
        # Fallback to deep copy if factory fails (shouldn't happen for valid specs)
        start_spec = copy.deepcopy(fs_spec)
        if hasattr(start_spec, "location"):
            start_spec.location = target_location
    try:
        entry = resolver.Resolver.OpenFileEntry(start_spec)
    except Exception:
        entry = None
    display = target_location.rstrip("/") or "/"
    return entry, display


def _list_directory_names(
    fs_type: str,
    fs_spec,
    start_location: str,
    cache: dict[str, list[str]],
) -> list[str]:
    if start_location in cache:
        return cache[start_location]
    entry, _ = _open_entry(fs_type, fs_spec, start_location)
    names: list[str] = []
    if entry:
        try:
            # Sort entries for deterministic iteration order
            entries = list(entry.sub_file_entries)
            entries.sort(key=lambda e: getattr(e, "name", ""))

            for child in entries:
                name = getattr(child, "name", None)
                if name and name not in {".", ".."}:
                    names.append(name)
        except Exception:
            names = []
    cache[start_location] = names
    return names


def _data_variants(pattern: str, target: str | None) -> list[str]:
    prefixes = ["Users/", "Library/", "private/", "var/"]
    normalized = pattern.replace("\\", "/")
    stripped = normalized.lstrip("/")
    variants: list[str] = []

    # Generate System/Volumes/Data variants
    for prefix in prefixes:
        if stripped.lower().startswith(prefix):
            candidate = "System/Volumes/Data/" + stripped
            candidate = candidate.replace("//", "/")
            if not candidate.startswith("/"):
                candidate = "/" + candidate
            variants.append(candidate)

    # Only add companion file variants for specific database files, not wildcard patterns
    # Wildcard patterns (containing *, ?, [, ]) should not have companion suffixes
    has_glob_chars = any(char in normalized for char in ["*", "?", "[", "]"])

    if not has_glob_chars:
        # Add companion file variants (WAL, SHM, Journal) for SQLite databases
        # These contain uncommitted data and should always be captured
        companion_suffixes = ["-wal", "-shm", "-journal"]
        base_variants = list(variants) if variants else [normalized]

        variants.extend(base_pattern + suffix for base_pattern in base_variants for suffix in companion_suffixes)

        # Also add companion suffixes to original pattern
        if not variants:  # If no System/Volumes/Data variants were created
            variants.extend(normalized + suffix for suffix in companion_suffixes)

    return variants
