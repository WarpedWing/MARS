"""Export catalog-defined files from an image using dfVFS."""

from __future__ import annotations

import fnmatch
import hashlib
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, cast

from dfdatetime import interface as dfdatetime_interface
from dfvfs.lib import errors as dfvfs_errors
from dfvfs.resolver import resolver

from mars.pipeline.mount_utils import dfvfs_glob_utils as glob_utils
from mars.utils.compression_utils import (
    get_archive_extension,
    is_archive,
)
from mars.utils.debug_logger import logger
from mars.utils.platform_utils import sanitize_windows_path

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass
class ExportRecord:
    """
    Record of a file exported from a forensic image via dfVFS.

    Timestamp fields are the ORIGINAL file timestamps from the forensic image,
    not when MARS processed them.
    """

    virtual_path: str
    export_path: Path
    target_name: str
    md5: str
    size: int
    # Original file timestamps from forensic image (not MARS processing time)
    file_created: datetime | None
    file_modified: datetime | None
    file_accessed: datetime | None


class DFVFSExporter:
    """Copy catalog matches out of an image into a workspace."""

    def __init__(
        self,
        image_path: Path,
        include_fs: Iterable[str] | None = None,
        console=None,
        partition_labels: list[str] | None = None,
        config=None,
    ):
        self.image_path = image_path
        # Include common filesystem types when scanning forensic images.
        self.include_fs = include_fs or {"apfs", "hfs", "tsk", "ntfs"}
        self.console = console
        self.config = config

        self.targets: list[glob_utils.TargetFS] = []
        if image_path.is_dir():
            from dfvfs.lib import definitions as dfvfs_definitions
            from dfvfs.path import os_path_spec

            # In TUI mode (console provided), suppress verbose debug messages
            logger.debug("[dfvfs] Source is a directory - creating synthetic OS target")
            path_spec = os_path_spec.OSPathSpec(location=str(image_path))
            synthetic_target = glob_utils.TargetFS(
                label="root",
                type_indicator=dfvfs_definitions.TYPE_INDICATOR_OS,
                fs_spec=path_spec,
                volume_name=str(image_path),
                size_bytes=None,
            )
            self.targets = [synthetic_target]
        else:
            # Check if source is an archive file
            archive_detected = is_archive(image_path)
            suffix = get_archive_extension(image_path)

            if archive_detected:
                # In TUI mode (console provided), suppress verbose debug messages
                logger.debug(f"[dfvfs] Source is an archive ({suffix}) - creating synthetic target")
                from dfvfs.lib import definitions as dfvfs_definitions
                from dfvfs.path import (
                    gzip_path_spec,
                    os_path_spec,
                    tar_path_spec,
                    zip_path_spec,
                )

                os_path = os_path_spec.OSPathSpec(location=str(image_path))

                if suffix == ".zip":
                    path_spec = zip_path_spec.ZipPathSpec(location="/", parent=os_path)
                    type_indicator = dfvfs_definitions.TYPE_INDICATOR_ZIP
                elif suffix in {".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2"}:
                    if suffix in {".tar.gz", ".tgz"}:
                        gzip_parent = gzip_path_spec.GzipPathSpec(parent=os_path)
                        path_spec = tar_path_spec.TARPathSpec(location="/", parent=gzip_parent)
                    elif suffix in {".tar.bz2", ".tbz2"}:
                        path_spec = tar_path_spec.TARPathSpec(location="/", parent=os_path)
                    else:
                        path_spec = tar_path_spec.TARPathSpec(location="/", parent=os_path)
                    type_indicator = dfvfs_definitions.TYPE_INDICATOR_TAR
                elif suffix == ".gz":
                    path_spec = gzip_path_spec.GzipPathSpec(parent=os_path)
                    type_indicator = dfvfs_definitions.TYPE_INDICATOR_GZIP
                else:
                    path_spec = tar_path_spec.TARPathSpec(location="/", parent=os_path)
                    type_indicator = dfvfs_definitions.TYPE_INDICATOR_TAR

                synthetic_target = glob_utils.TargetFS(
                    label=image_path.stem,
                    type_indicator=type_indicator,
                    fs_spec=path_spec,
                    volume_name=image_path.name,
                    size_bytes=(image_path.stat().st_size if image_path.exists() else None),
                )
                self.targets = [synthetic_target]
            else:
                self.targets = glob_utils.scan_sources(str(image_path), self.include_fs)

        # Filter targets by partition labels if provided
        if partition_labels:
            original_count = len(self.targets)
            self.targets = [t for t in self.targets if t.label in partition_labels]
            # In TUI mode (console provided), suppress verbose debug messages
            logger.debug(
                f"[dfvfs] Filtered {original_count} partition(s) to {len(self.targets)} (labels: {', '.join(partition_labels)})"
            )

        # In TUI mode (console provided), suppress verbose debug messages
        logger.debug(f"[dfvfs] Prepared exporter for {image_path} (targets={len(self.targets)})")

    def export_catalog(
        self,
        workspace: Path,
        target_names: set[str],
        catalog_path: Path,
    ) -> dict[Path, ExportRecord]:
        # Get excluded file types from config (e.g., {'cache'} to skip Firefox cache)
        excluded_file_types: set[str] = set()
        if self.config and hasattr(self.config, "exemplar"):
            excluded_list = getattr(self.config.exemplar, "excluded_file_types", [])
            excluded_file_types = set(excluded_list) if excluded_list else set()

        if excluded_file_types:
            logger.debug(f"[dfvfs] Excluding file types: {sorted(excluded_file_types)}")

        patterns_map, pattern_targets = glob_utils.load_catalog_patterns(
            catalog_path, excluded_file_types=excluded_file_types
        )
        # In TUI mode (console provided), suppress verbose debug messages
        logger.debug(f"[dfvfs] Catalog patterns loaded: {sum(len(v) for v in patterns_map.values())} variants")
        selected = []
        for target in target_names:
            selected.extend(patterns_map.get(target, []))

        # In TUI mode (console provided), suppress verbose debug messages
        logger.debug(f"[dfvfs] Exporting {len(selected)} pattern(s) across {len(target_names)} target(s)")
        start_map = glob_utils._group_patterns_by_start(selected, cast("dict[str, str | None]", pattern_targets))  # pylint: disable=protected-access
        return self._export_from_map(workspace, start_map, pattern_targets)

    def export_patterns(
        self,
        workspace: Path,
        patterns: Sequence[str],
        target_name: str = "custom",
    ) -> dict[Path, ExportRecord]:
        normalized = [glob_utils._normalize(pat) for pat in patterns]  # pylint: disable=protected-access
        # In TUI mode (console provided), suppress verbose debug messages
        logger.debug(f"[dfvfs] Exporting {len(normalized)} ad-hoc pattern(s) as target '{target_name}'")
        pattern_targets = dict.fromkeys(normalized, target_name)
        start_map = glob_utils._group_patterns_by_start(normalized, cast("dict[str, str | None]", pattern_targets))  # pylint: disable=protected-access
        return self._export_from_map(workspace, start_map, pattern_targets)

    def _export_from_map(
        self,
        workspace: Path,
        start_map,
        pattern_targets,
    ) -> dict[Path, ExportRecord]:
        exported: dict[Path, ExportRecord] = {}
        seen_virtual: set[str] = set()

        for fs in self.targets:
            from dfvfs.lib import definitions as dfvfs_definitions

            fs_name = fs.volume_name or fs.type_indicator
            # In TUI mode (console provided), suppress verbose debug messages
            logger.debug(f"[dfvfs] Walking {fs_name} ({fs.type_indicator})")
            # Show progress in TUI

            is_os_target = fs.type_indicator == dfvfs_definitions.TYPE_INDICATOR_OS
            base_path = fs.volume_name if is_os_target else None

            base = base_path.rstrip("/") if base_path else None

            if is_os_target and base_path:
                adjusted_start_map = {}
                for start_loc, entries in start_map.items():
                    actual_start = (base + start_loc).replace("//", "/")
                    adjusted_start_map.setdefault(actual_start, []).extend(entries)
            else:
                adjusted_start_map = start_map

            files_checked = 0
            files_exported = 0

            sorted_start_items = sorted(adjusted_start_map.items())
            # Check if progress bars should be shown
            show_progress_bars = self.config.ui.show_progress_bars if self.config else True
            use_progress = bool(self.console and sorted_start_items and show_progress_bars)

            # Print plain text header when progress bars are disabled
            if self.console and not show_progress_bars:
                self.console.print(f"[dim]→[/dim] dfVFS Export: ({fs.type_indicator}) {fs_name}")

            pattern_progress = None
            path_text = None
            stats_text = None
            live_context = nullcontext()
            if use_progress:
                from rich.console import Group
                from rich.live import Live
                from rich.panel import Panel
                from rich.progress import (
                    BarColumn,
                    MofNCompleteColumn,
                    Progress,
                    SpinnerColumn,
                    TimeElapsedColumn,
                )
                from rich.text import Text

                pattern_progress = Progress(
                    SpinnerColumn(),
                    "[progress.description]{task.description}",
                    BarColumn(),
                    MofNCompleteColumn(),
                    TimeElapsedColumn(),
                    console=self.console,
                )
                path_text = Text("", style="light_goldenrod3")
                stats_text = Text(
                    "\n  Progress: 0 files checked, 0 exported",
                    style="bold deep_sky_blue1",
                )
                fs_text = (
                    f"[bold cyan]Scanning filesystem[/bold cyan][cyan]: ({fs.type_indicator}) {fs_name}...[/cyan]\n"
                )

                # Wrap Group in Panel for enhanced visual presentation
                content = Group(fs_text, pattern_progress, path_text, stats_text)
                panel = Panel(
                    content,
                    title="[bold deep_sky_blue1]dfVFS Export[/bold deep_sky_blue1]",
                    border_style="deep_sky_blue3",
                    padding=(2, 2),
                )
                live_context = Live(
                    panel,
                    console=self.console,
                    refresh_per_second=10,
                    transient=True,
                )

            with live_context:
                pattern_task = None
                stats_callback = None
                if pattern_progress is not None:
                    pattern_task = pattern_progress.add_task(
                        "[bold]Matching catalog patterns[/bold]",
                        total=None,  # Set after pattern expansion completes
                    )
                    if stats_text is not None:

                        def _update_stats(checked: int, exported: int):
                            stats_text.plain = f"\n  Progress: {checked:,} files checked, {exported:,} exported"  # type: ignore
                            # Live context will auto-refresh, no manual refresh needed

                        stats_callback = _update_stats

                expanded_map: dict[str, list] = {}
                for start_location, pattern_entries in sorted_start_items:
                    expanded = glob_utils._expand_entries_for_start(  # pylint: disable=protected-access
                        fs.type_indicator, fs.fs_spec, start_location, pattern_entries
                    )
                    for new_start, new_entries in sorted(expanded.items()):
                        canonical_start = new_start
                        if is_os_target and base and canonical_start.startswith(base):
                            canonical_start = canonical_start[len(base) :]
                            if not canonical_start.startswith("/"):
                                canonical_start = "/" + canonical_start.lstrip("/")
                            canonical_start = canonical_start or "/"

                        converted_entries = []
                        for entry in new_entries:
                            canonical_pattern = entry.pattern
                            if is_os_target and base and canonical_pattern.startswith(base):
                                canonical_pattern = canonical_pattern[len(base) :]
                                if not canonical_pattern.startswith("/"):
                                    canonical_pattern = "/" + canonical_pattern.lstrip("/")
                                canonical_pattern = canonical_pattern or "/"

                            canonical_entry = glob_utils._make_pattern_entry(  # pylint: disable=protected-access
                                canonical_pattern,
                                entry.target,
                                canonical_start,
                            )
                            converted_entries.append(canonical_entry)

                        expanded_map.setdefault(canonical_start, []).extend(converted_entries)

                # Sort start locations by depth (deepest first) for correct precedence
                # This ensures /diagnostics/Special is processed before /diagnostics,
                # so specific patterns match files before catch-all patterns
                def start_location_key(item):
                    location, _ = item
                    depth = location.count("/")  # More slashes = deeper path
                    return (
                        -depth,
                        location,
                    )  # Negative depth for descending order, then alphabetical

                sorted_items = sorted(expanded_map.items(), key=start_location_key)

                prepared_sets: list[tuple[str, list]] = []
                for start_location, pattern_entries in sorted_items:
                    prepared_sets.append(
                        (
                            start_location,
                            sorted(
                                pattern_entries,
                                key=lambda e: _pattern_specificity(e.pattern),
                            ),
                        )
                    )

                total_patterns = sum(len(entries) for _, entries in prepared_sets)
                progress_ready = (
                    use_progress and pattern_progress is not None and pattern_task is not None and total_patterns > 0
                )
                if progress_ready and pattern_progress is not None and pattern_task is not None:
                    pattern_progress.update(pattern_task, total=total_patterns)
                    pattern_progress.refresh()
                    for start_location, pattern_entries in prepared_sets:
                        if path_text is not None:
                            display_location = start_location
                            max_width = 100
                            if len(display_location) > max_width:
                                display_location = "..." + display_location[-(max_width - 3) :]
                            path_text.plain = "  " + display_location
                        # Live context will auto-refresh, no manual refresh needed
                        files_checked, files_exported = self._process_start_location(
                            fs,
                            start_location,
                            pattern_entries,
                            is_os_target,
                            base_path,
                            pattern_targets,
                            workspace,
                            seen_virtual,
                            exported,
                            files_checked,
                            files_exported,
                            progress_callback=stats_callback,
                            path_text=path_text,
                        )
                        pattern_progress.advance(pattern_task, len(pattern_entries))
                else:
                    # Plain text output when progress bars are disabled
                    for start_location, pattern_entries in prepared_sets:
                        # Print location being scanned
                        if self.console and not show_progress_bars:
                            display_location = start_location
                            max_width = 100
                            if len(display_location) > max_width:
                                display_location = "..." + display_location[-(max_width - 3) :]
                            self.console.print(f"[dim]  ├─[/dim] {display_location}")

                        files_checked, files_exported = self._process_start_location(
                            fs,
                            start_location,
                            pattern_entries,
                            is_os_target,
                            base_path,
                            pattern_targets,
                            workspace,
                            seen_virtual,
                            exported,
                            files_checked,
                            files_exported,
                            progress_callback=stats_callback,
                            path_text=None,  # No live update for plain text mode
                        )

                # Progress context handled by Live; no explicit stop needed

            # Clear the progress line and show final count for this filesystem
            if self.console and files_checked > 0 and not show_progress_bars:
                # Use tree-style formatting for consistency
                self.console.print(
                    f"[dim]  └─[/dim] Completed: {files_checked:,} files checked, {files_exported:,} exported"
                )

        return exported

    def _process_start_location(
        self,
        fs,
        start_location: str,
        pattern_entries,
        is_os_target: bool,
        base_path: str | None,
        pattern_targets,
        workspace: Path,
        seen_virtual: set[str],
        exported: dict[Path, ExportRecord],
        files_checked: int,
        files_exported: int,
        progress_callback=None,
        path_text=None,
    ) -> tuple[int, int]:
        """Process a single start location and return updated counters."""
        dir_filters: set[str] = set()
        for entry in pattern_entries:
            dir_filters.update(entry.dir_prefixes)
        if dir_filters:
            root_filter = start_location.rstrip("/") or "/"
            dir_filters.add(root_filter if root_filter.startswith("/") else "/" + root_filter)

        actual_start_location = start_location
        if is_os_target and base_path:
            actual_start_location = (base_path.rstrip("/") + start_location).replace("//", "/")

        adjusted_filters = dir_filters
        if dir_filters and is_os_target and base_path:
            base = base_path.rstrip("/")
            adjusted_filters = set()
            for flt in dir_filters:
                prefix = flt if flt.startswith("/") else "/" + flt
                adjusted_filters.add((base + prefix).replace("//", "/"))

        # Debug: log the filters being used
        logger.debug(f"[dfvfs] start_location={start_location}, actual={actual_start_location}")
        logger.debug(f"[dfvfs] dir_filters count={len(adjusted_filters)}")
        if adjusted_filters:
            for flt in sorted(adjusted_filters)[:10]:  # Show first 10
                logger.debug(f"[dfvfs]   filter: {flt}")
            if len(adjusted_filters) > 10:
                logger.debug(f"[dfvfs]   ... and {len(adjusted_filters) - 10} more")

        report_interval = 500
        last_report = files_checked

        # Phase 1: Collect all matching files first (enables sorted/sequential export)
        pending_exports: list[tuple[str, str, str, object]] = []
        # Each tuple: (virtual_path, logical_virtual_path, target_name, file_entry)

        # Create status callback for directory enumeration progress
        def _update_enum_status(status: str) -> None:
            if path_text is not None:
                path_text.plain = "  " + status

        for virtual_path, file_entry in glob_utils.iter_entries(
            fs.type_indicator,
            fs.fs_spec,
            actual_start_location,
            dir_filters=adjusted_filters,
            status_callback=_update_enum_status if path_text else None,
        ):
            files_checked += 1

            logical_virtual_path = virtual_path
            if is_os_target and base_path:
                base = base_path.rstrip("/")
                if logical_virtual_path == base:
                    logical_virtual_path = "/"
                elif logical_virtual_path.startswith(base + "/"):
                    logical_virtual_path = logical_virtual_path[len(base) :]
                    if not logical_virtual_path.startswith("/"):
                        logical_virtual_path = "/" + logical_virtual_path.lstrip("/")

            # Report progress during collection phase
            if progress_callback and files_checked - last_report >= report_interval:
                last_report = files_checked
                progress_callback(files_checked, files_exported)
                # Update path display to show current directory being scanned
                if path_text is not None:
                    # Show directory portion of path (truncate if too long)
                    dir_path = "/".join(logical_virtual_path.split("/")[:-1]) or "/"
                    max_width = 100
                    if len(dir_path) > max_width:
                        dir_path = "..." + dir_path[-(max_width - 3) :]
                    path_text.plain = "  " + dir_path

            match = _match_target(pattern_entries, logical_virtual_path)
            if match is None:
                continue
            target_name, pattern = match

            key = f"{fs.label}:{logical_virtual_path}"
            if key in seen_virtual:
                continue
            seen_virtual.add(key)

            # Queue for export instead of exporting immediately
            resolved_target = target_name or pattern_targets.get(pattern, "unknown")
            pending_exports.append((virtual_path, logical_virtual_path, resolved_target, file_entry))

        # Phase 2: Sort by virtual path for sequential I/O access
        # Files in same directories will be exported together, reducing segment seeks
        pending_exports.sort(key=lambda x: x[0])

        logger.debug(f"Collected {len(pending_exports)} files to export, sorted for sequential access")

        # Phase 3: Export in sorted order
        total_to_export = len(pending_exports)

        for idx, (
            virtual_path,
            logical_virtual_path,
            target_name,
            file_entry,
        ) in enumerate(pending_exports):
            # Log progress every 100 files during export
            if idx % 100 == 0:
                logger.debug(f"Exporting file {idx + 1}/{total_to_export}: {logical_virtual_path}")

            # Update path_text to show current file being exported
            if path_text is not None:
                display_path = logical_virtual_path
                if len(display_path) > 80:
                    display_path = "..." + display_path[-77:]
                path_text.plain = f"  Exporting: {display_path}"

            record = self._export_entry(
                file_entry,
                logical_virtual_path,
                target_name,
                workspace,
            )
            if record:
                files_exported += 1
                exported[record.export_path] = record

            # Report progress during export phase
            if progress_callback and files_exported % 100 == 0:
                progress_callback(files_checked, files_exported)

        if progress_callback:
            progress_callback(files_checked, files_exported)

        return files_checked, files_exported

    def _export_entry(
        self,
        file_entry,
        virtual_path: str,
        target_name: str,
        workspace: Path,
    ) -> ExportRecord | None:
        # Sanitize path for Windows (replaces : and other reserved characters)
        safe_path = sanitize_windows_path(virtual_path.lstrip("/"))
        dest_path = workspace / Path(safe_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Skip entries that are not regular files
        try:
            is_file = file_entry.IsFile()
        except AttributeError:
            is_file = True
        if not is_file:
            return None

        try:
            file_object = resolver.Resolver.OpenFileObject(file_entry.path_spec)
        except dfvfs_errors.BackEndError as exc:
            logger.debug(f"[dfvfs] skipped {virtual_path}: unable to open ({exc})")
            return None

        md5 = hashlib.md5()
        try:
            # Add timeout for problematic files (especially in segmented Ex01 images)
            # Note: signal.SIGALRM is Unix-only, skip timeout on Windows
            import sys

            use_timeout = sys.platform != "win32"
            alarm_set = False
            if use_timeout:
                import signal

                def timeout_handler(signum, frame):
                    raise TimeoutError(f"File export timeout: {virtual_path}")

                # Set 30 second timeout for file export
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                alarm_set = True

            try:
                with dest_path.open("wb") as out_file:
                    while True:
                        data = file_object.read(4 * 1024 * 1024)
                        if not data:
                            break
                        out_file.write(data)
                        md5.update(data)
            finally:
                if alarm_set:
                    signal.alarm(0)  # pyright: ignore[reportPossiblyUnboundVariable] # Cancel alarm

        except KeyboardInterrupt:
            dest_path.unlink(missing_ok=True)
            raise
        except TimeoutError as exc:
            logger.debug(f"[dfvfs] timeout exporting {virtual_path}: {exc}")
            dest_path.unlink(missing_ok=True)
            return None
        except dfvfs_errors.BackEndError as exc:
            logger.debug(f"[dfvfs] backend error exporting {virtual_path}: {exc}")
            dest_path.unlink(missing_ok=True)
            return None
        except Exception as exc:
            logger.debug(f"[dfvfs] error exporting {virtual_path}: {exc}")
            dest_path.unlink(missing_ok=True)
            return None
        finally:
            file_object.close()

        size = getattr(file_entry, "size", dest_path.stat().st_size)
        created = _get_dfvfs_time(file_entry, "creation")
        modified = _get_dfvfs_time(file_entry, "modification")
        accessed = _get_dfvfs_time(file_entry, "access")

        # Debug: Log if timestamps couldn't be extracted from dfVFS
        if created is None and modified is None:
            logger.debug(
                f"[dfvfs] No timestamps from file_entry for {virtual_path}. "
                f"Available attrs: {[a for a in dir(file_entry) if 'time' in a.lower()]}"
            )

        return ExportRecord(
            virtual_path=virtual_path,
            export_path=dest_path,
            target_name=target_name,
            md5=md5.hexdigest(),
            size=size,
            file_created=created,
            file_modified=modified,
            file_accessed=accessed,
        )


def _pattern_specificity(pattern: str) -> tuple[int, int, int]:
    """
    Calculate pattern specificity for sorting (more specific = lower tuple value).

    Returns tuple of (negative_segments, has_doublestar, wildcards):
    - More specific patterns have MORE path segments (prioritized first)
    - Patterns without ** are more specific than with **
    - Among patterns with same segments, fewer wildcards is more specific
    - Used to sort patterns so most specific are matched first

    Examples:
        **/diagnostics/Special/*.tracev3 → (-2, 1, 3) [2 segments, has **, 3 wildcards]
        **/diagnostics/Persist/*.tracev3 → (-2, 1, 3) [2 segments, has **, 3 wildcards]
        **/diagnostics/*                 → (-1, 1, 2) [1 segment, has **, 2 wildcards]

    When sorted ascending, (-2, 1, 3) comes before (-1, 1, 2),
    so more specific patterns are matched first.
    """
    has_doublestar = 1 if "**" in pattern else 0
    wildcards = pattern.count("*") + pattern.count("?")
    segments = len([p for p in pattern.split("/") if p and p not in ("*", "**")])
    # Return tuple where lower values = more specific
    # IMPORTANT: Prioritize segments first (more segments = more specific)
    # segments: more is better (negate so -2 < -1)
    # has_doublestar: without ** is better (0 < 1)
    # wildcards: fewer is better (natural order)
    return (-segments, has_doublestar, wildcards)


def _match_target(pattern_entries, virtual_path: str):
    """Match a virtual path against pattern entries, supporting ** globstar syntax.

    Patterns are expected to be pre-sorted by specificity (most specific first).
    This ensures that more specific patterns (e.g., **/diagnostics/Special/*)
    are matched before less specific catch-alls (e.g., **/diagnostics/*).

    Uses pre-compiled regex patterns from PatternEntry.compiled_regex for
    patterns containing ** (globstar), avoiding repeated regex compilation.
    """
    for entry in pattern_entries:
        # Use pre-compiled regex for globstar patterns (much faster)
        if entry.compiled_regex is not None:
            if entry.compiled_regex.match(virtual_path):
                return entry.target, entry.pattern
            continue

        # For non-globstar patterns, use fnmatch (fast C implementation)
        if fnmatch.fnmatch(virtual_path, entry.pattern):
            return entry.target, entry.pattern

    return None


def _get_dfvfs_time(file_entry, attribute: str) -> datetime | None:
    """
    Extract timestamp from dfVFS file_entry using multiple methods.

    Different filesystem types expose timestamps in different ways:
    - Direct attributes: creation_time, modification_time, access_time
    - Via GetStatAttribute() method
    - Via _stat_object internal attribute
    """
    mapping = {
        "creation": "creation_time",
        "modification": "modification_time",
        "access": "access_time",
    }
    attr = mapping.get(attribute)
    if not attr:
        return None

    # Method 1: Direct attribute access (most common)
    timestamp = getattr(file_entry, attr, None)
    if timestamp is not None:
        result = _convert_timestamp(timestamp)
        if result is not None:
            return result

    # Method 2: Try GetStatAttribute method (some filesystem implementations)
    try:
        stat_attr_method = getattr(file_entry, "GetStatAttribute", None)
        if stat_attr_method:
            stat_value = stat_attr_method(attr)
            if stat_value is not None:
                result = _convert_timestamp(stat_value)
                if result is not None:
                    return result
    except Exception:
        pass

    # Method 3: Try _stat_object internal attribute
    try:
        stat_obj = getattr(file_entry, "_stat_object", None)
        if stat_obj:
            stat_timestamp = getattr(stat_obj, attr, None)
            if stat_timestamp is not None:
                result = _convert_timestamp(stat_timestamp)
                if result is not None:
                    return result
    except Exception:
        pass

    # Method 4: Try stat_attribute property with different naming
    alt_mapping = {
        "creation_time": ["crtime", "st_birthtime", "st_ctime"],
        "modification_time": ["mtime", "st_mtime"],
        "access_time": ["atime", "st_atime"],
    }
    for alt_attr in alt_mapping.get(attr, []):
        try:
            alt_timestamp = getattr(file_entry, alt_attr, None)
            if alt_timestamp is not None:
                result = _convert_timestamp(alt_timestamp)
                if result is not None:
                    return result
        except Exception:
            pass

    return None


def _convert_timestamp(value) -> datetime | None:
    if value is None:
        return None

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None

    # dfdatetime values - try multiple conversion methods
    date_time_values_cls = getattr(dfdatetime_interface, "DateTimeValues", None)
    if date_time_values_cls and isinstance(value, date_time_values_cls):
        # Method 1: Try CopyToPosixTimestamp (most reliable)
        try:
            posix_ts = value.CopyToPosixTimestamp()
            if posix_ts is not None:
                return datetime.fromtimestamp(posix_ts, tz=UTC)
        except (AttributeError, OverflowError, OSError, ValueError):
            pass

        # Method 2: Try CopyToDateTimeString and parse (fallback)
        try:
            dt_string = value.CopyToDateTimeString()
            if dt_string:
                # Format is typically "YYYY-MM-DD HH:MM:SS" or similar
                parsed = datetime.fromisoformat(dt_string.replace(" ", "T"))
                if parsed.tzinfo is None:
                    return parsed.replace(tzinfo=UTC)
                return parsed.astimezone(UTC)
        except (AttributeError, ValueError):
            pass

        # Method 3: Try CopyToDateTime (older dfdatetime versions)
        try:
            dt_object = value.CopyToDateTime()
            if dt_object is not None:
                if dt_object.tzinfo is None:
                    return dt_object.replace(tzinfo=UTC)
                return dt_object.astimezone(UTC)
        except AttributeError:
            pass

    # Fallback string conversion
    if isinstance(value, str):
        try:
            # Allow ISO formatted strings
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=UTC)
            return parsed.astimezone(UTC)
        except ValueError:
            return None

    # Unknown type
    return None
