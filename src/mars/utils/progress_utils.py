#!/usr/bin/env python3
"""
Progress Bar Utilities with Panel-based Status Display

Provides factory functions for creating consistent, well-formatted progress bars
across the application with optional Panel wrappers for enhanced visual presentation.

Design Features:
- Panel wrapper with optional header metadata
- Static labels (prevent layout shift) + dynamic content in separate columns
- Multithreaded progress support (ThreadPoolExecutor + as_completed)
- Configurable time display (elapsed, remaining, or countdown)
- Consistent formatting across all progress bars

Example:
    >>> with create_standard_progress(
    ...     label="Processing databases",
    ...     header_title="Database Analysis",
    ...     show_time="remaining"
    ... ) as progress:
    ...     task = progress.add_task("Processing...", total=100)
    ...     for i in range(100):
    ...         progress.update(task, advance=1, description=f"File: {i}.db")
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from rich.console import Console

    from mars.config import MARSConfig


# ============================================================================
# Core Progress Bar Factory Functions
# ============================================================================


class _NoOpTask:
    """Mock task object for _NoOpProgress."""

    def __init__(self, total=None):
        """Initialize mock task."""
        self.total = total
        self.completed = 0
        self.description = ""


class _NoOpProgress:
    """No-op Progress object for when progress bars are disabled.

    Implements the minimal Progress interface needed for compatibility.
    When progress bars are disabled, prints plain text progress updates
    to provide visibility into what's happening.
    """

    def __init__(self, console=None, config=None, verbose: bool = True):
        """
        Initialize with optional console for compatibility.

        Args:
            console: Rich Console for output
            config: Config object (to check debug settings)
            verbose: If True, print plain text progress updates
        """
        self.console = console
        self.config = config
        self.verbose = verbose
        self.tasks = {}  # Dictionary to store mock tasks
        self._last_update = {}  # Track last update per task to avoid spam

    def add_task(self, description: str, **kwargs) -> TaskID:
        """Add a task. Prints task description if verbose."""
        task_id = TaskID(len(self.tasks))
        total = kwargs.get("total")
        task = _NoOpTask(total=total)
        task.description = description
        self.tasks[task_id] = task

        # Print task start
        if self.verbose and self.console and description:
            # Format output based on whether we have a total
            if total is not None:
                self.console.print(f"[dim]→[/dim] {description} [dim]({total} items)[/dim]")
            else:
                self.console.print(f"[dim]→[/dim] {description}")

        return task_id

    def update(self, task_id: TaskID, **kwargs) -> None:
        """Update a task. Prints significant updates if verbose."""
        if task_id not in self.tasks:
            return

        task = self.tasks[task_id]
        description_changed = False
        completed_changed = False

        if "total" in kwargs:
            task.total = kwargs["total"]
        if "completed" in kwargs:
            old_completed = task.completed
            task.completed = kwargs["completed"]
            completed_changed = old_completed != task.completed
        if "description" in kwargs:
            old_desc = task.description
            task.description = kwargs["description"]
            description_changed = old_desc != task.description

        # Print updates for description changes (e.g., current file being processed)
        if self.verbose and self.console:
            if description_changed and task.description:
                if task.total and task.total > 50000:
                    # For large totals, print description on one line to prevent spam
                    self.console.print(f"[dim]  └─[/dim] {task.description}", end="\r")
                    self.console.print()  # Newline after overwrite
                else:
                    self.console.print(f"[dim]  ├─[/dim] {task.description}")
            # Print milestone completions (every 10%, or at specific intervals)
            elif completed_changed and task.total and task.total > 0:
                progress_pct = (task.completed / task.total) * 100
                last_pct = self._last_update.get(task_id, -1)
                # Print every 25% or at completion
                if progress_pct >= last_pct + 25 or task.completed == task.total or progress_pct == 100:
                    if task.completed == task.total:
                        self.console.print(f"[dim]  └─[/dim] Completed: {task.completed}/{task.total}")
                    else:
                        self.console.print(
                            f"[dim]  ├─[/dim] Progress: {task.completed}/{task.total} ({progress_pct:.0f}%)"
                        )
                    self._last_update[task_id] = progress_pct

    def remove_task(self, task_id: TaskID) -> None:
        """Remove a task."""
        if task_id in self.tasks:
            del self.tasks[task_id]
        if task_id in self._last_update:
            del self._last_update[task_id]

    def start_task(self, task_id: TaskID) -> None:
        """Start a task (no-op)."""
        pass

    def stop_task(self, task_id: TaskID) -> None:
        """Stop a task (no-op)."""
        pass


@contextmanager
def _no_op_progress() -> Iterator[_NoOpProgress]:
    """Context manager that yields a no-op progress object.

    Used when progress bars are disabled in config.
    """
    yield _NoOpProgress()


@contextmanager
def create_standard_progress(
    label: str,
    console: Console | None = None,
    header_title: str | None = None,
    header_subtitle: str | None = None,
    show_time: Literal["elapsed", "remaining", "none"] = "remaining",
    show_percentage: bool = True,
    show_count: bool = False,
    transient: bool = True,
    panel_style: str = "deep_sky_blue3",
    config: MARSConfig | None = None,
) -> Iterator[Progress | _NoOpProgress]:
    """
    Create a standard single-threaded progress bar with optional Panel wrapper.

    The progress bar uses a static label to prevent layout shift when the dynamic
    description (filename/path) changes.

    Args:
        label: Static label text (e.g., "Processing", "Scanning")
        console: Rich console instance (optional)
        header_title: Optional panel header title
        header_subtitle: Optional panel header subtitle (metadata, stats)
        show_time: Time display mode ("elapsed", "remaining", or "none")
        show_percentage: Show percentage complete
        show_count: Show count (n/total) instead of percentage
        transient: Hide progress bar after completion (default: False)
        panel_style: Panel border style/color (default: "deep_sky_blue3")
        config: MARS config (if provided, respects show_progress_bars setting)

    Yields:
        Progress context manager

    Example:
        >>> with create_standard_progress(
        ...     label="Processing",
        ...     header_title="Database Analysis"
        ... ) as progress:
        ...     task = progress.add_task("Loading...", total=100)
        ...     for i in range(100):
        ...         progress.update(task, advance=1, description=f"File: {i}.db")
    """
    # Check if progress bars are disabled in config
    if config is not None and not config.ui.show_progress_bars:
        yield _NoOpProgress(console=console, config=config)
        return

    # Build column layout: Spinner + Static Label + Bar + Metrics + Time + Dynamic Content
    columns = [
        SpinnerColumn(),
        TextColumn(f"[bold]{label}[/bold]"),
        BarColumn(),
    ]

    if show_percentage:
        columns.append(TaskProgressColumn())
    if show_count:
        columns.append(MofNCompleteColumn())

    if show_time == "elapsed":
        columns.append(TimeElapsedColumn())
    elif show_time == "remaining":
        columns.append(TimeRemainingColumn())

    # Dynamic content
    columns.append(TextColumn("{task.description}"))

    # Create progress bar
    progress = Progress(*columns, console=console)

    # Wrap in Panel with Live rendering
    progress_panel = create_panel_group(
        header_title,
        header_subtitle,
        progress,
        label,
        None,
        panel_style,
    )

    # Use Live to continuously update the Panel containing the Progress
    # Note: Only enter Live context, not progress context, to avoid double rendering
    with Live(progress_panel, console=console, refresh_per_second=10, transient=transient):
        yield progress


@contextmanager
def create_multithreaded_progress(
    label: str,
    console: Console | None = None,
    header_title: str | None = None,
    header_subtitle: str | None = None,
    show_time: Literal["elapsed", "remaining"] = "remaining",
    transient: bool = True,
    panel_style: str = "deep_sky_blue3",
    config: MARSConfig | None = None,
) -> Iterator[Progress | _NoOpProgress]:
    """
    Create a multithreaded progress bar for ThreadPoolExecutor + as_completed pattern.

    Optimized for parallel processing with dynamic updates as tasks complete.

    Args:
        label: Static label text (e.g., "Processing", "Categorizing")
        console: Rich console instance (optional)
        header_title: Optional panel header title
        header_subtitle: Optional panel header subtitle (metadata, stats)
        show_time: Time display mode ("elapsed" or "remaining")
        transient: Hide progress bar after completion (default: False)
        panel_style: Panel border style/color (default: "deep_sky_blue3")
        config: MARS config (if provided, respects show_progress_bars setting)

    Yields:
        Progress context manager

    Example:
        >>> with create_multithreaded_progress("Processing", "Database Processing") as progress:
        ...     task = progress.add_task("Processing...", total=len(files))
        ...     with ThreadPoolExecutor(max_workers=4) as executor:
        ...         futures = {executor.submit(process, f): f for f in files}
        ...         for future in as_completed(futures):
        ...             result = future.result()
        ...             progress.update(task, advance=1, description=f"Completed: {result}")
    """
    # Check if progress bars are disabled in config
    if config is not None and not config.ui.show_progress_bars:
        yield _NoOpProgress(console=console, config=config)
        return

    columns = [
        SpinnerColumn(),
        TextColumn(f"[bold]{label}[/bold]"),
        BarColumn(),
        TaskProgressColumn(),
    ]

    if show_time == "elapsed":
        columns.append(TimeElapsedColumn())
    else:
        columns.append(TimeRemainingColumn())

    # Dynamic content
    columns.append(TextColumn("{task.description}"))

    progress = Progress(*columns, console=console)

    # Wrap in Panel with Live rendering
    progress_panel = create_panel_group(header_title, header_subtitle, progress, label, None, panel_style)

    # Use Live to continuously update the Panel containing the Progress
    # Note: Only enter Live context, not progress context, to avoid double rendering
    with Live(progress_panel, console=console, refresh_per_second=10, transient=transient):
        yield progress


# ============================================================================
# Helper Functions for Common Patterns
# ============================================================================
def create_panel_group(
    header_title: str | None,
    header_subtitle: str | None,
    progress: Progress,
    label: str,
    dynamic_text: str | None,
    panel_style: str = "deep_sky_blue3",
) -> Panel:
    """
    Create a Panel with optional header metadata.

    Args:
        header_title: Panel header title (label will be used if this isn't set)
        header_subtitle: Panel header subtitle (metadata, stats)
        progress: Progress object to wrap
        label: Static label text (e.g., "Processing", "Categorizing")
        dynamic_text: Displays below the progress bar. Good for filenames to prevent layout shift.
        panel_style: Panel border style/color (default: "deep_sky_blue3")

    Returns:
        Panel with optional header metadata
    """
    panel_title = header_title if header_title else label
    panel_subtitle = f"[bold deep_sky_blue1]{header_subtitle}[/bold deep_sky_blue1]\n" if header_subtitle else None
    panel_dynamic_text = f"\n[bold light_goldenrod3]{dynamic_text}[/bold light_goldenrod3]" if dynamic_text else None

    panel_group = progress
    if panel_subtitle and panel_dynamic_text:
        panel_group = Group(panel_subtitle, progress, panel_dynamic_text)
    elif panel_subtitle and not panel_dynamic_text:
        panel_group = Group(panel_subtitle, progress)

    panel = Panel(
        panel_group,
        title=f"[bold deep_sky_blue1]{panel_title}[/bold deep_sky_blue1]",
        border_style=panel_style,
        padding=(2, 2),
    )

    return panel


def format_file_description(file_path: Path, max_length: int = 50) -> str:
    """
    Format a file path for display in progress bar dynamic content.

    Truncates long paths to fit within progress bar width.

    Args:
        file_path: Path to format
        max_length: Maximum length for display

    Returns:
        Formatted path string

    Example:
        >>> format_file_description(Path("/very/long/path/to/file.db"))
        " .../path/to/file.db"
    """
    path_str = str(file_path)
    if len(path_str) > max_length:
        # Keep filename and parent dirs, truncate middle
        parts = file_path.parts
        filename = file_path.name
        path_str = f".../{'/'.join(parts[-2:])}" if len(parts) > 2 else f".../{filename}"

    return f" {path_str}"


# ============================================================================
# Hierarchical Progress Context for LF Processing
# ============================================================================


class LFProgressContext:
    """
    Hierarchical progress context for LF (Lost & Found) processing.

    Manages a main task (showing phase) and sub-tasks (showing current operation).
    Sub-tasks appear below the main progress bar and are removed when complete.

    The main progress bar moves proportionally with sub-task completion:
    - If main task has 7 phases and current phase is 2 (base = 2/7)
    - And sub-task has 50 items, each sub-item adds (1/7)*(1/50) to main bar
    - This creates smooth progress rather than discrete phase jumps

    Example:
        >>> ctx = LFProgressContext(progress, main_task, total_phases=7)
        >>> ctx.update_main("Phase 1/7: Splitting databases", phase=0)
        >>> sub = ctx.create_sub_task("Splitting", total=50, phase_weight=1.0)
        >>> for i in range(50):
        ...     ctx.update_sub(sub, i+1, description=f"Database {i+1}/50")
        >>> ctx.remove_sub(sub)  # Main bar now at phase 1
        >>> ctx.update_main("Phase 2/7: Grouping...", phase=1)
    """

    def __init__(
        self,
        progress: Progress | None,
        main_task: TaskID | None,
        total_phases: int = 7,
    ):
        """
        Initialize progress context.

        Args:
            progress: Rich Progress object (or None if disabled)
            main_task: TaskID for the main phase progress bar
            total_phases: Total number of phases (for proportional calculation)
        """
        self.progress = progress
        self.main_task = main_task
        self.total_phases = total_phases
        self._active_sub_tasks: list[TaskID] = []
        # Track sub-task metadata for proportional progress
        self._sub_task_info: dict[TaskID, dict] = {}
        # Track current phase base for proportional updates
        self._current_phase_base: float = 0.0

    def update_main(
        self,
        description: str,
        completed: int | None = None,
        total: int | None = None,
        phase: int | None = None,
    ) -> None:
        """
        Update the main phase progress bar.

        Args:
            description: Phase description (e.g., "Phase 3/7: Merge (metamatch groups)")
            completed: Current completed value (deprecated, use phase instead)
            total: Total phases (optional, updates total_phases)
            phase: Current phase number (0-indexed). Sets the base for proportional sub-task progress.
        """
        if self.progress is None or self.main_task is None:
            return

        # Update total phases if provided
        if total is not None:
            self.total_phases = total

        # Track current phase base for proportional updates
        if phase is not None:
            self._current_phase_base = float(phase)

        kwargs: dict = {"description": f"[cyan]{description}"}

        # Use phase for completed if provided, else use completed directly
        if phase is not None:
            kwargs["completed"] = phase
        elif completed is not None:
            kwargs["completed"] = completed
            self._current_phase_base = float(completed)

        if total is not None:
            kwargs["total"] = total

        self.progress.update(self.main_task, **kwargs)

    def create_sub_task(
        self,
        description: str,
        total: int | None = None,
        phase_weight: float = 1.0,
    ) -> TaskID | None:
        """
        Create a sub-task for detailed operation tracking.

        Args:
            description: Sub-task description (e.g., "Splitting databases")
            total: Total items to process (optional, for percentage display)
            phase_weight: How much of one phase this sub-task represents (default 1.0 = full phase)

        Returns:
            TaskID for the sub-task, or None if progress is disabled
        """
        if self.progress is None:
            return None
        task_id = self.progress.add_task(
            f"[dim]  └─ {description}[/dim]",
            total=total if total else None,
        )
        self._active_sub_tasks.append(task_id)

        # Store metadata for proportional progress
        self._sub_task_info[task_id] = {
            "total": total or 1,
            "phase_weight": phase_weight,
            "phase_base": self._current_phase_base,
            "last_completed": 0,
        }
        return task_id

    def update_sub(
        self,
        task_id: TaskID | None,
        completed: int,
        total: int | None = None,
        description: str | None = None,
    ) -> None:
        """
        Update a sub-task's progress and proportionally update main bar.

        Args:
            task_id: TaskID from create_sub_task
            completed: Current item number
            total: Total items (optional, to update if changed)
            description: New description (optional)
        """
        if self.progress is None or task_id is None:
            return

        # Update sub-task
        kwargs: dict = {"completed": completed}
        if total is not None:
            kwargs["total"] = total
        if description is not None:
            kwargs["description"] = f"[dim]  └─ {description}[/dim]"
        self.progress.update(task_id, **kwargs)

        # Update main bar proportionally
        if task_id in self._sub_task_info and self.main_task is not None:
            info = self._sub_task_info[task_id]

            # Update total if provided
            if total is not None:
                info["total"] = total

            sub_total = info["total"]
            phase_weight = info["phase_weight"]
            phase_base = info["phase_base"]

            if sub_total > 0:
                # Calculate proportional progress within this phase
                # progress = phase_base + (completed / sub_total) * phase_weight
                sub_progress = (completed / sub_total) * phase_weight
                main_progress = phase_base + sub_progress

                self.progress.update(self.main_task, completed=main_progress)
                info["last_completed"] = completed

    def remove_sub(self, task_id: TaskID | None) -> None:
        """
        Remove a sub-task when operation completes.

        Args:
            task_id: TaskID from create_sub_task
        """
        if self.progress is None or task_id is None:
            return
        if task_id in self._active_sub_tasks:
            self._active_sub_tasks.remove(task_id)
        if task_id in self._sub_task_info:
            del self._sub_task_info[task_id]
        self.progress.remove_task(task_id)

    def cleanup(self) -> None:
        """Remove all active sub-tasks."""
        for task_id in self._active_sub_tasks[:]:  # Copy to avoid modification during iteration
            self.remove_sub(task_id)


class _NoOpProgressContext:
    """
    No-op progress context for when progress bars are disabled.

    Provides the same interface as LFProgressContext but does nothing,
    allowing code to use the same API regardless of progress bar settings.
    """

    progress: None
    main_task: None

    def __init__(self, console: Console | None = None):
        """Initialize with optional console for logging."""
        self.progress = None
        self.main_task = None
        self.console = console

    def update_main(
        self,
        description: str,
        completed: int | None = None,
        total: int | None = None,
        phase: int | None = None,
    ) -> None:
        """Log phase update to console if available."""
        if self.console:
            self.console.print(f"[dim]→[/dim] {description}")

    def create_sub_task(
        self,
        description: str,
        total: int | None = None,
        phase_weight: float = 1.0,
    ) -> None:
        """Return None (no sub-tasks when disabled)."""

    def update_sub(
        self,
        task_id: TaskID | None,
        completed: int,
        total: int | None = None,
        description: str | None = None,
    ) -> None:
        """No-op when progress is disabled."""
        pass

    def remove_sub(self, task_id: TaskID | None) -> None:
        """No-op when progress is disabled."""
        pass

    def cleanup(self) -> None:
        """No-op when progress is disabled."""
        pass


# Type alias for progress context (either real or no-op)
ProgressContextType = LFProgressContext | _NoOpProgressContext
