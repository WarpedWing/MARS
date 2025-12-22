"""
Progress interface for MARS report modules.

This module provides a way for report modules to communicate progress
back to the module_runner, which can then update the UI progress bar.

Usage in modules (optional - modules that don't use this will work fine):

    from mars.report_modules.progress_interface import get_progress

    def main():
        files = list(input_path.rglob("*"))

        progress = get_progress()
        if progress:
            progress.set_total(len(files))

        for f in files:
            process(f)
            if progress:
                progress.advance()  # or progress.update(message="Processing...")

If no progress interface is set by the runner (e.g., running standalone),
get_progress() returns None and the module runs without progress updates.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class ModuleProgress:
    """Interface for modules to report progress to the runner.

    Attributes:
        total: Total number of items to process (None for indeterminate)
        current: Current progress count
        message: Optional status message
    """

    _callback: Callable[[int, int | None, str | None], None] | None = None
    _current: int = 0
    _total: int | None = None
    _message: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def set_total(self, total: int | None) -> None:
        """Set total items to process.

        Args:
            total: Number of items, or None for indeterminate progress (throbbing bar)

        Example:
            progress.set_total(len(files))  # Known count
            progress.set_total(None)        # Unknown count (bar will throb)
        """
        with self._lock:
            self._total = total
            self._notify()

    def update(
        self,
        current: int | None = None,
        message: str | None = None,
        advance: int = 0,
    ) -> None:
        """Update progress.

        Args:
            current: Set absolute progress value (overrides advance)
            message: Optional status message to display
            advance: Increment progress by this amount (default 0)

        Examples:
            progress.update(current=5)           # Set to 5
            progress.update(advance=1)           # Increment by 1
            progress.update(message="Step 2")    # Update message only
            progress.update(advance=1, message="Processing file.db")
        """
        with self._lock:
            if current is not None:
                self._current = current
            elif advance:
                self._current += advance

            if message is not None:
                self._message = message

            self._notify()

    def advance(self, amount: int = 1, message: str | None = None) -> None:
        """Advance progress by amount.

        Convenience method for the common case of incrementing by 1.

        Args:
            amount: How much to increment (default 1)
            message: Optional status message

        Example:
            for file in files:
                process(file)
                progress.advance()  # Increment by 1
        """
        self.update(advance=amount, message=message)

    def set_message(self, message: str) -> None:
        """Update status message without changing progress count.

        Args:
            message: Status message to display
        """
        self.update(message=message)

    @property
    def total(self) -> int | None:
        """Get current total (read-only)."""
        return self._total

    @property
    def current(self) -> int:
        """Get current progress (read-only)."""
        return self._current

    @property
    def message(self) -> str | None:
        """Get current message (read-only)."""
        return self._message

    def _notify(self) -> None:
        """Notify callback of progress update (internal)."""
        if self._callback:
            self._callback(self._current, self._total, self._message)


# Module-level progress instance (set by ModuleRunner before executing module)
_progress: ModuleProgress | None = None
_progress_lock = threading.Lock()


def get_progress() -> ModuleProgress | None:
    """Get the current progress interface.

    Returns:
        ModuleProgress instance if running under ModuleRunner with progress,
        None if running standalone or progress tracking is disabled.

    Usage:
        progress = get_progress()
        if progress:
            progress.set_total(100)
            for i in range(100):
                do_work()
                progress.advance()
    """
    with _progress_lock:
        return _progress


def set_progress(progress: ModuleProgress | None) -> None:
    """Set the progress interface. Called by ModuleRunner before executing module.

    Note: This is intended for internal use by ModuleRunner. Modules should
    use get_progress() to access progress reporting.

    Args:
        progress: ModuleProgress instance, or None to clear
    """
    global _progress
    with _progress_lock:
        _progress = progress
