#!/usr/bin/env python3
"""
Global Debug Logging System

Provides centralized logging functionality that integrates with config.ui.debug
and Rich console output for consistent, well-formatted debug messages across
the application.

Design:
- Single global logger instance accessible throughout the application
- Reads config.ui.debug flag to control debug message visibility
- All output styled via Rich console.print for consistency with progress bars
- Thread-safe for multithreaded operations
- Supports both CLI and TUI modes (with shared console instance)

Usage:
    >>> from mars.utils.debug_logger import logger
    >>> logger.debug("Processing file...")      # Only shown if config.ui.debug=True
    >>> logger.info("Task completed")           # Always shown
    >>> logger.warning("Possible issue found")  # Always shown (yellow)
    >>> logger.error("Operation failed")        # Always shown (red)

Migration from existing patterns:
    self.log(msg, "INFO")     → logger.info(msg)
    self.log(msg, "WARNING")  → logger.warning(msg)
    self.log(msg, "ERROR")    → logger.error(msg)
    self._debug(msg)          → logger.debug(msg)
    print(msg)                → logger.info(msg) or logger.debug(msg)
"""

from __future__ import annotations

import contextlib
import threading
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

from rich.console import Console
from rich.text import Text

if TYPE_CHECKING:
    from mars.config import MARSConfig


class DebugLogger:
    """
    Global debug logger with Rich console integration.

    Provides styled logging output with automatic debug flag checking.
    Thread-safe for concurrent operations.
    """

    def __init__(self):
        """Initialize the logger with default console and no config."""
        self._console: Console | None = None
        self._config: MARSConfig | None = None
        self._lock = threading.Lock()
        self._default_console = Console(stderr=True)  # Default to stderr for logs
        self._log_file: TextIO | None = None
        self._log_path: Path | None = None

    def configure(
        self,
        config: MARSConfig,
        console: Console | None = None,
        project_dir: Path | None = None,
    ) -> None:
        """
        Configure the logger with application config and optional console.

        Args:
            config: Application configuration (for reading config.ui.debug)
            console: Optional Rich console instance (for TUI mode)
                     If None, creates a default stderr console
            project_dir: Optional project directory for log file output
        """
        with self._lock:
            self._config = config
            self._console = console

            # Handle log file - close existing if any
            self._close_log_file()

            # Open new log file if enabled and project dir provided
            if config.ui.debug and config.ui.debug_log_to_file and project_dir:
                self._open_log_file(project_dir)

    def _open_log_file(self, project_dir: Path) -> None:
        """Open log file in project directory."""
        self._log_path = project_dir / "mars_debug.log"
        try:
            self._log_file = Path.open(self._log_path, "a", encoding="utf-8")
            # Write session header
            self._log_file.write(f"\n{'=' * 70}\n")
            self._log_file.write(f"MARS Debug Log - {datetime.now().isoformat()}\n")
            self._log_file.write(f"{'=' * 70}\n\n")
            self._log_file.flush()
        except Exception:
            self._log_file = None
            self._log_path = None

    def _close_log_file(self) -> None:
        """Close log file if open."""
        if self._log_file:
            with contextlib.suppress(Exception):
                self._log_file.close()
            self._log_file = None
            self._log_path = None

    def _write_to_file(self, message: str, level: str = "") -> None:
        """Write message to log file (stripping Rich markup)."""
        if self._log_file:
            try:
                # Strip Rich markup for plain text file
                plain = Text.from_markup(message).plain
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                prefix = f"[{timestamp}] {level}: " if level else f"[{timestamp}] "
                self._log_file.write(f"{prefix}{plain}\n")
                self._log_file.flush()
            except Exception:
                pass

    def get_console(self) -> Console:
        """Get the active console instance (shared or default).

        Use this when you need the raw Console object (e.g., for Progress bars).
        For simple output, prefer using logger.info(), logger.warning(), etc.
        """
        if self._console is not None:
            return self._console
        return self._default_console

    def _should_show_debug(self) -> bool:
        """Check if debug messages should be displayed."""
        if self._config is None:
            return False  # No config = no debug output
        return self._config.ui.debug

    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message (only shown if config.ui.debug=True).

        Args:
            message: Debug message to display
            **kwargs: Additional arguments passed to console.print

        Example:
            >>> logger.debug("Processing database: [cyan]example.db[/cyan]")
        """
        if not self._should_show_debug():
            return

        console = self.get_console()
        with self._lock:
            console.print(f"[dim]DEBUG:[/dim] {message}", **kwargs)
            self._write_to_file(message, "DEBUG")

    def info(self, message: str, **kwargs) -> None:
        """
        Log an informational message (always shown).

        Args:
            message: Info message to display
            **kwargs: Additional arguments passed to console.print

        Example:
            >>> logger.info("Scan completed: [green]1,234[/green] files processed")
        """
        console = self.get_console()
        with self._lock:
            console.print(message, **kwargs)
            self._write_to_file(message, "INFO")

    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message (always shown in yellow).

        Args:
            message: Warning message to display
            **kwargs: Additional arguments passed to console.print

        Example:
            >>> logger.warning("Database file is empty, skipping")
        """
        console = self.get_console()
        with self._lock:
            console.print(f"[yellow]WARNING:[/yellow] {message}", **kwargs)
            self._write_to_file(message, "WARNING")

    def error(self, message: str, **kwargs) -> None:
        """
        Log an error message (always shown in red).

        Args:
            message: Error message to display
            **kwargs: Additional arguments passed to console.print

        Example:
            >>> logger.error("Failed to open database: permission denied")
        """
        console = self.get_console()
        with self._lock:
            console.print(f"[red]ERROR:[/red] {message}", **kwargs)
            self._write_to_file(message, "ERROR")

    def separator(self, char: str = "=", length: int = 70) -> None:
        """
        Print a separator line (useful for section breaks).

        Args:
            char: Character to use for separator
            length: Length of separator line

        Example:
            >>> logger.separator()
            >>> logger.info("Section Title")
            >>> logger.separator()
        """
        if not self._should_show_debug():
            return

        console = self.get_console()
        with self._lock:
            console.print(char * length)


# Global logger instance
# Configure with logger.configure(config, console) before use
logger = DebugLogger()


# ============================================================================
# Convenience Decorators for Migration
# ============================================================================


def debug_method(func):
    """
    Decorator to add debug logging to method calls.

    Example:
        >>> @debug_method
        ... def process_file(self, path: Path):
        ...     # Will log: "Calling process_file(path='/data/file.db')"
        ...     pass
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logger.debug(f"Calling {func_name}({args[1:]=}, {kwargs=})")
        result = func(*args, **kwargs)
        logger.debug(f"{func_name} completed")
        return result

    return wrapper
