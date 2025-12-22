"""
Module runner for MARS report modules.

Executes modules with proper sys.argv isolation and error handling.
"""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from rich.console import Console

from mars.report_modules.progress_interface import (
    ModuleProgress,
    set_progress,
)
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from pathlib import Path

    from mars.report_modules.module_config import ModuleConfig


@dataclass
class ModuleResult:
    """Result from running a module."""

    success: bool
    """Whether the module ran successfully"""

    module_name: str
    """Display name of the module"""

    duration_seconds: float
    """Execution time in seconds"""

    error: str | None = None
    """Error message if failed"""

    files_created: list[Path] | None = None
    """List of output files created (if available)"""

    records_processed: int | None = None
    """Number of records processed (if available)"""


class ModuleRunner:
    """Executes report modules with isolation and error handling."""

    def __init__(self, console: Console | None = None):
        """Initialize module runner.

        Args:
            console: Rich console for logging (optional)
        """
        self.console = console or Console()

    def run(
        self,
        module: ModuleConfig,
        args: list[str],
    ) -> ModuleResult:
        """Run a module with the given arguments.

        Args:
            module: Module configuration
            args: Command-line arguments (sys.argv format, excluding program name)

        Returns:
            ModuleResult with execution details

        Notes:
            - Saves and restores sys.argv automatically
            - Captures exceptions and returns structured errors
            - Uses dynamic import to load the module
        """
        start_time = time.time()

        # Run module
        try:
            # Import the module dynamically
            module_obj = self._import_module(module)

            # Save original sys.argv
            original_argv = sys.argv.copy()

            try:
                # Set sys.argv for this module
                # argv[0] should be the script name (use entry name)
                sys.argv = [module.entry] + args

                # Call the module's main() function
                if not hasattr(module_obj, "main"):
                    raise AttributeError(f"Module '{module.entry}' has no main() function")

                module_obj.main()

                # Success!
                duration = time.time() - start_time

                return ModuleResult(
                    success=True,
                    module_name=module.name,
                    duration_seconds=duration,
                )

            finally:
                # Always restore sys.argv
                sys.argv = original_argv

        except Exception as e:
            # Module execution failed
            duration = time.time() - start_time
            error_msg = f"{type(e).__name__}: {e}"

            logger.error(f"  Failed: {error_msg}")

            return ModuleResult(
                success=False,
                module_name=module.name,
                duration_seconds=duration,
                error=error_msg,
            )

    def _import_module(self, module: ModuleConfig):
        """Dynamically import a module.

        Args:
            module: Module configuration

        Returns:
            Imported module object

        Raises:
            ImportError: If module cannot be imported
        """
        # Build import path
        module_path = module.module_path
        relative_path = module_path.relative_to(module_path.parent.parent)  # Get path relative to report_modules/

        # Convert path to module name
        # report_modules/firefox_cache_parser -> mars.report_modules.firefox_cache_parser
        parts = ["mars"] + list(relative_path.parts)
        module_import_path = ".".join(parts) + f".{module.entry}"

        return importlib.import_module(module_import_path)

    def run_with_progress(
        self,
        module: ModuleConfig,
        args: list[str],
        task_id: Any = None,
        progress: Any = None,
    ) -> ModuleResult:
        """Run a module with Rich progress tracking.

        Args:
            module: Module configuration
            args: Command-line arguments
            task_id: Rich progress task ID (optional)
            progress: Rich Progress instance (optional)

        Returns:
            ModuleResult with execution details

        Note:
            Sets up a ModuleProgress interface that the module can optionally
            use to report fine-grained progress. Modules can import and use:

                from mars.report_modules.progress_interface import get_progress

                progress = get_progress()
                if progress:
                    progress.set_total(len(items))
                    for item in items:
                        process(item)
                        progress.advance()
        """
        # Update progress to show module is running
        if progress and task_id is not None:
            progress.update(task_id, description=f"[cyan]Running: {module.name}[/cyan]")

        # Set up module progress interface with callback to update Rich progress
        module_progress: ModuleProgress | None = None
        if progress and task_id is not None:
            module_progress = ModuleProgress()

            def progress_callback(current: int, total: int | None, message: str | None) -> None:
                """Callback to update Rich progress from module progress."""
                if total is not None and total > 0:
                    # Determinate progress - show percentage with filled bar
                    pct = min(99, int((current / total) * 100))  # Cap at 99% until done
                    desc = f"[cyan]{module.name}[/cyan] [{current}/{total}]"
                    if message:
                        desc = f"[cyan]{module.name}[/cyan] - {message}"
                    # Set total=100 so the bar fills based on completed percentage
                    progress.update(task_id, description=desc, completed=pct, total=100)
                elif message:
                    # Indeterminate but with message
                    progress.update(task_id, description=f"[cyan]{module.name}[/cyan] - {message}")

            module_progress._callback = progress_callback

        # Set the global progress interface for the module to use
        set_progress(module_progress)

        try:
            # Run the module
            result = self.run(module, args)
        finally:
            # Always clean up the progress interface
            set_progress(None)

        # Update progress based on result
        if progress and task_id is not None:
            if result.success:
                progress.update(
                    task_id,
                    description=f"[green][bold]✓[/bold] {module.name}[/green]",
                    completed=100,
                )
            else:
                progress.update(
                    task_id,
                    description=f"[red][bold]✗[/bold] {module.name}[/red] - {result.error}",
                    completed=100,
                )
            # Stop the task timer so elapsed time doesn't keep incrementing
            progress.stop_task(task_id)

        return result

    def validate_module(self, module: ModuleConfig) -> tuple[bool, list[str]]:
        """Validate that a module can be executed.

        Args:
            module: Module configuration

        Returns:
            Tuple of (is_valid, error_messages)

        Checks:
            - Module directory exists
            - Entry file exists
            - Module can be imported
            - Module has main() function
        """
        errors = []

        # Check module path
        if not module.module_path.exists():
            errors.append(f"Module directory not found: {module.module_path}")
            return False, errors

        # Check entry file
        entry_file = module.module_path / f"{module.entry}.py"
        if not entry_file.exists():
            errors.append(f"Entry file not found: {entry_file}")
            return False, errors

        # Try to import
        try:
            module_obj = self._import_module(module)

            # Check for main() function
            if not hasattr(module_obj, "main"):
                errors.append(f"Module '{module.entry}' has no main() function")
                return False, errors

        except ImportError as e:
            errors.append(f"Cannot import module: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Validation error: {e}")
            return False, errors

        return True, []
