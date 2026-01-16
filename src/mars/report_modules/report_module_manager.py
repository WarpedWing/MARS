#!/usr/bin/env python3
"""
Report Module Manager for MARS.

Discovers, configures, and executes report modules based on YAML configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from rich.console import Console

from mars.report_modules.argument_builder import ArgumentBuilder
from mars.report_modules.module_config import ModuleConfig
from mars.report_modules.module_runner import ModuleResult, ModuleRunner
from mars.report_modules.target_resolver import TargetResolver
from mars.utils.debug_logger import logger
from mars.utils.progress_utils import create_standard_progress

if TYPE_CHECKING:
    from mars.config import MARSConfig, ProjectPaths


class ReportModuleManager:
    """Manages discovery and execution of report modules."""

    def __init__(
        self,
        config: MARSConfig | None = None,
        paths: ProjectPaths | None = None,
        console: Console | None = None,
    ):
        """Initialize report module manager.

        Args:
            config: MARS configuration (optional)
            paths: Project paths configuration (optional)
            console: Rich console for output (optional, for TUI mode)
        """
        if console:
            self.console = console
        else:
            self.console = Console()

        # Handle config with backward compatibility
        if config is None:
            from mars.config import MARSConfig

            config = MARSConfig()

        self.config = config
        self.paths = paths

        # Initialize components
        self.arg_builder = ArgumentBuilder()
        self.runner = ModuleRunner(console=self.console)

        # Cache for discovered modules
        self._modules_cache: list[ModuleConfig] | None = None

    def discover_modules(self) -> list[ModuleConfig]:
        """Discover all modules by scanning report_modules/ directory.

        Returns:
            List of ModuleConfig objects (includes inactive modules)

        Note:
            Results are cached after first call.
        """
        if self._modules_cache is not None:
            return self._modules_cache

        # Find report_modules directory
        report_modules_dir = Path(__file__).parent

        modules = []
        errors = []

        # Scan for mars_module.yaml files
        for yaml_path in report_modules_dir.glob("*/mars_module.yaml"):
            try:
                module = ModuleConfig.from_yaml(yaml_path)
                modules.append(module)
            except Exception as e:
                errors.append(f"  Error loading {yaml_path.parent.name}: {e}")
                logger.error(f"  Error loading {yaml_path.parent.name}: {e}")

        if errors:
            logger.error(f"Found {len(modules)} modules with {len(errors)} errors")

        self._modules_cache = modules
        return modules

    def get_active_modules(self, scan_type: str, validate: bool = True) -> list[ModuleConfig]:
        """Get active modules for a specific scan type.

        Args:
            scan_type: 'exemplar', 'candidate', or 'free'
            validate: Whether to validate modules (default: True)

        Returns:
            List of active ModuleConfig objects matching scan_type

        Note:
            Only returns modules where active=True and scan_type matches.
            If validate=True, modules that fail validation are excluded.
        """
        all_modules = self.discover_modules()

        # Filter by active and scan_type
        active = [m for m in all_modules if m.active and m.matches_scan_type(scan_type)]

        if not validate:
            return active

        # Validate modules
        validated = []
        for module in active:
            is_valid, errors = self.runner.validate_module(module)
            if is_valid:
                validated.append(module)
            else:
                logger.error(f"  Module '{module.name}' failed validation: {'; '.join(errors)}")

        return validated

    def run_modules(
        self,
        scan_type: str,
        source_root: Path,
        reports_dir: Path,
        catalog: dict | None = None,
    ) -> dict[str, Any]:
        """Run all active modules for a scan type.

        Args:
            scan_type: 'exemplar', 'candidate', or 'free'
            source_root: Root path of the scan (exemplar or candidate root)
            reports_dir: Base reports directory for module outputs
            catalog: Database catalog dict (loaded from artifact_recovery_catalog.yaml)

        Returns:
            Execution summary dict with:
                - total: Total modules attempted
                - succeeded: Number of successful executions
                - failed: Number of failed executions
                - results: List of ModuleResult objects
                - duration: Total execution time
        """
        import time

        start_time = time.time()

        # Load catalog if not provided
        if catalog is None:
            catalog = self._load_catalog()

        # Get active modules
        modules = self.get_active_modules(scan_type)

        if not modules:
            logger.debug(f"No active modules for scan_type='{scan_type}'")
            return {
                "total": 0,
                "succeeded": 0,
                "failed": 0,
                "results": [],
                "duration": 0.0,
            }

        logger.debug(f"\n[bold cyan]Running {len(modules)} report modules...[/bold cyan]")

        # Create reports directory if it doesn't exist
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Execute modules with progress tracking
        results = []

        with create_standard_progress(
            "Report Modules",
            console=self.console,
            show_time="elapsed",
            show_percentage=False,
            config=self.config,
        ) as progress:
            for module in modules:
                task = progress.add_task(f"[cyan]{module.name}[/cyan]", total=None)

                try:
                    # Resolve target paths
                    resolver = TargetResolver(catalog, source_root)
                    target_paths = resolver.resolve_with_usernames(module.target)

                    if not target_paths:
                        logger.debug(f"  Note: No targets found for module '{module.name}' (target='{module.target}')")
                        progress.update(
                            task,
                            description=f"[yellow][bold]⊘[/bold] {module.name} (no targets)[/yellow]",
                            completed=100,
                        )
                        progress.stop_task(task)
                        continue

                    # Execute for each target match
                    for target_path, username in target_paths:
                        # Build output directory name
                        output_name = module.report_folder_name
                        if username:
                            output_name = f"{output_name}_{username}"

                        output_dir = reports_dir / output_name
                        output_dir.mkdir(parents=True, exist_ok=True)

                        # Build arguments
                        args_config = module.get_arguments()
                        args = self.arg_builder.build(args_config, target_path, output_dir)

                        # Run module
                        result = self.runner.run_with_progress(module, args, task, progress)
                        results.append(result)

                except Exception as e:
                    # Unexpected error during execution setup
                    error_msg = f"Setup error: {e}"
                    logger.error(f"  Error running '{module.name}': {error_msg}")

                    progress.update(
                        task,
                        description=f"[red][bold]✗[/bold] {module.name}[/red] - {error_msg}",
                        completed=100,
                    )
                    progress.stop_task(task)

                    results.append(
                        ModuleResult(
                            success=False,
                            module_name=module.name,
                            duration_seconds=0.0,
                            error=error_msg,
                        )
                    )

        # Calculate summary
        duration = time.time() - start_time
        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded

        summary = {
            "total": len(results),
            "succeeded": succeeded,
            "failed": failed,
            "results": results,
            "duration": duration,
        }

        return summary

    def _load_catalog(self) -> dict:
        """Load database catalog from YAML file.

        Returns:
            Catalog dict

        Note:
            Cached after first load.
        """
        if hasattr(self, "_catalog_cache"):
            return self._catalog_cache

        # Find catalog file
        catalog_path = Path(__file__).parent.parent / "catalog" / "artifact_recovery_catalog.yaml"

        if not catalog_path.exists():
            logger.error(f"Warning: Catalog not found at {catalog_path}")
            return {}

        with catalog_path.open() as f:
            catalog = yaml.safe_load(f)

        self._catalog_cache = catalog
        return catalog


# ============================================================================
# Main entry point
# ============================================================================


def main():
    """Test module discovery and listing."""
    manager = ReportModuleManager()

    logger.info("Discovering modules...")
    modules = manager.discover_modules()

    logger.info(f"\nFound {len(modules)} modules:\n")
    for module in modules:
        logger.info(f"  {module}")

    logger.info("\nActive modules for exemplar scan:")
    active = manager.get_active_modules("exemplar")
    for module in active:
        logger.info(f"  - {module.name}")

    logger.info("\nActive modules for candidate scan:")
    active = manager.get_active_modules("candidate")
    for module in active:
        logger.info(f"  - {module.name}")

    logger.info("\nActive modules for free scan:")
    active = manager.get_active_modules("free")
    for module in active:
        logger.info(f"  - {module.name}")


if __name__ == "__main__":
    main()
