#!/usr/bin/env python3
"""
Free Match Mode - Process SQLite files without catalog dependency.

Provides three modes:
1. Free Exemplar Scan - Create exemplars from user-provided SQLite files
2. Free Candidate Scan - Process corrupt SQLites through pipeline without catalog
3. Salvage SQLites - Quick SQLite carving and recovery
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table

from mars.cli.explorer import browse_for_directory, browse_for_file
from mars.config import MARSConfig
from mars.config.paths import ProjectPaths
from mars.pipeline.exemplar_scanner.schema_generator import (
    generate_schema_and_rubric,
    get_schema_fingerprint,
)
from mars.pipeline.raw_scanner.candidate_orchestrator import RawFileProcessor
from mars.utils.cleanup_utilities import cleanup_sqlite_directory
from mars.utils.platform_utils import open_help

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rich.console import Console

    from mars.pipeline.project.manager import MARSProject


class FreeMatchUI:
    """UI for Free Match Mode operations."""

    def __init__(self, console: Console, project: MARSProject | None, show_current_project_menu: Callable):
        """
        Initialize Free Match UI.

        Args:
            console: Rich console instance
            project: Current MARS project (may be None for standalone use)
        """
        self.console = console
        self.project = project
        self.show_current_project_menu = show_current_project_menu

    def show_menu(self) -> None:
        """Display Free Match Mode submenu and handle selections."""
        while True:
            self.show_current_project_menu()
            self._show_banner()

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold sky_blue1", width=4)
            table.add_column(style="bold deep_sky_blue2")
            table.add_column(style="grey69")

            table.add_row(
                "1.",
                "Free Exemplar Scan",
                "Create exemplars from any SQLite files",
            )
            table.add_row(
                "2.",
                "Free Candidate Scan",
                "Process corrupt SQLites (match against free exemplars)",
            )
            table.add_row(
                "3.",
                "Free Salvage",
                "Quick file recovery, no exemplar required",
            )
            table.add_row(None, "", "")
            table.add_row(
                "[bold hot_pink]h[/bold hot_pink]",
                "[bold hot_pink]Help[/bold hot_pink]",
                "",
            )
            table.add_row(None, "[dim](B)ack to Main Menu[/dim]", "")

            panel = Panel(
                table,
                border_style="deep_sky_blue3",
                padding=(1, 1),
            )
            self.console.print(panel)

            choice = (
                Prompt.ask(
                    "\n[bold cyan]Select option[/bold cyan]",
                    choices=["1", "2", "3", "h", "b"],
                    show_default=False,
                )
                .strip()
                .lower()
            )

            if choice == "1":
                if self._free_exemplar_scan():
                    break  # Return to main menu after scan
            elif choice == "2":
                if self._free_candidate_scan():
                    break  # Return to main menu after scan
            elif choice == "3":
                if self._salvage_sqlites():
                    break  # Return to main menu after scan
            elif choice == "h":
                open_help("free-match")
            elif choice == "b":
                break

    def _show_banner(self) -> None:
        """Display a mini banner for Free Match mode."""
        self.console.print(
            Panel(
                "[cyan]Process SQLite files without macOS catalog dependency[/cyan]",
                title="[bold deep_sky_blue2]Free Match Mode[/bold deep_sky_blue2]",
                border_style="dark_goldenrod",
            )
        )

    # =========================================================================
    # ======================== FREE EXEMPLAR SCAN =============================
    # =========================================================================

    def _free_exemplar_scan(self) -> bool:
        """
        Create exemplars from user-provided SQLite files.

        Flow:
        1. User selects SQLite file(s) or folder
        2. Generate schemas and rubrics for each
        3. Store as "free exemplar" in project output

        Returns:
            True if scan was completed, False if cancelled.
        """
        self.show_current_project_menu()
        self._show_banner()

        self.console.print(
            Panel(
                "Create exemplar schemas from your own SQLite files.\n"
                "These can then be used to match and recover data from\n"
                "corrupt versions of the same database types.",
                border_style="deep_sky_blue3",
                style="gray54",
                title_align="left",
                title="[bold deep_sky_blue2]Free Exemplar Scan[/bold deep_sky_blue2]",
                padding=(1, 1),
            )
        )

        # Check for project
        if not self.project:
            self.console.print("\n[red]Error: No project loaded. Free Exemplar Scan requires an active project.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return False

        # Ask user for input type
        self.console.print("\n[bold]Select input type:[/bold]")
        self.console.print("  1. Single SQLite file")
        self.console.print("  2. Folder containing SQLite files")
        self.console.print("  B. Back\n")

        input_choice = (
            Prompt.ask(
                "[cyan]Choice[/cyan]",
                choices=["1", "2", "b"],
                show_default=False,
            )
            .strip()
            .lower()
        )

        if input_choice == "b":
            return False

        self.show_current_project_menu()
        self._show_banner()

        # Browse for input
        sqlite_files: list[Path] = []

        if input_choice == "1":
            # Single file
            selected = browse_for_file(
                None,  # Use OS-appropriate default
                file_filter=[".sqlite", ".db", ".sqlite3"],
                title="Select SQLite File",
            )
            if selected and selected.is_file():
                sqlite_files = [selected]
        else:
            # Folder
            selected = browse_for_directory(
                None,  # Use OS-appropriate default
                title="Select Folder with SQLite Files",
            )
            if selected and selected.is_dir():
                # Find all SQLite files in folder (recursive)
                for ext in [".sqlite", ".db", ".sqlite3"]:
                    sqlite_files.extend(selected.rglob(f"*{ext}"))

        if not sqlite_files:
            self.console.print("\n[yellow]No SQLite files selected or found.[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return False

        # Validate SQLite files
        valid_files = self._validate_sqlite_files(sqlite_files)

        if not valid_files:
            self.console.print("\n[red]No valid SQLite files found in selection.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return False

        # Show what will be processed
        self.console.print(f"\n[green]Found {len(valid_files)} valid SQLite file(s):[/green]")
        for f in valid_files[:10]:  # Show first 10
            self.console.print(f"  • {f.name}")
        if len(valid_files) > 10:
            self.console.print(f"  [dim]... and {len(valid_files) - 10} more[/dim]")

        # Optional description
        description = Prompt.ask(
            "\n[cyan]Enter optional description (or press Enter to skip)[/cyan]",
            default="",
        ).strip()

        # Confirm
        if not Confirm.ask("\n[cyan]Proceed with exemplar generation?[/cyan]"):
            return False

        # Create output directory structure matching standard exemplar scans
        # Format: MARS_{project_name}_free_{timestamp}
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        project_name = self.project.config.get("project_name", "Unknown")
        # Sanitize project name for filesystem
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in project_name)
        root_folder = f"MARS_{safe_name}_free_{timestamp}"
        output_root = self.project.project_dir / "output" / root_folder

        # Create standard directory structure
        catalog_dir = output_root / "exemplar" / "databases" / "catalog"
        schemas_dir = output_root / "exemplar" / "databases" / "schemas"
        candidates_dir = output_root / "candidates"
        reports_dir = output_root / "reports"

        catalog_dir.mkdir(parents=True, exist_ok=True)
        schemas_dir.mkdir(parents=True, exist_ok=True)
        candidates_dir.mkdir(parents=True, exist_ok=True)
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Process each file
        results: list[dict] = []
        errors: list[tuple[Path, str]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing SQLite files...", total=len(valid_files))

            for db_path in valid_files:
                progress.update(task, description="Processing database...")

                try:
                    result = self._process_single_exemplar(db_path, catalog_dir, schemas_dir)
                    results.append(result)
                except Exception as e:
                    errors.append((db_path, str(e)))

                progress.advance(task)

        # Create manifest
        manifest = {
            "created": datetime.now(UTC).isoformat(),
            "description": description or None,
            "source_type": "free_exemplar",
            "total_files": len(valid_files),
            "successful": len(results),
            "failed": len(errors),
            "exemplars": results,
            "errors": [{"file": str(p), "error": e} for p, e in errors],
        }

        manifest_path = output_root / "_manifest.json"
        with manifest_path.open("w") as f:
            json.dump(manifest, f, indent=2)

        # Show summary
        self.show_current_project_menu()
        self.console.print(
            Panel(
                f"[bold dark_sea_green4]Free Exemplar Scan Complete[/bold dark_sea_green4]\n\n"
                f"Processed: {len(valid_files)} file(s)\n"
                f"Successful: [green]{len(results)}[/green]\n"
                f"Failed: [red]{len(errors)}[/red]\n\n"
                f"Output: {output_root}",
                border_style="green",
            )
        )

        if errors:
            self.console.print("\n[red]Errors:[/red]")
            for path, error in errors[:5]:
                self.console.print(f"  • {path.name}: {error}")
            if len(errors) > 5:
                self.console.print(f"  [dim]... and {len(errors) - 5} more[/dim]")

        # Generate HTML report
        from mars.cli.scan_report_generator import ScanReportGenerator

        report_gen = ScanReportGenerator(
            project_root=self.project.project_dir,
            project_name=self.project.config.get("project_name", "Unknown"),
        )
        report_gen.generate_free_report(
            scan_type="free_exemplar",
            stats={
                "total_files": len(valid_files),
                "successful": len(results),
                "failed": len(errors),
            },
            output_dir=output_root,
            description=description or "",
            console=self.console,
        )

        Prompt.ask("\nPress Enter to continue")
        return True

    def _validate_sqlite_files(self, files: list[Path]) -> list[Path]:
        """Validate that files are actual SQLite databases."""
        valid = []
        for f in files:
            if not f.is_file():
                continue
            # Check SQLite magic bytes
            try:
                with f.open("rb") as fp:
                    header = fp.read(16)
                    if header[:16] == b"SQLite format 3\x00":
                        valid.append(f)
            except (OSError, PermissionError):
                continue
        return valid

    def _process_single_exemplar(self, db_path: Path, catalog_dir: Path, schemas_dir: Path) -> dict:
        """
        Process a single SQLite file into a free exemplar.

        Args:
            db_path: Path to source SQLite file
            catalog_dir: Directory for database files (catalog/{db_name}/)
            schemas_dir: Directory for schema files (schemas/{db_name}/)

        Returns:
            Dict with exemplar metadata
        """
        db_name = db_path.stem

        # Strip common suffixes that shouldn't be part of the exemplar name
        # (e.g., "Chrome Cookies_admin_v1.combined" -> "Chrome Cookies_admin_v1")
        for suffix in [".combined", "_combined"]:
            if db_name.endswith(suffix):
                db_name = db_name[: -len(suffix)]
                break

        # Create output directories for this exemplar
        db_catalog_dir = catalog_dir / db_name
        db_schema_dir = schemas_dir / db_name
        db_catalog_dir.mkdir(parents=True, exist_ok=True)
        db_schema_dir.mkdir(parents=True, exist_ok=True)

        # Copy the database to catalog
        dest_db = db_catalog_dir / db_path.name
        shutil.copy2(db_path, dest_db)

        # Get config values for rubric generation
        config = MARSConfig()
        min_year = int(config.exemplar.epoch_min[:4])
        max_year = int(config.exemplar.epoch_max[:4])

        # Generate schema and rubric in schemas directory
        schema_path, rubric_path = generate_schema_and_rubric(
            dest_db,
            db_schema_dir,
            db_name,
            min_timestamp_rows=config.exemplar.min_timestamp_rows,
            min_role_sample_size=config.exemplar.min_role_sample_size,
            min_year=min_year,
            max_year=max_year,
        )

        # Get fingerprint
        fingerprint = get_schema_fingerprint(dest_db)

        # Get basic stats
        table_count = 0
        try:
            with sqlite3.connect(f"file:{dest_db}?mode=ro", uri=True) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                table_count = cur.fetchone()[0]
        except sqlite3.Error:
            pass

        return {
            "name": db_name,
            "source_path": str(db_path),
            "catalog_dir": str(db_catalog_dir),
            "schema_dir": str(db_schema_dir),
            "schema_path": str(schema_path) if schema_path else None,
            "rubric_path": str(rubric_path) if rubric_path else None,
            "fingerprint": fingerprint,
            "table_count": table_count,
        }

    # =========================================================================
    # ======================== FREE CANDIDATE SCAN ============================
    # =========================================================================

    def _find_free_exemplar_scans(self) -> list[dict]:
        """
        Find all Free Exemplar Scans in the current project.

        Returns:
            List of dicts with scan metadata:
                - path: Path to the scan root directory
                - name: Folder name
                - created: Creation timestamp
                - description: Optional description
                - exemplar_count: Number of exemplars
        """
        if not self.project:
            return []

        scans = []
        output_dir = self.project.project_dir / "output"

        if not output_dir.exists():
            return []

        # Look for _free_ folders with valid manifests
        for folder in output_dir.iterdir():
            if not folder.is_dir() or "_free_" not in folder.name:
                continue

            manifest_path = folder / "_manifest.json"
            if not manifest_path.exists():
                continue

            try:
                with manifest_path.open() as f:
                    manifest = json.load(f)

                # Only include free_exemplar type scans
                if manifest.get("source_type") != "free_exemplar":
                    continue

                # Check that exemplar directories exist
                exemplar_db_dir = folder / "exemplar" / "databases"
                if not exemplar_db_dir.exists():
                    continue

                scans.append(
                    {
                        "path": folder,
                        "name": folder.name,
                        "created": manifest.get("created", "Unknown"),
                        "description": manifest.get("description") or "No description",
                        "exemplar_count": manifest.get("successful", 0),
                    }
                )
            except (json.JSONDecodeError, OSError):
                continue

        # Sort by creation date (newest first)
        scans.sort(key=lambda x: x["created"], reverse=True)
        return scans

    def _free_candidate_scan(self) -> bool:
        """
        Process corrupt SQLites through the pipeline using free exemplars.

        Flow:
        1. User selects a Free Exemplar Scan to use for matching
        2. User selects corrupt SQLite file(s) or folder
        3. Run through full recovery pipeline with free exemplars
        4. Output recovered data with matches to free exemplar schemas

        Returns:
            True if scan was completed, False if cancelled.
        """
        self.show_current_project_menu()
        self._show_banner()

        self.console.print(
            Panel(
                "Process corrupt SQLite files using your own exemplars.\n"
                "Matches against schemas from a previous Free Exemplar Scan\n"
                "instead of the built-in macOS database catalog.",
                border_style="deep_sky_blue3",
                title="[bold deep_sky_blue2]Free Candidate Scan[/bold deep_sky_blue2]",
                title_align="left",
                style="grey54",
                padding=(1, 1),
            )
        )

        # Check for project
        if not self.project:
            self.console.print("\n[red]Error: No project loaded. Free Candidate Scan requires an active project.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return False

        # Find available Free Exemplar Scans
        free_scans = self._find_free_exemplar_scans()

        if not free_scans:
            self.console.print(
                "\n[yellow]No Free Exemplar Scans found in this project.[/yellow]\n"
                "[dim]Run 'Free Exemplar Scan' first to create exemplars from your\n"
                "known-good SQLite files, then use this feature to match against them.[/dim]"
            )
            Prompt.ask("\nPress Enter to continue")
            return False

        # Show available free exemplar scans
        self.console.print("\n[bold]Available Free Exemplar Scans:[/bold]\n")

        table = Table(show_header=True, box=None, padding=(0, 2))
        table.add_column("#", style="bold cyan", width=3)
        table.add_column("Name", style="bold")
        table.add_column("Exemplars", style="green", justify="right")
        table.add_column("Description", style="dim")
        table.add_column("Created", style="dim")

        for idx, scan in enumerate(free_scans, 1):
            # Parse and format date
            created = scan["created"]
            if "T" in created:
                created = created.split("T")[0]

            table.add_row(
                str(idx),
                scan["name"].split("_free_")[0].replace("MARS_", ""),
                str(scan["exemplar_count"]),
                (scan["description"][:30] + "..." if len(scan["description"]) > 30 else scan["description"]),
                created,
            )

        self.console.print(table)
        self.console.print()

        # Let user select a scan
        choices = [str(i) for i in range(1, len(free_scans) + 1)] + ["b"]
        scan_choice = (
            Prompt.ask(
                "[cyan]Select exemplar scan (or B to go back)[/cyan]",
                choices=choices,
                show_default=False,
            )
            .strip()
            .lower()
        )

        if scan_choice == "b":
            return False

        selected_scan = free_scans[int(scan_choice) - 1]
        exemplar_db_dir = selected_scan["path"] / "exemplar" / "databases"

        self.console.print(
            f"\n[green]Using exemplars from:[/green] {selected_scan['name']}"
            f"\n  ({selected_scan['exemplar_count']} exemplar(s))\n"
        )

        # Ask user for input type
        self.console.print("[bold]Select input type:[/bold]")
        self.console.print("  1. Single SQLite file")
        self.console.print("  2. Folder containing files (SQLites, archives, etc.)")
        self.console.print("  B. Back\n")

        input_choice = (
            Prompt.ask(
                "[cyan]Choice[/cyan]",
                choices=["1", "2", "b"],
                show_default=False,
            )
            .strip()
            .lower()
        )

        if input_choice == "b":
            return False

        # Browse for input
        input_path: Path | None = None

        if input_choice == "1":
            # Single file
            selected = browse_for_file(
                None,  # Use OS-appropriate default
                file_filter=[".sqlite", ".db", ".sqlite3", ".gz", ".bz2", ".zip", ".plist", ".log"],
                title="Select SQLite File (or archive)",
            )
            if selected and selected.is_file():
                input_path = selected
        else:
            # Folder
            selected = browse_for_directory(
                None,  # Use OS-appropriate default
                title="Select Folder with Files to Process",
            )
            if selected and selected.is_dir():
                input_path = selected

        if not input_path:
            self.console.print("\n[yellow]No files selected.[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return False

        self.show_current_project_menu()
        self._show_banner()

        # Show what will be processed
        if input_path.is_file():
            size_kb = input_path.stat().st_size / 1024
            self.console.print(f"\n[green]Selected file:[/green] {input_path.name} ({size_kb:.1f} KB)")
        else:
            self.console.print(f"\n[green]Selected folder:[/green] {input_path}")

        # Optional description
        description = Prompt.ask(
            "\n[cyan]Enter optional description (or press Enter to skip)[/cyan]",
            default="",
        ).strip()

        # Confirm
        if not Confirm.ask("\n[cyan]Proceed with free candidate scan?[/cyan]"):
            return False

        # Create output directory structure INSIDE the selected Free Exemplar Scan
        # This mirrors how standard candidate scans go inside standard exemplar scans
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_root = selected_scan["path"]  # Use the Free Exemplar Scan's folder

        # Create staging directory if single file was selected
        if input_path.is_file():
            staging_dir = output_root / "_staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_path, staging_dir / input_path.name)
            input_dir = staging_dir
        else:
            input_dir = input_path

        # Create candidates directory for this run inside the Free Exemplar folder
        candidates_root = output_root / "candidates"
        candidates_root.mkdir(parents=True, exist_ok=True)

        # Create ProjectPaths for the candidate run
        paths = ProjectPaths.create_candidate_run(
            candidates_root=candidates_root,
            run_name=timestamp,
        )
        paths.create_candidate_dirs()

        # Create config with dissect_all enabled (for databases without exact matches)
        config = MARSConfig()
        config.variant_selector.dissect_all = True

        # Run the full pipeline with free exemplars
        self.show_current_project_menu()
        self.console.print("\n[bold cyan]Starting free candidate scan...[/bold cyan]\n")

        try:
            processor = RawFileProcessor(
                input_dir=input_dir,
                config=config,
                paths=paths,
                exemplar_db_dir=exemplar_db_dir,
            )

            # Run all processing steps
            processor.process_all(richConsole=self.console, free_scan=True)

            # Get stats from processor
            stats = processor.stats

            # Create manifest
            manifest = {
                "created": datetime.now(UTC).isoformat(),
                "description": description or None,
                "source_type": "free_candidate",
                "input_path": str(input_path),
                "exemplar_scan": str(selected_scan["path"]),
                "exemplar_count": selected_scan["exemplar_count"],
                "stats": {
                    "total_files": stats.get("total_files", 0),
                    "sqlite_dbs_found": stats.get("sqlite_dbs_found", 0),
                    "sqlite_dbs_matched": stats.get("sqlite_dbs_matched", 0),
                    "sqlite_dbs_matched_nonempty": stats.get("sqlite_dbs_matched_nonempty", 0),
                    "sqlite_dbs_unmatched": stats.get("sqlite_dbs_unmatched", 0),
                    "metamatch_unique_schemas": stats.get("metamatch_unique_schemas", 0),
                    "metamatch_multi_member": stats.get("metamatch_multi_member", 0),
                    "carved_databases": stats.get("carved_databases", 0),
                    "carved_failed": stats.get("carved_failed", 0),
                },
            }

            # Save manifest in the candidate run folder (not the exemplar root)
            manifest_path = paths.root / "_manifest.json"
            with manifest_path.open("w") as f:
                json.dump(manifest, f, indent=2)

            # Clean up staging directory if created
            staging_dir = output_root / "_staging"
            if staging_dir.exists():
                with contextlib.suppress(Exception):
                    cleanup_sqlite_directory(staging_dir)

            # Show summary
            self.show_current_project_menu()
            self.console.print(
                Panel(
                    f"[bold dark_sea_green4]Free Candidate Scan Complete[/bold dark_sea_green4]\n\n"
                    f"Files scanned: {stats.get('total_files', 0)}\n"
                    f"SQLite DBs found: {stats.get('sqlite_dbs_found', 0)}\n"
                    f"Matched to exemplars: [green]{stats.get('sqlite_dbs_matched', 0)}[/green]\n"
                    f"  (with data: {stats.get('sqlite_dbs_matched_nonempty', 0)})\n"
                    f"Unmatched: {stats.get('sqlite_dbs_unmatched', 0)}\n"
                    f"Metamatch groups: {stats.get('metamatch_unique_schemas', 0)}\n"
                    f"Carved DBs: {stats.get('carved_databases', 0)}\n\n"
                    f"Output: {paths.root}",
                    border_style="green",
                )
            )

            # Generate HTML report
            from mars.cli.scan_report_generator import ScanReportGenerator

            report_gen = ScanReportGenerator(
                project_root=self.project.project_dir,
                project_name=self.project.config.get("project_name", "Unknown"),
            )
            report_gen.generate_free_report(
                scan_type="free_candidate",
                stats=stats,
                output_dir=output_root,
                description=description or "",
                console=self.console,
            )

        except Exception as e:
            self.console.print(f"\n[red]Error during processing: {e}[/red]")
            import traceback

            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

        Prompt.ask("\nPress Enter to continue")
        return True

    # =========================================================================
    # ========================== FREE SALVAGE =================================
    # =========================================================================

    def _salvage_sqlites(self) -> bool:
        """
        Full pipeline SQLite recovery without catalog matching.

        Flow:
        1. User selects SQLite file(s) or folder
        2. Run full recovery pipeline:
           - File categorization
           - Variant selection (find best O/C/R/D)
           - Metamatch grouping (schema-based)
           - Carving for failed databases
           - Will also process WiFi plists, text logs, jsonlz4, etc.
        3. Output recovered data with standard structure

        Returns:
            True if scan was completed, False if cancelled.
        """
        self.show_current_project_menu()
        self._show_banner()

        self.console.print(
            Panel(
                "Full pipeline recovery for corrupt SQLite files.\n"
                "Runs variant selection, metamatch grouping, and "
                "forensic carving - without catalog matching.\n"
                "Can also process archives containing SQLite, plists, logs, etc.",
                title="[bold deep_sky_blue2]Free Salvage[/bold deep_sky_blue2]",
                title_align="left",
                border_style="deep_sky_blue3",
                style="grey54",
                padding=(1, 1),
            )
        )

        # Check for project
        if not self.project:
            self.console.print("\n[red]Error: No project loaded. Free Salvage requires an active project.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return False

        # Ask user for input type
        self.console.print("\n[bold]Select input type:[/bold]")
        self.console.print("  1. Single file")
        self.console.print("  2. Folder containing files (SQLites, plists, archives, etc.)")
        self.console.print("  B. Back\n")

        input_choice = (
            Prompt.ask(
                "[cyan]Choice[/cyan]",
                choices=["1", "2", "b"],
                show_default=False,
            )
            .strip()
            .lower()
        )

        if input_choice == "b":
            return False

        self.show_current_project_menu()
        self._show_banner()

        # Browse for input
        input_path: Path | None = None

        if input_choice == "1":
            # Single file - we'll create a staging directory for it
            selected = browse_for_file(
                None,  # Use OS-appropriate default
                file_filter=[".sqlite", ".db", ".sqlite3", ".gz", ".bz2", ".zip", ".plist", ".log"],
                title="Select SQLite File (or archive)",
            )
            if selected and selected.is_file():
                input_path = selected
        else:
            # Folder - use directly as input
            selected = browse_for_directory(
                None,  # Use OS-appropriate default
                title="Select Folder with Files to Process",
            )
            if selected and selected.is_dir():
                input_path = selected

        if not input_path:
            self.console.print("\n[yellow]No files selected.[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return False

        # Show what will be processed
        if input_path.is_file():
            size_kb = input_path.stat().st_size / 1024
            self.console.print(f"\n[green]Selected file:[/green] {input_path.name} ({size_kb:.1f} KB)")
        else:
            self.console.print(f"\n[green]Selected folder:[/green] {input_path}")

        # Optional description
        description = Prompt.ask(
            "\n[cyan]Enter optional description (or press Enter to skip)[/cyan]",
            default="",
        ).strip()

        # Confirm
        if not Confirm.ask("\n[cyan]Proceed with salvage operation?[/cyan]", default=True):
            return False

        # Create output directory structure
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        project_name = self.project.config.get("project_name", "Unknown")
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in project_name)
        root_folder = f"MARS_{safe_name}_salvage_{timestamp}"
        output_root = self.project.project_dir / "output" / root_folder

        # Create empty exemplar directories (processor requires them but nothing will match)
        exemplar_db_dir = output_root / "exemplar" / "databases"
        exemplar_catalog = exemplar_db_dir / "catalog"
        exemplar_schemas = exemplar_db_dir / "schemas"
        exemplar_catalog.mkdir(parents=True, exist_ok=True)
        exemplar_schemas.mkdir(parents=True, exist_ok=True)

        # Create staging directory if single file was selected
        if input_path.is_file():
            staging_dir = output_root / "_staging"
            staging_dir.mkdir(parents=True, exist_ok=True)
            # Copy file to staging
            shutil.copy2(input_path, staging_dir / input_path.name)
            input_dir = staging_dir
        else:
            input_dir = input_path

        # Create candidates directory for this run
        candidates_root = output_root / "candidates"
        candidates_root.mkdir(parents=True, exist_ok=True)

        # Create ProjectPaths for the candidate run
        paths = ProjectPaths.create_candidate_run(
            candidates_root=candidates_root,
            run_name=timestamp,
        )
        paths.create_candidate_dirs()

        # Create config with dissect_all enabled (no catalog means no matches,
        # so we want to attempt sqlite_dissect on all recovered databases)
        config = MARSConfig()
        config.variant_selector.dissect_all = True

        # Run the full pipeline
        self.show_current_project_menu()

        try:
            processor = RawFileProcessor(
                input_dir=input_dir,
                config=config,
                paths=paths,
                exemplar_db_dir=exemplar_db_dir,
            )

            # Run all processing steps
            processor.process_all(richConsole=self.console, free_scan=True)

            # Get stats from processor
            stats = processor.stats

            # Create manifest
            manifest = {
                "created": datetime.now(UTC).isoformat(),
                "description": description or None,
                "source_type": "salvage",
                "input_path": str(input_path),
                "stats": {
                    "total_files": stats.get("total_files", 0),
                    "sqlite_dbs_found": stats.get("sqlite_dbs_found", 0),
                    "sqlite_dbs_matched": stats.get("sqlite_dbs_matched", 0),
                    "sqlite_dbs_unmatched": stats.get("sqlite_dbs_unmatched", 0),
                    "metamatch_unique_schemas": stats.get("metamatch_unique_schemas", 0),
                    "metamatch_multi_member": stats.get("metamatch_multi_member", 0),
                    "carved_databases": stats.get("carved_databases", 0),
                    "carved_failed": stats.get("carved_failed", 0),
                    "wifi_plists_kept:": stats.get("wifi_plists_kept", 0),
                    "text_logs:": stats.get("text_logs", 0),
                    "jsonlz4_kept:": stats.get("jsonlz4_kept", 0),
                },
            }

            manifest_path = output_root / "_manifest.json"
            with manifest_path.open("w") as f:
                json.dump(manifest, f, indent=2)

            # Clean up staging directory if created
            staging_dir = output_root / "_staging"
            if staging_dir.exists():
                with contextlib.suppress(Exception):
                    cleanup_sqlite_directory(staging_dir)

            # Show summary
            self.show_current_project_menu()
            self.console.print(
                Panel(
                    f"[bold dark_sea_green4]Salvage Operation Complete[/bold dark_sea_green4]\n\n"
                    f"Files scanned: {stats.get('total_files', 0)}\n"
                    f"SQLite DBs found: {stats.get('sqlite_dbs_found', 0)}\n"
                    f"Metamatch groups: {stats.get('metamatch_unique_schemas', 0)}\n"
                    f"Carved DBs: {stats.get('carved_databases', 0)}\n"
                    f"WiFi Plists: {stats.get('wifi_plists_kept', 0)}\n"
                    f"Text Logs: {stats.get('text_logs', 0)}\n"
                    f"JSONLZ4 Files: {stats.get('jsonlz4_kept', 0)}\n\n"
                    f"Output: {output_root}",
                    border_style="green",
                )
            )

            # Generate HTML report
            from mars.cli.scan_report_generator import ScanReportGenerator

            report_gen = ScanReportGenerator(
                project_root=self.project.project_dir,
                project_name=self.project.config.get("project_name", "Unknown"),
            )
            report_gen.generate_free_report(
                scan_type="salvage",
                stats=stats,
                output_dir=output_root,
                description=description or "",
                console=self.console,
            )

        except Exception as e:
            self.console.print(f"\n[red]Error during processing: {e}[/red]")
            import traceback

            self.console.print(f"[dim]{traceback.format_exc()}[/dim]")

        Prompt.ask("\nPress Enter to continue")
        return True
