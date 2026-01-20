#!/usr/bin/env python3
"""
MARS - Main CLI Interface
by WarpedWing Labs.
"""

from __future__ import annotations

import argparse
import atexit
import contextlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from time import sleep

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

from mars.cli.comparison_ui import ComparisonUI
from mars.cli.ewf_mount_ui import EWFMountUI
from mars.cli.exemplar_packager import ExemplarImporter
from mars.cli.exemplar_scan_ui import ExemplarScanUI
from mars.cli.explorer import browse_for_directory, browse_for_file
from mars.cli.export_ui import ExportUI
from mars.cli.free_match import FreeMatchUI
from mars.cli.plotter_ui import PlotterUI
from mars.cli.settings_ui import SettingsUI
from mars.cli.target_scan_ui import TargetScanUI
from mars.cli.time_machine_ui import TimeMachineScanUI
from mars.config import ConfigLoader
from mars.pipeline.project.manager import MARSProject
from mars.utils.compression_utils import (
    get_archive_extension,
    is_archive,
)
from mars.utils.debug_logger import logger
from mars.utils.platform_utils import is_admin, is_windows, open_help

# Rich Colors
DSB1 = "deep_sky_blue1"
DSB2 = "deep_sky_blue2"
DSB3 = "deep_sky_blue3"
BDSB1 = "bold deep_sky_blue1"
BDSB2 = "bold deep_sky_blue2"
BDSB3 = "bold deep_sky_blue3"


def _get_mars_sudo_path() -> Path | None:
    """Find the mars-sudo wrapper script for privilege elevation.

    Searches common locations for the wrapper script that handles
    sudo invocation with proper uv cache isolation.

    Returns:
        Path to mars-sudo if found and executable, None otherwise
    """
    # First try relative to source tree (development) - BEFORE checking PATH
    # This ensures we use the same version we're running from, not an old install
    src_root = Path(__file__).parent.parent.parent.parent  # cli/app.py -> project root
    dev_path = src_root / "bin" / "mars-sudo"
    if dev_path.exists() and os.access(dev_path, os.X_OK):
        return dev_path

    # Then check if mars-sudo is in PATH (installed MARS)
    which_result = shutil.which("mars-sudo")
    if which_result:
        return Path(which_result)

    # Try other common locations
    candidates = [
        Path("/usr/local/bin/mars-sudo"),
        Path.home() / ".local" / "bin" / "mars-sudo",
    ]
    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return path
    return None


def _cleanup_uv_cache_permissions() -> None:
    """Restore uv cache ownership to original user if running via sudo.

    When running `sudo uv run mars`, uv creates cache files owned by root
    in the original user's ~/.cache/uv. This cleanup restores ownership
    so the user can manage the cache after the scan completes.
    """
    sudo_user = os.environ.get("SUDO_USER")
    if not sudo_user or os.geteuid() != 0:
        return

    uv_cache = Path.home() / ".cache" / "uv"
    if uv_cache.exists():
        with contextlib.suppress(Exception):
            subprocess.run(
                ["chown", "-R", f"{sudo_user}:staff", str(uv_cache)],
                check=False,
                capture_output=True,
            )


# Register cleanup to run on exit (safety net for direct sudo usage)
atexit.register(_cleanup_uv_cache_permissions)


class MARSCLI:
    """Main CLI application."""

    def __init__(self):
        """Initialize CLI."""
        self.console = Console()
        self.project: MARSProject | None = None

        # Configure global debug logger
        # Load config to get debug flag (without project context)
        config = ConfigLoader.load()
        logger.configure(config, console=self.console)

    def run(self):
        """Run the main CLI loop."""
        self.show_banner()

        while True:
            if self.project is None:
                # No project loaded - show project menu
                action = self.show_no_project_menu()
                if action == "quit":
                    break
            else:
                # Project loaded - show main menu
                action = self.show_current_project_menu()
                action_2 = self.show_main_menu()
                if action or action_2 == "quit":
                    break

        self.console.clear()

    def run_direct_exemplar_scan(self, source_path: Path) -> None:
        """Run exemplar scan directly without menu navigation.

        Used after sudo elevation to skip directly to the scan confirmation
        page, preserving the user's navigation progress.

        Args:
            source_path: Path to scan (typically "/" for live system scans)
        """

        # Validate source path
        if not source_path.exists():
            self.console.print(f"[bold red]Error:[/bold red] Path does not exist: {source_path}")
            sys.exit(1)

        # Load last-used project from config
        last_project = self._get_last_project()
        if not last_project:
            self.console.print("[bold red]Error:[/bold red] No recent project found.")
            self.console.print("Please run MARS normally first to create or open a project.")
            sys.exit(1)

        if not self._open_last_project(last_project):
            self.console.print(f"[bold red]Error:[/bold red] Could not open project: {last_project}")
            sys.exit(1)

        # Jump directly to ExemplarScanUI
        # Use is_image=True to leverage dfVFS for proper file enumeration (matches menu-driven flow)
        self.show_current_project_menu()
        scanner_ui = ExemplarScanUI(self.console, self.project)  # type: ignore[arg-type]
        scanner_ui.run_scan(source_path, is_image=True, show_header_callback=self.show_current_project_menu)

        # After scan completes, continue with normal menu flow
        self.run()

    def run_direct_tm_scan(self, tm_volume: Path) -> None:
        """Run Time Machine scan directly without menu navigation.

        Used after sudo elevation to skip directly to the TM backup selection
        and scan, preserving the user's navigation progress.

        Args:
            tm_volume: Path to Time Machine volume root
        """
        from mars.utils.time_machine_utils import find_time_machine_volume, parse_backup_manifest

        # Validate TM volume
        validated_volume = find_time_machine_volume(tm_volume)
        if validated_volume is None:
            self.console.print(f"[bold red]Error:[/bold red] Not a valid Time Machine volume: {tm_volume}")
            sys.exit(1)

        # Load last-used project from config
        last_project = self._get_last_project()
        if not last_project:
            self.console.print("[bold red]Error:[/bold red] No recent project found.")
            self.console.print("Please run MARS normally first to create or open a project.")
            sys.exit(1)

        if not self._open_last_project(last_project):
            self.console.print(f"[bold red]Error:[/bold red] Could not open project: {last_project}")
            sys.exit(1)

        # Show admin status
        self.show_current_project_menu()
        if is_admin():
            self.console.print("[bold dark_sea_green4][✓] Running with administrator privileges[/bold dark_sea_green4]")
            sleep(0.7)

        # Select exemplar scan to use for matching
        exemplar_scan = self._select_exemplar_scan()
        if exemplar_scan is None:
            self.console.print("[yellow]No exemplar scan selected.[/yellow]")
            sleep(0.7)
            self.run()
            return

        # Parse manifest and get available backups
        try:
            backups = parse_backup_manifest(validated_volume / "backup_manifest.plist")
        except Exception as e:
            self.console.print(f"[bold red]Error reading backup manifest:[/bold red] {e}")
            sleep(1)
            self.run()
            return

        if not backups:
            self.console.print("[yellow]No backups found in the selected volume.[/yellow]")
            sleep(1)
            self.run()
            return

        # Initialize Time Machine UI and continue with backup selection + scan
        tm_ui = TimeMachineScanUI(self.console, self.project, lambda: self.show_current_project_menu())  # type: ignore[arg-type]

        # Select backups to scan
        selected_backups = tm_ui.select_backups(backups)
        if not selected_backups:
            self.run()
            return

        # Run the scan
        tm_ui.run_scan(
            tm_volume=validated_volume,
            selected_backups=selected_backups,
            exemplar_scan=exemplar_scan,
            show_header_callback=self.show_current_project_menu,
        )

        # After scan completes, continue with normal menu flow
        self.run()

    def show_banner(self):
        """Display application banner."""

        banner = r"""
[bold red3]╔═════════════════════════════════════╗
║            [red3]   ▛▛▌▀▌▛▘▛▘[/red3]             ║
║            [red3]▄▄▖▌▌▌█▌▌ ▄▌[/red3]             ║
║  ─────────────────────────────────  ║
║    [navajo_white1]macOS Artifact Recovery Suite[/navajo_white1]    ║
║                                     ║
║           [grey42]WarpedWing Labs[/grey42]           ║
║                [grey23]v1.1.0[/grey23]               ║
╚═════════════════════════════════════╝[/bold red3]
"""
        self.console.print(banner)

    def show_no_project_menu(self) -> str:
        """
        Show project creation/loading menu.

        Returns:
            Action taken: 'loaded', 'created', 'quit'
        """
        self.console.clear()
        self.show_banner()

        # Check for last project
        last_project = self._get_last_project()

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold sky_blue2", width=4)
        table.add_column()

        table.add_row("1.", f"[{BDSB1}]New[/{BDSB1}] project")
        table.add_row("2.", f"[{BDSB1}]Open[/{BDSB1}] existing project")

        # Track next menu number
        next_num = 3

        if last_project:
            table.add_row(
                f"{next_num}.",
                f"[{BDSB1}]Last[/{BDSB1}] project: [light_goldenrod2]{last_project.name.removesuffix('.marsproj')}[/light_goldenrod2]",
            )
            next_num += 1

        # Utilities menu only available on macOS (EWF mounting requires hdiutil)
        utilities_num = None
        if not is_windows():
            utilities_num = next_num
            table.add_row("", "")
            table.add_row(
                f"{next_num}.",
                "[bold light_goldenrod2]Utilities[/bold light_goldenrod2]",
            )

        table.add_row(None, "")
        table.add_row(
            "[bold hot_pink]h[/bold hot_pink]",
            "[bold hot_pink]Help[/bold hot_pink]",
        )

        table.add_row("[red bold]q[/red bold]", "[red bold]Quit[/red bold]")

        panel = Panel(
            table,
            title=f"[{BDSB1}]Load Project[/{BDSB1}]",
            border_style=f"{DSB3}",
        )
        self.console.print(panel)

        # Build choices list
        choices = ["1", "2"]
        if last_project:
            choices.append("3")
        if utilities_num:
            choices.append(str(utilities_num))
        choices.extend(["h", "q"])

        choice = Prompt.ask(
            "\n[bold cyan]Select option[/bold cyan]",
            choices=choices,
            show_default=False,
        ).lower()

        if choice == "1":
            if self._create_project():
                self._save_last_project()
                return "created"
        elif choice == "2":
            if self._open_project():
                self._save_last_project()
                return "loaded"
        elif choice == "3" and last_project:
            if self._open_last_project(last_project):
                return "loaded"
        elif utilities_num and choice == str(utilities_num):
            self._utilities_menu()
        elif choice == "h":
            open_help()
        elif choice == "q":
            return "quit"

        return "continue"

    def show_current_project_menu(self, show_banner: bool | None = False) -> str | None:
        """
        Show main application menu.

        Returns:
            Action taken: 'continue', 'quit'
        """
        self.console.clear()
        if show_banner:
            self.show_banner()

        # Case/Examiner/Exemplar Status table
        if self.project:
            # Gather project info
            project = ""
            examiner = ""
            case_number = ""
            description = ""

            def _create_header_text(descriptor: str, value: str) -> Text:
                text = Text(overflow="ellipsis", no_wrap=True)
                text.append(f"{descriptor}: ", style=f"{BDSB3}")
                text.append(value, style="grey69")
                return text

            project = self.project.config["project_name"]
            if self.project.config.get("examiner_name"):
                examiner = self.project.config["examiner_name"]
            if self.project.config.get("case_number"):
                case_number = self.project.config["case_number"]
            if self.project.config.get("description"):
                description = self.project.config["description"]

            project_text = _create_header_text("Project", project)
            description_text = _create_header_text("Description", description)
            examiner_text = _create_header_text("Examiner", examiner)
            case_text = _create_header_text("Case #", case_number)

            project_description_table = Table.grid(expand=True, collapse_padding=True, pad_edge=False, padding=(0, 0))
            project_description_table.add_column(ratio=1)
            project_description_table.add_column(ratio=2, justify="right")
            project_description_table.add_row(project_text, description_text)
            project_description_table.add_row(examiner_text, case_text)

            logo = r"""
   ▛▛▌
▄▄▖▌▌▌
"""
            logo_text = Text(logo, style="red3", no_wrap=True)
            title_text = Text("MARS: macOS Artifact Recovery Suite")
            title_text.stylize_before("bold red3", 0, 4)
            title_text.stylize_before("bold navajo_white1", 5)

            # pseudo flexbox
            header_grid = Table.grid(expand=True, collapse_padding=True, pad_edge=False, padding=(0, 0))
            header_grid.add_column(width=9)
            header_grid.add_column(ratio=1, vertical="middle")
            header_grid.add_row(logo_text, project_description_table)

            self.console.print(
                Panel(
                    header_grid,
                    title=title_text,
                    border_style="red3",
                )
            )
            self.console.print()

    # ==========================================================================
    # ============================ PROJECT MENU ================================
    # ==========================================================================

    def show_main_menu(self) -> str:
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold sky_blue2", width=4)
        table.add_column(style=f"{BDSB1}")
        table.add_column(style="grey69")

        table.add_row(
            "1.",
            "Exemplar Scan",
            "Extract target catalog files from a reference system",
        )
        # Check if exemplar analysis is complete
        exemplar_complete = self._is_exemplar_complete()

        # Candidates Scan - always available (gating moved to subpage)
        table.add_row(
            "2.",
            "Candidates Scan",
            "Classify files against exemplar scan and recover lost data",
        )

        # Free Match Mode - always available
        table.add_row(
            "3.",
            f"[{BDSB2}]Free Match Mode[/{BDSB2}]",
            "Process files without using the artifact catalog",
        )
        table.add_row("", "", "")

        if exemplar_complete:
            table.add_row(
                "4.",
                "[bold dark_sea_green4]Reports & Visualization[/bold dark_sea_green4]",
                "Comparison reports and timeline charts",
            )
        else:
            table.add_row(
                "4.",
                "[dim]Reports & Visualization[/dim]",
                "Requires Exemplar Analysis",
            )

        table.add_row("", "", "")

        # Export Data - available if any scan exists
        if exemplar_complete:
            table.add_row(
                "5.",
                "[bold orange_red1]Export Data[/bold orange_red1]",
                "Save data for external apps and export exemplar packages",
            )
        else:
            table.add_row(
                "5.",
                "[dim]Export Data[/dim]",
                "Requires Exemplar or Candidate Scan",
            )

        # Import Data - always available
        table.add_row(
            "6.",
            "[bold orange_red1]Import Data[/bold orange_red1]",
            "Import exemplar packages from other projects",
        )
        table.add_row("", "", "")

        # Utilities only available on macOS (EWF mounting requires hdiutil)
        if not is_windows():
            table.add_row(
                "7.",
                "[bold light_goldenrod2]Utilities[/bold light_goldenrod2]",
                "Mount EWF Image (requires FUSE-T)",
            )
        table.add_row(
            "8.",
            "[bold light_goldenrod2]Settings[/bold light_goldenrod2]",
            "Set project scan and module options",
        )
        table.add_row("", "", "")

        table.add_row("9.", f"[bold dim {DSB1}]Close Project[/bold dim {DSB1}]")
        table.add_row(None, "")
        table.add_row(
            "[bold hot_pink]h[/bold hot_pink]",
            "[bold hot_pink]Help[/bold hot_pink]",
        )
        table.add_row("[red bold]q[/red bold]", "[red bold]Quit[/red bold]")

        panel = Panel(
            table,
            title=f"[{BDSB1}]Main Menu[/{BDSB1}]",
            border_style=f"{DSB3}",
        )
        self.console.print(panel)

        # Build valid choices based on exemplar status and platform
        # Note: "2" (Candidates Scan) is always available - gating is in subpage
        # "6" (Import Data) is always available
        valid_choices = ["1", "2", "3", "6", "8", "9", "h", "q"]
        if not is_windows():
            valid_choices.append("7")
        if exemplar_complete:
            valid_choices.extend(["4", "5"])

        sorted_choices = sorted(valid_choices, key=lambda x: (x, x.isdigit()))

        choice = Prompt.ask(
            "\n[bold cyan]Select option[/bold cyan]",
            choices=sorted_choices,
            show_default=False,
        ).lower()

        if choice == "1":
            self._exemplar_analysis()
        elif choice == "2":
            self._candidates_scan_menu()
        elif choice == "3":
            self._free_match_mode()
        elif choice == "4" and exemplar_complete:
            self._create_reports()
        elif choice == "5" and exemplar_complete:
            self._export_data()
        elif choice == "6":
            self._import_data_menu()
        elif choice == "7" and not is_windows():
            self._utilities_menu()
        elif choice == "8":
            self._project_settings()
        elif choice == "9":
            self._close_project()
        elif choice == "h":
            open_help()
        elif choice == "q":
            return "quit"

        return "continue"

    def _create_project(self) -> bool:
        """
        Create a new project.

        Returns:
            True if project was created, False otherwise
        """
        self.console.clear()
        self.console.print(
            Panel(
                "[bold cyan]Create New Project[/bold cyan]",
                border_style=f"{DSB1}",
            )
        )

        # Browse for directory
        directory = browse_for_directory(
            None,  # Use OS-appropriate default
            title="Select Project Directory",
            explanation="A new project folder will be created here.",
        )

        if not directory:
            self.console.print("[yellow]Project creation cancelled[/yellow]")
            sleep(0.8)
            return False

        # Get project metadata
        self.console.clear()
        self.console.print(
            Panel(
                f"[bold cyan]Creating project in:[/bold cyan]\n{directory}",
                border_style="cyan",
            )
        )

        project_name = Prompt.ask("\n[bold cyan]Project name[/bold cyan]")
        if not project_name:
            self.console.print("[red]Project name is required[/red]")
            Prompt.ask("\nPress Enter to continue")
            return False

        examiner_name = Prompt.ask("[bold cyan]Examiner name[/bold cyan]", default="")
        case_number = Prompt.ask("[bold cyan]Case number[/bold cyan]", default="")
        description = Prompt.ask("[bold cyan]Description[/bold cyan]", default="")

        # Confirm
        confirm_table = Table(show_header=False, box=None)
        confirm_table.add_column(style="bold", width=15)
        confirm_table.add_column()

        confirm_table.add_row("Directory:", str(directory))
        confirm_table.add_row("Project:", project_name)
        if examiner_name:
            confirm_table.add_row("Examiner:", examiner_name)
        if case_number:
            confirm_table.add_row("Case #:", case_number)
        if description:
            confirm_table.add_row("Description:", description)

        self.console.print("\n")
        self.console.print(confirm_table)

        if not Confirm.ask("\n[bold cyan]Create this project?[/bold cyan]", default=True):
            self.console.print("[yellow]Project creation cancelled[/yellow]")
            sleep(0.7)
            # Prompt.ask("\nPress Enter to continue")
            return False

        # Create project
        try:
            self.project = MARSProject.create(
                directory=directory,
                project_name=project_name,
                examiner_name=examiner_name or None,
                case_number=case_number or None,
                description=description or None,
            )

            # Reconfigure logger with project-specific settings (.marsproj)
            config = ConfigLoader.load(project_dir=self.project.project_dir)
            logger.configure(config, console=self.console, project_dir=self.project.project_dir)

            self.console.print(f"\n[{BDSB1}]Project created successfully![/{BDSB1}]")
            self.console.print(f"[dim]Location: {self.project.project_path}[/dim]")
            sleep(0.7)
            # Prompt.ask("\nPress Enter to continue")
            return True

        except Exception as e:
            self.console.print(f"\n[bold red]Error creating project:[/bold red] {e}")
            Prompt.ask("\nPress Enter to continue")
            return False

    def _open_project(self) -> bool:
        """
        Open an existing project.

        Returns:
            True if project was opened, False otherwise
        """
        self.console.clear()
        self.console.print(
            Panel(
                "[bold cyan]Open Existing Project[/bold cyan]",
                border_style=f"{DSB1}",
            )
        )

        # Browse for .marsproj file
        self.console.print("\n[bold]Select .marsproj file:[/bold]")
        project_file = browse_for_file(file_filter=".marsproj", title="Select Project File (.marsproj)")

        if not project_file:
            self.console.print("[yellow]Open cancelled[/yellow]")
            sleep(0.7)
            # Prompt.ask("\nPress Enter to continue")
            return False

        # Load project
        try:
            self.project = MARSProject.load(project_file)

            # Reconfigure logger with project-specific settings (.marsproj)
            config = ConfigLoader.load(project_dir=self.project.project_dir)
            logger.configure(config, console=self.console, project_dir=self.project.project_dir)

            self.console.print("\n[bold dark_sea_green4]Project opened successfully![/bold dark_sea_green4]")
            self.console.print(f"[dim]Project: {self.project.config['project_name']}[/dim]")
            return True

        except Exception as e:
            self.console.print(f"\n[bold red]Error opening project:[/bold red] {e}")
            Prompt.ask("\nPress Enter to continue")
            return False

    def _get_install_dir(self) -> Path | None:
        """Get MARS installation directory from executable path.

        Derives install directory from sys.executable:
            sys.executable = /path/to/MARS-x.x.x/.venv/bin/python3
            Install dir    = /path/to/MARS-x.x.x/

        Returns:
            Path to install directory, or None if not in a proper install
        """
        try:
            # .venv/bin/python3 -> .venv/bin -> .venv -> install_dir
            venv_dir = Path(sys.executable).parent.parent
            install_dir = venv_dir.parent
            # Verify this looks like a MARS install (has .venv)
            if (install_dir / ".venv").is_dir():
                return install_dir
        except Exception:
            pass
        return None

    def _get_config_file(self) -> Path | None:
        """Get config file path in install directory.

        Returns:
            Path to config.json in install directory, or None if not available
        """
        install_dir = self._get_install_dir()
        if install_dir:
            return install_dir / "config.json"
        return None

    def _get_last_project(self) -> Path | None:
        """Get the last opened project path from config.

        Reads from config.json in the MARS installation directory.

        Returns:
            Path to last project file or None
        """
        config_file = self._get_config_file()
        if not config_file or not config_file.exists():
            return None

        try:
            import json

            with Path.open(config_file) as f:
                config = json.load(f)
                last_project_path = config.get("last_project")

                if last_project_path:
                    path = Path(last_project_path)
                    if path.exists():
                        return path
        except Exception:
            pass

        return None

    def _save_last_project(self):
        """Save current project as last opened project.

        Writes to config.json in the MARS installation directory.
        Silently fails if not in a proper install (e.g., dev environment).
        """
        if not self.project:
            return

        config_file = self._get_config_file()
        if not config_file:
            return  # Not in a proper install - don't persist

        try:
            import json

            config = {}
            if config_file.exists():
                with Path.open(config_file) as f:
                    config = json.load(f)

            config["last_project"] = str(self.project.project_path)

            with Path.open(config_file, "w") as f:
                json.dump(config, f, indent=2)
        except Exception:
            pass  # Silent fail - not critical

    def _open_last_project(self, project_path: Path) -> bool:
        """
        Open the last project directly.

        Args:
            project_path: Path to project file

        Returns:
            True if successful
        """
        try:
            self.project = MARSProject.load(project_path)

            # Reconfigure logger with project-specific settings (.marsproj)
            config = ConfigLoader.load(project_dir=self.project.project_dir)
            logger.configure(config, console=self.console, project_dir=self.project.project_dir)

            with self.console.status("Loading..."):
                sleep(0.3)
            return True
        except Exception as e:
            self.console.print(f"\n[bold red]Error opening project:[/bold red] {e}")
            Prompt.ask("\nPress Enter to continue")
            return False

    def _close_project(self):
        """Close current project."""
        self.project = None
        self.console.print("[yellow]Project closed[/yellow]")
        sleep(0.7)

    # ==========================================================================
    # ============================ EXEMPLAR ANALYSIS ===========================
    # ==========================================================================
    def _exemplar_analysis(self):
        """Exemplar analysis workflow."""
        self.show_current_project_menu()

        # Show submenu
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="bold sky_blue2", width=4)
        table.add_column(style="grey69")

        table.add_row(
            "1.",
            f"[{BDSB1}]Disk Image[/{BDSB1}]   (E01, Ex01, DD, DMG, etc.)",
        )
        table.add_row(
            "2.",
            f"[{BDSB1}]Directory[/{BDSB1}]    (mounted macOS drive or folder)",
        )
        table.add_row(
            "3.",
            f"[{BDSB1}]Live System[/{BDSB1}]  (scan running macOS)",
        )
        table.add_row(
            "4.",
            f"[{BDSB1}]Archive[/{BDSB1}]      (TAR, ZIP, GZIP, etc.)",
        )
        table.add_row(None, "")
        table.add_row("[bold hot_pink]h[/bold hot_pink]", "[bold hot_pink]Help[/bold hot_pink]")
        table.add_row(None, "[dim](B)ack to main menu[/dim]")
        ref_source_group = Group("[bold]Select reference system source:[/bold]\n", table)

        self.console.print(
            Panel(
                ref_source_group,
                title=f"[{BDSB1}]Exemplar Analysis - Reference System[/{BDSB1}]",
                border_style=f"{DSB3}",
                padding=(0, 2),
            )
        )

        choice = Prompt.ask(
            "\n[bold cyan]Select option[/bold cyan]",
            choices=["1", "2", "3", "4", "h", "b"],
            show_default=False,
        ).lower()

        if choice == "1":
            self._exemplar_from_image()
        elif choice == "2":
            self._exemplar_from_directory()
        elif choice == "3":
            self._exemplar_from_live_system()
        elif choice == "4":
            self._exemplar_from_archive()
        elif choice == "h":
            open_help("exemplar-scan")
        # choice "b" - just return

    def _exemplar_from_directory(self):
        """Scan exemplar from mounted directory."""
        self.show_current_project_menu()

        source_path = browse_for_directory(
            start_path=None,  # Use OS-appropriate default
            title="Select System Root Directory",
            explanation="dfVFS can scan directly from directories or mounted disks.",
        )

        if not source_path:
            self.console.print("[yellow]Cancelled[/yellow]")
            # Prompt.ask("\nPress Enter to continue")
            sleep(0.3)
            return

        # Confirm and run scan
        # Use dfVFS for directories too (same indexed approach as images/archives)
        if self.project:
            ui = ExemplarScanUI(self.console, self.project)
            ui.run_scan(
                source_path,
                is_image=True,
                show_header_callback=self.show_current_project_menu,
            )

    def _exemplar_from_image(self):
        """Mount and scan exemplar from disk image."""
        self.show_current_project_menu()

        # Browse for image file
        self.console.print(f"[{BDSB1}]Browse to disk image:[/{BDSB1}]")
        image_path = browse_for_file(
            start_path=None,  # Use OS-appropriate default
            title="Select Disk Image (.E01, .dd, .dmg, etc.)",
            explanation="dfVFS will extract exemplar data directly from selected partitions.\n[bold red]⚠ NOTE:[/bold red] Compressed, segmented images may take a very long time to process.",
        )

        if not image_path:
            self.console.print("[yellow]Cancelled[/yellow]")
            # Prompt.ask("\nPress Enter to continue")
            sleep(0.3)
            return

        # Ensure we have a supported disk image extension
        suffix = image_path.suffix.lower()
        supported = {
            ".e01",
            ".ex01",
            ".s01",
            ".dd",
            ".raw",
            ".img",
            ".dmg",
            ".sparsebundle",
        }
        if suffix not in supported:
            self.console.print(f"\n[red]Unsupported image format: {suffix}[/red]")
            Prompt.ask("\nPress Enter to continue")
            return

        # Scan for available partitions and let user select
        if not self.project:
            return
        ui = ExemplarScanUI(self.console, self.project)
        selected_labels = ui.select_partitions(image_path, show_header_callback=self.show_current_project_menu)
        if not selected_labels:
            self.console.print("[yellow]Cancelled[/yellow]")
            sleep(0.3)
            return

        # Run exemplar scan directly via dfVFS (ExemplarScanner handles export)
        ui.run_scan(
            image_path,
            is_image=True,
            partition_labels=selected_labels,
            show_header_callback=self.show_current_project_menu,
        )

    # Archive file
    def _exemplar_from_archive(self):
        """Scan exemplar from archive file (TAR, ZIP, etc.)."""
        self.show_current_project_menu()

        archive_path = browse_for_file(
            start_path=None,  # Use OS-appropriate default
            title="Select Archive (.tar, .tar.gz, .zip, etc.)",
            explanation="[bold red]⚠ ALERT:[/bold red] Pre-extracting the archive is currently recommended.\ndfVFS can scan directly from archives, but it may take a very long time.",
        )

        if not archive_path:
            self.console.print("[yellow]Cancelled[/yellow]")
            sleep(0.3)
            return

        # Ensure we have a supported archive extension
        if not is_archive(archive_path):
            suffix = get_archive_extension(archive_path)
            self.console.print(f"\n[red]Unsupported archive format: {suffix}[/red]")
            Prompt.ask("\nPress Enter to continue")
            return

        # Archives are scanned directly without partition selection
        # dfVFS will handle extraction on-the-fly
        if self.project:
            ui = ExemplarScanUI(self.console, self.project)
            ui.run_scan(
                archive_path,
                is_image=True,
                show_header_callback=self.show_current_project_menu,
            )

    def _exemplar_from_live_system(self):
        """Scan exemplar from the live running macOS system."""
        # Windows gate - this feature requires macOS-specific paths and permissions
        if is_windows():
            self.console.print("[yellow]Live System Scan is not available on Windows.[/yellow]")
            self.console.print("[dim]This feature requires macOS Full Disk Access permissions.[/dim]")
            Prompt.ask("\nPress Enter to continue")
            return

        self.show_current_project_menu()
        self.console.print(
            Panel(
                f"[{BDSB1}]Scan Live System[/{BDSB1}]\n[italic grey50]Gather files directly from the running macOS system.[/italic grey50]",
                border_style=f"{DSB3}",
            )
        )

        # Show warning about Full Disk Access
        self.console.print("\n[bold yellow]This will scan your currently running macOS system.[/bold yellow]")
        self.console.print("[dim]This requires Full Disk Access (FDA) permissions to access protected databases.[/dim]")
        self.console.print(
            "[dim]FDA allows access to TCC, Powerlog, Contact Interactions, and other protected databases.[/dim]"
        )

        # Ask about sudo for maximum access
        self.console.print("\n[bold]Administrator Privileges:[/bold]")
        self.console.print("[dim]For maximum access (auth.db, known-networks.plist, SystemPolicy, etc.).[/dim]")
        self.console.print("[dim]the application must run with administrator privileges (sudo).[/dim]")

        # Ask non-admin user if they want admin scan
        if not is_admin():
            use_sudo = Prompt.ask(
                "\n[bold]Scan with administrator privileges?[/bold]",
                choices=["y", "n"],
                default="y",
            ).lower()

            if use_sudo == "y":
                # Try to auto-relaunch with sudo using the wrapper script
                mars_sudo = _get_mars_sudo_path()
                if mars_sudo:
                    self.console.print("\n[bold cyan]Relaunching with administrator privileges...[/bold cyan]")
                    self.console.print("[dim]You may be prompted for your password.[/dim]\n")
                    # Replace current process with sudo version, skip to scan confirmation
                    os.execvp(str(mars_sudo), [str(mars_sudo), "--exemplar-scan", "/"])
                else:
                    # Fallback: manual instructions if wrapper not found
                    self.console.print("\n[bold red]Error: mars-sudo wrapper not found[/bold red]")
                    self.console.print("[yellow]Please run manually:[/yellow]")
                    self.console.print("[dim]  sudo UV_CACHE_DIR=/tmp/uv-cache uv run mars --exemplar-scan /[/dim]")
                    self.console.print("\n[dim]This will skip directly to the scan confirmation.[/dim]")
                    Prompt.ask("\nPress Enter to continue")
                    return
            if use_sudo == "n":
                continue_scan = Prompt.ask(
                    "[bold yellow]\nContinue as non-sudo?[/bold yellow]",
                    choices=["y", "n"],
                    default="n",
                ).lower()
                if continue_scan != "y":
                    self.console.print("\n[bold red]Declined non-sudo scan.[/bold red]")
                    sleep(0.7)
                    return

        # Already running with elevated privileges
        if is_admin():
            self.console.print(
                "\n[bold dark_sea_green4][✓] Running with administrator privileges[/bold dark_sea_green4]"
            )
            sleep(0.7)

        # Scan from system root "/"
        # This leverages FDA permissions to access protected databases
        if self.project:
            ui = ExemplarScanUI(self.console, self.project)
            ui.run_scan(
                Path("/"),
                is_image=True,
                show_header_callback=self.show_current_project_menu,
            )

    # ==========================================================================
    # ============================ SCAN CANDIDATES =============================
    # ==========================================================================
    def _is_exemplar_complete(self) -> bool:
        """
        Check if exemplar analysis has been completed.

        Returns:
            True if at least one completed exemplar scan exists
        """
        project = self.project
        if project is None:
            return False

        # Check database for completed scans
        scans = project.get_exemplar_scans(active_only=True)
        return len(scans) > 0

    def _select_exemplar_scan(self) -> dict | None:
        """
        Select an exemplar scan to use for matching.

        Returns:
            Selected scan dict or None if cancelled
        """
        project = self.project
        if project is None:
            self.console.print("[bold red]No project loaded. Unable to select exemplar scan.[/bold red]")
            return None

        scans = project.get_exemplar_scans(active_only=True)

        if len(scans) == 0:
            return None

        # Validate that output directories still exist
        valid_scans = []
        stale_scans = []

        for scan in scans:
            output_dir = project.project_dir / "output" / scan["output_dir"]
            if output_dir.exists():
                valid_scans.append(scan)
            else:
                stale_scans.append(scan)

        # Mark stale scans as inactive in database
        if stale_scans:
            for scan in stale_scans:
                project.mark_exemplar_scan_inactive(scan["id"])

            # Show warning about stale scans
            self.console.print(
                f"\n[yellow]Found {len(stale_scans)} exemplar scan(s) with missing output directories.[/yellow]"
            )
            self.console.print("[dim]These have been marked as inactive and hidden from the list.[/dim]\n")
            sleep(1.5)

        if len(valid_scans) == 0:
            return None

        # If only one valid scan, auto-select it
        if len(valid_scans) == 1:
            return valid_scans[0]

        # Multiple scans - show selection menu
        self.show_current_project_menu()
        self.console.print(
            Panel(
                Group(
                    f"[{BDSB1}]Select Exemplar Scan[/{BDSB1}][italic grey50]\n\nThis is the scanned set of files to use for rubric matching.[/italic grey50]"
                ),
                border_style=f"{DSB3}",
            )
        )

        # Check for last used
        last_used = project.get_last_used_exemplar_scan()

        # Validate last_used still exists
        if last_used:
            last_used_dir = project.project_dir / "output" / last_used["output_dir"]
            if not last_used_dir.exists():
                last_used = None

        table = Table(
            show_header=True,
            header_style=f"{BDSB1}",
            box=box.ROUNDED,
            border_style="grey54",
        )
        table.add_column("#", style="bold", width=4)
        table.add_column("Timestamp", style=f"{BDSB3}")
        table.add_column("Description", style="italic")
        table.add_column("Files", justify="right")
        table.add_column("Schemas", justify="right")
        table.add_column("Status")

        for idx, scan in enumerate(valid_scans, start=1):
            timestamp = scan["timestamp"][:19].replace("T", " ") + "Z"  # Truncate and format
            status = "[green]✓[/green]" if scan["status"] == "completed" else ""

            # Highlight last used
            if last_used and scan["id"] == last_used["id"]:
                status += " [yellow](Last Used)[/yellow]"

            # Get description (truncate if too long)
            desc = scan.get("description") or ""
            if len(desc) > 30:
                desc = desc[:27] + "..."

            table.add_row(
                str(idx),
                timestamp,
                desc,
                str(scan["databases_found"]),
                str(scan["schemas_generated"]),
                status,
            )

        self.console.print(table)

        # Build choices
        choices = [str(i) for i in range(1, len(valid_scans) + 1)]
        if last_used:
            choices.append("l")
            self.console.print("\n[dim]L - Load last used[/dim]")

        choices.append("b")
        self.console.print("[dim]B - Back[/dim]")

        choice = Prompt.ask(
            "\n[bold cyan]Select exemplar scan[/bold cyan]",
            choices=choices,
            show_default=False,
        ).lower()

        if choice == "b":
            return None

        if choice == "l" and last_used:
            selected = last_used
        else:
            idx = int(choice) - 1
            selected = valid_scans[idx]

        # Mark as last used
        project.set_last_used_exemplar_scan(selected["id"])

        return selected

    def _has_imported_packages(self) -> bool:
        """Check if any imported exemplar packages exist.

        Returns:
            True if at least one valid imported package exists
        """
        if self.project is None:
            return False

        imports_dir = self.project.project_dir / "imports"
        if not imports_dir.exists():
            return False

        return any((d / "manifest.json").exists() for d in imports_dir.iterdir() if d.is_dir())

    def _candidates_scan_menu(self):
        """Candidates scan subpage - go directly to exemplar selection."""
        # Check what's available
        has_project_exemplar = self._is_exemplar_complete()
        has_imported = self._has_imported_packages()
        can_select = has_project_exemplar or has_imported

        if not can_select:
            # No exemplars available - show message and return
            self.show_current_project_menu()
            self.console.print(
                Panel(
                    "[bold]Candidates Scan[/bold]\n\n"
                    "[yellow]No exemplar scans or imported packages available.[/yellow]\n\n"
                    "[grey54]To run a Candidates Scan, you need either:[/grey54]\n"
                    "[grey54]  • A completed Exemplar Scan from this project[/grey54]\n"
                    "[grey54]  • An imported Exemplar Package (use [bold orange_red1]Import Data[/bold orange_red1] from Main Menu)[/grey54]",
                    border_style=f"{DSB3}",
                )
            )
            Prompt.ask("\nPress Enter to return to Main Menu")
            return

        # Run the scan - returns True if completed, False if cancelled
        if self._select_and_run_candidate_scan():
            # Scan completed - return to main menu
            return

    def _select_and_run_candidate_scan(self) -> bool:
        """Select exemplar (project or imported) and run candidate scan.

        Returns:
            True if a scan was completed, False if cancelled or no scan run.
        """
        project = self.project
        if project is None:
            return False

        has_project_exemplar = self._is_exemplar_complete()
        has_imported = self._has_imported_packages()

        # If both options exist, show unified selection
        self.show_current_project_menu()

        if has_project_exemplar and has_imported:
            # Show option to choose between project exemplar and imported
            self.console.print(
                Panel(
                    f"[{BDSB1}]Select Exemplar Source[/{BDSB1}]\nChoose the source of the exemplar data for your scan.",
                    title=f"[{BDSB1}]Candidates Scan[/{BDSB1}]",
                    border_style=f"{DSB3}",
                )
            )

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style=f"{BDSB1}", width=4)
            table.add_column()

            table.add_row(
                "1.",
                f"[{BDSB1}]Project Exemplar Scan[/{BDSB1}]         [grey69]Use a completed catalog exemplar scan from this project[/grey69]",
            )
            table.add_row(
                "2.",
                "[bold orange_red1]Imported Exemplar Package[/bold orange_red1]     [grey69]Use an imported exemplar package[/grey69]",
            )
            table.add_row("", "")
            table.add_row("[bold hot_pink]h[/bold hot_pink]", "[bold hot_pink]Help[/bold hot_pink]")
            table.add_row("", "[dim](B)ack[/dim]")

            self.console.print(
                Panel(
                    table,
                    border_style="grey69",
                )
            )

            choice = Prompt.ask(
                "\n[bold cyan]Select source[/bold cyan]",
                choices=["1", "2", "h", "b"],
                show_default=False,
            ).lower()

            if choice == "b":
                return False
            if choice == "1":
                return self._run_candidate_scan_with_project_exemplar()
            if choice == "2":
                return self._run_candidate_scan_with_imported_package()
            if choice == "h":
                open_help("candidates-scan")

        elif has_project_exemplar:
            # Only project exemplar available
            return self._run_candidate_scan_with_project_exemplar()

        elif has_imported:
            # Only imported packages available
            return self._run_candidate_scan_with_imported_package()

        return False

    def _run_candidate_scan_with_project_exemplar(self) -> bool:
        """Run candidate scan using project exemplar.

        Returns:
            True if scan was completed, False if cancelled.
        """
        exemplar_scan = self._select_exemplar_scan()

        if not exemplar_scan:
            self.console.print("[yellow]No exemplar scan selected[/yellow]")
            sleep(0.5)
            return False

        # Show scan type selection menu
        return self._select_scan_type_and_run(exemplar_scan)

    def _select_scan_type_and_run(self, exemplar_scan: dict) -> bool:
        """Select scan type (Raw Files or Time Machine) and run.

        Args:
            exemplar_scan: Selected exemplar scan dictionary

        Returns:
            True if scan was completed, False if cancelled.
        """
        self.show_current_project_menu()

        exemplar_display = exemplar_scan.get("description") or exemplar_scan["output_dir"]

        self.console.print(
            Panel(
                f"[{BDSB1}]Select Scan Type[/{BDSB1}]\nExemplar: [cyan]{exemplar_display}[/cyan]",
                title=f"[{BDSB1}]Candidates Scan[/{BDSB1}]",
                border_style=f"{DSB3}",
            )
        )

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style=f"{BDSB1}", width=4)
        table.add_column()

        table.add_row(
            "1.",
            f"[{BDSB1}]Raw Files Scan[/{BDSB1}]         [grey69]Scan a folder of carved/recovered files[/grey69]",
        )
        table.add_row(
            "2.",
            f"[{BDSB1}]Time Machine Scan[/{BDSB1}]      [grey69]Scan Time Machine backup volume[/grey69]",
        )
        table.add_row("", "")
        table.add_row("", "[dim](B)ack[/dim]")

        self.console.print(
            Panel(
                table,
                border_style="grey69",
            )
        )

        choice = Prompt.ask(
            "\n[bold cyan]Select scan type[/bold cyan]",
            choices=["1", "2", "b"],
            show_default=False,
        ).lower()

        if choice == "b":
            return False
        if choice == "1":
            return self._run_raw_files_scan(exemplar_scan)
        if choice == "2":
            return self._run_time_machine_scan(exemplar_scan)

        return False

    def _run_raw_files_scan(self, exemplar_scan: dict) -> bool:
        """Run raw files (carved) scan.

        Args:
            exemplar_scan: Selected exemplar scan dictionary

        Returns:
            True if scan was completed, False if cancelled.
        """
        self.show_current_project_menu()

        target_path = browse_for_directory(
            start_path=None,
            title="Select Carved Files Directory",
            explanation="Select the directory containing the carved files to be scanned using the selected exemplar scan.",
        )

        if not target_path:
            self.console.print("[yellow]Cancelled[/yellow]")
            sleep(0.3)
            return False

        # Run the scanning workflow
        self._run_target_scan(target_path, exemplar_scan)
        return True

    def _run_time_machine_scan(self, exemplar_scan: dict) -> bool:
        """Run Time Machine backup scan.

        Args:
            exemplar_scan: Selected exemplar scan dictionary

        Returns:
            True if scan was completed, False if cancelled.
        """
        from mars.utils.time_machine_utils import parse_backup_manifest

        project = self.project
        if project is None:
            return False

        # Initialize Time Machine UI
        tm_ui = TimeMachineScanUI(self.console, project, lambda: self.show_current_project_menu())

        # Select Time Machine volume
        tm_volume = tm_ui.select_tm_volume()
        if tm_volume is None:
            return False

        # Ask about sudo for maximum access (like exemplar scan)
        self.show_current_project_menu()
        self.console.print("\n[bold]Administrator Privileges:[/bold]")
        self.console.print("[dim]Some Time Machine files (auth.db, known-networks.plist, etc.)[/dim]")
        self.console.print("[dim]require administrator privileges to access.[/dim]")

        if not is_admin():
            use_sudo = Prompt.ask(
                "\n[bold]Scan with administrator privileges?[/bold]",
                choices=["y", "n"],
                default="y",
            ).lower()

            if use_sudo == "y":
                # Try to auto-relaunch with sudo using the wrapper script
                mars_sudo = _get_mars_sudo_path()
                if mars_sudo:
                    self.console.print("\n[bold cyan]Relaunching with administrator privileges...[/bold cyan]")
                    self.console.print("[dim]You may be prompted for your password.[/dim]\n")
                    # Replace current process with sudo version, skip to TM scan
                    os.execvp(str(mars_sudo), [str(mars_sudo), "--tm-scan", str(tm_volume)])
                else:
                    # Fallback: manual instructions if wrapper not found
                    self.console.print("\n[bold red]Error: mars-sudo wrapper not found[/bold red]")
                    self.console.print("[yellow]Please run manually:[/yellow]")
                    self.console.print(
                        f"[dim]  sudo UV_CACHE_DIR=/tmp/uv-cache uv run mars --tm-scan {tm_volume}[/dim]"
                    )
                    self.console.print("\n[dim]This will skip directly to the TM backup selection.[/dim]")
                    Prompt.ask("\nPress Enter to continue")
                    return False
            if use_sudo == "n":
                continue_scan = Prompt.ask(
                    "[bold yellow]\nContinue as non-sudo?[/bold yellow]",
                    choices=["y", "n"],
                    default="n",
                ).lower()
                if continue_scan != "y":
                    self.console.print("\n[bold red]Declined non-sudo scan.[/bold red]")
                    sleep(0.7)
                    return False

        # Already running with elevated privileges
        if is_admin():
            self.console.print(
                "\n[bold dark_sea_green4][✓] Running with administrator privileges[/bold dark_sea_green4]"
            )
            sleep(0.7)

        # Parse manifest and get available backups
        try:
            backups = parse_backup_manifest(tm_volume / "backup_manifest.plist")
        except Exception as e:
            self.console.print(f"[bold red]Error reading backup manifest:[/bold red] {e}")
            sleep(1)
            return False

        if not backups:
            self.console.print("[yellow]No backups found in the selected volume.[/yellow]")
            sleep(1)
            return False

        # Select backups to scan
        selected_backups = tm_ui.select_backups(backups)
        if not selected_backups:
            return False

        # Run the scan
        tm_ui.run_scan(
            tm_volume=tm_volume,
            selected_backups=selected_backups,
            exemplar_scan=exemplar_scan,
            show_header_callback=self.show_current_project_menu,
        )

        return True

    def _run_candidate_scan_with_imported_package(self) -> bool:
        """Run candidate scan using imported exemplar package.

        Returns:
            True if scan was completed, False if cancelled.
        """
        project = self.project
        if project is None:
            return False

        # Select imported package
        selected_package = self._select_imported_package()
        if not selected_package:
            return False

        package_path, package_name = selected_package

        self.show_current_project_menu()

        target_path = browse_for_directory(
            start_path=None,
            title="Select Carved Files Directory",
            explanation=f"Select the directory to scan using imported exemplar: {package_name}",
        )

        if not target_path:
            self.console.print("[yellow]Cancelled[/yellow]")
            sleep(0.3)
            return False

        # Run the scanning workflow with imported package
        self._run_target_scan_with_import(target_path, package_path, package_name)
        return True

    def _select_imported_package(self) -> tuple[Path, str] | None:
        """Select an imported exemplar package.

        Returns:
            Tuple of (package_path, package_name) or None if cancelled
        """
        project = self.project
        if project is None:
            return None

        imports_dir = project.project_dir / "imports"
        if not imports_dir.exists():
            self.console.print("[yellow]No imported packages found.[/yellow]")
            sleep(0.5)
            return None

        # Find all valid packages
        packages = []
        for d in sorted(imports_dir.iterdir()):
            if not d.is_dir():
                continue
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                try:
                    import json

                    with manifest_path.open() as f:
                        manifest = json.load(f)
                    packages.append(
                        {
                            "path": d,
                            "name": manifest.get("name", d.name),
                            "description": manifest.get("description", ""),
                            "db_count": manifest.get("database_count", 0),
                            "os_info": manifest.get("os_info", {}),
                        }
                    )
                except Exception:
                    continue

        if not packages:
            self.console.print("[yellow]No valid imported packages found.[/yellow]")
            sleep(0.5)
            return None

        # Show selection menu
        self.show_current_project_menu()
        self.console.print(
            Panel(
                "[bold]Select Imported Exemplar Package[/bold]",
                border_style="light_goldenrod3",
            )
        )

        table = Table(
            show_header=True,
            header_style=f"{BDSB1}",
            box=box.ROUNDED,
            border_style="grey54",
        )
        table.add_column("#", style="bold", width=4)
        table.add_column("Name", style="light_goldenrod3")
        table.add_column("Description", style="italic")
        table.add_column("DBs", justify="right")
        table.add_column("OS", style="dim")

        for idx, pkg in enumerate(packages, start=1):
            os_str = ""
            if pkg["os_info"]:
                os_name = pkg["os_info"].get("name", "")
                os_ver = pkg["os_info"].get("version", "")
                os_str = f"{os_name} {os_ver}".strip()

            table.add_row(
                str(idx),
                pkg["name"],
                pkg["description"][:30] + ("..." if len(pkg["description"]) > 30 else ""),
                str(pkg["db_count"]),
                os_str,
            )

        self.console.print(table)
        self.console.print("\n[dim]B - Back[/dim]")

        choices = [str(i) for i in range(1, len(packages) + 1)] + ["b"]
        choice = Prompt.ask(
            "\n[bold cyan]Select package[/bold cyan]",
            choices=choices,
            show_default=False,
        ).lower()

        if choice == "b":
            return None

        selected = packages[int(choice) - 1]
        return (selected["path"], selected["name"])

    def _import_data_menu(self):
        """Import Data menu for importing exemplar packages."""
        while True:
            self.show_current_project_menu()

            self.console.print(
                Panel(
                    "[bold]Import Data[/bold]\n\n"
                    "[grey54]Import exemplar packages exported from other MARS projects.\n"
                    "Imported packages can be used as reference databases for Candidates Scans,\n"
                    "allowing you to recover data without scanning an exemplar system directly.[/grey54]",
                    border_style="orange_red1",
                )
            )

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold orange_red1", width=4)
            table.add_column()
            table.add_column(style="grey69")

            table.add_row(
                "1.",
                "[bold orange_red1]Import Exemplar Package[/bold orange_red1]",
                "Import reference databases from another system",
            )
            table.add_row("", "")
            table.add_row(
                "[bold hot_pink]h[/bold hot_pink]",
                "[bold hot_pink]Help[/bold hot_pink]",
            )
            table.add_row("", "[dim](B)ack[/dim]")

            self.console.print(table)

            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=["1", "h", "b"],
                show_default=False,
            ).lower()

            if choice == "b":
                return
            if choice == "h":
                open_help("import-data")
                continue
            if choice == "1":
                self._import_exemplar_package()

    def _import_exemplar_package(self):
        """Import an exemplar package into the project."""
        project = self.project
        if project is None:
            self.console.print("[bold red]No project loaded.[/bold red]")
            return

        self.show_current_project_menu()

        self.console.print(
            Panel(
                "[bold light_goldenrod3]Import Exemplar Package[/bold light_goldenrod3]\n\n"
                "Import an exemplar package created by another MARS project.\n"
                "This allows you to run candidate scans using reference databases\n"
                "without needing to scan an exemplar system.",
                border_style="light_goldenrod3",
            )
        )

        # Browse for package directory
        package_path = browse_for_directory(
            start_path=None,
            title="Select Exemplar Package Directory",
            explanation="Select a folder containing an exported exemplar package (with manifest.json).",
        )

        if not package_path:
            self.console.print("[yellow]Cancelled[/yellow]")
            sleep(0.3)
            return

        # Validate it's an exemplar package
        manifest_path = package_path / "manifest.json"
        if not manifest_path.exists():
            self.console.print("[bold red]Invalid exemplar package:[/bold red] manifest.json not found.")
            self.console.print("[dim]Select a folder created by 'Export Exemplar Package'.[/dim]")
            Prompt.ask("\nPress Enter to continue")
            return

        # Load manifest for display
        try:
            import json

            with manifest_path.open() as f:
                manifest = json.load(f)
        except Exception as e:
            self.console.print(f"[bold red]Error reading manifest:[/bold red] {e}")
            Prompt.ask("\nPress Enter to continue")
            return

        # Show package info
        self.console.print("\n[bold]Package Info:[/bold]")
        self.console.print(f"  Name: [green]{manifest.get('exemplar_name', 'Unknown')}[/green]")
        if manifest.get("description"):
            self.console.print(f"  Description: {manifest['description']}")
        if manifest.get("os_info"):
            os_info = manifest["os_info"]
            os_str = f"{os_info.get('name', '')} {os_info.get('version', '')}".strip()
            if os_str:
                self.console.print(f"  OS: {os_str}")
        self.console.print(f"  Databases: {manifest.get('database_count', 0)}")

        if not Confirm.ask("\n[bold cyan]Import this package?[/bold cyan]", default=True):
            return

        # Perform import
        self.console.print()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Importing exemplar package...", total=None)

            imports_dir = project.project_dir / "imports"
            importer = ExemplarImporter(imports_dir)

            result = importer.import_package(package_path)

            progress.update(task, completed=True)

        # Show results
        if result["success"]:
            # Get package name from manifest
            manifest = result.get("manifest")
            package_name = manifest.exemplar_name if manifest else package_path.name
            import_location = imports_dir / package_path.name

            self.show_current_project_menu()
            self.console.print(
                Panel(
                    f"[bold dark_sea_green4]Import Complete![/bold dark_sea_green4]\n\n"
                    f"Package: [cyan]{package_name}[/cyan]\n"
                    f"Location: [dim]{import_location}[/dim]",
                    border_style="green",
                )
            )

            # Ask if user wants to proceed with scan
            if Confirm.ask(
                "\n[bold cyan]Run candidate scan now?[/bold cyan]",
                default=True,
            ):
                self._run_candidate_scan_with_imported_package()
        else:
            self.console.print(
                Panel(
                    "[bold red]Import Failed[/bold red]",
                    border_style="red",
                )
            )
            error_msg = result.get("error", "Unknown error")
            self.console.print(f"  [red]{error_msg}[/red]")

            Prompt.ask("\nPress Enter to continue")

    def _run_target_scan_with_import(self, target_path: Path, package_path: Path, package_name: str):
        """Run target scan using an imported exemplar package.

        Args:
            target_path: Directory to scan
            package_path: Path to imported exemplar package
            package_name: Display name of the package
        """
        if self.project is None:
            self.console.print("[bold red]No project loaded. Unable to scan.[/bold red]")
            return

        ui = TargetScanUI(self.console, self.project)
        ui.run_scan(
            target_path,
            exemplar_scan=None,  # No project exemplar
            imported_package_path=package_path,
            imported_package_name=package_name,
            show_header_callback=self.show_current_project_menu,
        )

    def _run_target_scan(self, target_path: Path, exemplar_scan: dict):
        """Run raw scanner on target directory."""
        if self.project is None:
            self.console.print("[bold red]No project loaded. Unable to scan carved files.[/bold red]")
            return

        ui = TargetScanUI(self.console, self.project)
        ui.run_scan(
            target_path,
            exemplar_scan,
            show_header_callback=self.show_current_project_menu,
        )

    # ==========================================================================
    # =========================== CREATE CHARTS ================================
    # ==========================================================================

    def _create_reports(self):
        """Create reports and charts."""
        exemplar_complete = self._is_exemplar_complete()

        # First, choose which tool to use
        self.show_current_project_menu()

        reports_explain_text_1 = (
            f"[{BDSB2}]Data Recovery Comparison[/{BDSB2}] includes a timeline analysis of exemplar vs. recovered data."
        )
        reports_explain_text_2 = f"[{BDSB2}]Chart Plotter[/{BDSB2}] creates graphs of data against timeline columns [italic](e.g., Powerlog level vs. time).[/italic]"
        reports_explain_text_3 = f"[{BDSB2}]Report Index[/{BDSB2}] lists all scans and their output files."
        reports_text_group = Group(reports_explain_text_1, reports_explain_text_2, reports_explain_text_3)

        self.console.print(
            Panel(
                reports_text_group,
                border_style=f"{DSB3}",
                style="grey54",
                title=f"[{BDSB1}]Reports & Visualization[/{BDSB1}]",
            )
        )

        # Show tool options
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style=f"{BDSB1}", width=4)
        table.add_column()

        table.add_row(
            "1.",
            "[bold dark_sea_green4]Data Recovery Comparison[/bold dark_sea_green4] - Exemplar vs Candidate analysis",
        )
        table.add_row(
            "2.",
            "[bold dark_sea_green4]Chart Plotter[/bold dark_sea_green4] - Visualize database timelines",
        )
        if exemplar_complete:
            table.add_row(
                "3.",
                "[bold dark_sea_green4]Report Index[/bold dark_sea_green4] - Lists all scans",
            )
        table.add_row(None, "")
        table.add_row("[bold hot_pink]h[/bold hot_pink]", "[bold hot_pink]Help[/bold hot_pink]")
        table.add_row(None, "(B)ack to main menu")

        self.console.print(
            Panel(
                table,
                border_style="grey69",
            )
        )

        choices = ["1", "2", "h", "b"]
        if exemplar_complete:
            choices = ["1", "2", "3", "h", "b"]

        choice = Prompt.ask(
            "\n[bold cyan]Select option[/bold cyan]",
            choices=choices,
            show_default=False,
        ).lower()

        if choice == "b":
            return
        if choice == "h":
            open_help("reports")
            return

        # Open the index file HTML that links to all scans
        if choice == "3" and Confirm.ask("\nOpen report in browser?", default=True):
            project = self.project
            if project is None:
                return

            index_file = project.project_dir / "output" / "index.html"

            if not index_file.is_file:
                return

            import webbrowser

            webbrowser.open(f"file://{index_file}")
            return

        if choice == "2":
            # Chart Plotter - ask for source type FIRST
            self._run_chart_plotter_flow()
            return

        # Data Recovery Comparison (choice == "1")
        # Select exemplar scan first
        exemplar_scan = self._select_exemplar_scan()

        if not exemplar_scan:
            self.console.print("[yellow]No exemplar scan selected[/yellow]")
            sleep(0.5)
            return

        project = self.project
        if project is None:
            self.console.print("[bold red]No project loaded.[/bold red]")
            return

        exemplar_output_dir = project.project_dir / "output" / exemplar_scan["output_dir"]

        # Check if candidates exist for comparison
        candidates_root = exemplar_output_dir / "candidates"
        if not candidates_root.exists() or not list(candidates_root.iterdir()):
            self.console.print(
                "[yellow]No candidate scans found for this exemplar. Run a candidate scan first.[/yellow]"
            )
            Prompt.ask("\nPress Enter to continue")
            return
        self._generate_comparison_report(exemplar_scan, exemplar_output_dir)

    def _run_chart_plotter_flow(self):
        """Run chart plotter - goes directly to Selection Buffer."""
        if self.project is None:
            self.console.print("[bold red]No project loaded.[/bold red]")
            return

        # Use PlotterUI with buffer-based workflow
        # Exemplar selection is deferred to when user clicks "Add Buffer"
        ui = PlotterUI(self.console, self.project)
        ui.show_menu_with_buffer(show_header_callback=self.show_current_project_menu)

    def _generate_comparison_report(self, exemplar_scan: dict, exemplar_output_dir: Path):
        """Generate data recovery comparison report."""
        if self.project is None:
            return

        ui = ComparisonUI(self.console, self.project)
        ui.generate_report(
            exemplar_scan,
            exemplar_output_dir,
            show_header_callback=self.show_current_project_menu,
        )

    def _free_match_mode(self):
        """Launch Free Match Mode submenu."""
        ui = FreeMatchUI(self.console, self.project, lambda: self.show_current_project_menu())
        ui.show_menu()

    def _export_data(self):
        """Launch Export Data menu for external tool packaging."""
        # Get selected exemplar scan (if any)
        exemplar_scan = self._select_exemplar_scan()
        if not exemplar_scan:
            return

        if self.project is None:
            self.console.print("[bold red]No project loaded.[/bold red]")
            return

        ui = ExportUI(self.console, self.project)
        ui.show_menu(
            exemplar_scan=exemplar_scan,
            show_header_callback=self.show_current_project_menu,
        )

    def _utilities_menu(self):
        """Utilities submenu."""
        while True:
            self.console.clear()
            self.show_banner()

            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold light_goldenrod2", width=4)
            table.add_column()

            table.add_row(
                "1.",
                "[bold light_goldenrod2]Mount EWF/E01 Image[/bold light_goldenrod2]",
            )
            table.add_row(None, "")
            table.add_row(
                "[bold hot_pink]h[/bold hot_pink]",
                "[bold hot_pink]Help[/bold hot_pink]",
            )
            table.add_row(None, "[bold red](B)ack to Main Menu[/bold red]")

            panel = Panel(
                table,
                title=f"[{BDSB1}]Utilities[/{BDSB1}]",
                border_style=f"{BDSB3}",
            )
            self.console.print(panel)

            choice = (
                Prompt.ask(
                    "\n[bold cyan]Select option[/bold cyan]",
                    choices=["1", "h", "b"],
                    show_default=False,
                )
                .strip()
                .lower()
            )

            if choice == "1":
                self._mount_ewf_image()
            elif choice == "h":
                open_help("utilities")
            elif choice == "b":
                break

    def _mount_ewf_image(self):
        """Mount an EWF/E01 forensic image."""
        # Windows gate - EWF mounting requires macOS-specific tools
        if is_windows():
            self.console.print("[yellow]EWF mounting is not yet available on Windows.[/yellow]")
            self.console.print("[dim]This feature requires macOS tools (ewfmount, hdiutil).[/dim]")
            Prompt.ask("\nPress Enter to continue")
            return

        ui = EWFMountUI(self.console)
        ui.show_menu(show_banner_callback=self.show_banner)

    def _project_settings(self):
        """Settings menu using dedicated SettingsUI module."""
        # Load current config
        config = ConfigLoader.load(
            project_dir=self.project.project_dir if self.project else None,
        )

        # Get project directory (None if no project loaded)
        project_dir = self.project.project_dir if self.project else None

        # Launch settings UI
        settings_ui = SettingsUI(
            config,
            self.console,
            project_dir,
            lambda: self.show_current_project_menu(),
        )

        settings_ui.show_main_menu()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="MARS - macOS Artifact Recovery Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--exemplar-scan",
        metavar="PATH",
        help="Skip to exemplar scan confirmation for the given source path (used by mars-sudo)",
    )
    parser.add_argument(
        "--tm-scan",
        metavar="VOLUME_PATH",
        help="Skip to Time Machine scan for the given TM volume path (used by mars-sudo)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    # Fix Windows console encoding for Unicode characters
    if sys.platform == "win32":
        import os

        os.system("")  # Enable ANSI escape sequences on Windows
        # reconfigure is available on TextIOWrapper but not in TextIO type stub
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    args = parse_args()

    try:
        cli = MARSCLI()

        if args.exemplar_scan:
            # Skip main menu, go directly to exemplar scan (used after sudo elevation)
            cli.run_direct_exemplar_scan(Path(args.exemplar_scan))
        elif args.tm_scan:
            # Skip main menu, go directly to TM scan (used after sudo elevation)
            cli.run_direct_tm_scan(Path(args.tm_scan))
        else:
            cli.run()
    except KeyboardInterrupt:
        Console().print("\n\n[bold yellow]Interrupted by user[/bold yellow]")
        sys.exit(0)
    except Exception as e:
        Console().print(f"\n[bold red]Fatal error:[/bold red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
