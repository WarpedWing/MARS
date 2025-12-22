#!/usr/bin/env python3
"""Settings UI module for MARS TUI.

Provides section-based settings management with validation, module toggles,
and persistence to user preferences.
"""

from __future__ import annotations

import copy
import re
from datetime import datetime
from time import sleep
from typing import TYPE_CHECKING

import yaml
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from mars.cli.arc_manager_ui import ARCManagerUI
from mars.config import ConfigLoader
from mars.config.schema import (
    CarverConfig,
    ExemplarScanConfig,
    UIConfig,
    VariantSelectorConfig,
    get_user_configurable_fields,
)
from mars.pipeline.common.catalog_manager import CatalogManager
from mars.report_modules.report_module_manager import ReportModuleManager
from mars.utils.debug_logger import logger
from mars.utils.platform_utils import open_help

if TYPE_CHECKING:
    from pathlib import Path

    from rich.console import Console

    from mars.config import MARSConfig
    from mars.report_modules.module_config import ModuleConfig


# ============================================================================
# Validation Helpers
# ============================================================================


def validate_date_format(date_str: str) -> tuple[bool, str]:
    """Validate date string format (YYYY-MM-DD).

    Args:
        date_str: Date string to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check format with regex
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return False, "Invalid format. Use YYYY-MM-DD (e.g., 2025-01-15)"

    # Check if it's a valid date
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError as e:
        return False, f"Invalid date: {e}"

    # Check year is reasonable (2000-2035)
    year = parsed_date.year
    if year < 2000 or year > 2035:
        return False, f"Year {year} is out of reasonable range (2000-2035)"

    return True, ""


def validate_date_range(start_str: str, end_str: str) -> tuple[bool, str]:
    """Validate that end date is after start date.

    Args:
        start_str: Start date string (YYYY-MM-DD)
        end_str: End date string (YYYY-MM-DD)

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        start = datetime.strptime(start_str, "%Y-%m-%d")
        end = datetime.strptime(end_str, "%Y-%m-%d")

        if end <= start:
            return False, "End date must be after start date"

        return True, ""
    except ValueError:
        return False, "Invalid date format"


def sanitize_text(text: str) -> str:
    """Sanitize user text input.

    Args:
        text: Raw text input

    Returns:
        Sanitized text
    """
    # Strip whitespace
    text = text.strip()

    # Reject dangerous characters (basic protection)
    if any(c in text for c in [";", "&", "|", "`", "$", "(", ")"]):
        return ""

    return text


# ============================================================================
# Settings UI Class
# ============================================================================


class SettingsUI:
    """Interactive settings UI with section-based navigation."""

    def __init__(
        self,
        config: MARSConfig,
        console: Console,
        project_dir: Path | None,
        proj_header=None,
    ):
        """Initialize settings UI.

        Args:
            config: MARS configuration object
            console: Rich console for output
            project_dir: Project directory for saving config (None = no save)
            proj_header: Persistent project header
        """
        self.config = config
        self.console = console
        self.project_dir = project_dir
        self.module_manager = ReportModuleManager(config=config, console=console)
        self.proj_header = proj_header
        # Store snapshot for unsaved changes detection
        self._original_config = copy.deepcopy(config.to_user_configurable_dict())

    def _has_unsaved_changes(self) -> bool:
        """Check if config has been modified since last save."""
        current = self.config.to_user_configurable_dict()
        return current != self._original_config

    def show_main_menu(self) -> None:
        """Show main settings menu with section navigation."""
        while True:
            # Brought the Current Project header over
            if self.proj_header:
                self.proj_header()

            # Build menu
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold cyan", width=4)
            table.add_column()

            table.add_row("1.", "[bold light_goldenrod2]General Settings[/bold light_goldenrod2]")
            table.add_row("", "")
            table.add_row("2.", "[bold blue]Exemplar Scan Settings[/bold blue]")
            table.add_row("3.", "[bold blue]Candidates Scan Settings[/bold blue]")
            table.add_row("", "")
            table.add_row("4.", "[bold indian_red]Artifact Recovery Catalog (ARC) Management[/bold indian_red]")
            table.add_row("5.", "[bold indian_red]Report Module Management[/bold indian_red]")
            table.add_row("", "")
            table.add_row("[green]s[/green]", "[bold dark_sea_green4]Save Settings to Project[/bold dark_sea_green4]")
            table.add_row("", "")
            table.add_row(
                "[bold hot_pink]h[/bold hot_pink]",
                "[bold hot_pink]Help[/bold hot_pink]",
            )
            table.add_row(None, "[bold red](B)ack to Main Menu[/bold red]")

            panel = Panel(
                table,
                title="[bold deep_sky_blue1]Settings Manager[/bold deep_sky_blue1]",
                border_style="deep_sky_blue3",
            )
            self.console.print(panel)

            choice = Prompt.ask(
                "\n[bold cyan]Select section[/bold cyan]",
                choices=["1", "2", "3", "4", "5", "s", "h", "b"],
                show_default=False,
            ).lower()

            if choice == "h":
                open_help("settings")
                continue
            if choice == "b":
                if self._has_unsaved_changes():
                    if self._confirm_discard_changes():
                        break
                    # else: stay in menu (continue loop)
                else:
                    break
            elif choice == "1":
                self._general_settings()
            elif choice == "2":
                self._exemplar_scan_settings()
            elif choice == "3":
                self._candidate_scan_settings()
            elif choice == "4":
                arc_manager = ARCManagerUI(self.console)
                arc_manager.show_menu(show_header_callback=self.proj_header)
            elif choice == "5":
                self._module_settings()
            elif choice == "s":
                save_successful = self._save_preferences()
                if save_successful:
                    break

    # ========================================================================
    # Section: General Settings
    # ========================================================================

    def _general_settings(self) -> None:
        """General settings section (debug mode toggles progress bars automatically)."""
        # Get UIConfig fields
        ui_fields = get_user_configurable_fields(UIConfig, category="basic")

        while True:
            if self.proj_header:
                self.proj_header()

            # Build field map
            field_map = {}
            idx = 1
            for field_name, field_info in ui_fields.items():
                field_map[idx] = ("ui", field_name, field_info)
                idx += 1

            # Build menu
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold cyan", width=4)
            table.add_column()
            table.add_column(style="grey69")

            table.add_row("", "[bold dark_sea_green4]General / Debug Settings[/bold dark_sea_green4]", "")
            table.add_row("", "", "")

            for menu_idx in sorted(field_map.keys()):
                section_name, field_name, field_info = field_map[menu_idx]
                current_value = getattr(self.config.ui, field_name)
                label = field_info["metadata"]["label"]
                description = field_info["metadata"]["description"]

                if isinstance(current_value, bool):
                    state = "[green]ON" if current_value else "[red]OFF"
                    display = f"[bold]{label}[/bold]: {state}[/]"
                else:
                    display = f"[bold]{label}[/bold]: [cyan]{current_value}[/cyan]"

                table.add_row(f"{menu_idx}.", display, description)

            table.add_row("", "", "")
            table.add_row(None, "[bold red](B)ack to Settings Menu[/bold red]")

            panel = Panel(
                table,
                title="[bold dark_sea_green4]General Settings[/bold dark_sea_green4]",
                border_style="green",
            )
            self.console.print(panel)

            valid_choices = [str(i) for i in range(1, len(field_map) + 1)] + ["b"]
            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=valid_choices,
                show_default=False,
            ).lower()

            if choice == "b":
                break

            # Toggle field
            menu_idx = int(choice)
            if menu_idx in field_map:
                section_name, field_name, field_info = field_map[menu_idx]
                self._toggle_boolean_field("ui", field_name)

    # ========================================================================
    # Section: Candidates Scan Settings
    # ========================================================================

    def _candidate_scan_settings(self) -> None:
        """Candidates scan settings (CarverConfig + VariantSelectorConfig)."""
        # Get fields from both configs
        carver_fields = get_user_configurable_fields(CarverConfig, category="advanced")
        variant_fields = get_user_configurable_fields(VariantSelectorConfig, category="advanced")

        while True:
            if self.proj_header:
                self.proj_header()

            # Build field map
            field_map = {}
            idx = 1

            # Carver fields
            for field_name, field_info in carver_fields.items():
                field_map[idx] = ("carver", field_name, field_info)
                idx += 1

            # Variant selector fields
            for field_name, field_info in variant_fields.items():
                field_map[idx] = ("variant_selector", field_name, field_info)
                idx += 1

            # Build menu
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold cyan", width=4)
            table.add_column()
            table.add_column(style="grey69")

            table.add_row("", "[bold yellow]Carving Settings[/bold yellow]", "")
            table.add_row("", "", "")

            # Show carver fields
            for menu_idx, (section_name, field_name, field_info) in field_map.items():
                if section_name != "carver":
                    continue

                section_obj = getattr(self.config, section_name)
                current_value = getattr(section_obj, field_name)
                label = field_info["metadata"]["label"]
                description = field_info["metadata"]["description"]

                if isinstance(current_value, bool):
                    state = "[green]ON" if current_value else "[red]OFF"
                    display = f"[bold]{label}[/bold]: {state}[/]"
                else:
                    display = f"[bold]{label}[/bold]: [cyan]{current_value}[/cyan]"

                table.add_row(f"{menu_idx}.", display, description)

            table.add_row("", "", "")
            table.add_row("", "[bold yellow]Variant Selection Settings[/bold yellow]")
            table.add_row("", "", "")

            # Show variant selector fields
            for menu_idx, (section_name, field_name, field_info) in field_map.items():
                if section_name != "variant_selector":
                    continue

                section_obj = getattr(self.config, section_name)
                current_value = getattr(section_obj, field_name)
                label = field_info["metadata"]["label"]
                description = field_info["metadata"]["description"]

                if isinstance(current_value, bool):
                    state = "[green]ON" if current_value else "[red]OFF"
                    display = f"[bold]{label}[/bold]: {state}[/]"
                else:
                    display = f"[bold]{label}[/bold]: [cyan]{current_value}[/cyan]"

                table.add_row(f"{menu_idx}.", display, description)

            table.add_row("", "", "")
            table.add_row(None, "[bold red](B)ack to Settings Menu[/bold red]")

            panel = Panel(
                table,
                title="[bold yellow]Candidates Scan Settings[/bold yellow]",
                border_style="yellow",
            )
            self.console.print(panel)

            valid_choices = [str(i) for i in range(1, len(field_map) + 1)] + ["b"]
            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=valid_choices,
                show_default=False,
            ).lower()

            if choice == "b":
                break

            # Update selected field
            menu_idx = int(choice)
            if menu_idx in field_map:
                section_name, field_name, field_info = field_map[menu_idx]
                self._validate_and_update_field(section_name, field_name, field_info)

    # ========================================================================
    # Section: Exemplar Scan Settings
    # ========================================================================

    def _exemplar_scan_settings(self) -> None:
        """Exemplar scan settings (epoch bounds, rubric params, catalog groups)."""
        # Define category groups with display labels
        exemplar_categories = [
            ("exemplar_timestamp", "Timestamp Validation"),
            ("exemplar_rubric", "Rubric Generation"),
            ("exemplar_catalog", "Catalog Groups"),
            ("exemplar_filetype", "File Type Filtering"),
        ]

        while True:
            if self.proj_header:
                self.proj_header()

            cfg = self.config.exemplar

            # Build field map with menu indices
            field_map: dict[int, tuple[str, dict]] = {}
            idx = 1

            # Collect fields for each category
            for cat_name, _ in exemplar_categories:
                cat_fields = get_user_configurable_fields(ExemplarScanConfig, category=cat_name)
                for field_name, field_info in cat_fields.items():
                    field_map[idx] = (field_name, field_info)
                    idx += 1

            # Build menu table
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold cyan", width=4)
            table.add_column()
            table.add_column(style="grey69")

            # Render fields grouped by category
            current_idx = 1
            for cat_name, cat_label in exemplar_categories:
                cat_fields = get_user_configurable_fields(ExemplarScanConfig, category=cat_name)
                if not cat_fields:
                    continue

                # Add section header
                table.add_row("", f"[bold blue]{cat_label}[/bold blue]", "")

                for field_name, field_info in cat_fields.items():
                    metadata = field_info["metadata"]
                    current_value = getattr(cfg, field_name)
                    label = metadata.get("label", field_name)
                    description = metadata.get("description", "")

                    # Format display based on field type
                    if isinstance(current_value, list):
                        value_str = self._format_list_field_display(field_name, current_value)
                        display = f"[bold]{label}[/bold]: {value_str}"
                    elif isinstance(current_value, bool):
                        state = "[green]ON" if current_value else "[red]OFF"
                        display = f"[bold]{label}[/bold]: {state}[/]"
                    else:
                        display = f"[bold]{label}[/bold]: [cyan]{current_value}[/cyan]"

                    table.add_row(f"{current_idx}.", display, description)
                    current_idx += 1

                table.add_row("", "", "")

            table.add_row(None, "[bold red](B)ack to Settings Menu[/bold red]")

            panel = Panel(
                table,
                title="[bold blue]Exemplar Scan Settings[/bold blue]",
                border_style="blue",
            )
            self.console.print(panel)

            valid_choices = [str(i) for i in range(1, len(field_map) + 1)] + ["b"]
            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=valid_choices,
                show_default=False,
            ).lower()

            if choice == "b":
                break

            # Handle field selection
            menu_idx = int(choice)
            if menu_idx in field_map:
                field_name, field_info = field_map[menu_idx]
                metadata = field_info["metadata"]

                # Check for special handler
                special_handler = metadata.get("special_handler")
                if special_handler:
                    handler_method = getattr(self, special_handler)
                    handler_method()
                else:
                    self._validate_and_update_field("exemplar", field_name, field_info)

    def _format_list_field_display(self, field_name: str, current_value: list) -> str:
        """Format display string for list-type fields in exemplar settings.

        Args:
            field_name: Name of the field
            current_value: Current list value

        Returns:
            Formatted display string for the menu
        """
        if field_name == "enabled_catalog_groups":
            all_groups = CatalogManager().get_all_group_names()
            if not current_value:
                return f"[green]All {len(all_groups)} enabled[/green]"
            return f"[cyan]{len(current_value)}/{len(all_groups)} enabled[/cyan]"

        if field_name == "excluded_file_types":
            if not current_value:
                return "[green]None[/green]"
            return f"[yellow]{', '.join(current_value)}[/yellow]"

        # Default list display
        if not current_value:
            return "[dim]empty[/dim]"
        return f"[cyan]{len(current_value)} items[/cyan]"

    def _edit_exemplar_epoch_min(self) -> None:
        """Wrapper to edit epoch_min field."""
        self._edit_exemplar_epoch("epoch_min")

    def _edit_exemplar_epoch_max(self) -> None:
        """Wrapper to edit epoch_max field."""
        self._edit_exemplar_epoch("epoch_max")

    def _edit_exemplar_epoch(self, field_name: str) -> None:
        """Edit epoch_min or epoch_max field.

        Args:
            field_name: Either 'epoch_min' or 'epoch_max'
        """
        cfg = self.config.exemplar
        current_value = getattr(cfg, field_name)
        label = "Epoch Minimum" if field_name == "epoch_min" else "Epoch Maximum"

        while True:
            new_value = Prompt.ask(
                f"\n[cyan]{label} (YYYY-MM-DD)[/cyan]\n"
                f"[dim]Current: {current_value}[/dim]\n"
                f"[dim]Enter new value (or 'b' to cancel)[/dim]",
                default=current_value,
            )

            if new_value.lower() == "b":
                return

            # Validate format
            is_valid, error_msg = validate_date_format(new_value)
            if not is_valid:
                self.console.print(f"[red]{error_msg}[/red]")
                continue

            # For epoch bounds, allow wider year range (2000-2050)
            try:
                parsed = datetime.strptime(new_value, "%Y-%m-%d")
                if parsed.year < 1970 or parsed.year > 2050:
                    self.console.print(f"[red]Year {parsed.year} is out of range (1970-2050)[/red]")
                    continue
            except ValueError:
                self.console.print("[red]Invalid date[/red]")
                continue

            # Cross-validate epoch_min < epoch_max
            if field_name == "epoch_min":
                is_valid, error_msg = validate_date_range(new_value, cfg.epoch_max)
            else:
                is_valid, error_msg = validate_date_range(cfg.epoch_min, new_value)

            if not is_valid:
                self.console.print(f"[red]{error_msg}[/red]")
                continue

            # All valid
            setattr(cfg, field_name, new_value)
            self.console.print(f"[green]✓ {label} updated to {new_value}[/green]")
            break

    def _edit_catalog_groups(self) -> None:
        """Edit enabled_catalog_groups with multi-column display."""
        cfg = self.config.exemplar
        catalog_mgr = CatalogManager()
        all_groups = catalog_mgr.get_all_group_names()

        if not all_groups:
            self.console.print("[yellow]No catalog groups found[/yellow]")
            Prompt.ask("\nPress Enter to continue")
            return

        # Working copy of enabled groups
        # Empty list means "all enabled"
        enabled_set = set(all_groups) if not cfg.enabled_catalog_groups else set(cfg.enabled_catalog_groups)

        while True:
            if self.proj_header:
                self.proj_header()

            # Build catalog table
            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("#", style="bold cyan", width=4)
            table.add_column("Group", width=20)
            table.add_column("#", style="bold cyan", width=4)
            table.add_column("Group", width=20)
            table.add_column("#", style="bold cyan", width=4)
            table.add_column("Group", width=20)

            # Split groups into 3 columns
            col_size = (len(all_groups) + 2) // 3
            for i in range(col_size):
                row_data = []
                for col in range(3):
                    idx = i + col * col_size
                    if idx < len(all_groups):
                        group = all_groups[idx]
                        num = str(idx + 1)
                        name = f"[green]{group}[/green]" if group in enabled_set else f"[dim]{group}[/dim]"
                        row_data.extend([num, name])
                    else:
                        row_data.extend(["", ""])
                table.add_row(*row_data)

            # Show summary
            enabled_count = len(enabled_set)
            total_count = len(all_groups)

            panel = Panel(
                table,
                title="[bold blue]Catalog Groups[/bold blue]",
                subtitle=f"[dim]{enabled_count}/{total_count} enabled | "
                "[green]green[/green]=enabled [dim]dim[/dim]=disabled[/dim]",
                border_style="blue",
            )
            self.console.print(panel)

            self.console.print(
                "\n[dim]Commands: (a) Select All | (n) Select None | (1-N) Toggle group | (b) Back[/dim]"
            )

            # Get input
            valid_nums = [str(i) for i in range(1, len(all_groups) + 1)]
            choice = Prompt.ask(
                "\n[bold cyan]Enter command[/bold cyan]",
                default="b",
            ).lower()

            if choice == "b":
                # Validate at least one group is selected
                if not enabled_set:
                    self.console.print(
                        "[yellow]At least one group must be selected. "
                        "Use (a) to select all or toggle individual groups.[/yellow]"
                    )
                    continue
                # Save changes - if all enabled, store empty list
                if enabled_set == set(all_groups):
                    cfg.enabled_catalog_groups = []
                else:
                    cfg.enabled_catalog_groups = sorted(enabled_set)
                break

            if choice == "a":
                # Select all
                enabled_set = set(all_groups)
                self.console.print("[green]✓ All groups enabled[/green]")

            elif choice == "n":
                # Select none - clears selection, user must select at least one before exiting
                enabled_set.clear()
                self.console.print("[dim]All groups cleared. Select at least one before exiting.[/dim]")

            elif choice in valid_nums:
                # Toggle specific group
                idx = int(choice) - 1
                group = all_groups[idx]
                if group in enabled_set:
                    enabled_set.remove(group)
                    self.console.print(f"[dim]Disabled: {group}[/dim]")
                else:
                    enabled_set.add(group)
                    self.console.print(f"[green]Enabled: {group}[/green]")

            else:
                self.console.print("[red]Invalid input[/red]")

    def _edit_excluded_file_types(self) -> None:
        """Edit excluded_file_types to skip certain file types during scan.

        Available file types from catalog: cache, log, keychain, database
        """
        cfg = self.config.exemplar
        # Known file types in the catalog
        all_file_types = ["cache", "log", "keychain", "database"]

        # Working copy of excluded types
        excluded_set = set(cfg.excluded_file_types) if cfg.excluded_file_types else set()

        while True:
            if self.proj_header:
                self.proj_header()

            # Build menu table
            table = Table(show_header=False, box=None, padding=(0, 2))
            table.add_column(style="bold cyan", width=4)
            table.add_column()

            table.add_row("", "[bold blue]File Type Exclusions[/bold blue]")
            table.add_row("", "[dim]Filter out general file types from scan[/dim]")
            table.add_row("", "")

            for idx, file_type in enumerate(all_file_types, start=1):
                status = "[yellow]EXCLUDED[/yellow]" if file_type in excluded_set else "[green]INCLUDED[/green]"
                filetype_colon = f"{file_type}:"
                padded_filetype = f"{filetype_colon: <9}"
                table.add_row(f"{idx}.", f"[bold]{padded_filetype}[/bold] {status}")

            table.add_row("", "")
            table.add_row("", "[dim](a) Include All | (n) Exclude All[/dim]")
            table.add_row("", "")
            table.add_row(None, "[bold red](B)ack to Exemplar Settings[/bold red]")

            panel = Panel(
                table,
                title="[bold blue]File Type Filtering[/bold blue]",
                subtitle="[dim]Toggle to exclude file types from scan[/dim]",
                border_style="blue",
            )
            self.console.print(panel)

            valid_choices = [str(i) for i in range(1, len(all_file_types) + 1)] + [
                "a",
                "n",
                "b",
            ]
            choice = Prompt.ask(
                "\n[bold cyan]Select option[/bold cyan]",
                choices=valid_choices,
                show_default=False,
            ).lower()

            if choice == "b":
                # Save changes
                cfg.excluded_file_types = sorted(excluded_set) if excluded_set else []
                break

            if choice == "a":
                # Include all (clear exclusions)
                excluded_set.clear()
                self.console.print("[green]✓ All file types will be included[/green]")

            elif choice == "n":
                # Exclude all
                excluded_set = set(all_file_types)
                self.console.print("[yellow]All file types excluded (no files will be scanned!)[/yellow]")

            elif choice.isdigit():
                # Toggle specific type
                idx = int(choice) - 1
                if 0 <= idx < len(all_file_types):
                    file_type = all_file_types[idx]
                    if file_type in excluded_set:
                        excluded_set.remove(file_type)
                        self.console.print(f"[green]Included: {file_type}[/green]")
                    else:
                        excluded_set.add(file_type)
                        self.console.print(f"[yellow]Excluded: {file_type}[/yellow]")

    # ========================================================================
    # Section: Module Management
    # ========================================================================

    def _module_settings(self) -> None:
        """Module management section (toggle active/inactive)."""
        while True:
            if self.proj_header:
                self.proj_header()

            # Discover modules
            try:
                modules = self.module_manager.discover_modules()
            except Exception as e:
                self.console.print(f"[red]Error discovering modules: {e}[/red]")
                Prompt.ask("\nPress Enter to continue")
                break

            if not modules:
                self.console.print("[yellow]No modules found[/yellow]")
                Prompt.ask("\nPress Enter to continue")
                break

            # Build module map
            module_map = {}
            for idx, module in enumerate(modules, start=1):
                module_map[idx] = module

            # Build menu table
            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("#", style="bold cyan", width=4)
            table.add_column("Module Name", style="bold")
            table.add_column("Status", width=12)
            table.add_column("Scan Type", width=20)
            table.add_column("Description")

            for idx, module in module_map.items():
                status = "[green]ACTIVE[/green]" if module.active else "[red]INACTIVE[/red]"
                scan_type = ", ".join(module.scan_type)
                description = module.description[:50] + ("..." if len(module.description) > 50 else "")

                table.add_row(
                    f"{idx}.",
                    module.name,
                    status,
                    scan_type,
                    f"[dim]{description}[/dim]",
                )

            panel = Panel(
                table,
                title="[bold magenta]Module Management[/bold magenta]",
                subtitle="[dim]Select module to toggle active/inactive[/dim]",
                border_style="magenta",
            )
            self.console.print(panel)

            self.console.print("\n[dim]Tip: Inactive modules will not run during scans[/dim]")

            valid_choices = [str(i) for i in range(1, len(module_map) + 1)] + ["b"]
            choice = Prompt.ask(
                "\n[bold cyan]Select module to toggle (or B to go back)[/bold cyan]",
                choices=valid_choices,
                show_default=False,
            ).lower()

            if choice == "b":
                break

            # Toggle selected module
            module_idx = int(choice)
            if module_idx in module_map:
                module = module_map[module_idx]
                new_state = not module.active
                self._toggle_module_active(module, new_state)

    def _toggle_module_active(self, module: ModuleConfig, new_state: bool) -> None:
        """Toggle module active state and persist to YAML.

        Args:
            module: Module to toggle
            new_state: New active state (True/False)
        """
        try:
            # Load YAML
            with module.yaml_path.open() as f:
                data = yaml.safe_load(f)

            # Update active flag
            if "module_info" not in data:
                raise ValueError("Invalid YAML structure: missing module_info")

            data["module_info"]["active"] = new_state

            # Write back to YAML
            with module.yaml_path.open("w") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

            # Update in-memory object
            module.active = new_state

            # Clear module cache to force rediscovery
            self.module_manager._modules_cache = None

            state_str = "ACTIVE" if new_state else "INACTIVE"
            self.console.print(f"\n[green]✓ Module '{module.name}' set to {state_str}[/green]")

        except Exception as e:
            self.console.print(f"\n[red]Failed to toggle module: {e}[/red]\n[dim]YAML path: {module.yaml_path}[/dim]")
            Prompt.ask("\nPress Enter to continue")

    # ========================================================================
    # Field Update Helpers
    # ========================================================================

    def _validate_and_update_field(self, section_name: str, field_name: str, field_info: dict) -> None:
        """Validate and update a config field.

        Args:
            section_name: Config section name (e.g., 'carver', 'ui')
            field_name: Field name within section
            field_info: Field metadata dictionary
        """
        section_obj = getattr(self.config, section_name)
        current_value = getattr(section_obj, field_name)
        metadata = field_info["metadata"]

        if isinstance(current_value, bool):
            # Toggle boolean
            self._toggle_boolean_field(section_name, field_name)

        elif isinstance(current_value, str):
            # Handle string field
            description = metadata.get("description", field_name)
            choices = metadata.get("choices")
            validation_pattern = metadata.get("validation")

            if choices:
                # Cycle through choices with single key press
                self._cycle_choices_field(section_name, field_name, choices)

            else:
                # Free text with optional validation
                while True:
                    new_value = Prompt.ask(
                        f"\n[cyan]{description}[/cyan]\n"
                        f"[dim]Current: {current_value}[/dim]\n"
                        f"[dim]Enter new value (or 'b' to cancel)[/dim]",
                        default=current_value,
                    ).lower()

                    if new_value == "b":
                        return

                    # Sanitize
                    new_value = sanitize_text(new_value)
                    if not new_value:
                        self.console.print("[red]Invalid input (contains special characters)[/red]")
                        continue

                    # Validate date format if applicable
                    date_pattern = "\\d{4}-\\d{2}-\\d{2}"
                    if validation_pattern and date_pattern in validation_pattern:
                        is_valid, error_msg = validate_date_format(new_value)
                        if not is_valid:
                            self.console.print(f"[red]{error_msg}[/red]")
                            continue

                        # Check date range if this is ts_start or ts_end
                        if field_name == "ts_end":
                            ts_start = self.config.carver.ts_start
                            is_valid, error_msg = validate_date_range(ts_start, new_value)
                            if not is_valid:
                                self.console.print(f"[red]{error_msg}[/red]")
                                continue

                        elif field_name == "ts_start":
                            ts_end = self.config.carver.ts_end
                            is_valid, error_msg = validate_date_range(new_value, ts_end)
                            if not is_valid:
                                self.console.print(f"[red]{error_msg}[/red]")
                                continue

                    # All validation passed
                    setattr(section_obj, field_name, new_value)
                    break

        elif isinstance(current_value, int):
            # Handle integer field with min/max validation
            label = metadata.get("label", field_name)
            description = metadata.get("description", field_name)
            min_val = metadata.get("min")
            max_val = metadata.get("max")

            range_hint = ""
            if min_val is not None and max_val is not None:
                range_hint = f" ({min_val}-{max_val})"

            while True:
                new_value = Prompt.ask(
                    f"\n[cyan]{description}[/cyan]\n"
                    f"[dim]Current: {current_value}[/dim]\n"
                    f"[dim]Enter value{range_hint} (or 'b' to cancel)[/dim]",
                    default=str(current_value),
                )

                if new_value.lower() == "b":
                    return

                try:
                    int_value = int(new_value)

                    # Validate range if min/max provided
                    if min_val is not None and int_value < min_val:
                        self.console.print(f"[red]Value must be at least {min_val}[/red]")
                        continue
                    if max_val is not None and int_value > max_val:
                        self.console.print(f"[red]Value must be at most {max_val}[/red]")
                        continue

                    setattr(section_obj, field_name, int_value)
                    self.console.print(f"[green]{label} updated to {int_value}[/green]")
                    break

                except ValueError:
                    self.console.print("[red]Please enter a valid integer[/red]")
                    continue

    def _toggle_boolean_field(self, section_name: str, field_name: str) -> None:
        """Toggle a boolean field.

        Args:
            section_name: Config section name
            field_name: Field name within section
        """
        section_obj = getattr(self.config, section_name)
        current_value = getattr(section_obj, field_name)
        new_value = not current_value

        # Special handling for debug mode
        if section_name == "ui" and field_name == "debug":
            # Prevent turning OFF debug mode while Save Debug to File is ON
            if not new_value and self.config.ui.debug_log_to_file:
                self.console.print(
                    "\n[yellow]Cannot disable Debug Mode while 'Save Debug to File' is enabled.[/yellow]"
                )
                self.console.print("[dim]Turn off 'Save Debug to File' first.[/dim]")
                Prompt.ask("\nPress Enter to continue")
                return

            setattr(section_obj, field_name, new_value)
            logger.configure(self.config, console=self.console, project_dir=self.project_dir)
            # Debug mode and progress bars are mutually exclusive
            # When debug is ON, progress bars OFF; when debug is OFF, progress bars ON
            self.config.ui.show_progress_bars = not new_value

        # Special handling for debug log file toggle
        elif section_name == "ui" and field_name == "debug_log_to_file":
            setattr(section_obj, field_name, new_value)
            # When enabling Save Debug to File, also enable Debug Mode
            if new_value:
                self.config.ui.debug = True
                # Debug mode and progress bars are mutually exclusive
                self.config.ui.show_progress_bars = False
            logger.configure(self.config, console=self.console, project_dir=self.project_dir)

        else:
            # Default toggle for other boolean fields
            setattr(section_obj, field_name, new_value)

    def _cycle_choices_field(self, section_name: str, field_name: str, choices: list[str]) -> None:
        """Cycle through predefined choices for a string field.

        Args:
            section_name: Config section name
            field_name: Field name within section
            choices: List of valid choices to cycle through
        """
        section_obj = getattr(self.config, section_name)
        current_value = getattr(section_obj, field_name)

        # Find current index and cycle to next
        try:
            current_idx = choices.index(current_value)
            next_idx = (current_idx + 1) % len(choices)
        except ValueError:
            # Current value not in choices, start at first choice
            next_idx = 0

        new_value = choices[next_idx]
        setattr(section_obj, field_name, new_value)

    # ========================================================================
    # Save Preferences
    # ========================================================================

    def _confirm_discard_changes(self) -> bool:
        """Prompt user about unsaved changes.

        Returns:
            True if user wants to exit (saved or discarded), False to stay.
        """
        self.console.print("\n[yellow]You have unsaved changes.[/yellow]")
        choice = Prompt.ask(
            "[bold]Save before exiting?[/bold]",
            choices=["y", "n", "c"],
            default="c",
        ).lower()

        if choice == "y":
            # Try to save - if successful, exit; if failed, stay in menu
            return bool(self._save_preferences())
        # "n" = discard and exit, "c" = cancel (stay in settings)
        return choice == "n"

    def _save_preferences(self) -> bool | None:
        """Save current configuration to project config file."""
        if self.project_dir is None:
            self.console.print(
                "\n[yellow]No project loaded - cannot save settings[/yellow]\n"
                "[dim]Settings are per-case and stored in the case directory.[/dim]\n"
                "[dim]Create or open a project first.[/dim]"
            )
            Prompt.ask("\nPress Enter to continue")
            return False

        try:
            ConfigLoader.save_project_config(self.config, self.project_dir)
            config_file = self.project_dir / ".marsproj"
            self.console.print(f"\n[green]✓ Configuration saved to project[/green]\n[dim]{config_file}[/dim]\n")
            sleep(0.8)
            # Reset snapshot so we don't prompt again
            self._original_config = copy.deepcopy(self.config.to_user_configurable_dict())
            return True
        except Exception as e:
            self.console.print(f"\n[red]Failed to save configuration: {e}[/red]")
            Prompt.ask("\nPress Enter to continue")
