"""ARC Manager UI module for viewing and editing the artifact recovery catalog.

Provides interactive navigation for groups, targets, and field editing,
plus a wizard for creating new entries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from mars.catalog.catalog_editor import CatalogEditor
from mars.catalog.glob_validator import (
    generate_exemplar_pattern,
    has_extra_wildcards,
    infer_file_type_from_glob,
    suggest_multi_profile,
    user_glob_parser,
    validate_entry,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rich.console import Console

# prompt_toolkit for better text input handling (arrow keys, backspace on wrapped lines)
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

# Shared session for text inputs (preserves history across prompts)
_text_prompt_session: PromptSession | None = None


def get_text_input(prompt_text: str, default: str = "") -> str:
    """Get text input with proper line editing support.

    Uses prompt_toolkit for better handling of long text that wraps
    across multiple lines, with proper backspace and arrow key support.

    Args:
        prompt_text: The prompt to display.
        default: Default value to pre-fill.

    Returns:
        The user's input, stripped of whitespace.
    """
    global _text_prompt_session
    if _text_prompt_session is None:
        _text_prompt_session = PromptSession(history=InMemoryHistory())

    try:
        result: str = _text_prompt_session.prompt(f"{prompt_text}: ", default=default)  # type: ignore[union-attr]
        return result.strip()
    except (EOFError, KeyboardInterrupt):
        return default


# Field definitions for display and editing
FIELD_DEFINITIONS = {
    "name": {"label": "Name", "type": "text", "required": True},
    "description": {"label": "Description", "type": "text", "required": False},
    "notes": {"label": "Notes", "type": "text", "required": False},
    "glob_pattern": {"label": "Glob Pattern", "type": "text", "required": False},
    "scope": {
        "label": "Scope",
        "type": "choice",
        "choices": ["user", "system"],
        "required": True,
    },
    "file_type": {
        "label": "File Type",
        "type": "choice",
        "choices": ["database", "log", "cache"],
        "required": False,
        "default": "database",
    },
    "exemplar_pattern": {"label": "Exemplar Pattern", "type": "text", "required": True},
    "multi_profile": {"label": "Multi-Profile", "type": "bool", "required": False},
    "has_archives": {"label": "Has Archives", "type": "bool", "required": False},
    "preserve_structure": {
        "label": "Preserve Structure",
        "type": "bool",
        "required": False,
    },
    "ignorable_tables": {
        "label": "Ignorable Tables",
        "type": "list",
        "required": False,
    },
    "combine_strategy": {
        "label": "Combine Strategy",
        "type": "choice",
        "choices": [
            "decompress_and_merge",
            "decompress_and_concatenate",
            "decompress_only",
        ],
        "required": False,
    },
}


@dataclass
class NavigationState:
    """Tracks navigation history for back button support."""

    history: list[str] = field(default_factory=list)
    current_group: str | None = None
    current_target_index: int | None = None

    def push(self, location: str) -> None:
        """Push a location onto the history stack."""
        self.history.append(location)

    def pop(self) -> str | None:
        """Pop and return the last location, or None if empty."""
        if self.history:
            return self.history.pop()
        return None


class ARCManagerUI:
    """Interactive UI for managing the Artifact Recovery Catalog."""

    def __init__(
        self,
        console: Console,
        catalog_path: Path | None = None,
    ):
        """Initialize the ARC Manager UI.

        Args:
            console: Rich console for output.
            catalog_path: Optional path to catalog file. Uses default if None.
        """
        self.console = console
        self.editor = CatalogEditor(catalog_path)
        self.nav = NavigationState()

    def show_menu(
        self,
        show_header_callback: Callable[[], None] | None = None,
    ) -> None:
        """Show the main ARC Manager menu.

        Args:
            show_header_callback: Optional callback to display persistent header.
        """
        # Load catalog
        if not self.editor.load():
            self.console.print("[red]Failed to load catalog. Check file exists and is valid YAML.[/red]")
            Prompt.ask("\nPress Enter to continue")
            return

        while True:
            self._show_group_list(show_header_callback)

            # Prompt to save if there are unsaved changes
            if self.editor.has_unsaved_changes:
                choice = Prompt.ask(
                    "\n[yellow]You have unsaved changes. Save before exiting?[/yellow] (y/n/c)",
                    choices=["y", "n", "c"],
                    default="y",
                ).lower()

                if choice == "c":
                    # Cancel - stay in manager
                    continue
                if choice == "y":
                    if self.editor.save():
                        self.console.print("[green]Catalog saved successfully.[/green]")
                    else:
                        self.console.print("[red]Failed to save catalog.[/red]")
            # Exit the manager
            break

    def _show_group_list(
        self,
        show_header_callback: Callable[[], None] | None = None,
    ) -> None:
        """Display the list of artifact groups."""
        while True:
            if show_header_callback:
                show_header_callback()

            groups = self.editor.get_groups()
            if not groups:
                self.console.print("[yellow]No artifact groups found.[/yellow]")
                Prompt.ask("\nPress Enter to continue")
                return

            # Build multi-column table (3 columns)
            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("#", style="bold cyan", width=4)
            table.add_column("Group", width=23)
            table.add_column("#", style="bold cyan", width=4)
            table.add_column("Group", width=23)
            table.add_column("#", style="bold cyan", width=4)
            table.add_column("Group", width=23)

            # Split groups into 3 columns
            col_size = (len(groups) + 2) // 3
            for i in range(col_size):
                row_data: list[str] = []
                for col in range(3):
                    idx = i + col * col_size
                    if idx < len(groups):
                        group = groups[idx]
                        count = self.editor.get_target_count(group)
                        num = str(idx + 1)
                        # Show target count in parentheses
                        display = f"{group} ({count})"
                        row_data.extend([num, display])
                    else:
                        row_data.extend(["", ""])
                table.add_row(*row_data)

            # Unsaved indicator
            unsaved_indicator = "[yellow]*[/yellow] " if self.editor.has_unsaved_changes else ""

            panel = Panel(
                table,
                title=f"{unsaved_indicator}[bold indian_red]Artifact Recovery Catalog[/bold indian_red]",
                subtitle=f"[dim]{len(groups)} groups[/dim]",
                border_style="indian_red",
            )
            self.console.print(panel)

            self.console.print(
                "\n[dim]Commands: (1-N) Select group | (n) New group | (d) Delete group | (s) Save | (b) Back[/dim]"
            )

            valid_nums = [str(i) for i in range(1, len(groups) + 1)]
            choice = (
                Prompt.ask(
                    "\n[bold cyan]Enter command[/bold cyan]",
                    default="b",
                )
                .lower()
                .strip()
            )

            if choice == "b":
                return

            if choice == "s":
                if self.editor.save():
                    self.console.print("[green]Catalog saved successfully.[/green]")
                else:
                    self.console.print("[red]Failed to save catalog.[/red]")
                continue

            if choice == "n":
                self._add_new_group(show_header_callback)
                continue

            if choice == "d":
                self._delete_group_prompt(groups)
                continue

            if choice in valid_nums:
                idx = int(choice) - 1
                group = groups[idx]
                self.nav.current_group = group
                self._show_target_list(group, show_header_callback)
                continue

            self.console.print("[red]Invalid input.[/red]")

    def _add_new_group(self, show_header_callback: Callable[[], None] | None = None) -> None:
        """Add a new artifact group and auto-enter it."""
        self.console.print("\n[bold]Add New Group[/bold]")
        name = Prompt.ask("Enter group name (lowercase, no spaces)", default="").strip()

        if not name:
            self.console.print("[yellow]Cancelled.[/yellow]")
            return

        # Validate: lowercase, no spaces, alphanumeric + underscore
        if not name.replace("_", "").isalnum() or " " in name:
            self.console.print("[red]Group name must be lowercase alphanumeric with underscores only.[/red]")
            return

        name = name.lower()

        if self.editor.add_group(name):
            self.console.print(f"[green]Group '{name}' created.[/green]")
            # Auto-enter the new group
            self.nav.current_group = name
            self._show_target_list(name, show_header_callback)
        else:
            self.console.print(f"[red]Group '{name}' already exists.[/red]")

    def _delete_group_prompt(self, groups: list[str]) -> None:
        """Prompt to delete a group and all its targets."""
        if not groups:
            self.console.print("[yellow]No groups to delete.[/yellow]")
            return

        del_choice = Prompt.ask("Enter group number to delete (or 'c' to cancel)", default="c").strip()
        if del_choice.lower() == "c":
            return

        try:
            idx = int(del_choice) - 1
            if 0 <= idx < len(groups):
                group = groups[idx]
                target_count = self.editor.get_target_count(group)
                self.console.print(
                    f"[yellow]Warning: This will delete '{group}' and all {target_count} target(s).[/yellow]"
                )
                if Confirm.ask("Are you sure?", default=False):
                    if self.editor.delete_group(group):
                        self.console.print(f"[green]Deleted group '{group}'.[/green]")
                    else:
                        self.console.print("[red]Failed to delete group.[/red]")
            else:
                self.console.print("[red]Invalid group number.[/red]")
        except ValueError:
            self.console.print("[red]Invalid input.[/red]")

    def _show_target_list(
        self,
        group: str,
        show_header_callback: Callable[[], None] | None = None,
    ) -> None:
        """Display the list of targets in a group."""
        while True:
            if show_header_callback:
                show_header_callback()

            targets = self.editor.get_targets(group)

            # Build target table
            table = Table(show_header=True, box=None, padding=(0, 1))
            table.add_column("#", style="bold cyan", width=4)
            table.add_column("Name", width=35)
            table.add_column("Type", width=10)
            table.add_column("Scope", width=8)
            table.add_column("Flags", width=8)

            for idx, target in enumerate(targets, start=1):
                name = target.get("name", "(unnamed)")
                file_type = target.get("file_type", "database")
                scope = target.get("scope", "system")

                # Build flags string
                flags = []
                if target.get("has_archives"):
                    flags.append("[blue]A[/blue]")
                if target.get("multi_profile"):
                    flags.append("[magenta]M[/magenta]")
                if target.get("preserve_structure"):
                    flags.append("[cyan]P[/cyan]")
                flags_str = " ".join(flags) if flags else ""

                table.add_row(str(idx), name, file_type, scope, flags_str)

            # Unsaved indicator
            unsaved_indicator = "[yellow]*[/yellow] " if self.editor.has_unsaved_changes else ""

            panel = Panel(
                table,
                title=f"{unsaved_indicator}[bold blue]{group}[/bold blue]",
                subtitle="[dim]A: has-archives M: multi-profile P: preserve-structure[/dim]",
                border_style="blue",
            )
            self.console.print(panel)
            self.console.print("")

            if targets:
                self.console.print("To [bold blue]edit[/bold blue] a target, select its number.\n")
            self.console.print("[bold dark_sea_green4]To add a new target:[/bold dark_sea_green4]")
            self.console.print("  - Enter 'n' to [bold blue]directly add[/bold blue] fields, [italic]or[/italic]")
            self.console.print("  - Enter 'w' to run the [bold blue]ARC Wizard[/bold blue] for guided assistance\n")

            self.console.print(
                "\n[dim]Commands: (1-N) Edit target | (n) New target | (w) Wizard | (d) Delete target | (b) Back[/dim]"
            )

            valid_nums = [str(i) for i in range(1, len(targets) + 1)]
            choice = (
                Prompt.ask(
                    "\n[bold cyan]Enter command[/bold cyan]",
                    default="b",
                )
                .lower()
                .strip()
            )

            if choice == "b":
                self.nav.current_group = None
                return

            if choice == "n":
                self._add_blank_target(group, show_header_callback)
                continue

            if choice == "w":
                wizard = ARCWizard(self.console, self.editor, group, show_header_callback)
                wizard.run()
                continue

            if choice == "d":
                self._delete_target_prompt(group, targets)
                continue

            if choice in valid_nums:
                idx = int(choice) - 1
                self.nav.current_target_index = idx
                self._show_field_editor(group, idx, show_header_callback)
                continue

            self.console.print("[red]Invalid input.[/red]")

    def _add_blank_target(
        self,
        group: str,
        show_header_callback: Callable[[], None] | None = None,
    ) -> None:
        """Add a new blank target to a group."""
        name = get_text_input("Enter target name", default="")
        if not name or name.lower() == "c":
            self.console.print("[yellow]Cancelled.[/yellow]")
            return

        # Create minimal target - only name and file_type to start
        target = {
            "name": name,
            "file_type": "database",
            "scope": "system",
            "glob_pattern": "",
            "description": "",
            "exemplar_pattern": generate_exemplar_pattern(name, "database"),
        }

        if self.editor.add_target(group, target):
            self.console.print(f"[green]Target '{name}' added.[/green]")
            # Auto-open editor for the new target
            new_index = self.editor.get_target_count(group) - 1
            deleted = self._show_field_editor(group, new_index, show_header_callback, is_new_target=True)
            if deleted:
                self.console.print(f"[yellow]Target '{name}' was cancelled and removed.[/yellow]")
        else:
            self.console.print("[red]Failed to add target.[/red]")

    def _delete_target_prompt(self, group: str, targets: list[dict]) -> None:
        """Prompt to delete a target."""
        if not targets:
            self.console.print("[yellow]No targets to delete.[/yellow]")
            return

        choice = Prompt.ask(
            "Enter target number to delete (or 'c' to cancel)",
            default="c",
        ).strip()

        if choice.lower() == "c":
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(targets):
                target_name = targets[idx].get("name", "(unnamed)")
                if Confirm.ask(f"Delete '{target_name}'?", default=False):
                    if self.editor.delete_target(group, idx):
                        self.console.print(f"[green]Deleted '{target_name}'.[/green]")
                    else:
                        self.console.print("[red]Failed to delete target.[/red]")
            else:
                self.console.print("[red]Invalid target number.[/red]")
        except ValueError:
            self.console.print("[red]Invalid input.[/red]")

    def _get_available_fields(self, target: dict) -> set[str]:
        """Determine which fields are editable based on current target state.

        Progressive unlocking rules:
        - Always available: name, glob_pattern, description, notes
        - After glob set: file_type
        - After both glob and file_type: exemplar_pattern, has_archives, scope (read-only)
        - Database only: ignorable_tables, multi_profile (if has extra wildcards)
        - If glob ends with **/*: preserve_structure
        - If has_archives=true: combine_strategy, primary, archives
        """
        # Always available
        available = {"name", "glob_pattern", "description", "notes"}

        glob_pattern = target.get("glob_pattern") or (target.get("primary", {}) or {}).get("glob_pattern")
        file_type = target.get("file_type")
        scope = target.get("scope")
        has_archives = target.get("has_archives", False)

        # File type unlocked after glob is set
        if glob_pattern:
            available.add("file_type")

        # Rest unlocked after both glob and file_type are set
        if glob_pattern and file_type:
            available.add("exemplar_pattern")
            available.add("has_archives")
            available.add("scope")  # Read-only, auto-set from glob

            if file_type == "database":
                available.add("ignorable_tables")
                # Multi-profile only if user scope and has extra wildcards
                if scope == "user" and has_extra_wildcards(glob_pattern):
                    available.add("multi_profile")

            if glob_pattern.endswith("**/*"):
                available.add("preserve_structure")

            if has_archives:
                available.add("combine_strategy")
                available.add("primary.glob_pattern")
                available.add("archives")

        return available

    def _show_field_editor(
        self,
        group: str,
        target_index: int,
        show_header_callback: Callable[[], None] | None = None,
        is_new_target: bool = False,
    ) -> bool:
        """Show the field editor for a specific target.

        Args:
            group: The group name.
            target_index: Index of target within the group.
            show_header_callback: Optional callback for header display.
            is_new_target: If True, show (c) Cancel option to delete this target.

        Returns:
            True if target was deleted (cancelled), False otherwise.
        """
        while True:
            if show_header_callback:
                show_header_callback()

            target = self.editor.get_target(group, target_index)
            if target is None:
                self.console.print("[red]Target not found.[/red]")
                return False

            # Determine available fields
            available_fields = self._get_available_fields(target)

            # Build field table
            table = Table(show_header=True, box=None, padding=(0, 1), expand=True)
            table.add_column("#", style="bold cyan", width=3)
            table.add_column("Field", width=20)
            table.add_column("Value", overflow="ellipsis")

            # Standard fields in display order
            display_order = [
                "name",
                "description",
                "notes",
                "glob_pattern",
                "file_type",
                "scope",
                "exemplar_pattern",
                "multi_profile",
                "has_archives",
                "preserve_structure",
                "ignorable_tables",
                "combine_strategy",
            ]

            # Show guidance message based on state
            glob_pattern = target.get("glob_pattern") or (target.get("primary", {}) or {}).get("glob_pattern")
            file_type = target.get("file_type")
            if not glob_pattern:
                self.console.print("[yellow]→ Set 'Glob Pattern' to unlock file type and other fields.[/yellow]\n")
            elif not file_type:
                self.console.print("[yellow]→ Set 'File Type' to unlock remaining fields.[/yellow]\n")

            field_list: list[str] = []
            for field_name in display_order:
                if field_name in target or field_name in FIELD_DEFINITIONS:
                    field_list.append(field_name)
                    value = target.get(field_name)
                    # For glob_pattern, also check primary.glob_pattern
                    if field_name == "glob_pattern" and value is None:
                        value = (target.get("primary", {}) or {}).get("glob_pattern")
                    field_def = FIELD_DEFINITIONS.get(field_name, {})
                    label = field_def.get("label", field_name)

                    is_available = field_name in available_fields

                    # Format value for display
                    if value is None:
                        display_value = "[dim](not set)[/dim]"
                    elif isinstance(value, bool):
                        display_value = "[green]Yes[/green]" if value else "[red]No[/red]"
                    elif isinstance(value, list):
                        display_value = ", ".join(str(v) for v in value) if value else "[dim](empty)[/dim]"
                    else:
                        display_value = str(value) if value else "[dim](empty)[/dim]"

                    # Dim unavailable fields
                    if is_available:
                        table.add_row(str(len(field_list)), label, display_value)
                    else:
                        table.add_row(
                            f"[dim]{len(field_list)}[/dim]",
                            f"[dim]{label}[/dim]",
                            f"[dim]{display_value}[/dim]",
                        )

            # Handle has_archives special fields (primary, archives)
            if target.get("has_archives"):
                primary = target.get("primary", {})
                if primary:
                    field_list.append("primary.glob_pattern")
                    table.add_row(
                        str(len(field_list)),
                        "Primary Glob",
                        primary.get("glob_pattern", "[dim](not set)[/dim]"),
                    )

                archives = target.get("archives", [])
                for i, archive in enumerate(archives):
                    field_list.append(f"archives.{i}")
                    archive_desc = (
                        f"{archive.get('name', 'Archive')}: "
                        f"subpath='{archive.get('subpath', '')}' "
                        f"pattern='{archive.get('pattern', '*')}'"
                    )
                    table.add_row(str(len(field_list)), f"Archive {i + 1}", archive_desc)

            # Unsaved indicator
            unsaved_indicator = "[yellow]*[/yellow] " if self.editor.has_unsaved_changes else ""
            target_name = target.get("name", "(unnamed)")

            panel = Panel(
                table,
                title=f"{unsaved_indicator}[bold green]{target_name}[/bold green]",
                border_style="green",
            )
            self.console.print(panel)

            # Show commands - include (c) Cancel for new targets
            if is_new_target:
                self.console.print(
                    "\n[dim]Commands: (1-N) Edit field | (s) Save | (v) Validate | (c) Cancel | (b) Back[/dim]"
                )
            else:
                self.console.print("\n[dim]Commands: (1-N) Edit field | (s) Save | (v) Validate | (b) Back[/dim]")

            valid_nums = [str(i) for i in range(1, len(field_list) + 1)]
            choice = (
                Prompt.ask(
                    "\n[bold cyan]Enter command[/bold cyan]",
                    default="b",
                )
                .lower()
                .strip()
            )

            if choice == "b":
                self.nav.current_target_index = None
                return False

            if choice == "c" and is_new_target:
                # Cancel - delete this new target
                if Confirm.ask("Delete this incomplete target?", default=True):
                    self.editor.delete_target(group, target_index)
                    self.nav.current_target_index = None
                    return True
                continue

            if choice == "s":
                if self.editor.save():
                    self.console.print("[green]Catalog saved successfully.[/green]")
                else:
                    self.console.print("[red]Failed to save catalog.[/red]")
                continue

            if choice == "v":
                self._validate_target(target)
                continue

            if choice in valid_nums:
                idx = int(choice) - 1
                field_name = field_list[idx]
                # Check if field is available for editing
                if field_name not in available_fields and not field_name.startswith("archives."):
                    self.console.print(f"[yellow]'{field_name}' is locked. Set prerequisites first.[/yellow]")
                    continue
                self._edit_field(group, target_index, field_name, target)
                continue

            self.console.print("[red]Invalid input.[/red]")

    def _validate_target(self, target: dict) -> None:
        """Validate a target and display results."""
        is_valid, errors = validate_entry(target)
        if is_valid:
            self.console.print("[green]Target is valid.[/green]")
        else:
            self.console.print("[red]Validation errors:[/red]")
            for error in errors:
                self.console.print(f"  [red]- {error}[/red]")
        Prompt.ask("\nPress Enter to continue")

    def _edit_field(
        self,
        group: str,
        target_index: int,
        field_name: str,
        target: dict,
    ) -> None:
        """Edit a specific field of a target."""
        # Handle special compound fields
        if field_name == "primary.glob_pattern":
            self._edit_primary_glob(group, target_index, target)
            return

        if field_name.startswith("archives."):
            archive_idx = int(field_name.split(".")[1])
            self._edit_archive(group, target_index, target, archive_idx)
            return

        # Scope is auto-set from glob, show as read-only
        if field_name == "scope":
            self.console.print(
                f"[yellow]Scope is auto-detected from glob pattern: {target.get('scope', 'system')}[/yellow]"
            )
            return

        field_def = FIELD_DEFINITIONS.get(field_name, {"type": "text"})
        field_type = field_def.get("type", "text")
        label = field_def.get("label", field_name)
        current_value = target.get(field_name)

        if field_type == "text":
            # Use prompt_toolkit for text fields (better line editing for long text)
            if field_name in ("glob_pattern", "name", "description", "notes"):
                new_value = get_text_input(
                    f"{label} (current: {current_value or '(empty)'})",
                    default=str(current_value) if current_value else "",
                )
            else:
                new_value = Prompt.ask(
                    f"\n[cyan]{label}[/cyan] (current: {current_value or '(empty)'})",
                    default=str(current_value) if current_value else "",
                ).strip()

            if new_value or new_value == "":
                # Special handling for glob_pattern - validate, auto-set scope, and infer file_type
                if field_name == "glob_pattern":
                    file_type = target.get("file_type", "database")
                    result = user_glob_parser(new_value, file_type)
                    if not result.is_valid:
                        self.console.print("[red]Validation errors:[/red]")
                        for error in result.errors:
                            self.console.print(f"  [red]- {error}[/red]")
                        if not Confirm.ask("Save anyway?", default=False):
                            return
                    # Auto-set scope from glob
                    self.editor.update_target_field(group, target_index, "scope", result.scope)
                    self.console.print(f"[dim]Scope auto-set to: {result.scope}[/dim]")

                    # Infer file_type from glob pattern
                    suggested_type, available_choices = infer_file_type_from_glob(new_value)
                    current_file_type = target.get("file_type")

                    if suggested_type and suggested_type != current_file_type:
                        # Suggest updating file_type
                        if Confirm.ask(
                            f"Detected file type '[cyan]{suggested_type}[/cyan]' from pattern. Update?",
                            default=True,
                        ):
                            self.editor.update_target_field(group, target_index, "file_type", suggested_type)
                            self.console.print(f"[dim]File type set to: {suggested_type}[/dim]")
                    elif current_file_type and current_file_type not in available_choices:
                        # Current file_type is not valid for this glob pattern
                        self.console.print(
                            f"[yellow]Warning: Current file type '{current_file_type}' "
                            f"may not be appropriate for this pattern.[/yellow]"
                        )
                        self.console.print(f"[dim]Available types: {', '.join(available_choices)}[/dim]")

                self.editor.update_target_field(group, target_index, field_name, new_value)
                self.console.print(f"[green]{label} updated.[/green]")

        elif field_type == "bool":
            # Toggle boolean
            new_value = not bool(current_value)
            self.editor.update_target_field(group, target_index, field_name, new_value)
            status = "enabled" if new_value else "disabled"
            self.console.print(f"[green]{label} {status}.[/green]")

        elif field_type == "choice":
            choices = field_def.get("choices", [])
            self.console.print(f"\n[cyan]{label}[/cyan] options:")
            for i, opt in enumerate(choices, start=1):
                marker = "[green]*[/green]" if opt == current_value else " "
                self.console.print(f"  {marker} {i}. {opt}")

            choice = Prompt.ask("Select option number", default="").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(choices):
                    new_value = choices[idx]
                    self.editor.update_target_field(group, target_index, field_name, new_value)
                    self.console.print(f"[green]{label} set to '{new_value}'.[/green]")
            except (ValueError, IndexError):
                self.console.print("[yellow]Cancelled.[/yellow]")

        elif field_type == "list":
            self._edit_list_field(group, target_index, field_name, target)

    def _edit_list_field(
        self,
        group: str,
        target_index: int,
        field_name: str,
        target: dict,
    ) -> None:
        """Edit a list field (like ignorable_tables)."""
        current = target.get(field_name, [])
        if not isinstance(current, list):
            current = []

        label = FIELD_DEFINITIONS.get(field_name, {}).get("label", field_name)
        self.console.print(f"\n[cyan]{label}[/cyan]")

        if current:
            self.console.print("Current values:")
            for i, item in enumerate(current, start=1):
                self.console.print(f"  {i}. {item}")
        else:
            self.console.print("  [dim](empty)[/dim]")

        self.console.print("\n[dim]Commands: (a) Add item | (d) Delete item | (c) Clear all | (b) Back[/dim]")

        choice = Prompt.ask("Enter command", default="b").lower().strip()

        if choice == "a":
            new_item = Prompt.ask("Enter new value", default="").strip()
            if new_item and new_item not in current:
                current.append(new_item)
                self.editor.update_target_field(group, target_index, field_name, current)
                self.console.print(f"[green]Added '{new_item}'.[/green]")

        elif choice == "d":
            if not current:
                self.console.print("[yellow]No items to delete.[/yellow]")
                return
            # Show numbered list again for clarity
            self.console.print("\nSelect item to delete:")
            for i, item in enumerate(current, start=1):
                self.console.print(f"  {i}. {item}")

            del_choice = Prompt.ask("Enter number to delete (or 'c' to cancel)", default="c").strip()
            if del_choice.lower() == "c":
                return

            try:
                idx = int(del_choice) - 1
                if 0 <= idx < len(current):
                    removed = current.pop(idx)
                    self.editor.update_target_field(group, target_index, field_name, current)
                    self.console.print(f"[green]Removed '{removed}'.[/green]")
                else:
                    self.console.print("[red]Invalid number.[/red]")
            except ValueError:
                self.console.print("[red]Enter a number or 'c' to cancel.[/red]")

        elif choice == "c":
            if Confirm.ask("Clear all values?", default=False):
                self.editor.update_target_field(group, target_index, field_name, [])
                self.console.print("[green]Cleared.[/green]")

    def _edit_primary_glob(
        self,
        group: str,
        target_index: int,
        target: dict,
    ) -> None:
        """Edit the primary.glob_pattern field."""
        primary = target.get("primary", {})
        current = primary.get("glob_pattern", "")

        new_value = Prompt.ask(
            f"\n[cyan]Primary Glob Pattern[/cyan] (current: {current or '(empty)'})",
            default=current,
        ).strip()

        if new_value:
            if "primary" not in target:
                target["primary"] = {}
            target["primary"]["glob_pattern"] = new_value
            self.editor.update_target(group, target_index, target)
            self.console.print("[green]Primary glob pattern updated.[/green]")

    def _edit_archive(
        self,
        group: str,
        target_index: int,
        target: dict,
        archive_idx: int,
    ) -> None:
        """Edit an archive entry."""
        archives = target.get("archives", [])
        if not (0 <= archive_idx < len(archives)):
            self.console.print("[red]Archive not found.[/red]")
            return

        archive = archives[archive_idx]

        self.console.print(f"\n[cyan]Archive {archive_idx + 1}[/cyan]")
        self.console.print(f"  Name: {archive.get('name', '')}")
        self.console.print(f"  Subpath: {archive.get('subpath', '')}")
        self.console.print(f"  Pattern: {archive.get('pattern', '*')}")

        self.console.print("\n[dim]Edit: (1) Name | (2) Subpath | (3) Pattern | (d) Delete | (b) Back[/dim]")

        choice = Prompt.ask("Enter command", default="b").lower().strip()

        if choice == "1":
            new_name = Prompt.ask("Archive name", default=archive.get("name", "")).strip()
            archive["name"] = new_name

        elif choice == "2":
            new_subpath = Prompt.ask("Subpath", default=archive.get("subpath", "")).strip()
            archive["subpath"] = new_subpath

        elif choice == "3":
            new_pattern = Prompt.ask("Pattern", default=archive.get("pattern", "*")).strip()
            archive["pattern"] = new_pattern

        elif choice == "d":
            if Confirm.ask("Delete this archive entry?", default=False):
                archives.pop(archive_idx)
                self.console.print("[green]Archive deleted.[/green]")

        # Update the target
        self.editor.update_target(group, target_index, target)


class ARCWizard:
    """Guided wizard for creating new catalog entries."""

    def __init__(
        self,
        console: Console,
        editor: CatalogEditor,
        group: str,
        show_header_callback: Callable[[], None] | None = None,
    ):
        """Initialize the wizard.

        Args:
            console: Rich console for output.
            editor: CatalogEditor instance.
            group: The group to add the entry to.
        """
        self.console = console
        self.show_header_callback = show_header_callback
        self.editor = editor
        self.group = group
        self.state: dict[str, Any] = {}
        self.step = 1

    def run(self) -> bool:
        """Run the wizard.

        Returns:
            True if an entry was created, False if cancelled.
        """

        steps = [
            self._step_basic_info,
            self._step_glob_pattern,
            self._step_file_type,
            self._step_database_options,
            self._step_preserve_structure,
            self._step_archives,
            self._step_review,
        ]

        self.step = 0
        while 0 <= self.step < len(steps):
            result = steps[self.step]()
            if result == "next":
                self.step += 1
            elif result == "back":
                self.step -= 1
            elif result == "cancel":
                self.console.print("[yellow]Wizard cancelled.[/yellow]")
                return False
            elif result == "done":
                return True

        return False

    def _step_basic_info(self) -> str:
        """Step 1: Get basic info (name and description only)."""
        if self.show_header_callback:
            self.show_header_callback()

        self.console.print("[bold]ARC Wizard - Create New Entry[/bold]")
        self.console.print("[dim]Enter 'b' at any step to go back, 'c' to cancel.[/dim]\n")
        self.console.print("[bold cyan]Step 1: Basic Information[/bold cyan]")

        # Name (using prompt_toolkit for better editing)
        name = get_text_input(
            "Target name (e.g., 'Safari History')",
            default=self.state.get("name", ""),
        )

        if name.lower() == "c":
            return "cancel"
        if name.lower() == "b":
            return "back"
        if not name:
            self.console.print("[red]Name is required.[/red]")
            return "stay"

        self.state["name"] = name

        # Description (using prompt_toolkit for better editing)
        desc = get_text_input(
            "Description (optional)",
            default=self.state.get("description", ""),
        )

        if desc.lower() == "c":
            return "cancel"
        if desc.lower() == "b":
            return "back"

        self.state["description"] = desc

        return "next"

    def _step_glob_pattern(self) -> str:
        """Step 2: Get and validate glob pattern."""
        if self.show_header_callback:
            self.show_header_callback()
        self.console.print("[bold cyan]Step 2: Glob Pattern[/bold cyan]")
        self.console.print("\n[dim]Glob pattern syntax guide:[/dim]")
        self.console.print("If under the Users/ directory, replace the username with [blue]*[/blue]")
        self.console.print("  - [blue]Users/*/Library/Application Support/com.apple.TCC/TCC.db[/blue]")
        self.console.print(
            "\nUse [blue]*[/blue] to represent variable folder names with a fixed depth as well as variable file names"
        )
        self.console.print("  - [blue]private/var/folders/*/*/0/com.apple.routined/dv/Cache/Cloud*.sqlite[/blue]")
        self.console.print("\nIf the file exists at an unknown folder depth, you can use [blue]**[/blue]")
        self.console.print("  - [blue]Users/*/Library/Caches/**/Cache.db[/blue]")
        self.console.print(
            "\nIf the file is associated with multiple profiles, replace the profile name with [blue]*[/blue]"
        )
        self.console.print(
            "  - [blue]Users/*/Library/Application Support/Google/Chrome/*/databases/Databases.db[/blue]"
        )
        self.console.print(
            "\nTo select all files, use [blue]*[/blue] instead of a filename (type [blue]log[/blue] or [blue]cache[/blue] only)"
        )
        self.console.print("  - [blue]Users/*/Library/Caches/Firefox/Profiles/*/cache2/*[/blue]")
        self.console.print("\nYou can also use regular expressions in the glob pattern")
        self.console.print(
            "  - [blue]/Users/*/Library/Group Containers/group.com.example/[0-9][0-9][0-9][0-9][0-9]*.db[/blue]"
        )
        self.console.print(
            "\nAnd if you want to copy a folder with structure intact, use [blue]**/*[/blue] at the root level."
        )
        self.console.print("  - [blue]private/var/db/uuidtext/**/*[/blue]\n")

        glob_str = get_text_input(
            "Enter glob pattern",
            default=self.state.get("glob_pattern", ""),
        )

        if glob_str.lower() == "c":
            return "cancel"
        if glob_str.lower() == "b":
            return "back"
        if not glob_str:
            self.console.print("[red]Glob pattern is required.[/red]")
            return "stay"

        # Parse and validate (use "cache" for permissive validation - file_type comes next)
        result = user_glob_parser(glob_str, "cache")

        if not result.is_valid:
            self.console.print("[red]Validation errors:[/red]")
            for error in result.errors:
                self.console.print(f"  [red]- {error}[/red]")
            return "stay"

        self.state["glob_pattern"] = result.glob_pattern
        self.state["scope"] = result.scope

        self.console.print(f"\n[green]Detected scope: {result.scope}[/green]")

        # Suggest multi_profile for browsers
        if suggest_multi_profile(result.glob_pattern, result.scope):
            self.console.print("[yellow]This looks like a browser path with multiple profiles.[/yellow]")
            self.state["_suggest_multi_profile"] = True
        else:
            self.state["_suggest_multi_profile"] = False

        return "next"

    def _step_file_type(self) -> str:
        """Step 3: Confirm or select file type based on glob inference."""
        if self.show_header_callback:
            self.show_header_callback()
        self.console.print("[bold cyan]Step 3: File Type[/bold cyan]")

        glob_pattern = self.state.get("glob_pattern", "")
        suggested, choices = infer_file_type_from_glob(glob_pattern)

        if suggested:
            self.console.print(f"\n[dim]Detected from pattern: {suggested}[/dim]")
            if Confirm.ask(f"Use '{suggested}'?", default=True):
                self.state["file_type"] = suggested
                return "next"

        # Show filtered menu
        self.console.print("\nFile type:")
        type_map = {}
        labels = {"database": "database (SQLite)", "log": "log (text file)", "cache": "cache (binary)"}
        for i, choice in enumerate(choices, 1):
            self.console.print(f"  {i}. {labels.get(choice, choice)}")
            type_map[str(i)] = choice

        current_type = self.state.get("file_type")
        default_choice = "1"
        if current_type:
            for k, v in type_map.items():
                if v == current_type:
                    default_choice = k
                    break

        selection = Prompt.ask("Select type", default=default_choice).strip()
        if selection.lower() == "c":
            return "cancel"
        if selection.lower() == "b":
            return "back"

        selected_type = type_map.get(selection, choices[0])

        # Warn if selecting database but terminal is wildcard
        if selected_type == "database":
            terminal = glob_pattern.split("/")[-1] if "/" in glob_pattern else glob_pattern
            if terminal in ("*", "**/*", "**"):
                self.console.print(f"[yellow]Warning: '{terminal}' terminal is not recommended for databases.[/yellow]")
                if not Confirm.ask("Continue anyway?", default=False):
                    return "stay"

        self.state["file_type"] = selected_type
        return "next"

    def _step_database_options(self) -> str:
        """Step 4: Database-specific options (multi_profile, ignorable_tables)."""
        if self.state.get("file_type") != "database":
            return "next"

        if self.show_header_callback:
            self.show_header_callback()
        self.console.print("[bold cyan]Step 4: Database Options[/bold cyan]")

        # Multi-profile - only ask if glob has wildcards beyond Users/*/
        glob_pattern = self.state.get("glob_pattern", "")
        scope = self.state.get("scope", "system")

        if scope == "user" and has_extra_wildcards(glob_pattern):
            # Eligible for multi-profile
            if self.state.get("_suggest_multi_profile"):
                multi = Confirm.ask(
                    "Does this database have multiple profiles (e.g., browser profiles)?",
                    default=True,
                )
            else:
                multi = Confirm.ask(
                    "Does this database have multiple profiles?",
                    default=False,
                )
            self.state["multi_profile"] = multi
        else:
            # Not eligible for multi-profile
            self.state["multi_profile"] = False

        # Ignorable tables
        if Confirm.ask("Are there tables to ignore during matching?", default=False):
            tables: list[str] = self.state.get("ignorable_tables", [])
            while True:
                table = Prompt.ask(
                    "Table name (Enter when done)",
                    default="",
                    show_default=False,
                ).strip()
                if not table:  # Empty = done
                    break
                if table.lower() == "b":
                    return "back"
                if table.lower() == "c":
                    return "cancel"
                if table not in tables:
                    tables.append(table)
                    self.console.print(f"  Added: {table}")
            self.state["ignorable_tables"] = tables
        else:
            self.state["ignorable_tables"] = []

        return "next"

    def _step_preserve_structure(self) -> str:
        """Step 4: Preserve structure option (for logs/caches)."""
        file_type = self.state.get("file_type", "database")

        if file_type == "database":
            return "next"

        if self.show_header_callback:
            self.show_header_callback()

        glob_pattern = self.state.get("glob_pattern", "")
        if "**/*" in glob_pattern:
            self.console.print("\n[bold cyan]Step 5: Preserve Structure[/bold cyan]")
            preserve = Confirm.ask(
                "Preserve the entire folder structure when copying?",
                default=True,
            )
            self.state["preserve_structure"] = preserve
        else:
            self.state["preserve_structure"] = False

        return "next"

    def _step_archives(self) -> str:
        """Step 5: Archives configuration."""
        if self.show_header_callback:
            self.show_header_callback()

        self.console.print("[bold cyan]Step 6: Archives[/bold cyan]")

        has_archives = Confirm.ask(
            "Does this target have associated archive files?",
            default=False,
        )

        if not has_archives:
            self.state["has_archives"] = False
            return "next"

        self.state["has_archives"] = True

        # For has_archives, we need primary.glob_pattern
        self.state["primary"] = {"glob_pattern": self.state.get("glob_pattern", "")}
        # Remove the regular glob_pattern since we're using primary
        if "glob_pattern" in self.state:
            del self.state["glob_pattern"]

        archives: list[dict] = []
        while True:
            self.console.print(f"\n[dim]Archive {len(archives) + 1}[/dim]")

            name = Prompt.ask("Archive name (e.g., 'Archives')", default="Archives").strip()
            if name.lower() == "b":
                return "back"
            if name.lower() == "c":
                return "cancel"

            subpath = Prompt.ask(
                "Subpath relative to primary (empty for same folder)",
                default="",
            ).strip()

            pattern = Prompt.ask("File pattern (e.g., '*.gz')", default="*").strip()

            archives.append(
                {
                    "name": name,
                    "subpath": subpath,
                    "pattern": pattern,
                    "forensic_note": "",
                }
            )

            if not Confirm.ask("Add another archive location?", default=False):
                break

        self.state["archives"] = archives

        # Combine strategy
        file_type = self.state.get("file_type", "database")
        if Confirm.ask("Combine archives with primary?", default=True):
            if file_type == "database":
                self.state["combine_strategy"] = "decompress_and_merge"
            else:
                self.state["combine_strategy"] = "decompress_and_concatenate"
        else:
            self.state["combine_strategy"] = "decompress_only"

        return "next"

    def _step_review(self) -> str:
        """Step 6: Review and confirm."""
        if self.show_header_callback:
            self.show_header_callback()

        self.console.print("[bold cyan]Step 7: Review[/bold cyan]")

        # Generate exemplar pattern
        name = self.state.get("name", "")
        file_type = self.state.get("file_type", "database")
        self.state["exemplar_pattern"] = generate_exemplar_pattern(name, file_type)

        # Build the final entry
        entry = self._build_entry()

        # Display as formatted output
        self.console.print("\n[bold]Entry to be created:[/bold]")
        for key, value in entry.items():
            if isinstance(value, dict):
                self.console.print(f"  {key}:")
                for k, v in value.items():
                    self.console.print(f"    {k}: {v}")
            elif isinstance(value, list):
                if value:
                    self.console.print(f"  {key}:")
                    for item in value:
                        if isinstance(item, dict):
                            self.console.print(f"    - {item}")
                        else:
                            self.console.print(f"    - {item}")
                else:
                    self.console.print(f"  {key}: []")
            else:
                self.console.print(f"  {key}: {value}")

        # Validate
        is_valid, errors = validate_entry(entry)
        if not is_valid:
            self.console.print("\n[red]Validation errors:[/red]")
            for error in errors:
                self.console.print(f"  [red]- {error}[/red]")
            if not Confirm.ask("Continue anyway?", default=False):
                return "back"

        # Confirm
        choice = (
            Prompt.ask(
                "\n[bold]Confirm[/bold]: (s)ave, (b)ack, (c)ancel",
                default="s",
            )
            .lower()
            .strip()
        )

        if choice == "s":
            if self.editor.add_target(self.group, entry):
                # Also save to disk
                if self.editor.save():
                    self.console.print(f"[green]Entry '{name}' added and saved to '{self.group}'.[/green]")
                else:
                    self.console.print("[yellow]Entry added but failed to save to disk.[/yellow]")
                return "done"
            self.console.print("[red]Failed to add entry.[/red]")
            return "cancel"
        if choice == "b":
            return "back"
        return "cancel"

    def _build_entry(self) -> dict[str, Any]:
        """Build the final entry dictionary from state."""
        entry: dict[str, Any] = {}

        # Required fields
        entry["name"] = self.state.get("name", "")
        entry["scope"] = self.state.get("scope", "system")
        entry["description"] = self.state.get("description", "")
        entry["exemplar_pattern"] = self.state.get("exemplar_pattern", "")

        # File type - always include for progressive field unlocking
        entry["file_type"] = self.state.get("file_type", "database")

        # Glob pattern or primary
        if "primary" in self.state:
            entry["has_archives"] = True
            entry["primary"] = self.state["primary"]
            entry["archives"] = self.state.get("archives", [])
            if "combine_strategy" in self.state:
                entry["combine_strategy"] = self.state["combine_strategy"]
        else:
            entry["glob_pattern"] = self.state.get("glob_pattern", "")

        # Optional fields (only include if set)
        if self.state.get("multi_profile"):
            entry["multi_profile"] = True

        if self.state.get("preserve_structure"):
            entry["preserve_structure"] = True

        ignorable = self.state.get("ignorable_tables", [])
        if ignorable:
            entry["ignorable_tables"] = ignorable
        else:
            entry["ignorable_tables"] = []

        if self.state.get("notes"):
            entry["notes"] = self.state["notes"]

        return entry
