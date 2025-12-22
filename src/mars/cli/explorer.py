#!/usr/bin/env python3
"""
CLI File Explorer for MARS
by WarpedWing Labs

Simple curses-style file browser using rich for CLI navigation.
"""

from __future__ import annotations

import string
import sys
from pathlib import Path

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text


class FileExplorer:
    """Simple file/directory explorer for CLI."""

    def __init__(
        self,
        start_path: Path | None = None,
        file_filter: str | list[str] | None = None,
    ):
        """
        Initialize file explorer.

        Args:
            start_path: Starting directory (default: OS-appropriate default)
            file_filter: File extension filter (e.g., ".marsproj", ".db") or list of extensions (e.g., [".E01", ".ex01"])
        """
        self.console = Console()
        self._cancelled_initial_drive = False

        if start_path is None:
            start_path = self._get_default_start_path()

        # Normalize file_filter to always be a list for consistency
        if file_filter is None:
            self.file_filter = None
        elif isinstance(file_filter, str):
            self.file_filter = [file_filter]
        else:
            self.file_filter = file_filter

        # On Windows, show drive picker first if starting from default
        if sys.platform == "win32" and start_path == Path.home():
            picked_drive = self._show_drive_picker(is_initial=True)
            if picked_drive:
                start_path = picked_drive
            else:
                # User cancelled initial drive selection - flag for browse() to return None
                self._cancelled_initial_drive = True
                self.current_path = start_path  # Set a default to avoid None errors
                return

        self.current_path = start_path.resolve()

    def browse(self, title: str = "Select File or Directory", explanation: str | None = None) -> Path | None:
        """
        Browse for a file or directory.

        Args:
            title: Title to display
            explanation: Explanation to display

        Returns:
            Selected path or None if cancelled
        """
        # If user cancelled initial drive selection, return None immediately
        if self._cancelled_initial_drive:
            return None

        while True:
            # Display current directory
            self._display_directory(title, explanation)

            # Get user choice
            choice = (
                Prompt.ask(
                    "\n[bold cyan]Enter choice[/bold cyan]",
                    show_default=False,
                )
                .strip()
                .lower()
            )

            # Handle commands
            if choice == "b":
                return None
            if choice == "d" and sys.platform == "win32":
                # Change drive (Windows only)
                self.console.clear()
                picked_drive = self._show_drive_picker()
                if picked_drive:
                    self.current_path = picked_drive
            elif choice == "..":
                # Go up one level, or show drive picker if at root on Windows
                if self._is_drive_root():
                    picked_drive = self._show_drive_picker()
                    if picked_drive:
                        self.current_path = picked_drive
                elif self.current_path.parent != self.current_path:
                    self.current_path = self.current_path.parent
            elif choice == ".":
                # Select current directory
                return self.current_path
            elif choice == "n":
                # Create new folder
                folder_name = Prompt.ask("[bold cyan]New folder name[/bold cyan]")
                if folder_name:
                    new_folder = self.current_path / folder_name
                    try:
                        new_folder.mkdir(parents=True, exist_ok=True)
                        self.console.print(f"[green]Created folder: {new_folder}[/green]")
                        return new_folder
                    except Exception as e:
                        self.console.print(f"[red]Error creating folder: {e}[/red]")
            elif choice.isdigit():
                # Select by number
                idx = int(choice) - 1
                items = self._get_directory_items()

                if 0 <= idx < len(items):
                    selected = items[idx]

                    if selected.is_dir():
                        # Enter directory
                        self.current_path = selected
                    else:
                        # Select file
                        return selected
                else:
                    self.console.print("[red]Invalid selection[/red]")
            else:
                self.console.print("[red]Invalid choice. Try again.[/red]")

    def _get_directory_items(self) -> list[Path]:
        """Get sorted list of items in current directory."""
        try:
            # Files to ignore (macOS metadata and system files)
            ignore_files = {
                ".DS_Store",
                ".localized",
                ".Spotlight-V100",
                ".Trashes",
                ".DocumentRevisions-V100",
                ".fseventsd",
                ".TemporaryItems",
                "$RECYCLE.BIN",
            }

            items = [
                item
                for item in self.current_path.iterdir()
                if item.name not in ignore_files and not item.name.startswith(".")
            ]

            # Filter if specified
            if self.file_filter:
                filtered_items = []
                for item in items:
                    if item.is_dir() or item.suffix in self.file_filter:
                        filtered_items.append(item)
                items = filtered_items

            # Sort: directories first, then files
            items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            return items
        except PermissionError:
            return []

    def _display_directory(self, title: str, explanation: str | None = None) -> None:
        """Display current directory contents."""
        self.console.clear()

        title_text = f"[bold deep_sky_blue2]{title}[/bold deep_sky_blue2]"
        explanation_text = f"[italic grey50]{explanation}[/italic grey50]\n" if explanation else ""
        path_text = Text.assemble(("Current: ", "dim"), (str(self.current_path), "dim"))

        title_group = Group(title_text, explanation_text, path_text)

        title_panel = Panel(
            title_group,
            title="[bold deep_sky_blue1]File Explorer[/bold deep_sky_blue1]",
            border_style="deep_sky_blue3",
        )
        self.console.print(title_panel)

        # Create table
        table = Table(
            show_header=True,
            header_style="bold deep_sky_blue1",
            border_style="grey54",
            box=box.ROUNDED,
            expand=True,
        )
        table.add_column("#", style="bold dim", width=4)
        table.add_column("Type", width=6)
        table.add_column("Name", min_width=30)
        table.add_column("Size", justify="right", width=8)

        # Add parent directory option
        if self.current_path.parent != self.current_path:
            table.add_row(
                "..",
                "[bold deep_sky_blue3][DIR][/bold deep_sky_blue3]",
                "[grey54].. (Parent Directory)[/grey54]",
                "",
            )

        # Add items
        items = self._get_directory_items()

        for idx, item in enumerate(items, start=1):
            try:
                if item.is_dir():
                    type_str = "[bold deep_sky_blue3][DIR][/bold deep_sky_blue3]"
                    name_str = f"[dark_slate_gray1]{item.name}/[/dark_slate_gray1]"
                    size_str = ""
                else:
                    type_str = "[bold deep_sky_blue1][FILE][bold deep_sky_blue1]"
                    name_str = f"[dark_slate_gray1]{item.name}[/dark_slate_gray1]"
                    size_str = self._format_size(item.stat().st_size)

                    # Highlight files matching filter
                    if self.file_filter and item.suffix in self.file_filter:
                        name_str = f"[bold dark_sea_green4]{item.name}[/bold dark_sea_green4]"

                table.add_row(str(idx), type_str, name_str, size_str)
            except (PermissionError, OSError):
                continue

        nav_text_left = (
            "[bold cyan]..[/bold cyan] -  Go up one level\n"
            "[bold cyan]#[/bold cyan]  -  [bold dark_sea_green4]Select file/directory\n[/bold dark_sea_green4]"
            "[bold cyan].[/bold cyan]  -  [bold dark_sea_green4]Select current directory[/bold dark_sea_green4]  "
        )

        nav_text_right = "[bold cyan]n[/bold cyan]  -  Create new folder\n"
        if sys.platform == "win32":
            nav_text_right += "[bold cyan]d[/bold cyan]  -  Change drive\n"
        nav_text_right += "[bold cyan]b[/bold cyan]  -  Back to previous menu"

        nav_grid = Table.grid()
        nav_grid.add_column(ratio=1)
        nav_grid.add_column(ratio=1)
        nav_grid.add_row(nav_text_left, nav_text_right)

        nav_panel = Panel(
            nav_grid,
            title="[bold deep_sky_blue3]Navigation:[/bold deep_sky_blue3]",
            title_align="left",
            border_style="dark_goldenrod",
            expand=False,
        )

        layout_grid = Table.grid()
        layout_grid.add_column(no_wrap=True)
        layout_grid.add_row(table)
        layout_grid.add_row(nav_panel)
        self.console.print(layout_grid)

    @staticmethod
    def _format_size(size_bytes: float) -> str:
        """Format file size in human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    @staticmethod
    def _get_default_start_path() -> Path:
        """
        Return platform-appropriate default starting path for file browsing.

        Returns:
            - Windows: User's home directory (C:\\Users\\username)
            - macOS: /Volumes (mounted drives)
            - Linux: /mnt if exists, otherwise home directory
        """
        if sys.platform == "win32":
            return Path.home()
        elif sys.platform == "darwin":  # noqa: RET505
            return Path("/Volumes")
        else:
            # Linux - prefer /mnt if it exists, otherwise home
            mnt_path = Path("/mnt")
            return mnt_path if mnt_path.exists() else Path.home()

    @staticmethod
    def _get_windows_drives() -> list[tuple[Path, str]]:
        """Get list of available drive letters on Windows with volume labels."""
        import ctypes

        drives = []
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]  # Windows-only

        for letter in string.ascii_uppercase:
            drive_path = Path(f"{letter}:/")
            if drive_path.exists():
                # Get volume label using Windows API
                volume_name = ctypes.create_unicode_buffer(261)
                try:
                    result = kernel32.GetVolumeInformationW(
                        f"{letter}:\\",
                        volume_name,
                        261,
                        None,
                        None,
                        None,
                        None,
                        0,
                    )
                    label = volume_name.value if result else ""
                except Exception:
                    label = ""
                drives.append((drive_path, label))
        return drives

    def _show_drive_picker(self, is_initial: bool = False) -> Path | None:
        """
        Show Windows drive selection menu.

        Args:
            is_initial: If True, this is the initial drive selection on startup

        Returns:
            Selected drive path or None if cancelled
        """
        with self.console.status("Retrieving available drives..."):
            drives = self._get_windows_drives()

        if not drives:
            return None

        # Build drive selection table
        table = Table(
            show_header=True,
            header_style="bold deep_sky_blue1",
            border_style="grey54",
            box=box.ROUNDED,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("Drive", width=8)
        table.add_column("Label", min_width=20)

        for idx, (drive, label) in enumerate(drives, start=1):
            table.add_row(str(idx), f"[bold]{drive.drive}[/bold]", label)

        panel = Panel(
            table,
            title="[bold deep_sky_blue1]Select Drive[/bold deep_sky_blue1]",
            border_style="deep_sky_blue3",
        )
        self.console.print(panel)
        self.console.print("\n[dim]b - Back/Cancel[/dim]")

        choice = (
            Prompt.ask(
                "\n[bold cyan]Select drive[/bold cyan]",
                show_default=False,
            )
            .strip()
            .lower()
        )

        if choice == "b":
            return None

        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(drives):
                return drives[idx][0]  # Return just the Path, not the label

        # If user typed a drive letter directly (e.g., "d" or "d:")
        letter = choice.rstrip(":").upper()
        if len(letter) == 1 and letter in string.ascii_uppercase:
            drive_path = Path(f"{letter}:/")
            if drive_path.exists():
                return drive_path

        return None

    def _is_drive_root(self) -> bool:
        """Check if current path is a drive root on Windows."""
        if sys.platform != "win32":
            return False
        # On Windows, drive roots are like C:\ - parent equals self
        return self.current_path.parent == self.current_path


def browse_for_file(
    start_path: Path | None = None,
    file_filter: str | list[str] | None = None,
    title: str = "Select File",
    explanation: str | None = None,
) -> Path | None:
    """
    Convenience function to browse for a file.

    Args:
        start_path: Starting directory
        file_filter: File extension filter (e.g., ".marsproj") or list of extensions (e.g., [".E01", ".ex01"])
        title: Title to display
        explanation: Explanation to display

    Returns:
        Selected file path or None if cancelled
    """
    explorer = FileExplorer(start_path, file_filter)
    return explorer.browse(title, explanation)


def browse_for_directory(
    start_path: Path | None = None,
    title: str = "Select Directory",
    explanation: str | None = None,
) -> Path | None:
    """
    Convenience function to browse for a directory.

    Args:
        start_path: Starting directory
        title: Title to display
        explanation: Explanation to display

    Returns:
        Selected directory path or None if cancelled
    """
    explorer = FileExplorer(start_path)
    return explorer.browse(title, explanation)
