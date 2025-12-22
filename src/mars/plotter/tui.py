#!/usr/bin/env python3

"""
Rich-based styled output functions for the plotter.
"""

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

# Shared console instance
console = Console()


def cprint(msg: str, style: str = ""):
    """Print a message with optional style."""
    console.print(msg, style=style)


def cinfo(msg: str):
    """Print an info message in cyan."""
    console.print(f"[cyan]{msg}[/cyan]")


def cgood(msg: str):
    """Print a success message in green."""
    console.print(f"[green]{msg}[/green]")


def cwarn(msg: str):
    """Print a warning message in yellow."""
    console.print(f"[yellow]{msg}[/yellow]")


def cerr(msg: str):
    """Print an error message in red."""
    console.print(f"[bold red]{msg}[/bold red]")


def chead(msg: str):
    """Print a header/title."""
    console.print(Panel(msg, style="bold cyan", border_style="cyan"))


def cdim(msg: str):
    """Print dimmed/muted text."""
    console.print(f"[dim]{msg}[/dim]")


def show_menu_table(
    title: str,
    options: list[tuple[str, str, bool]],
    show_back: bool = True,
) -> Table:
    """
    Create a styled menu table.

    Args:
        title: Menu title
        options: List of (number, label, enabled) tuples
        show_back: Whether to show back option

    Returns:
        Rich Table object
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold cyan", width=4)
    table.add_column()

    for num, label, enabled in options:
        if enabled:
            table.add_row(f"{num}.", f"[bold]{label}[/bold]")
        else:
            table.add_row(f"[dim]{num}.[/dim]", f"[dim]{label}[/dim]")

    if show_back:
        table.add_row("", "[dim](b) Back[/dim]")

    return table


def prompt_input(prompt_text: str = ">") -> str:
    """Get user input with styled prompt."""
    return Prompt.ask(f"[cyan]{prompt_text}[/cyan]", show_default=False) or ""


def prompt_confirm(prompt_text: str, default: bool = False) -> bool:
    """Ask a yes/no question."""
    return Confirm.ask(prompt_text, default=default)


# Legacy C class for backwards compatibility during transition
class C:
    """Legacy color class - use Rich markup instead."""

    RESET = ""
    BOLD = ""

    class FG:
        GRAY = ""
        RED = ""
        GREEN = ""
        YELLOW = ""
        CYAN = ""
