#!/usr/bin/env python3
"""
File Categorization Statistics Reporter

Provides formatted output for file categorization statistics from RawFileProcessor.
Extracted from processor.py to improve modularity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.live import Live

from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from rich.console import Console

    from mars.config import MARSConfig


class CategorizationReporter:
    """Reporter for file categorization statistics with debug and rich output modes."""

    def __init__(self, stats: dict[str, Any]):
        """
        Initialize the reporter.

        Args:
            stats: Statistics dictionary from RawFileProcessor
        """
        self.stats = stats

    def print_stats(
        self,
        console: Console | None = None,
        config: MARSConfig | None = None,
    ):
        """
        Print categorization statistics.

        Args:
            console: Rich console for formatted output (optional)
            config: Config to check debug mode (optional)
        """
        # Determine if we should use debug output or rich output
        use_debug = config.ui.debug if config else False

        if use_debug or console is None:
            self._print_debug_stats()
        else:
            self._print_rich_stats(console)

    def _print_debug_stats(self):
        """Print plain text debug statistics."""
        stats = self.stats
        logger.separator()
        logger.info("File Categorization Summary")
        logger.separator()
        logger.info(f"Total files scanned: {stats['total_files']}")
        logger.info(f"Files classified: {stats['classified']}")
        logger.info(f"Ignored files: {stats['unknown']}")
        logger.info("")
        logger.info("Collected artifacts:")
        logger.info(f"  WiFi plists: {stats['wifi_plists_kept']}")
        logger.info(f"  ASL logs: {stats['asl_logs_kept']}")
        logger.info(f"  JSONLZ4 files: {stats['jsonlz4_kept']}")
        logger.info(f"  Text logs: {stats['text_logs']}")
        logger.info(f"  SQLite databases: {stats['sqlite_dbs']}")
        logger.info("")

        # Archive processing stats
        if stats.get("archives_processed", 0) > 0:
            logger.info("Archive processing (inline decompression):")
            logger.info(f"  Archives processed: {stats.get('archives_processed', 0)}")
            if stats.get("archives_failed", 0) > 0:
                logger.info(f"  Failed to decompress: {stats.get('archives_failed', 0)}")
            if stats.get("sqlite_from_archives", 0) > 0:
                logger.info(f"  SQLite databases from archives: {stats.get('sqlite_from_archives', 0)}")
            if stats.get("wifi_plists_from_archives", 0) > 0:
                logger.info(f"  WiFi plists from archives: {stats.get('wifi_plists_from_archives', 0)}")
            if stats.get("text_logs_from_archives", 0) > 0:
                logger.info(f"  Text logs from archives: {stats.get('text_logs_from_archives', 0)}")
            if stats.get("asl_logs_from_archives", 0) > 0:
                logger.info(f"  ASL logs from archives: {stats.get('asl_logs_from_archives', 0)}")
            if stats.get("jsonlz4_from_archives", 0) > 0:
                logger.info(f"  JSONLZ4 from archives: {stats.get('jsonlz4_from_archives', 0)}")
            logger.info("")

        if stats.get("plists_skipped", 0) > 0:
            saved_gb = stats.get("plists_skipped_bytes", 0) / (1024**3)
            logger.info(f"Disk space saved: {saved_gb:.2f} GB ({stats['plists_skipped']} non-WiFi plists skipped)")
        logger.separator()

    def _print_rich_stats(self, console: Console):
        """Print Rich formatted statistics in a compact panel."""
        from rich.panel import Panel
        from rich.table import Table

        stats = self.stats

        # Create a multi-column table for compact display
        table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        table.add_column("Label1", style="dim", width=18)
        table.add_column("Value1", style="bold cyan", width=8, justify="right")
        table.add_column("Label2", style="dim", width=18)
        table.add_column("Value2", style="bold cyan", width=8, justify="right")
        table.add_column("Label3", style="dim", width=18)
        table.add_column("Value3", style="bold cyan", width=8, justify="right")

        # Row 1: Overview stats
        table.add_row(
            "Files scanned:",
            f"{stats['total_files']:,}",
            "Classified:",
            f"{stats['classified']:,}",
            "Ignored:",
            f"{stats['unknown']:,}",
        )

        # Row 2: Collected artifacts
        table.add_row(
            "SQLite databases:",
            f"{stats['sqlite_dbs']:,}",
            "Text logs:",
            f"{stats['text_logs']:,}",
            "WiFi plists:",
            f"{stats['wifi_plists_kept']:,}",
        )

        # Row 3: More artifacts
        table.add_row(
            "ASL logs:",
            f"{stats['asl_logs_kept']:,}",
            "JSONLZ4 files:",
            f"{stats['jsonlz4_kept']:,}",
            "",
            "",
        )

        # Add archive stats if any were processed
        archives_processed = stats.get("archives_processed", 0)
        if archives_processed > 0:
            table.add_row("", "", "", "", "", "")  # Spacer row
            table.add_row(
                "Archives processed:",
                f"{archives_processed:,}",
                "Failed:",
                f"{stats.get('archives_failed', 0):,}",
                "SQLite from archives:",
                f"{stats.get('sqlite_from_archives', 0):,}",
            )

        # Add disk space saved if plists were skipped
        plists_skipped = stats.get("plists_skipped", 0)
        if plists_skipped > 0:
            saved_gb = stats.get("plists_skipped_bytes", 0) / (1024**3)
            table.add_row("", "", "", "", "", "")  # Spacer row
            table.add_row(
                "Plists skipped:",
                f"{plists_skipped:,}",
                "Space saved:",
                f"{saved_gb:.1f} GB",
                "",
                "",
            )

        def make_live_summary() -> None:
            panel = Panel(
                table,
                title="[bold deep_sky_blue1]File Categorization Summary[/bold deep_sky_blue1]",
                border_style="deep_sky_blue3",
                padding=(0, 1),
            )
            with Live(panel, console=console):
                pass

        make_live_summary()
