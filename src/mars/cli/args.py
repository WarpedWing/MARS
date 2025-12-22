"""Unified CLI argument parser for MARS.

This module provides reusable argument parsers for all MARS CLI tools,
with automatic integration to the centralized configuration system.

Example:
    >>> parser = create_processor_parser()
    >>> args = parser.parse_args()
    >>> config = args_to_config(args)
    >>> processor = RawFileProcessor(input_dir, config=config)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from mars.config import ConfigLoader, MARSConfig
from mars.utils.debug_logger import logger


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Add configuration management arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    config_group = parser.add_argument_group("Configuration Management")
    config_group.add_argument(
        "--load-config",
        type=Path,
        metavar="FILE",
        help="Load configuration from JSON file",
    )
    config_group.add_argument(
        "--save-config",
        type=Path,
        metavar="FILE",
        help="Save current configuration to JSON file (and exit)",
    )
    config_group.add_argument(
        "--show-config",
        action="store_true",
        help="Display effective configuration (and exit)",
    )


def add_matching_args(parser: argparse.ArgumentParser) -> None:
    """Add matching/classification arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    matching = parser.add_argument_group("Matching Options")
    matching.add_argument(
        "--min-confidence",
        type=float,
        metavar="FLOAT",
        help="Minimum match confidence (0.0-1.0). Default: 0.7",
    )
    matching.add_argument(
        "--min-rows",
        type=int,
        metavar="N",
        help="Minimum rows for valid match. Default: 10",
    )
    matching.add_argument(
        "--min-columns",
        type=int,
        metavar="N",
        help="Minimum columns for valid match. Default: 3",
    )
    matching.add_argument(
        "--exemplar-confidence",
        type=float,
        metavar="FLOAT",
        help="Threshold to use exemplar rubrics. Default: 0.85",
    )


def add_scanner_args(parser: argparse.ArgumentParser) -> None:
    """Add scanner/recovery arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    scanner = parser.add_argument_group("Scanner Options")
    scanner.add_argument(
        "--timestamp-start",
        type=str,
        metavar="DATE",
        help="Start timestamp for filtering (YYYY-MM-DD)",
    )
    scanner.add_argument(
        "--timestamp-end",
        type=str,
        metavar="DATE",
        help="End timestamp for filtering (YYYY-MM-DD)",
    )
    scanner.add_argument(
        "--max-db-size",
        type=int,
        metavar="MB",
        help="Maximum database size to process (MB). Default: 1000",
    )
    scanner.add_argument(
        "--save-self-match-rubrics",
        action="store_true",
        help="Save self-match rubrics for unidentified databases",
    )


def add_output_args(parser: argparse.ArgumentParser) -> None:
    """Add output configuration arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    output = parser.add_argument_group("Output Options")
    output.add_argument(
        "--output-dir",
        type=Path,
        metavar="DIR",
        help="Base output directory",
    )
    output.add_argument(
        "--case-name",
        type=str,
        metavar="NAME",
        help="Case name for folder naming. Default: 'Case'",
    )
    output.add_argument(
        "--no-auto-timestamp",
        action="store_true",
        help="Disable automatic timestamp in folder names",
    )
    output.add_argument(
        "--output-prefix",
        type=str,
        metavar="PREFIX",
        help="Prefix for output folder names. Default: 'MARS'",
    )


def add_ui_args(parser: argparse.ArgumentParser) -> None:
    """Add UI/logging arguments to parser.

    Args:
        parser: ArgumentParser to add arguments to
    """
    ui = parser.add_argument_group("UI Options")
    ui.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    ui.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress non-error output",
    )
    ui.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        metavar="LEVEL",
        help="Logging level. Default: INFO",
    )


def create_base_parser(
    prog: str | None = None,
    description: str | None = None,
) -> argparse.ArgumentParser:
    """Create base argument parser with all common options.

    Args:
        prog: Program name for help text
        description: Program description for help text

    Returns:
        ArgumentParser with all common argument groups
    """
    parser = argparse.ArgumentParser(
        prog=prog,
        description=description or "MARS - SQLite database recovery and analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Add all argument groups
    add_matching_args(parser)
    add_scanner_args(parser)
    add_output_args(parser)
    add_ui_args(parser)
    add_config_args(parser)

    return parser


def create_processor_parser() -> argparse.ArgumentParser:
    """Create argument parser for RawFileProcessor CLI.

    Returns:
        ArgumentParser configured for processor.py
    """
    parser = create_base_parser(
        prog="mars process",
        description="Process recovered/carved files to identify and recover macOS forensic artifacts",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing recovered files (e.g., from PhotoRec)",
    )
    parser.add_argument(
        "--exemplar-dir",
        type=Path,
        metavar="DIR",
        help="Path to exemplar scan output for rubric matching",
    )

    return parser


def create_scanner_parser() -> argparse.ArgumentParser:
    """Create argument parser for ExemplarScanner CLI.

    Returns:
        ArgumentParser configured for scanner.py
    """
    parser = create_base_parser(
        prog="mars scan",
        description="Scan exemplar macOS system to generate database rubrics",
    )

    parser.add_argument(
        "--source",
        type=Path,
        metavar="PATH",
        help="Path to exemplar macOS system (e.g., /Volumes/MacintoshHD)",
    )
    parser.add_argument(
        "--e01-image",
        type=Path,
        metavar="FILE",
        help="Path to E01 forensic image (will be mounted automatically)",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        metavar="FILE",
        help="Path to database catalog YAML (defaults to built-in catalog)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        metavar="CAT",
        help="Filter by categories (e.g., browser security system)",
    )
    parser.add_argument(
        "--custom-paths",
        nargs="+",
        metavar="PATTERN",
        help="Additional custom glob patterns to scan",
    )

    return parser


def args_to_config(args: argparse.Namespace) -> MARSConfig:
    """Convert argparse namespace to MARSConfig.

    Args:
        args: Parsed arguments from ArgumentParser

    Returns:
        MARSConfig with values from CLI arguments
    """
    # Build CLI args dict (only include non-None values)
    cli_dict: dict[str, Any] = {}

    # Matching options
    if hasattr(args, "min_confidence") and args.min_confidence is not None:
        cli_dict.setdefault("matching", {})["min_confidence"] = args.min_confidence
    if hasattr(args, "min_rows") and args.min_rows is not None:
        cli_dict.setdefault("matching", {})["min_rows"] = args.min_rows
    if hasattr(args, "min_columns") and args.min_columns is not None:
        cli_dict.setdefault("matching", {})["min_columns"] = args.min_columns
    if hasattr(args, "exemplar_confidence") and args.exemplar_confidence is not None:
        cli_dict.setdefault("matching", {})["exemplar_confidence_threshold"] = args.exemplar_confidence

    # Scanner options
    if hasattr(args, "timestamp_start") and args.timestamp_start:
        cli_dict.setdefault("scanner", {})["target_timestamp_start"] = args.timestamp_start
    if hasattr(args, "timestamp_end") and args.timestamp_end:
        cli_dict.setdefault("scanner", {})["target_timestamp_end"] = args.timestamp_end
    if hasattr(args, "max_db_size") and args.max_db_size is not None:
        cli_dict.setdefault("scanner", {})["max_db_size_mb"] = args.max_db_size
    if hasattr(args, "save_self_match_rubrics") and args.save_self_match_rubrics:
        cli_dict.setdefault("scanner", {})["save_self_match_rubrics"] = True

    # Output options
    if hasattr(args, "case_name") and args.case_name:
        cli_dict.setdefault("output", {})["case_name_default"] = args.case_name
    if hasattr(args, "no_auto_timestamp") and args.no_auto_timestamp:
        cli_dict.setdefault("output", {})["auto_timestamp"] = False
    if hasattr(args, "output_prefix") and args.output_prefix:
        cli_dict.setdefault("output", {})["prefix"] = args.output_prefix

    # UI options
    if hasattr(args, "verbose") and args.verbose:
        cli_dict.setdefault("ui", {})["verbose"] = True
    if hasattr(args, "quiet") and args.quiet:
        cli_dict.setdefault("ui", {})["log_level"] = "ERROR"
    elif hasattr(args, "log_level") and args.log_level:
        cli_dict.setdefault("ui", {})["log_level"] = args.log_level

    # Load config with proper priority
    # Check if user wants to load from specific file
    if hasattr(args, "load_config") and args.load_config:
        config = ConfigLoader.load_from_file(args.load_config)
        # Still apply CLI args on top
        if cli_dict:
            config = ConfigLoader._merge_config(config, cli_dict)
    else:
        # Normal priority: CLI > Project > Defaults
        config = ConfigLoader.load(
            cli_args=cli_dict if cli_dict else None,
            project_dir=Path.cwd(),
        )

    return config


def handle_config_commands(args: argparse.Namespace, config: MARSConfig) -> bool:
    """Handle config management commands (--show-config, --save-config).

    Args:
        args: Parsed arguments
        config: Loaded configuration

    Returns:
        True if command was handled (should exit), False otherwise
    """
    # Show config
    if hasattr(args, "show_config") and args.show_config:
        logger.debug("=" * 60)
        logger.debug("MARS - Effective Configuration")
        logger.debug("=" * 60)
        logger.debug(str(config))
        logger.debug("\nConfiguration sources (priority order):")
        logger.debug("  1. CLI arguments (highest)")
        logger.debug("  2. Project config (.marsproj in current directory)")
        logger.debug("  3. User preferences (~/.config/mars/preferences.json)")
        logger.debug("  4. Built-in defaults (lowest)")
        return True

    # Save config
    if hasattr(args, "save_config") and args.save_config:
        ConfigLoader.save_to_file(config, args.save_config)
        logger.debug(f"✓ Configuration saved to {args.save_config}")
        return True

    return False


def create_config_subcommands() -> argparse.ArgumentParser:
    """Create parser for config management subcommands.

    Returns:
        ArgumentParser with config subcommands (show, save-prefs, reset)
    """
    parser = argparse.ArgumentParser(
        prog="mars config",
        description="Manage MARS configuration",
    )

    subparsers = parser.add_subparsers(dest="command", help="Configuration commands")

    # Show command
    show_parser = subparsers.add_parser("show", help="Display effective configuration")
    show_parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )

    # Save-prefs command
    save_parser = subparsers.add_parser("save-prefs", help="Save current settings as user preferences")
    add_matching_args(save_parser)
    add_scanner_args(save_parser)
    add_output_args(save_parser)
    add_ui_args(save_parser)

    # Reset command
    reset_parser = subparsers.add_parser("reset", help="Reset user preferences to defaults")
    reset_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export configuration to file")
    export_parser.add_argument("file", type=Path, help="Output file (JSON format)")
    add_matching_args(export_parser)
    add_scanner_args(export_parser)
    add_output_args(export_parser)
    add_ui_args(export_parser)

    return parser
