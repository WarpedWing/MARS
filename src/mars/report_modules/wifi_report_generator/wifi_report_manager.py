#!/usr/bin/env python3
"""
WiFi Report Manager for MARS.

Orchestrates wifi_summary and wifi_report to generate comprehensive WiFi analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mars.report_modules.progress_interface import get_progress
from mars.report_modules.wifi_report_generator import (
    wifi_report,
    wifi_summary,
)
from mars.utils.debug_logger import logger


def main() -> None:
    """
    Main entry point for WiFi report generation.

    Coordinates the two-stage process:
    1. wifi_summary: Parse WiFi artifacts and generate JSON summary + events
    2. wifi_report: Generate HTML report from summary data
    """
    parser = argparse.ArgumentParser(description="Generate WiFi summary and HTML report from exemplar scan.")

    # Primary arguments
    parser.add_argument("input_path", type=Path, help="Path to exemplar root or run directory")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for reports")

    # wifi_summary options
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    parser.add_argument(
        "--oui-map",
        type=Path,
        help="Optional CSV mapping OUIs to vendor names",
    )
    parser.add_argument(
        "--presence-window",
        type=int,
        default=900,
        help="Seconds around each DHCP lease to correlate presence events (default: 900)",
    )
    parser.add_argument(
        "--max-plist-mb",
        type=int,
        default=50,
        help="Maximum plist size to fully parse (default: 50)",
    )
    parser.add_argument(
        "--tz",
        type=str,
        help="Optional IANA timezone (e.g., 'UTC', 'America/Los_Angeles')",
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.input_path.exists():
        logger.info(f"Error: Input directory not found: {args.input_path}")
        return

    # Create output directory
    output_dir = args.out
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up progress reporting (6 steps)
    progress = get_progress()
    if progress:
        progress.set_total(6)

    try:
        # ==================================================================
        # Stage 1: Build summary from exemplar directory
        # ==================================================================

        # Set up OUI lookup
        oui_map_path = args.oui_map
        if oui_map_path is None and wifi_summary.DEFAULT_OUI_MAP_PATH.exists():
            oui_map_path = wifi_summary.DEFAULT_OUI_MAP_PATH

        oui_lookup = wifi_summary.enrich_with_oui_vendor(oui_map_path)

        # Create event collector
        collector = wifi_summary.NormalizedEventCollector(oui_lookup)

        # Parse timezone if provided
        tzinfo = None
        if args.tz:
            import zoneinfo

            try:
                tzinfo = zoneinfo.ZoneInfo(args.tz)
            except Exception as e:
                logger.info(f"Warning: Invalid timezone '{args.tz}': {e}")
                tzinfo = None

        # Build summary
        summary = wifi_summary.build_summary(
            args.input_path,
            tzinfo=tzinfo,
            max_plist_mb=args.max_plist_mb,
            collector=collector,
        )
        if progress:
            progress.advance(message="Built summary")

        # Write JSON summary
        json_path = output_dir / "wifi_summary.json"
        json_text = json.dumps(summary, indent=2) if args.pretty else json.dumps(summary)
        json_path.write_text(json_text + "\n", encoding="utf-8")
        if progress:
            progress.advance(message="Wrote JSON summary")

        # Write normalized events (NDJSON)
        events_path = output_dir / "wifi_events.ndjson"
        wifi_summary.write_ndjson(events_path, collector.events)
        if progress:
            progress.advance(message="Wrote events")

        # Write presence rollup (CSV)
        presence_rows = wifi_summary.build_presence_rollups(
            summary.get("dhcp_leases", []),
            collector.events,
            window_seconds=args.presence_window,
            tzinfo=tzinfo,
        )
        presence_path = output_dir / "wifi_presence_daily.csv"
        wifi_summary.write_presence_csv(presence_path, presence_rows)
        if progress:
            progress.advance(message="Wrote presence CSV")

        # ==================================================================
        # Stage 2: Generate HTML report
        # ==================================================================

        # Load the data we just created
        events_df = wifi_report.load_ndjson(events_path)
        presence_df = wifi_report.load_csv(presence_path)
        if progress:
            progress.advance(message="Loaded data for HTML")

        # Build HTML sections
        sections = wifi_report.build_summary_sections(summary, events_df, presence_df)

        # Compose HTML
        html = wifi_report.compose_html(summary.get("exemplar_root", str(args.input_path)), sections)

        # Write HTML report
        html_path = output_dir / "wifi_report.html"
        html_path.write_text(html, encoding="utf-8")
        if progress:
            progress.advance(message="Generated HTML report")

        # ==================================================================
        # Summary
        # ==================================================================

    except Exception as e:
        logger.info(f"    ✗ Error generating WiFi report: {e}")
        raise


if __name__ == "__main__":
    main()
