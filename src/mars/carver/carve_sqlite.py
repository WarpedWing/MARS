#!/usr/bin/env python3

"""
SQLite Carver v3.4
by WarpedWing Labs

A page-oriented forensic SQLite data carver with intelligent timestamp validation.

 - Outputs:
     • carved_databases/<base>_<UTC>/<base>_Carved_Recovered.sqlite (includes ts_confidence column)
     • carved_databases/<base>_<UTC>/<base>_carved_protobufs.jsonl   ← JSON Lines
     • (optional) carved_databases/<base>_<UTC>/carved_all.csv (includes ts_confidence column)
 - Protobuf decoding ON by default (disable with --no-protobuf)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add src to path for direct execution
if __name__ == "__main__" or __name__.endswith(".__main__"):
    src_dir = Path(__file__).resolve().parent.parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

# Import our extracted modules
from mars.carver.finders import (
    find_blob_candidates,
    find_text_runs,
    find_urls,
)
from mars.carver.output.writers import (
    analyze_text_quality,
    append_csv_batch,
    append_jsonl,
    open_out_db,
    write_csv,
)

# Protobuf decoder (bbpb is required)
from mars.carver.protobuf.decoder import decode_protobuf_bbpb

# Protobuf timestamp extractor
from mars.carver.protobuf.timestamp_extractor import should_keep_protobuf
from mars.carver.sqlite_utils import (
    check_schema_has_blobs,
    read_sqlite_header_info,
)
from mars.carver.timestamp.classifier import (
    classify_page_timestamps,
    validate_protobuf_timestamps,
)
from mars.carver.timestamp.patterns import (
    find_timestamp_candidates,
    set_timestamp_range,
)
from mars.carver.timestamp.text_scanner import scan_text_for_timestamps

# Timestamp validation (V2 - Classification-based)
from mars.carver.timestamp.types import (
    ClassificationStats,
    should_keep_timestamp,
)
from mars.config import ConfigLoader
from mars.utils.debug_logger import logger

# ---------------- Config ----------------
PROTOBUF_DECODER_NAME = "blackboxprotobuf"

BATCH_SIZE = 100  # Commit every N pages

# ---------------- Parallel Processing ----------------


def process_page_parallel(page_data):
    """
    Worker function for parallel page processing.

    Processes a single page and returns structured results.
    Must be a module-level function for multiprocessing.pickle.

    Args:
        page_data: Tuple of (page_num, page_bytes, page_size, cluster_pages,
                             clustering_on, should_decode_protobuf, filter_mode)

    Returns:
        Dict with timestamps, urls, text_runs, protobufs
    """
    (
        pno,
        page,
        page_size,
        _cluster_pages,
        _clustering_on,
        should_decode_protobuf,
        filter_mode,
    ) = page_data

    abs_page_start = pno * page_size

    results = {
        "page_no": pno,
        "timestamps": [],
        "urls": [],
        "text_runs": [],
        "protobufs": [],
    }

    # Extract URLs
    for off, url in find_urls(page):
        results["urls"].append((off, abs_page_start + off, url))

    # Extract text runs
    for start_off, end_off, txt in find_text_runs(page):
        results["text_runs"].append((start_off, end_off, abs_page_start + start_off, txt))

    # Extract timestamps (V2 classifier)
    page_url_offsets = find_urls(page)
    raw_candidates = find_timestamp_candidates(page)
    classified = classify_page_timestamps(page, raw_candidates, page_url_offsets)

    for finding in classified:
        # Filter by mode
        keep = should_keep_timestamp(finding.classification, filter_mode)
        if keep:
            results["timestamps"].append(
                {
                    "offset": finding.offset,
                    "abs_offset": abs_page_start + finding.offset,
                    "value": str(finding.value),
                    "format_type": finding.format_type,
                    "human": finding.human_readable,
                    "classification": finding.classification.value,
                    "reason": finding.reason,
                    "source_url": finding.source_url,
                    "field_name": finding.field_name,
                }
            )

    # Also get hex timestamps
    try:
        from mars.carver.timestamp.patterns import (
            find_hex_timestamp_candidates,
        )

        hex_timestamps = find_hex_timestamp_candidates(page)
        for offset, hex_val, kind, human in hex_timestamps:
            results["timestamps"].append(
                {
                    "offset": offset,
                    "abs_offset": abs_page_start + offset,
                    "value": hex_val,
                    "format_type": kind,
                    "human": human,
                    "classification": "confirmed_timestamp",
                    "reason": "Hexadecimal timestamp format",
                    "source_url": None,
                    "field_name": None,
                }
            )
    except (ImportError, AttributeError):
        pass  # Hex timestamp support not available

    # Extract protobufs
    if should_decode_protobuf:
        for off, blob in find_blob_candidates(page, min_len=16):
            abs_off = abs_page_start + off

            result = decode_protobuf_bbpb(blob)

            if not result:
                continue

            parsed = result["message"]

            # Filter out garbage protobufs
            keep, _ = should_keep_protobuf(parsed)
            if not keep:
                continue

            # Validate timestamps using V2 classifier (uses field names + Unfurl logic)
            ts_validation = validate_protobuf_timestamps(
                message=parsed,
            )

            results["protobufs"].append(
                {
                    "offset": off,
                    "abs_offset": abs_off,
                    "json": result["json"],
                    "schema": (json.dumps(result["schema"]) if result.get("schema") else None),
                    "field_count": len(parsed) if isinstance(parsed, dict) else 0,
                    "timestamp_count": ts_validation["timestamp_count"],
                    "timestamp_fields": ts_validation["timestamp_fields"],
                }
            )

    return results


# ---------------- Main ----------------


def main():
    # Load config for defaults
    config = ConfigLoader.load()
    carver_cfg = config.carver

    ap = argparse.ArgumentParser(description="SQLite Carver (forensic)")
    ap.add_argument("db", help="SQLite file to carve")
    ap.add_argument("--no-cluster", action="store_true", help="Disable page clustering")
    ap.add_argument(
        "--no-protobuf",
        action="store_true",
        default=not carver_cfg.decode_protobuf,
        help=f"Skip protobuf decoding (default: {'disabled' if not carver_cfg.decode_protobuf else 'enabled'})",
    )
    ap.add_argument(
        "--no-pretty-protobuf",
        action="store_true",
        default=not carver_cfg.pretty_json,
        help=f"Compact JSON output for protobufs (default: {'compact' if not carver_cfg.pretty_json else 'pretty'})",
    )
    ap.add_argument(
        "--ts-start",
        type=str,
        default=carver_cfg.ts_start,
        help=f"Start of timestamp validity range (YYYY-MM-DD, default: {carver_cfg.ts_start})",
    )
    ap.add_argument(
        "--ts-end",
        type=str,
        default=carver_cfg.ts_end,
        help=f"End of timestamp validity range (YYYY-MM-DD, default: {carver_cfg.ts_end})",
    )
    ap.add_argument(
        "--filter-mode",
        type=str,
        choices=["strict", "balanced", "permissive", "all"],
        default=carver_cfg.filter_mode,
        help=f"Timestamp filtering mode: strict (only confirmed), balanced (confirmed+likely), "
        f"permissive (exclude only confirmed IDs), all (no filtering) (default: {carver_cfg.filter_mode})",
    )
    ap.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for carved data (default: same directory as source database)",
    )
    ap.add_argument(
        "--csv",
        action="store_true",
        default=carver_cfg.csv_export,
        help=f"Enable CSV output (default: {carver_cfg.csv_export})",
    )
    ap.add_argument(
        "--parallel",
        action="store_true",
        default=carver_cfg.parallel_processing,
        help=f"Enable parallel processing for large files (default: {carver_cfg.parallel_processing})",
    )
    ap.add_argument(
        "--parallel-threshold",
        type=int,
        default=carver_cfg.parallel_threshold,
        help=f"Minimum file size in MB for parallel processing (default: {carver_cfg.parallel_threshold})",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=carver_cfg.max_workers,
        help="Number of worker processes (default: CPU count)",
    )
    args = ap.parse_args()

    # Parse timestamp range
    global TARGET_START, TARGET_END
    try:
        TARGET_START = datetime.strptime(args.ts_start, "%Y-%m-%d").replace(tzinfo=UTC)
        TARGET_END = datetime.strptime(args.ts_end, "%Y-%m-%d").replace(tzinfo=UTC)
        # Update timestamp_patterns module if available
        if set_timestamp_range:
            set_timestamp_range(TARGET_START, TARGET_END)
    except ValueError as e:
        logger.debug(f"Invalid timestamp range: {e}")
        sys.exit(1)

    src = Path(args.db)
    if not src.is_file():
        logger.debug(f"File not found: {src}")
        sys.exit(1)

    try:
        page_size, encoding, page_count = read_sqlite_header_info(src)
    except Exception as e:
        logger.debug(f"Failed to read SQLite header from {src}: {e}")
        sys.exit(1)

    # Check if schema has BLOB columns (informational only)
    schema_has_blobs = check_schema_has_blobs(src)

    # Determine if we should decode protobufs
    # DEFAULT: Always decode (forensic recovery priority)
    # Logic:
    # - If --no-protobuf: never decode
    # - Otherwise: always decode (even without BLOB columns)
    #   Rationale: Protobufs may exist in TEXT/INTEGER fields, deleted pages,
    #   or overflow pages. BlackBoxProtobuf is good at rejecting garbage.
    should_decode_protobuf = not args.no_protobuf

    # Determine output directory (default: same directory as source database)
    output_base = Path(args.output_dir) if args.output_dir else src.parent

    # Timestamped outputs
    now_tag = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    base = src.stem
    FILE_DIR = output_base / f"{base}_{now_tag}"
    FILE_DIR.mkdir(parents=True, exist_ok=False)

    out_sqlite = FILE_DIR / f"{base}_Carved_Recovered.sqlite"
    out_csv_main = FILE_DIR / f"{base}_carved_all.csv"
    out_jsonl_pb = FILE_DIR / f"{base}_carved_protobufs.jsonl"

    # Protobuf status reporting
    if should_decode_protobuf:
        decoder_info = f" (using {PROTOBUF_DECODER_NAME})"
        if schema_has_blobs:
            logger.debug(f"Protobuf decoding enabled - schema contains BLOBs{decoder_info}")
        else:
            logger.debug(f"Protobuf decoding enabled - searching all pages{decoder_info}")
    else:
        logger.debug("Protobuf decoding disabled by --no-protobuf")

    logger.debug("Using V2 classifier (Unfurl + time_decode)")

    db = open_out_db(out_sqlite)
    cur = db.cursor()

    total = page_count or (src.stat().st_size // page_size)
    cluster_pages = 16
    clustering_on = not args.no_cluster

    logger.debug("─────────────────────────────────────────────")
    seen_url_abs: set[int] = set()  # absolute offsets for URLs
    seen_txt_abs: set[int] = set()  # absolute offsets for text
    seen_pb_abs: set[int] = set()  # absolute offsets for protobufs

    # For CSVs - batch buffer
    rows_batch: list[list[Any]] = []

    # Initialize CSV with header (only if CSV output is enabled)
    if args.csv:
        write_csv(
            out_csv_main,
            [
                "page_no",
                "page_offset",
                "abs_offset",
                "cluster_id",
                "kind",
                "value_text",
                "value_text_clean",
                "text_quality",
                "timestamp_count",
                "timestamp_fields",
            ],
            [],
        )

    # Track classification stats
    classification_stats = ClassificationStats()

    # Determine if we should use parallel processing
    file_size_mb = src.stat().st_size / (1024 * 1024)
    use_parallel = (
        args.parallel and file_size_mb >= args.parallel_threshold and total > 100  # Need enough pages to benefit
    )

    if use_parallel:
        num_workers = args.workers if args.workers else min(os.cpu_count() or 4, 8)
        logger.debug(f"Parallel mode: {num_workers} workers ({file_size_mb:.1f} MB)")
        logger.debug(f"Estimated speedup: 2-3x on {num_workers}+ core systems")

        # Process pages in parallel
        processed = 0
        # Create pool BEFORE loading pages so workers initialize in parallel
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Read all pages into memory for parallel processing
            # Workers spawn and import modules while this happens
            logger.debug("Loading pages into memory...")
            pages_data = []
            with src.open("rb") as f:
                for pno in range(total):
                    page = f.read(page_size)
                    if not page:
                        break
                    pages_data.append(
                        (
                            pno,
                            page,
                            page_size,
                            cluster_pages,
                            clustering_on,
                            should_decode_protobuf,
                            args.filter_mode,
                        )
                    )

            logger.debug(f"Loaded {len(pages_data)} pages ({file_size_mb:.1f} MB)")
            logger.debug(f"Processing with {num_workers} workers...")

            # Use imap_unordered for better throughput (no head-of-line blocking)
            # Reduced chunksize from 100 to 20 for faster first progress
            for page_results in pool.imap_unordered(process_page_parallel, pages_data, chunksize=20):
                processed += 1
                pno = page_results["page_no"]
                cluster_id = (pno // cluster_pages) if clustering_on else None

                # Write URLs
                for off, abs_off, url in page_results["urls"]:
                    if abs_off in seen_url_abs:
                        continue
                    seen_url_abs.add(abs_off)

                    # Analyze text quality
                    quality_info = analyze_text_quality(str(url))

                    cur.execute(
                        "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                        "kind,value_text,value_text_clean,text_quality,timestamp_count,timestamp_fields) "
                        "VALUES (?,?,?,?,'url',?,?,?,0,NULL)",
                        (
                            pno,
                            off,
                            abs_off,
                            cluster_id,
                            url,
                            quality_info["cleaned"],
                            quality_info["quality"],
                        ),
                    )

                    if args.csv:
                        rows_batch.append(
                            [
                                pno,
                                off,
                                abs_off,
                                cluster_id,
                                "url",
                                url,
                                quality_info["cleaned"],
                                quality_info["quality"],
                                0,  # timestamp_count
                                None,  # timestamp_fields
                            ]
                        )

                # Write text runs
                for start_off, end_off, abs_off, txt in page_results["text_runs"]:
                    if abs_off in seen_txt_abs:
                        continue
                    seen_txt_abs.add(abs_off)

                    # Analyze text quality
                    quality_info = analyze_text_quality(txt)

                    # Scan text for timestamps
                    ts_info = scan_text_for_timestamps(txt)

                    # Pretty print timestamp fields (unless disabled)
                    if ts_info["timestamp_fields"]:
                        if args.no_pretty_protobuf:
                            ts_fields_json = json.dumps(ts_info["timestamp_fields"])
                        else:
                            ts_fields_json = json.dumps(ts_info["timestamp_fields"], indent=2)
                    else:
                        ts_fields_json = None

                    cur.execute(
                        "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                        "kind,value_text,value_text_clean,text_quality,"
                        "timestamp_count,timestamp_fields) "
                        "VALUES (?,?,?,?,'text',?,?,?,?,?)",
                        (
                            pno,
                            start_off,
                            abs_off,
                            cluster_id,
                            txt,
                            quality_info["cleaned"],
                            quality_info["quality"],
                            ts_info["timestamp_count"],
                            ts_fields_json,
                        ),
                    )

                    if args.csv:
                        rows_batch.append(
                            [
                                pno,
                                start_off,
                                abs_off,
                                cluster_id,
                                "text",
                                txt,
                                quality_info["cleaned"],
                                quality_info["quality"],
                                ts_info["timestamp_count"],
                                ts_fields_json,
                            ]
                        )

                # Write protobufs
                for pb in page_results["protobufs"]:
                    if pb["abs_offset"] in seen_pb_abs:
                        continue
                    seen_pb_abs.add(pb["abs_offset"])

                    # Pretty print timestamp fields (unless disabled)
                    if pb["timestamp_fields"]:
                        if args.no_pretty_protobuf:
                            ts_fields_json = json.dumps(pb["timestamp_fields"])
                        else:
                            ts_fields_json = json.dumps(pb["timestamp_fields"], indent=2)
                    else:
                        ts_fields_json = None
                    cur.execute(
                        "INSERT INTO carved_protobufs(parent_abs_offset,page_no,abs_offset,"
                        "json_pretty,schema,field_count,"
                        "timestamp_count,timestamp_fields,decoder_used) "
                        "VALUES (?,?,?,?,?,?,?,?,?)",
                        (
                            pb["abs_offset"],
                            pno,
                            pb["abs_offset"],
                            pb["json"],
                            pb["schema"],
                            pb["field_count"],
                            pb["timestamp_count"],
                            ts_fields_json,
                            PROTOBUF_DECODER_NAME,
                        ),
                    )

                    # Write to JSONL
                    jsonl_entry = {
                        "parent_abs_offset": pb["abs_offset"],
                        "page_no": pno,
                        "abs_offset": pb["abs_offset"],
                        "decoder": PROTOBUF_DECODER_NAME,
                        "field_count": pb["field_count"],
                        "protobuf": (json.loads(pb["json"]) if not args.no_pretty_protobuf else pb["json"]),
                        "timestamp_count": pb["timestamp_count"],
                        "timestamps": pb["timestamp_fields"],
                    }
                    with out_jsonl_pb.open("a", encoding="utf-8") as jf:
                        jf.write(json.dumps(jsonl_entry, default=str) + "\n")

                # Batch commit every 100 pages
                if processed % 100 == 0:
                    db.commit()
                    if args.csv and rows_batch:
                        append_csv_batch(out_csv_main, rows_batch)
                        rows_batch.clear()

        # Final commit
        db.commit()
        if args.csv and rows_batch:
            append_csv_batch(out_csv_main, rows_batch)

    else:
        # Sequential processing
        if args.parallel:
            logger.debug(f"File too small for parallel ({file_size_mb:.1f} MB < {args.parallel_threshold} MB)")

        with src.open("rb") as f:
            for pno in range(total):
                try:
                    abs_page_start = pno * page_size
                    page = f.read(page_size)
                    if not page:
                        break

                    cluster_id = (pno // cluster_pages) if clustering_on else None

                    # URLs
                    for off, url in find_urls(page):
                        abs_off = abs_page_start + off
                        if abs_off in seen_url_abs:
                            continue
                        seen_url_abs.add(abs_off)

                        # Analyze text quality
                        quality_info = analyze_text_quality(str(url))

                        cur.execute(
                            "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                            "kind,value_text,value_text_clean,text_quality,"
                            "timestamp_count,timestamp_fields) "
                            "VALUES (?,?,?,?, 'url',?,?,?,0,NULL)",
                            (
                                pno,
                                off,
                                abs_off,
                                cluster_id,
                                url,
                                quality_info["cleaned"],
                                quality_info["quality"],
                            ),
                        )
                        rows_batch.append(
                            [
                                pno,
                                off,
                                abs_off,
                                cluster_id,
                                "url",
                                url,
                                quality_info["cleaned"],
                                quality_info["quality"],
                                0,  # timestamp_count
                                None,  # timestamp_fields
                            ]
                        )

                    # Text
                    for off_start, off_end, txt in find_text_runs(page, min_len=8):
                        abs_off = abs_page_start + off_start
                        if abs_off in seen_txt_abs:
                            continue
                        seen_txt_abs.add(abs_off)

                        # Analyze text quality
                        quality_info = analyze_text_quality(txt)

                        # Scan text for timestamps
                        ts_info = scan_text_for_timestamps(txt)

                        # Pretty print timestamp fields (unless disabled)
                        if ts_info["timestamp_fields"]:
                            if args.no_pretty_protobuf:
                                ts_fields_json = json.dumps(ts_info["timestamp_fields"])
                            else:
                                ts_fields_json = json.dumps(ts_info["timestamp_fields"], indent=2)
                        else:
                            ts_fields_json = None

                        cur.execute(
                            "INSERT OR IGNORE INTO carved_all(page_no,page_offset,abs_offset,cluster_id,"
                            "kind,value_text,value_text_clean,text_quality,"
                            "timestamp_count,timestamp_fields) "
                            "VALUES (?,?,?,?, 'text',?,?,?,?,?)",
                            (
                                pno,
                                off_start,
                                abs_off,
                                cluster_id,
                                txt,
                                quality_info["cleaned"],
                                quality_info["quality"],
                                ts_info["timestamp_count"],
                                ts_fields_json,
                            ),
                        )

                        rows_batch.append(
                            [
                                pno,
                                off_start,
                                abs_off,
                                cluster_id,
                                "text",
                                txt,
                                quality_info["cleaned"],
                                quality_info["quality"],
                                ts_info["timestamp_count"],
                                ts_fields_json,
                            ]
                        )

                    # Protobufs → JSONL (using blackboxprotobuf for better decoding)
                    if should_decode_protobuf:
                        for off, blob in find_blob_candidates(page, min_len=16):
                            abs_off = abs_page_start + off
                            # Deduplicate by absolute offset
                            if abs_off in seen_pb_abs:
                                continue

                            # Use bbpb decoder
                            result = decode_protobuf_bbpb(blob)
                            if not result:
                                continue
                            parsed = result["message"]
                            jtxt = result["json"]
                            schema = json.dumps(result["schema"])
                            field_count = len(parsed) if isinstance(parsed, dict) else 0

                            # Filter out garbage protobufs
                            keep, _ = should_keep_protobuf(parsed)
                            if not keep:
                                continue  # Skip this protobuf - it's noise

                            seen_pb_abs.add(abs_off)

                            # Validate timestamps using V2 classifier (uses field names + Unfurl logic)
                            ts_validation = validate_protobuf_timestamps(
                                message=parsed,
                            )

                            # Prepare timestamp analysis for database
                            ts_count = ts_validation["timestamp_count"]
                            ts_entries = ts_validation["timestamp_fields"]

                            # Convert to JSON string for database storage
                            if ts_entries:
                                if args.no_pretty_protobuf:
                                    ts_fields = json.dumps(ts_entries)
                                else:
                                    ts_fields = json.dumps(ts_entries, indent=2)
                            else:
                                ts_fields = None

                            # Insert into database with rich metadata
                            cur.execute(
                                "INSERT INTO carved_protobufs(parent_abs_offset,page_no,abs_offset,"
                                "json_pretty,schema,field_count,"
                                "timestamp_count,timestamp_fields,decoder_used) "
                                "VALUES (?,?,?,?,?,?,?,?,?)",
                                (
                                    abs_off,
                                    pno,
                                    abs_off,
                                    jtxt,
                                    schema,
                                    field_count,
                                    ts_count,
                                    ts_fields,
                                    PROTOBUF_DECODER_NAME,
                                ),
                            )

                            # Build JSONL output with richer metadata
                            jsonl_entry = {
                                "parent_abs_offset": abs_off,
                                "page_no": pno,
                                "abs_offset": abs_off,
                                "decoder": PROTOBUF_DECODER_NAME,
                                "field_count": field_count,
                                "protobuf": (json.loads(jtxt) if not args.no_pretty_protobuf else parsed),
                            }

                            # Add schema if available
                            if schema:
                                jsonl_entry["schema"] = json.loads(schema)

                            # Add timestamp analysis to JSONL if available
                            if ts_count > 0:
                                jsonl_entry["timestamp_analysis"] = {
                                    "timestamp_count": ts_count,
                                    "timestamps": ts_entries,
                                }

                            append_jsonl(out_jsonl_pb, jsonl_entry)

                    # Batch commit every BATCH_SIZE pages
                    if (pno + 1) % BATCH_SIZE == 0:
                        db.commit()
                        if args.csv:
                            append_csv_batch(out_csv_main, rows_batch)
                            rows_batch.clear()

                except Exception as e:
                    logger.debug(f"Error processing page {pno}: {e}")
                    # Commit what we have so far to avoid losing all progress
                    db.commit()
                    if args.csv:
                        append_csv_batch(out_csv_main, rows_batch)
                        rows_batch.clear()
                    # Continue with next page
                    continue

    # Final commit for remaining rows
    if rows_batch and args.csv:
        append_csv_batch(out_csv_main, rows_batch)
        rows_batch.clear()

    # Finalize DB (single file, no WAL/SHM)
    db.commit()
    cur.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    cur.execute("PRAGMA journal_mode=DELETE;")
    db.commit()
    db.close()

    logger.debug("─────────────────────────────────────────────")

    # Print classification summary
    if classification_stats:
        logger.debug(classification_stats.get_summary())

    logger.debug("Exported:")
    logger.debug(f"   • {out_sqlite}")
    if args.csv:
        logger.debug(f"   • {out_csv_main}")
    logger.debug(f"   • {out_jsonl_pb}")
    logger.debug("Done.")


if __name__ == "__main__":
    main()
