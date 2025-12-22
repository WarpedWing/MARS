#!/usr/bin/env python3
"""
Biome SEGB Parser for MARS.

Parses macOS Biome SEGB files (v1 and v2) and extracts timestamps and data records.
Optionally decodes protobuf payloads within the records.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ccl_segb

from mars.carver.protobuf.decoder import decode_protobuf_bbpb, extract_strings_from_message
from mars.report_modules.progress_interface import get_progress
from mars.utils.debug_logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

# Pattern to match bundle identifiers in data (com.*, etc.)
BUNDLE_ID_PATTERN = re.compile(rb"(?:com|net|org|io|app)\.[a-zA-Z0-9_-]+(?:\.[a-zA-Z0-9_-]+)+")


def try_protobuf_decode(data: bytes) -> tuple[dict[str, Any] | None, bool]:
    """Try protobuf decode with and without skipping header bytes.

    Some SEGB records have protobuf starting at byte 0, others at byte 8.
    This function tries both approaches.

    Returns:
        Tuple of (decoded_dict, needed_skip) where needed_skip indicates if 8-byte skip was used.
    """
    if not data:
        return None, False

    # Try without skip first (many records like finder have protobuf at byte 0)
    decoded = decode_protobuf_bbpb(data)
    if decoded and decoded.get("message"):
        return decoded, False

    # Try with skip (some records like siri.metrics have 8-byte header)
    if len(data) > 8:
        decoded = decode_protobuf_bbpb(data[8:])
        if decoded and decoded.get("message"):
            return decoded, True

    return None, False


def iter_segb_files(root: Path) -> Iterator[Path]:
    """Iterate over all files in directory that might be SEGB files.

    SEGB files don't have a standard extension - they're typically named
    with numeric identifiers.
    """
    for path in root.rglob("*"):
        if path.is_file() and not path.name.startswith("."):
            yield path


def extract_bundle_id(data: bytes) -> str | None:
    """Extract bundle identifier from SEGB record data.

    Bundle IDs like 'com.apple.siri.metrics.ExperimentationExtension.hourly'
    are embedded in the data content (search entire data, not just after header).
    """
    if not data:
        return None
    # Search entire data for bundle ID pattern (it's usually after protobuf header bytes)
    match = BUNDLE_ID_PATTERN.search(data)
    if match:
        return match.group(0).decode("utf-8", errors="ignore")
    return None


def is_empty_record(data: bytes) -> bool:
    """Check if record data is all zeros or empty."""
    if not data:
        return True
    return all(b == 0 for b in data)


def decode_data_text(data: bytes, skip_header: bool = False) -> str | None:
    """Attempt to decode data as readable text.

    Args:
        data: Raw data bytes
        skip_header: If True, skip first 8 bytes before decoding

    Returns printable text content or None if not decodable.
    """
    if not data:
        return None
    # Skip first 8 bytes if needed
    text_data = data[8:] if skip_header and len(data) > 8 else data
    try:
        text = text_data.decode("utf-8", errors="replace")
        # Keep only printable characters, replace others with space
        readable = "".join(c if c.isprintable() or c in "\n\t" else " " for c in text)
        # Collapse multiple spaces and strip
        readable = " ".join(readable.split())
        return readable if readable else None
    except Exception:
        return None


def format_timestamp(ts: Any) -> str | None:
    """Format timestamp to ISO format string."""
    if ts is None:
        return None
    try:
        return ts.isoformat()
    except (ValueError, OSError, AttributeError):
        return str(ts)


def parse_segb_file(file_path: Path) -> list[dict[str, Any]]:
    """Parse a single SEGB file and return list of record dicts.

    Returns empty list if file is not a valid SEGB file.
    """
    records: list[dict[str, Any]] = []

    try:
        for record in ccl_segb.read_segb_file(file_path):
            # Determine SEGB version based on available attributes
            # v2 records have a 'metadata' attribute, v1 records have 'timestamp1'/'timestamp2'
            metadata = getattr(record, "metadata", None)
            is_v2 = metadata is not None

            offset = getattr(record, "data_start_offset", 0)
            data = getattr(record, "data", b"")

            # Skip empty records (all zeros or no data)
            if is_empty_record(data):
                continue

            if is_v2:
                # SEGB v2 record structure
                metadata_offset = getattr(metadata, "metadata_offset", None)
                state_obj = getattr(metadata, "state", None)
                state = getattr(state_obj, "name", str(state_obj)) if state_obj else None
                timestamp1 = format_timestamp(getattr(metadata, "creation", None))
                timestamp2 = None
            else:
                # SEGB v1 record structure
                metadata_offset = None
                state_obj = getattr(record, "state", None)
                state = str(state_obj) if state_obj is not None else None
                timestamp1 = format_timestamp(getattr(record, "timestamp1", None))
                timestamp2 = format_timestamp(getattr(record, "timestamp2", None))

            # Extract bundle identifier from data content
            bundle_id = extract_bundle_id(data)

            # Attempt protobuf decode (try both with and without 8-byte skip)
            protobuf_json: str | None = None
            protobuf_strings: list[str] = []
            needed_skip = False
            if data:
                decoded, needed_skip = try_protobuf_decode(data)
                if decoded:
                    protobuf_json = decoded.get("json")
                    if decoded.get("message"):
                        protobuf_strings = extract_strings_from_message(decoded["message"])

            # For data_text, prefer protobuf_strings (cleaner) over raw text decode
            if protobuf_strings:
                data_text = "; ".join(protobuf_strings)
            else:
                # Fall back to raw text decode if protobuf failed
                data_text = decode_data_text(data, skip_header=needed_skip)

            records.append(
                {
                    "offset": offset,
                    "metadata_offset": metadata_offset,
                    "state": state,
                    "timestamp1": timestamp1,
                    "timestamp2": timestamp2,
                    "stream_type": "v2" if is_v2 else "v1",
                    "bundle_id": bundle_id,
                    "data_size": len(data) if data else 0,
                    "data_text": data_text,
                    "data_hex": data.hex()
                    if data and len(data) <= 1024
                    else (data[:1024].hex() + "..." if data else ""),
                    "protobuf_json": protobuf_json,
                    "protobuf_strings": "; ".join(protobuf_strings) if protobuf_strings else None,
                }
            )

    except ValueError as e:
        # Not a valid SEGB file - this is expected for non-SEGB files
        logger.debug(f"Not a SEGB file: {file_path}: {e}")
    except Exception as e:
        logger.debug(f"Error parsing SEGB file {file_path}: {e}")

    return records


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Write parsed records to CSV file."""
    fieldnames = [
        "source_file",
        "bundle_id",
        "stream_type",
        "offset",
        "metadata_offset",
        "state",
        "timestamp1",
        "timestamp2",
        "data_size",
        "data_text",
        "data_hex",
        "protobuf_json",
        "protobuf_strings",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse macOS Biome SEGB files and extract records.")
    parser.add_argument("indir", type=Path, help="Input directory containing Biome stream files")
    parser.add_argument("--out", type=Path, default=Path(), help="Output directory (default: cwd)")
    args = parser.parse_args()

    indir = args.indir
    if not indir.exists():
        parser.error(f"{indir} does not exist")

    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    # Collect all potential SEGB files
    segb_files = list(iter_segb_files(indir))
    if not segb_files:
        logger.debug(f"[warn] No files detected under {indir}")
        return

    # Set up progress reporting
    progress = get_progress()
    if progress:
        progress.set_total(len(segb_files))

    all_rows: list[dict[str, Any]] = []
    parsed_count = 0
    failed_count = 0

    for file_path in segb_files:
        records = parse_segb_file(file_path)

        if records:
            for record in records:
                record["source_file"] = str(file_path)
            all_rows.extend(records)
            parsed_count += 1
        else:
            failed_count += 1

        if progress:
            progress.advance()

    # Write output CSV
    csv_path = outdir / "biome_records.csv"
    write_csv(all_rows, csv_path)

    logger.debug(f"Parsed {parsed_count} SEGB files, {failed_count} non-SEGB files, {len(all_rows)} total records")


if __name__ == "__main__":
    main()
