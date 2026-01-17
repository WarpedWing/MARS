#!/usr/bin/env python3
"""dfVFS workspace management utilities for exemplar scanner.

Handles decompression of archived databases and logs in the dfVFS workspace,
making them available for normal processing by the scanner.
"""

from __future__ import annotations

import bz2
import gzip
import hashlib
import shutil
import zipfile
from typing import IO, TYPE_CHECKING, cast

from mars.utils.compression_utils import recover_gzip

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from mars.pipeline.mount_utils.dfvfs_exporter import ExportRecord


def decompress_workspace_archives(
    workspace: Path,
    manifest: dict[Path, ExportRecord],
    debug_callback: Callable[[str], None] | None = None,
) -> dict[Path, ExportRecord]:
    """Decompress .gz, .bz2, and .zip files in dfVFS workspace after export.

    This processes archived databases (like Powerlog .gz files), logs,
    and ZIP archives (like Biome streams), decompressing them so they
    can be processed normally by the scanner.
    Returns new manifest entries for decompressed files.

    Uses centralized compression_utils for decompression and recovery.

    Args:
        workspace: Path to dfVFS workspace directory
        manifest: Current manifest dict (will not be modified)
        debug_callback: Optional callback for debug messages

    Returns:
        Dictionary of new manifest entries to add (decompressed files)
    """

    def debug(message: str):
        if debug_callback:
            debug_callback(message)

    if not workspace or not workspace.exists():
        return {}

    # Find all compressed files in workspace (case-insensitive for .ZIP)
    compressed_files = []
    for pattern in ["**/*.gz", "**/*.bz2", "**/*.zip", "**/*.ZIP"]:
        # Sort for deterministic order
        compressed_files.extend(sorted(workspace.glob(pattern)))

    # Deduplicate (in case .zip and .ZIP match same files on case-insensitive FS)
    compressed_files = sorted(set(compressed_files))

    if not compressed_files:
        return {}

    debug(f"[decompress] Found {len(compressed_files)} archived files to decompress")

    success_count = 0
    failed_count = 0

    # Collect new manifest entries to add
    new_manifest_entries: dict[Path, ExportRecord] = {}

    for compressed_file in compressed_files:
        try:
            # Handle ZIP files differently - they contain multiple files
            if compressed_file.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(compressed_file, "r") as zip_ref:
                        # Extract to same directory as the ZIP file
                        extract_dir = compressed_file.parent
                        zip_ref.extractall(extract_dir)
                        extracted_names = zip_ref.namelist()

                    debug(f"[decompress] Extracted {len(extracted_names)} file(s) from {compressed_file.name}")
                    success_count += 1

                    # Find the original manifest entry for the ZIP file
                    for export_path, record in sorted(manifest.items()):
                        if export_path == compressed_file:
                            # Create manifest entries for each extracted file
                            from mars.pipeline.mount_utils.dfvfs_exporter import (
                                ExportRecord,
                            )

                            for extracted_name in extracted_names:
                                # Skip directories
                                if extracted_name.endswith("/"):
                                    continue

                                extracted_path = extract_dir / extracted_name
                                if not extracted_path.exists():
                                    continue

                                # Calculate MD5
                                md5_hash = hashlib.md5()
                                with extracted_path.open("rb") as f:
                                    for chunk in iter(lambda: f.read(65536), b""):
                                        md5_hash.update(chunk)

                                # Virtual path: parent ZIP path + extracted filename
                                virtual_base = record.virtual_path.rsplit(".", 1)[0]  # Remove .zip
                                extracted_record = ExportRecord(
                                    virtual_path=f"{virtual_base}/{extracted_name}",
                                    export_path=extracted_path,
                                    target_name=record.target_name,
                                    md5=md5_hash.hexdigest(),
                                    size=extracted_path.stat().st_size,
                                    file_created=record.file_created,
                                    file_modified=record.file_modified,
                                    file_accessed=record.file_accessed,
                                )
                                new_manifest_entries[extracted_path] = extracted_record
                            break
                except zipfile.BadZipFile as e:
                    debug(f"[decompress] Bad ZIP file {compressed_file.name}: {e}")
                    failed_count += 1
                except Exception as e:
                    debug(f"[decompress] ZIP extraction failed for {compressed_file.name}: {e}")
                    failed_count += 1
                continue

            # Determine output filename (strip .gz or .bz2 extension)
            if compressed_file.suffix in (".gz", ".bz2"):
                decompressed_file = compressed_file.with_suffix("")
            else:
                continue

            # Try standard decompression first
            decompressed_successfully = False

            if compressed_file.suffix == ".gz":
                try:
                    with (
                        gzip.open(compressed_file, "rb") as f_in,
                        decompressed_file.open("wb") as f_out,
                    ):
                        fin = cast("IO[bytes]", f_in)
                        fout = cast("IO[bytes]", f_out)
                        shutil.copyfileobj(fin, fout)
                    decompressed_successfully = True
                except Exception as e:
                    # Try gzrecover
                    debug(f"[decompress] Standard gzip failed for {compressed_file.name}, trying gzrecover: {e}")
                    recovered_path = recover_gzip(
                        compressed_file,
                        decompressed_file,
                        timeout=30,
                    )
                    if recovered_path:
                        decompressed_successfully = True
            elif compressed_file.suffix == ".bz2":
                try:
                    with (
                        bz2.open(compressed_file, "rb") as f_in,
                        decompressed_file.open("wb") as f_out,
                    ):
                        fin = cast("IO[bytes]", f_in)
                        fout = cast("IO[bytes]", f_out)
                        shutil.copyfileobj(fin, fout)
                    decompressed_successfully = True
                except Exception as e:
                    # bzip2recover is not usable - it creates multiple block files
                    # that need individual testing and concatenation
                    debug(f"[decompress] bz2 decompression failed for {compressed_file.name}: {e}")

            if decompressed_successfully:
                success_count += 1

                # Update manifest if this file is tracked
                # Find the corresponding export record and add decompressed version
                for export_path, record in sorted(manifest.items()):
                    if export_path == compressed_file:
                        # Calculate MD5 of decompressed file
                        md5_hash = hashlib.md5()
                        with decompressed_file.open("rb") as f:
                            for chunk in iter(lambda: f.read(65536), b""):
                                md5_hash.update(chunk)

                        # Create new record for decompressed version
                        from mars.pipeline.mount_utils.dfvfs_exporter import (
                            ExportRecord,
                        )

                        decompressed_record = ExportRecord(
                            virtual_path=record.virtual_path.rsplit(".", 1)[0],  # Remove .gz/.bz2
                            export_path=decompressed_file,
                            target_name=record.target_name,
                            md5=md5_hash.hexdigest(),
                            size=decompressed_file.stat().st_size,
                            file_created=record.file_created,
                            file_modified=record.file_modified,
                            file_accessed=record.file_accessed,
                        )
                        # Collect for return
                        new_manifest_entries[decompressed_file] = decompressed_record
                        break
            else:
                failed_count += 1

        except Exception as e:
            debug(f"[decompress] Failed to decompress {compressed_file.name}: {e}")
            failed_count += 1

    # Report results
    if new_manifest_entries:
        debug(f"[decompress] Added {len(new_manifest_entries)} decompressed files to manifest")

    if success_count > 0:
        debug(f"[decompress] Successfully decompressed {success_count} archive(s)")
    if failed_count > 0:
        debug(f"[decompress] Failed to decompress {failed_count} archive(s)")

    return new_manifest_entries
