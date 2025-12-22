#!/usr/bin/env python3
"""
Batch SQLite Carver
Orchestrates carving for multiple databases in the candidate pipeline.

Integrates with the candidate processing workflow to carve databases
that failed validation or are empty.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from mars.utils.debug_logger import logger


def carve_database(
    source_db: Path,
    output_dir: Path,
    output_name: str | None = None,
    ts_start: str = "2015-01-01",
    ts_end: str = "2030-01-01",
    filter_mode: str = "permissive",
    enable_protobuf: bool = True,
    enable_csv: bool = False,
    enable_parallel: bool = False,
    parallel_threshold: int = 50,
) -> dict:
    """
    Carve a single SQLite database using the forensic carver.

    Args:
        source_db: Path to source SQLite database
        output_dir: Directory for carved output
        output_name: Optional custom output name (default: source_db.stem + "_carved")
        ts_start: Start of timestamp validity range (YYYY-MM-DD)
        ts_end: End of timestamp validity range (YYYY-MM-DD)
        filter_mode: Timestamp filtering mode (strict/balanced/permissive/all)
        enable_protobuf: Enable protobuf decoding
        enable_csv: Enable CSV output
        enable_parallel: Enable parallel processing
        parallel_threshold: Minimum file size in MB for parallel processing

    Returns:
        Dict with carving results:
            - success: bool
            - output_folder: Path (output directory with carved database)
            - carved_db: Path (carved SQLite database)
            - protobuf_jsonl: Path (protobuf JSONL file)
            - csv_file: Path | None (CSV file if enabled)
            - error: str | None (error message if failed)

    Output structure:
        output_dir/
            {output_name}/
                {output_name}.sqlite
                {output_name}_protobufs.jsonl
    """
    if not source_db.exists():
        error = f"Source database not found: {source_db}"
        logger.debug(error)
        return {
            "success": False,
            "output_folder": None,
            "carved_db": None,
            "protobuf_jsonl": None,
            "csv_file": None,
            "error": error,
        }

    # Determine output name
    if output_name is None:
        output_name = f"{source_db.stem}_carved"

    # Create output directory structure
    output_folder = output_dir / output_name
    output_folder.mkdir(parents=True, exist_ok=True)

    # Output paths
    carved_db = output_folder / f"{output_name}.sqlite"
    protobuf_jsonl = output_folder / f"{output_name}_protobufs.jsonl"
    csv_file = output_folder / f"{output_name}_all.csv" if enable_csv else None

    # Build carver command
    carver_script = Path(__file__).parent / "carve_sqlite.py"
    cmd = [
        sys.executable,
        str(carver_script),
        str(source_db),
        "--output-dir",
        str(output_folder),
        "--ts-start",
        ts_start,
        "--ts-end",
        ts_end,
        "--filter-mode",
        filter_mode,
    ]

    # Add optional flags
    if not enable_protobuf:
        cmd.append("--no-protobuf")

    if enable_csv:
        cmd.append("--csv")

    if enable_parallel:
        cmd.append("--parallel")
        cmd.extend(["--parallel-threshold", str(parallel_threshold)])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error = f"Carver failed with exit code {result.returncode}"
            logger.debug(error)
            if result.stderr:
                logger.debug(f"  Stderr: {result.stderr[:500]}")
            return {
                "success": False,
                "output_folder": output_folder,
                "carved_db": None,
                "protobuf_jsonl": None,
                "csv_file": None,
                "error": error,
            }

        # The carver creates timestamped output dirs by default
        # Find the carved database in the output folder
        carved_dbs = list(output_folder.glob("**/*_Carved_Recovered.sqlite"))
        if not carved_dbs:
            error = "Carved database not found in output"
            logger.debug(error)
            return {
                "success": False,
                "output_folder": output_folder,
                "carved_db": None,
                "protobuf_jsonl": None,
                "csv_file": None,
                "error": error,
            }

        # Move carved database to our target location
        actual_carved_db = carved_dbs[0]
        actual_carved_db.rename(carved_db)

        # Find and move protobuf JSONL
        pb_jsonls = list(output_folder.glob("**/*_carved_protobufs.jsonl"))
        if pb_jsonls:
            pb_jsonls[0].rename(protobuf_jsonl)

        # Find and move CSV if enabled
        if enable_csv:
            csvs = list(output_folder.glob("**/*_carved_all.csv"))
            if csvs and csv_file:
                csvs[0].rename(csv_file)

        # Remove the timestamped subdirectory created by carver
        for subdir in output_folder.iterdir():
            if subdir.is_dir() and subdir != output_folder:
                import shutil

                shutil.rmtree(subdir)

        return {
            "success": True,
            "output_folder": output_folder,
            "carved_db": carved_db,
            "protobuf_jsonl": protobuf_jsonl,
            "csv_file": csv_file,
            "error": None,
        }

    except Exception as e:
        error = f"Carver exception: {e}"
        logger.debug(error)
        return {
            "success": False,
            "output_folder": output_folder,
            "carved_db": None,
            "protobuf_jsonl": None,
            "csv_file": None,
            "error": error,
        }


def batch_carve_databases(
    databases: list[Path],
    output_dir: Path,
    ts_start: str = "2015-01-01",
    ts_end: str = "2030-01-01",
    filter_mode: str = "permissive",
    enable_protobuf: bool = True,
    enable_csv: bool = False,
    enable_parallel: bool = True,
    parallel_threshold: int = 50,
    console: Any | None = None,
    config: Any | None = None,
) -> dict:
    """
    Carve multiple SQLite databases in batch.

    Args:
        databases: List of paths to SQLite databases to carve
        output_dir: Base directory for carved output
        ts_start: Start of timestamp validity range (YYYY-MM-DD)
        ts_end: End of timestamp validity range (YYYY-MM-DD)
        filter_mode: Timestamp filtering mode (strict/balanced/permissive/all)
        enable_protobuf: Enable protobuf decoding
        enable_csv: Enable CSV output
        enable_parallel: Enable parallel processing
        parallel_threshold: Minimum file size in MB for parallel processing
        console: Optional Rich console for progress display

    Returns:
        Dict with batch results:
            - total: int (total databases to carve)
            - success: int (successfully carved)
            - failed: int (failed to carve)
            - results: list[dict] (individual carving results)
    """
    results = []
    success_count = 0
    failed_count = 0

    # Use Rich progress bar with parallel processing if console is provided
    if console:
        import threading
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from mars.utils.progress_utils import create_standard_progress

        # Thread-safe counters
        results_lock = threading.Lock()

        def carve_single_db(db_path: Path) -> dict:
            """Carve a single database (thread-safe wrapper)."""
            result = carve_database(
                source_db=db_path,
                output_dir=output_dir,
                output_name=None,  # Auto-generate from source name
                ts_start=ts_start,
                ts_end=ts_end,
                filter_mode=filter_mode,
                enable_protobuf=enable_protobuf,
                enable_csv=enable_csv,
                enable_parallel=enable_parallel,
                parallel_threshold=parallel_threshold,
            )
            return {"source_db": db_path, "result": result}

        with create_standard_progress(
            "Carving databases",
            console=console,
            show_count=True,
            show_percentage=False,
            show_time="elapsed",
            config=config,
        ) as progress:
            task = progress.add_task("Byte carving...", total=len(databases))

            # Process databases in parallel
            # Use min of 4 workers or number of databases
            num_workers = min(4, len(databases))

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit all carving jobs
                future_to_db = {executor.submit(carve_single_db, db_path): db_path for db_path in databases}

                # Collect results as they complete
                for future in as_completed(future_to_db):
                    try:
                        result_dict = future.result()

                        with results_lock:
                            results.append(result_dict)

                            if result_dict["result"]["success"]:
                                success_count += 1
                            else:
                                failed_count += 1

                        progress.update(task, completed=len(results))

                    except Exception as exc:
                        db_path = future_to_db[future]
                        logger.debug(f"  [red]✗[/red] Exception carving {db_path}: {exc}")
                        with results_lock:
                            failed_count += 1
                        progress.update(task, completed=len(results))

    else:
        # Fallback to simple loop without progress bar
        for idx, db_path in enumerate(databases, 1):
            result = carve_database(
                source_db=db_path,
                output_dir=output_dir,
                output_name=None,  # Auto-generate from source name
                ts_start=ts_start,
                ts_end=ts_end,
                filter_mode=filter_mode,
                enable_protobuf=enable_protobuf,
                enable_csv=enable_csv,
                enable_parallel=enable_parallel,
                parallel_threshold=parallel_threshold,
            )

            results.append(
                {
                    "source_db": db_path,
                    "result": result,
                }
            )

            if result["success"]:
                success_count += 1
            else:
                failed_count += 1
                logger.debug(f"  [red]✗[/red] Failed: {result.get('error')}")

    logger.debug(f"[green][✓][/green] Batch carving complete: {success_count} succeeded, {failed_count} failed")

    return {
        "total": len(databases),
        "success": success_count,
        "failed": failed_count,
        "results": results,
    }
