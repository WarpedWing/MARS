#!/usr/bin/env python3
"""Variant creation and selection operations for database recovery.

This module handles the creation of database variants through various recovery methods:
- Original (O): Direct file access
- Clone (C): SQLite .clone operation
- Recover (R): SQLite .recover operation

It also implements the selection logic to choose the best variant based on data quality metrics.
"""

from __future__ import annotations

import bz2
import contextlib
import gzip
import shutil
import sqlite3
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from mars.pipeline.lf_processor.db_reconstructor import cleanup_wal_files
from mars.utils.debug_logger import logger

from .db_variant_selector_helpers import (
    _which_sqlite3,
    make_case_dir,
    safe_copy,
)
from .models import DBMeta, Variant

if TYPE_CHECKING:
    from collections.abc import Sequence

# Import centralized compression utilities for .gz handling
try:
    from mars.utils.compression_utils import recover_gzip

    COMPRESSION_UTILS_AVAILABLE = True
except ImportError:
    COMPRESSION_UTILS_AVAILABLE = False

    # Define stub function to satisfy type checker
    def recover_gzip(*args, **kwargs) -> Path | None:  # type: ignore
        return None


# Salvage table configuration - imported or fallback
try:
    from mars.config.schema import SchemaComparisonConfig

    _config = SchemaComparisonConfig()
    SALVAGE_TABLE_CANON = _config.salvage_tables
except ImportError:
    # Fallback if config not available (standalone mode)
    SALVAGE_TABLE_CANON = {"lost_and_found"}


# Variant priority for choosing best candidate (higher wins).
# When all variants have equal data scores, prefer: O > C > R > D (closest to original)
# Note: Variant X is not listed here - it's a semantic marker applied AFTER selection
#       (X uses same file as O, but signals "empty database for byte carving")
VARIANT_PRIORITY: dict[str, tuple[int, int]] = {
    "O": (10, 0),  # highest priority: original file, closest to original
    "C": (8, 1),  # clone (integrity check)
    "R": (6, 2),  # recovered (sqlite3 .recover)
    "D": (4, 3),  # dissect_rebuilt (lowest priority: most reconstructed)
}


def resolve_variant_priority(tag: str) -> tuple[int, int]:
    """Get priority tuple for a variant tag.

    Args:
        tag: Variant tag ("O", "C", "R", "D", or "X")

    Returns:
        Tuple of (priority, closeness) where higher priority wins,
        and lower closeness wins when priorities are equal
    """
    return VARIANT_PRIORITY.get(tag, (0, 0))


def is_salvage_name(name: str) -> bool:
    """Check if a table name is a salvage/lost & found table.

    Args:
        name: Table name to check

    Returns:
        True if name matches salvage table patterns
    """
    return name.lower() in SALVAGE_TABLE_CANON


def _variant_score(meta: DBMeta) -> tuple:
    """Calculate data quality score for a database variant.

    Higher tuple is better. Scoring heuristic:
      1) integrity ok (True > False)
      2) total rowcount across effective tables
      3) number of non-empty tables
      4) number of effective tables
      5) ddl parse rate
      6) does it open (True > False)

    Args:
        meta: Database metadata to score

    Returns:
        Tuple of scores for comparison (higher is better)
    """
    return (
        1 if meta.integrity_ok else 0,
        int(meta.total_rows or 0),
        len(meta.nonempty_tables or []),
        len(meta.effective_table_names or []),
        meta.ddl_parse_rate or 0.0,
        1 if meta.opens else 0,
    )


def choose_best_variant(
    variants: Sequence[Variant],
    profile_scores: dict[str, float | None] | None = None,
) -> Variant:
    """Select the best variant using base heuristic plus optional profile scores.

    Profile scores (0..1) boost rowcount weight; missing scores are penalized slightly.
    When data quality is equal, prefers variants closer to the original (O > C > R > D).

    Args:
        variants: List of database variants to choose from
        profile_scores: Optional dict mapping variant path to profile similarity score (0..1)

    Returns:
        The best variant according to scoring heuristic
    """
    profile_scores = profile_scores or {}

    def _score(variant: Variant) -> tuple:
        integrity, total_rows, nonempty, effective, ddl_rate, opens_flag = _variant_score(variant.meta)
        prof = profile_scores.get(variant.path.as_posix())
        prof_val = -1.0 if prof is None else max(0.0, min(1.0, float(prof)))

        priority, closeness = resolve_variant_priority(variant.tag)

        return (
            integrity,
            total_rows,
            prof_val,
            nonempty,
            effective,
            ddl_rate,
            opens_flag,
            priority,
            -closeness,
        )

    return max(variants, key=_score)


def attempt_clone(
    src: Path,
    tmpdir: Path,
    dest_dir: Path | None = None,
    sqlite3_path: str | None = None,
    original_name: Path | None = None,
) -> Path | None:
    """Attempt to clone a database using SQLite recovery methods.

    Tries multiple approaches in order of preference:
      1) .clone  -> <dest> (best option)
      2) .backup -> <dest> (good fallback)
      3) VACUUM INTO <dest> (works when source can open RW)
      4) .dump | sqlite3 <dest> (last resort)

    Writes final artifact into dest_dir (else tmpdir).

    Args:
        src: Source database path (may be decompressed temp file)
        tmpdir: Temporary directory for intermediate files
        dest_dir: Final destination directory (defaults to tmpdir)
        sqlite3_path: Path to sqlite3 binary (auto-detected if None)
        original_name: Original filename (before decompression) for .gz/.bz2 files

    Returns:
        Path to cloned database if successful, None otherwise
    """
    final_dir = dest_dir or tmpdir
    final_dir.mkdir(parents=True, exist_ok=True)

    # Use original filename if provided (for .gz/.bz2), otherwise use src filename
    # For files like "f121661224.sqlite.gz", we want base name "f121661224"
    if original_name:
        base_name = original_name.stem  # e.g., "f121661224.sqlite" from "f121661224.sqlite.gz"
        # Strip .sqlite if present to avoid "f121661224.sqlite.clone.sqlite"
        if base_name.endswith(".sqlite"):
            base_name = base_name[:-7]  # Remove ".sqlite"
    else:
        base_name = src.stem

    out_tmp = tmpdir / (base_name + ".clone.sqlite")
    out_final = final_dir / (base_name + ".clone.sqlite")

    sqlite3_bin = sqlite3_path or _which_sqlite3()

    # --- 1) .clone (best option) ---
    if sqlite3_bin:
        try:
            # sqlite3 <src> ".clone <out_tmp>"
            # Use native paths for subprocess on Windows
            # Use encoding with errors="replace" to handle binary data in corrupted DBs
            r = subprocess.run(
                [sqlite3_bin, str(src), f".clone {str(out_tmp)}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if r.returncode == 0 and out_tmp.exists() and out_tmp.stat().st_size > 0:
                shutil.move(str(out_tmp), str(out_final))
                return out_final
        except Exception:
            pass

        # --- 2) .backup (good fallback) ---
        try:
            # sqlite3 <src> ".backup <out_tmp>"
            r = subprocess.run(
                [sqlite3_bin, str(src), f".backup {str(out_tmp)}"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            if r.returncode == 0 and out_tmp.exists() and out_tmp.stat().st_size > 0:
                shutil.move(str(out_tmp), str(out_final))
                return out_final
        except Exception:
            pass

    # --- 3) VACUUM INTO (works when source can open RW) ---
    try:
        rw_copy = tmpdir / (src.stem + ".rwcopy.sqlite")
        shutil.copy2(src, rw_copy)
        # SQLite URI format uses forward slashes on all platforms
        uri = f"file:{rw_copy.as_posix()}?mode=rw"
        con = sqlite3.connect(uri, uri=True, timeout=3.0)
        try:
            con.execute(f"VACUUM INTO '{out_tmp.as_posix()}'")
        finally:
            with contextlib.suppress(Exception):
                con.close()
        if out_tmp.exists() and out_tmp.stat().st_size > 0:
            shutil.move(str(out_tmp), str(out_final))
            return out_final
    except Exception:
        pass

    # --- 4) .dump | sqlite3 (last resort; keeps going past minor errors) ---
    if sqlite3_bin:
        p1 = None
        p2 = None
        try:
            # Use encoding with errors="replace" to handle binary data in corrupted DBs
            p1 = subprocess.Popen(
                [sqlite3_bin, str(src), ".dump"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            p2 = subprocess.Popen(
                [sqlite3_bin, str(out_tmp)],
                stdin=p1.stdout,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            # Allow p1 to receive SIGPIPE if p2 exits
            if p1.stdout:
                p1.stdout.close()

            _, _ = p2.communicate(timeout=180)

            # CRITICAL: Wait for p1 to finish and reap its process to avoid FD leak
            p1.wait(timeout=10)

            if out_tmp.exists() and out_tmp.stat().st_size > 0:
                shutil.move(str(out_tmp), str(out_final))
                return out_final
        except subprocess.TimeoutExpired:
            # Kill both processes on timeout to avoid zombie processes
            if p2:
                p2.kill()
                p2.wait()
            if p1:
                p1.kill()
                p1.wait()
        except Exception:
            pass
        finally:
            # Close all pipe handles before process cleanup to prevent FD leaks
            # Pipes remain open even after process termination until explicitly closed
            for proc in [p1, p2]:
                if proc is not None:
                    for pipe in [proc.stdin, proc.stdout, proc.stderr]:
                        if pipe is not None:
                            with contextlib.suppress(Exception):
                                pipe.close()
                    # Reap zombie processes
                    if proc.poll() is None:
                        try:
                            proc.terminate()
                            proc.wait(timeout=1)
                        except Exception:
                            with contextlib.suppress(Exception):
                                proc.kill()
                                proc.wait()

    return None


def _sanitize_recover_sql(sql: str) -> str:
    """Remove DEFAULT CURRENT_TIMESTAMP and similar dynamic defaults from schema.

    When .recover produces partial rows (only some columns recovered), SQLite fills
    in DEFAULT values for missing columns. For DEFAULT CURRENT_TIMESTAMP, this
    fabricates timestamps with today's date - a serious forensic integrity issue.

    By stripping these defaults from CREATE TABLE statements before import,
    missing timestamp columns become NULL instead of fabricated values.

    Args:
        sql: Raw .recover SQL output

    Returns:
        Sanitized SQL with dynamic time defaults removed
    """
    import re

    # Patterns to remove (order matters - most specific first):
    # We remove both the DEFAULT clause AND any NOT NULL constraint for time columns,
    # because partial rows should be allowed to have NULL timestamps
    patterns = [
        # NOT NULL DEFAULT CURRENT_TIMESTAMP -> remove both (allow NULL)
        (r"\bNOT\s+NULL\s+DEFAULT\s+CURRENT_TIMESTAMP\b", ""),
        # DEFAULT CURRENT_TIMESTAMP alone
        (r"\bDEFAULT\s+CURRENT_TIMESTAMP\b", ""),
        # DEFAULT (datetime(...)) with nested parentheses and arguments
        (r"\bDEFAULT\s+\(datetime\([^)]*\)\)", ""),
        # DEFAULT CURRENT_DATE
        (r"\bDEFAULT\s+CURRENT_DATE\b", ""),
        # DEFAULT CURRENT_TIME
        (r"\bDEFAULT\s+CURRENT_TIME\b", ""),
    ]

    sanitized = sql
    for pattern, replacement in patterns:
        sanitized = re.sub(pattern, replacement, sanitized, flags=re.IGNORECASE)

    return sanitized


def attempt_recover(
    src: Path,
    tmpdir: Path,
    sqlite3_path: str | None = None,
    dest_dir: Path | None = None,
    original_name: Path | None = None,
) -> Path | None:
    """Attempt to recover a database using sqlite3 .recover.

    Always saves the raw .recover SQL (for later salvage of lost_and_found).
    Imports with '.bail off' so a single failing statement doesn't abort import.
    Returns the recovered DB path if any statements succeeded; else None.

    Args:
        src: Source database path (may be decompressed temp file)
        tmpdir: Temporary directory for intermediate files
        sqlite3_path: Path to sqlite3 binary (auto-detected if None)
        dest_dir: Final destination directory (defaults to tmpdir)
        original_name: Original filename (before decompression) for .gz/.bz2 files

    Returns:
        Path to recovered database if successful, None otherwise
    """
    sqlite3_bin = sqlite3_path or _which_sqlite3()
    if not sqlite3_bin:
        return None

    final_dir = dest_dir or tmpdir
    final_dir.mkdir(parents=True, exist_ok=True)

    # Use original filename if provided (for .gz/.bz2), otherwise use src filename
    # For files like "f121661224.sqlite.gz", we want base name "f121661224"
    if original_name:
        base_name = original_name.stem  # e.g., "f121661224.sqlite" from "f121661224.sqlite.gz"
        # Strip .sqlite if present to avoid "f121661224.sqlite.recover.sqlite"
        if base_name.endswith(".sqlite"):
            base_name = base_name[:-7]  # Remove ".sqlite"
    else:
        base_name = src.stem

    raw_sql_path = final_dir / (base_name + ".recover.sql")
    out_db_tmp = tmpdir / (base_name + ".recover.sqlite")
    out_db_final = final_dir / (base_name + ".recover.sqlite")

    try:
        # 1) Produce raw recover SQL and save it verbatim (keep lost_and_found content!)
        # Use native paths for subprocess on Windows
        # Use encoding with errors="replace" to handle binary data in corrupted DBs
        p = subprocess.run(
            [sqlite3_bin, str(src), ".recover"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
        )
        if p.returncode != 0 or not p.stdout:
            return None

        raw_sql_path.write_text(p.stdout, encoding="utf-8", errors="ignore")

        # 2) Sanitize SQL to prevent DEFAULT CURRENT_TIMESTAMP contamination.
        # When .recover produces partial rows, SQLite fills DEFAULT values for
        # missing columns. For CURRENT_TIMESTAMP, this fabricates timestamps
        # with today's date - a serious forensic integrity issue.
        sanitized_sql = _sanitize_recover_sql(p.stdout)

        # 3) Import with '.bail off' so import continues past any single failing stmt
        # (We do NOT strip lost_and_found statements; we just don't let them abort.)
        import_script = (
            ".bail off\n"
            ".echo off\n"
            # speed up large imports a bit
            "PRAGMA journal_mode=OFF;\n"
            "PRAGMA synchronous=OFF;\n"
            "PRAGMA temp_store=MEMORY;\n"
        )

        # Feed the control commands + the sanitized SQL to sqlite3 building the OUT DB
        subprocess.run(
            [sqlite3_bin, str(out_db_tmp)],
            input=import_script + sanitized_sql,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=300,
        )

        # Even if p2.returncode != 0, many statements may have applied;
        # accept the DB if it exists and is non-empty.
        if out_db_tmp.exists() and out_db_tmp.stat().st_size > 0:
            shutil.move(str(out_db_tmp), str(out_db_final))
            # Switch to DELETE journal mode to avoid WAL file locks on Windows
            try:
                with sqlite3.connect(out_db_final) as conn:
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
                    conn.execute("PRAGMA journal_mode=DELETE;")
                # Explicitly delete WAL files (Windows compatibility)
                cleanup_wal_files(out_db_final)
            except Exception:
                pass  # Best effort - don't fail if WAL cleanup fails
            return out_db_final

    except Exception:
        return None

    return None


def introspect_with_repairs(
    src: Path,
    ignore_tables: set[str],
    ignore_prefixes: set[str],
    ignore_suffixes: set[str],
    try_clone: bool = True,
    try_recover: bool = True,
    sqlite3_path: str | None = None,
    save_dir: Path | None = None,
    introspect_func=None,  # Will be injected to avoid circular import
) -> tuple[Variant, list[Variant], dict[str, str]]:
    """Introspect a database and create variants through recovery methods.

    Decompresses .gz/.bz2 files if needed, then:
    1. Attempts to open original database
    2. If needed, attempts clone operation
    3. If needed, attempts recover operation
    4. Selects best variant based on data quality

    Args:
        src: Source database path
        ignore_tables: Table names to ignore during introspection
        ignore_prefixes: Table name prefixes to ignore
        ignore_suffixes: Table name suffixes to ignore
        try_clone: Whether to attempt clone operation if needed
        try_recover: Whether to attempt recover operation if needed
        sqlite3_path: Path to sqlite3 binary (auto-detected if None)
        save_dir: Directory to save variant files (uses temp dir if None)
        introspect_func: Function to introspect database (injected to avoid circular import)

    Returns:
        Tuple of (best_variant, all_variants, saved_paths_map)
        saved_paths_map includes any of:
            {
                "original": "...",
                "clone": "...",
                "recover": "...",
                "recover_sql_raw": "...",
                "dissect": "...",
                "dissect_rebuilt": "...",
                "decompressed": "..."
            }
    """
    if introspect_func is None:
        # Import here to avoid circular dependency
        from .db_variant_selector import (
            introspect_db as introspect_func,  # type: ignore
        )

    variants: list[Variant] = []
    saved: dict[str, str] = {}

    # Decompress .gz/.bz2 files first
    actual_src = src
    decompressed_path: Path | None = None

    suffix = src.suffix.lower()
    if suffix in (".gz", ".bz2"):
        try:
            # Try standard decompression first
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".sqlite", delete=False) as tmp:
                decompressed_path = Path(tmp.name)

                if suffix == ".gz":
                    try:
                        with gzip.open(src, "rb") as f_in:
                            shutil.copyfileobj(f_in, tmp)
                        actual_src = decompressed_path
                    except Exception:
                        # Standard decompression failed, try gzrecover
                        decompressed_path.unlink()
                        if COMPRESSION_UTILS_AVAILABLE:
                            recovered = recover_gzip(src, decompressed_path, timeout=30)
                            if recovered:
                                actual_src = recovered
                                decompressed_path = recovered
                            else:
                                raise RuntimeError("gzrecover failed")
                        else:
                            raise RuntimeError("compression_utils not available")

                elif suffix == ".bz2":
                    try:
                        with bz2.open(src, "rb") as f_in:
                            shutil.copyfileobj(f_in, tmp)
                        actual_src = decompressed_path
                    except Exception:
                        # Standard decompression failed - bzip2recover is not usable
                        # because it creates multiple block files that need manual
                        # testing and concatenation
                        decompressed_path.unlink()
                        raise RuntimeError("bz2 decompression failed")

        except Exception as e:
            # Decompression completely failed, continue with original compressed file (will likely fail)
            logger.warning(f"Failed to decompress {src.name}: {e}")
            actual_src = src

    try:
        # Optional per-case folder and copy of original
        case_dir: Path | None = None
        if save_dir:
            case_dir = make_case_dir(save_dir, src)
            orig_copy = case_dir / src.name
            if safe_copy(src, orig_copy):  # type: ignore
                saved["original"] = orig_copy.as_posix()

            # Also save decompressed version if we created one
            if decompressed_path and decompressed_path.exists():
                decompressed_copy = case_dir / (src.stem + ".sqlite")
                if safe_copy(decompressed_path, decompressed_copy):  # type: ignore
                    saved["decompressed"] = decompressed_copy.as_posix()

        # --- Original (no salvage detection) ---
        # Use decompressed file if available, otherwise original
        Original = introspect_func(
            actual_src,
            ignore_tables,
            ignore_prefixes,
            ignore_suffixes,
            detect_salvage=False,
        )
        variants.append(Variant("O", actual_src, Original))

        need_help = (
            (not Original.tables)
            or ("sqlite_master unreadable" in " ".join(Original.notes))
            or (not Original.integrity_ok)
        )

        cpath: Path | None = None
        rpath: Path | None = None

        with tempfile.TemporaryDirectory(prefix="sqfix_") as td:
            tdp = Path(td)

            # --- Clone (shell-first) ---
            if try_clone and need_help:
                cpath = attempt_clone(
                    actual_src,
                    tdp,
                    dest_dir=case_dir,
                    sqlite3_path=sqlite3_path,
                    original_name=src,  # Pass original filename for .gz/.bz2 files
                )
                if cpath:
                    C = introspect_func(
                        cpath,
                        ignore_tables,
                        ignore_prefixes,
                        ignore_suffixes,
                        detect_salvage=False,  # never detect salvage on clones
                    )
                    C.notes.append(f"origin: cloned from {src.name}")
                    variants.append(Variant("C", cpath, C))
                    if case_dir:
                        saved["clone"] = cpath.as_posix()

            # --- Recover (.recover with raw SQL saved) ---
            if try_recover and need_help:
                rpath = attempt_recover(
                    actual_src,
                    tdp,
                    sqlite3_path=sqlite3_path,
                    dest_dir=case_dir,
                    original_name=src,  # Pass original filename for .gz/.bz2 files
                )
                if rpath:
                    R = introspect_func(
                        rpath,
                        ignore_tables,
                        ignore_prefixes,
                        ignore_suffixes,
                        detect_salvage=True,  # ONLY detect salvage here
                    )
                    R.notes.append(f"origin: recovered from {src.name}")
                    variants.append(Variant("R", rpath, R))
                    if case_dir:
                        saved["recover"] = rpath.as_posix()
                        raw_sql = case_dir / (src.stem + ".recover.sql")
                        if raw_sql.exists():
                            saved["recover_sql_raw"] = raw_sql.as_posix()

        # Pick the best variant by heuristic with explicit tag priorities.
        # When variants have equal data quality scores, prefer original over reconstructed:
        # Prefer original ("O") > clone ("C") > recovered ("R") > dissect_rebuilt ("D").
        # This prioritizes files closest to the original for maximum data fidelity.
        best = choose_best_variant(variants)
        return best, variants, saved

    finally:
        # Clean up temporary decompressed file
        if decompressed_path and decompressed_path.exists():
            with contextlib.suppress(Exception):
                decompressed_path.unlink()
