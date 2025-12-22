"""Compression utilities for gzip and bzip2 files.

Provides centralized access to gzrecover for recovering data from corrupted
gzip archives. Bzip2 files use Python's native bz2 module for decompression.

Note: bzip2recover is NOT supported because it creates multiple block files
that need individual testing and concatenation, making it unsuitable for
automated recovery pipelines.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from mars.utils.debug_logger import logger


def find_recovery_tool(tool_name: str) -> Path | None:
    """
    Find recovery tool binary (e.g., gzrecover).

    Searches in order:
    1. Bundled binary in resources/<os>/bin/
    2. System PATH

    Args:
        tool_name: Tool name (e.g., 'gzrecover')

    Returns:
        Path to tool if found, None otherwise
    """
    # Determine OS-specific subdirectory
    system = platform.system().lower()
    if system == "darwin":
        os_dir = "macos"
    elif system == "linux":
        os_dir = "linux"
    elif system == "windows":
        os_dir = "windows"
        if not tool_name.endswith(".exe"):
            tool_name = f"{tool_name}.exe"
    else:
        os_dir = "macos"  # fallback

    # Check bundled binary first
    # Navigate from src/mars/utils/ to src/resources/<os>/bin/
    bundled_path = Path(__file__).parent.parent.parent / "resources" / os_dir / "bin" / tool_name
    if bundled_path.exists() and bundled_path.is_file():
        return bundled_path

    # Check system PATH
    path_binary = shutil.which(tool_name)
    if path_binary:
        return Path(path_binary)

    return None


def recover_gzip(
    file_path: Path,
    output_path: Path | None = None,
    timeout: int = 30,
) -> Path | None:
    """
    Attempt to recover data from corrupted gzip file using gzrecover.

    Args:
        file_path: Path to corrupted gzip file
        output_path: Optional output path (if None, creates temp file)
        timeout: Timeout in seconds (default: 30)

    Returns:
        Path to recovered file, or None if recovery failed
    """
    gzrecover = find_recovery_tool("gzrecover")
    if not gzrecover:
        logger.debug("gzrecover not found in resources or PATH, cannot recover")
        return None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Copy input file to temp dir - gzrecover writes output next to input file,
            # so we need the input in temp dir to keep output there too
            temp_input = tmpdir_path / file_path.name
            shutil.copy(file_path, temp_input)

            # Run gzrecover on the temp copy
            subprocess.run(
                [str(gzrecover), str(temp_input)],
                cwd=tmpdir,
                capture_output=True,
                timeout=timeout,
                check=False,
            )

            # Find recovered files (gzrecover creates <filename>.recovered next to input)
            # Filter out the input file we copied
            recovered_files = [f for f in tmpdir_path.glob("*") if f != temp_input]
            if not recovered_files:
                logger.debug(f"gzrecover produced no output for {file_path.name}")
                return None

            # Use provided output path or create one
            if output_path is None:
                # Create temp file that caller is responsible for cleaning up
                fd, temp_name = tempfile.mkstemp(suffix=".recovered")
                import os

                os.close(fd)
                output_path = Path(temp_name)

            # Copy recovered data to output
            shutil.copy(recovered_files[0], output_path)

            logger.debug(f"Recovered {file_path.name} with gzrecover")

            return output_path

    except subprocess.TimeoutExpired:
        logger.debug(f"gzrecover timeout on {file_path.name}")
    except Exception as e:
        logger.debug(f"gzrecover failed on {file_path.name}: {e}")

    return None


def read_compressed_with_recovery(
    file_path: Path,
    size: int = 4096,
    try_recovery: bool = True,
    compression_type: str | None = None,
) -> bytes:
    """
    Read compressed file with automatic recovery fallback for corrupted files.

    Attempts standard decompression first, then tries recovery tools if it fails.
    Supports gzip and bzip2 formats.

    Args:
        file_path: Path to compressed file
        size: Number of bytes to read (default: 4096)
        try_recovery: Whether to attempt recovery if standard decompression fails
        compression_type: Optional hint ("gzip" or "bzip2"). If provided, tries this
                         type first regardless of magic bytes. This is useful for files
                         with .gz/.bz2 extensions but corrupted magic bytes.

    Returns:
        Decompressed bytes, or empty bytes if all attempts fail
    """
    import gzip

    # If caller specified compression type, try that without magic byte validation
    if compression_type:
        if compression_type == "gzip":
            try:
                with gzip.open(file_path, "rb") as gz:
                    return gz.read(size)
            except Exception:
                # Try gzrecover for corrupted files
                if try_recovery:
                    recovered_path = recover_gzip(file_path)
                    if recovered_path:
                        try:
                            # Read recovered data
                            with recovered_path.open("rb") as f:
                                data = f.read(size)
                            # Clean up temp file
                            recovered_path.unlink(missing_ok=True)
                            return data
                        except Exception:
                            recovered_path.unlink(missing_ok=True)
                            pass
            return b""

        if compression_type == "bzip2":
            import bz2

            try:
                with bz2.open(file_path, "rb") as bz:
                    return bz.read(size)
            except Exception:
                # bzip2recover doesn't work for our use case - it creates multiple
                # block files that need manual testing and concatenation
                pass
            return b""

    # No type hint provided - use magic bytes to detect format
    try:
        # Read magic bytes to detect format
        with file_path.open("rb") as f:
            header = f.read(2)

        if header[:2] == b"\x1f\x8b":  # gzip magic
            try:
                with gzip.open(file_path, "rb") as gz:
                    return gz.read(size)
            except Exception:
                # Try gzrecover for corrupted files
                if try_recovery:
                    recovered_path = recover_gzip(file_path)
                    if recovered_path:
                        try:
                            # Read recovered data
                            with recovered_path.open("rb") as f:
                                data = f.read(size)
                            # Clean up temp file
                            recovered_path.unlink(missing_ok=True)
                            return data
                        except Exception:
                            recovered_path.unlink(missing_ok=True)
                            pass

        elif header[:2] == b"BZ":  # bzip2 magic
            import bz2

            try:
                with bz2.open(file_path, "rb") as bz:
                    return bz.read(size)
            except Exception:
                # bzip2recover doesn't work for our use case - it creates multiple
                # block files that need manual testing and concatenation
                pass

    except Exception:
        pass

    return b""


def decompress_file_with_recovery(
    file_path: Path,
    output_path: Path,
    compression_type: str = "gzip",
    try_recovery: bool = True,
) -> bool:
    """
    Decompress entire file with automatic recovery fallback for corrupted files.

    Attempts standard full decompression first, then tries recovery tools if it fails.
    Supports gzip and bzip2 formats.

    Args:
        file_path: Path to compressed file
        output_path: Path where decompressed file should be written
        compression_type: "gzip" or "bzip2"
        try_recovery: Whether to attempt recovery if standard decompression fails

    Returns:
        True if decompression succeeded, False otherwise
    """
    import shutil

    if compression_type == "gzip":
        import gzip

        # Try direct gzip decompression first
        try:
            with gzip.open(file_path, "rb") as gz_in, output_path.open("wb") as out:
                shutil.copyfileobj(gz_in, out)
            return True
        except Exception as e:
            logger.debug(
                f"Direct gzip decompression failed for {file_path.name}: {e}",
            )

            # Try gzrecover for corrupted files
            if try_recovery:
                recovered_path = recover_gzip(file_path)
                if recovered_path:
                    try:
                        # Recovered file is already decompressed - just copy it
                        shutil.copy(recovered_path, output_path)
                        # Clean up temp recovered file
                        recovered_path.unlink(missing_ok=True)
                        logger.debug(
                            f"Successfully recovered {file_path.name} with gzrecover",
                        )
                        return True
                    except Exception as e2:
                        logger.debug(
                            f"Failed to copy recovered file {file_path.name}: {e2}",
                        )
                        recovered_path.unlink(missing_ok=True)

    elif compression_type == "bzip2":
        import bz2

        # Try direct bzip2 decompression first
        try:
            with bz2.open(file_path, "rb") as bz_in, output_path.open("wb") as out:
                shutil.copyfileobj(bz_in, out)
            return True
        except Exception as e:
            logger.debug(
                f"Direct bz2 decompression failed for {file_path.name}: {e}",
            )
            # bzip2recover doesn't work for our use case - it creates multiple
            # block files that need manual testing and concatenation

    return False


# ============================================================================
# Archive Extension Detection
# ============================================================================

# Supported archive file extensions
ARCHIVE_EXTENSIONS = {
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
    ".tbz2",
    ".gz",
    ".bz2",
}

# Compression magic bytes for format detection
COMPRESSION_MAGIC_BYTES = {
    "gzip": b"\x1f\x8b",
    "bzip2": b"BZ",
    "zip": b"\x50\x4b\x03\x04",
}


def get_archive_extension(path: Path) -> str:
    """
    Get the full archive extension, handling compound extensions like .tar.gz.

    Args:
        path: Path to check

    Returns:
        Full extension string (e.g., ".tar.gz", ".zip", ".bz2")

    Example:
        >>> get_archive_extension(Path("file.tar.gz"))
        '.tar.gz'
        >>> get_archive_extension(Path("file.zip"))
        '.zip'
    """
    path_lower = path.name.lower()

    # Check for compound extensions first
    if path_lower.endswith(".tar.gz"):
        return ".tar.gz"
    if path_lower.endswith(".tar.bz2"):
        return ".tar.bz2"

    # Fall back to simple extension
    return path.suffix.lower()


def is_archive(path: Path) -> bool:
    """
    Check if a file is a supported archive format.

    Args:
        path: Path to check

    Returns:
        True if file has a recognized archive extension

    Example:
        >>> is_archive(Path("data.tar.gz"))
        True
        >>> is_archive(Path("data.txt"))
        False
    """
    extension = get_archive_extension(path)
    return extension in ARCHIVE_EXTENSIONS


def get_compression_type(path: Path) -> str | None:
    """
    Detect compression type from file extension or magic bytes.

    Args:
        path: Path to file

    Returns:
        Compression type ("gzip", "bzip2", "zip") or None if not compressed

    Example:
        >>> get_compression_type(Path("file.gz"))
        'gzip'
        >>> get_compression_type(Path("file.tar.bz2"))
        'bzip2'
        >>> get_compression_type(Path("file.txt"))
        None
    """
    extension = get_archive_extension(path)

    # Check extension first
    if extension in {".gz", ".tar.gz", ".tgz"}:
        return "gzip"
    if extension in {".bz2", ".tar.bz2", ".tbz2"}:
        return "bzip2"
    if extension == ".zip":
        return "zip"

    # Fall back to magic byte detection if file exists
    if path.exists() and path.is_file():
        try:
            with path.open("rb") as f:
                header = f.read(4)

            for comp_type, magic in COMPRESSION_MAGIC_BYTES.items():
                if header.startswith(magic):
                    return comp_type
        except Exception:
            pass

    return None
