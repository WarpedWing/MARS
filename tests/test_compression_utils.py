"""Tests for compression utility functions."""

import gzip
from pathlib import Path

import pytest

from mars.utils.compression_utils import (
    ARCHIVE_EXTENSIONS,
    get_archive_extension,
    get_compression_type,
    is_archive,
)


class TestGetArchiveExtension:
    """Test archive extension detection."""

    def test_tar_gz_extension(self):
        """Test detection of .tar.gz compound extension."""
        assert get_archive_extension(Path("file.tar.gz")) == ".tar.gz"
        assert get_archive_extension(Path("FILE.TAR.GZ")) == ".tar.gz"

    def test_tar_bz2_extension(self):
        """Test detection of .tar.bz2 compound extension."""
        assert get_archive_extension(Path("file.tar.bz2")) == ".tar.bz2"
        assert get_archive_extension(Path("FILE.TAR.BZ2")) == ".tar.bz2"

    def test_simple_zip_extension(self):
        """Test detection of simple .zip extension."""
        assert get_archive_extension(Path("file.zip")) == ".zip"
        assert get_archive_extension(Path("FILE.ZIP")) == ".zip"

    def test_simple_gz_extension(self):
        """Test detection of simple .gz extension."""
        assert get_archive_extension(Path("file.gz")) == ".gz"

    def test_simple_bz2_extension(self):
        """Test detection of simple .bz2 extension."""
        assert get_archive_extension(Path("file.bz2")) == ".bz2"

    def test_tgz_extension(self):
        """Test detection of .tgz shorthand extension."""
        assert get_archive_extension(Path("file.tgz")) == ".tgz"

    def test_non_archive_extension(self):
        """Test handling of non-archive extensions."""
        assert get_archive_extension(Path("file.txt")) == ".txt"
        assert get_archive_extension(Path("file.db")) == ".db"

    def test_no_extension(self):
        """Test handling of files with no extension."""
        assert get_archive_extension(Path("filename")) == ""


class TestIsArchive:
    """Test archive file detection."""

    def test_zip_is_archive(self):
        """Test that .zip files are detected as archives."""
        assert is_archive(Path("file.zip")) is True

    def test_tar_gz_is_archive(self):
        """Test that .tar.gz files are detected as archives."""
        assert is_archive(Path("file.tar.gz")) is True

    def test_tar_bz2_is_archive(self):
        """Test that .tar.bz2 files are detected as archives."""
        assert is_archive(Path("file.tar.bz2")) is True

    def test_tgz_is_archive(self):
        """Test that .tgz files are detected as archives."""
        assert is_archive(Path("file.tgz")) is True

    def test_tbz2_is_archive(self):
        """Test that .tbz2 files are detected as archives."""
        assert is_archive(Path("file.tbz2")) is True

    def test_gz_is_archive(self):
        """Test that .gz files are detected as archives."""
        assert is_archive(Path("file.gz")) is True

    def test_bz2_is_archive(self):
        """Test that .bz2 files are detected as archives."""
        assert is_archive(Path("file.bz2")) is True

    def test_txt_is_not_archive(self):
        """Test that .txt files are not detected as archives."""
        assert is_archive(Path("file.txt")) is False

    def test_db_is_not_archive(self):
        """Test that .db files are not detected as archives."""
        assert is_archive(Path("file.db")) is False

    def test_no_extension_is_not_archive(self):
        """Test that files without extensions are not detected as archives."""
        assert is_archive(Path("filename")) is False


class TestGetCompressionType:
    """Test compression type detection."""

    def test_gz_extension(self):
        """Test gzip detection from .gz extension."""
        assert get_compression_type(Path("file.gz")) == "gzip"

    def test_tar_gz_extension(self):
        """Test gzip detection from .tar.gz extension."""
        assert get_compression_type(Path("file.tar.gz")) == "gzip"

    def test_tgz_extension(self):
        """Test gzip detection from .tgz extension."""
        assert get_compression_type(Path("file.tgz")) == "gzip"

    def test_bz2_extension(self):
        """Test bzip2 detection from .bz2 extension."""
        assert get_compression_type(Path("file.bz2")) == "bzip2"

    def test_tar_bz2_extension(self):
        """Test bzip2 detection from .tar.bz2 extension."""
        assert get_compression_type(Path("file.tar.bz2")) == "bzip2"

    def test_tbz2_extension(self):
        """Test bzip2 detection from .tbz2 extension."""
        assert get_compression_type(Path("file.tbz2")) == "bzip2"

    def test_zip_extension(self):
        """Test zip detection from .zip extension."""
        assert get_compression_type(Path("file.zip")) == "zip"

    def test_non_compressed_extension(self):
        """Test that non-compressed files return None."""
        assert get_compression_type(Path("file.txt")) is None
        assert get_compression_type(Path("file.db")) is None

    def test_gzip_magic_bytes(self, tmp_path):
        """Test gzip detection from magic bytes."""
        gzip_file = tmp_path / "compressed.bin"
        # Write gzip-compressed content
        with gzip.open(gzip_file, "wb") as f:
            f.write(b"Hello World")  # type: ignore

        assert get_compression_type(gzip_file) == "gzip"

    def test_unknown_magic_bytes(self, tmp_path):
        """Test handling of unknown magic bytes."""
        unknown_file = tmp_path / "unknown.bin"
        unknown_file.write_bytes(b"UNKN" + b"\x00" * 100)

        assert get_compression_type(unknown_file) is None


class TestArchiveExtensionsConstant:
    """Test ARCHIVE_EXTENSIONS constant."""

    def test_archive_extensions_is_set(self):
        """Test that ARCHIVE_EXTENSIONS is a set."""
        assert isinstance(ARCHIVE_EXTENSIONS, set)

    def test_archive_extensions_contains_common_formats(self):
        """Test that common archive formats are included."""
        assert ".zip" in ARCHIVE_EXTENSIONS
        assert ".tar" in ARCHIVE_EXTENSIONS
        assert ".tar.gz" in ARCHIVE_EXTENSIONS
        assert ".gz" in ARCHIVE_EXTENSIONS
        assert ".bz2" in ARCHIVE_EXTENSIONS

    def test_archive_extensions_lowercase(self):
        """Test that all extensions are lowercase."""
        for ext in ARCHIVE_EXTENSIONS:
            assert ext == ext.lower()

    def test_archive_extensions_start_with_dot(self):
        """Test that all extensions start with a dot."""
        for ext in ARCHIVE_EXTENSIONS:
            assert ext.startswith(".")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
