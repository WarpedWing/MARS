"""Tests for platform utility functions."""

import pytest

from mars.utils.platform_utils import (
    is_linux,
    is_macos,
    is_windows,
    sanitize_windows_filename,
    sanitize_windows_path,
)


class TestPlatformDetection:
    """Test platform detection functions."""

    def test_platform_detection_mutually_exclusive(self):
        """Test that platform detection is mutually exclusive."""
        # At least one should be true, and at most one should be true
        platforms = [is_windows(), is_macos(), is_linux()]
        true_count = sum(platforms)
        # Should have exactly 0 or 1 (could be an unknown platform)
        assert true_count <= 1

    def test_is_windows_returns_bool(self):
        """Test that is_windows returns a boolean."""
        assert isinstance(is_windows(), bool)

    def test_is_macos_returns_bool(self):
        """Test that is_macos returns a boolean."""
        assert isinstance(is_macos(), bool)

    def test_is_linux_returns_bool(self):
        """Test that is_linux returns a boolean."""
        assert isinstance(is_linux(), bool)


class TestSanitizeWindowsFilename:
    """Test Windows filename sanitization."""

    def test_normal_filename_unchanged(self):
        """Test that normal filenames are not modified."""
        result = sanitize_windows_filename("normal.txt")
        # On non-Windows, input is returned unchanged
        assert result == "normal.txt"

    def test_returns_string(self):
        """Test that function always returns a string."""
        assert isinstance(sanitize_windows_filename("test"), str)

    def test_empty_string(self):
        """Test handling of empty string."""
        assert sanitize_windows_filename("") == ""

    @pytest.mark.skipif(not is_windows(), reason="Windows-specific test")
    def test_reserved_chars_replaced_on_windows(self):
        """Test that reserved characters are replaced on Windows."""
        assert sanitize_windows_filename("file:name") == "file_name"
        assert sanitize_windows_filename("file<>name") == "file__name"
        assert sanitize_windows_filename('file"name') == "file_name"
        assert sanitize_windows_filename("file|name") == "file_name"
        assert sanitize_windows_filename("file?name") == "file_name"
        assert sanitize_windows_filename("file*name") == "file_name"

    @pytest.mark.skipif(not is_windows(), reason="Windows-specific test")
    def test_control_chars_removed_on_windows(self):
        """Test that control characters are removed on Windows."""
        # Control character (ASCII 0x01)
        assert sanitize_windows_filename("file\x01name") == "file_name"


class TestSanitizeWindowsPath:
    """Test Windows path sanitization."""

    def test_normal_path_unchanged(self):
        """Test that normal paths are not modified."""
        result = sanitize_windows_path("Users/admin/Documents")
        # On non-Windows, input is returned unchanged
        assert result == "Users/admin/Documents"

    def test_returns_string(self):
        """Test that function always returns a string."""
        assert isinstance(sanitize_windows_path("test/path"), str)

    def test_empty_string(self):
        """Test handling of empty string."""
        assert sanitize_windows_path("") == ""

    @pytest.mark.skipif(not is_windows(), reason="Windows-specific test")
    def test_path_components_sanitized_on_windows(self):
        """Test that each path component is sanitized on Windows."""
        result = sanitize_windows_path("Users/name:test/file")
        assert ":" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
