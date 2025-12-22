"""Tests for glob utility functions."""

import pytest

from mars.pipeline.mount_utils.dfvfs_glob_utils import (
    _compile_globstar_regex,
    _normalize,
    _segment_has_wildcard,
)


class TestNormalize:
    """Test path normalization function."""

    def test_add_leading_slash(self):
        """Test that leading slash is added if missing."""
        assert _normalize("Library/Caches") == "/Library/Caches"
        assert _normalize("Users/admin") == "/Users/admin"

    def test_preserve_existing_slash(self):
        """Test that existing leading slash is preserved."""
        assert _normalize("/Library/Caches") == "/Library/Caches"

    def test_strip_trailing_slash(self):
        """Test that trailing slash is stripped."""
        assert _normalize("/Library/Caches/") == "/Library/Caches"
        assert _normalize("Library/Caches/") == "/Library/Caches"

    def test_convert_backslash(self):
        """Test that backslashes are converted to forward slashes."""
        assert _normalize("\\Users\\Admin") == "/Users/Admin"
        assert _normalize("Library\\Caches") == "/Library/Caches"

    def test_root_path(self):
        """Test handling of root path."""
        assert _normalize("/") == "/"
        assert _normalize("") == "/"

    def test_complex_path(self):
        """Test complex path with mixed separators."""
        assert _normalize("Users\\admin/Documents/") == "/Users/admin/Documents"


class TestSegmentHasWildcard:
    """Test wildcard detection in path segments."""

    def test_asterisk_wildcard(self):
        """Test detection of asterisk wildcard."""
        assert _segment_has_wildcard("*.db") is True
        assert _segment_has_wildcard("file*") is True
        assert _segment_has_wildcard("*") is True

    def test_question_mark_wildcard(self):
        """Test detection of question mark wildcard."""
        assert _segment_has_wildcard("file?.txt") is True
        assert _segment_has_wildcard("?") is True

    def test_bracket_wildcard(self):
        """Test detection of bracket wildcard."""
        assert _segment_has_wildcard("[abc]file") is True
        assert _segment_has_wildcard("file[0-9]") is True

    def test_globstar(self):
        """Test detection of globstar (**)."""
        assert _segment_has_wildcard("**") is True

    def test_no_wildcard(self):
        """Test segments without wildcards."""
        assert _segment_has_wildcard("normal") is False
        assert _segment_has_wildcard("file.txt") is False
        assert _segment_has_wildcard("Library") is False

    def test_empty_segment(self):
        """Test empty segment."""
        assert _segment_has_wildcard("") is False


class TestCompileGlobstarRegex:
    """Test globstar regex compilation."""

    def test_no_globstar_returns_none(self):
        """Test that patterns without ** return None."""
        assert _compile_globstar_regex("/Users/*/Documents") is None
        assert _compile_globstar_regex("/simple/path") is None

    def test_globstar_mid_pattern(self):
        """Test ** in middle of pattern."""
        regex = _compile_globstar_regex("/Users/**/Library")
        assert regex is not None
        # Should match direct path
        assert regex.match("/Users/admin/Library")
        # Should match deep path
        assert regex.match("/Users/admin/deep/path/Library")
        # Should match zero segments (direct child)
        assert regex.match("/Users/Library")

    def test_globstar_end_pattern(self):
        """Test /** at end of pattern."""
        regex = _compile_globstar_regex("/Users/admin/**")
        assert regex is not None
        # Should match the exact path
        assert regex.match("/Users/admin")
        # Should match subdirectories
        assert regex.match("/Users/admin/Documents")
        assert regex.match("/Users/admin/deep/nested/path")

    def test_globstar_start_pattern(self):
        """Test **/ at start of pattern."""
        regex = _compile_globstar_regex("**/Library")
        assert regex is not None
        # Should match from root
        assert regex.match("/Library")
        # Should match with prefix
        assert regex.match("/Users/admin/Library")

    def test_special_characters_escaped(self):
        """Test that regex special characters are escaped."""
        regex = _compile_globstar_regex("/path/**/file.txt")
        assert regex is not None
        # Should match literal dot
        assert regex.match("/path/sub/file.txt")
        # Should not match different character
        assert not regex.match("/path/sub/filextxt")

    def test_single_asterisk_in_globstar_pattern(self):
        """Test that single * still works in patterns with **."""
        regex = _compile_globstar_regex("/Users/**/*.db")
        assert regex is not None
        # Should match .db files at various depths
        assert regex.match("/Users/data.db")
        assert regex.match("/Users/admin/data.db")
        assert regex.match("/Users/admin/Library/Caches/data.db")

    def test_complex_pattern(self):
        """Test complex pattern with multiple wildcards."""
        regex = _compile_globstar_regex("/Users/**/Library/Caches/*.sqlite")
        assert regex is not None
        assert regex.match("/Users/admin/Library/Caches/cache.sqlite")
        assert regex.match("/Users/admin/deep/path/Library/Caches/data.sqlite")

    def test_returns_compiled_pattern(self):
        """Test that result is a compiled regex pattern."""
        import re

        regex = _compile_globstar_regex("/path/**/file")
        assert regex is not None
        assert isinstance(regex, re.Pattern)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
