"""Tests for rubric utility functions."""

import pytest

from mars.pipeline.matcher.rubric_utils import (
    date_to_epoch,
    detect_pattern_type,
    detect_programming_case_role,
    epoch_to_year,
    is_filesystem_path,
    is_multi_word_identifier,
    matches_programming_case,
)


class TestIsFilesystemPath:
    """Test filesystem path detection."""

    def test_absolute_unix_path(self):
        """Test detection of absolute Unix paths."""
        assert is_filesystem_path("/Users/admin/file.txt") is True
        assert is_filesystem_path("/etc/passwd") is True
        assert is_filesystem_path("/tmp") is True

    def test_home_relative_path(self):
        """Test detection of home-relative paths."""
        assert is_filesystem_path("~/Documents") is True
        assert is_filesystem_path("~/Library/Caches/file.db") is True

    def test_base64_starting_with_slash(self):
        """Test rejection of base64 strings starting with slash."""
        # Base64 often starts with / and contains + and = which are invalid in paths
        # The function allows short segments with + (< 20 chars with + is OK)
        # Only long segments with + are rejected
        assert is_filesystem_path("/abc+def+ghi") is True  # Short + is OK
        # Long base64-like string with + should be rejected
        long_base64 = "/KS47+1OObRwg" + "A" * 30 + "+more"
        assert is_filesystem_path(long_base64) is False

    def test_base64_with_equals(self):
        """Test rejection of paths with equals (base64 padding)."""
        assert is_filesystem_path("/some/path/data==") is False

    def test_empty_string(self):
        """Test handling of empty string."""
        assert is_filesystem_path("") is False

    def test_relative_path(self):
        """Test rejection of relative paths."""
        assert is_filesystem_path("relative/path") is False
        assert is_filesystem_path("./local/path") is False

    def test_long_base64_segment(self):
        """Test rejection of long base64-like segments."""
        long_base64 = "/" + "A" * 50 + "B" * 50  # 100-char segment
        assert is_filesystem_path(long_base64) is False


class TestMatchesProgrammingCase:
    """Test programming case pattern matching."""

    def test_snake_case(self):
        """Test detection of snake_case."""
        assert matches_programming_case("snake_case_var") is True
        assert matches_programming_case("my_variable") is True
        assert matches_programming_case("foo_bar_baz") is True

    def test_camel_case(self):
        """Test detection of camelCase."""
        assert matches_programming_case("camelCaseVar") is True
        assert matches_programming_case("myVariable") is True

    def test_pascal_case(self):
        """Test detection of PascalCase."""
        assert matches_programming_case("PascalCase") is True
        assert matches_programming_case("MyClassName") is True

    def test_kebab_case(self):
        """Test detection of kebab-case."""
        assert matches_programming_case("kebab-case-name") is True
        assert matches_programming_case("my-component") is True

    def test_screaming_snake_case(self):
        """Test detection of SCREAMING_SNAKE_CASE."""
        assert matches_programming_case("SCREAMING_SNAKE") is True
        assert matches_programming_case("MY_CONSTANT") is True

    def test_single_word_no_match(self):
        """Test that single words don't match."""
        assert matches_programming_case("single") is False
        assert matches_programming_case("word") is False

    def test_all_caps_single_word(self):
        """Test that all-caps single words don't match."""
        assert matches_programming_case("UUID") is False
        assert matches_programming_case("HTTP") is False


class TestIsMultiWordIdentifier:
    """Test multi-word identifier detection."""

    def test_snake_case_is_multi_word(self):
        """Test that snake_case is detected as multi-word."""
        assert is_multi_word_identifier("my_var") is True
        assert is_multi_word_identifier("foo_bar_baz") is True

    def test_kebab_case_is_multi_word(self):
        """Test that kebab-case is detected as multi-word."""
        assert is_multi_word_identifier("my-component") is True

    def test_camel_case_is_multi_word(self):
        """Test that camelCase is detected as multi-word."""
        assert is_multi_word_identifier("myVariable") is True
        assert is_multi_word_identifier("firstName") is True

    def test_all_caps_not_multi_word(self):
        """Test that all-caps single words are not multi-word."""
        assert is_multi_word_identifier("UUID") is False
        assert is_multi_word_identifier("HTTP") is False
        assert is_multi_word_identifier("API") is False

    def test_single_lowercase_not_multi_word(self):
        """Test that single lowercase words are not multi-word."""
        assert is_multi_word_identifier("word") is False
        assert is_multi_word_identifier("a") is False


class TestDetectPatternType:
    """Test semantic pattern type detection."""

    def test_url_detection(self):
        """Test detection of URLs."""
        assert detect_pattern_type("https://example.com") == "url"
        assert detect_pattern_type("http://localhost:8080/api") == "url"
        assert detect_pattern_type("ftp://files.example.com") == "url"

    def test_uuid_detection(self):
        """Test detection of UUIDs."""
        assert detect_pattern_type("550e8400-e29b-41d4-a716-446655440000") == "uuid"
        assert detect_pattern_type("A550E840-E29B-41D4-A716-446655440000") == "uuid"

    def test_email_detection(self):
        """Test detection of email addresses."""
        assert detect_pattern_type("user@example.com") == "email"
        assert detect_pattern_type("test.user@subdomain.example.org") == "email"

    def test_path_detection(self):
        """Test detection of filesystem paths."""
        assert detect_pattern_type("/Users/admin/file.txt") == "path"
        assert detect_pattern_type("~/Documents/report.pdf") == "path"

    def test_domain_detection(self):
        """Test detection of domain names."""
        assert detect_pattern_type("example.com") == "domain"
        assert detect_pattern_type("subdomain.example.org") == "domain"

    def test_timestamp_text_detection(self):
        """Test detection of human-readable timestamps."""
        assert detect_pattern_type("2020-07-26T20:40:53Z") == "timestamp_text"
        assert detect_pattern_type("2020/07/26-20:40:53.493") == "timestamp_text"

    def test_random_text_no_pattern(self):
        """Test that random text returns None."""
        assert detect_pattern_type("random text here") is None
        assert detect_pattern_type("12345") is None

    def test_empty_string(self):
        """Test handling of empty string."""
        assert detect_pattern_type("") is None

    def test_non_string_input(self):
        """Test handling of non-string input."""
        assert detect_pattern_type(None) is None  # type: ignore
        assert detect_pattern_type(123) is None  # type: ignore


class TestDetectProgrammingCaseRole:
    """Test programming_case role detection."""

    def test_snake_case_column(self):
        """Test detection of snake_case column."""
        values = ["my_var", "foo_bar", "test_value", "another_one"]
        assert detect_programming_case_role(values) == "programming_case"

    def test_camel_case_column(self):
        """Test detection of camelCase column."""
        values = ["myVar", "fooBar", "testValue", "anotherOne"]
        assert detect_programming_case_role(values) == "programming_case"

    def test_mixed_with_single_words(self):
        """Test column with mix of multi-word and single words."""
        # Single words don't count toward threshold
        values = ["snake_case", "foo", "camelCase", "bar", "PascalCase"]
        result = detect_programming_case_role(values)
        # 3/5 = 60% which meets threshold
        assert result == "programming_case"

    def test_natural_language_disqualified(self):
        """Test that spaces disqualify the column."""
        values = ["my_var", "foo bar", "test_value"]  # "foo bar" has space
        assert detect_programming_case_role(values) is None

    def test_empty_values(self):
        """Test handling of empty/null values."""
        values = [None, "", None, ""]
        assert detect_programming_case_role(values) is None


class TestDateToEpoch:
    """Test date string to epoch conversion."""

    def test_valid_date(self):
        """Test conversion of valid date."""
        epoch = date_to_epoch("2020-01-01")
        assert isinstance(epoch, float)
        assert epoch == 1577836800.0

    def test_another_date(self):
        """Test conversion of another date."""
        epoch = date_to_epoch("2000-01-01")
        assert epoch == 946684800.0

    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        with pytest.raises(ValueError):
            date_to_epoch("01-01-2020")  # Wrong format

    def test_invalid_date(self):
        """Test that invalid date raises ValueError."""
        with pytest.raises(ValueError):
            date_to_epoch("2020-13-01")  # Month 13 doesn't exist


class TestEpochToYear:
    """Test epoch timestamp to year conversion."""

    def test_year_2020(self):
        """Test conversion to year 2020."""
        year = epoch_to_year(1577836800.0)  # 2020-01-01
        assert year == 2020

    def test_year_2000(self):
        """Test conversion to year 2000."""
        year = epoch_to_year(946684800.0)  # 2000-01-01
        assert year == 2000

    def test_year_1970(self):
        """Test conversion to year 1970 (Unix epoch)."""
        year = epoch_to_year(0.0)
        assert year == 1970


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
