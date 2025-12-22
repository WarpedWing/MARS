"""Tests for timestamp utility functions."""

import pytest

from mars.carver.protobuf.timestamp_extractor import (
    analyze_protobuf_for_timestamps,
    extract_timestamps_from_protobuf,
    looks_like_timestamp,
)
from mars.pipeline.matcher.rubric_utils import TimestampFormat


class TestTimestampFormatValidation:
    """Test TimestampFormat.validate_timestamp method."""

    def test_valid_unix_seconds(self):
        """Test validation of Unix seconds timestamp."""
        # 2021-01-01 00:00:00 UTC
        assert TimestampFormat.validate_timestamp(1609459200, "unix_seconds") is True

    def test_valid_unix_milliseconds(self):
        """Test validation of Unix milliseconds timestamp."""
        # 2021-01-01 00:00:00 UTC in milliseconds
        assert TimestampFormat.validate_timestamp(1609459200000, "unix_milliseconds") is True

    def test_valid_unix_microseconds(self):
        """Test validation of Unix microseconds timestamp."""
        # 2021-01-01 00:00:00 UTC in microseconds
        assert TimestampFormat.validate_timestamp(1609459200000000, "unix_microseconds") is True

    def test_valid_cocoa_seconds(self):
        """Test validation of Cocoa seconds timestamp."""
        # Cocoa epoch is 2001-01-01, so 2021 would be about 20 years of seconds
        # 2021-01-01 = about 631152000 Cocoa seconds
        assert TimestampFormat.validate_timestamp(631152000, "cocoa_seconds") is True

    def test_out_of_range_timestamp(self):
        """Test rejection of out-of-range timestamps."""
        # Very old timestamp (year 1980)
        assert TimestampFormat.validate_timestamp(315532800, "unix_seconds") is False

    def test_invalid_input(self):
        """Test rejection of invalid input."""
        assert TimestampFormat.validate_timestamp("not a number", "unix_seconds") is False
        assert TimestampFormat.validate_timestamp(None, "unix_seconds") is False

    def test_unknown_format(self):
        """Test 'unknown' format detection."""
        # Should auto-detect valid timestamp in any format
        assert TimestampFormat.validate_timestamp(1609459200, "unknown") is True


class TestTimestampFormatDetection:
    """Test TimestampFormat.detect_timestamp_format method."""

    def test_detect_unix_seconds(self):
        """Test detection of Unix seconds format."""
        # 2021-01-01 00:00:00 UTC
        result = TimestampFormat.detect_timestamp_format(1609459200)
        assert result == "unix_seconds"

    def test_detect_unix_milliseconds(self):
        """Test detection of Unix milliseconds format."""
        # 2021-01-01 00:00:00 UTC in milliseconds
        result = TimestampFormat.detect_timestamp_format(1609459200000)
        assert result == "unix_milliseconds"

    def test_small_number_rejected(self):
        """Test that small numbers are not detected as timestamps."""
        assert TimestampFormat.detect_timestamp_format(1000) is None
        assert TimestampFormat.detect_timestamp_format(9999) is None

    def test_invalid_value_rejected(self):
        """Test rejection of invalid values."""
        assert TimestampFormat.detect_timestamp_format("not a number") is None
        assert TimestampFormat.detect_timestamp_format(None) is None


class TestAppleTimestampSentinel:
    """Test Apple timestamp sentinel detection."""

    def test_distant_past(self):
        """Test detection of distantPast sentinel."""
        assert TimestampFormat.is_apple_timestamp_sentinel(-63114076800.0) is True

    def test_distant_future(self):
        """Test detection of distantFuture sentinel."""
        assert TimestampFormat.is_apple_timestamp_sentinel(63114076800.0) is True

    def test_normal_timestamp_not_sentinel(self):
        """Test that normal timestamps are not sentinels."""
        assert TimestampFormat.is_apple_timestamp_sentinel(631152000) is False
        assert TimestampFormat.is_apple_timestamp_sentinel(0) is False


class TestLooksLikeTimestamp:
    """Test the looks_like_timestamp function."""

    def test_valid_unix_seconds(self):
        """Test detection of valid Unix seconds timestamp."""
        is_ts, human = looks_like_timestamp(1609459200)
        assert is_ts is True
        assert human is not None
        assert "2021" in human

    def test_valid_with_time_keyword(self):
        """Test detection with time-related field name."""
        is_ts, human = looks_like_timestamp(1609459200, "created_time")
        assert is_ts is True

    def test_small_number_rejected(self):
        """Test that small numbers are rejected."""
        is_ts, human = looks_like_timestamp(100)
        assert is_ts is False
        assert human is None

    def test_invalid_type_rejected(self):
        """Test rejection of invalid types."""
        is_ts, human = looks_like_timestamp("not a number")  # type: ignore
        assert is_ts is False


class TestExtractTimestampsFromProtobuf:
    """Test timestamp extraction from protobuf data."""

    def test_extract_from_simple_dict(self):
        """Test extraction from simple dict with timestamp."""
        pb_data = {"timestamp": 1609459200}
        timestamps = extract_timestamps_from_protobuf(pb_data)
        assert len(timestamps) > 0

    def test_extract_from_nested_dict(self):
        """Test extraction from nested dict structure."""
        pb_data = {"outer": {"timestamp": 1609459200}}
        timestamps = extract_timestamps_from_protobuf(pb_data)
        assert len(timestamps) > 0

    def test_no_timestamps_found(self):
        """Test handling of data with no timestamps."""
        pb_data = {"name": "test", "count": 42}
        timestamps = extract_timestamps_from_protobuf(pb_data)
        assert len(timestamps) == 0

    def test_empty_dict(self):
        """Test handling of empty dict."""
        timestamps = extract_timestamps_from_protobuf({})
        assert timestamps == []


class TestAnalyzeProtobufForTimestamps:
    """Test the analyze_protobuf_for_timestamps function."""

    def test_analyze_dict_with_timestamp(self):
        """Test analysis of dict containing timestamp."""
        pb_data = {"created": 1609459200}
        result = analyze_protobuf_for_timestamps(pb_data)
        assert "has_timestamps" in result
        assert "timestamp_count" in result
        assert "timestamps" in result

    def test_analyze_json_string(self):
        """Test analysis of JSON string input."""
        json_str = '{"timestamp": 1609459200}'
        result = analyze_protobuf_for_timestamps(json_str)
        assert result["has_timestamps"] is True

    def test_analyze_invalid_json(self):
        """Test handling of invalid JSON string."""
        result = analyze_protobuf_for_timestamps("not valid json")
        assert "error" in result

    def test_analyze_empty_dict(self):
        """Test analysis of empty dict."""
        result = analyze_protobuf_for_timestamps({})
        assert result["has_timestamps"] is False
        assert result["timestamp_count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
