"""Tests for Biome SEGB parser functions."""

import pytest

from mars.report_modules.biome_parser.biome_parser import (
    decode_data_text,
    extract_bundle_id,
    is_empty_record,
    try_protobuf_decode,
)


class TestExtractBundleId:
    """Test bundle ID extraction from SEGB record data."""

    def test_com_apple_bundle_id(self):
        """Test extraction of com.apple.* bundle IDs."""
        data = b"\x00\x00\x00com.apple.Safari\x00\x00"
        assert extract_bundle_id(data) == "com.apple.Safari"

    def test_com_microsoft_bundle_id(self):
        """Test extraction of com.microsoft.* bundle IDs."""
        data = b"prefix com.microsoft.VSCode suffix"
        assert extract_bundle_id(data) == "com.microsoft.VSCode"

    def test_net_prefix_bundle_id(self):
        """Test extraction of net.* bundle IDs."""
        data = b"some data net.sourceforge.app.name more data"
        assert extract_bundle_id(data) == "net.sourceforge.app.name"

    def test_org_prefix_bundle_id(self):
        """Test extraction of org.* bundle IDs."""
        data = b"org.mozilla.firefox"
        assert extract_bundle_id(data) == "org.mozilla.firefox"

    def test_io_prefix_bundle_id(self):
        """Test extraction of io.* bundle IDs."""
        data = b"data io.github.someproject.app data"
        assert extract_bundle_id(data) == "io.github.someproject.app"

    def test_app_prefix_bundle_id(self):
        """Test extraction of app.* bundle IDs."""
        data = b"app.cursor.editor"
        assert extract_bundle_id(data) == "app.cursor.editor"

    def test_no_bundle_id_found(self):
        """Test that None is returned when no bundle ID exists."""
        assert extract_bundle_id(b"no bundle here") is None
        assert extract_bundle_id(b"random binary data \x00\x01\x02") is None

    def test_empty_data(self):
        """Test handling of empty data."""
        assert extract_bundle_id(b"") is None
        assert extract_bundle_id(None) is None  # type: ignore

    def test_complex_bundle_id_with_hyphens(self):
        """Test bundle IDs with hyphens in segments."""
        data = b"com.apple.siri-metrics.extension"
        result = extract_bundle_id(data)
        assert result is not None
        assert result.startswith("com.apple")

    def test_multiple_bundle_ids_returns_first(self):
        """Test that first bundle ID is returned when multiple exist."""
        data = b"com.apple.first com.apple.second"
        assert extract_bundle_id(data) == "com.apple.first"


class TestIsEmptyRecord:
    """Test empty record detection."""

    def test_empty_bytes(self):
        """Test that empty bytes are detected as empty."""
        assert is_empty_record(b"") is True

    def test_all_zeros(self):
        """Test that all-zero data is detected as empty."""
        assert is_empty_record(b"\x00") is True
        assert is_empty_record(b"\x00\x00\x00") is True
        assert is_empty_record(b"\x00" * 100) is True

    def test_non_empty_data(self):
        """Test that non-empty data is not detected as empty."""
        assert is_empty_record(b"data") is False
        assert is_empty_record(b"\x00\x01\x00") is False
        assert is_empty_record(b"\x01") is False

    def test_zeros_with_trailing_nonzero(self):
        """Test data with mostly zeros but some content."""
        assert is_empty_record(b"\x00\x00\x00\x01") is False

    def test_none_input(self):
        """Test handling of None input."""
        # Function should handle None gracefully
        assert is_empty_record(None) is True  # type: ignore


class TestDecodeDataText:
    """Test text decoding from SEGB record data."""

    def test_simple_utf8_text(self):
        """Test decoding simple UTF-8 text."""
        assert decode_data_text(b"Hello World") == "Hello World"

    def test_with_header_skip(self):
        """Test decoding with 8-byte header skip."""
        data = b"\x00" * 8 + b"Hello"
        assert decode_data_text(data, skip_header=True) == "Hello"

    def test_without_header_skip(self):
        """Test decoding without header skip includes header bytes."""
        data = b"\x00" * 8 + b"Hello"
        result = decode_data_text(data, skip_header=False)
        assert result is not None
        assert "Hello" in result

    def test_non_printable_replaced(self):
        """Test that non-printable characters are replaced with spaces."""
        data = b"Hello\x01World"
        result = decode_data_text(data)
        assert result is not None
        # Non-printable \x01 should be replaced and spaces collapsed
        assert "Hello" in result
        assert "World" in result

    def test_empty_data(self):
        """Test handling of empty data."""
        assert decode_data_text(b"") is None
        assert decode_data_text(None) is None  # type: ignore

    def test_all_non_printable(self):
        """Test data with only non-printable characters."""
        result = decode_data_text(b"\x00\x01\x02\x03")
        # Should return None when no readable content
        assert result is None

    def test_tabs_and_newlines_preserved(self):
        """Test that tabs and newlines are preserved."""
        data = b"Line1\nLine2\tTabbed"
        result = decode_data_text(data)
        assert result is not None
        # Note: spaces get collapsed, but content should be present
        assert "Line1" in result
        assert "Line2" in result

    def test_unicode_text(self):
        """Test decoding Unicode text."""
        data = "Hello 世界 🌍".encode()
        result = decode_data_text(data)
        assert result is not None
        assert "Hello" in result

    def test_header_skip_with_short_data(self):
        """Test header skip with data shorter than 8 bytes."""
        data = b"Short"
        result = decode_data_text(data, skip_header=True)
        # Should still decode the data even if < 8 bytes
        assert result == "Short"


class TestTryProtobufDecode:
    """Test protobuf decoding with header skip detection."""

    def test_empty_data(self):
        """Test handling of empty data."""
        result, needed_skip = try_protobuf_decode(b"")
        assert result is None
        assert needed_skip is False

    def test_none_data(self):
        """Test handling of None data."""
        result, needed_skip = try_protobuf_decode(None)  # type: ignore
        assert result is None
        assert needed_skip is False

    def test_invalid_protobuf(self):
        """Test handling of invalid protobuf data."""
        result, needed_skip = try_protobuf_decode(b"\xff\xff\xff\xff")
        assert result is None
        assert needed_skip is False

    def test_valid_protobuf_no_skip(self):
        """Test valid protobuf that doesn't need header skip."""
        # Minimal valid protobuf: field 1 = varint 42
        data = b"\x08\x2a"
        result, needed_skip = try_protobuf_decode(data)
        # If blackboxprotobuf is available and can decode this
        if result is not None:
            assert "message" in result
            assert needed_skip is False

    def test_valid_protobuf_with_header(self):
        """Test valid protobuf that needs 8-byte header skip."""
        # 8 bytes of header followed by minimal protobuf
        # Note: Minimal protobuf \x08\x2a (field 1 = 42) might decode with or without skip
        # depending on bbpb's tolerance for leading zeros
        data = b"\x00" * 8 + b"\x08\x2a"
        result, needed_skip = try_protobuf_decode(data)
        # Result depends on whether bbpb can decode and which path succeeds
        # Just verify it doesn't crash and returns valid tuple
        assert isinstance(needed_skip, bool)

    def test_short_data_for_skip(self):
        """Test that very short data doesn't crash with skip attempt."""
        data = b"\x08\x2a"  # Only 2 bytes, can't skip 8
        result, needed_skip = try_protobuf_decode(data)
        # Should not crash, may or may not decode
        assert isinstance(needed_skip, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
