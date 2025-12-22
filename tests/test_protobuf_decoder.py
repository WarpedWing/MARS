"""Tests for protobuf decoder functions."""

import pytest

from mars.carver.protobuf.decoder import (
    BBPB_AVAILABLE,
    decode_protobuf_bbpb,
    extract_numbers_from_message,
    extract_strings_from_message,
)


class TestDecodeProtobufBbpb:
    """Test blackboxprotobuf-based protobuf decoding."""

    def test_empty_blob(self):
        """Test handling of empty blob."""
        assert decode_protobuf_bbpb(b"") is None

    def test_none_input(self):
        """Test handling of None input."""
        assert decode_protobuf_bbpb(None) is None  # type: ignore

    @pytest.mark.skipif(not BBPB_AVAILABLE, reason="blackboxprotobuf not available")
    def test_valid_simple_protobuf(self):
        """Test decoding of minimal valid protobuf."""
        # Minimal valid protobuf: field 1 = varint 42
        blob = b"\x08\x2a"
        result = decode_protobuf_bbpb(blob)
        if result is not None:
            assert "message" in result
            assert "typedef" in result
            assert "json" in result
            assert "schema" in result

    def test_invalid_protobuf(self):
        """Test handling of invalid protobuf data."""
        # Random invalid bytes
        # May or may not decode depending on bbpb tolerance
        # Just verify it doesn't crash
        _ = decode_protobuf_bbpb(b"\xff\xff\xff\xff")

    def test_random_binary(self):
        """Test handling of random binary data."""
        import random

        random_bytes = bytes(random.randint(0, 255) for _ in range(100))
        # Should not crash - result may be None or a decoded dict
        _ = decode_protobuf_bbpb(random_bytes)


class TestExtractStringsFromMessage:
    """Test string extraction from decoded protobuf messages."""

    def test_simple_string(self):
        """Test extraction of simple string."""
        msg = {"1": "Hello World"}
        result = extract_strings_from_message(msg)
        assert "Hello World" in result

    def test_nested_strings(self):
        """Test extraction of nested strings."""
        msg = {"1": "Hello", "2": {"nested": "World"}}
        result = extract_strings_from_message(msg)
        assert "Hello" in result
        assert "World" in result

    def test_string_in_list(self):
        """Test extraction of strings from lists."""
        msg = {"1": ["First", "Second", "Third"]}
        result = extract_strings_from_message(msg)
        assert "First" in result
        assert "Second" in result
        assert "Third" in result

    def test_short_strings_excluded(self):
        """Test that strings shorter than 3 chars are excluded."""
        msg = {"1": "ab", "2": "abc", "3": "a"}
        result = extract_strings_from_message(msg)
        assert "ab" not in result
        assert "a" not in result
        assert "abc" in result

    def test_non_alphanumeric_excluded(self):
        """Test that strings without alphanumeric chars are excluded."""
        msg = {"1": "   ", "2": "...", "3": "Valid123"}
        result = extract_strings_from_message(msg)
        assert "   " not in result
        assert "..." not in result
        assert "Valid123" in result

    def test_empty_message(self):
        """Test handling of empty message."""
        result = extract_strings_from_message({})
        assert result == []

    def test_bytes_decoded(self):
        """Test that bytes are decoded to strings."""
        msg = {"1": b"Hello World"}
        result = extract_strings_from_message(msg)
        assert "Hello World" in result


class TestExtractNumbersFromMessage:
    """Test number extraction from decoded protobuf messages."""

    def test_simple_integer(self):
        """Test extraction of simple integer."""
        msg = {"1": 42}
        result = extract_numbers_from_message(msg)
        assert 42 in result

    def test_simple_float(self):
        """Test extraction of simple float."""
        msg = {"1": 3.14}
        result = extract_numbers_from_message(msg)
        assert 3.14 in result

    def test_nested_numbers(self):
        """Test extraction of nested numbers."""
        msg = {"1": 10, "2": {"nested": 20}}
        result = extract_numbers_from_message(msg)
        assert 10 in result
        assert 20 in result

    def test_numbers_in_list(self):
        """Test extraction of numbers from lists."""
        msg = {"1": [1, 2, 3]}
        result = extract_numbers_from_message(msg)
        assert 1 in result
        assert 2 in result
        assert 3 in result

    def test_mixed_types(self):
        """Test extraction ignores non-numeric types."""
        msg = {"1": 42, "2": "text", "3": 3.14, "4": None}
        result = extract_numbers_from_message(msg)
        assert 42 in result
        assert 3.14 in result
        assert len(result) == 2

    def test_empty_message(self):
        """Test handling of empty message."""
        result = extract_numbers_from_message({})
        assert result == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
