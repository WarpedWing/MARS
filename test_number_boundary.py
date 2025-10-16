#!/usr/bin/env python3
"""Test that we don't extract numbers from within larger numbers."""

import re

# New regexes with word boundaries
TIMESTAMP_REGEX = re.compile(rb"(?<![0-9])[0-9]{9,}(?:\.[0-9]+)?(?![0-9])")
HEX_TIMESTAMP_REGEX = re.compile(rb"(?<![0-9a-fA-F])[0-9a-fA-F]{8}(?:\.[0-9a-fA-F]{8})?(?![0-9a-fA-F])")

# Test cases
test_cases = [
    # (input, expected_decimal_matches, expected_hex_matches, description)
    (b"<20180321165313.CE7A338000096@example.com>", [], [], "ISO-8601 in email - should not match substring"),
    (b"message:%3C20180321165313.CE7A338000096@example.com%3E", [], [], "URL-encoded ISO-8601 - should not match substring"),
    (b"<2018-03-21 16:53:13>", [], [], "Formatted ISO-8601 - should not match"),
    (b"timestamp:1521649993", [b"1521649993"], [], "Valid Unix timestamp - should match"),
    (b"etPan.5b994548.5ad909e0.fe@example.org", [b"5ad909e0"], [b"5b994548", b"5ad909e0"], "Hex timestamps in Message-ID - should match both"),
    (b"unix:1521649993.123", [b"1521649993.123"], [], "Unix timestamp with fractional seconds"),
    (b"id:123456789012345", [b"123456789012345"], [], "Long ID - should match as complete number"),
    (b"5b994548", [], [b"5b994548"], "Standalone 8-char hex - should match"),
    (b"abc5b994548def", [], [], "Hex in middle of word - should NOT match"),
    (b"123456789", [b"123456789"], [], "9-digit number - should match"),
    (b"12345678", [], [], "8-digit number - too short for decimal, not valid standalone hex context"),
    (b"before 1521649993 after", [b"1521649993"], [], "Number with spaces - should match"),
    (b"5b994548.5ad909e0", [], [b"5b994548.5ad909e0"], "Dot-separated hex pair - should match as one"),
]

def url_decode_bytes(data: bytes) -> bytes:
    """Simple URL decoder for testing."""
    decoded = bytearray()
    i = 0
    length = len(data)
    while i < length:
        b = data[i]
        if b == 0x25 and i + 2 < length:  # '%'
            hex_chunk = data[i + 1 : i + 3]
            try:
                decoded_byte = int(hex_chunk, 16)
                decoded.append(decoded_byte)
                i += 3
                continue
            except ValueError:
                pass
        decoded.append(b)
        i += 1
    return bytes(decoded)

print("Testing timestamp extraction with word boundaries...\n")
print("=" * 80)

all_passed = True
for test_input, expected_decimal, expected_hex, description in test_cases:
    print(f"\nTest: {description}")
    print(f"Input: {test_input}")

    # Test on both original and URL-decoded
    decoded = url_decode_bytes(test_input)
    if decoded != test_input:
        print(f"Decoded: {decoded}")

    # Find decimal matches
    decimal_matches = [m.group(0) for m in TIMESTAMP_REGEX.finditer(test_input)]
    decimal_matches_decoded = [m.group(0) for m in TIMESTAMP_REGEX.finditer(decoded)]
    all_decimal = sorted(set(decimal_matches + decimal_matches_decoded))

    # Find hex matches
    hex_matches = [m.group(0) for m in HEX_TIMESTAMP_REGEX.finditer(test_input)]
    hex_matches_decoded = [m.group(0) for m in HEX_TIMESTAMP_REGEX.finditer(decoded)]
    all_hex = sorted(set(hex_matches + hex_matches_decoded))

    # Check results
    decimal_pass = sorted(all_decimal) == sorted(expected_decimal)
    hex_pass = sorted(all_hex) == sorted(expected_hex)

    print(f"Expected decimal: {expected_decimal}")
    print(f"Got decimal:      {all_decimal}")
    print(f"Decimal: {'✓ PASS' if decimal_pass else '✗ FAIL'}")

    print(f"Expected hex:     {expected_hex}")
    print(f"Got hex:          {all_hex}")
    print(f"Hex: {'✓ PASS' if hex_pass else '✗ FAIL'}")

    if not (decimal_pass and hex_pass):
        all_passed = False

    print("-" * 80)

print("\n" + "=" * 80)
if all_passed:
    print("✓ ALL TESTS PASSED")
else:
    print("✗ SOME TESTS FAILED")
