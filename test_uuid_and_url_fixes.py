#!/usr/bin/env python3
"""Test UUID exclusion and URL decoder byte range fix."""

import sys
sys.path.insert(0, 'src')

from mac_log_sleuth.carver.timestamp_patterns import (
    find_hex_timestamp_candidates,
    url_decode_bytes,
)

print("=" * 80)
print("Test 1: UUID Exclusion")
print("=" * 80)

# Test UUID exclusion
test_uuid = b"6E1DD98F-67C5-4222-A810-5BD8C88D8745"
print(f"\nInput: {test_uuid}")
print("Expected: No hex timestamps (entire string is a UUID)")

hex_timestamps = find_hex_timestamp_candidates(test_uuid)
print(f"Found {len(hex_timestamps)} hex timestamps")
for offset, hex_val, kind, human in hex_timestamps:
    print(f"  Offset {offset}: {hex_val} ({kind}) = {human}")

if len(hex_timestamps) == 0:
    print("✓ PASS - UUID correctly excluded")
else:
    print("✗ FAIL - UUID segments were incorrectly parsed as timestamps")

print("\n" + "=" * 80)
print("Test 2: Valid hex timestamp still works")
print("=" * 80)

test_valid = b"etPan.5b994548.5ad909e0.fe@example.org"
print(f"\nInput: {test_valid}")
print("Expected: Two hex timestamps (5b994548 and 5ad909e0)")

hex_timestamps = find_hex_timestamp_candidates(test_valid)
print(f"Found {len(hex_timestamps)} hex timestamp(s)")
for offset, hex_val, kind, human in hex_timestamps:
    print(f"  Offset {offset}: {hex_val} ({kind}) = {human}")

if len(hex_timestamps) >= 1:
    print("✓ PASS - Valid hex timestamps still detected")
else:
    print("✗ FAIL - Valid hex timestamps were not detected")

print("\n" + "=" * 80)
print("Test 3: URL decoder byte range error fix")
print("=" * 80)

# Test cases that might cause byte range errors
test_cases = [
    b"normal%20string",  # Valid URL encoding
    b"%GG invalid hex",  # Invalid hex
    b"%999 out of range",  # Out of range value (0x999 = 2457, > 255)
    b"test%FFF extra",  # Way out of range (0xFFF = 4095)
    b"%00 null byte",  # Valid: 0
    b"%FF max byte",  # Valid: 255
]

all_passed = True
for test_input in test_cases:
    print(f"\nInput: {test_input}")
    try:
        decoded, mapping = url_decode_bytes(test_input)
        print(f"Decoded: {decoded}")
        print(f"✓ No error")
    except Exception as e:
        print(f"✗ Error: {e}")
        all_passed = False

if all_passed:
    print("\n✓ PASS - All URL decode cases handled without errors")
else:
    print("\n✗ FAIL - Some URL decode cases caused errors")

print("\n" + "=" * 80)
print("Test 4: Mixed UUID and valid timestamps")
print("=" * 80)

# More realistic test: UUID and valid timestamp in same data
test_mixed = b"user_id=6E1DD98F-67C5-4222-A810-5BD8C88D8745&timestamp=5b994548"
print(f"\nInput: {test_mixed}")
print("Expected: 1 hex timestamp (5b994548), UUID excluded")

hex_timestamps = find_hex_timestamp_candidates(test_mixed)
print(f"Found {len(hex_timestamps)} hex timestamp(s)")
for offset, hex_val, kind, human in hex_timestamps:
    print(f"  Offset {offset}: {hex_val} ({kind}) = {human}")

# Check that we got the timestamp but not the UUID
found_timestamp = any(hex_val == "5b994548" for _, hex_val, _, _ in hex_timestamps)
found_uuid_segment = any(hex_val.upper() in ["6E1DD98F", "67C5", "4222", "A810"]
                         for _, hex_val, _, _ in hex_timestamps)

if found_timestamp and not found_uuid_segment:
    print("✓ PASS - Valid timestamp found, UUID segments excluded")
elif not found_timestamp:
    print("✗ FAIL - Valid timestamp not found")
else:
    print("✗ FAIL - UUID segments incorrectly parsed as timestamps")

print("\n" + "=" * 80)
