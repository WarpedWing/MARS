#!/usr/bin/env python3

"""
protobuf_extractor.py

Schema-agnostic, heuristic Protocol Buffers decoder.

Exports:
  - maybe_decode_protobuf(blob: bytes, max_depth: int = 4) -> dict | list | None
      Tries to decode a blob as a protobuf message (unknown schema). Returns a
      Python object (dict or list) when confident; otherwise returns None.

  - to_json(obj, pretty: bool = True) -> str
      JSON serializer for decoded structures.

Notes:
  * Handles wire types: varint(0), 64-bit(1), length-delimited(2), 32-bit(5).
  * Detects:
      - packed varints inside length-delimited fields,
      - embedded messages (recursive parse),
      - printable UTF-8 strings,
      - 64/32-bit numbers (also surfaces float/double when they look plausible).
  * Field keys are numeric ("f1", "f2", …). Repeated fields become lists.
  * Depth/budget limits keep it safe on random data.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any

# ------------ Varint helpers ------------


def _read_varint(buf: bytes, i: int) -> tuple[int | None, int]:
    """Return (value, next_index) or (None, i) if invalid."""
    result = 0
    shift = 0
    start = i
    while i < len(buf) and shift <= 70:
        b = buf[i]
        result |= (b & 0x7F) << shift
        i += 1
        if not (b & 0x80):
            return result, i
        shift += 7
    return None, start


def _zigzag(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


def _is_printable_utf8(b: bytes) -> bool:
    try:
        s = b.decode("utf-8")
    except Exception:
        return False
    # heuristic: require at least one letter and not too many controls
    if not any(c.isalpha() for c in s):
        return False
    return not sum(1 for ch in s if ord(ch) < 32 and ch not in ("\t", "\n", "\r")) > 0


# ------------ Core parsing ------------


class _Budget:
    def __init__(
        self, max_depth: int = 4, max_fields: int = 2048, max_bytes: int = 1_000_000
    ):
        self.max_depth = max_depth
        self.max_fields = max_fields
        self.max_bytes = max_bytes


def _add_field(obj: dict, field_no: int, value: Any):
    key = f"f{field_no}"
    if key in obj:
        exist = obj[key]
        if isinstance(exist, list):
            exist.append(value)
        else:
            obj[key] = [exist, value]
    else:
        obj[key] = value


def _parse_message(
    buf: bytes, i: int, end: int, depth: int, budget: _Budget
) -> tuple[dict | None, int]:
    if depth > budget.max_depth:
        return None, i
    out = {}
    fields = 0
    while i < end:
        key, i2 = _read_varint(buf, i)
        if key is None:
            break
        i = i2
        field_no = key >> 3
        wire_type = key & 0x7
        fields += 1
        if fields > budget.max_fields:
            break

        # wire type dispatch
        if wire_type == 0:
            # varint
            v, i = _read_varint(buf, i)
            if v is None:
                break
            # expose both raw and (optionally) zigzag if it looks signed
            _add_field(out, field_no, v)
        elif wire_type == 1:
            # 64-bit
            if i + 8 > end:
                break
            raw = buf[i : i + 8]
            i += 8
            u64 = int.from_bytes(raw, "little", signed=False)
            # also attempt double
            try:
                dbl = struct.unpack("<d", raw)[0]
                # prefer float if finite and "reasonable" magnitude
                if math.isfinite(dbl) and (abs(dbl) < 1e308):
                    _add_field(out, field_no, {"u64": u64, "f64": dbl})
                else:
                    _add_field(out, field_no, u64)
            except Exception:
                _add_field(out, field_no, u64)
        elif wire_type == 2:
            # length-delimited
            ln, i3 = _read_varint(buf, i)
            if ln is None or i3 + ln > end:
                break
            i = i3
            seg = buf[i : i + ln]
            i += ln

            # Heuristics for length-delimited:
            # 1) printable UTF-8 → string
            if _is_printable_utf8(seg):
                _add_field(out, field_no, seg.decode("utf-8", errors="replace"))
                continue

            # 2) packed varints? Try to parse all as varints
            j = 0
            packed = []
            ok = True
            while j < len(seg):
                v, j2 = _read_varint(seg, j)
                if v is None:
                    ok = False
                    break
                packed.append(v)
                j = j2
            if ok and packed:
                _add_field(out, field_no, {"packed_varint": packed})
                continue

            # 3) embedded message (recursive)
            submsg, _ = _parse_message(seg, 0, len(seg), depth + 1, budget)
            if submsg is not None and submsg != {}:
                _add_field(out, field_no, submsg)
            else:
                # raw bytes fallback (hex)
                _add_field(out, field_no, {"bytes_hex": seg.hex()})
        elif wire_type == 5:
            # 32-bit
            if i + 4 > end:
                break
            raw = buf[i : i + 4]
            i += 4
            u32 = int.from_bytes(raw, "little", signed=False)
            try:
                f32 = struct.unpack("<f", raw)[0]
                if math.isfinite(f32) and (abs(f32) < 1e38):
                    _add_field(out, field_no, {"u32": u32, "f32": f32})
                else:
                    _add_field(out, field_no, u32)
            except Exception:
                _add_field(out, field_no, u32)
        else:
            # unsupported wire types (3,4 deprecated / groups)
            break

        # size budget
        if end - i > budget.max_bytes:
            break

    return out if out else {}, i


def maybe_decode_protobuf(blob: bytes, max_depth: int = 4):
    """
    Try to parse blob as a protobuf message. Returns a Python dict/list or None.
    """
    if not blob or len(blob) < 2:
        return None
    budget = _Budget(max_depth=max_depth)
    msg, pos = _parse_message(blob, 0, len(blob), 1, budget)
    # Very small or empty → probably not a protobuf message
    if msg is None or msg == {}:
        return None
    return msg


def to_json(obj, pretty: bool = True) -> str:
    return json.dumps(obj, indent=2 if pretty else None, ensure_ascii=False)
