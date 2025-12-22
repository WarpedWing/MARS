# Biome SEGB Parser

This module parses macOS Biome SEGB (Sequential Event Grid Block) files to extract system telemetry and application metrics.

## Disclaimer

**This report is for informational and investigative purposes only.** The data presented should be
independently verified before being relied upon for any legal, regulatory, or evidentiary purpose.

### Limitations

- **Protobuf decoding**: Apple does not publish Biome protobuf schemas. Decoding attempts are
best-effort and may yield incomplete or incorrectly interpreted data.
- **Bundle ID extraction**: Bundle identifiers (e.g., `com.apple.*`) are extracted using pattern
matching heuristics. Some records may have missing or incorrectly parsed bundle IDs.
- **Binary data interpretation**: Non-protobuf binary data is presented as hex dumps with attempted
text extraction. Interpretation requires domain knowledge of specific Biome stream types.
- **Timestamp formats**: Biome timestamps use Mac Absolute Time (seconds since 2001-01-01).
Conversion accuracy depends on timezone assumptions.
- **Stream coverage**: Not all Biome stream types are fully understood. Some streams may contain
data that cannot be meaningfully parsed.

## What Are Biome Files?

Biome is Apple's system telemetry framework introduced in macOS 12 (Monterey). It records
application usage, system events, and device metrics in binary SEGB format files stored under `~/Library/Biome/`.

Forensically relevant data includes:

- Application launch/termination events
- Device wake/sleep cycles
- Location context changes
- Intents and Siri interactions
- Media playback events

## Parsed Data

### SEGB Records

Each SEGB file contains a stream of timestamped records. The parser extracts:

- **Timestamps**: Two timestamp fields per record (typically event start/end or creation/modification)
- **Bundle ID**: Application identifier when present (e.g., `com.apple.Safari`)
- **Stream type**: v1 or v2 format indicator
- **State flags**: Record state metadata
- **Protobuf content**: Decoded protobuf fields when parsing succeeds
- **Raw data**: Hex dump and extracted text for manual analysis

### Empty Record Detection

Records containing only null bytes are flagged as empty and can be filtered during analysis.

## Data Sources

| Artifact | Path Pattern | macOS Versions |
| ---------- | -------------- | ---------------- |
| Biome Streams | `~/Library/Biome/streams/**/*.segb` | 12.0+ (Monterey) |
| Biome Streams | `~/Library/Biome/streams/**/*store` | 12.0+ |

Common stream locations include:

- `public/` - General system telemetry
- `restricted/` - Privacy-sensitive metrics (requires elevated access)

## Not Currently Parsed

- **Protobuf schema mapping**: Without Apple's schemas, field names and types are inferred
- **Biome database files**: SQLite databases in Biome directories are not parsed by this module
- **Restricted streams**: Some streams require root access and may not be available from standard extractions

## Output Files

- `biome_records.csv` - All extracted records with the following columns:
  - `source_file` - Path to source SEGB file
  - `bundle_id` - Extracted application bundle identifier
  - `stream_type` - v1 or v2 format
  - `offset` - Byte offset in source file
  - `metadata_offset` - Metadata section offset
  - `state` - Record state flags
  - `timestamp1`, `timestamp2` - Extracted timestamps (ISO 8601)
  - `data_size` - Raw data length in bytes
  - `data_text` - Decoded text content
  - `data_hex` - Hexadecimal data dump
  - `protobuf_json` - Decoded protobuf as JSON (when successful)
  - `protobuf_strings` - Extracted string fields from protobuf

## Dependencies

This module requires the `ccl_segb` library for SEGB file parsing.

## Scan Type

**Exemplar only** - This module runs during live system scans to capture current Biome state. It is
not used for carved/recovered file processing.
