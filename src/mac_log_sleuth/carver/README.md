# SQLite Carver v3.4

A forensic SQLite data carver with intelligent timestamp validation.

## Quick Start

```bash
# Basic usage (default confidence threshold: 0.5)
python carve_sqlite.py database.sqlite

# Strict mode (high precision, fewer false positives)
python carve_sqlite.py database.sqlite --min-confidence 0.7

# Permissive mode (high recall, more candidates)
python carve_sqlite.py database.sqlite --min-confidence 0.3

# Custom timestamp range (for older/newer data)
python carve_sqlite.py database.sqlite --ts-start 2010-01-01 --ts-end 2035-01-01
```

## New in v3.4: Timestamp Confidence Scoring

**Problem**: Modern apps use timestamp-like IDs (Snowflake, Facebook event IDs) that look like timestamps but aren't.

**Solution**: Every timestamp gets a confidence score (0.0-1.0) based on:

- **Context analysis**: Field names like `created_at` vs `event_id`
- **Pattern detection**: Sequential IDs vs scattered timestamps
- **Clustering**: Timestamps often appear in pairs (`created_at`/`updated_at`)
- **Value patterns**: Suspicious prefixes, digit patterns

**Result**: Filter out false positives while keeping real timestamps.

See [TIMESTAMP_VALIDATION.md](TIMESTAMP_VALIDATION.md) for complete documentation.

## Features

### Core Capabilities
- ✅ Extracts deleted/unallocated SQLite data page by page
- ✅ Detects timestamps: Unix (sec/ms/μs/ns), Apple Cocoa, WebKit
- ✅ Extracts URLs, UTF-8 text runs, protobuf blobs
- ✅ Deduplicates artifacts by absolute file offset
- ✅ Optional page clustering for grouped analysis

### Timestamp Validation (NEW)
- ✅ Confidence scoring (0.0-1.0) for each timestamp
- ✅ Context-aware analysis (field names, keywords)
- ✅ Sequential pattern detection (filters ID sequences)
- ✅ Temporal clustering detection (timestamp pairs/groups)
- ✅ Configurable threshold (`--min-confidence`)

### Performance & Reliability
- ✅ Batch commits every 100 pages (crash recovery)
- ✅ Memory-efficient streaming (handles multi-GB files)
- ✅ Graceful error handling (continues on page errors)
- ✅ Progress bar with real-time updates

### Output
- ✅ SQLite database: `Carved/<db>_<timestamp>_Carved_Recovered.sqlite`
- ✅ CSV export: `carved_all.csv` (includes confidence scores)
- ✅ Protobuf JSON Lines: `carved_protobufs.jsonl`

## Command-Line Options

```
positional arguments:
  db                    SQLite file to carve

optional arguments:
  --no-cluster          Disable page clustering
  --no-protobuf         Skip protobuf decoding
  --no-pretty-protobuf  Compact JSON output for protobufs

  --ts-start YYYY-MM-DD
                        Start of timestamp validity range (default: 2015-01-01)
  --ts-end YYYY-MM-DD
                        End of timestamp validity range (default: 2030-01-01)

  --min-confidence FLOAT
                        Minimum confidence score for timestamps (0.0-1.0, default: 0.5)
```

## Output Schema

### carved_all Table/CSV

| Column | Type | Description |
|--------|------|-------------|
| `page_no` | INTEGER | Page number (0-indexed) |
| `page_offset` | INTEGER | Byte offset within page |
| `abs_offset` | INTEGER | Absolute file offset |
| `cluster_id` | INTEGER | Page cluster ID (or NULL) |
| `kind` | TEXT | `ts`, `url`, `text` |
| `value_text` | TEXT | String value (URLs, text) |
| `value_num` | INTEGER | Numeric value (timestamps) |
| `ts_kind_guess` | TEXT | `unix_sec`, `unix_milli`, `cocoa_sec`, etc. |
| `ts_human` | TEXT | Human-readable GMT timestamp |
| **`ts_confidence`** | REAL | **Confidence 0.0-1.0 (NEW)** |

### carved_protobufs Table

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `parent_abs_offset` | INTEGER | Offset in file |
| `page_no` | INTEGER | Page number |
| `abs_offset` | INTEGER | Absolute offset |
| `json_pretty` | TEXT | JSON representation |

Indexes on `page_no` and `abs_offset` for fast queries.

## Example Queries

```sql
-- High-confidence timestamps only
SELECT ts_human, value_num, ts_confidence
FROM carved_all
WHERE kind = 'ts' AND ts_confidence >= 0.8
ORDER BY value_num;

-- Confidence distribution
SELECT
    CASE
        WHEN ts_confidence >= 0.8 THEN 'HIGH'
        WHEN ts_confidence >= 0.6 THEN 'MEDIUM'
        WHEN ts_confidence >= 0.4 THEN 'LOW'
        ELSE 'VERY_LOW'
    END as confidence_level,
    COUNT(*) as count
FROM carved_all
WHERE kind = 'ts'
GROUP BY confidence_level;

-- Find URLs from specific time period
SELECT DISTINCT value_text
FROM carved_all
WHERE kind = 'url'
  AND page_no IN (
    SELECT page_no FROM carved_all
    WHERE kind = 'ts' AND ts_confidence >= 0.7
      AND value_num BETWEEN 1609459200 AND 1640995200  -- 2021
  );

-- Temporal correlation (timestamps + nearby text)
SELECT
    t.ts_human,
    t.ts_confidence,
    txt.value_text
FROM carved_all t
JOIN carved_all txt
    ON ABS(t.abs_offset - txt.abs_offset) < 200
WHERE t.kind = 'ts' AND t.ts_confidence >= 0.6
  AND txt.kind = 'text'
ORDER BY t.value_num
LIMIT 100;
```

## Use Cases

### Social Media Forensics
```bash
# Facebook/Instagram: Heavy ID pollution
python carve_sqlite.py facebook_messages.db --min-confidence 0.7
```

### Browser History Analysis
```bash
# Chrome/Firefox: Mix of timestamps and IDs
python carve_sqlite.py places.sqlite --min-confidence 0.5
```

### System Forensics
```bash
# macOS/iOS databases: Usually clean timestamps
python carve_sqlite.py fsevents.db --min-confidence 0.4
```

### Exploratory Analysis
```bash
# Unknown database: See everything first
python carve_sqlite.py unknown.db --min-confidence 0.0

# Review distribution, then re-run with appropriate threshold
python carve_sqlite.py unknown.db --min-confidence 0.6
```

## Architecture

```
carver/
├── carve_sqlite.py           # Main carver (orchestrates everything)
├── timestamp_validator.py    # Confidence scoring engine
├── timestamp_patterns.py     # Timestamp detection & interpretation
├── protobuf_extractor.py     # Protobuf parsing
├── TIMESTAMP_VALIDATION.md   # Detailed validation docs
└── README.md                 # This file
```

## Version History

### v3.4 (Latest)
- **NEW: Timestamp confidence scoring**
- Filters Snowflake IDs, Facebook IDs, and other timestamp-like values
- Context-aware analysis (field names, keywords)
- Sequential pattern detection
- `--min-confidence` CLI argument
- Updated schema with `ts_confidence` column

### v3.3
- Batch commits for crash recovery
- Memory-efficient CSV streaming
- Improved deduplication (absolute offsets)
- Error handling with graceful degradation
- Configurable timestamp range
- Database indexes for protobufs
- Removed unnecessary page dumps

### v3.2
- Initial release with basic carving
- Timestamp detection (multiple formats)
- URL and text extraction
- Protobuf decoding

## Requirements

- Python 3.9+
- Standard library only (no external dependencies for core functionality)
- Optional: `protobuf_extractor.py` for protobuf decoding

## License

By WarpedWing Labs

## Support

For issues, questions, or feature requests, see the main project repository.
