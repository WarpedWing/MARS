# SQLite Carver v3.4 - CLI Usage Guide

## Overview

The SQLite Carver is designed as a **standalone CLI tool** that can also be integrated into larger forensic pipelines. All functionality is accessible via command-line arguments with sensible defaults.

## Standalone CLI Usage

### Basic Usage

```bash
# Carve a corrupt database with default settings
python3 carve_sqlite.py /path/to/corrupt_database.sqlite
```

This will:
- Check database integrity and warn if intact
- Carve all pages for timestamps, URLs, text, and protobufs
- Use "balanced" filtering mode (confirmed + likely timestamps)
- Output to `Carved/{dbname}_{timestamp}/` in same directory as source database

### Output Files

```
Carved/database_20250116_143022/
├── database_Carved_Recovered.sqlite  # SQLite database with all artifacts
├── database_carved_all.csv           # CSV export of all artifacts
└── database_carved_protobufs.jsonl   # JSON Lines file with protobuf extractions
```

## Command-Line Options

### Timestamp Filtering

Control which timestamps are kept based on classification confidence:

```bash
# Only keep confirmed timestamps (highest precision)
python3 carve_sqlite.py database.sqlite --filter-mode strict

# Keep confirmed + likely timestamps (default, recommended)
python3 carve_sqlite.py database.sqlite --filter-mode balanced

# Keep everything except confirmed IDs (permissive)
python3 carve_sqlite.py database.sqlite --filter-mode permissive

# Keep all timestamps including ambiguous ones (kitchen sink)
python3 carve_sqlite.py database.sqlite --filter-mode all
```

**Filter Mode Comparison:**

| Mode | Keeps | Use Case |
|------|-------|----------|
| `strict` | Only CONFIRMED_TIMESTAMP | High-precision, low false positives |
| `balanced` | CONFIRMED_TIMESTAMP + LIKELY_TIMESTAMP | Recommended default |
| `permissive` | Everything except CONFIRMED_ID and INVALID | Maximum recall |
| `all` | Everything (no filtering) | Research, debugging |

### Timestamp Validity Range

Restrict timestamps to a specific time window:

```bash
# Only accept timestamps from 2020-2025
python3 carve_sqlite.py database.sqlite \
    --ts-start 2020-01-01 \
    --ts-end 2025-12-31
```

**Use Cases:**
- **Incident response**: Focus on incident timeframe (e.g., 2024-03-01 to 2024-03-31)
- **Historical analysis**: Filter out modern timestamps in old data
- **Noise reduction**: Exclude obviously invalid timestamps

### Output Directory

Control where carved data is written:

```bash
# Default: same directory as source database
python3 carve_sqlite.py database.sqlite

# Custom output directory
python3 carve_sqlite.py database.sqlite --output-dir /tmp/carved_output

# Output to current working directory
python3 carve_sqlite.py database.sqlite --output-dir .
```

### Protobuf Options

Control protobuf extraction and formatting:

```bash
# Disable protobuf extraction entirely
python3 carve_sqlite.py database.sqlite --no-protobuf

# Compact JSON (no pretty-printing)
python3 carve_sqlite.py database.sqlite --no-pretty-protobuf
```

**Protobuf Filtering:**
- Automatically filters out garbage protobufs with generic field names (`f1`, `f2`, etc.)
- Keeps protobufs with timestamp fields (`endTime`, `expiry`, etc.)
- Keeps protobufs with significant structure (10+ meaningful fields)

### Page Clustering

Control whether similar pages are clustered together:

```bash
# Disable clustering (process each page independently)
python3 carve_sqlite.py database.sqlite --no-cluster
```

**Clustering Benefits:**
- Groups related data from fragmented pages
- Identifies patterns across multiple pages
- **Default: Enabled** (recommended for most use cases)

## Complete Examples

### Example 1: Incident Response (Focused Timeframe)

```bash
# Carve database from compromised system
# Incident occurred March 15-20, 2024
python3 carve_sqlite.py /evidence/browser_cache.sqlite \
    --filter-mode strict \
    --ts-start 2024-03-15 \
    --ts-end 2024-03-20 \
    --output-dir /cases/2024-03-incident/carved
```

### Example 2: General Forensics (Maximum Data)

```bash
# Carve everything, let analyst filter later
python3 carve_sqlite.py /evidence/messages.sqlite \
    --filter-mode all \
    --output-dir /cases/case-12345/raw_carved
```

### Example 3: Quick Triage (High Confidence Only)

```bash
# Fast triage with only confirmed timestamps
python3 carve_sqlite.py /evidence/logs.sqlite \
    --filter-mode strict \
    --no-protobuf
```

### Example 4: Research Mode (No Filtering)

```bash
# Extract everything for analysis
python3 carve_sqlite.py sample.sqlite \
    --filter-mode all \
    --ts-start 1990-01-01 \
    --ts-end 2040-01-01
```

## Integration into Larger Pipelines

The carver is designed to be easily integrated into automated forensic workflows.

### Python Integration

```python
import subprocess
from pathlib import Path

def carve_database(db_path: Path, output_dir: Path, filter_mode: str = "balanced"):
    """Carve a SQLite database as part of a larger pipeline."""

    cmd = [
        "python3",
        "carve_sqlite.py",
        str(db_path),
        "--filter-mode", filter_mode,
        "--output-dir", str(output_dir),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception(f"Carving failed: {result.stderr}")

    return result.stdout

# Example usage in pipeline
for db_file in evidence_dir.glob("*.sqlite"):
    print(f"Processing {db_file.name}...")
    carve_database(db_file, output_dir / db_file.stem)
```

### Bash Pipeline

```bash
#!/bin/bash

# Process all SQLite databases in evidence directory
for db in /evidence/*.sqlite; do
    echo "Carving $db..."
    python3 carve_sqlite.py "$db" \
        --filter-mode balanced \
        --output-dir /cases/case-123/carved/
done

# Combine all CSV outputs
cat /cases/case-123/carved/*/carved_all.csv > /cases/case-123/all_artifacts.csv

# Search for specific timestamps
grep "2024-03-15" /cases/case-123/all_artifacts.csv > /cases/case-123/incident_day.csv
```

### Make/Task Integration

```makefile
# Makefile for forensic pipeline

EVIDENCE_DIR := /evidence
OUTPUT_DIR := /cases/case-123

.PHONY: carve
carve:
	@for db in $(EVIDENCE_DIR)/*.sqlite; do \
		python3 carve_sqlite.py $$db \
			--filter-mode balanced \
			--output-dir $(OUTPUT_DIR)/carved; \
	done

.PHONY: analyze
analyze: carve
	python3 analyze_timestamps.py $(OUTPUT_DIR)/carved
	python3 generate_report.py $(OUTPUT_DIR)
```

## Output Format Details

### SQLite Database Schema

The `*_Carved_Recovered.sqlite` database contains a single table:

```sql
CREATE TABLE carved_all (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    page_no INTEGER,              -- Page number in source database
    page_offset INTEGER,          -- Offset within page
    abs_offset INTEGER,           -- Absolute offset in file
    cluster_id INTEGER,           -- Cluster ID (if clustering enabled)
    kind TEXT,                    -- Artifact type: 'ts', 'url', 'text', 'protobuf'
    value_text TEXT,              -- Text content (for URLs, text)
    value_num INTEGER,            -- Numeric value (for timestamps)
    ts_kind_guess TEXT,           -- Timestamp format: unix_sec, unix_milli, etc.
    ts_human TEXT,                -- Human-readable timestamp
    ts_classification TEXT,       -- Classification: confirmed_timestamp, ambiguous, etc.
    ts_reason TEXT,               -- Why this classification was chosen
    ts_source_url TEXT,           -- URL this timestamp came from (if applicable)
    ts_field_name TEXT            -- Field name like 'endTime', 'created_at'
);
```

### CSV Format

Same columns as SQLite database, suitable for importing into Excel, Splunk, etc.

### JSONL Format (Protobufs)

```json
{
  "parent_abs_offset": 123456,
  "page_no": 42,
  "abs_offset": 123456,
  "protobuf": {
    "state": 1,
    "endTime": 1592503917395,
    "fileSize": 392223
  },
  "timestamp_analysis": {
    "timestamp_count": 1,
    "timestamps": [
      {
        "field": "endTime",
        "value": 1592503917395,
        "human_readable": "2020-06-18 18:11:57 GMT"
      }
    ]
  }
}
```

## Database Integrity Warning

When you run the carver on an **intact** (non-corrupt) database, you'll see this warning:

```
[!] WARNING: Database appears to be intact!
    Reason: Database integrity check passed (PRAGMA integrity_check = ok)
    This carver is designed for CORRUPT databases.
    For intact databases, use standard SQLite tools instead.
    If you want to proceed anyway, press Ctrl+C to cancel or wait 5 seconds...
```

**What This Means:**
- The database can be queried normally with `sqlite3` or DB Browser
- Using the carver on intact databases is unnecessary and slow
- You should use standard SQL queries instead

**To Query Intact Databases:**
```bash
# Use sqlite3 CLI
sqlite3 database.sqlite "SELECT * FROM table_name"

# Use DB Browser for SQLite (GUI)
# https://sqlitebrowser.org/
```

## Timestamp Format Support

The carver detects these timestamp formats:

### Decimal Formats
- **Unix seconds**: 1609459200 → 2021-01-01 00:00:00 GMT
- **Unix milliseconds**: 1609459200000 → 2021-01-01 00:00:00 GMT
- **Unix microseconds**: 1609459200000000 → 2021-01-01 00:00:00 GMT
- **Unix nanoseconds**: 1609459200000000000 → 2021-01-01 00:00:00 GMT
- **Apple Cocoa seconds**: 662688000 → 2022-01-01 00:00:00 GMT (since 2001)
- **Apple Cocoa nanoseconds**: 662688000000000000 → 2022-01-01 00:00:00 GMT
- **WebKit microseconds**: 13258032000000000 → 2021-01-01 00:00:00 GMT (since 1601)

### Hexadecimal Formats (NEW)
- **Hex Unix timestamps**: 5b994548 → 1536771400 → 2018-09-12 16:56:40 GMT
- **Hex pairs**: 5b994548.5ad909e0 → Two separate timestamps

**Example hex timestamp:**
```
Input bytes:  5b 99 45 48 2e 5a d9 09 e0
ASCII:        [ \x99 E H  .  Z \xd9 \t \xe0
Hex string:   5b994548.5ad909e0
Timestamp 1:  5b994548 = 1536771400 = 2018-09-12 16:56:40 GMT
Timestamp 2:  5ad909e0 = 1524173280 = 2018-04-19 21:28:00 GMT
```

## Classification System

The V2 classifier uses these categories:

| Classification | Symbol | Description | Example |
|----------------|--------|-------------|---------|
| CONFIRMED_TIMESTAMP | [+] | Field name or URL confirms it's a timestamp | `{"endTime": 1592503917395}` |
| CONFIRMED_ID | [-] | Snowflake ID or URL-based ID | Facebook post ID: 123456789012345678 |
| LIKELY_TIMESTAMP | [~] | Field name suggests timestamp | `{"lastSeen": 1609459200}` |
| LIKELY_ID | [x] | Field name suggests ID | `{"userId": 1234567890}` |
| AMBIGUOUS | [?] | Valid format, no context | Random number: 1609459200 |
| INVALID | [!] | Not a valid timestamp | Out of range or malformed |

## Performance Tips

1. **Use `--filter-mode strict`** for fastest processing (fewer false positives to analyze)
2. **Use `--no-protobuf`** if you don't need protobuf extraction (faster)
3. **Use `--ts-start` and `--ts-end`** to reduce search space
4. **Keep clustering enabled** for better context (minimal performance impact)

## Troubleshooting

### "No timestamps found" on intact database

**Problem:** Running carver on intact macOS powerlog shows "no timestamps found"

**Solution:** The database is not corrupt! Use standard SQLite queries instead:
```bash
# Query the powerlog directly
sqlite3 powerlog.sqlite "SELECT * FROM PLBatteryAgent_EventBackward_Battery LIMIT 10"
```

The carver is designed for **corrupt or deleted** SQLite files where normal queries fail.

### Too many false positives

**Problem:** Getting too many IDs classified as timestamps

**Solution:** Use stricter filtering:
```bash
python3 carve_sqlite.py database.sqlite --filter-mode strict
```

### Missing legitimate timestamps

**Problem:** Known timestamps not appearing in output

**Solution:** Use permissive or all mode:
```bash
python3 carve_sqlite.py database.sqlite --filter-mode all
```

## Exit Codes

- **0**: Success
- **1**: Invalid arguments or file not found
- **2**: Database header invalid (not SQLite)

## Dependencies

Required Python packages:
- `unfurl` (URL timestamp extraction)
- `time_decode` (timestamp format detection)
- Standard library: `sqlite3`, `argparse`, `datetime`, `pathlib`, `csv`, `json`

Install dependencies:
```bash
pip install unfurl-app time-decode
```

## License

By WarpedWing Labs

## Support

For issues, feature requests, or questions:
- File an issue on GitHub
- Check existing documentation in the `carver/` directory
- Review `V2_SYSTEM_OVERVIEW.md` for technical details
