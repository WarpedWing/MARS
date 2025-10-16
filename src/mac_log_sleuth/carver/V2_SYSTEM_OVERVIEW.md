# SQLite Carver V2 Classification System

## Overview

The V2 system replaces fuzzy confidence scores (0.0-1.0) with **clear categorical classifications** using industry-standard tools (Unfurl and time_decode).

## Architecture

```bash
┌─────────────────────────────────────────────────────────┐
│                  SQLite Carver v3.4                     │
│              (carve_sqlite.py - main)                   │
└─────────────────────────────────────────────────────────┘
                         │
                         ├──> Find URLs (existing)
                         ├──> Find raw numbers
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         timestamp_patterns.py                           │
│  • find_timestamp_candidates()                          │
│  • Pre-filter by digit count (10, 13, 16-17, 19)        │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         url_analyzer.py                                 │
│  • parse_url_with_unfurl()   ← Uses Unfurl              │
│  • Extracts timestamps from URLs                        │
│  • Identifies IDs in URLs (Snowflake, etc.)             │
│  • Returns URLContext with all findings                 │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         timestamp_classifier_v2.py                      │
│  • classify_page_timestamps()                           │
│  • Integrates all detection methods                     │
│  • Priority order:                                      │
│    1. URL context (highest)                             │
│    2. Field name analysis                               │
│    3. Snowflake ID detection                            │
│    4. time_decode format validation                     │
│    5. Ambiguous (valid format, no context)              │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│         timestamp_classifier.py                         │
│  • TimestampClassification enum                         │
│  • should_keep_timestamp() - filter logic               │
│  • ClassificationStats - reporting                      │
└─────────────────────────────────────────────────────────┘
                         │
                         ▼
                    Database & CSV
          (includes classification + reason)
```

## Classification Categories

| Classification | Meaning | Keep in balanced mode? |
|---|---|---|
| **CONFIRMED_TIMESTAMP** | Field name + format match, or URL timestamp param | [+] Yes |
| **CONFIRMED_ID** | Snowflake structure, or URL ID field | [-] No |
| **LIKELY_TIMESTAMP** | Field name suggests timestamp, valid format | [+] Yes |
| **LIKELY_ID** | Field name suggests ID, or near URL with IDs | [-] No |
| **AMBIGUOUS** | Valid timestamp format, no context | [?] Depends on mode |
| **INVALID** | Doesn't match any known format | [-] No |

## Filter Modes

```bash
--filter-mode strict      # Only CONFIRMED_TIMESTAMP
--filter-mode balanced    # CONFIRMED_TIMESTAMP + LIKELY_TIMESTAMP (default)
--filter-mode permissive  # Everything except CONFIRMED_ID and INVALID
--filter-mode all         # No filtering (for debugging)
```

## Example Usage

### Basic (Balanced Mode)

```bash
python carve_sqlite.py /path/to/facebook.db
```

Output includes all confirmed and likely timestamps, filters out Snowflake IDs.

### Strict (High Precision)

```bash
python carve_sqlite.py /path/to/facebook.db --filter-mode strict
```

Only keeps timestamps with strong evidence (field names or URL context).

### See Everything

```bash
python carve_sqlite.py /path/to/facebook.db --filter-mode all
```

Includes all candidates for manual review.

### Custom Output Directory

```bash
python carve_sqlite.py /path/to/db.sqlite --output-dir /output/path
```

## Output Schema

### CSV Columns (carved_all.csv)

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `page_no` | INTEGER | Page number | 42 |
| `page_offset` | INTEGER | Byte offset within page | 256 |
| `abs_offset` | INTEGER | Absolute file offset | 172288 |
| `cluster_id` | INTEGER | Page cluster ID | 2 |
| `kind` | TEXT | Artifact type | `ts`, `url`, `text` |
| `value_text` | TEXT | String value | URL or text |
| `value_num` | INTEGER | Numeric value | 1609459200 |
| `ts_kind_guess` | TEXT | Format type | `unix_sec`, `snowflake` |
| `ts_human` | TEXT | Human-readable | `2021-01-01 00:00:00 GMT` |
| **`ts_classification`** | TEXT | **Classification** | `confirmed_timestamp` |
| **`ts_reason`** | TEXT | **Why classified** | `Field name indicates timestamp` |
| **`ts_source_url`** | TEXT | **Source URL** | `https://...` (if applicable) |
| **`ts_field_name`** | TEXT | **Field name** | `created_at` (if detected) |

### Example Rows

```csv
page_no,page_offset,abs_offset,cluster_id,kind,value_text,value_num,ts_kind_guess,ts_human,ts_classification,ts_reason,ts_source_url,ts_field_name
42,120,172408,2,ts,,1609459200,unix_sec,2021-01-01 00:00:00 GMT,confirmed_timestamp,Field name indicates timestamp,,created_at
42,450,172738,2,ts,,123456789012345678,snowflake,Snowflake ID (dc:5 worker:3...),confirmed_id,Matches Snowflake ID structure,,
42,567,172855,2,url,https://facebook.com/posts/123456789012345678,,,,,,https://facebook.com/posts/123456789012345678,
```

## SQL Queries

### Get Only Confirmed Timestamps

```sql
SELECT ts_human, value_num, ts_reason, ts_field_name
FROM carved_all
WHERE kind = 'ts'
  AND ts_classification = 'confirmed_timestamp'
ORDER BY value_num;
```

### Classification Distribution

```sql
SELECT
    ts_classification,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
FROM carved_all
WHERE kind = 'ts'
GROUP BY ts_classification
ORDER BY count DESC;
```

### Find Snowflake IDs with Embedded Timestamps

```sql
SELECT
    value_num as snowflake_id,
    ts_human as embedded_timestamp,
    ts_source_url
FROM carved_all
WHERE ts_kind_guess = 'snowflake'
  AND ts_classification = 'confirmed_id';
```

### Timestamps from Specific Platform

```sql
SELECT DISTINCT ts_human, ts_reason
FROM carved_all
WHERE ts_source_url LIKE '%facebook.com%'
  AND ts_classification IN ('confirmed_timestamp', 'likely_timestamp')
ORDER BY value_num;
```

## Terminal Output

```bash
╭──────────────────────────────╮
│      SQLite Carver v3.4      │
│      by WarpedWing Labs      │
╰──────────────────────────────╯

[[~]] File loaded: facebook_messages.db
[•] Page size: 4096 bytes  |  Encoding: UTF-8
[•] Timestamp filter mode: balanced
[[~]] Using V2 classifier (Unfurl + time_decode)

─────────────────────────────────────────────
[========================] 100%  Parsing page 1024/1024
─────────────────────────────────────────────

Timestamp Classification Summary (15,234 total):
──────────────────────────────────────────────────
[+] Confirmed Timestamps:   3,421 ( 22.5%)
[~]  Likely Timestamps:      1,205 (  7.9%)
[?]  Ambiguous:             2,103 ( 13.8%)
[x]  Likely IDs:               892 (  5.9%)
[-] Confirmed IDs:           7,613 ( 49.9%)
[!] Invalid:                    0 (  0.0%)
──────────────────────────────────────────────────
Kept (balanced mode): 4,626/15,234 (30.4%)

[[~]] Exported:
   • /path/to/Carved/facebook_messages_20250116_143022/facebook_messages_Carved_Recovered.sqlite
   • /path/to/Carved/facebook_messages_20250116_143022/facebook_messages_carved_all.csv
   • /path/to/Carved/facebook_messages_20250116_143022/facebook_messages_carved_protobufs.jsonl

[[~]] Done.
```

## Key Improvements Over V1

### V1 (Confidence Scores)

- [-] Fuzzy scores (0.5-0.7) weren't actionable
- [-] Hard to understand why a score was given
- [-] No URL awareness
- [-] No Snowflake ID detection

### V2 (Categorical Classification)

- [+] Clear labels: "CONFIRMED_ID" vs "CONFIRMED_TIMESTAMP"
- [+] Transparent reasoning in `ts_reason` column
- [+] URL-aware (uses Unfurl)
- [+] Detects Snowflake IDs structurally
- [+] Integrates time_decode (170+ formats)
- [+] Easy to filter: SQL or `--filter-mode`

## Dependencies

```bash
pip install dfir-unfurl time-decode
# Or with uv:
uv pip install dfir-unfurl time-decode
```

## Testing

Test on a known database:

```bash
# Facebook database (heavy ID pollution expected)
python carve_sqlite.py facebook.db --filter-mode balanced

# Check distribution
sqlite3 Carved/facebook_*/facebook_Carved_Recovered.sqlite \
  "SELECT ts_classification, COUNT(*) FROM carved_all
   WHERE kind='ts' GROUP BY ts_classification;"
```

Expected: 40-60% CONFIRMED_ID (Snowflake IDs), 20-30% CONFIRMED_TIMESTAMP

## Troubleshooting

### "Warning: V2 classifier not available"

- Unfurl or time_decode not installed
- Falls back to simple classifier
- Run: `uv pip install dfir-unfurl time-decode`

### Too many/few timestamps kept

- Adjust `--filter-mode`:
  - Too many false positives → use `strict`
  - Missing real timestamps → use `permissive`

### Want to see what was filtered

```sql
-- See all candidates, even filtered ones
SELECT * FROM carved_all WHERE kind = 'ts' ORDER BY ts_classification;

-- Or run with --filter-mode all to keep everything
```

## Future Enhancements

- [ ] Add more platform patterns to `url_analyzer.py`
- [ ] Machine learning classifier (optional)
- [ ] Fuzzy field name matching
- [ ] Custom pattern configuration file
- [ ] Integration with timeline tools
