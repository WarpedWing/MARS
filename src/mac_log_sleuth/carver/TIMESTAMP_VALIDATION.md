# Timestamp Validation System

## Overview

The SQLite Carver now includes an intelligent timestamp validation system that distinguishes **real timestamps** from **timestamp-like IDs** (Snowflake IDs, Facebook event IDs, etc.).

## The Problem

Modern distributed systems often generate IDs that look like timestamps:

- **Twitter/X Snowflake IDs**: 64-bit integers with embedded millisecond timestamp
- **Facebook/Instagram IDs**: Often encode creation time + shard information
- **MongoDB ObjectIDs**: First 4 bytes contain Unix timestamp
- **Discord Snowflake IDs**: Similar to Twitter's format
- **UUID v1**: Contains timestamp component

These IDs pass basic timestamp validation (correct digit count, valid date range) but **aren't actually time values**.

## Solution: Confidence Scoring

Every detected timestamp receives a **confidence score** (0.0 to 1.0):

- **1.0** = Definitely a timestamp
- **0.8-0.9** = Very likely a timestamp (HIGH confidence)
- **0.6-0.7** = Probably a timestamp (MEDIUM confidence)
- **0.4-0.5** = Uncertain (LOW confidence)
- **0.0-0.3** = Probably an ID (VERY LOW confidence)

## Validation Factors

### 1. Field Name Detection (+0.35 / -0.40)

**Strongest signal**. Extracts field names from context:

```bash
✓ created_at: 1609459200      → +0.35 (timestamp keyword)
✗ event_id: 1234567890123456  → -0.40 (ID keyword)
```

Timestamp keywords: `time`, `date`, `created`, `modified`, `updated`, `expire`, `last`, `timestamp`, `accessed`

ID keywords: `id`, `uid`, `event_id`, `notification_id`, `user_id`, `msg_id`, `token`, `key`, `snowflake`

### 2. Context Keywords (+0.20 / -0.25)

**Medium signal**. Checks surrounding text (±50 bytes):

```bash
✓ "last modified time" 1609459200     → +0.20
✗ "user id value" 1234567890123456    → -0.25
```

### 3. Temporal Clustering (+0.15 / +0.05)

**Medium signal**. Real timestamps often appear in pairs/groups:

```bash
✓ created_at: 1609459200
  updated_at: 1609459300              → +0.15 (2+ nearby timestamps)

✓ Single timestamp with one nearby   → +0.05
```

### 4. Sequential Pattern Detection (-0.30 batch penalty)

**Medium signal**. IDs increment sequentially, timestamps scatter:

```bash
✗ 1000000001, 1000000002, 1000000003  → -0.30 (sequential = IDs)
✓ 1609459200, 1612137600, 1614556800  → No penalty (scattered)
```

### 5. Value Pattern Analysis (-0.20)

**Weak-medium signal**. Suspicious patterns suggest IDs:

```bash
✗ 10000000123456789  → -0.20 (starts with 10000...)
✗ 11111111111111111  → -0.20 (too few unique digits)
```

### 6. Binary Alignment (+0.05 / +0.02)

**Weak signal**. Real timestamps are often 4 or 8-byte aligned in binary formats:

```bash
✓ Offset 0x0100 (8-byte aligned)  → +0.05
✓ Offset 0x0104 (4-byte aligned)  → +0.02
✗ Offset 0x0103 (unaligned)       → No bonus
```

## Usage

### Basic Usage (Default: min-confidence 0.5)

```bash
python carve_sqlite.py database.sqlite
```

This filters out low-confidence timestamps (< 0.5), removing most IDs while keeping real timestamps.

### Strict Mode (High Precision)

```bash
python carve_sqlite.py database.sqlite --min-confidence 0.8
```

Only keeps HIGH confidence timestamps. Use when:

- You want **very clean** results
- You're okay missing some edge cases
- You're analyzing databases with lots of ID pollution

### Permissive Mode (High Recall)

```bash
python carve_sqlite.py database.sqlite --min-confidence 0.2
```

Includes more uncertain values. Use when:

- You want to see **everything** for manual review
- You're doing exploratory analysis
- You can filter results later in your workflow

### See All Candidates (No Filtering)

```bash
python carve_sqlite.py database.sqlite --min-confidence 0.0
```

Includes all detected numeric values, even obvious IDs. Useful for debugging or understanding what the carver is finding.

## Output

### CSV Columns

The `carved_all.csv` now includes:

| Column | Description |
|--------|-------------|
| `page_no` | Page number in database |
| `page_offset` | Byte offset within page |
| `abs_offset` | Absolute byte offset in file |
| `cluster_id` | Page cluster ID (or NULL) |
| `kind` | Artifact type: `ts`, `url`, `text` |
| `value_text` | Text content (URLs, text runs) |
| `value_num` | Numeric value (timestamps) |
| `ts_kind_guess` | Timestamp format (`unix_sec`, `unix_milli`, etc.) |
| `ts_human` | Human-readable timestamp |
| **`ts_confidence`** | **Confidence score 0.0-1.0** |

### SQLite Database

The `carved_all` table includes the same `ts_confidence` column for querying:

```sql
-- Get only high-confidence timestamps
SELECT * FROM carved_all
WHERE kind = 'ts' AND ts_confidence >= 0.8
ORDER BY ts_confidence DESC;

-- Distribution of confidence scores
SELECT
    ROUND(ts_confidence, 1) as confidence_bucket,
    COUNT(*) as count
FROM carved_all
WHERE kind = 'ts'
GROUP BY confidence_bucket
ORDER BY confidence_bucket DESC;

-- Find suspicious IDs that were flagged
SELECT * FROM carved_all
WHERE kind = 'ts' AND ts_confidence < 0.3
LIMIT 100;
```

## Examples

### Facebook Database (High ID Pollution)

```bash
# Default filtering removes most event/notification IDs
python carve_sqlite.py facebook_messages.db

# Strict mode for cleanest results
python carve_sqlite.py facebook_messages.db --min-confidence 0.7
```

### macOS System Database (Clean Timestamps)

```bash
# Default works well, or be more permissive
python carve_sqlite.py system.db --min-confidence 0.4
```

### Unknown Database (Exploratory)

```bash
# Start permissive to see what's there
python carve_sqlite.py unknown.db --min-confidence 0.0

# Then review CSV and adjust threshold based on confidence distribution
python carve_sqlite.py unknown.db --min-confidence 0.6
```

## Tuning Recommendations

### By Confidence Distribution

After running with `--min-confidence 0.0`, check the distribution:

```sql
SELECT
    CASE
        WHEN ts_confidence >= 0.8 THEN 'HIGH (0.8-1.0)'
        WHEN ts_confidence >= 0.6 THEN 'MEDIUM (0.6-0.8)'
        WHEN ts_confidence >= 0.4 THEN 'LOW (0.4-0.6)'
        ELSE 'VERY_LOW (0.0-0.4)'
    END as category,
    COUNT(*) as count,
    ROUND(AVG(ts_confidence), 2) as avg_confidence
FROM carved_all
WHERE kind = 'ts'
GROUP BY category;
```

**Typical Results:**

- **Many in VERY_LOW** → Database has heavy ID pollution, use 0.5-0.7 threshold
- **Mostly in MEDIUM/HIGH** → Clean database, can use 0.3-0.4 threshold
- **Bimodal distribution** → Clear separation between IDs and timestamps, 0.5 is ideal

### By Application Type

| Application | Recommended Threshold | Why |
|-------------|----------------------|-----|
| Social Media Apps | 0.6-0.7 | Heavy ID usage (Snowflake, etc.) |
| System Databases | 0.4-0.5 | Cleaner timestamp usage |
| Browser History | 0.5-0.6 | Mix of IDs and timestamps |
| Chat/Messaging | 0.6-0.8 | Message IDs often look like timestamps |
| File Metadata | 0.3-0.4 | Usually clean timestamp fields |

## Architecture

### Modules

```bash
carver/
├── carve_sqlite.py           # Main carver (v3.4)
├── timestamp_validator.py    # Confidence scoring engine
├── timestamp_patterns.py     # Timestamp detection & interpretation
└── protobuf_extractor.py     # Protobuf parsing (existing)
```

### Flow

```bash
1. find_timestamps_with_interpretation()
   └─> Detects potential timestamps (10-19 digit numbers)

2. validate_timestamp_batch()
   ├─> Analyzes context for each candidate
   ├─> Checks field names, keywords, clustering
   ├─> Detects sequential patterns across batch
   └─> Returns TimestampCandidate objects with confidence scores

3. Filter by --min-confidence threshold
   └─> Only keep candidates above threshold

4. Write to database & CSV with confidence scores
```

## False Positives vs False Negatives

### High Threshold (0.7-0.9)

**Fewer False Positives** (fewer IDs slip through)
**More False Negatives** (might miss valid timestamps)

Use when: You need clean results for timeline construction

### Low Threshold (0.2-0.4)

**More False Positives** (some IDs included)
**Fewer False Negatives** (catches more real timestamps)

Use when: Doing exploratory analysis, can manually review

### Default (0.5)

**Balanced** - Good starting point for most use cases

## Known Limitations

1. **Novel ID formats**: New ID generation schemes may not be detected
2. **Context-free binary data**: Without field names, scoring is less accurate
3. **Compressed/encrypted data**: Validation not effective on opaque data
4. **Language dependency**: Keyword detection assumes English field names

## Future Enhancements

Potential improvements:

- [ ] Machine learning classifier trained on known timestamp vs ID patterns
- [ ] Multi-language field name detection
- [ ] Fuzzy clustering of similar values (statistical analysis)
- [ ] Integration with schema detection for structured formats
- [ ] Confidence calibration based on user feedback
