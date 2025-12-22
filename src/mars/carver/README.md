# Carver Module

Deep data carving and extraction for SQLite databases and binary artifacts.

## Purpose

Extract forensic artifacts from:

- Corrupt SQLite databases
- Database pages outside proper structure
- Binary blobs (protobuf, plist, JSON)
- Timestamp data in various formats
- URLs and identifiers

## Module Components

### Main SQLite Carver

`carve_sqlite.py`

**Extracts data from SQLite database pages**

**Extraction Targets:**

- Timestamps (multiple formats)
- URLs and web artifacts
- Text strings
- Protobuf blobs
- JSON structures
- Binary data patterns

**Features:**

- Parallel page processing
- Multiple timestamp classifiers
- URL analysis with Unfurl
- Protobuf decoding
- CSV/JSONL/DB output formats

## Submodules

### Protobuf Handling

`protobuf/`

**Decode and analyze protobuf binary data**

- `decoder.py` - Decode protobuf blobs using blackboxprotobuf
- `extractor.py` - Extract protobuf data from database pages
- `timestamp_extractor.py` - Extract timestamps from protobuf

### Timestamp Detection

`timestamp/`

**Classify and extract timestamp data**

- `types.py` - Classification types and data structures
- `classifier.py` - Timestamp classification implementation
- `patterns.py` - Timestamp pattern definitions

**Supported Formats:**

- Unix timestamps (seconds, milliseconds, nanoseconds)
- Cocoa timestamps (macOS/iOS)
- Windows FILETIME
- Chrome timestamps
- WebKit timestamps
- ISO 8601 strings
- RFC 3339 strings

### URL Analysis

`unfurl/`

**Extract and analyze URLs and identifiers**

- `analyzer.py` - URL parsing and timestamp extraction
- `id_detectors.py` - Detect IDs in URLs (UUIDs, timestamps, etc.)

**Features:**

- Extract timestamps from URLs
- Detect UUID patterns
- Identify session IDs
- Parse query parameters
- Analyze URL structure

---

## How Carving Works

### 1. Database Page Iteration

```python
# carve_sqlite.py main loop
for page_num in range(total_pages):
    page_data = read_page(db_file, page_num, page_size)
    results = process_page(page_data, page_num)
```

### 2. Data Extraction Per Page

For each database page:

**1. Timestamp Detection**

- Scan for numeric values
- Classify as timestamp types
- Filter by confidence threshold
- Deduplicate close timestamps

**2. URL Extraction**

- Regex patterns for URLs
- HTTP/HTTPS schemes
- Parse with Unfurl
- Extract embedded timestamps/IDs

**3. Text Extraction**

- Printable ASCII strings
- Minimum length filtering
- Context preservation

**4. Binary Blob Extraction**

- Protobuf detection
- Decode with blackboxprotobuf
- Extract nested timestamps

### 3. Output Generation

Results written to:

- **CSV**: Timestamp inventory (optional)
- **JSONL**: Detailed extraction data
- **Database**: Structured SQLite output

### Basic Database Carving

**Output:**

```text
Carved/
  corrupt_timestamps.csv      # All timestamps found
  corrupt_carved.jsonl        # Detailed extraction data
  corrupt_carved.db           # Structured database
```
