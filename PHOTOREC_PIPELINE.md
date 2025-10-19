# PhotoRec Forensic Pipeline for Mac Log Sleuth

## Overview

Complete forensic pipeline for processing PhotoRec carved output from macOS disk images. Handles classification, decompression, recovery, and chronological merging of fragmented artifacts.

## Status: Phase 1-3 Complete ✅

**Tested on:** 377,000 PhotoRec files from KB laptop case
**Result:** 70 WiFi log fragments → 122,100 chronologically sorted lines

---

## Pipeline Architecture

```
PhotoRec Output (377K files)
    ↓
Phase 1: Smart Classification (232 relevant files)
    ├── Text Logs (70 WiFi logs)
    ├── SQLite DBs (0 - to be implemented)
    └── Other Artifacts (115 JSON files)
    ↓
Phase 2: Decompression & Recovery
    ├── Native Python (gzip/bz2)
    ├── Recovery Tools (gzrecover/bzip2recover)
    └── Corrupted Archive Quarantine
    ↓
Phase 3: Chronological Merging
    ├── Timestamp Parsing (year inference)
    ├── Heap-based Streaming Merge
    └── Combined Logs (122K lines)
    ↓
Phase 4: SQLite Combining [TODO]
    ├── Schema Matching
    ├── Fragment Merging
    └── Row Deduplication
```

---

## Implementation Details

### Phase 1: Smart Classification

**File:** `photorec_processor.py`

- **Smart Filtering**: Only scans relevant PhotoRec folders (sqlite, gz, bz2, txt, log, plist, json, jsonlz4, asl)
- **Pattern-Based Detection**: Uses `text_log_fingerprinter.py` for content analysis
- **Confidence Scoring**: Rejects low-confidence matches to avoid false positives

**Supported Formats:**

- WiFi logs (`wifi.log`, `wifi.log.*.bz2`)
- System logs (`system.log`, `system.log.*.gz`)
- Install logs (`install.log`)
- ASL binary files (`*.asl`)
- Plists (XML & binary)
- JSON files
- JSONLZ4 (Firefox compressed JSON)

### Phase 2: Decompression & Recovery

**Multi-Tier Recovery:**

1. **Native Decompression** (gzip/bz2 Python libraries)
2. **Recovery Tools** (gzrecover, bzip2recover for corrupted files)
3. **Corrupted Archive Quarantine** (unsalvageable files)

**Statistics from KB Case:**

- 70 compressed files processed
- 100% success rate (0 failures)
- All files decompressed and re-classified

### Phase 3: Chronological Merging

**Algorithm:**

- Heap-based K-way merge
- Memory-efficient streaming (doesn't load all files into RAM)
- Multi-line log entry support (continuation lines inherit timestamp)

**Timestamp Parsing:**

```python
# WiFi log format
Thu Jul 23 00:48:51.636 <airportd[128]> message

# System/Install log format
Nov 23 17:45:15 hostname process[pid]: message
```

**Year Inference Strategy:**

macOS logs don't include year in timestamps. We infer year from:

1. **File metadata** (modification time)
2. **Most common year** across all fragments
3. **Fallback** to current year

**Known Limitation:** Year ambiguity cannot be fully resolved without external context (e.g., case timeline, exemplar metadata).

**Output:**

- Parser-compatible format (mac_apt, APOLLO)
- No unknown lines or markers
- Pure chronological log data

---

## Usage

### Basic Usage

```bash
python photorec_processor.py \
  --photorec-dir /path/to/PhotoRec_Output \
  --output-dir /path/to/recovered
```

### With Verbosity

```bash
python photorec_processor.py \
  --photorec-dir /Volumes/Crux/KB_Case/PhotoRec \
  --output-dir ./kb_recovered \
  --verbose
```

### Lower Confidence (More Aggressive Recovery)

```bash
python photorec_processor.py \
  --photorec-dir /path/to/PhotoRec_Output \
  --output-dir /path/to/recovered \
  --min-confidence 0.5
```

---

## Output Structure

```
output_dir/
├── combined_logs/
│   ├── combined_wifi_log.log          # Merged WiFi logs
│   ├── combined_system_log.log        # Merged system logs
│   └── combined_install_log.log       # Merged install logs
│
├── decompressed/
│   ├── f34064264                      # Decompressed fragments
│   └── ...
│
├── corrupted_archives/
│   └── (files that couldn't be recovered)
│
├── photorec_processing_report.json   # Complete audit trail
│
└── _temp/                             # Working directory
```

---

## Processing Report

JSON report includes:

```json
{
  "generated_at": "2025-10-17T...",
  "photorec_dir": "/path/to/photorec",
  "statistics": {
    "total_files": 232,
    "classified": 185,
    "unknown": 47,
    "text_logs": 70,
    "sqlite_dbs": 0,
    "other": 115
  },
  "text_logs": {
    "wifi_log": [
      {
        "source_path": "...",
        "file_type": "wifi_log",
        "confidence": 1.0,
        "first_timestamp": "Oct 6 00:49:53.207",
        "last_timestamp": "Oct 6 02:06:13.931",
        "decompressed_path": "...",
        "md5": "e849d6a41cbbb59cc8041bf8585c06e4"
      }
    ]
  },
  "errors": []
}
```

---

## Phase 4: SQLite Combining [TODO]

### Planned Features

1. **Schema Fingerprinting**
   - Use existing rubrics from `database_catalog.yaml`
   - Match fragments by table structure
   - Identify databases (Safari History, chat.db, etc.)

2. **Multi-Tier Recovery**
   - **Tier 1**: Intact databases (direct ATTACH and merge)
   - **Tier 2**: Corrupted databases (`.recover` command)
   - **Tier 3**: Severely damaged (sqlite_dissect)
   - **Tier 4**: Page-level carving (send to carve_sqlite.py)

3. **Row Deduplication**
   - By primary key
   - By timestamp + unique fields
   - Preserve provenance (source file tracking)

4. **WAL/SHM Handling**
   - Auto-detect WAL/SHM files
   - Merge into main database
   - Extract deleted rows from freelist

### Implementation Strategy

```python
class SQLiteCombiner:
    def combine_databases(self, schema_name, db_fragments):
        # 1. Verify schema match
        # 2. Create combined database
        # 3. ATTACH each fragment
        # 4. Merge data with deduplication
        # 5. Extract deleted rows
        # 6. Generate provenance report
```

---

## Known Limitations

### Year Ambiguity (macOS Logs)

macOS system logs don't include year in timestamps. Our inference strategy:

- Uses file modification time as primary hint
- Falls back to most common year across fragments
- Cannot definitively resolve year without external context

**Impact:** Logs from different years may be interleaved if file metadata is lost.

**Mitigation:**

- Use case timeline to manually set year
- Cross-reference with databases that have full timestamps
- Document year inference in processing report

### PhotoRec File Organization

PhotoRec output can be organized in two ways:

1. **Type folders** (sqlite/, gz/, bz2/) - Current implementation
2. **recup_dir.* folders** (numbered recovery directories) - Not yet supported

**TODO:** Add support for both organizational schemes.

---

## Dependencies

### Python Libraries

```bash
pip install pyyaml lz4
```

### External Tools (Optional)

```bash
# For corrupted archive recovery
brew install gzrecover  # macOS
apt-get install gzrt    # Linux (gzrecover)

# bzip2recover (usually included with bzip2)
which bzip2recover
```

### For SQLite Recovery (Phase 4)

```bash
# sqlite3 with .recover command (built-in on macOS/most Linux)
sqlite3 --version  # Should be 3.32+

# sqlite_dissect (optional, for advanced recovery)
# TODO: Add installation instructions
```

---

## Performance

### KB Case Statistics

- **Input:** 377,000 PhotoRec files
- **Filtered to:** 232 relevant files
- **Processing time:** ~20 seconds (Phases 1-3)
- **Output:** 9.3MB combined WiFi log (122,100 lines)
- **Memory usage:** Peak ~50MB (streaming merge)

### Scalability

- **Large files:** Automatic streaming for files > 100MB
- **Many fragments:** Heap-based merge handles hundreds of files
- **Compressed data:** Progressive decompression

---

## Future Enhancements

### Short Term

- [ ] Phase 4: SQLite database combining
- [ ] Firefox JSONLZ4 decompression (using lz4 library)
- [ ] Plist parsing and extraction
- [ ] Support for recup_dir.* organization scheme

### Medium Term

- [ ] GUI for year override per log type
- [ ] Advanced year inference (analyze log content for clues)
- [ ] Timeline visualization
- [ ] Integration with mac_apt/APOLLO

### Long Term

- [ ] Machine learning for artifact classification
- [ ] Automated timeline correlation
- [ ] Cloud-based processing for large cases

---

## References

- **Text Log Fingerprinter:** `src/mac_log_sleuth/text_log_fingerprinter.py`
- **PhotoRec Processor:** `src/mac_log_sleuth/pipeline/photorec_processor.py`
- **Database Catalog:** `src/mac_log_sleuth/catalog/database_catalog.yaml`
- **Firefox JSONLZ4:** [GitHub - Firefox-File-Utilities](https://github.com/jscher2000/Firefox-File-Utilities)

---

## Contributing

When adding new log types:

1. Add detection patterns to `text_log_fingerprinter.py`
2. Add timestamp parsing to `TimestampParser` class
3. Update `database_catalog.yaml` with metadata
4. Add tests with real forensic data

---

## License

Part of Mac Log Sleuth by WarpedWing Labs
