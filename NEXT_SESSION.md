# Mac Log Sleuth - Next Session Plan

## Current Status - KB Laptop Case

### Completed Today

- Built complete E01 forensic workflow (FUSE-free!)
- Scanned 750GB forensic image (4 partitions)
- Found 3,317 database files on macOS partition
- **Successfully extracted 190 forensic databases**

### Key Databases Extracted

- Safari History.db + WAL/SHM files
- Chrome History + Journal
- Firefox places.sqlite (5MB of browsing history)
- Messages chat.db + WAL/SHM (iMessage/SMS)
- 186 other databases

Location: `./kb_extracted_databases/`

### Disk Space Usage

- Raw DD file: ~750GB at `/Volumes/Crux/Kelley_Brannon_Case/KB_Laptop_Image/DD/KBLaptopRAW.dd`
- Extracted databases: ~50MB
- **To reclaim space**: Run `./CLEANUP_E01.sh` to delete DD (keeps E01 original)

---

## Next Steps for Morning Session

### 1. Generate Schemas/Rubrics from Extracted Databases

Run the exemplar scanner on extracted databases:

```bash
# Scan all extracted databases
uv run src/mac_log_sleuth/pipeline/exemplar_scanner.py \
    --source ./kb_extracted_databases \
    --case-name "KB_Laptop_Forensic_Analysis" \
    --output-dir ./kb_analysis_output
```

This will create:

- Schema CSV files (table/column definitions)
- Rubric JSON files (metadata, types, examples, roles)
- Organized output with provenance tracking

### 2. Analyze Key Databases

**Safari History:**

```bash
# View schema
cat kb_analysis_output/schemas/History/History.schema.csv

# Check rubric for timestamp detection
jq '.tables.history_visits' kb_analysis_output/schemas/History/History.rubric.json
```

**Messages (iMessage/SMS):**

```bash
# Explore chat.db schema
cat kb_analysis_output/schemas/chat/chat.schema.csv

# Look for dual-nature fields (notif_id, etc)
jq '.tables.message.columns' kb_analysis_output/schemas/chat/chat.rubric.json
```

**Firefox Browsing History:**

```bash
# Check places.sqlite
cat kb_analysis_output/schemas/places/places.schema.csv
```

### 3. Run the Carver on Damaged Databases

If any databases are corrupted:

```bash
# Carve timestamps and data
PYTHONPATH=src uv run python3 -m mac_log_sleuth.carver.carve_sqlite \
    --db ./kb_extracted_databases/Users_admin_Library_Safari_History.db \
    --output ./carved_safari_history.sqlite
```

### 4. Scan Windows Partition (Boot Camp)

The KB Laptop has a 250GB Windows partition (partition 4):

```bash
# Scan Windows partition for IE/Edge/Chrome databases
uv run src/mac_log_sleuth/pipeline/multi_partition_scanner.py \
    --image /Volumes/Crux/Kelley_Brannon_Case/KB_Laptop_Image/DD/KBLaptopRAW.dd \
    --output-dir ./kb_windows_scan \
    --partition 4
```

Then extract Windows databases (Chrome, Edge, IE history, Registry hives, etc.)

---

## Tools Built Today

### 1. `e01_workflow.py` - E01 Handler

- Convert E01 → DD: `--convert`
- Get E01 info: `--info`
- Diagnose DD structure: `--diagnose`
- Mount DD: `--mount-dd`

### 2. `multi_partition_scanner.py` - Partition Scanner

- Lists all database files using Sleuth Kit
- Works on HFS+, APFS, NTFS, FAT32
- No mounting required

### 3. `extract_databases.py` - Database Extractor

- Extracts files by inode using `icat`
- Pattern-based filtering
- MD5 hashing for integrity
- Provenance tracking

### 4. `exemplar_scanner.py` - Schema/Rubric Generator

- Scans databases and generates schemas
- Creates rubrics with metadata
- Detects timestamp fields
- Tracks dual-nature fields (IDs that are also timestamps)

### 5. Database Catalog

- 50+ macOS/Windows databases cataloged
- Safari, Chrome, Firefox, Edge
- Messages, Mail, TCC, Knowledge Store
- Ready for automated detection

---

## Dependencies

### Python (via uv)

- pyyaml>=6.0
- bbpb>=1.4.2 (BlackBoxProtobuf)
- plotly>=6.3.1
- time-decode>=10.1.0
- dfir-unfurl>=20250810

### External Tools (via Homebrew)

- sleuthkit (fls, icat, mmls, fsstat) - `brew install sleuthkit`
- libewf (ewfexport, ewfinfo) - `brew install libewf`
- hdiutil (macOS built-in)

---

## Quick Reference Commands

### Free up disk space

```bash
./CLEANUP_E01.sh
```

### List extracted databases

```bash
ls -lh kb_extracted_databases/ | head -20
```

### Check extraction report

```bash
jq '.successfully_extracted' kb_extracted_databases/extraction_report.json
```

### Generate schemas for all databases

```bash
uv run src/mac_log_sleuth/pipeline/exemplar_scanner.py \
    --source ./kb_extracted_databases \
    --case-name "KB_Laptop"
```

---

## Notes

- **E01 integrity preserved**: Original E01 file untouched
- **DD file**: Can be safely deleted after extraction (750GB reclaimed)
- **Extracted databases**: Keep these - only ~50MB total
- **FUSE not needed**: Our workflow works without FUSE mounting
- **Ready for analysis**: 190 databases ready to process

Have a great rest! Looking forward to analyzing these databases in the morning!
