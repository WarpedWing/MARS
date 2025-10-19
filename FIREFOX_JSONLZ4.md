# Firefox JSONLZ4 Decompression

## Overview

Firefox uses a custom JSONLZ4 format for various files:

- Session restore: `sessionstore-backups/*.jsonlz4`
- Bookmark backups: `bookmarkbackups/*.jsonlz4`
- Search engines: `search.json.mozlz4`

## Format

- Magic: `mozLz40\x00` (8 bytes)
- Followed by: LZ4-compressed JSON

## Decompression Tools

### Python Implementation

```python
import lz4.block

def decompress_mozlz4(file_path):
    with open(file_path, 'rb') as f:
        magic = f.read(8)
        if magic != b'mozLz40\x00':
            raise ValueError("Not a mozLz4 file")
        
        compressed = f.read()
        decompressed = lz4.block.decompress(compressed)
        return json.loads(decompressed)
```

### External Tools

- **Firefox-File-Utilities**: <https://github.com/jscher2000/Firefox-File-Utilities>
  - Comprehensive toolkit for Firefox file formats
  - Handles JSONLZ4, session store, bookmarks
  - JavaScript-based utilities

### Dependencies

```bash
# Python LZ4
pip install lz4

# Or use bundled Firefox utilities
git clone https://github.com/jscher2000/Firefox-File-Utilities
```

## Integration Points

### PhotoRec Processor

1. Detect JSONLZ4 files (already implemented in text_log_fingerprinter.py)
2. Decompress to JSON
3. Extract forensic artifacts:
   - Session tabs (URLs, timestamps)
   - Bookmark history
   - Form data

### Files to Target

- `recovery.jsonlz4` - Session recovery data
- `previous.jsonlz4` - Previous session
- `bookmarks-YYYY-MM-DD_*.jsonlz4` - Bookmark backups

## TODO

- [ ] Add lz4 Python library to dependencies
- [ ] Implement JSONLZ4 decompressor in text_log_fingerprinter.py
- [ ] Add Firefox artifact extraction to PhotoRec processor
- [ ] Test with real Firefox profile data
