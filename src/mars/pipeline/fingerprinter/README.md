# Fingerprinter Module

File fingerprinting for accurate type detection.

## Purpose

Analyze files to determine their type and structure **without** relying solely
on file extensions or metadata. Essential for processing carved files where
metadata is often lost.

## Module Components

### [text_fingerprinter.py](text_fingerprinter.py) - Text Log Detection

**Identifies log types from text content**

```python
from mars.pipeline.fingerprinter.text_fingerprinter import (
    LogType,
    identify_log_type,
)

# Identify log type
result = identify_log_type(
    file_path=Path("./unknown.log"),
    min_confidence=0.6
)

print(f"Type: {result.log_type}")
print(f"Confidence: {result.confidence}")
print(f"Reasons: {result.reasons}")
print(f"First timestamp: {result.first_timestamp}")
```

**Supported Log Types:**

- `WIFI_LOG` - WiFi connection logs
- `SYSTEM_LOG` - macOS system logs
- `INSTALL_LOG` - Software installation logs
- `SQLITE` - SQLite database files
- `JSONLZ4` - Firefox JSONLZ4 compressed files
- `JSON` - JSON files
- `PLIST` - Property list files
- `ASL` - Apple System Log files

**Detection Methods:**

- Magic byte patterns (SQLite, JSONLZ4, etc.)
- Content pattern matching (timestamps, log prefixes)
- Structure analysis (JSON, plist format detection)
- Statistical heuristics for text logs

**Returns:**

- `LogType` enum
- Confidence score (0.0-1.0)
- Detection reasons (for debugging)
- First/last timestamps (if found)

---

## How Fingerprinting Works

### Text Fingerprinting Process

1. **Quick Checks** (fast rejection)
   - File size < 50 bytes → UNKNOWN
   - Binary magic bytes → classify as binary type

2. **Magic Byte Detection**
   - JSONLZ4: `b"mozLz40\x00"`
   - ASL: `b"ASL DB\x00"`

3. **Content Analysis**
   - Read first 1000-10000 lines
   - Look for timestamp patterns
   - Match against known log prefixes
   - Calculate pattern frequency

4. **Scoring**
   - Each detection method adds to confidence
   - Multiple matching patterns → higher confidence
   - Return best match above threshold

## Usage Examples

### Fingerprint Unknown Files

```python
from pathlib import Path
from mars.pipeline.fingerprinter.text_fingerprinter import identify_log_type

file_path = Path("./unknown_file")

# Try text fingerprinting
result = identify_log_type(file_path, min_confidence=0.6)
if result.log_type != LogType.UNKNOWN:
    print(f"Identified as: {result.log_type}")
    print(f"Confidence: {result.confidence:.2%}")
```

## Performance Notes

- **Text fingerprinting**: O(n) where n = number of lines sampled (max 10000)
- **Magic byte checks**: O(1) - very fast, always run first
