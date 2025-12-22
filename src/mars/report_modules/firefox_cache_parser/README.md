# Firefox Cache Parser

This module parses Firefox's cache2 directory structure to extract HTTP artifacts including URLs,
response headers, and cached content.

## Disclaimer

**This report is for informational and investigative purposes only.** The data presented should be
independently verified before being relied upon for any legal, regulatory, or evidentiary purpose.

### Limitations

- **Cache eviction**: Firefox automatically evicts old cache entries. The cache represents a partial
snapshot of browsing activity, not a complete history.
- **Content-Type detection**: When HTTP headers are unavailable or incomplete, file type detection
falls back to magic byte signatures. This heuristic may misidentify some content types.
- **Compressed content**: Gzip-compressed responses are automatically decompressed. Truncated or
corrupted compressed data may fail to decompress fully.
- **Private browsing**: Content accessed in private/incognito mode is not written to the disk cache.
- **Cache integrity**: Individual cache entries may be corrupted, truncated, or partially
overwritten. SHA256 hashes verify extracted content integrity.
- **Timestamp accuracy**: Cache metadata timestamps reflect when content was cached, not necessarily when it was first accessed.

## What Is Firefox Cache2?

Firefox stores HTTP cache data in a binary format under the `cache2/` directory within each profile.
Each cached resource is stored as a separate file with a SHA1-based filename in the `entries/` subdirectory.

Cache entries contain:

- The cached HTTP response body
- HTTP response headers (status, content-type, cache-control, etc.)
- Request metadata (URL, method, security info)
- Cache management data (frecency scores, expiration times)

## Parsed Data

### Cache Entry Metadata

For each cache entry, the parser extracts:

- **URL**: The original request URL
- **HTTP headers**: Response status, content-type, content-length, encoding, date, expires, ETag, cache-control
- **Security info**: Base64-encoded TLS/SSL certificate data
- **Cache metadata**: Entry size, frecency score, fetch count, last fetch/modification times

### Index and Journal Files

When available, the parser also processes:

- **index file**: Master cache index with entry metadata and file size information
- **index.log (journal)**: Recent cache operations log

### Body Extraction

Optionally, cached content bodies can be exported:

- **Copy mode**: Extracts and decompresses content to a separate directory
- **Symlink mode**: Creates symlinks to original cache files (preserves disk space)

File extensions are determined by content-type headers or magic byte detection.

## Data Sources

| Artifact | Path Pattern | Firefox Versions |
| ---------- | -------------- | ------------------ |
| Cache entries | `~/Library/Caches/Firefox/Profiles/*/cache2/entries/*` | 32+ |
| Cache index | `~/Library/Caches/Firefox/Profiles/*/cache2/index` | 32+ |
| Cache journal | `~/Library/Caches/Firefox/Profiles/*/cache2/index.log` | 32+ |

Note: Firefox 32 (2014) introduced the cache2 format, replacing the older cache v1 format.

## Not Currently Parsed

- **Cache v1 format**: Legacy Firefox cache format (pre-Firefox 32)
- **Memory cache**: In-memory cached content is not persisted to disk
- **Service worker caches**: Stored separately from the HTTP cache
- **Encrypted content**: HTTPS response bodies are stored encrypted in some configurations

## Output Files

- `firefox_cache_entries.csv` - All parsed cache entries with 50+ columns including:
  - `entry_name` - SHA1-based cache filename
  - `source_file` - Full path to cache entry file
  - `url` - Original request URL
  - `request_method` - HTTP method (GET, POST, etc.)
  - `status_line` - HTTP response status
  - `content_type`, `content_length`, `content_encoding`
  - `response_date`, `expires`, `etag`, `last_modified`
  - `cache_control`, `pragma`
  - `security_info_b64` - TLS certificate data
  - `metadata_size`, `body_size`, `decompressed_size`
  - `body_sha256` - Content hash for integrity verification
  - Index metadata (when available): `frecency`, `flags`, `last_fetched`, `last_modified_idx`, `file_size_kb`
  - `dump_body_path` - Path to extracted body (if `--dump-bodies` used)

- `firefox_cache_bodies/` (optional) - Directory containing extracted/decompressed cache bodies with appropriate file extensions

## Command-Line Options

| Option | Description |
| -------- | ------------- |
| `--index-file` | Path to cache index file |
| `--journal-file` | Path to index.log journal file |
| `--dump-bodies` | Extract cached content to output directory |
| `--dump-mode` | `copy` (default) or `link` for body extraction |

## Magic Byte Detection

When content-type headers are missing, the parser detects file types by magic bytes:

- Images: JPEG, PNG, GIF, WebP, ICO, BMP, TIFF, AVIF
- Documents: PDF, HTML
- Archives: ZIP, GZIP
- Media: MP4, WebM, MP3, OGG, WAV, FLAC
- Fonts: WOFF, WOFF2, TTF, OTF
- Data: JSON, XML, JavaScript, CSS, WASM

## Scan Type

**Exemplar only** - This module runs during live system scans to capture current Firefox cache state.
