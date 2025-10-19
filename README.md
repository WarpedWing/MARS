# MacLogSleuth

Parses carved SQLite and Gzip files from MacOS, extracts logs from corrupt compressed files, and will reassemble log data from corrupt database files.

Currently supports deep recovery of Powerlog files, with general recovery of FSEvents, WiFi, System, Chrome, cfurl, Firefox, and other uncategorized SQLite files.

Coming soon.

## MacOS Powerlog locations

- `/private/var/db/powerlog/Library/BatteryLife/CurrentPowerlog.PLSQL`
  - (& `-shm` & `-wal`)
- `/private/var/db/powerlog/Library/BatteryLife/Archives/*.gz` (gzipped logs)

## License

MacLogSleuth is licensed under the MIT License. See [LICENSE.md](LICENSE.md) for details.

### Third-Party Licenses

This project uses or references the following third-party software:

- **zlib** - Compression library (zlib License) - [src/resources/docs/zlib-LICENSE](src/resources/docs/zlib-LICENSE)
- **gzrecover** - Corrupted gzip file recovery (GPL v2) - [src/resources/docs/gzrecover-LICENSE-GPLv2.txt](src/resources/docs/gzrecover-LICENSE-GPLv2.txt)
- **bzip2** - Compression library (BSD-style License) - [src/resources/docs/bzip2-LICENSE](src/resources/docs/bzip2-LICENSE)
- **libewf** - Expert Witness Compression Format library (LGPL v3) - [src/resources/docs/libewf-COPYING](src/resources/docs/libewf-COPYING) and [src/resources/docs/libewf-COPYING.LESSER](src/resources/docs/libewf-COPYING.LESSER)
- **SQLite Dissect** - SQLite parser by DC3 (DC3 Open Source License) - [src/resources/docs/sqlite-dissect-LICENSE.txt](src/resources/docs/sqlite-dissect-LICENSE.txt)

For detailed license information and documentation, see the [src/resources/docs](src/resources/docs) directory.
