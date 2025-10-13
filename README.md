# MacLogSleuth

Parses carved SQLite and Gzip files from MacOS, extracts logs from corrupt compressed files, and will reassemble log data from corrupt database files.

Currently supports deep recovery of Powerlog files, with general recovery of FSEvents, WiFi, System, Chrome, cfurl, Firefox, and other uncategorized SQLite files.

Coming soon.

## MacOS Powerlog locations

- `/private/var/db/powerlog/Library/BatteryLife/CurrentPowerlog.PLSQL`
  - (& `-shm` & `-wal`)
- `/private/var/db/powerlog/Library/BatteryLife/Archives/*.gz` (gzipped logs)
