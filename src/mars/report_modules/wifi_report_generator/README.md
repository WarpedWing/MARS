# Wi-Fi / Network Report Generator

This module parses macOS Wi-Fi and network artifacts to generate an HTML report summarizing network activity.

## Disclaimer

**This report is for informational and investigative purposes only.** The data presented should be
independently verified before being relied upon for any legal, regulatory, or evidentiary purpose.

### Limitations

- **Timestamp accuracy**: Wi-Fi log timestamps lack year information. The module infers the year by
correlating with DHCP lease timestamps when available. If no DHCP data exists, it falls back to the
current/previous year. The inferred year should be validated when possible.
- **Channel/band ambiguity**: WiFi 6E (6GHz) channel numbers overlap with 2.4GHz and 5GHz ranges.
Without frequency data, channels 1-196 are attributed to legacy bands; only channels 197-233 are
unambiguously identified as 6GHz. When frequency data is available, it provides unambiguous band identification.
- **Incomplete coverage**: This module parses traditional plist-based artifacts and legacy wifi.log
files. Unified logs (tracev3 format) are not currently parsed, which may omit events on macOS 10.12+.
- **Locally Administered Addresses (LAA)**: macOS uses randomized (locally administered) MAC
addresses in certain contexts. The LAA flag indicates this but does not guarantee the address is or
is not the device's true hardware address.
- **DHCP lease interpretation**: Lease timestamps indicate when the device obtained an IP address,
not necessarily continuous network presence.

## Report Sections

### Known Networks

Displays Wi-Fi networks the device has connected to, extracted from:

- `com.apple.airport.preferences.plist` (macOS 10.15 and earlier)
- `com.apple.wifi.known-networks.plist` (macOS 11 Big Sur and later)

Shows SSID, BSSID, security type, first/last connection times, and auto-join settings.

### BSSID Activity

Aggregates BSSID (access point MAC address) sightings across all parsed sources.
Includes vendor lookup from OUI database, signal strength (RSSI), and channel/band information where available.

### BSSID Cross-References

Cross-references BSSIDs found in multiple sources (logs, plists, DHCP leases) to identify consistent network associations.

### Wi-Fi Log Activity

Events extracted from traditional Wi-Fi log files (`/var/log/wifi.log` and rotated variants).
Note: Modern macOS increasingly uses unified logging, which is not parsed by this module.

### Scan Burst Summary

Groups Wi-Fi scanning events that occurred in rapid succession, which may indicate device wake events,
location changes, or active network searching.

### Daily Presence Rollup

When DHCP lease data is available, summarizes daily network presence by correlating lease timestamps with Wi-Fi events.

### Sightings Around Leases

Shows Wi-Fi events that occurred near DHCP lease acquisition times, helping correlate
network connections with specific access points.

### DHCP Leases

Parses DHCP lease plists from `/var/db/dhcpclient/leases/` showing IP address assignments,
lease durations, and router information.

### Interfaces & Service Mapping

Network interface configuration from `NetworkInterfaces.plist`, showing hardware addresses and interface types.

### Wi-Fi Message Tracer Metrics

Parses `com.apple.wifi.message-tracer.plist` for aggregate Wi-Fi statistics and metrics.

### Network Locations Map

When geographic data is available (from CoreLocation caches or similar), displays an interactive map of network locations.

## Data Sources

| Artifact | Path Pattern | macOS Versions |
| ---------- | -------------- | ---------------- |
| Airport Preferences | `com.apple.airport.preferences.plist` | 10.6 - 10.15 |
| Known Networks (Modern) | `com.apple.wifi.known-networks.plist` | 11.0+ |
| Wi-Fi Message Tracer | `com.apple.wifi.message-tracer.plist` | Various |
| DHCP Leases | `/var/db/dhcpclient/leases/*` | All |
| Network Interfaces | `NetworkInterfaces.plist` | All |
| Network Preferences | `preferences.plist` | All |
| EAPOL Client | `com.apple.eapolclient.plist` | All |
| Wi-Fi Logs | `/var/log/wifi.log*` | Pre-10.12 (legacy) |

## Not Currently Parsed

- **Unified Logs (tracev3)**: Modern macOS stores WiFi events in unified logs under the `com.apple.wifi` subsystem.
Parsing these requires Apple-proprietary format handling.
- **Private MAC Mappings**: `com.apple.wifi-private-mac-networks.plist` (private MAC address associations)
- **Keychain entries**: Wi-Fi passwords and certificates stored in Keychain

## Output Files

- `wifi_summary.json` - Raw parsed data in JSON format (includes file provenance)
- `wifi_events.ndjson` - Normalized event stream (newline-delimited JSON)
- `wifi_presence_daily.csv` - Daily presence rollup
- `wifi_report.html` - Human-readable HTML report

## File Provenance

The `wifi_summary.json` output includes a `file_provenance` array with cryptographic hashes and metadata
for all parsed source files:

```json
{
  "file_provenance": [
    {
      "path": "/path/to/com.apple.wifi.known-networks.plist",
      "sha256": "abc123...",
      "size_bytes": 12345,
      "mtime_iso": "2024-03-15T10:23:45+00:00"
    }
  ]
}
```

This enables verification that source files have not been modified since analysis.
The SHA256 hash can be compared against known-good values or chain-of-custody records.
