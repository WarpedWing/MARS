#!/usr/bin/env python3

"""
Test script for exemplar_scanner.py

Tests the exemplar scanner against the current macOS system.
Can run in different modes depending on what's available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mac_log_sleuth.pipeline.exemplar_scanner import ExemplarScanner
from mac_log_sleuth.pipeline.output_structure import OutputStructure


def test_local_system():
    """
    Test scanner against the current macOS system's user directories.

    This is the safest test - scans only the current user's Library folder.
    """
    print("=" * 80)
    print("TEST MODE: Local User Library")
    print("=" * 80)
    print("\nThis will scan your user Library folder for common databases.")
    print("No system files will be accessed.\n")

    # Use current user's home directory
    home = Path.home()
    source = home / "Library"

    if not source.exists():
        print(f"Error: {source} does not exist")
        return False

    # Create test output
    output = OutputStructure(
        base_output_dir=Path.cwd() / "test_output",
        case_name="Test_LocalUser"
    )
    output.create()

    print(f"Source: {source}")
    print(f"Output: {output.root}\n")

    # Create scanner
    scanner = ExemplarScanner(
        source_path=source,
        output_structure=output
    )

    # Scan only user-accessible, high-priority databases
    summary = scanner.scan(
        categories=["browser", "communication", "productivity"],
        priorities=["critical", "high"]
    )

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Databases found: {summary['total_found']}")
    print(f"Databases failed: {summary['total_failed']}")
    print(f"Schemas generated: {summary['schemas_generated']}")
    print(f"Rubrics generated: {summary['rubrics_generated']}")
    print(f"\nOutput directory: {output.root}")

    if summary['found_databases']:
        print("\nFound databases:")
        for db in summary['found_databases']:
            status = "[OK]" if db.get('schema_generated') else "[WARNING]"
            print(f"  {status} {db['name']}")
            print(f"     Path: {db['relative_path']}")
            if db.get('schema_path'):
                print(f"     Schema: {Path(db['schema_path']).name}")

    if summary['failed_databases']:
        print("\nFailed databases:")
        for db in summary['failed_databases']:
            print(f"  [ERROR] {db['name']}")
            print(f"     Error: {db['error']}")

    return True


def test_mounted_volume():
    """
    Test scanner against a mounted macOS volume.

    Use this if you have a forensic image or external drive mounted.
    """
    print("=" * 80)
    print("TEST MODE: Mounted Volume")
    print("=" * 80)
    print()

    # List available volumes
    volumes = Path("/Volumes")
    if volumes.exists():
        print("Available volumes:")
        for vol in sorted(volumes.iterdir()):
            if vol.is_dir():
                print(f"  - {vol.name}")

    volume_name = input("\nEnter volume name to scan (or press Enter to cancel): ").strip()

    if not volume_name:
        print("Cancelled.")
        return False

    source = volumes / volume_name

    if not source.exists():
        print(f"Error: Volume not found: {source}")
        return False

    # Create test output
    output = OutputStructure(
        base_output_dir=Path.cwd() / "test_output",
        case_name=f"Test_{volume_name.replace(' ', '_')}"
    )
    output.create()

    print(f"\nSource: {source}")
    print(f"Output: {output.root}\n")

    # Ask for filtering
    print("Filter options:")
    print("  1. All databases (critical + high + medium + low)")
    print("  2. Critical and high priority only")
    print("  3. Browsers only")
    print("  4. System databases only")

    choice = input("\nSelect filter (1-4, default=2): ").strip() or "2"

    if choice == "1":
        priorities = None
        categories = None
    elif choice == "2":
        priorities = ["critical", "high"]
        categories = None
    elif choice == "3":
        priorities = None
        categories = ["browser"]
    elif choice == "4":
        priorities = None
        categories = ["security", "system", "logs"]
    else:
        priorities = ["critical", "high"]
        categories = None

    # Create scanner
    scanner = ExemplarScanner(
        source_path=source,
        output_structure=output
    )

    # Run scan
    print("\nStarting scan...\n")
    summary = scanner.scan(
        categories=categories,
        priorities=priorities
    )

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Databases found: {summary['total_found']}")
    print(f"Databases failed: {summary['total_failed']}")
    print(f"Schemas generated: {summary['schemas_generated']}")
    print(f"Rubrics generated: {summary['rubrics_generated']}")
    print(f"\nOutput directory: {output.root}")

    if summary['found_databases']:
        print(f"\nFound {len(summary['found_databases'])} databases:")
        for db in summary['found_databases'][:10]:  # Show first 10
            status = "[OK]" if db.get('schema_generated') else "[WARNING]"
            print(f"  {status} {db['name']}")
            print(f"     Category: {db['category']}")
            print(f"     Path: {db['relative_path']}")

        if len(summary['found_databases']) > 10:
            print(f"  ... and {len(summary['found_databases']) - 10} more")

    return True


def test_custom_path():
    """
    Test scanner with a custom path (for testing specific databases).
    """
    print("=" * 80)
    print("TEST MODE: Custom Path")
    print("=" * 80)
    print()

    print("Examples:")
    print("  - /Users/yourname/Library/Safari")
    print("  - /Users/yourname/Library/Messages")
    print("  - /Volumes/MacintoshHD/Users/alice/Library")

    custom_path = input("\nEnter path to scan: ").strip()

    if not custom_path:
        print("Cancelled.")
        return False

    source = Path(custom_path)

    if not source.exists():
        print(f"Error: Path not found: {source}")
        return False

    # Create test output
    output = OutputStructure(
        base_output_dir=Path.cwd() / "test_output",
        case_name=f"Test_Custom_{source.name}"
    )
    output.create()

    print(f"\nSource: {source}")
    print(f"Output: {output.root}\n")

    # Custom glob patterns
    print("Enter glob patterns to search (comma-separated):")
    print("Examples: *.db, *.sqlite, **/*.db")

    patterns_input = input("Patterns (default: **/*.db,**/*.sqlite): ").strip()

    if not patterns_input:
        patterns = ["**/*.db", "**/*.sqlite"]
    else:
        patterns = [p.strip() for p in patterns_input.split(",")]

    # Create scanner
    scanner = ExemplarScanner(
        source_path=source,
        output_structure=output
    )

    # Run scan with custom paths
    print("\nStarting scan...\n")
    summary = scanner.scan(
        custom_paths=patterns
    )

    # Print results
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"Databases found: {summary['total_found']}")
    print(f"Databases failed: {summary['total_failed']}")
    print(f"Schemas generated: {summary['schemas_generated']}")
    print(f"Rubrics generated: {summary['rubrics_generated']}")
    print(f"\nOutput directory: {output.root}")

    if summary['found_databases']:
        print(f"\nFound databases:")
        for db in summary['found_databases']:
            status = "[OK]" if db.get('schema_generated') else "[WARNING]"
            print(f"  {status} {db.get('name', 'Unknown')}")
            print(f"     Path: {db['relative_path']}")

    return True


def test_single_database():
    """
    Test schema/rubric generation on a single database file.

    This is the simplest test - just tests the core functionality.
    """
    print("=" * 80)
    print("TEST MODE: Single Database")
    print("=" * 80)
    print()

    print("Enter path to a SQLite database file:")
    print("Examples:")
    print(f"  - {Path.home()}/Library/Safari/History.db")
    print(f"  - {Path.home()}/Library/Messages/chat.db")

    db_path_input = input("\nDatabase path: ").strip()

    if not db_path_input:
        print("Cancelled.")
        return False

    db_path = Path(db_path_input)

    if not db_path.exists():
        print(f"Error: File not found: {db_path}")
        return False

    if not db_path.is_file():
        print(f"Error: Not a file: {db_path}")
        return False

    # Create test output
    output = OutputStructure(
        base_output_dir=Path.cwd() / "test_output",
        case_name=f"Test_SingleDB_{db_path.stem}"
    )
    output.create()

    print(f"\nDatabase: {db_path}")
    print(f"Output: {output.root}\n")

    # Create scanner with parent directory as source
    scanner = ExemplarScanner(
        source_path=db_path.parent,
        output_structure=output
    )

    # Create a minimal database definition
    db_def = {
        "name": db_path.name,
        "path": str(db_path),
        "description": f"Test database: {db_path.name}",
        "priority": "test",
        "category": "test",
    }

    # Process the single database
    print("Processing database...\n")
    try:
        scanner._process_database(db_path, db_def, "test")

        print("[OK] Success!")
        print(f"\nOutput directory: {output.root}")

        # Show what was created
        schema_dir = output.get_schema_dir(db_path.name)
        if schema_dir.exists():
            print(f"\nGenerated files:")
            for file in sorted(schema_dir.iterdir()):
                print(f"  - {file.name}")

        # Show original copy
        original_dir = output.get_original_db_dir(db_path.name)
        if original_dir.exists():
            print(f"\nOriginal copy:")
            for file in sorted(original_dir.iterdir()):
                print(f"  - {file.name}")

        return True

    except Exception as e:
        print(f"[ERROR] Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Test the exemplar scanner with different modes"
    )
    parser.add_argument(
        "--mode",
        choices=["local", "volume", "custom", "single", "interactive"],
        default="interactive",
        help="Test mode (default: interactive menu)"
    )

    args = parser.parse_args()

    if args.mode == "interactive":
        # Interactive menu
        print("\n" + "=" * 80)
        print("Mac Log Sleuth - Exemplar Scanner Test")
        print("=" * 80)
        print("\nSelect test mode:")
        print("  1. Local User Library (safest - scans ~/Library)")
        print("  2. Mounted Volume (e.g., /Volumes/MacintoshHD)")
        print("  3. Custom Path (specify any directory)")
        print("  4. Single Database (test one .db file)")
        print("  0. Exit")

        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            test_local_system()
        elif choice == "2":
            test_mounted_volume()
        elif choice == "3":
            test_custom_path()
        elif choice == "4":
            test_single_database()
        else:
            print("Cancelled.")
            return

    elif args.mode == "local":
        test_local_system()
    elif args.mode == "volume":
        test_mounted_volume()
    elif args.mode == "custom":
        test_custom_path()
    elif args.mode == "single":
        test_single_database()

    print("\n[OK] Test complete!")


if __name__ == "__main__":
    main()
