#!/bin/bash

# Cleanup script for E01 forensic analysis
# Removes large temporary files to save disk space

echo "Mac Log Sleuth - E01 Analysis Cleanup"
echo "======================================"
echo ""

# Check if DD file exists
DD_FILE="/Volumes/Crux/Kelley_Brannon_Case/KB_Laptop_Image/DD/KBLaptopRAW.dd"

if [ -f "$DD_FILE" ]; then
    SIZE=$(du -h "$DD_FILE" | awk '{print $1}')
    echo "Found raw DD image: $DD_FILE"
    echo "Size: $SIZE"
    echo ""
    echo "WARNING: This will delete the raw DD conversion."
    echo "The original E01 file will NOT be touched."
    echo ""
    echo "You can always reconvert from E01 if needed using:"
    echo "  uv run src/mac_log_sleuth/pipeline/e01_workflow.py --convert [E01_FILE] --output [DD_FILE]"
    echo ""
    read -p "Delete DD file? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Deleting $DD_FILE..."
        rm "$DD_FILE"
        echo "[OK] Deleted"
    else
        echo "Skipped."
    fi
else
    echo "DD file not found: $DD_FILE"
fi

echo ""
echo "Disk space saved. Extracted databases are kept in:"
echo "  ./kb_extracted_databases/ (190 files, ~50MB)"
echo ""
echo "E01 forensic image remains intact at:"
echo "  [original E01 location]"
echo ""
