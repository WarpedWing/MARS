#!/bin/bash

# Mount a specific partition from a DD image
# Usage: ./mount_partition.sh /path/to/image.dd partition_number

if [ $# -lt 2 ]; then
    echo "Usage: $0 <dd_image> <partition_number>"
    echo ""
    echo "Example: $0 image.dd 2"
    exit 1
fi

DD_IMAGE="$1"
PARTITION_NUM="$2"

echo "Attempting to mount partition $PARTITION_NUM from $DD_IMAGE"
echo ""

# Attach without mounting
echo "[1/3] Attaching image without mounting..."
DEVICE=$(hdiutil attach -nomount -imagekey diskimage-class=CRawDiskImage "$DD_IMAGE" 2>&1 | grep -E '^/dev/disk' | awk '{print $1}' | head -1)

if [ -z "$DEVICE" ]; then
    echo "[ERROR] Could not attach image"
    exit 1
fi

echo "[OK] Attached as $DEVICE"
echo ""

# Show partitions
echo "[2/3] Partition layout:"
diskutil list "$DEVICE"
echo ""

# Mount the requested partition
PARTITION_DEVICE="${DEVICE}s${PARTITION_NUM}"
echo "[3/3] Mounting $PARTITION_DEVICE..."

diskutil mount readOnly "$PARTITION_DEVICE"

if [ $? -eq 0 ]; then
    echo ""
    echo "[OK] Partition mounted successfully"
    echo ""
    MOUNT_POINT=$(diskutil info "$PARTITION_DEVICE" | grep "Mount Point" | awk -F': ' '{print $2}')
    if [ -n "$MOUNT_POINT" ]; then
        echo "Mount point: $MOUNT_POINT"
        echo ""
        echo "To unmount:"
        echo "  hdiutil detach $DEVICE"
    fi
else
    echo "[ERROR] Failed to mount partition"
    echo "Detaching device..."
    hdiutil detach "$DEVICE"
    exit 1
fi
