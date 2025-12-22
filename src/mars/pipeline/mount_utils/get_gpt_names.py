#!/usr/bin/env python3
from __future__ import annotations

import json
import struct
import sys

from dfvfs.lib import definitions
from dfvfs.path import factory as psfactory
from dfvfs.resolver import resolver
from dfvfs.volume import apfs_volume_system

from mars.utils.debug_logger import logger

# ---------- minimal GPT parser (little-endian) ----------
GPT_HDR_FMT = "<8sIIIIQQQQ16sQIII420s"  # up to 512 bytes
# fields (subset): signature, rev, hdrsz, crc, reserved, current_lba, backup_lba, first_usable_lba,
# last_usable_lba, disk_guid, part_entry_lba, num_part_entries, part_entry_size, part_crc, pad...


def _root_ps(path: str):
    os_ps = psfactory.Factory.NewPathSpec(definitions.TYPE_INDICATOR_OS, location=path)
    if path.lower().endswith((".e01", ".e02", ".e03", ".ex01")):
        return psfactory.Factory.NewPathSpec(definitions.TYPE_INDICATOR_EWF, parent=os_ps)
    return psfactory.Factory.NewPathSpec(definitions.TYPE_INDICATOR_RAW, parent=os_ps)


def _open_file(pathspec):
    return resolver.Resolver.OpenFileObject(pathspec)


def _read_at(pathspec, offset: int, size: int) -> bytes:
    f = _open_file(pathspec)
    f.seek(offset)
    return f.read(size)


def _read_sector(pathspec, lba: int, bytes_per_sector: int) -> bytes:
    return _read_at(pathspec, lba * bytes_per_sector, bytes_per_sector)


def _parse_gpt(root_ps, bytes_per_sector=512):
    # Read LBA 1 (GPT header)
    hdr = _read_sector(root_ps, 1, bytes_per_sector)
    sig = hdr[:8]
    if sig != b"EFI PART":
        raise RuntimeError("Not a GPT disk (signature mismatch).")
    # unpack header
    (
        signature,
        rev,
        hdrsz,
        crc,
        _res,
        cur_lba,
        bak_lba,
        first_use,
        last_use,
        disk_guid,
        part_entry_lba,
        num_part,
        part_esize,
        part_crc,
        _pad,
    ) = struct.unpack(GPT_HDR_FMT, hdr[:512])

    # Read partition entries array
    table_size = num_part * part_esize
    # align to sector boundary read
    n_sectors = (table_size + bytes_per_sector - 1) // bytes_per_sector
    table = _read_at(root_ps, part_entry_lba * bytes_per_sector, n_sectors * bytes_per_sector)

    parts = []
    off = 0
    for idx in range(num_part):
        entry = table[off : off + part_esize]
        off += part_esize
        if len(entry) < 128:  # minimal entry size guard
            break
        # Partition entry layout (UEFI spec):
        #  0x00: type GUID (16)
        #  0x10: unique GUID (16)
        #  0x20: first LBA (8)
        #  0x28: last LBA  (8)
        #  0x30: attributes (8)
        #  0x38: name UTF-16LE (72 bytes / 36 chars)
        ptype_guid = entry[0:16]
        first_lba = struct.unpack("<Q", entry[0x20:0x28])[0]
        last_lba = struct.unpack("<Q", entry[0x28:0x30])[0]
        name_utf16 = entry[0x38 : 0x38 + 72]
        # skip unused entries (type GUID == zero)
        if ptype_guid == b"\x00" * 16 or first_lba == 0 or last_lba == 0:
            continue

        # human GUID
        def _fmt_guid(g):
            d = g.hex()
            return f"{d[6:8]}{d[4:6]}{d[2:4]}{d[0:2]}-{d[10:12]}{d[8:10]}-{d[14:16]}{d[12:14]}-{d[16:20]}-{d[20:32]}".upper()

        guid = _fmt_guid(ptype_guid)
        # decode UTF-16LE name (strip trailing nulls)
        try:
            name = name_utf16.decode("utf-16le").rstrip("\x00")
        except Exception:
            name = None
        size_bytes = (last_lba - first_lba + 1) * bytes_per_sector
        parts.append(
            {
                "index": idx,  # 0-based table index
                "first_lba": first_lba,
                "last_lba": last_lba,
                "offset": first_lba * bytes_per_sector,
                "size": size_bytes,
                "gpt_type_guid": guid,
                "gpt_entry_name": name,
            }
        )
    return parts


# ---------- FS detection & label helpers (work off DATA_RANGE) ----------
def _slice_ps(root_ps, offset: int, size: int):
    return psfactory.Factory.NewPathSpec(
        definitions.TYPE_INDICATOR_DATA_RANGE,
        parent=root_ps,
        range_offset=offset,
        range_size=size,
    )


def _read_slice(dr_ps, size: int, offset: int = 0) -> bytes:
    f = _open_file(dr_ps)
    f.seek(offset)
    return f.read(size)


def _detect_fs(dr_ps):
    b0 = _read_slice(dr_ps, 4096, 0)
    if len(b0) < 512:
        return None
    if b0[:4] == b"NXSB":
        return "apfs"
    if b0[3:11] == b"NTFS    ":
        return "ntfs"
    if b0[3:11] == b"EXFAT   ":
        return "fatlike"
    if b0[510:512] == b"\x55\xaa" and any(sig in b0[:128] for sig in (b"FAT12", b"FAT16", b"FAT32")):
        return "fatlike"
    b1k = _read_slice(dr_ps, 512, 1024)
    if len(b1k) >= 2 and b1k[:2] in (b"H+", b"HX"):
        return "hfs"
    return None


def _label_hfs(dr_ps):
    ps = psfactory.Factory.NewPathSpec(definitions.TYPE_INDICATOR_HFS, location="/", parent=dr_ps)
    fs = resolver.Resolver.OpenFileSystem(ps)
    vol = getattr(fs, "GetHFSVolume", lambda: None)()
    return getattr(vol, "name", None) or getattr(vol, "volume_name", None) or getattr(fs, "volume_name", None)


def _label_ntfs(dr_ps):
    ps = psfactory.Factory.NewPathSpec(definitions.TYPE_INDICATOR_NTFS, location="/", parent=dr_ps)
    fs = resolver.Resolver.OpenFileSystem(ps)
    vol = getattr(fs, "GetNTFSVolume", lambda: None)()
    return getattr(vol, "label", None) or getattr(vol, "name", None) or getattr(fs, "label", None)


def _label_fatlike(dr_ps):
    ps = psfactory.Factory.NewPathSpec(definitions.TYPE_INDICATOR_FAT, location="/", parent=dr_ps)
    fs = resolver.Resolver.OpenFileSystem(ps)
    return getattr(fs, "label", None) or getattr(fs, "volume_label", None) or getattr(fs, "name", None)


def _apfs_names(dr_ps):
    container_ps = psfactory.Factory.NewPathSpec(definitions.TYPE_INDICATOR_APFS_CONTAINER, parent=dr_ps)
    vs = apfs_volume_system.APFSVolumeSystem()
    vs.Open(container_ps)
    out = []
    for idx, v in enumerate(vs.volumes):
        a = v.GetAttribute("name")
        nm = a.value if a else None
        if not nm:
            vol_ps = psfactory.Factory.NewPathSpec(
                definitions.TYPE_INDICATOR_APFS,
                location="/",
                parent=container_ps,
                volume_index=idx,
            )
            try:
                fs = resolver.Resolver.OpenFileSystem(vol_ps)
                vol_obj = getattr(fs, "GetAPFSVolume", lambda: None)()
                nm = getattr(vol_obj, "name", None) or getattr(fs, "name", None)
            except Exception:
                nm = None
        out.append(nm)
    return out


# ---------- main ----------
def enumerate_volume_names(image_path: str, sector_size=512):
    root_ps = _root_ps(image_path)
    parts = _parse_gpt(root_ps, sector_size)
    results = []
    # Known APFS partition type GUID
    APFS_GUID = "7C3457EF-0000-11AA-AA11-00306543ECAC"

    for p in parts:
        dr = _slice_ps(root_ps, p["offset"], p["size"])
        rec = {
            "id": f"p{p['index'] + 1}",
            "gpt_entry_name": p["gpt_entry_name"],
            "gpt_type_guid": p["gpt_type_guid"],
            "fs": None,
            "label": None,
            "apfs": None,
        }
        kind = _detect_fs(dr)

        # If filesystem detection failed but GPT type indicates APFS, try APFS anyway
        if kind is None and p["gpt_type_guid"] == APFS_GUID:
            kind = "apfs"

        try:
            if kind == "apfs":
                rec["fs"] = "APFS (container)"
                rec["apfs"] = _apfs_names(dr)
            elif kind == "hfs":
                rec["fs"] = "HFS+"
                rec["label"] = _label_hfs(dr)
            elif kind == "ntfs":
                rec["fs"] = "NTFS"
                rec["label"] = _label_ntfs(dr)
            elif kind == "fatlike":
                rec["fs"] = "FAT/exFAT"
                rec["label"] = _label_fatlike(dr)
        except Exception:
            pass
        results.append(rec)
    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error(f"Usage: {sys.argv[0]} /path/to/disk.img_or.E01")
        sys.exit(1)
    logger.info(json.dumps(enumerate_volume_names(sys.argv[1]), indent=2))
