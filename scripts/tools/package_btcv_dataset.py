#!/usr/bin/env python3
"""
tools/package_btcv_dataset.py
Packages converted BTCV dataset (data/btcv and data/lists_Synapse)
into a single compact zip archive for Colab local SSD fast extraction.
"""

import os
import sys
import zipfile
from pathlib import Path
from tqdm import tqdm

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import shutil

_HERE = Path(__file__).resolve().parent
REPO_ROOT = _HERE.parent.parent if _HERE.parent.name == "scripts" else _HERE.parent
OUTPUT_ZIP = REPO_ROOT / "outputs" / "btcv_data.zip"

DRIVE_CANDIDATES = [
    Path(r"I:\My Drive\IAD"),
    Path(r"I:\My Drive\boost"),
    Path(r"H:\My Drive\IAD"),
    Path(r"H:\My Drive\boost"),
    Path(r"G:\.shortcut-targets-by-id\1UYom4Idt529jZ6efqZ24lXiJG6zt7lO7\IAD"),
    Path(r"G:\My Drive\IAD"),
    Path(r"/content/drive/MyDrive/IAD"),
    Path(r"/content/drive/MyDrive/boost"),
]


def package_btcv_data():
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT_ZIP.exists() and OUTPUT_ZIP.stat().st_size > 400 * 1024 * 1024):
        print(f"[Dataset] Packaging BTCV dataset into {OUTPUT_ZIP}...")
        items_to_pack = [
            REPO_ROOT / "data" / "btcv",
            REPO_ROOT / "data" / "lists_Synapse",
        ]

        total_files = 0
        for item in items_to_pack:
            if item.is_dir():
                for root, _, files in os.walk(item):
                    total_files += len(files)

        with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
            with tqdm(total=total_files, desc="Archiving dataset") as pbar:
                for item in items_to_pack:
                    for root, _, files in os.walk(item):
                        for f in files:
                            fp = Path(root) / f
                            arcname = fp.relative_to(REPO_ROOT).as_posix()
                            zf.write(fp, arcname)
                            pbar.update(1)

        print(f"[Dataset] Successfully created {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size / (1024*1024):.2f} MB)")
    else:
        print(f"[Dataset] {OUTPUT_ZIP} already exists ({OUTPUT_ZIP.stat().st_size / (1024*1024):.1f} MB). Skipping creation.")

    # Sync to parent workspace outputs if nested
    parent_outputs = REPO_ROOT.parent / "outputs"
    if parent_outputs.exists() and parent_outputs.resolve() != OUTPUT_ZIP.parent.resolve():
        dst = parent_outputs / "btcv_data.zip"
        if not dst.exists() or dst.stat().st_size != OUTPUT_ZIP.stat().st_size:
            shutil.copy2(OUTPUT_ZIP, dst)
            print(f"[Dataset] Synced package to workspace outputs: {dst}")

    # Sync to drive candidates
    for drive_dir in DRIVE_CANDIDATES:
        if drive_dir.exists():
            for dst_rel in ["btcv_data.zip", "outputs/btcv_data.zip"]:
                dst = drive_dir / dst_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists() or dst.stat().st_size != OUTPUT_ZIP.stat().st_size:
                    shutil.copy2(OUTPUT_ZIP, dst)
                    print(f"[Dataset] Synced btcv_data.zip to {dst}")

    return OUTPUT_ZIP


if __name__ == "__main__":
    package_btcv_data()
