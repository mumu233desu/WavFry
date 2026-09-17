#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and archive all breathy voice audio, TextGrids, EGG, and documentation
from the UCLA Voice Quality Project (NSF BCS-0720304) to E:\\Corpus\\Breathy\\UCLA.

Breathy voice languages in the project:
1. Gujarati (Gujarati_Audio.zip, Gujarati_TextGrids.zip, EGG, Notes, WordList)
2. Mazatec (Mazatec_Audio.zip, Mazatec_TextGrids.zip, Notes, WordList)
3. Hmong / White Hmong (Hmong_Audio.zip, Hmong_TextGrids.zip, EGG, Notes, WordList)
4. Miao / Black Miao (Miao_Audio.zip, Miao_TextGrids.zip, EGG, Notes, WordList)
5. Zapotec (Zapotec_Audio.zip, Zapotec_TextGrids.zip, EGG, Notes, WordList)

Plus the master spreadsheets:
- Voice_Master.xlsx
- Voice_EGG.xlsx
- Voice_Readme.pdf
- egg_notes.html
"""

import os
import sys
import time
import zipfile
import requests
from pathlib import Path

BASE_URL = "http://www.phonetics.ucla.edu/voiceproject"
TARGET_ROOT = Path(r"E:\Corpus\Breathy\UCLA")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Master files
MASTER_FILES = [
    ("Spreadsheet/Voice_Master.xlsx", "_master/Voice_Master.xlsx"),
    ("Spreadsheet/Voice_EGG.xlsx", "_master/Voice_EGG.xlsx"),
    ("Spreadsheet/Voice_Readme.pdf", "_master/Voice_Readme.pdf"),
    ("egg_notes.html", "_master/egg_notes.html"),
]

# Language packages that contain Breathy Voice
BREATHY_LANGUAGES = {
    "Gujarati": [
        "Gujarati_Audio.zip",
        "Gujarati_TextGrids.zip",
        "Gujarati_Notes.pdf",
        "Gujarati_WordList.pdf",
        "Gujarati_EGG.zip",
    ],
    "Mazatec": [
        "Mazatec_Audio.zip",
        "Mazatec_TextGrids.zip",
        "Mazatec_Notes.pdf",
        "Mazatec_WordList.pdf",
    ],
    "Hmong": [
        "Hmong_Audio.zip",
        "Hmong_TextGrids.zip",
        "Hmong_Notes.pdf",
        "Hmong_WordList.pdf",
        "Hmong_EGG.zip",
    ],
    "Miao": [
        "Miao_Audio.zip",
        "Miao_TextGrids.zip",
        "Miao_Notes.pdf",
        "Miao_WordList.pdf",
        "Miao_EGG.zip",
    ],
    "Zapotec": [
        "Zapotec_Audio.zip",
        "Zapotec_TextGrids.zip",
        "Zapotec_Notes.pdf",
        "Zapotec_WordList.pdf",
        "Zapotec_EGG.zip",
    ],
}


def download_file(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download a file with streaming and resume/verification."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{attempt}/{max_retries}] Connecting to {url} ...", flush=True)
            with requests.get(url, headers=HEADERS, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    print(f"  -> HTTP 404 Not Found: {url}", flush=True)
                    return False
                r.raise_for_status()

                total_size = int(r.headers.get("content-length", 0))
                
                # Check if already fully downloaded
                if dest_path.exists():
                    existing_size = dest_path.stat().st_size
                    if total_size > 0 and existing_size == total_size:
                        print(f"  -> File already exists and matches size ({existing_size / 1024 / 1024:.2f} MB): {dest_path.name}", flush=True)
                        return True
                    elif total_size == 0 and existing_size > 0:
                        print(f"  -> File already exists: {dest_path.name}", flush=True)
                        return True

                print(f"  -> Downloading {dest_path.name} ({total_size / 1024 / 1024:.2f} MB)...", flush=True)
                downloaded = 0
                start_time = time.time()
                last_log_time = start_time

                with open(temp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            now = time.time()
                            if now - last_log_time > 3.0:
                                elapsed = now - start_time
                                speed = (downloaded / 1024 / 1024) / max(elapsed, 0.001)
                                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                                print(f"     Progress: {downloaded / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB ({pct:.1f}%) - {speed:.2f} MB/s", flush=True)
                                last_log_time = now

                # Rename upon successful completion
                if temp_path.exists():
                    if dest_path.exists():
                        dest_path.unlink()
                    temp_path.rename(dest_path)
                    print(f"  -> Done! Saved to {dest_path}", flush=True)
                    return True

        except Exception as e:
            print(f"  -> Download attempt {attempt} failed: {e}", flush=True)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception:
                    pass
            if attempt < max_retries:
                time.sleep(3)

    return False


def main():
    print("=" * 70, flush=True)
    print("UCLA Voice Quality Project - Breathy Voice Archiver", flush=True)
    print(f"Target Directory: {TARGET_ROOT}", flush=True)
    print("=" * 70, flush=True)

    TARGET_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Download Master Files
    print("\n>>> 1. Downloading Master Documentation & Spreadsheets...", flush=True)
    for rel_url, rel_dest in MASTER_FILES:
        url = f"{BASE_URL}/{rel_url}"
        dest = TARGET_ROOT / rel_dest
        download_file(url, dest)

    # 2. Download Breathy Voice Languages
    total_files = 0
    success_files = 0
    for lang, files in BREATHY_LANGUAGES.items():
        print(f"\n>>> 2. Downloading {lang} (Breathy Voice Package)...", flush=True)
        lang_dir = TARGET_ROOT / lang
        for item in files:
            url = f"{BASE_URL}/{lang}/{item}"
            dest = lang_dir / item
            total_files += 1
            ok = download_file(url, dest)
            if ok:
                success_files += 1

    print("\n" + "=" * 70, flush=True)
    print(f"Archiving complete! Successfully archived {success_files}/{total_files} files.", flush=True)
    print(f"Destination: {TARGET_ROOT}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
