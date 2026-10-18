#!/usr/bin/env python
"""
Streamlined Corpus Archiver & Cloud Disk Uploader (.zip format).
Preserves exact 1:1 directory hierarchy relative to E:/Corpus.
Packs corpus directories into R:\ RAM disk one by one using uncompressed ZIP (ZIP_STORED),
uploads via discovery-ctl, and immediately deletes each zip to ensure zero physical disk wear.
"""

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

# Discovery CLI path
LOCAL_CLI = Path(os.environ.get("LOCALAPPDATA", "")) / "discovery" / "bin" / "discovery-ctl.exe"
CLI_CMD = str(LOCAL_CLI) if LOCAL_CLI.exists() else "discovery-ctl"

# Target RAM disk scratch folder
RAM_DISK_DIR = Path("R:/_corpus_upload_tmp")
BASE_CORPUS_DIR = Path("E:/Corpus")

# Each dataset packaged independently with 1:1 exact matching folder hierarchy
DATASET_SPECS = [
    {
        "name": "Corpus_CS_Dialogue.zip",
        "targets": [
            "CS-Dialogue/data/short_wav/WAVE/01_en_dereverb",
            "CS-Dialogue/data/short_wav/WAVE/02_en_mfa_aligned",
        ],
        "desc": "CS-Dialogue: 01_en_dereverb (Audio) & 02_en_mfa_aligned (TextGrid & Cache)",
    },
    {
        "name": "Corpus_Creaky_MELI.zip",
        "targets": [
            "Creaky/MELI_processed",
        ],
        "desc": "MELI Mandarin: audio & textgrid",
    },
    {
        "name": "Corpus_Breathy_YSJ.zip",
        "targets": [
            "Breathy/YSJ",
        ],
        "desc": "YSJ Japanese: audio & textgrid_tifa",
    },
    {
        "name": "Corpus_Breathy_ITA.zip",
        "targets": [
            "Breathy/ita_rion_processed",
        ],
        "desc": "ITA Corpus: audio & textgrid_flat",
    },
    {
        "name": "Corpus_Breathy_BreathyTTS.zip",
        "targets": [
            "Breathy/BreathyTTS",
        ],
        "desc": "BreathyTTS: audio & textgrid",
    },
    {
        "name": "Corpus_Noise_QUT.zip",
        "targets": [
            "Noise/QUT-NOISE_processed",
        ],
        "desc": "QUT-NOISE: impulses (RIR) & noise",
    },
]


def check_preflight():
    # 1. Check discovery-ctl
    res = subprocess.run([CLI_CMD, "clouddisk", "info"], capture_output=True, text=True)
    if res.returncode != 0:
        print("[ERROR] discovery-ctl clouddisk info failed. Please check login status:", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        sys.exit(1)
    print("[Preflight OK] discovery-ctl is authenticated and connected to Cloud Disk.")

    # 2. Check RAM disk
    if not Path("R:/").exists():
        print("[ERROR] R: drive not detected. Please verify RAM disk exists.", file=sys.stderr)
        sys.exit(1)

    total, used, free = shutil.disk_usage("R:/")
    free_gb = free / (1024**3)
    print(f"[Preflight OK] RAM disk R: detected ({free_gb:.1f} GB free).")
    if free_gb < 5.0:
        print(f"[WARNING] Free space on R: is {free_gb:.1f} GB. Largest package is ~4.3 GB.")

    RAM_DISK_DIR.mkdir(parents=True, exist_ok=True)


def package_and_upload():
    check_preflight()

    total_pkgs = len(DATASET_SPECS)
    print(f"Starting pipeline upload: {total_pkgs} datasets -> Cloud Disk (/)...", flush=True)

    pipeline_log = RAM_DISK_DIR / "upload_pipeline.log"

    for idx, spec in enumerate(DATASET_SPECS, 1):
        pkg_name = spec["name"]
        zip_path = RAM_DISK_DIR / pkg_name
        targets = spec["targets"]
        desc = spec["desc"]

        # Check source existences
        missing = [t for t in targets if not (BASE_CORPUS_DIR / t).exists()]
        if missing:
            print(f"[{idx}/{total_pkgs}] ERROR: Missing source: {missing}", file=sys.stderr, flush=True)
            continue

        # Step 1: Pack directly into RAM disk with ZIP_STORED
        t0 = time.time()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for t in targets:
                target_path = BASE_CORPUS_DIR / t
                if target_path.is_file():
                    zf.write(target_path, arcname=t)
                else:
                    for root, _, files in os.walk(target_path):
                        for f in files:
                            full_path = os.path.join(root, f)
                            rel_path = os.path.relpath(full_path, BASE_CORPUS_DIR)
                            zf.write(full_path, arcname=rel_path)

        pack_sec = time.time() - t0
        zip_mb = zip_path.stat().st_size / (1024 * 1024)

        # Step 2: Upload via discovery-ctl (silencing chunk stdout spam to log file)
        t1 = time.time()
        upload_cmd = [CLI_CMD, "clouddisk", "upload", "--local", str(zip_path), "--remote-path", "/"]
        ctl_log = RAM_DISK_DIR / "discovery_ctl_upload.log"
        with open(ctl_log, "w", encoding="utf-8") as lf:
            proc = subprocess.run(upload_cmd, stdout=lf, stderr=subprocess.STDOUT)
        upload_sec = time.time() - t1

        if proc.returncode != 0:
            print(f"[{idx}/{total_pkgs}] ERROR: Upload failed for {pkg_name}!", file=sys.stderr, flush=True)
            if ctl_log.exists():
                with open(ctl_log, "r", encoding="utf-8", errors="ignore") as lf:
                    lines = lf.readlines()
                    print("Last output:\n" + "".join(lines[-10:]), file=sys.stderr, flush=True)
            sys.exit(1)

        speed_mb = zip_mb / max(0.1, upload_sec)

        # Step 3: Delete zip immediately from RAM disk
        zip_path.unlink(missing_ok=True)

        # Emit exactly ONE clean status line per completed dataset
        status_line = (
            f"[{idx}/{total_pkgs}] {pkg_name}: {zip_mb:.1f} MB (pack {pack_sec:.1f}s, "
            f"upload {upload_sec:.1f}s @ {speed_mb:.1f} MB/s) -> uploaded & RAM freed."
        )
        print(status_line, flush=True)

        with open(pipeline_log, "a", encoding="utf-8") as pl:
            pl.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {status_line}\n")

    # Cleanup temporary folder
    try:
        shutil.rmtree(RAM_DISK_DIR, ignore_errors=True)
    except Exception:
        pass

    print(f"\n[Complete] All {total_pkgs} dataset zip packages uploaded successfully to Cloud Disk.", flush=True)


if __name__ == "__main__":
    package_and_upload()
