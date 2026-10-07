#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare benchmark dataset for MLLM breathy/whispery voice detection testing.

Constructs 3 subsets:
1. Positive (Ishi real paralinguistic breathy/whispery dialog samples, 40 items)
2. Negative (ITA full sentences with confirmed non-breathy modal voice, 40 items)
3. Transition (Synthesized transitions: ITA modal full sentence concatenated with Ishi breathy sample, 20 items)

Outputs:
manifests/test_annot_mllm/benchmark_manifest.json
manifests/test_annot_mllm/audio/ (for transition audio)
"""

from __future__ import annotations

import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import soundfile as sf

# Reconfigure stdout for UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ISHI_DIR = Path(r"C:\Users\mumu\Downloads\breathywhispery")
ITA_AUDIO_DIR = Path(r"E:\Corpus\Breathy\ita_rion_processed\audio")
ITA_CACHE_DIR = Path(r"E:\Corpus\Breathy\ita_rion_processed\textgrid_flat\_breathy_cache")
OUTPUT_MANIFEST_DIR = Path(r"manifests\test_annot_mllm")
OUTPUT_AUDIO_DIR = OUTPUT_MANIFEST_DIR / "audio"

TARGET_SR = 16000


def get_audio_info(path: Path) -> Tuple[int, float]:
    """Get samplerate and duration in seconds."""
    info = sf.info(str(path))
    return info.samplerate, info.duration


def load_ishi_samples() -> List[Dict[str, Any]]:
    """Load Ishi metadata and map audio paths."""
    metadata_path = ISHI_DIR / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Ishi metadata not found at {metadata_path}")

    samples = []
    with open(metadata_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            new_fn = row["new_filename"]
            wav_path = ISHI_DIR / new_fn
            if not wav_path.exists():
                # Check without prefix or by category
                alt_path = ISHI_DIR / row["orig_filename"]
                if alt_path.exists():
                    wav_path = alt_path
                else:
                    continue

            sr, dur = get_audio_info(wav_path)
            samples.append({
                "sample_id": f"ishi_{row['index']}",
                "source": "ishi_corpus",
                "category": row.get("major_name", ""),
                "sub_cat": row.get("sub_cat", ""),
                "context": row.get("context", ""),
                "audio_path": str(wav_path),
                "duration_sec": round(dur, 3),
                "gt_has_breathy": True,  # Ishi corpus samples are known breathy/whispery
                "gt_type": "breathy_or_whispery",
                "gt_segments": [{
                    "start_sec": 0.0,
                    "end_sec": round(dur, 3),
                    "phonation_type": "breathy_or_whispery"
                }]
            })
    return samples


def find_ita_pure_modal_sentences(min_vowels: int = 5, limit: int = 100) -> List[Dict[str, Any]]:
    """Find ITA sentences where all annotated vowels are strictly 'non_breathy'."""
    candidates = []

    # Check batch shards in batches/batch_1 through batch_6
    shard_dirs = []
    batches_dir = ITA_CACHE_DIR / "batches"
    if batches_dir.exists():
        for b_dir in batches_dir.iterdir():
            s_dir = b_dir / "manual_breathy_annotations.json.d"
            if s_dir.exists():
                shard_dirs.append(s_dir)

    global_shard_dir = ITA_CACHE_DIR / "manual_breathy_annotations.json.d"
    if global_shard_dir.exists():
        shard_dirs.append(global_shard_dir)

    seen_keys = set()
    for s_dir in shard_dirs:
        for json_file in s_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                fk = data.get("file_key")
                if not fk or fk in seen_keys:
                    continue

                annotations = data.get("annotations", {})
                if not annotations:
                    continue

                # Ensure all annotations are 'non_breathy'
                labels = list(annotations.values())
                if len(labels) >= min_vowels and all(lbl == "non_breathy" for lbl in labels):
                    # Check audio file
                    audio_path = ITA_AUDIO_DIR / f"{fk}.flac"
                    if not audio_path.exists():
                        audio_path = ITA_AUDIO_DIR / f"{fk}.wav"
                    if audio_path.exists():
                        seen_keys.add(fk)
                        sr, dur = get_audio_info(audio_path)
                        candidates.append({
                            "file_key": fk,
                            "audio_path": str(audio_path),
                            "duration_sec": round(dur, 3),
                            "speaker_id": data.get("speaker_id", ""),
                            "num_vowels": len(labels)
                        })
            except Exception:
                continue

    # Sort or shuffle
    random.seed(42)
    random.shuffle(candidates)
    return candidates[:limit]


def create_transition_sample(
    modal_sample: Dict[str, Any],
    breathy_sample: Dict[str, Any],
    out_path: Path
) -> Dict[str, Any]:
    """Concatenate modal sentence with breathy sample with a 150ms crossfade/silence."""
    wav_modal, sr1 = sf.read(modal_sample["audio_path"])
    wav_breathy, sr2 = sf.read(breathy_sample["audio_path"])

    # Ensure mono
    if wav_modal.ndim > 1:
        wav_modal = wav_modal.mean(axis=1)
    if wav_breathy.ndim > 1:
        wav_breathy = wav_breathy.mean(axis=1)

    # Resample if needed (simplistic or assuming 16kHz/44.1kHz, let's resample using scipy/librosa/numpy)
    if sr1 != sr2:
        import torchaudio.transforms as T
        import torch
        resampler = T.Resample(sr2, sr1)
        wav_breathy_t = resampler(torch.tensor(wav_breathy, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        wav_breathy = wav_breathy_t
        sr_out = sr1
    else:
        sr_out = sr1

    # Insert 100ms pause
    pause_len = int(0.100 * sr_out)
    pause = np.zeros(pause_len, dtype=np.float32)

    combined = np.concatenate([wav_modal.astype(np.float32), pause, wav_breathy.astype(np.float32)])
    sf.write(str(out_path), combined, sr_out)

    modal_dur = len(wav_modal) / sr_out
    pause_dur = pause_len / sr_out
    total_dur = len(combined) / sr_out
    trans_start = modal_dur + pause_dur

    return {
        "sample_id": f"trans_{out_path.stem}",
        "source": "synthesized_transition",
        "audio_path": str(out_path),
        "duration_sec": round(total_dur, 3),
        "gt_has_breathy": True,
        "gt_type": "transition_modal_to_breathy",
        "modal_source": modal_sample["file_key"],
        "breathy_source": breathy_sample["sample_id"],
        "gt_segments": [{
            "start_sec": round(trans_start, 3),
            "end_sec": round(total_dur, 3),
            "phonation_type": "breathy_or_whispery"
        }]
    }


def main():
    OUTPUT_MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Ishi positive samples...")
    ishi_samples = load_ishi_samples()
    print(f"Loaded {len(ishi_samples)} Ishi positive samples.")

    print("Searching for ITA pure modal negative sentences...")
    ita_modal_pool = find_ita_pure_modal_sentences(min_vowels=5, limit=100)
    print(f"Found {len(ita_modal_pool)} qualified ITA modal sentences.")

    if len(ita_modal_pool) < 40:
        raise RuntimeError(f"Not enough ITA modal sentences ({len(ita_modal_pool)} < 40)")

    # 1. 40 Positive
    pos_samples = ishi_samples[:40]

    # 2. 40 Negative
    neg_selected = ita_modal_pool[:40]
    neg_samples = []
    for i, item in enumerate(neg_selected):
        neg_samples.append({
            "sample_id": f"ita_modal_{i+1:02d}",
            "source": "ita_modal_corpus",
            "file_key": item["file_key"],
            "speaker_id": item["speaker_id"],
            "audio_path": item["audio_path"],
            "duration_sec": item["duration_sec"],
            "gt_has_breathy": False,
            "gt_type": "pure_modal",
            "gt_segments": []
        })

    # 3. 20 Transition samples (ITA modal + Ishi breathy)
    trans_samples = []
    trans_ita_pool = ita_modal_pool[40:60]
    trans_ishi_pool = pos_samples[:20]

    print("Generating 20 synthesized transition samples...")
    for i in range(20):
        out_wav = OUTPUT_AUDIO_DIR / f"trans_{i+1:02d}.wav"
        t_sample = create_transition_sample(trans_ita_pool[i], trans_ishi_pool[i], out_wav)
        trans_samples.append(t_sample)

    all_samples = pos_samples + neg_samples + trans_samples
    manifest_path = OUTPUT_MANIFEST_DIR / "benchmark_manifest.json"

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "version": 1,
            "total_count": len(all_samples),
            "counts": {
                "positive_ishi": len(pos_samples),
                "negative_ita": len(neg_samples),
                "transition": len(trans_samples)
            },
            "samples": all_samples
        }, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Benchmark manifest generated successfully: {manifest_path}")
    print(f"Total samples: {len(all_samples)} (Pos: {len(pos_samples)}, Neg: {len(neg_samples)}, Trans: {len(trans_samples)})")


if __name__ == "__main__":
    main()
