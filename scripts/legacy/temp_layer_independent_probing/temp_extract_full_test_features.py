"""
Extracts 13-layer pooled features for the FULL canonical test sets:
- Creaky: N = 3,147 tokens across 18 speakers (79 unique audio files)
- Breathy: N = 3,932 tokens across 29 speakers (146 unique audio files)
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch

BASE_DIR = Path(r"E:\Coding\WavFry")
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from wavfry.eval.unified import load_model_from_checkpoint
from wavfry.model.wavlm_classifier import WavFryClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("temp_extract_full")

CACHE_DIR = BASE_DIR / "data" / "experiments" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_P = CODE_DIR / "manifests" / "segments_multi.csv"


def extract_full_test_for_task(task: str, device: torch.device):
    out_cache = CACHE_DIR / f"temp_full_test_features_cache_{task}.npz"
    if out_cache.exists():
        logger.info(f"Full test feature cache already exists: {out_cache}")
        return

    logger.info(f"Loading full test manifest for {task}...")
    df = pd.read_csv(MANIFEST_P, low_memory=False)
    test_df = df[df["batch_split"] == "test"].copy()
    label_col = f"label_{task}"
    task_df = test_df[test_df[label_col].notna()].copy().reset_index(drop=True)
    logger.info(f"Task {task}: {len(task_df)} tokens across {task_df['speaker_id'].nunique()} speakers.")

    ckpt_path = CODE_DIR / "exp_results" / "checkpoints" / "ablation_backbone_wavlm_baseplus" / "best_model.pt"
    model, _ = load_model_from_checkpoint(ckpt_path, device, WavFryClassifier)
    model.eval()

    staged_clean_dir = CODE_DIR / "data" / "experiments" / "cache" / "staged_audio" / "clean"
    unique_fkeys = task_df["file_key"].unique()
    fkey_to_path: Dict[str, Path] = {}
    for _, row in task_df.iterrows():
        fkey = str(row["file_key"])
        if fkey not in fkey_to_path:
            staged_p = staged_clean_dir / f"{fkey}.wav"
            if staged_p.exists():
                fkey_to_path[fkey] = staged_p
            elif "audio_path" in row and Path(str(row["audio_path"])).exists():
                fkey_to_path[fkey] = Path(str(row["audio_path"]))

    file_hidden_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    t0 = time.time()
    logger.info(f"Running WavLM forward pass for {len(unique_fkeys)} unique audio files ({task})...")

    with torch.no_grad():
        for i, fkey in enumerate(unique_fkeys):
            if fkey not in fkey_to_path:
                logger.warning(f"Audio missing for {fkey}")
                continue
            wav_p = fkey_to_path[fkey]
            audio, sr = sf.read(str(wav_p), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != 16000:
                import torchaudio.functional as AF
                audio_t = AF.resample(torch.from_numpy(audio), orig_freq=sr, new_freq=16000)
                audio = audio_t.numpy()

            x = torch.from_numpy(audio).unsqueeze(0).to(device)
            res = model.wavlm(x, output_hidden_states=True)
            hs_stack = torch.stack(res.hidden_states, dim=1).squeeze(0).cpu().numpy()  # [13, T, 768]
            T = hs_stack.shape[1]
            timestamps = (np.arange(T, dtype=np.float32) * 320.0 + 160.0) / 16000.0
            file_hidden_cache[fkey] = (timestamps, hs_stack)
            if (i + 1) % 20 == 0 or (i + 1) == len(unique_fkeys):
                logger.info(f"  Processed {i+1}/{len(unique_fkeys)} files ({time.time()-t0:.1f}s)...")

    # Pool interval features across all tokens
    all_feats = []
    labels = []
    speakers = []
    interval_keys = []
    phone_types = []

    for _, row in task_df.iterrows():
        fkey = str(row["file_key"])
        s_time = float(row["start_s"])
        e_time = float(row["end_s"])
        y_val = int(row[label_col])
        spk = str(row["speaker_id"])
        ptype = str(row.get("phone_type", "unknown"))
        int_key = f"{row.get('dataset_id', 'ds')}::{fkey}::{s_time:.4f}_{e_time:.4f}"

        if fkey not in file_hidden_cache:
            continue
        timestamps, hs_stack = file_hidden_cache[fkey]
        mask = (timestamps >= s_time) & (timestamps <= e_time)
        if np.any(mask):
            feat_pooled = np.mean(hs_stack[:, mask, :], axis=1)  # [13, 768]
        else:
            idx = int(np.argmin(np.abs(timestamps - (s_time + e_time) / 2.0)))
            feat_pooled = hs_stack[:, idx, :]  # [13, 768]

        all_feats.append(feat_pooled)
        labels.append(y_val)
        speakers.append(spk)
        interval_keys.append(int_key)
        phone_types.append(ptype)

    features_3d = np.array(all_feats, dtype=np.float32)  # [N, 13, 768]
    labels_arr = np.array(labels, dtype=np.int32)
    speakers_arr = np.array(speakers, dtype=object)
    keys_arr = np.array(interval_keys, dtype=object)
    ptypes_arr = np.array(phone_types, dtype=object)

    np.savez_compressed(
        out_cache,
        features_3d=features_3d,
        labels=labels_arr,
        speakers=speakers_arr,
        interval_keys=keys_arr,
        phone_types=ptypes_arr,
    )
    logger.info(f"Saved full test feature cache: {out_cache} (Shape: {features_3d.shape})")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    extract_full_test_for_task("creaky", device)
    extract_full_test_for_task("breathy", device)
    logger.info("Full test feature extraction completed.")


if __name__ == "__main__":
    main()
