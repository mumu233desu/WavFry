r"""
High-throughput Frame-level Inference for WavFry on E:\Corpus\zt\v2.

Outputs:
  - Per-audio compressed .npz: timestamps, prob_creak, prob_breath
    saved to E:\Corpus\zt\v2\wavfry_probs\<speaker>\<file_key>.npz
  - Global metadata index: summary.parquet and summary.csv

Usage:
  uv run python scripts/infer_zt_v2_frame_probs.py --batch-size 64
"""

from __future__ import annotations

import os
# Force HuggingFace offline mode immediately
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import argparse
import logging
import math
from pathlib import Path
import sys
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

# Add 06_wavfry root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from wavfry.eval.unified import load_model_from_checkpoint
from wavfry.model.wavlm_classifier import WavFryClassifier

# Setup clean, non-flooding logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("infer_zt_v2")

DEFAULT_DATA_DIR = Path(r"E:\Corpus\zt\v2")
DEFAULT_OUTPUT_DIR = Path(r"E:\Corpus\zt\v2\wavfry_probs")
DEFAULT_CKPT = ROOT_DIR / "exp_results" / "checkpoints" / "ablation_backbone_wavlm_baseplus" / "best_model.pt"
OPTIMAL_THRESHOLD = 0.560
TARGET_SR = 16000


class AudioFileDataset(Dataset):
    """Dataset for audio reading and resampling to 16kHz."""

    def __init__(self, file_list: List[Path], target_sr: int = 16000):
        self.file_list = file_list
        self.target_sr = target_sr
        self.resampler = torchaudio.transforms.Resample(
            22050, self.target_sr, resampling_method="sinc_interp_hann"
        )
        self.fallback_resamplers = {}

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, str, Path]:
        path = self.file_list[idx]
        speaker = path.parent.name
        file_key = path.stem

        try:
            audio_data, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)

            waveform = torch.from_numpy(audio_data)

            if sr == 22050:
                waveform = self.resampler(waveform)
            elif sr != self.target_sr:
                if sr not in self.fallback_resamplers:
                    self.fallback_resamplers[sr] = torchaudio.transforms.Resample(
                        sr, self.target_sr, resampling_method="sinc_interp_hann"
                    )
                waveform = self.fallback_resamplers[sr](waveform)

            return waveform, speaker, file_key, path
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return torch.zeros(16000, dtype=torch.float32), speaker, file_key, path


def collate_audio_batch(batch):
    waveforms, speakers, file_keys, paths = zip(*batch)
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)

    padded_wavs = torch.zeros(len(waveforms), max_len, dtype=torch.float32)
    for i, w in enumerate(waveforms):
        padded_wavs[i, : w.shape[0]] = w

    return padded_wavs, torch.tensor(lengths, dtype=torch.int64), speakers, file_keys, paths


def run_inference(
    data_dir: Path,
    output_dir: Path,
    ckpt_path: Path,
    batch_size: int = 64,
    device_str: str = "cuda",
    num_workers: int = 0,
    force_recompute: bool = False,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info(f"Target device: {device}")
    logger.info(f"Scanning audio files in {data_dir}...")

    all_wavs = sorted(list(data_dir.glob("*/*.wav")))
    total_found = len(all_wavs)
    logger.info(f"Total audio files found: {total_found}")

    if not force_recompute:
        to_process = []
        for p in all_wavs:
            npz_p = output_dir / p.parent.name / f"{p.stem}.npz"
            if not (npz_p.exists() and npz_p.stat().st_size > 500):
                to_process.append(p)
        logger.info(f"Files needing processing (resume enabled): {len(to_process)} / {total_found}")
    else:
        to_process = all_wavs

    if not to_process:
        logger.info("All files already processed! Building summary index...")
        build_summary_index(output_dir, all_wavs)
        return

    logger.info(f"Loading checkpoint from {ckpt_path}...")
    model, ckpt_dict = load_model_from_checkpoint(ckpt_path, device, WavFryClassifier)
    model.eval()

    best_thresh = float(ckpt_dict.get("best_threshold", OPTIMAL_THRESHOLD))
    logger.info(f"Model loaded successfully. Optimal threshold: {best_thresh:.3f}")

    dataset = AudioFileDataset(to_process, target_sr=TARGET_SR)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=(device.type == "cuda"),
    )

    total_batches = len(loader)
    logger.info(f"Starting batch inference: {len(to_process)} files in {total_batches} batches (batch_size={batch_size})...")

    t_start = time.time()
    processed_count = 0
    stride_samples = None
    receptive_samples = None

    log_interval = max(1, total_batches // 10)  # Log every 10%

    for batch_idx, (padded_wavs, lengths, speakers, file_keys, paths) in enumerate(loader):
        padded_wavs = padded_wavs.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        with torch.no_grad():
            frames = model(padded_wavs, lengths=lengths, return_frames=True)
            c_logits = frames["frame_logits"]["creaky"]
            b_logits = frames["frame_logits"]["breathy"]

            c_probs = torch.sigmoid(c_logits).cpu().numpy().astype(np.float32)
            b_probs = torch.sigmoid(b_logits).cpu().numpy().astype(np.float32)

            out_lengths = model.wavlm._get_feat_extract_output_lengths(lengths).cpu().numpy()

            if stride_samples is None:
                stride_samples = frames.get("frame_stride_samples", 320)
                receptive_samples = frames.get("frame_receptive_samples", 400)

        # Save each file's npz
        for i in range(len(speakers)):
            spk = speakers[i]
            k = file_keys[i]
            n_f = int(out_lengths[i])

            p_creak = c_probs[i, :n_f]
            p_breath = b_probs[i, :n_f]
            timestamps = (np.arange(n_f, dtype=np.float32) * stride_samples + receptive_samples / 2.0) / TARGET_SR

            spk_dir = output_dir / spk
            spk_dir.mkdir(parents=True, exist_ok=True)
            out_npz = spk_dir / f"{k}.npz"

            np.savez_compressed(
                out_npz,
                timestamps=timestamps,
                prob_creak=p_creak,
                prob_breath=p_breath,
            )

        processed_count += len(speakers)

        # Infrequent milestone logging
        if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == total_batches:
            elapsed = time.time() - t_start
            speed = processed_count / max(1e-5, elapsed)
            remaining_batches = total_batches - (batch_idx + 1)
            eta_sec = (remaining_batches * batch_size) / speed
            logger.info(
                f"Progress: [{batch_idx + 1}/{total_batches}] batches | "
                f"{processed_count}/{len(to_process)} files ({(batch_idx + 1)/total_batches*100:.1f}%) | "
                f"Speed: {speed:.1f} files/s | ETA: {eta_sec/60:.1f} min",
                extra={"flush": True}
            )

    elapsed_total = time.time() - t_start
    logger.info(f"Inference complete in {elapsed_total/60:.2f} minutes ({processed_count / elapsed_total:.1f} files/s).")

    # Build summary index
    logger.info("Building summary manifest across all files...")
    build_summary_index(output_dir, all_wavs, best_thresh)


def build_summary_index(output_dir: Path, all_wavs: List[Path], threshold: float = OPTIMAL_THRESHOLD):
    """Aggregate per-file summary statistics into summary.parquet and summary.csv."""
    t0 = time.time()
    records = []

    for idx, p in enumerate(all_wavs):
        spk = p.parent.name
        k = p.stem
        npz_p = output_dir / spk / f"{k}.npz"
        if not npz_p.exists():
            continue

        try:
            with np.load(npz_p) as data:
                p_creak = data["prob_creak"]
                p_breath = data["prob_breath"]
                ts = data["timestamps"]

            n_frames = len(p_creak)
            duration_s = float(ts[-1]) if n_frames > 0 else 0.0
            creak_count = int(np.sum(p_creak >= threshold))
            creak_ratio = creak_count / max(1, n_frames)

            records.append({
                "speaker": spk,
                "file_key": k,
                "duration_s": round(duration_s, 3),
                "num_frames": n_frames,
                "mean_creak_prob": round(float(np.mean(p_creak)), 4),
                "max_creak_prob": round(float(np.max(p_creak)), 4),
                "creak_frames_056": creak_count,
                "creak_ratio_056": round(creak_ratio, 4),
                "mean_breath_prob": round(float(np.mean(p_breath)), 4),
                "max_breath_prob": round(float(np.max(p_breath)), 4),
            })
        except Exception as e:
            logger.warning(f"Error reading {npz_p}: {e}")

    df = pd.DataFrame(records)
    parquet_path = output_dir / "summary.parquet"
    csv_path = output_dir / "summary.csv"

    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved summary index with {len(df)} records in {time.time()-t0:.2f}s:")
    logger.info(f"  - {parquet_path}")
    logger.info(f"  - {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="High-throughput frame inference on zt/v2")
    parser.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--force", action="store_true", help="Force recomputation of existing files")
    args = parser.parse_args()

    run_inference(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        ckpt_path=Path(args.checkpoint),
        batch_size=args.batch_size,
        device_str=args.device,
        num_workers=args.num_workers,
        force_recompute=args.force,
    )


if __name__ == "__main__":
    main()
