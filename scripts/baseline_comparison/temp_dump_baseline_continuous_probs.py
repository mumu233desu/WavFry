"""
Temporary Frame-Level Probability Dumper for Baseline Re-Evaluation.

Dumps continuous frame/window creak probabilities into reusable compressed .npz files
for CreaPy (10ms hop) and DeepFry (5ms hop) across the 77 canonical test audio files
and 5 acoustic degradation tiers (clean, mild_15db, moderate_5db, severe_0db, compound).

Usage:
    uv run python scripts/baseline_comparison/temp_dump_baseline_continuous_probs.py
    uv run python scripts/baseline_comparison/temp_dump_baseline_continuous_probs.py --limit-files 2 --tiers clean
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress noisy third-party warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import soundfile as sf
import torch

# Base directories
BASELINES_DIR_PRIMARY = Path(r"E:\Coding\WavFry\data\experiments\baselines")
if BASELINES_DIR_PRIMARY.exists():
    BASELINES_DIR = BASELINES_DIR_PRIMARY
    ROOT_DIR = Path(r"E:\Coding\WavFry")
else:
    SCRIPT_DIR = Path(__file__).resolve().parent
    CODE_DIR = SCRIPT_DIR.parent.parent
    ROOT_DIR = CODE_DIR.parent if (CODE_DIR.parent / "data").exists() else CODE_DIR
    BASELINES_DIR = ROOT_DIR / "data" / "experiments" / "baselines"

CODE_DIR = Path(r"E:\Coding\WavFry\code") if (Path(r"E:\Coding\WavFry\code")).exists() else Path(__file__).resolve().parent.parent.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from wavfry.eval.unified import build_test_manifest

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("temp_dump_continuous_probs")

CANONICAL_TIERS = ["clean", "mild_15db", "moderate_5db", "severe_0db", "compound"]
CREAPY_DIR = BASELINES_DIR / "creapy-main"
DEEPFRY_DIR = BASELINES_DIR / "DeepFry"
DEEPFRY_CKPT = (
    DEEPFRY_DIR
    / "models"
    / "CREAK-220lr_0.001_decay_21_input_size_512_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_.pth"
)
WAVFRY_CKPT = CODE_DIR / "exp_results" / "checkpoints" / "ablation_backbone_wavlm_baseplus" / "best_model.pt"
DEFAULT_STAGED_DIR = CODE_DIR / "data" / "experiments" / "cache" / "staged_audio"
DEFAULT_OUTPUT_CACHE = ROOT_DIR / "data" / "experiments" / "cache" / "temp_frame_probabilities"
KANE_SPANS_DIR = CODE_DIR / "data" / "experiments" / "cache" / "baseline_spans" / "kane"
if not KANE_SPANS_DIR.exists():
    KANE_SPANS_DIR = ROOT_DIR / "data" / "experiments" / "cache" / "baseline_spans" / "kane"


# ---------------------------------------------------------------------------
# CreaPy Extractor
# ---------------------------------------------------------------------------
def run_creapy_single_file(
    wav_path: Path, creapy_dir: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs CreaPy on a 16kHz mono audio file.

    Returns:
        timestamps: (N,) float32 array of frame centers in seconds.
        prob_creak: (N,) float32 array of continuous creak posterior in [0, 1].
    """
    if str(creapy_dir) not in sys.path:
        sys.path.insert(0, str(creapy_dir))

    import creapy
    from creapy.model.classify import process_file
    from creapy.utils.helpers import get_time_vector

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = process_file(str(wav_path))

    y_pred = res[1] if isinstance(res, tuple) and len(res) >= 2 else None
    if y_pred is None or len(y_pred) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # CreaPy hop_size = 10ms, block_size = 40ms.
    # get_time_vector returns block centers: N/2 + t0 + i*R
    t = get_time_vector(y_pred, 16000, 0.0)
    p = np.clip(y_pred.astype(np.float32), 0.0, 1.0)
    return t.astype(np.float32), p


# ---------------------------------------------------------------------------
# DeepFry Extractor
# ---------------------------------------------------------------------------
def load_deepfry_model(device: torch.device, deepfry_dir: Path, ckpt_path: Path):
    """Loads pre-trained DeepFry PyTorch model."""
    if str(deepfry_dir) not in sys.path:
        sys.path.insert(0, str(deepfry_dir))
    from models import load_model
    import argparse as ap

    parser = ap.ArgumentParser()
    parser.add_argument("--enc_hidden", type=int, default=128)
    args = parser.parse_args([])
    model, _ = load_model(str(ckpt_path), args, device=str(device))
    model = model.to(device).eval()
    return model


def run_deepfry_single_file(
    model, wav_path: Path, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs DeepFry on a 16kHz mono audio file.

    Returns:
        timestamps: (N,) float32 array of frame centers in seconds (stride = 5ms).
        prob_creak: (N,) float32 array of creak posterior in [0, 1].
    """
    audio_data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr}Hz for {wav_path}")

    # Process in chunks if longer than 30s to be memory safe, though 1D CNN is very light
    chunk_samples = 480000  # 30s
    p_creak_chunks = []
    
    with torch.no_grad():
        for start_idx in range(0, len(audio_data), chunk_samples):
            ch = audio_data[start_idx : start_idx + chunk_samples]
            if len(ch) < 512:
                continue
            x = torch.from_numpy(ch).unsqueeze(0).unsqueeze(0).to(device)
            _, probs = model(x)
            # Channel 0: voicing, Channel 1: creaky
            p_ch = probs[:, 1].squeeze(0).cpu().numpy()
            p_creak_chunks.append(p_ch)

    if not p_creak_chunks:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    p_creak = np.concatenate(p_creak_chunks).astype(np.float32)
    n_frames = len(p_creak)
    # Stride = 80 samples (5ms), effective receptive field center ~ 40 samples (2.5ms)
    t = (np.arange(n_frames, dtype=np.float32) * 80.0 + 40.0) / 16000.0
    return t, p_creak


# ---------------------------------------------------------------------------
# WavFry Extractor
# ---------------------------------------------------------------------------
def load_wavfry_model(device: torch.device, ckpt_path: Path):
    """Loads pre-trained WavFry PyTorch model."""
    from wavfry.eval.unified import load_model_from_checkpoint
    from wavfry.model.wavlm_classifier import WavFryClassifier

    model, ckpt_info = load_model_from_checkpoint(ckpt_path, device, WavFryClassifier)
    return model


def run_wavfry_single_file(
    model, wav_path: Path, device: torch.device, task: str = "creaky"
) -> Tuple[np.ndarray, np.ndarray]:
    """Runs WavFry on a 16kHz mono audio file via frame-level head.

    Returns:
        timestamps: (N,) float32 array of frame centers in seconds (stride = 20ms).
        prob_creak: (N,) float32 array of continuous posterior in [0, 1].
    """
    audio_data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    if sr != 16000:
        raise ValueError(f"Expected 16kHz audio, got {sr}Hz for {wav_path}")

    x = torch.from_numpy(audio_data).unsqueeze(0).to(device)
    lengths = torch.tensor([len(audio_data)], device=device)
    with torch.no_grad():
        frames = model(x, lengths=lengths, return_frames=True)
        frame_logits = frames["frame_logits"][task].squeeze(0)
        p_creak = torch.sigmoid(frame_logits).cpu().numpy().astype(np.float32)

    n_frames = len(p_creak)
    stride_samples = frames.get("frame_stride_samples", 320)
    receptive_samples = frames.get("frame_receptive_samples", 400)
    t = (np.arange(n_frames, dtype=np.float32) * stride_samples + receptive_samples / 2.0) / 16000.0
    return t, p_creak


# ---------------------------------------------------------------------------
# Praat Extractor (Parselmouth with Speech Intensity Gating)
# ---------------------------------------------------------------------------
def run_praat_single_file(wav_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Runs Praat pitch perturbation baseline on 16kHz audio with intensity silence gating.

    Mitigates the critical silence pitfall by suppressing false alarms in pauses/silence
    where standard Praat pitch tracking trivially fails (frequency == 0).

    Returns:
        timestamps: (N,) float32 array of frame centers in seconds (stride = 15ms).
        prob_creak: (N,) float32 array of creak posterior in [0, 1].
    """
    import parselmouth

    sound = parselmouth.Sound(str(wav_path))
    pitch = sound.to_pitch(time_step=0.015, pitch_floor=50.0, pitch_ceiling=350.0)
    intensity = sound.to_intensity(time_step=0.015, minimum_pitch=50.0)

    xs = np.array(pitch.xs(), dtype=np.float32)
    freqs = np.array(pitch.selected_array["frequency"])

    if len(xs) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Intensity gating: filter out non-speech silence
    ivals = np.array([intensity.get_value(float(t)) for t in xs], dtype=np.float32)
    ivals = np.nan_to_num(ivals, nan=0.0)
    max_i = float(np.max(ivals)) if len(ivals) > 0 else 0.0
    # Standard Praat speech silence threshold: max - 30 dB, and absolute >= 35 dB
    voiced_mask = (ivals >= (max_i - 30.0)) & (ivals >= 35.0)

    # Creak decision: active speech frames where pitch tracker fails (unvoiced/subharmonic)
    prob_creak = np.where(voiced_mask & (freqs == 0.0), 1.0, 0.0).astype(np.float32)
    return xs, prob_creak


# ---------------------------------------------------------------------------
# Kane Extractor (Rasterized from Pre-computed Covarep ANN Decision Spans)
# ---------------------------------------------------------------------------
def run_kane_single_file(
    fkey: str, tier: str, wav_path: Path, kane_spans_cache: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterizes Kane (2013 / Covarep) official ANN decision spans to 10ms frame grid.

    Re-uses official ANN classifications (with 3-tap median filtering and tau=0.3 threshold)
    to achieve 100% faithful alignment with Table 1 baseline extraction without redundant
    Octave execution overhead.

    Returns:
        timestamps: (N,) float32 array of frame centers in seconds (stride = 10ms).
        prob_creak: (N,) float32 array of binary creak decisions in {0.0, 1.0}.
    """
    import json
    info = sf.info(str(wav_path))
    dur = float(info.duration)
    # 10ms frame grid matching COVAREP detect_creaky_voice frame centers (t = (1:N)*0.010)
    timestamps = np.arange(0.010, dur, 0.010, dtype=np.float32)
    prob_creak = np.zeros_like(timestamps, dtype=np.float32)

    span_json = kane_spans_cache / tier / f"{fkey}.json"
    if span_json.exists():
        try:
            with open(span_json, "r", encoding="utf-8") as f:
                spans = json.load(f)
            for s, e in spans:
                prob_creak[(timestamps >= float(s) - 1e-4) & (timestamps <= float(e) + 1e-4)] = 1.0
        except Exception as e:
            logger.warning(f"Error reading Kane span for {fkey} ({tier}): {e}")
    else:
        logger.warning(f"Kane span JSON missing: {span_json}")

    return timestamps, prob_creak


# ---------------------------------------------------------------------------
# Main Dumping Routine
# ---------------------------------------------------------------------------
def dump_baseline_probabilities(
    manifest_path: Path,
    staged_audio_dir: Path,
    output_cache_dir: Path,
    models: List[str],
    tiers: List[str],
    limit_files: Optional[int] = None,
    device_str: str = "cuda",
    overwrite: bool = False,
):
    output_cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    logger.info(f"Using device: {device}")

    # Build canonical test manifest
    full_manifest = pd.read_csv(manifest_path, low_memory=False)
    test_manifest = build_test_manifest(full_manifest, task="creaky")
    
    unique_files = test_manifest.drop_duplicates(subset=["file_key"])
    if limit_files is not None and limit_files > 0:
        unique_files = unique_files.iloc[:limit_files].copy()
    
    file_keys = unique_files["file_key"].tolist()
    logger.info(f"Total unique test audio files to process: {len(file_keys)}")

    # Load DeepFry if needed
    deepfry_model = None
    if "DeepFry" in models:
        logger.info("Loading DeepFry model checkpoint...")
        deepfry_model = load_deepfry_model(device, DEEPFRY_DIR, DEEPFRY_CKPT)

    # Load WavFry if needed
    wavfry_model = None
    if "WavFry" in models:
        logger.info(f"Loading WavFry model checkpoint from {WAVFRY_CKPT}...")
        wavfry_model = load_wavfry_model(device, WAVFRY_CKPT)

    total_ops = len(models) * len(tiers) * len(file_keys)
    done_ops = 0
    t0 = time.time()

    for tier in tiers:
        tier_staged = staged_audio_dir / tier
        if not tier_staged.exists():
            logger.warning(f"Tier staged directory missing: {tier_staged}. Skipping.")
            continue

        for model_name in models:
            model_tier_cache = output_cache_dir / model_name / tier
            model_tier_cache.mkdir(parents=True, exist_ok=True)

            logger.info(f"=== Dumping [{model_name}] on [{tier}] ({len(file_keys)} files) ===")

            for i, fkey in enumerate(file_keys, 1):
                out_npz = model_tier_cache / f"{fkey}.npz"
                if not overwrite and out_npz.exists() and out_npz.stat().st_size > 100:
                    done_ops += 1
                    continue

                wav_p = tier_staged / f"{fkey}.wav"
                if not wav_p.exists():
                    logger.warning(f"Missing audio: {wav_p}")
                    done_ops += 1
                    continue

                try:
                    if model_name == "CreaPy":
                        t_arr, p_arr = run_creapy_single_file(wav_p, CREAPY_DIR)
                    elif model_name == "DeepFry":
                        t_arr, p_arr = run_deepfry_single_file(deepfry_model, wav_p, device)
                    elif model_name == "WavFry":
                        t_arr, p_arr = run_wavfry_single_file(wavfry_model, wav_p, device, task="creaky")
                    elif model_name == "Praat":
                        t_arr, p_arr = run_praat_single_file(wav_p)
                    elif model_name == "Kane":
                        t_arr, p_arr = run_kane_single_file(fkey, tier, wav_p, KANE_SPANS_DIR)
                    else:
                        raise ValueError(f"Unknown model: {model_name}")

                    # Save compressed npz
                    np.savez_compressed(
                        out_npz,
                        timestamps=t_arr,
                        prob_creak=p_arr,
                    )
                except Exception as e:
                    logger.error(f"Failed [{model_name}] on {fkey} ({tier}): {e}", exc_info=True)

                done_ops += 1
                if i % 10 == 0 or i == len(file_keys):
                    elapsed = time.time() - t0
                    logger.info(
                        f"[{model_name}][{tier}] Processed {i}/{len(file_keys)} files | "
                        f"Overall: {done_ops}/{total_ops} ({done_ops/total_ops*100:.1f}%) | "
                        f"Elapsed: {elapsed:.1f}s"
                    )

    logger.info("=== Frame probability dump complete! ===")


def main():
    parser = argparse.ArgumentParser(description="Dump baseline continuous frame probabilities.")
    parser.add_argument("--manifest", type=str, default="manifests/segments_multi.csv")
    parser.add_argument("--staged-dir", type=str, default=str(DEFAULT_STAGED_DIR))
    parser.add_argument("--output-cache", type=str, default=str(DEFAULT_OUTPUT_CACHE))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["CreaPy", "DeepFry", "WavFry", "Praat", "Kane"],
        choices=["CreaPy", "DeepFry", "WavFry", "Praat", "Kane"],
    )
    parser.add_argument("--tiers", nargs="+", default=CANONICAL_TIERS, choices=CANONICAL_TIERS)
    parser.add_argument("--limit-files", type=int, default=None, help="Limit number of files for smoke testing.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cached npz files.")

    args = parser.parse_args()

    # Resolve manifest path relative to code dir
    manifest_p = Path(args.manifest)
    if not manifest_p.is_absolute():
        manifest_p = CODE_DIR / manifest_p

    dump_baseline_probabilities(
        manifest_path=manifest_p,
        staged_audio_dir=Path(args.staged_dir),
        output_cache_dir=Path(args.output_cache),
        models=args.models,
        tiers=args.tiers,
        limit_files=args.limit_files,
        device_str=args.device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
