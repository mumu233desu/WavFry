"""
Glottal Physical Interpretability & External EGG Validity Audit (AVID Corpus).

Canonical Data-Producing Script:
- Measures statistical correlations between WavFry predicted probabilities
  (Breathy Voice and Creaky Voice) and ground-truth electroglottographic (EGG)
  glottal physical metrics extracted on the AVID corpus:
  * Open Quotient (Oq): Glottal open phase duration relative to period (Henrich et al. 2004 DECOM)
  * Contact Quotient (CQ = 1 - Oq): Glottal contact phase duration
- Dual-Scale Evaluation:
  1. Microscopic Frame-Level: Continuous time-series correlation on voiced EGG frames.
  2. Macroscopic Utterance-Level: Stratification across acoustic intensity conditions
     (Soft, Normal, Loud, Very Loud) and population-wide speaker consistency (All 50 AVID speakers).
- Direct Continuous Long-Recording Support:
  Directly extracts and evaluates from the original continuous 44.1kHz dual-channel recordings
  (E:\\Corpus\\EGG_RAW\\AVID\\Repository 1) or pre-sliced flac segments.
  Automatically skips initial 94dB calibration tones, uses chunked WavFry inference to
  prevent GPU OOM, and utilizes parallel Octave DECOM workers.
- Decoupled Pipeline Architecture: Outputs ONLY structured data (JSON, CSV, MD).
  Visualization is handled by plot_egg_correlations.py.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import scipy.signal as signal
import soundfile as sf
import torch
import torchaudio

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.eval.unified import load_model_from_checkpoint
from wavfry.model.wavlm_classifier import WavFryClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_egg_correlations")

# Canonical default paths
DEFAULT_AVID_DIR = Path(r"E:\Corpus\EGG_RAW\AVID\Repository 1")
FALLBACK_AVID_DIR = Path(r"E:\Corpus\EGG\AVID")
DEFAULT_COVAREP_DIR = Path(r"E:\Coding\EGG_Improve_PE\knowledge_base\raw_materials\covarep\glottalsource\egg\decom")
DEFAULT_OCTAVE_BIN = Path(r"D:\Tools\octave-11.3.0-w64\mingw64\bin\octave-cli.exe")

CANDIDATE_CHECKPOINTS = [
    PROJECT_ROOT / "exp_results/checkpoints/frozen_creaky_breathy_block/best_model.pt",
    PROJECT_ROOT / "exp_results/checkpoints/ablation_backbone_wavlm_baseplus/latest.pt",
    PROJECT_ROOT / "exp_results/checkpoints/csl_frozen_wavlm_base_plus/best_model.pt",
]

# Official AVID speaker gender ground-truth (from AVID/Repository 2/Metadata_with_labels_SENT.csv)
AVID_SPEAKER_GENDER: Dict[str, str] = {
    "Spk1": "F", "Spk2": "M", "Spk3": "M", "Spk4": "M", "Spk5": "M",
    "Spk6": "M", "Spk7": "M", "Spk8": "M", "Spk9": "F", "Spk10": "M",
    "Spk11": "F", "Spk12": "F", "Spk13": "F", "Spk14": "F", "Spk15": "M",
    "Spk16": "M", "Spk17": "M", "Spk18": "F", "Spk19": "M", "Spk20": "F",
    "Spk21": "F", "Spk22": "M", "Spk23": "F", "Spk24": "F", "Spk25": "F",
    "Spk26": "F", "Spk27": "M", "Spk28": "M", "Spk29": "F", "Spk30": "M",
    "Spk31": "F", "Spk32": "M", "Spk33": "F", "Spk34": "M", "Spk35": "F",
    "Spk36": "F", "Spk37": "M", "Spk38": "M", "Spk39": "F", "Spk40": "F",
    "Spk41": "M", "Spk42": "F", "Spk43": "M", "Spk44": "M", "Spk45": "F",
    "Spk46": "F", "Spk47": "F", "Spk48": "F", "Spk49": "M", "Spk50": "M",
}


def resolve_octave(custom_path: Optional[str] = None) -> Path:
    if custom_path and Path(custom_path).is_file():
        return Path(custom_path)
    if DEFAULT_OCTAVE_BIN.is_file():
        return DEFAULT_OCTAVE_BIN
    which_bin = shutil.which("octave-cli") or shutil.which("octave")
    if which_bin:
        return Path(which_bin)
    raise FileNotFoundError(
        "GNU Octave executable not found. Please install Octave or specify --octave-bin."
    )


def resolve_checkpoint(custom_path: Optional[str] = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p
        raise FileNotFoundError(f"Specified checkpoint not found: {custom_path}")

    for p in CANDIDATE_CHECKPOINTS:
        if p.is_file():
            return p

    raise FileNotFoundError(
        "No default checkpoint found in candidate paths. Please provide --checkpoint."
    )


def parse_avid_filename(filename: str) -> Dict[str, str]:
    """Parse AVID file metadata from standard or raw filename."""
    stem = Path(filename).stem
    # Match SpkXX
    m = re.search(r"(Spk\d+)", stem)
    spk_id = m.group(1) if m else "Unknown"
    gender = AVID_SPEAKER_GENDER.get(spk_id, "Unknown")
    return {
        "file_key": stem,
        "speaker_id": spk_id,
        "gender": gender,
    }


def extract_egg_decom(
    egg_signal: np.ndarray,
    fs: int,
    octave_bin: Path,
    covarep_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract Open Quotient (Oq), F0, and timestamps from EGG signal using
    Henrich et al. (2004) DECOM algorithm via Octave.
    Returns: (timestamps, f0, oq)
    """
    # 1. High-pass filter at 30 Hz (2nd order Butterworth) to remove macro laryngeal drift
    b, a = signal.butter(2, 30.0 / (fs / 2.0), btype="high")
    egg_hp = signal.filtfilt(b, a, egg_signal)

    # 2. First derivative (dEGG)
    degg = np.diff(egg_hp)
    degg = np.append(degg, degg[-1])
    max_val = np.max(np.abs(degg))
    if max_val > 0:
        degg = degg / max_val

    # 3. Call Octave decom_sig
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_wav = Path(tmpdir) / "degg.wav"
        out_csv = Path(tmpdir) / "out.csv"
        sf.write(str(tmp_wav), degg.astype(np.float32), int(fs))

        octave_script = f"""
        pkg load signal;
        addpath('{covarep_dir.as_posix()}');
        addpath('{covarep_dir.as_posix()}/private');
        [s, fs] = audioread('{tmp_wav.as_posix()}');
        [Oq, f0, npicf, npico, atimes] = decom_sig(s, fs);
        res = [atimes(:), f0(:), Oq(:)];
        csvwrite('{out_csv.as_posix()}', res);
        """
        oct_m = Path(tmpdir) / "run_decom.m"
        oct_m.write_text(octave_script, encoding="utf-8")

        cmd = [str(octave_bin), "--no-gui", "--no-window-system", str(oct_m)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning(f"Octave decom_sig failed: {proc.stderr}")
            return np.array([]), np.array([]), np.array([])

        if not out_csv.exists() or out_csv.stat().st_size == 0:
            return np.array([]), np.array([]), np.array([])

        try:
            res = np.loadtxt(str(out_csv), delimiter=",")
            if res.ndim == 1:
                res = res.reshape(1, -1)
            times = res[:, 0]
            f0 = res[:, 1]
            oq = res[:, 2]
            return times, f0, oq
        except Exception as e:
            logger.warning(f"Failed to parse Octave DECOM output: {e}")
            return np.array([]), np.array([]), np.array([])


def run_wavfry_frames_long(
    model: WavFryClassifier,
    speech_waveform: np.ndarray,
    orig_sr: int,
    device: torch.device,
    chunk_sec: float = 30.0,
    time_offset: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Run continuous frame-level inference on 16kHz speech in blocks to prevent CUDA OOM."""
    speech_tensor = torch.from_numpy(speech_waveform).float().unsqueeze(0)
    if orig_sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=16000)
        speech_16k = resampler(speech_tensor)[0]
    else:
        speech_16k = speech_tensor[0]

    chunk_samples = int(chunk_sec * 16000)
    chunks = [speech_16k[i : i + chunk_samples] for i in range(0, len(speech_16k), chunk_samples)]

    all_logits: Dict[str, List[np.ndarray]] = {t: [] for t in model.tasks}
    with torch.no_grad():
        for chunk in chunks:
            lens = torch.tensor([len(chunk)], device=device)
            out = model(chunk.unsqueeze(0).to(device), lengths=lens, return_frames=True)
            for t in model.tasks:
                all_logits[t].append(out["frame_logits"][t][0].cpu().numpy())

    probs_dict = {}
    for t in model.tasks:
        cat_logits = np.concatenate(all_logits[t])
        probs_dict[t] = 1.0 / (1.0 + np.exp(-cat_logits))

    first_task = model.tasks[0]
    n_frames = len(probs_dict[first_task])
    # 20ms stride = 320 samples at 16kHz
    times = (np.arange(n_frames) * 320.0 + 160.0) / 16000.0 + time_offset
    return times, probs_dict


def extract_single_audio_egg(
    audio_file: Path,
    octave_bin: Path,
    covarep_dir: Path,
    duration_limit: Optional[float] = None,
    skip_lead_s: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """Extract EGG metrics for a single audio file in a thread-safe manner."""
    try:
        info = sf.info(str(audio_file))
        if info.channels < 2:
            return None

        fs = info.samplerate
        is_long = info.duration > 30.0
        lead_s = skip_lead_s if is_long else 0.0
        start_frame = int(lead_s * fs)
        frames_to_read = int(duration_limit * fs) if duration_limit is not None else -1

        data, _ = sf.read(str(audio_file), start=start_frame, frames=frames_to_read)
        if data.ndim < 2 or data.shape[1] < 2:
            return None

        speech_audio = data[:, 0]
        egg_audio = data[:, 1]
        speech_rms = float(np.sqrt(np.mean(speech_audio**2)))

        t_egg, f0_egg, oq_egg = extract_egg_decom(egg_audio, fs, octave_bin, covarep_dir)
        if len(t_egg) == 0:
            return None

        t_egg = t_egg + lead_s
        valid_mask = (f0_egg > 50.0) & (f0_egg < 800.0) & (oq_egg >= 0.20) & (oq_egg <= 0.95)
        if np.sum(valid_mask) < 10:
            return None

        return {
            "audio_file": audio_file,
            "speech_audio": speech_audio,
            "fs": fs,
            "speech_rms": speech_rms,
            "lead_s": lead_s,
            "t_v": t_egg[valid_mask],
            "f0_v": f0_egg[valid_mask],
            "oq_v": oq_egg[valid_mask],
        }
    except Exception as e:
        logger.warning(f"Error extracting EGG for {audio_file.name}: {e}")
        return None


def evaluate_egg_correlations(
    avid_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    covarep_dir: Path,
    octave_bin: Path,
    speakers: Optional[List[str]] = None,
    all_speakers: bool = False,
    duration_per_speaker: Optional[float] = None,
    limit_files: Optional[int] = None,
    num_workers: int = 4,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)
    logger.info(f"Loading WavFry model from {checkpoint_path} on {device}...")
    model, config = load_model_from_checkpoint(checkpoint_path, device, WavFryClassifier)
    model.eval()

    logger.info(f"Model tasks: {model.tasks}")

    # Gather audio files (supports .wav and .flac)
    all_audio_files = sorted(list(avid_dir.glob("*.wav")) + list(avid_dir.glob("*.flac")))
    if not all_audio_files and avid_dir != FALLBACK_AVID_DIR and FALLBACK_AVID_DIR.exists():
        logger.info(f"No audio files found in {avid_dir}, falling back to {FALLBACK_AVID_DIR}...")
        avid_dir = FALLBACK_AVID_DIR
        all_audio_files = sorted(list(avid_dir.glob("*.wav")) + list(avid_dir.glob("*.flac")))

    if not all_audio_files:
        raise FileNotFoundError(f"No .wav or .flac files found in directory: {avid_dir}")

    # Target speakers
    target_speakers = set(speakers) if speakers else None

    filtered_files = []
    seen_spks = set()
    for f in all_audio_files:
        meta = parse_avid_filename(f.name)
        spk = meta["speaker_id"]
        if target_speakers is not None and spk not in target_speakers:
            continue
        # If long recording directory, ensure 1 primary file per speaker
        if "Repository 1" in str(avid_dir):
            if spk in seen_spks:
                continue
            seen_spks.add(spk)
        filtered_files.append(f)

    if limit_files is not None and limit_files > 0:
        filtered_files = filtered_files[:limit_files]

    logger.info(
        f"Evaluating {len(filtered_files)} audio files across {len(set(parse_avid_filename(f.name)['speaker_id'] for f in filtered_files))} speakers."
    )

    # Phase 1: Identify and extract uncached files
    uncached_files = []
    for f in filtered_files:
        meta = parse_avid_filename(f.name)
        cache_file = cache_dir / f"{meta['file_key']}.npz"
        if not cache_file.exists():
            uncached_files.append(f)

    if uncached_files:
        logger.info(
            f"Found {len(uncached_files)} uncached files. Extracting EGG & running inference with {num_workers} parallel workers..."
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_file = {
                executor.submit(
                    extract_single_audio_egg, f, octave_bin, covarep_dir, duration_per_speaker
                ): f
                for f in uncached_files
            }
            done_count = 0
            for future in as_completed(future_to_file):
                done_count += 1
                egg_res = future.result()
                if egg_res is None:
                    continue
                f_file = egg_res["audio_file"]
                f_meta = parse_avid_filename(f_file.name)
                cache_file = cache_dir / f"{f_meta['file_key']}.npz"

                t_model, probs_dict = run_wavfry_frames_long(
                    model, egg_res["speech_audio"], egg_res["fs"], device, time_offset=egg_res["lead_s"]
                )
                probs_cached = {}
                for task in model.tasks:
                    probs_cached[task] = np.interp(egg_res["t_v"], t_model, probs_dict[task])

                save_dict = {
                    "t_v": egg_res["t_v"],
                    "oq_v": egg_res["oq_v"],
                    "f0_v": egg_res["f0_v"],
                    "speech_rms": egg_res["speech_rms"],
                    **{f"p_{t}": probs_cached[t] for t in model.tasks},
                }
                np.savez_compressed(cache_file, **save_dict)
                if done_count % 5 == 0 or done_count == len(uncached_files):
                    logger.info(f"Processed and cached [{done_count}/{len(uncached_files)}] audio files...")

    # Phase 2: Load all cached files and compile records
    all_frame_records = []
    utterance_records = []

    for audio_file in filtered_files:
        file_meta = parse_avid_filename(audio_file.name)
        file_key = file_meta["file_key"]
        spk_id = file_meta["speaker_id"]
        spk_gender = file_meta["gender"]
        cache_file = cache_dir / f"{file_key}.npz"

        if not cache_file.exists():
            continue

        try:
            cached = np.load(cache_file)
            t_v = cached["t_v"]
            oq_v = cached["oq_v"]
            f0_v = cached["f0_v"]
            probs_cached = {t: cached[f"p_{t}"] for t in model.tasks if f"p_{t}" in cached}
            speech_rms = float(cached["speech_rms"])
        except Exception:
            continue

        cq_v = 1.0 - oq_v

        # Collect frame records
        for i in range(len(t_v)):
            rec = {
                "file_key": file_key,
                "speaker_id": spk_id,
                "gender": spk_gender,
                "time_s": round(float(t_v[i]), 4),
                "f0_hz": round(float(f0_v[i]), 2),
                "egg_oq": round(float(oq_v[i]), 4),
                "egg_cq": round(float(cq_v[i]), 4),
            }
            for t in model.tasks:
                rec[f"prob_{t}"] = round(float(probs_cached[t][i]), 4)
            all_frame_records.append(rec)

        # Collect utterance-level record
        utt_rec = {
            "file_key": file_key,
            "speaker_id": spk_id,
            "gender": spk_gender,
            "voiced_frames": len(t_v),
            "speech_rms": speech_rms,
            "speech_rms_db": round(20.0 * np.log10(max(speech_rms, 1e-6)), 2),
            "mean_f0_hz": round(float(np.mean(f0_v)), 2),
            "mean_egg_oq": round(float(np.mean(oq_v)), 4),
            "mean_egg_cq": round(float(np.mean(cq_v)), 4),
        }
        for t in model.tasks:
            utt_rec[f"mean_prob_{t}"] = round(float(np.mean(probs_cached[t])), 4)
        utterance_records.append(utt_rec)

    frame_df = pd.DataFrame(all_frame_records)
    utterance_df = pd.DataFrame(utterance_records)

    logger.info(f"Total voiced frame samples: {len(frame_df)}")
    logger.info(f"Total valid utterance samples: {len(utterance_df)}")

    # Assign intensity quartiles based on Speech RMS dB
    if len(utterance_df) >= 4:
        utterance_df["intensity_tier"] = pd.qcut(
            utterance_df["speech_rms_db"],
            q=4,
            labels=["Soft", "Normal", "Loud", "Very_Loud"],
        )

    # 4. Statistical Correlations
    metrics_summary: Dict[str, Any] = {
        "dataset": "AVID (Speech & EGG Dual-Channel Corpus)",
        "algorithm": "Henrich et al. (2004) DECOM (Octave + COVAREP)",
        "total_files": len(utterance_df),
        "total_voiced_frames": len(frame_df),
        "evaluated_speakers": int(utterance_df["speaker_id"].nunique()),
        "correlations": {},
    }

    # Microscopic frame-level statistics
    frame_corrs = []
    if "prob_breathy" in frame_df.columns:
        p_b = frame_df["prob_breathy"].values
        oq = frame_df["egg_oq"].values
        r_b, p_rb = pearsonr(p_b, oq)
        rho_b, p_rhob = spearmanr(p_b, oq)
        frame_corrs.append({
            "Target": "Breathy vs. EGG Oq (Open Quotient)",
            "Hypothesis": "Positive (Glottal gap / incomplete closure)",
            "Pearson_r": round(float(r_b), 4),
            "Pearson_p": float(p_rb),
            "Spearman_rho": round(float(rho_b), 4),
            "Spearman_p": float(p_rhob),
            "N_samples": len(frame_df),
        })

    if "prob_creaky" in frame_df.columns:
        p_c = frame_df["prob_creaky"].values
        cq = frame_df["egg_cq"].values
        oq = frame_df["egg_oq"].values
        r_c_cq, p_rc_cq = pearsonr(p_c, cq)
        rho_c_cq, p_rhoc_cq = spearmanr(p_c, cq)
        r_c_oq, p_rc_oq = pearsonr(p_c, oq)
        rho_c_oq, p_rhoc_oq = spearmanr(p_c, oq)
        frame_corrs.append({
            "Target": "Creaky vs. EGG Oq (Open Quotient)",
            "Hypothesis": "Negative (Opposing Glottal Pole / Low Open Quotient)",
            "Pearson_r": round(float(r_c_oq), 4),
            "Pearson_p": float(p_rc_oq),
            "Spearman_rho": round(float(rho_c_oq), 4),
            "Spearman_p": float(p_rhoc_oq),
            "N_samples": len(frame_df),
        })
        frame_corrs.append({
            "Target": "Creaky vs. EGG CQ (Contact Quotient)",
            "Hypothesis": "Positive (Medial compression / prolonged closure)",
            "Pearson_r": round(float(r_c_cq), 4),
            "Pearson_p": float(p_rc_cq),
            "Spearman_rho": round(float(rho_c_cq), 4),
            "Spearman_p": float(p_rhoc_cq),
            "N_samples": len(frame_df),
        })

    metrics_summary["correlations"]["frame_level"] = frame_corrs

    # Per-speaker correlation distribution
    spk_records = []
    for spk, g in frame_df.groupby("speaker_id"):
        spk_gender = g["gender"].iloc[0] if "gender" in g.columns else "Unknown"
        n_v = len(g)
        rec = {
            "speaker_id": spk,
            "gender": spk_gender,
            "voiced_frames": n_v,
        }
        if "prob_breathy" in g.columns and n_v >= 20:
            rb, pb = pearsonr(g["prob_breathy"], g["egg_oq"])
            rhob, prhob = spearmanr(g["prob_breathy"], g["egg_oq"])
            rec["breathy_oq_r"] = round(float(rb), 4)
            rec["breathy_oq_p"] = float(pb)
            rec["breathy_oq_rho"] = round(float(rhob), 4)
        if "prob_creaky" in g.columns and n_v >= 20:
            rc_oq, pc_oq = pearsonr(g["prob_creaky"], g["egg_oq"])
            rc_cq, pc_cq = pearsonr(g["prob_creaky"], g["egg_cq"])
            rhoc_cq, prhoc_cq = spearmanr(g["prob_creaky"], g["egg_cq"])
            rec["creaky_oq_r"] = round(float(rc_oq), 4)
            rec["creaky_cq_r"] = round(float(rc_cq), 4)
            rec["creaky_cq_p"] = float(pc_cq)
            rec["creaky_cq_rho"] = round(float(rhoc_cq), 4)
        spk_records.append(rec)

    speaker_df = pd.DataFrame(spk_records)

    # Calculate speaker-level summary stats
    spk_stats = {}
    if "breathy_oq_r" in speaker_df.columns:
        b_r = speaker_df["breathy_oq_r"].dropna()
        q25, q75 = np.percentile(b_r, [25, 75])
        spk_stats["breathy_oq"] = {
            "mean_r": round(float(np.mean(b_r)), 4),
            "median_r": round(float(np.median(b_r)), 4),
            "iqr_r": round(float(q75 - q25), 4),
            "std_r": round(float(np.std(b_r)), 4),
            "positive_ratio": round(float(np.mean(b_r > 0)), 4),
            "total_speakers": len(b_r),
            "male_median_r": round(float(speaker_df[speaker_df["gender"] == "M"]["breathy_oq_r"].median()), 4),
            "female_median_r": round(float(speaker_df[speaker_df["gender"] == "F"]["breathy_oq_r"].median()), 4),
        }
    if "creaky_cq_r" in speaker_df.columns:
        c_r = speaker_df["creaky_cq_r"].dropna()
        q25, q75 = np.percentile(c_r, [25, 75])
        spk_stats["creaky_cq"] = {
            "mean_r": round(float(np.mean(c_r)), 4),
            "median_r": round(float(np.median(c_r)), 4),
            "iqr_r": round(float(q75 - q25), 4),
            "std_r": round(float(np.std(c_r)), 4),
            "positive_ratio": round(float(np.mean(c_r > 0)), 4),
            "total_speakers": len(c_r),
            "male_median_r": round(float(speaker_df[speaker_df["gender"] == "M"]["creaky_cq_r"].median()), 4),
            "female_median_r": round(float(speaker_df[speaker_df["gender"] == "F"]["creaky_cq_r"].median()), 4),
        }
    metrics_summary["correlations"]["speaker_level"] = spk_stats

    # Macroscopic condition statistics
    if "intensity_tier" in utterance_df.columns:
        grouped = (
            utterance_df.groupby("intensity_tier", observed=True)
            .agg({
                "mean_egg_oq": ["mean", "std"],
                "mean_egg_cq": ["mean", "std"],
                **{f"mean_prob_{t}": ["mean", "std"] for t in model.tasks},
            })
        )
        cond_stats = {}
        for (col, stat), tier_dict in grouped.to_dict().items():
            key = f"{col}_{stat}"
            cond_stats[key] = {
                str(tier): round(float(v), 4) if pd.notnull(v) and np.isfinite(v) else None
                for tier, v in tier_dict.items()
            }
        metrics_summary["condition_stratification"] = cond_stats

    # Export outputs
    frame_csv_path = output_dir / "egg_frame_correlations.csv"
    utt_csv_path = output_dir / "egg_utterance_stratification.csv"
    speaker_csv_path = output_dir / "egg_speaker_correlations.csv"
    json_path = output_dir / "table_egg_correlations_metrics.json"
    table_csv_path = output_dir / "table_egg_correlations.csv"
    md_path = output_dir / "table_egg_correlations.md"

    frame_df.to_csv(frame_csv_path, index=False, encoding="utf-8")
    utterance_df.to_csv(utt_csv_path, index=False, encoding="utf-8")
    speaker_df.to_csv(speaker_csv_path, index=False, encoding="utf-8")
    pd.DataFrame(frame_corrs).to_csv(table_csv_path, index=False, encoding="utf-8")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table: Glottal Physical Interpretability on Synchronized EGG (AVID Corpus)\n\n")
        f.write(f"- **Corpus**: `AVID` ({len(utterance_df)} audio files across {utterance_df['speaker_id'].nunique()} speakers)\n")
        f.write("- **Gold-Standard Ground Truth**: Electroglottographic (EGG) open quotient ($O_q$) via Henrich et al. (2004) DECOM\n")
        f.write(f"- **Voiced Physical Frames**: $N = {len(frame_df):,}$ frames\n\n")
        f.write("### 1. Microscopic Frame-Level Correlations\n\n")
        f.write("| Target Relation | Expected Physical Trend | Pearson $r$ | Pearson $p$ | Spearman $\\rho$ | Spearman $p$ | N Samples |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in frame_corrs:
            p_r_str = f"{r['Pearson_p']:.2e}" if r['Pearson_p'] < 0.001 else f"{r['Pearson_p']:.4f}"
            p_rho_str = f"{r['Spearman_p']:.2e}" if r['Spearman_p'] < 0.001 else f"{r['Spearman_p']:.4f}"
            f.write(
                f"| **{r['Target']}** | `{r['Hypothesis']}` | **{r['Pearson_r']:+.4f}** | {p_r_str} | "
                f"**{r['Spearman_rho']:+.4f}** | {p_rho_str} | {r['N_samples']:,} |\n"
            )
        f.write("\n### 2. Population-Wide Speaker Consistency (All Evaluated Speakers)\n\n")
        f.write("| Acoustic Dimension | Evaluated Speakers | Median Pearson $r$ | Mean Pearson $r$ | IQR | Male Median $r$ | Female Median $r$ | Win Rate ($r > 0$) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        if "breathy_oq" in spk_stats:
            sb = spk_stats["breathy_oq"]
            f.write(f"| **Breathy vs. EGG $O_q$** | {sb['total_speakers']} | **{sb['median_r']:+.3f}** | {sb['mean_r']:+.3f} | {sb['iqr_r']:.3f} | {sb['male_median_r']:+.3f} | {sb['female_median_r']:+.3f} | {sb['positive_ratio']*100:.1f}% |\n")
        if "creaky_cq" in spk_stats:
            sc = spk_stats["creaky_cq"]
            f.write(f"| **Creaky vs. EGG $CQ$** | {sc['total_speakers']} | **{sc['median_r']:+.3f}** | {sc['mean_r']:+.3f} | {sc['iqr_r']:.3f} | {sc['male_median_r']:+.3f} | {sc['female_median_r']:+.3f} | {sc['positive_ratio']*100:.1f}% |\n")
        f.write("\n---\n")

    logger.info(f"Saved frame correlations to {frame_csv_path}")
    logger.info(f"Saved utterance stratification to {utt_csv_path}")
    logger.info(f"Saved speaker correlations to {speaker_csv_path}")
    logger.info(f"Saved Table CSV to {table_csv_path}")
    logger.info(f"Saved Markdown report to {md_path}")

    return frame_df, utterance_df, speaker_df, metrics_summary


def main():
    parser = argparse.ArgumentParser(
        description="Glottal Physical Interpretability & External EGG Validity Audit (AVID Corpus)."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to WavFry model checkpoint.")
    parser.add_argument("--avid-dir", type=str, default=str(DEFAULT_AVID_DIR), help="Path to AVID corpus (Repository 1).")
    parser.add_argument(
        "--covarep-dir", type=str, default=str(DEFAULT_COVAREP_DIR), help="Path to COVAREP decom directory."
    )
    parser.add_argument(
        "--octave-bin", type=str, default=str(DEFAULT_OCTAVE_BIN), help="Path to GNU Octave executable."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/experiments/interpretability/egg_audit",
        help="Directory to save audit data.",
    )
    parser.add_argument(
        "--speakers", type=str, default=None, help="Comma-separated speaker IDs (e.g. Spk1,Spk2)."
    )
    parser.add_argument(
        "--all-speakers", action="store_true", help="Evaluate all 50 speakers."
    )
    parser.add_argument(
        "--duration-per-speaker",
        type=float,
        default=None,
        help="Duration in seconds to extract per speaker (e.g. 180s for 3 mins, or None for full recording).",
    )
    parser.add_argument(
        "--limit-files", type=int, default=None, help="Limit total files for smoke testing."
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of parallel worker threads for Octave EGG extraction."
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    octave_path = resolve_octave(args.octave_bin)
    ckpt_path = resolve_checkpoint(args.checkpoint)

    spk_list = [s.strip() for s in args.speakers.split(",")] if args.speakers else None

    _, _, _, summary = evaluate_egg_correlations(
        avid_dir=Path(args.avid_dir),
        checkpoint_path=ckpt_path,
        output_dir=PROJECT_ROOT / args.output_dir,
        covarep_dir=Path(args.covarep_dir),
        octave_bin=octave_path,
        speakers=spk_list,
        all_speakers=args.all_speakers,
        duration_per_speaker=args.duration_per_speaker,
        limit_files=args.limit_files,
        num_workers=args.num_workers,
        device_str=args.device,
    )

    print("\n=== EGG Glottal Physical Interpretability Summary ===")
    for row in summary["correlations"].get("frame_level", []):
        print(f"[{row['Target']}] Pearson r = {row['Pearson_r']:+.4f} (p={row['Pearson_p']:.2e}), Spearman rho = {row['Spearman_rho']:+.4f}")


if __name__ == "__main__":
    main()
