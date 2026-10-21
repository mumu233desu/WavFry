"""
Unified Baseline Extractors for Table 1 and Figure 2.

Provides modular, robust, and reproducible extraction wrappers for the four
classical and deep learning voice quality / creaky voice baselines:
1. Praat (Boersma & Weenink, 2024 / Parselmouth):
   Measures pitch perturbation and unvoiced segment ratio across vowel intervals.
2. CreaPy (Paierl & Röck):
   Decision tree & acoustic feature extractor based on ZCR, STE, and MFCCs.
3. DeepFry (Chernyak et al., 2022):
   Pre-trained CNN-BiLSTM raw audio creaky voice detector.
4. Kane et al. (2013 / Covarep):
   Glottal source parameter extraction (SRH, GCI, H2-H1) via GNU Octave.

All extractors share:
- Strict 16kHz mono audio staging with deterministic acoustic degradation via
  wavfry.eval.acoustic_degradation.AcousticDegradationEngine.
- Intermediate span-level caching to prevent redundant CLI / tool executions.
- Standardized vowel interval overlap computation returning continuous creak
  posteriors for ROC-AUC / PR-AUC and F1 evaluation.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

# Suppress known benign third-party warnings from CreaPy and NumPy
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value encountered.*")

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

from wavfry.eval.acoustic_degradation import AcousticDegradationEngine, get_acoustic_degradation_engine

logger = logging.getLogger("baseline_extractors")

# Default tool and model paths
OCTAVE_CLI_DEFAULT = Path(r"D:\Tools\octave-11.3.0-w64\mingw64\bin\octave-cli.exe")
BASELINES_DIR_DEFAULT = Path(r"E:\Coding\WavFry\data\experiments\baselines").resolve()
COVAREP_DIR_DEFAULT = BASELINES_DIR_DEFAULT / "covarep-master"
CREAPY_DIR_DEFAULT = BASELINES_DIR_DEFAULT / "creapy-main"
DEEPFRY_DIR_DEFAULT = BASELINES_DIR_DEFAULT / "DeepFry"
DEEPFRY_MODEL_DEFAULT = (
    DEEPFRY_DIR_DEFAULT
    / "models"
    / "CREAK-220lr_0.001_decay_21_input_size_512_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_.pth"
)


def resolve_octave_cli() -> Optional[Path]:
    """Resolve GNU Octave CLI path from default location or PATH."""
    if OCTAVE_CLI_DEFAULT.exists():
        return OCTAVE_CLI_DEFAULT
    which_path = shutil.which("octave-cli")
    if which_path:
        return Path(which_path)
    return None


def compute_interval_overlap_ratio(
    start_s: float,
    end_s: float,
    spans: List[Tuple[float, float]],
    min_overlap_sec: float = 0.020,
) -> float:
    """
    Compute overlap ratio between an interval [start_s, end_s] and detected spans.

    Returns:
        Continuous creak posterior in [0.0, 1.0].
    """
    dur = max(1e-6, end_s - start_s)
    ov_total = 0.0
    for s, e in spans:
        ov = min(end_s, float(e)) - max(start_s, float(s))
        if ov > 0.0:
            ov_total += ov
    return float(np.clip(ov_total / dur, 0.0, 1.0))


def stage_degraded_audio_pool(
    test_manifest: pd.DataFrame,
    tier: str,
    degradation_engine: AcousticDegradationEngine,
    staging_base_dir: Path,
    seed: int = 42,
) -> Dict[str, Path]:
    """
    Stage 16kHz mono degraded audio files for a given tier.

    All external baselines require 16kHz mono PCM_16 audio on disk.
    Staging once per tier ensures exact acoustic parity across tools and
    avoids redundant audio synthesis.

    Returns:
        Mapping of file_key -> staged wav Path.
    """
    tier_stage_dir = (staging_base_dir / tier).resolve()
    tier_stage_dir.mkdir(parents=True, exist_ok=True)

    unique_files = test_manifest.drop_duplicates(subset=["file_key"])
    staged_map: Dict[str, Path] = {}

    for _, row in unique_files.iterrows():
        file_key = str(row["file_key"])
        src_audio_path = Path(str(row["audio_path"])).resolve()
        dst_wav_path = (tier_stage_dir / f"{file_key}.wav").resolve()
        staged_map[file_key] = dst_wav_path

        if dst_wav_path.exists() and dst_wav_path.stat().st_size > 1000:
            continue

        if not src_audio_path.exists():
            logger.warning(f"Source audio file not found: {src_audio_path}")
            continue

        try:
            # Read and convert to 16kHz mono float32
            data, sr = sf.read(str(src_audio_path), always_2d=False, dtype="float32")
            if data.ndim > 1:
                data = np.mean(data, axis=1)

            if sr != 16000:
                t = torch.from_numpy(data).unsqueeze(0).float()
                t = torchaudio.functional.resample(t, sr, 16000)
                data = t.squeeze(0).numpy()
                sr = 16000

            # Apply degradation condition
            if tier != "clean":
                sub_seed = (seed + abs(hash(file_key)) % 10007) if seed is not None else None
                degraded = degradation_engine.apply_numpy(data, sr, condition=tier, seed=sub_seed)
            else:
                degraded = data

            # Peak normalization if clipping
            max_val = float(np.max(np.abs(degraded)))
            if max_val > 1.0:
                degraded = degraded / (max_val + 1e-4) * 0.99

            sf.write(str(dst_wav_path), degraded, 16000, subtype="PCM_16")
        except Exception as e:
            logger.error(f"Error staging audio for {file_key} at tier {tier}: {e}")

    return staged_map


# ==============================================================================
# 1. Praat Extractor
# ==============================================================================

def _extract_single_praat_record(item: Tuple[str, str, float, float, str, str, str]) -> Dict[str, Any]:
    audio_path_str, interval_key, start_s, end_s, dataset_id, file_key, tier = item
    audio_path = Path(audio_path_str)
    if not audio_path.exists():
        prob = 0.0
    else:
        try:
            import parselmouth

            sound = parselmouth.Sound(str(audio_path))
            pitch = sound.to_pitch(time_step=0.015, pitch_floor=50.0, pitch_ceiling=350.0)
            xs = np.array(pitch.xs())
            freqs = np.array(pitch.selected_array["frequency"])
            mask = (xs >= start_s) & (xs <= end_s)
            if np.sum(mask) == 0:
                prob = 0.0
            else:
                sub_f = freqs[mask]
                uv_ratio = np.mean(sub_f == 0.0)
                prob = float(np.clip(uv_ratio, 0.0, 1.0))
        except Exception:
            prob = 0.0

    return {
        "interval_key": interval_key,
        "dataset_id": dataset_id,
        "file_key": file_key,
        "start_s": start_s,
        "end_s": end_s,
        "model_name": "Praat",
        "tier": tier,
        "prob_creaky": prob,
    }


def extract_praat_predictions(
    manifest_df: pd.DataFrame,
    staged_wavs: Dict[str, Path],
    tier: str,
    max_workers: Optional[int] = None,
) -> pd.DataFrame:
    """Extract Praat baseline predictions using ThreadPoolExecutor."""
    if manifest_df.empty:
        return pd.DataFrame()

    items = [
        (
            str(staged_wavs.get(str(row["file_key"]), row["audio_path"])),
            str(row["interval_key"]),
            float(row["start_s"]),
            float(row["end_s"]),
            str(row.get("dataset_id", "cs_dialogue")),
            str(row["file_key"]),
            tier,
        )
        for _, row in manifest_df.iterrows()
    ]

    from concurrent.futures import ThreadPoolExecutor

    workers = max_workers or min(8, os.cpu_count() or 4)
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record in executor.map(_extract_single_praat_record, items):
            rows.append(record)

    return pd.DataFrame(rows)


# ==============================================================================
# 2. CreaPy Extractor
# ==============================================================================

def extract_creapy_predictions(
    manifest_df: pd.DataFrame,
    staged_wavs: Dict[str, Path],
    tier: str,
    cache_spans_dir: Path,
    creapy_dir: Path = CREAPY_DIR_DEFAULT,
) -> pd.DataFrame:
    """Extract CreaPy predictions with file-level span caching."""
    if manifest_df.empty:
        return pd.DataFrame()

    if str(creapy_dir) not in sys.path:
        sys.path.insert(0, str(creapy_dir))

    import creapy
    from creapy.model.classify import process_file
    from creapy.utils.helpers import get_creak_intervals, get_time_vector

    tier_cache_dir = (cache_spans_dir / tier).resolve()
    tier_cache_dir.mkdir(parents=True, exist_ok=True)

    file_keys = sorted(manifest_df["file_key"].unique().tolist())
    spans_by_file: Dict[str, List[Tuple[float, float]]] = {}

    for key in file_keys:
        json_cache = tier_cache_dir / f"{key}.json"
        if json_cache.exists():
            try:
                with open(json_cache, "r", encoding="utf-8") as f:
                    spans_by_file[key] = [tuple(x) for x in json.load(f)]
                continue
            except Exception:
                pass

        wav_path = staged_wavs.get(key)
        if not wav_path or not wav_path.exists():
            spans_by_file[key] = []
            continue

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = process_file(str(wav_path))

            y_pred = res[1] if isinstance(res, tuple) and len(res) >= 2 else None
            if y_pred is not None and len(y_pred) > 0:
                intervals = get_creak_intervals(
                    y_pred, get_time_vector(y_pred, 16000, 0), tgt_intervals=False
                )
                spans = [(float(s), float(e)) for s, e in intervals]
            else:
                spans = []

            with open(json_cache, "w", encoding="utf-8") as f:
                json.dump(spans, f)
            spans_by_file[key] = spans
        except Exception as e:
            logger.warning(f"[CreaPy] Failed on {key}: {e}")
            spans_by_file[key] = []

    records = []
    for _, row in manifest_df.iterrows():
        key = str(row["file_key"])
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        spans = spans_by_file.get(key, [])
        prob = compute_interval_overlap_ratio(start_s, end_s, spans)

        records.append({
            "interval_key": str(row["interval_key"]),
            "dataset_id": str(row.get("dataset_id", "cs_dialogue")),
            "file_key": key,
            "start_s": start_s,
            "end_s": end_s,
            "model_name": "Creapy",
            "tier": tier,
            "prob_creaky": prob,
        })

    return pd.DataFrame(records)


# ==============================================================================
# 3. DeepFry Extractor
# ==============================================================================

def extract_deepfry_predictions(
    manifest_df: pd.DataFrame,
    staged_wavs: Dict[str, Path],
    tier: str,
    cache_spans_dir: Path,
    deepfry_dir: Path = DEEPFRY_DIR_DEFAULT,
    deepfry_model: Path = DEEPFRY_MODEL_DEFAULT,
) -> pd.DataFrame:
    """Extract DeepFry baseline predictions via TextGrid inference."""
    if manifest_df.empty:
        return pd.DataFrame()

    tier_cache_dir = (cache_spans_dir / tier).resolve()
    tier_cache_dir.mkdir(parents=True, exist_ok=True)

    file_keys = sorted(manifest_df["file_key"].unique().tolist())
    missing_keys = [k for k in file_keys if not (tier_cache_dir / f"{k}.TextGrid").exists()]

    if missing_keys:
        sub_stage = (tier_cache_dir / "_sub_stage").resolve()
        sub_stage_test = (sub_stage / "test").resolve()
        if sub_stage.exists():
            shutil.rmtree(sub_stage)
        sub_stage_test.mkdir(parents=True, exist_ok=True)

        for k in missing_keys:
            src_wav = staged_wavs.get(k)
            if src_wav and src_wav.exists():
                shutil.copy2(src_wav, sub_stage_test / f"{k}.wav")

        # Run DeepFry run.py CLI with resolved absolute paths
        cmd = [
            sys.executable,
            str(deepfry_dir.resolve() / "run.py"),
            "--data_dir",
            str(sub_stage),
            "--model_name",
            str(deepfry_model.resolve()),
            "--output_dir",
            str(tier_cache_dir),
            "--custom",
            "--workers",
            "0",
            "--batch_size",
            "8",
            "--tv",
            "0.5",
            "--tc",
            "0.5",
        ]
        proc = subprocess.run(cmd, cwd=str(deepfry_dir.resolve()), text=True, capture_output=True)
        if proc.returncode != 0:
            logger.warning(f"[DeepFry] CLI Output/Warning:\n{proc.stderr[:400]}")

        if sub_stage.exists():
            shutil.rmtree(sub_stage)

    # Parse resulting TextGrids
    import tgt

    spans_by_file: Dict[str, List[Tuple[float, float]]] = {}
    for key in file_keys:
        tg_path = tier_cache_dir / f"{key}.TextGrid"
        spans = []
        if tg_path.exists():
            try:
                tg = tgt.io.read_textgrid(str(tg_path), encoding="utf-8")
                for tier_obj in tg.tiers:
                    if "pred-creaky" in tier_obj.name.lower():
                        for interval in tier_obj:
                            if interval.text.strip().lower() == "c":
                                spans.append((float(interval.start_time), float(interval.end_time)))
                        break
            except Exception as e:
                logger.warning(f"[DeepFry] Error parsing TextGrid {key}: {e}")
        spans_by_file[key] = spans

    records = []
    for _, row in manifest_df.iterrows():
        key = str(row["file_key"])
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        spans = spans_by_file.get(key, [])
        prob = compute_interval_overlap_ratio(start_s, end_s, spans)

        records.append({
            "interval_key": str(row["interval_key"]),
            "dataset_id": str(row.get("dataset_id", "cs_dialogue")),
            "file_key": key,
            "start_s": start_s,
            "end_s": end_s,
            "model_name": "DeepFry",
            "tier": tier,
            "prob_creaky": prob,
        })

    return pd.DataFrame(records)


# ==============================================================================
# 4. Kane et al. (2013 / Covarep) Extractor
# ==============================================================================

def extract_kane_predictions(
    manifest_df: pd.DataFrame,
    staged_wavs: Dict[str, Path],
    tier: str,
    cache_spans_dir: Path,
    octave_cli: Optional[Path] = None,
    covarep_dir: Path = COVAREP_DIR_DEFAULT,
    batch_kane_m: Path = BASELINES_DIR_DEFAULT / "batch_kane.m",
) -> pd.DataFrame:
    """Extract Kane (2013) baseline predictions via GNU Octave Covarep."""
    if manifest_df.empty:
        return pd.DataFrame()

    oct_bin = octave_cli or resolve_octave_cli()
    if not oct_bin or not oct_bin.exists():
        logger.error(f"[Kane] GNU Octave CLI not found at {oct_bin}; skipping Kane baseline.")
        return pd.DataFrame()

    tier_cache_dir = (cache_spans_dir / tier).resolve()
    tier_cache_dir.mkdir(parents=True, exist_ok=True)

    file_keys = sorted(manifest_df["file_key"].unique().tolist())
    missing_keys = [k for k in file_keys if not (tier_cache_dir / f"{k}.json").exists()]

    if missing_keys:
        n_workers = min(6, len(missing_keys))
        chunks = [missing_keys[i::n_workers] for i in range(n_workers)]
        chunks = [c for c in chunks if c]
        logger.info(f"[Kane] Extracting {len(missing_keys)} missing files using {len(chunks)} parallel Octave workers...")

        def _run_octave_chunk(worker_idx: int, keys_subset: List[str]):
            list_file = (tier_cache_dir / f"wav_list_w{worker_idx}.txt").resolve()
            with open(list_file, "w", encoding="utf-8") as f:
                for k in keys_subset:
                    p = staged_wavs.get(k)
                    if p and p.exists():
                        f.write(f"{p.resolve().as_posix()}\n")

            cmd = [
                str(oct_bin),
                "--eval",
                f"addpath('{BASELINES_DIR_DEFAULT.as_posix()}'); batch_kane('{list_file.as_posix()}', '{tier_cache_dir.as_posix()}');",
            ]
            proc = subprocess.run(cmd, cwd=str(BASELINES_DIR_DEFAULT), capture_output=True, text=True)
            if proc.returncode != 0:
                logger.warning(f"[Kane W{worker_idx}] Octave warning/error:\n{proc.stderr[:400]}")
            list_file.unlink(missing_ok=True)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            futures = [executor.submit(_run_octave_chunk, idx, c) for idx, c in enumerate(chunks)]
            for fut in futures:
                fut.result()

    spans_by_file: Dict[str, List[Tuple[float, float]]] = {}
    for key in file_keys:
        json_file = tier_cache_dir / f"{key}.json"
        spans = []
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    spans = [tuple(x) for x in json.load(f)]
            except Exception:
                pass
        spans_by_file[key] = spans

    records = []
    for _, row in manifest_df.iterrows():
        key = str(row["file_key"])
        start_s = float(row["start_s"])
        end_s = float(row["end_s"])
        spans = spans_by_file.get(key, [])
        prob = compute_interval_overlap_ratio(start_s, end_s, spans)

        records.append({
            "interval_key": str(row["interval_key"]),
            "dataset_id": str(row.get("dataset_id", "cs_dialogue")),
            "file_key": key,
            "start_s": start_s,
            "end_s": end_s,
            "model_name": "Kane",
            "tier": tier,
            "prob_creaky": prob,
        })

    return pd.DataFrame(records)
