#!/usr/bin/env python
"""
External Baseline Evaluation Script for Creaky Voice Detection.
Evaluates DeepFry and Creapy on the same annotated speakers as WavFry LOOCV,
and generates a combined Markdown report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import yaml


WORKSPACE_ROOT = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE")
# Local baselines dir where we copied deepfry and creapy
BASELINES_DIR = Path(r"E:\Coding\WavFry\data\experiments\baselines")
DEEPFRY_DIR = BASELINES_DIR / "DeepFry"
CREAPY_DIR = BASELINES_DIR / "creapy-main"

import parselmouth

OCTAVE_CLI = r"D:\Tools\octave-11.3.0-w64\mingw64\bin\octave-cli.exe"

def run_praat(val_df: pd.DataFrame, audio_dir: Path, cache_dir: Path) -> dict[tuple[str, str], int]:
    """
    Evaluates Praat baseline for creaky voice detection.
    Uses Parselmouth to extract pitch (F0). Unvoiced regions (where F0 = 0) are considered creaky.
    """
    print("[praat] Starting Praat evaluation...")
    file_keys = sorted(val_df["file_key"].unique().tolist())
    
    hash_str = f"praat|{audio_dir.name}|{'|'.join(file_keys)}"
    input_hash = hashlib.md5(hash_str.encode("utf-8")).hexdigest()
    
    base_dir = cache_dir / "cache_praat"
    done_file = base_dir / "_DONE.json"
    spans_file = base_dir / "spans.json"
    
    cache_valid = False
    if done_file.exists() and spans_file.exists():
        try:
            cache_data = json.loads(done_file.read_text(encoding="utf-8"))
            if cache_data.get("input_hash") == input_hash:
                cache_valid = True
        except Exception:
            pass
            
    if cache_valid:
        print("[praat] Cache is valid. Skipping inference.")
        spans_by_file = json.loads(spans_file.read_text(encoding="utf-8"))
    else:
        print("[praat] Cache invalid or missing. Running inference...")
        base_dir.mkdir(parents=True, exist_ok=True)
        
        spans_by_file = {}
        
        from tqdm import tqdm
        for key in tqdm(file_keys, desc="[praat] Inferring", ncols=80):
            src_wav = audio_dir / f"{key}.wav"
            src_flac = audio_dir / f"{key}.flac"
            
            src_path = None
            if src_wav.exists(): 
                src_path = src_wav
            elif src_flac.exists(): 
                src_path = src_flac
            else:
                print(f"Warning: Audio for {key} not found in {audio_dir}")
                continue
                
            try:
                snd = parselmouth.Sound(str(src_path))
                pitch = snd.to_pitch(time_step=0.015, pitch_floor=50, pitch_ceiling=350)
                
                freqs = pitch.selected_array['frequency']
                xs = pitch.xs()
                
                time_step = pitch.time_step
                half_step = time_step / 2.0
                
                spans = []
                in_unvoiced = False
                start_time = 0.0
                
                for i, f in enumerate(freqs):
                    if f == 0:
                        if not in_unvoiced:
                            start_time = float(xs[i] - half_step)
                            in_unvoiced = True
                    else:
                        if in_unvoiced:
                            end_time = float(xs[i - 1] + half_step)
                            spans.append((start_time, end_time))
                            in_unvoiced = False
                            
                if in_unvoiced:
                    end_time = float(xs[-1] + half_step)
                    spans.append((start_time, end_time))
                    
                spans_by_file[key] = spans
            except Exception as e:
                print(f"Error processing {key} with Praat: {e}")
                spans_by_file[key] = []
                
        spans_file.write_text(json.dumps(spans_by_file, indent=2), encoding="utf-8")
        done_file.write_text(json.dumps({
            "model": "praat",
            "input_hash": input_hash,
            "expected_files": file_keys,
            "timestamp": time.time(),
        }, indent=2), encoding="utf-8")
        
    return spans_to_pred_map(val_df, spans_by_file)


def run_kane(val_df: pd.DataFrame, audio_dir: Path, cache_dir: Path) -> dict[tuple[str, str], int]:
    """
    Evaluates Kane (2013) baseline for creaky voice detection.
    Invokes the external MATLAB/Octave script `kane_baseline.m` to extract glottal pulses and detect creak.
    """
    print("[kane] Starting Kane (2013) baseline evaluation...")
    file_keys = sorted(val_df["file_key"].unique().tolist())
    
    hash_str = f"kane_v1|{audio_dir.name}|{'|'.join(file_keys)}"
    input_hash = hashlib.md5(hash_str.encode("utf-8")).hexdigest()
    
    base_dir = cache_dir / "cache_kane"
    data_dir = base_dir / "test"
    pred_dir = base_dir / "pred"
    done_file = base_dir / "_DONE.json"
    
    if check_cache_valid(done_file, input_hash):
        print("[kane] Cache is valid. Skipping inference.")
    else:
        print("[kane] Cache invalid or missing. Running inference...")
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        stage_audio(val_df, audio_dir, data_dir)
        
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def process_key(key):
            in_wav = data_dir / f"{key}.wav"
            out_json = pred_dir / f"{key}.json"
            
            kane_script_dir = r"E:\Coding\WavFry\data\experiments\baselines"
            eval_str = f"kane_baseline('{in_wav.as_posix()}', '{out_json.as_posix()}')"
            cmd = [OCTAVE_CLI, "--no-history", "--no-gui", "--quiet", "--eval", eval_str]
            
            proc = subprocess.run(cmd, cwd=kane_script_dir, text=True, capture_output=True)
            if proc.returncode != 0:
                return f"[kane] Error running Octave for {key}:\n{proc.stderr}\n{proc.stdout}"
            return None

        max_workers = min(os.cpu_count() or 4, 16)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_key, key): key for key in file_keys}
            for future in tqdm(as_completed(futures), total=len(file_keys), desc="[kane] Inferring", ncols=80):
                err = future.result()
                if err:
                    print(err)
                
        for key in file_keys:
            dummy_tg = pred_dir / f"{key}.TextGrid"
            dummy_tg.write_text("Dummy TextGrid for caching", encoding="utf-8")
        
        done_file.write_text(json.dumps({
            "model": "kane",
            "input_hash": input_hash,
            "expected_files": file_keys,
            "timestamp": time.time(),
        }, indent=2), encoding="utf-8")
        
    spans_by_file = {}
    for key in file_keys:
        out_json = pred_dir / f"{key}.json"
        if out_json.exists():
            try:
                spans = json.loads(out_json.read_text(encoding="utf-8"))
                spans_by_file[key] = [(float(s[0]), float(s[1])) for s in spans]
            except Exception as e:
                print(f"[kane] Failed to parse JSON for {key}: {e}")
                
    return spans_to_pred_map(val_df, spans_by_file)


DEEPFRY_MODEL = DEEPFRY_DIR / "models" / "CREAK-220lr_0.001_decay_21_input_size_512_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_.pth"
DEEPFRY_USE_CUSTOM = True

ANNOTATION_PATH = WORKSPACE_ROOT / "02_en_mfa_aligned" / "_creaky_cache" / "manual_creaky_annotations.json"

DEEPFRY_PRED_TIER = "pred-creaky"
DEEPFRY_POS_LABEL = "c"
CREAPY_PRED_SUFFIX = ""
CREAPY_PRED_TIER = "creapy"
CREAPY_POS_LABEL = "c"

MIN_OVERLAP_SEC = 0.020
MIN_OVERLAP_RATIO = 0.0

VOWEL_SET = {
    # ARPAbet
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW",
    "AA0", "AA1", "AA2", "AE0", "AE1", "AE2", "AH0", "AH1", "AH2", "AO0", "AO1", "AO2",
    "AW0", "AW1", "AW2", "AY0", "AY1", "AY2", "EH0", "EH1", "EH2", "ER0", "ER1", "ER2",
    "EY0", "EY1", "EY2", "IH0", "IH1", "IH2", "IY0", "IY1", "IY2", "OW0", "OW1", "OW2",
    "OY0", "OY1", "OY2", "UH0", "UH1", "UH2", "UW0", "UW1", "UW2",
    # MFA English
    "a", "aː", "aj", "aw", "e", "eː", "ej", "i", "iː", "o", "oː", "ow", "u", "uː", "æ", 
    "ɑ", "ɑː", "ɐ", "ɒ", "ɒː", "ɔ", "ɔj", "ə", "əw", "ɚ", "ɛ", "ɛː", "ɜ", "ɜː", "ɝ", "ɪ", "ʉ", "ʉː", "ʊ",
    # HFA Mandarin
    "v", "ai", "an", "ao", "ei", "er", "i0", "ia", "ie", "in", "io", "ir", "iu", "ix",
    "iy", "iz", "o0", "ong", "ou", "ua", "uai", "uan", "ue", "ui", "un", "uo", "van",
    "ve", "vn"
}

VALID_STATES = {"creaky", "non_creaky", "uncertain"}

def parse_speaker_id(file_key: str) -> str:
    parts = str(file_key).split("_")
    if len(parts) >= 6:
        return str(parts[5])
    return str(file_key)

def parse_gender(file_key: str) -> str:
    parts = str(file_key).split("_")
    if len(parts) >= 1:
        return str(parts[0])
    return ""

def parse_region(file_key: str) -> str:
    parts = str(file_key).split("_")
    if len(parts) >= 3:
        return str(parts[2])
    return ""

def is_vowel(label: str) -> bool:
    return label.upper() in VOWEL_SET or label.lower() in VOWEL_SET

def make_interval_id(start: float, end: float, label: str) -> str:
    return f"{start:.4f}|{end:.4f}|{label}"

def load_wavfry_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def _normalize_entry(entry: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    raw_annotations = entry.get("annotations", {})
    annotations: Dict[str, str] = {}
    if isinstance(raw_annotations, dict):
        for interval_id, state in raw_annotations.items():
            if state in VALID_STATES:
                annotations[str(interval_id)] = str(state)
    normalized: Dict[str, Any] = {"annotations": annotations}
    return normalized if annotations else None

def load_annotation_shards(manifest_path: Path) -> dict[str, dict[str, str]]:
    payload = {"files": {}}
    shard_root = manifest_path.with_name(manifest_path.name + ".d")
    
    # Try monolithic file first
    try:
        if manifest_path.exists():
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and "storage_mode" not in loaded:
                 if isinstance(loaded.get("files"), dict):
                     for file_key, entry in loaded["files"].items():
                         norm = _normalize_entry(entry)
                         if norm: payload["files"][str(file_key)] = norm
    except Exception:
        pass
        
    # Try sharded
    if shard_root.exists():
        for shard_path in sorted(shard_root.glob("*.json")):
            try:
                shard_payload = json.loads(shard_path.read_text(encoding="utf-8"))
                file_key = shard_payload.get("file_key")
                if isinstance(file_key, str) and file_key:
                    norm = _normalize_entry(shard_payload)
                    if norm: payload["files"][file_key] = norm
            except Exception:
                continue
                
    return payload["files"]

def parse_textgrid_intervals(tg_path: Path, preferred_tier_name: str) -> list[tuple[float, float, str]]:
    intervals = []
    
    # Try using tgt for robust parsing of both short and long formats
    try:
        import tgt
        tg = tgt.io.read_textgrid(str(tg_path), encoding="utf-8")
        for tier in tg.tiers:
            if preferred_tier_name.lower() in tier.name.lower() or tier.name.lower() in preferred_tier_name.lower():
                return [(float(interval.start_time), float(interval.end_time), str(interval.text)) for interval in tier]
    except Exception:
        pass
        
    # Fallback lightweight parser (long format only)
    try:
        lines = [line.strip() for line in tg_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return intervals
        
    in_target_tier = False
    in_interval = False
    current_start = 0.0
    current_end = 0.0
    
    for i, line in enumerate(lines):
        if line.startswith("class = \"IntervalTier\""):
            # Check next few lines for name
            tier_name = ""
            for j in range(i+1, min(i+5, len(lines))):
                if lines[j].startswith("name = "):
                    tier_name = lines[j].split("=", 1)[1].strip().strip('"')
                    break
            if tier_name.lower() == preferred_tier_name.lower():
                in_target_tier = True
            elif tier_name.lower() in preferred_tier_name.lower() or preferred_tier_name.lower() in tier_name.lower():
                in_target_tier = True
            else:
                in_target_tier = False
                
        if not in_target_tier:
            continue
            
        if line.startswith("intervals [") and "]:" in line:
            in_interval = True
            continue
            
        if in_interval:
            if line.startswith("xmin ="):
                current_start = float(line.split("=")[1].strip())
            elif line.startswith("xmax ="):
                current_end = float(line.split("=")[1].strip())
            elif line.startswith("text ="):
                label = line.split("=")[1].strip().strip('"')
                intervals.append((current_start, current_end, label))
                in_interval = False
                
    return intervals

def build_val_manifest(
    speakers: list[str],
    audio_dir: Path,
    tg_dir: Path,
    annotation_path: Path,
    phone_tier: str = "phones",
    target_file_keys: set[str] = None,
) -> pd.DataFrame:
    print("[val_manifest] Building validation manifest...")
    target_set = set(speakers)
    ann_files = load_annotation_shards(annotation_path)
    
    rows = []
    
    for tg_path in sorted(tg_dir.glob("*.TextGrid")):
        file_key = tg_path.stem
        speaker_id = parse_speaker_id(file_key)
        
        if target_file_keys is not None:
            if file_key not in target_file_keys:
                continue
        elif speaker_id not in target_set:
            continue
            
        gender = parse_gender(file_key)
        region = parse_region(file_key)
        
        entry = ann_files.get(file_key, {})
        annotations = entry.get("annotations", {})
        
        intervals = parse_textgrid_intervals(tg_path, phone_tier)
        for start, end, label in intervals:
            if not is_vowel(label):
                continue
                
            interval_id = make_interval_id(start, end, label)
            state = annotations.get(interval_id)
            
            if state == "creaky":
                gold_label = 1
            elif state == "non_creaky":
                gold_label = 0
            else:
                continue # uncertain or unreviewed
                
            rows.append({
                "file_key": file_key,
                "speaker_id": speaker_id,
                "gender": gender,
                "region": region,
                "start": start,
                "end": end,
                "label": label,
                "interval_id": interval_id,
                "gold_label": gold_label,
            })
            
    df = pd.DataFrame(rows)
    print(f"[val_manifest] Found {len(df)} reviewed vowel intervals for target speakers.")
    return df

def stage_audio(
    val_df: pd.DataFrame,
    audio_dir: Path,
    stage_dir: Path,
    target_sr: int = 16000,
) -> dict[str, Path]:
    print("[staging] Staging audio files...")
    stage_dir.mkdir(parents=True, exist_ok=True)
    
    file_keys = sorted(val_df["file_key"].unique().tolist())
    staged_map = {}
    
        
    for key in file_keys:
        dst_wav = stage_dir / f"{key}.wav"
        staged_map[key] = dst_wav
        
        if dst_wav.exists():
            continue
            
        # Try to find source audio
        src_wav = audio_dir / f"{key}.wav"
        src_flac = audio_dir / f"{key}.flac"
        
        src_path = None
        if src_wav.exists(): src_path = src_wav
        elif src_flac.exists(): src_path = src_flac
        else:
            print(f"Warning: Audio for {key} not found in {audio_dir}")
            continue
            
        audio, sr = sf.read(str(src_path), always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            
        if sr != target_sr:
            import torch, torchaudio
            audio_t = torch.from_numpy(audio).unsqueeze(0).float()
            audio = torchaudio.functional.resample(audio_t, sr, target_sr).squeeze(0).numpy()
            sr = target_sr
            
        sf.write(str(dst_wav), np.asarray(audio, dtype=np.float32), int(sr), subtype="PCM_16")
        
    return staged_map


def check_cache_valid(cache_done_file: Path, input_hash: str) -> bool:
    if not cache_done_file.exists():
        return False
    try:
        cache_data = json.loads(cache_done_file.read_text(encoding="utf-8"))
        if cache_data.get("input_hash") != input_hash:
            return False
        
        # Verify output files exist
        pred_dir = cache_done_file.parent / "pred"
        for stem in cache_data.get("expected_files", []):
            if cache_data.get("model") == "creapy":
                f = pred_dir / f"{stem}{CREAPY_PRED_SUFFIX}.TextGrid"
            else:
                f = pred_dir / f"{stem}.TextGrid"
            if not f.exists():
                print(f"[cache] Missing output file: {f.name}, cache invalidated.")
                return False
        return True
    except Exception:
        return False


def run_deepfry(val_df: pd.DataFrame, audio_dir: Path, cache_dir: Path) -> dict[tuple[str, str], int]:
    """
    Evaluates DeepFry baseline for creaky voice detection.
    Invokes the external DeepFry tool which runs a deep learning model to predict creak probability.
    """
    print("[deepfry] Starting DeepFry evaluation...")
    file_keys = sorted(val_df["file_key"].unique().tolist())
    
    # Compute hash based on model, keys, and settings
    hash_str = f"deepfry_v1_custom|{audio_dir.name}|{'|'.join(file_keys)}"
    input_hash = hashlib.md5(hash_str.encode("utf-8")).hexdigest()
    
    base_dir = cache_dir / "deepfry"
    data_dir = base_dir / "test"
    pred_dir = base_dir / "pred"
    done_file = base_dir / "_DONE.json"
    
    if check_cache_valid(done_file, input_hash):
        print("[deepfry] Cache is valid. Skipping inference.")
    else:
        print("[deepfry] Cache invalid or missing. Running inference...")
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        stage_audio(val_df, audio_dir, data_dir)
        
        cmd = [
            sys.executable, str(DEEPFRY_DIR / "run.py"),
            "--data_dir", str(base_dir),
            "--model_name", str(DEEPFRY_MODEL),
            "--output_dir", str(pred_dir),
            "--custom",
            "--workers", "0",
            "--batch_size", "8",
            "--tv", "0.5",
            "--tc", "0.5",
        ]
        
        proc = subprocess.run(cmd, cwd=str(DEEPFRY_DIR), text=True, capture_output=True)
        if proc.returncode != 0:
            print(f"[deepfry] Error running DeepFry:\n{proc.stderr}")
            return {}
            
        # Write Cache Done
        done_file.write_text(json.dumps({
            "model": "deepfry",
            "input_hash": input_hash,
            "expected_files": file_keys,
            "timestamp": time.time(),
        }, indent=2), encoding="utf-8")
        
    # Read predictions
    spans_by_file = {}
    for key in file_keys:
        tg_path = pred_dir / f"{key}.TextGrid"
        if tg_path.exists():
            intervals = parse_textgrid_intervals(tg_path, DEEPFRY_PRED_TIER)
            spans_by_file[key] = [(s, e) for s, e, l in intervals if l.lower() == DEEPFRY_POS_LABEL]
            
    return spans_to_pred_map(val_df, spans_by_file)


def run_creapy(val_df: pd.DataFrame, audio_dir: Path, source_tg_dir: Path, cache_dir: Path) -> dict[tuple[str, str], int]:
    """
    Evaluates CreaPy baseline for creaky voice detection.
    Uses the CreaPy Python API to predict creak probability based on Zero-Crossing-Rate, Short-Term-Energy, and classifiers.
    """
    print("[creapy] Starting Creapy evaluation...")
    file_keys = sorted(val_df["file_key"].unique().tolist())
    
    hash_str = f"creapy|{audio_dir.name}|{'|'.join(file_keys)}"
    input_hash = hashlib.md5(hash_str.encode("utf-8")).hexdigest()
    
    base_dir = cache_dir / "cache_creapy"
    audio_stage_dir = base_dir / "audio"
    tg_stage_dir = base_dir / "pred"
    done_file = base_dir / "_DONE.json"
    
    if check_cache_valid(done_file, input_hash):
        print("[creapy] Cache is valid. Skipping inference.")
    else:
        print("[creapy] Cache invalid or missing. Running inference...")
        if base_dir.exists():
            shutil.rmtree(base_dir, ignore_errors=True)
        audio_stage_dir.mkdir(parents=True, exist_ok=True)
        tg_stage_dir.mkdir(parents=True, exist_ok=True)
        
        stage_audio(val_df, audio_dir, audio_stage_dir)
        
        # Stage minimal textgrids for Creapy (it requires matching TGs)
        for key in file_keys:
            src_tg = source_tg_dir / f"{key}.TextGrid"
            dst_tg = tg_stage_dir / f"{key}.TextGrid"
            if src_tg.exists():
                shutil.copy2(src_tg, dst_tg)
                
        # Run Creapy
        if str(CREAPY_DIR) not in sys.path:
            sys.path.insert(0, str(CREAPY_DIR))
            
        try:
            import creapy
            from creapy.model.classify import process_folder
            import warnings
            
            creapy.set_config(
                audio_suffix=".wav",
                textgrid_suffix=".TextGrid",
                gender_model="all",
                tier_name=CREAPY_PRED_TIER,
                filename_extension=CREAPY_PRED_SUFFIX,
                zcr_threshold=1.0,
            )
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                process_folder(audio_directory=str(audio_stage_dir), textgrid_directory=str(tg_stage_dir), csv_directory=None)
                
            done_file.write_text(json.dumps({
                "model": "creapy",
                "input_hash": input_hash,
                "expected_files": file_keys,
                "timestamp": time.time(),
            }, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[creapy] Error running Creapy: {e}")
            return {}
            
    # Read predictions
    spans_by_file = {}
    for key in file_keys:
        tg_path = tg_stage_dir / f"{key}{CREAPY_PRED_SUFFIX}.TextGrid"
        if tg_path.exists():
            intervals = parse_textgrid_intervals(tg_path, CREAPY_PRED_TIER)
            spans_by_file[key] = [(s, e) for s, e, l in intervals if l.lower() == CREAPY_POS_LABEL]
            
    return spans_to_pred_map(val_df, spans_by_file)


def spans_to_pred_map(
    val_df: pd.DataFrame,
    spans_by_file: dict[str, list[tuple[float, float]]],
) -> dict[tuple[str, str], int]:
    out = {}
    for row in val_df.itertuples(index=False):
        spans = spans_by_file.get(str(row.file_key), [])
        dur = max(1e-6, float(row.end) - float(row.start))
        required = max(MIN_OVERLAP_SEC, MIN_OVERLAP_RATIO * dur)
        
        ov_total = 0.0
        start_s, end_s = float(row.start), float(row.end)
        for s, e in spans:
            ov = min(end_s, float(e)) - max(start_s, float(s))
            if ov > 0.0:
                ov_total += ov
                
        out[(str(row.file_key), str(row.interval_id))] = 1 if (ov_total > 0.0 and ov_total >= required) else 0
    return out


def evaluate_by_groups(val_df: pd.DataFrame, pred_map: dict[tuple[str, str], int]) -> pd.DataFrame:
    if val_df.empty or not pred_map:
        return pd.DataFrame()

    pred_values = []
    for row in val_df.itertuples(index=False):
        key = (str(row.file_key), str(row.interval_id))
        pred_values.append(pred_map.get(key, 0))
        
    work = val_df.copy()
    work["pred_label"] = np.asarray(pred_values, dtype=np.int8)
    
    # 1. First calculate metrics per speaker
    spk_metrics = []
    for spk_id, gdf in work.groupby("speaker_id"):
        y_true = gdf["gold_label"].to_numpy(dtype=np.int8)
        y_pred = gdf["pred_label"].to_numpy(dtype=np.int8)
        
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        tn = float(np.sum((y_true == 0) & (y_pred == 0)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        n = float(y_true.shape[0])
        
        acc = (tp + tn) / n if n > 0 else 0.0
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
        
        spk_metrics.append({
            "speaker_id": spk_id,
            "gender": gdf["gender"].iloc[0],
            "region": gdf["region"].iloc[0],
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1
        })
    
    spk_df = pd.DataFrame(spk_metrics)
    
    # 2. Compute macro-averages for groups
    rows = []
    
    # Per-speaker (individual)
    for row in spk_df.itertuples(index=False):
        rows.append({
            "split": "speaker",
            "group": row.speaker_id,
            "accuracy": row.accuracy,
            "precision": row.precision,
            "recall": row.recall,
            "f1": row.f1
        })
        
    # Overall and other categories
    def add_macro_avg(split_name: str, group_col: str | None):
        if group_col is None:
            # Overall macro-avg
            rows.append({
                "split": "overall",
                "group": "all",
                "accuracy": spk_df["accuracy"].mean(),
                "precision": spk_df["precision"].mean(),
                "recall": spk_df["recall"].mean(),
                "f1": spk_df["f1"].mean()
            })
        else:
            for g_name, gdf in spk_df.groupby(group_col):
                rows.append({
                    "split": split_name,
                    "group": str(g_name),
                    "accuracy": gdf["accuracy"].mean(),
                    "precision": gdf["precision"].mean(),
                    "recall": gdf["recall"].mean(),
                    "f1": gdf["f1"].mean()
                })
                
    add_macro_avg("overall", None)
    add_macro_avg("gender", "gender")
    add_macro_avg("region", "region")
    
    return pd.DataFrame(rows)



def run_wavfry(val_df: pd.DataFrame, audio_dir: Path, tg_dir: Path, cache_dir: Path, checkpoint_path: Path) -> dict[tuple[str, str], int]:
    """
    Evaluates WavFry model for creaky voice detection.
    Calls `predict.py` to run the ONNX or PyTorch WavFry model on the target intervals.
    """
    print("[wavfry] Starting WavFry evaluation...")
    file_keys = sorted(val_df["file_key"].unique().tolist())
    
    hash_str = f"wavfry|{audio_dir.name}|{checkpoint_path.name}|{int(checkpoint_path.stat().st_mtime)}|{'|'.join(file_keys)}"
    input_hash = hashlib.md5(hash_str.encode("utf-8")).hexdigest()
    
    base_dir = cache_dir / "cache_wavfry"
    audio_stage_dir = base_dir / "audio"
    tg_stage_dir = base_dir / "textgrid"
    preds_csv = base_dir / "wavfry_preds.csv"
    done_file = base_dir / "_DONE.json"
    
    # WavFry uses preds CSV, not pred/*.TextGrid — use a direct hash+file check
    _wf_cache_valid = False
    if done_file.exists() and preds_csv.exists():
        try:
            _stored = json.loads(done_file.read_text(encoding="utf-8"))
            _wf_cache_valid = (_stored.get("input_hash") == input_hash)
        except Exception:
            pass
    
    if _wf_cache_valid:
        print("[wavfry] Cache is valid. Skipping inference.")
    else:
        print("[wavfry] Cache invalid or missing. Running inference...")
        if base_dir.exists():
            import shutil
            shutil.rmtree(base_dir, ignore_errors=True)
        audio_stage_dir.mkdir(parents=True, exist_ok=True)
        tg_stage_dir.mkdir(parents=True, exist_ok=True)
        
        stage_audio(val_df, audio_dir, audio_stage_dir)
        import shutil
        for key in file_keys:
            src_tg = tg_dir / f"{key}.TextGrid"
            dst_tg = tg_stage_dir / f"{key}.TextGrid"
            if src_tg.exists():
                shutil.copy2(src_tg, dst_tg)
                
        cmd = [
            sys.executable, str(WORKSPACE_ROOT / "06_wavfry" / "scripts" / "predict.py"),
            "--checkpoint", str(checkpoint_path),
            "--audio-dir", str(audio_stage_dir),
            "--textgrid-dir", str(tg_stage_dir),
            "--output-csv", str(preds_csv),
            "--device", "auto",
            "--num-workers", "0"
        ]
        
        proc = subprocess.run(cmd, cwd=str(WORKSPACE_ROOT / "06_wavfry"), text=True, capture_output=True)
        if proc.returncode != 0:
            print(f"[wavfry] Error running WavFry:\n{proc.stderr}\n{proc.stdout}")
            return {}
        
        if not preds_csv.exists():
            print(f"[wavfry] WavFry ran but produced no output CSV.\nstdout: {proc.stdout}\nstderr: {proc.stderr}")
            return {}
            
        done_file.write_text(json.dumps({
            "model": "wavfry",
            "input_hash": input_hash,
            "expected_files": file_keys,
            "timestamp": time.time(),
        }, indent=2), encoding="utf-8")
        
    out = {}
    if preds_csv.exists():
        df_preds = pd.read_csv(preds_csv)
        for row in df_preds.itertuples(index=False):
            iid = f"{float(row.start_s):.4f}|{float(row.end_s):.4f}|{row.phone}"
            out[(str(row.file_key), iid)] = int(row.pred_label)
            
    return out



def write_report(
    out_path: Path,
    wavfry_eval: pd.DataFrame,
    deepfry_eval: pd.DataFrame,
    creapy_eval: pd.DataFrame,
    praat_eval: pd.DataFrame,
    kane_eval: pd.DataFrame,
    speaker_meta: dict[str, dict],
    details: str = "",
):
    print(f"[report] Writing report to {out_path}...")
    
    pass
                
    # 2. Write Report
    lines = [
        "# External Baseline Evaluation Report",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        details,
        "",
        "## Per-Speaker Results",
        "",
        "> **Abbreviations:** WF=WavFry, Pr=Praat, Kn=Kane, DF=DeepFry, Cr=Creapy. Acc=Accuracy, P=Precision, R=Recall",
        "",
        "| Speaker | Gender | Region | WF Acc | WF P | WF R | WF F1 | Pr Acc | Pr P | Pr R | Pr F1 | Kn Acc | Kn P | Kn R | Kn F1 | DF Acc | DF P | DF R | DF F1 | Cr Acc | Cr P | Cr R | Cr F1 |",
        "| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    ]
    
    df_spk = deepfry_eval[deepfry_eval["split"] == "speaker"].set_index("group") if not deepfry_eval.empty else pd.DataFrame()
    cp_spk = creapy_eval[creapy_eval["split"] == "speaker"].set_index("group") if not creapy_eval.empty else pd.DataFrame()
    pr_spk = praat_eval[praat_eval["split"] == "speaker"].set_index("group") if not praat_eval.empty else pd.DataFrame()
    kn_spk = kane_eval[kane_eval["split"] == "speaker"].set_index("group") if not kane_eval.empty else pd.DataFrame()
    
    wf_spk = wavfry_eval[wavfry_eval["split"] == "speaker"].set_index("group") if not wavfry_eval.empty else pd.DataFrame()
    
    for spk, meta in sorted(speaker_meta.items()):
        g, r = meta.get("gender", "U"), meta.get("region", "Unknown")
        
        def _get_metrics(df, key):
            row = df.loc[key] if key in df.index else None
            return (
                row["accuracy"] if row is not None else float('nan'),
                row["precision"] if row is not None else float('nan'),
                row["recall"] if row is not None else float('nan'),
                row["f1"] if row is not None else float('nan'),
            )
            
        def _fmt(val):
            return f"{val:.4f}" if pd.notna(val) else "N/A"
            
        wf_acc, wf_p, wf_r, wf_f1_val = _get_metrics(wf_spk, spk)
        pr_acc, pr_p, pr_r, pr_f1 = _get_metrics(pr_spk, spk)
        kn_acc, kn_p, kn_r, kn_f1 = _get_metrics(kn_spk, spk)
        df_acc, df_p, df_r, df_f1 = _get_metrics(df_spk, spk)
        cp_acc, cp_p, cp_r, cp_f1 = _get_metrics(cp_spk, spk)
        
        if pd.notna(wf_f1_val):
            wf_f1_fmt = f"**{wf_f1_val:.4f}**" if wf_f1_val > 0.1 else f"<span style='color:red'>**{wf_f1_val:.4f}**</span>"
        else:
            wf_f1_fmt = "N/A"
        
        cols = [
            spk, g, r,
            _fmt(wf_acc), _fmt(wf_p), _fmt(wf_r), wf_f1_fmt,
            _fmt(pr_acc), _fmt(pr_p), _fmt(pr_r), _fmt(pr_f1),
            _fmt(kn_acc), _fmt(kn_p), _fmt(kn_r), _fmt(kn_f1),
            _fmt(df_acc), _fmt(df_p), _fmt(df_r), _fmt(df_f1),
            _fmt(cp_acc), _fmt(cp_p), _fmt(cp_r), _fmt(cp_f1),
        ]
        
        lines.append("| " + " | ".join(cols) + " |")
        
    lines.extend([
        "",
        "## Table 2: Performance on Different Subgroups",
        "",
        "> **Note:** All values are **macro-averages** per subgroup.",
        "",
        "| Group | WF Acc | WF P | WF R | WF F1 | Pr Acc | Pr P | Pr R | Pr F1 | Kn Acc | Kn P | Kn R | Kn F1 | DF Acc | DF P | DF R | DF F1 | Cr Acc | Cr P | Cr R | Cr F1 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    ])
    
    def add_table_2_row(label: str, cat: str, val: str):
        def _get_sub(df):
            sub = df[(df["split"] == cat) & (df["group"] == val)] if not df.empty else pd.DataFrame()
            return sub.iloc[0].to_dict() if not sub.empty else {}
            
        def _fmt(d, key):
            return f"{d.get(key, 0):.4f}" if d else "N/A"
            
        wf = _get_sub(wavfry_eval)
        df_sub = _get_sub(deepfry_eval)
        cp_sub = _get_sub(creapy_eval)
        pr_sub = _get_sub(praat_eval)
        kn_sub = _get_sub(kane_eval)
        
        wf_f1 = f"**{wf.get('f1', 0):.4f}**" if wf else "N/A"
        
        cols = [
            label,
            _fmt(wf, 'accuracy'), _fmt(wf, 'precision'), _fmt(wf, 'recall'), wf_f1,
            _fmt(pr_sub, 'accuracy'), _fmt(pr_sub, 'precision'), _fmt(pr_sub, 'recall'), _fmt(pr_sub, 'f1'),
            _fmt(kn_sub, 'accuracy'), _fmt(kn_sub, 'precision'), _fmt(kn_sub, 'recall'), _fmt(kn_sub, 'f1'),
            _fmt(df_sub, 'accuracy'), _fmt(df_sub, 'precision'), _fmt(df_sub, 'recall'), _fmt(df_sub, 'f1'),
            _fmt(cp_sub, 'accuracy'), _fmt(cp_sub, 'precision'), _fmt(cp_sub, 'recall'), _fmt(cp_sub, 'f1'),
        ]
        
        lines.append("| " + " | ".join(cols) + " |")
        
    add_table_2_row("Overall", "overall", "all")
    add_table_2_row("Female", "gender", "F")
    add_table_2_row("Male", "gender", "M")
    add_table_2_row("Beijing", "region", "Beijing")
    add_table_2_row("Guangdong", "region", "Guangdong")
    add_table_2_row("Sichuan", "region", "Sichuan")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="External Baseline Evaluation")
    parser.add_argument("--wavfry-config", type=str, default="06_wavfry/config/_defaults.yaml")
    parser.add_argument("--wavfry-checkpoint", type=str, default=r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE\04_creaky_tools\best_ep13_prauc0.9572_f10.8923_sampled.pt")
    parser.add_argument("--output", type=str, default="evaluate_setup_b_batch_{batch_id}.md")
    parser.add_argument("--cache-dir", type=str, default="_ext_eval_cache")
    parser.add_argument("--skip-wavfry", action="store_true")
    parser.add_argument("--skip-deepfry", action="store_true")
    parser.add_argument("--skip-creapy", action="store_true")
    parser.add_argument("--skip-praat", action="store_true")
    parser.add_argument("--skip-kane", action="store_true")
    parser.add_argument("--sample-json", type=str, default="02_en_mfa_aligned/_creaky_cache/sampling_state.json")
    parser.add_argument("--batch-id", type=int, default=2)
    args = parser.parse_args()
    
    wf_config_path = WORKSPACE_ROOT / args.wavfry_config
    out_path = BASELINES_DIR / args.output.format(batch_id=args.batch_id)
    cache_dir = WORKSPACE_ROOT / "02_en_mfa_aligned" / "_creaky_cache" / f"batch_{args.batch_id}"
    
    cfg = load_wavfry_config(wf_config_path)
    speakers = cfg.get("data", {}).get("annotated_speakers", [])
    audio_dir = WORKSPACE_ROOT / "01_en_dereverb"  # default en audio dir
    tg_dir = WORKSPACE_ROOT / "02_en_mfa_aligned"
    
    sample_json_path = WORKSPACE_ROOT / args.sample_json
    target_file_keys = None
    if sample_json_path.exists():
        with open(sample_json_path, "r", encoding="utf-8") as f:
            st = json.load(f)
            target_file_keys = set()
            batch_found = False
            for b in st.get("batches", []):
                if b.get("batch_id") == args.batch_id:
                    batch_found = True
                    b_split = str(b.get("split", "")).lower()
                    if b_split == "train":
                        raise ValueError(
                            f"[DATA LEAKAGE ERROR] Batch {args.batch_id} ('{b.get('name')}') is marked as split='{b_split}'! "
                            f"Running test benchmark on training data is strictly forbidden in academic evaluation."
                        )
                    for fh in b.get("files", []):
                        if fh.get("annotated", False):
                            target_file_keys.add(fh["file_key"])
                    break
            if not batch_found:
                raise ValueError(f"Batch ID {args.batch_id} not found in {args.sample_json}")
        print(f"[val_manifest] Loaded {len(target_file_keys)} sampled files for validation from {args.sample_json}")
    else:
        print(f"[val_manifest] Warning: {args.sample_json} not found. Falling back to default speakers from YAML config.")
    
    val_df = build_val_manifest(
        speakers=speakers,
        audio_dir=audio_dir,
        tg_dir=tg_dir,
        annotation_path=ANNOTATION_PATH,
        phone_tier=cfg.get("data", {}).get("phone_tier_name", "phones"),
        target_file_keys=target_file_keys
    )
    
    speaker_meta = {}
    for row in val_df.itertuples(index=False):
        speaker_meta[row.speaker_id] = {"gender": row.gender, "region": row.region}
        
    checkpoint_path = Path(args.wavfry_checkpoint)
    wavfry_eval = pd.DataFrame()
    if not args.skip_wavfry:
        if checkpoint_path.exists():
            wf_pred = run_wavfry(val_df, audio_dir, tg_dir, cache_dir, checkpoint_path)
            wavfry_eval = evaluate_by_groups(val_df, wf_pred)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found.")
            
    deepfry_eval = pd.DataFrame()
    if not args.skip_deepfry:
        df_pred = run_deepfry(val_df, audio_dir, cache_dir)
        deepfry_eval = evaluate_by_groups(val_df, df_pred)
        
    creapy_eval = pd.DataFrame()
    if not args.skip_creapy:
        cp_pred = run_creapy(val_df, audio_dir, tg_dir, cache_dir)
        creapy_eval = evaluate_by_groups(val_df, cp_pred)
        
    praat_eval = pd.DataFrame()
    if not args.skip_praat:
        try:
            pr_pred = run_praat(val_df, audio_dir, cache_dir)
            praat_eval = evaluate_by_groups(val_df, pr_pred)
        except NameError:
            print("Warning: run_praat not available.")
            
    kane_eval = pd.DataFrame()
    if not args.skip_kane:
        try:
            kn_pred = run_kane(val_df, audio_dir, cache_dir)
            kane_eval = evaluate_by_groups(val_df, kn_pred)
        except NameError:
            print("Warning: run_kane not available.")
            
    # Generate report with test details
    details = f"""
## Test Details
- **Batch ID**: {args.batch_id}
- **Number of Test Files**: {len(target_file_keys) if target_file_keys else val_df['file_key'].nunique()}
- **Number of Annotated Intervals**: {len(val_df)}
- **WavFry Checkpoint**: `{args.wavfry_checkpoint}`
- **Cache Directory**: `{cache_dir}`
    """
        
    write_report(out_path, wavfry_eval, deepfry_eval, creapy_eval, praat_eval, kane_eval, speaker_meta, details)
    print("[done] Evaluation complete.")

if __name__ == "__main__":
    main()
