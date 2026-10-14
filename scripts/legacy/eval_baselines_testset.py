# -*- coding: utf-8 -*-
"""
Comprehensive Baseline Evaluation Script for NCMMSC 2026 Camera-Ready.
Evaluates 4 baselines (Praat, Kane 2013, DeepFry, Creapy) on the strict,
leakage-free 2,392 test segments (43 unseen speakers, 95 audio files)
across Clean and 6 degraded acoustic environments.
"""

import os
import sys
import json
import time
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import soundfile as sf
import parselmouth
import tgt
import torch
import torchaudio

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
WORKSPACE_ROOT = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE")
BASELINES_DIR = Path(r"E:\Coding\WavFry\data\experiments\baselines")
CACHE_DIR = BASELINES_DIR / "_ext_eval_cache"

DEEPFRY_DIR = BASELINES_DIR / "DeepFry"
DEEPFRY_MODEL = DEEPFRY_DIR / "models" / "CREAK-220lr_0.001_decay_21_input_size_512_hidden_size_256_channels_512_normalize_False_measure_ff1_dropout_0.1_classes_3_.pth"
CREAPY_DIR = BASELINES_DIR / "creapy-main"
OCTAVE_CLI = Path(r"D:\Tools\octave-11.3.0-w64\mingw64\bin\octave-cli.exe")

MFA_TG_DIR = WORKSPACE_ROOT / "02_en_mfa_aligned"

ENVIRONMENTS = [
    {"name": "Clean", "dir": WORKSPACE_ROOT / "01_en_dereverb"},
    {"name": "20dB", "dir": WORKSPACE_ROOT / "01_en_dereverb_SNR20"},
    {"name": "10dB", "dir": WORKSPACE_ROOT / "01_en_dereverb_SNR10"},
    {"name": "0dB", "dir": WORKSPACE_ROOT / "01_en_dereverb_SNR0"},
    {"name": "-10dB", "dir": WORKSPACE_ROOT / "01_en_dereverb_SNR-10"},
    {"name": "-20dB", "dir": WORKSPACE_ROOT / "01_en_dereverb_SNR-20"},
    {"name": "Rev(0.5s)", "dir": WORKSPACE_ROOT / "01_en_dereverb_Reverb"},
]

MIN_OVERLAP_SEC = 0.020


def parse_gender(file_key: str) -> str:
    parts = str(file_key).split("_")
    return parts[0] if len(parts) >= 1 else "U"


def parse_region(file_key: str) -> str:
    parts = str(file_key).split("_")
    return parts[2] if len(parts) >= 3 else "Unknown"


def load_test_manifest() -> pd.DataFrame:
    """Unified test-set construction — must match eval_wavfry_testset exactly."""
    from wavfry.eval.unified import build_test_manifest

    manifest_path = CODE_ROOT / "manifests" / "segments_all.csv"
    full_df = pd.read_csv(manifest_path, low_memory=False)
    test_df = build_test_manifest(full_df, task="creaky")
    test_df["gender"] = test_df["file_key"].apply(parse_gender)
    test_df["region"] = test_df["file_key"].apply(parse_region)
    return test_df


def stage_audio_files(test_df: pd.DataFrame, src_dir: Path, target_dir: Path) -> Dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    file_keys = sorted(test_df["file_key"].unique().tolist())
    staged = {}
    for key in file_keys:
        dst = target_dir / f"{key}.wav"
        staged[key] = dst
        if dst.exists() and dst.stat().st_size > 1000:
            continue
        src_flac = src_dir / f"{key}.flac"
        src_wav = src_dir / f"{key}.wav"
        src = src_flac if src_flac.exists() else src_wav
        if not src.exists():
            print(f"[Warning] Audio file missing for {key} in {src_dir}")
            continue
        data, sr = sf.read(str(src), always_2d=False)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != 16000:
            t = torch.from_numpy(data).unsqueeze(0).float()
            t = torchaudio.functional.resample(t, sr, 16000)
            data = t.squeeze(0).numpy()
        sf.write(str(dst), np.asarray(data, dtype=np.float32), 16000, subtype="PCM_16")
    return staged


def run_praat_baseline(file_keys: List[str], staged_wavs: Dict[str, Path], cache_dir: Path) -> Dict[str, List[Tuple[float, float]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_spans = {}
    for key in file_keys:
        json_file = cache_dir / f"{key}.json"
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    all_spans[key] = [tuple(x) for x in json.load(f)]
                continue
            except Exception:
                pass

        wav_path = staged_wavs.get(key)
        if not wav_path or not wav_path.exists():
            all_spans[key] = []
            continue

        try:
            sound = parselmouth.Sound(str(wav_path))
            pitch = sound.to_pitch(time_step=0.015, pitch_floor=50.0, pitch_ceiling=350.0)
            freqs = pitch.selected_array['frequency']
            xs = pitch.xs()
            time_step = pitch.time_step
            half_step = time_step / 2.0

            spans = []
            in_unvoiced = False
            start_t = 0.0
            for i, f in enumerate(freqs):
                if f == 0:
                    if not in_unvoiced:
                        start_t = max(0.0, float(xs[i] - half_step))
                        in_unvoiced = True
                else:
                    if in_unvoiced:
                        end_t = float(xs[i - 1] + half_step)
                        spans.append((start_t, end_t))
                        in_unvoiced = False
            if in_unvoiced and len(xs) > 0:
                end_t = float(xs[-1] + half_step)
                spans.append((start_t, end_t))

            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(spans, f)
            all_spans[key] = spans
        except Exception as e:
            print(f"[Praat] Error on {key}: {e}")
            all_spans[key] = []
    return all_spans


def run_kane_baseline(file_keys: List[str], staged_wavs: Dict[str, Path], cache_dir: Path) -> Dict[str, List[Tuple[float, float]]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    missing_keys = []
    for key in file_keys:
        json_file = cache_dir / f"{key}.json"
        if not json_file.exists():
            missing_keys.append(key)

    if missing_keys:
        num_workers = min(8, os.cpu_count() or 4)
        chunks = [missing_keys[i::num_workers] for i in range(num_workers)]
        chunks = [c for c in chunks if c]

        from concurrent.futures import ThreadPoolExecutor

        def process_chunk(args: Tuple[int, List[str]]):
            chunk_idx, keys_subset = args
            list_file = cache_dir / f"wav_list_{chunk_idx}.txt"
            with open(list_file, "w", encoding="utf-8") as f:
                for k in keys_subset:
                    p = staged_wavs.get(k)
                    if p and p.exists():
                        f.write(f"{p.as_posix()}\n")

            cmd = [
                str(OCTAVE_CLI),
                "--eval",
                f"addpath('{BASELINES_DIR.as_posix()}'); batch_kane('{list_file.as_posix()}', '{cache_dir.as_posix()}');"
            ]
            subprocess.run(cmd, cwd=str(BASELINES_DIR), text=True, capture_output=True)

        with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
            list(executor.map(process_chunk, enumerate(chunks)))

    all_spans = {}
    for key in file_keys:
        json_file = cache_dir / f"{key}.json"
        if json_file.exists():
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    all_spans[key] = [tuple(x) for x in json.load(f)]
            except Exception:
                all_spans[key] = []
        else:
            all_spans[key] = []
    return all_spans


def run_deepfry_baseline(file_keys: List[str], stage_base_dir: Path, pred_dir: Path) -> Dict[str, List[Tuple[float, float]]]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    done_file = pred_dir / "_DONE.txt"

    missing_keys = [k for k in file_keys if not (pred_dir / f"{k}.TextGrid").exists()]
    if missing_keys:
        # Create an isolated staging directory containing only the missing keys
        sub_stage = pred_dir / "_sub_stage"
        sub_stage_test = sub_stage / "test"
        if sub_stage.exists():
            shutil.rmtree(sub_stage)
        sub_stage_test.mkdir(parents=True, exist_ok=True)
        staged_test = stage_base_dir / "test"
        for k in missing_keys:
            src_wav = staged_test / f"{k}.wav"
            if src_wav.exists():
                shutil.copy2(src_wav, sub_stage_test / f"{k}.wav")

        cmd = [
            sys.executable, str(DEEPFRY_DIR / "run.py"),
            "--data_dir", str(sub_stage),
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
            print(f"[DeepFry] Error:\n{proc.stderr}")
        if sub_stage.exists():
            shutil.rmtree(sub_stage)
        done_file.write_text("done", encoding="utf-8")

    all_spans = {}
    for key in file_keys:
        tg_path = pred_dir / f"{key}.TextGrid"
        spans = []
        if tg_path.exists():
            try:
                tg = tgt.io.read_textgrid(str(tg_path), encoding="utf-8")
                for tier in tg.tiers:
                    if "pred-creaky" in tier.name.lower():
                        for interval in tier:
                            if interval.text.strip().lower() == "c":
                                spans.append((float(interval.start_time), float(interval.end_time)))
                        break
            except Exception as e:
                print(f"[DeepFry] Error parsing TG {key}: {e}")
        all_spans[key] = spans
    return all_spans


def run_creapy_baseline(file_keys: List[str], staged_audio_dir: Path, pred_dir: Path) -> Dict[str, List[Tuple[float, float]]]:
    pred_dir.mkdir(parents=True, exist_ok=True)
    done_file = pred_dir / "_DONE.txt"

    # Identify files that don't have a valid creapy tier in their TextGrid
    missing_keys = []
    for k in file_keys:
        tg_path = pred_dir / f"{k}.TextGrid"
        if not tg_path.exists():
            missing_keys.append(k)
        else:
            try:
                tg = tgt.io.read_textgrid(str(tg_path), encoding="utf-8")
                if not any("creapy" in tier.name.lower() for tier in tg.tiers):
                    missing_keys.append(k)
            except Exception:
                missing_keys.append(k)

    if missing_keys:
        sub_stage_audio = pred_dir / "_sub_audio"
        if sub_stage_audio.exists():
            shutil.rmtree(sub_stage_audio)
        sub_stage_audio.mkdir(parents=True, exist_ok=True)

        for k in missing_keys:
            # Stage base textgrid if needed
            src_tg = MFA_TG_DIR / f"{k}.TextGrid"
            dst_tg = pred_dir / f"{k}.TextGrid"
            if src_tg.exists() and not dst_tg.exists():
                shutil.copy2(src_tg, dst_tg)
            src_wav = staged_audio_dir / f"{k}.wav"
            if src_wav.exists():
                shutil.copy2(src_wav, sub_stage_audio / f"{k}.wav")

        if str(CREAPY_DIR) not in sys.path:
            sys.path.insert(0, str(CREAPY_DIR))
        import creapy
        from creapy.model.classify import process_folder
        import warnings

        creapy.set_config(
            audio_suffix=".wav",
            textgrid_suffix=".TextGrid",
            gender_model="all",
            tier_name="creapy",
            filename_extension="",
            zcr_threshold=1.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            process_folder(
                audio_directory=str(sub_stage_audio),
                textgrid_directory=str(pred_dir),
                csv_directory=None,
            )
        if sub_stage_audio.exists():
            shutil.rmtree(sub_stage_audio)
        done_file.write_text("done", encoding="utf-8")

    all_spans = {}
    for key in file_keys:
        tg_path = pred_dir / f"{key}.TextGrid"
        spans = []
        if tg_path.exists():
            try:
                tg = tgt.io.read_textgrid(str(tg_path), encoding="utf-8")
                for tier in tg.tiers:
                    if "creapy" in tier.name.lower():
                        for interval in tier:
                            if interval.text.strip().lower() == "c":
                                spans.append((float(interval.start_time), float(interval.end_time)))
                        break
            except Exception as e:
                print(f"[Creapy] Error parsing TG {key}: {e}")
        all_spans[key] = spans
    return all_spans


def evaluate_predictions(test_df: pd.DataFrame, spans_by_file: Dict[str, List[Tuple[float, float]]]) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    preds = []
    for row in test_df.itertuples(index=False):
        spans = spans_by_file.get(str(row.file_key), [])
        start_s = float(row.start_s)
        end_s = float(row.end_s)
        dur = max(1e-6, end_s - start_s)
        required = MIN_OVERLAP_SEC

        ov_total = 0.0
        for s, e in spans:
            ov = min(end_s, float(e)) - max(start_s, float(s))
            if ov > 0.0:
                ov_total += ov
        preds.append(1 if ov_total >= required else 0)

    work_df = test_df.copy()
    work_df["pred"] = np.array(preds, dtype=np.int32)
    work_df["gold"] = work_df["label"].to_numpy().astype(np.int32)

    # 1. Segment-level pooled metrics
    y_true = work_df["gold"].to_numpy()
    y_pred = work_df["pred"].to_numpy()
    tp = float(np.sum((y_true == 1) & (y_pred == 1)))
    tn = float(np.sum((y_true == 0) & (y_pred == 0)))
    fp = float(np.sum((y_true == 0) & (y_pred == 1)))
    fn = float(np.sum((y_true == 1) & (y_pred == 0)))
    n = float(len(y_true))

    pooled_acc = (tp + tn) / n if n > 0 else 0.0
    pooled_p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pooled_r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pooled_f1 = 2.0 * pooled_p * pooled_r / (pooled_p + pooled_r) if (pooled_p + pooled_r) > 0 else 0.0

    pooled_metrics = {
        "accuracy": pooled_acc,
        "precision": pooled_p,
        "recall": pooled_r,
        "f1": pooled_f1,
    }

    # 2. Speaker-level macro metrics
    spk_metrics = []
    for spk_id, gdf in work_df.groupby("speaker_id"):
        yt = gdf["gold"].to_numpy()
        yp = gdf["pred"].to_numpy()
        s_tp = float(np.sum((yt == 1) & (yp == 1)))
        s_tn = float(np.sum((yt == 0) & (yp == 0)))
        s_fp = float(np.sum((yt == 0) & (yp == 1)))
        s_fn = float(np.sum((yt == 1) & (yp == 0)))
        s_n = float(len(yt))

        s_acc = (s_tp + s_tn) / s_n if s_n > 0 else 0.0
        s_p = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
        s_r = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
        s_f1 = 2.0 * s_p * s_r / (s_p + s_r) if (s_p + s_r) > 0 else 0.0

        spk_metrics.append({
            "speaker_id": spk_id,
            "gender": gdf["gender"].iloc[0],
            "region": gdf["region"].iloc[0],
            "accuracy": s_acc,
            "precision": s_p,
            "recall": s_r,
            "f1": s_f1,
        })
    spk_df = pd.DataFrame(spk_metrics)

    subgroups = {
        "overall": {
            "accuracy": float(spk_df["accuracy"].mean()),
            "precision": float(spk_df["precision"].mean()),
            "recall": float(spk_df["recall"].mean()),
            "f1": float(spk_df["f1"].mean()),
        }
    }
    for g in ["F", "M"]:
        sub = spk_df[spk_df["gender"] == g]
        subgroups["Female" if g == "F" else "Male"] = {
            "accuracy": float(sub["accuracy"].mean()) if not sub.empty else 0.0,
            "precision": float(sub["precision"].mean()) if not sub.empty else 0.0,
            "recall": float(sub["recall"].mean()) if not sub.empty else 0.0,
            "f1": float(sub["f1"].mean()) if not sub.empty else 0.0,
        }
    for r in ["Beijing", "Guangdong", "Sichuan"]:
        sub = spk_df[spk_df["region"] == r]
        subgroups[r] = {
            "accuracy": float(sub["accuracy"].mean()) if not sub.empty else 0.0,
            "precision": float(sub["precision"].mean()) if not sub.empty else 0.0,
            "recall": float(sub["recall"].mean()) if not sub.empty else 0.0,
            "f1": float(sub["f1"].mean()) if not sub.empty else 0.0,
        }

    return pooled_metrics, subgroups


def main():
    print("=" * 70)
    print("  External Baselines Evaluation on Unified Test Set (Clean + Noise)")
    print("=" * 70)

    test_df = load_test_manifest()
    file_keys = sorted(test_df["file_key"].unique().tolist())
    print(f"Loaded {len(test_df)} test segments across {test_df['speaker_id'].nunique()} speakers and {len(file_keys)} audio files.")

    all_results: Dict[str, Any] = {
        "test_segments": len(test_df),
        "test_speakers": int(test_df["speaker_id"].nunique()),
        "test_files": len(file_keys),
        "baselines": {
            "Praat": {},
            "Kane": {},
            "DeepFry": {},
            "Creapy": {},
        }
    }

    for env in ENVIRONMENTS:
        env_name = env["name"]
        audio_dir = env["dir"]
        print(f"\n>>> Processing Environment: {env_name} (Dir: {audio_dir.name}) <<<")

        if not audio_dir.exists():
            print(f"[Warning] Directory {audio_dir} does not exist! Skipping.")
            continue

        # 1. Stage audio
        env_cache = CACHE_DIR / env_name
        deepfry_stage_base = env_cache / "deepfry_stage"
        staged_test_dir = deepfry_stage_base / "test"
        print("[1/5] Staging audio to 16kHz PCM_16 WAV...", flush=True)
        staged_wavs = stage_audio_files(test_df, audio_dir, staged_test_dir)

        # 2. Praat
        print("[2/5] Running Praat baseline...", flush=True)
        praat_spans = run_praat_baseline(file_keys, staged_wavs, env_cache / "praat")
        praat_pooled, praat_subgroups = evaluate_predictions(test_df, praat_spans)
        all_results["baselines"]["Praat"][env_name] = {"pooled": praat_pooled, "subgroups": praat_subgroups}
        print(f"  Praat [{env_name}]: F1={praat_pooled['f1']:.4f}, Acc={praat_pooled['accuracy']:.4f}, P={praat_pooled['precision']:.4f}, R={praat_pooled['recall']:.4f} (Speaker Macro F1={praat_subgroups['overall']['f1']:.4f})", flush=True)

        # 3. Kane
        print("[3/5] Running Kane (2013) baseline...", flush=True)
        kane_spans = run_kane_baseline(file_keys, staged_wavs, env_cache / "kane")
        kane_pooled, kane_subgroups = evaluate_predictions(test_df, kane_spans)
        all_results["baselines"]["Kane"][env_name] = {"pooled": kane_pooled, "subgroups": kane_subgroups}
        print(f"  Kane  [{env_name}]: F1={kane_pooled['f1']:.4f}, Acc={kane_pooled['accuracy']:.4f}, P={kane_pooled['precision']:.4f}, R={kane_pooled['recall']:.4f} (Speaker Macro F1={kane_subgroups['overall']['f1']:.4f})", flush=True)

        # 4. DeepFry
        print("[4/5] Running DeepFry baseline...", flush=True)
        deepfry_spans = run_deepfry_baseline(file_keys, deepfry_stage_base, env_cache / "deepfry_pred")
        df_pooled, df_subgroups = evaluate_predictions(test_df, deepfry_spans)
        all_results["baselines"]["DeepFry"][env_name] = {"pooled": df_pooled, "subgroups": df_subgroups}
        print(f"  DeepFry [{env_name}]: F1={df_pooled['f1']:.4f}, Acc={df_pooled['accuracy']:.4f}, P={df_pooled['precision']:.4f}, R={df_pooled['recall']:.4f} (Speaker Macro F1={df_subgroups['overall']['f1']:.4f})", flush=True)

        # 5. Creapy
        print("[5/5] Running Creapy baseline...", flush=True)
        creapy_spans = run_creapy_baseline(file_keys, staged_test_dir, env_cache / "creapy_pred")
        cp_pooled, cp_subgroups = evaluate_predictions(test_df, creapy_spans)
        all_results["baselines"]["Creapy"][env_name] = {"pooled": cp_pooled, "subgroups": cp_subgroups}
        print(f"  Creapy [{env_name}]: F1={cp_pooled['f1']:.4f}, Acc={cp_pooled['accuracy']:.4f}, P={cp_pooled['precision']:.4f}, R={cp_pooled['recall']:.4f} (Speaker Macro F1={cp_subgroups['overall']['f1']:.4f})", flush=True)

    # Save all results
    out_json = BASELINES_DIR / "all_baselines_eval_results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Done] All Baseline results successfully saved to: {out_json}")


if __name__ == "__main__":
    main()

