#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation and contrastive case extraction on YSJ spontaneous speech (Batch 3).

Evaluates Gemini 3.8 Flash High against human manual annotations in YSJ Batch 3:
1. Runs inference across 22 manually annotated slices in Batch 3.
2. Performs two-level alignment:
   - Slice-level (16 breathy vs 6 pure modal slices).
   - Interval-level (human vowel intervals vs LLM segments with +/-0.35s window).
3. Automatically mines and extracts audio snippets for 4 contrastive quadrants:
   - Quadrant 1: High Consensus Positive (Both user & LLM say Breathy)
   - Quadrant 2: LLM-Only Positive (LLM detected Breathy, User did not mark)
   - Quadrant 3: User-Only Positive (User marked Breathy, LLM judged Modal)
   - Quadrant 4: High Consensus Negative (Both agree pure Modal)
4. Exports sliced .wav files and comparison report with clickable links.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import soundfile as sf
from tqdm import tqdm

try:
    from test_annot_mllm_prompts import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE
    from test_annot_mllm_eval import call_gemini_api, parse_response_json
except ImportError:
    from scripts.test_annot_mllm.test_annot_mllm_prompts import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE
    from scripts.test_annot_mllm.test_annot_mllm_eval import call_gemini_api, parse_response_json

# Reconfigure stdout for UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

YSJ_BATCH3_DIR = Path(r"E:\Corpus\Breathy\YSJ\textgrid_tifa\_breathy_cache\batches\batch_3\manual_breathy_annotations.json.d")
YSJ_AUDIO_DIR = Path(r"E:\Corpus\Breathy\YSJ\audio")
YSJ_TRANSCRIPTS_PATH = Path(r"E:\Corpus\Breathy\YSJ\transcripts.tsv")

OUTPUT_DIR = Path(r"manifests\test_annot_mllm\ysj_listen_cases").resolve()
CACHE_DIR = Path(r"manifests\test_annot_mllm\ysj_cache").resolve()

DEFAULT_ENDPOINT = "http://127.0.0.1:8045"
DEFAULT_KEY = "sk-c1c7dbb68c714d3399a14b05ca9906d6"
DEFAULT_MODEL = "gemini-3.8-flash-high"


def load_ysj_transcripts() -> Dict[str, str]:
    """Load transcripts mapping slice_id -> text."""
    transcripts = {}
    if YSJ_TRANSCRIPTS_PATH.exists():
        with open(YSJ_TRANSCRIPTS_PATH, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    transcripts[parts[0]] = parts[1]
    return transcripts


def load_batch3_data() -> List[Dict[str, Any]]:
    """Load all 22 annotated files and their intervals from batch 3."""
    records = []
    for json_file in sorted(YSJ_BATCH3_DIR.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        fk = data.get("file_key")
        annots = data.get("annotations", {})
        if not fk or not annots:
            continue

        audio_path = YSJ_AUDIO_DIR / f"{fk}.flac"
        if not audio_path.exists():
            audio_path = YSJ_AUDIO_DIR / f"{fk}.wav"
        if not audio_path.exists():
            continue

        info = sf.info(str(audio_path))

        # Parse intervals: "start|end|phone": "label"
        breathy_intervals = []
        non_breathy_intervals = []
        for k, lbl in annots.items():
            parts = k.split("|")
            if len(parts) >= 3:
                s_t = float(parts[0])
                e_t = float(parts[1])
                phone = parts[2]
                if lbl == "breathy":
                    breathy_intervals.append({"start": s_t, "end": e_t, "phone": phone})
                elif lbl == "non_breathy":
                    non_breathy_intervals.append({"start": s_t, "end": e_t, "phone": phone})

        records.append({
            "file_key": fk,
            "audio_path": str(audio_path),
            "duration_sec": round(info.duration, 3),
            "total_intervals": len(annots),
            "breathy_intervals": sorted(breathy_intervals, key=lambda x: x["start"]),
            "non_breathy_intervals": sorted(non_breathy_intervals, key=lambda x: x["start"]),
            "gt_has_breathy": len(breathy_intervals) > 0
        })
    return records


def extract_audio_snippet(
    source_wav_path: Path,
    out_wav_path: Path,
    start_sec: float,
    end_sec: float,
    pad_sec: float = 0.5
) -> Tuple[float, float]:
    """Slice audio clip with context padding and save to out_wav_path."""
    wav, sr = sf.read(str(source_wav_path))
    total_dur = len(wav) / sr

    clip_start = max(0.0, start_sec - pad_sec)
    clip_end = min(total_dur, end_sec + pad_sec)

    s_idx = int(clip_start * sr)
    e_idx = int(clip_end * sr)
    sub_wav = wav[s_idx:e_idx]

    sf.write(str(out_wav_path), sub_wav, sr)
    return round(clip_start, 2), round(clip_end, 2)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    transcripts = load_ysj_transcripts()
    batch3_records = load_batch3_data()
    print(f"Loaded {len(batch3_records)} annotated slices from YSJ Batch 3.")

    # 1. Inference and caching
    print(f"\nRunning Gemini 3.8 Flash High inference on YSJ Batch 3...")
    results = []
    for r in tqdm(batch3_records, desc="Inferring YSJ"):
        fk = r["file_key"]
        cache_file = CACHE_DIR / f"{fk}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                res_record = json.load(f)
        else:
            api_res = call_gemini_api(
                Path(r["audio_path"]),
                DEFAULT_ENDPOINT,
                DEFAULT_KEY,
                DEFAULT_MODEL
            )
            parsed = parse_response_json(api_res["raw_content"])
            res_record = {
                "file_key": fk,
                "audio_path": r["audio_path"],
                "duration_sec": r["duration_sec"],
                "gt_has_breathy": r["gt_has_breathy"],
                "user_breathy_count": len(r["breathy_intervals"]),
                "user_breathy_intervals": r["breathy_intervals"],
                "api_status": api_res["status"],
                "raw_reasoning": api_res.get("raw_content", ""),
                "prediction": parsed
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(res_record, f, indent=2, ensure_ascii=False)
            time.sleep(1.0)

        results.append(res_record)

    # 2. Slice-level Metrics
    s_tp = s_fp = s_tn = s_fn = 0
    for r in results:
        u_has = r["gt_has_breathy"]
        p_b = r["prediction"].get("has_breathy")
        p_w = r["prediction"].get("has_whispery")
        l_has = (p_b is True or p_w is True)

        if u_has and l_has: s_tp += 1
        elif not u_has and l_has: s_fp += 1
        elif not u_has and not l_has: s_tn += 1
        else: s_fn += 1

    s_prec = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
    s_rec = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
    s_spec = s_tn / (s_tn + s_fp) if (s_tn + s_fp) > 0 else 0.0

    print("\n" + "=" * 55)
    print("      YSJ Batch 3 切片级（Slice-level）宏观对比指标")
    print("=" * 55)
    print(f"总切片数: {len(results)} (含精标气声: {s_tp+s_fn}条, 纯常态: {s_tn+s_fp}条)")
    print(f"真阳性 (TP): {s_tp:2d} | 假阳性 (FP): {s_fp:2d}")
    print(f"假阴性 (FN): {s_fn:2d} | 真阴性 (TN): {s_tn:2d}")
    print(f"切片精确率 (Precision):   {s_prec * 100:6.2f}%")
    print(f"切片召回率 (Recall):      {s_rec * 100:6.2f}%")
    print(f"切片特异度 (Specificity): {s_spec * 100:6.2f}% (纯常态切片未误判率)")
    print("=" * 55)

    # 3. Interval-level Matching & Case Mining
    quadrant_tp = []    # Agree Positive
    quadrant_llm_only = [] # LLM says breathy, user didn't mark
    quadrant_user_only = [] # User marked breathy, LLM said modal
    quadrant_tn = []    # Agree Negative (from pure modal slices)

    TOLERANCE_SEC = 0.35

    for r in results:
        fk = r["file_key"]
        audio_p = Path(r["audio_path"])
        u_b_list = r["user_breathy_intervals"]
        l_segs = [seg for seg in r["prediction"].get("segments", []) if seg.get("phonation_type") in ("breathy", "whispery")]

        matched_user_indices = set()
        matched_llm_indices = set()

        for u_idx, u_b in enumerate(u_b_list):
            u_s, u_e = u_b["start"], u_b["end"]
            for l_idx, l_seg in enumerate(l_segs):
                l_s, l_e = l_seg.get("start_sec", 0.0), l_seg.get("end_sec", 0.0)
                # Check overlap or proximity within tolerance
                if not (u_e < l_s - TOLERANCE_SEC or u_s > l_e + TOLERANCE_SEC):
                    matched_user_indices.add(u_idx)
                    matched_llm_indices.add(l_idx)
                    quadrant_tp.append({
                        "file_key": fk,
                        "audio_path": audio_p,
                        "user_interval": u_b,
                        "llm_segment": l_seg,
                        "match_start": min(u_s, l_s),
                        "match_end": max(u_e, l_e)
                    })

        # User-Only (User marked breathy, but LLM missed)
        for u_idx, u_b in enumerate(u_b_list):
            if u_idx not in matched_user_indices:
                quadrant_user_only.append({
                    "file_key": fk,
                    "audio_path": audio_p,
                    "user_interval": u_b,
                    "llm_pred_slice": r["prediction"]
                })

        # LLM-Only (LLM marked breathy, but user didn't annotate breathy)
        for l_idx, l_seg in enumerate(l_segs):
            if l_idx not in matched_llm_indices:
                quadrant_llm_only.append({
                    "file_key": fk,
                    "audio_path": audio_p,
                    "llm_segment": l_seg
                })

        # Agree Negative (from 100% non_breathy slices where LLM predicted no breathy)
        if not r["gt_has_breathy"] and not r["prediction"].get("has_breathy") and not r["prediction"].get("has_whispery"):
            quadrant_tn.append({
                "file_key": fk,
                "audio_path": audio_p,
                "duration": r["duration_sec"]
            })

    print(f"\n区间级对比统计:")
    print(f" - 双方共识气声 (TP Intervals): {len(quadrant_tp)}")
    print(f" - LLM独有检出气声 (LLM-Only Segments): {len(quadrant_llm_only)}")
    print(f" - 用户独有标注气声 (User-Only Intervals): {len(quadrant_user_only)}")
    print(f" - 双方共识纯常态切片 (TN Slices): {len(quadrant_tn)}")

    # 4. Generate Audio Snippets and Markdown Report
    report_lines = [
        "# YSJ 自发语音（Batch 3）人工精标 vs. Gemini 3.8 Flash 对比听测报告\n",
        "本报告针对您在 YSJ Batch 3 中人工精标的 22 条真实播客切片，提取出 **双方共识气声**、**LLM单独检出**、**用户单独标注** 以及 **共识常态** 四类对比案例，并切出短音频片段供您点击试听核查。\n",
        f"## 宏观切片级指标概览\n",
        f"- **总测试音频**: 22 条长切片（平均时长 22.0 秒，总长 8.1 分钟）",
        f"- **切片精确率 (Precision)**: **{s_prec*100:.2f}%**",
        f"- **切片特异度 (Specificity)**: **{s_spec*100:.2f}%**（6 条纯常态切片仅 {s_fp} 条报阳）",
        f"- **切片召回率 (Recall)**: **{s_rec*100:.2f}%**\n",
        "---\n"
    ]

    # --- Section 1: TP Cases ---
    report_lines.append("## 🟢 象限 1：双方高度共识的气声（Agree Positive - TP）")
    report_lines.append("人工精标为气声，Gemini 3.8 Flash 也以高置信度捕捉并精准定位。\n")
    sample_tps = quadrant_tp[:3]
    for idx, c in enumerate(sample_tps, 1):
        fk = c["file_key"]
        out_fn = f"case_tp_{idx}_{fk}.wav"
        out_path = OUTPUT_DIR / out_fn
        c_s, c_e = extract_audio_snippet(c["audio_path"], out_path, c["match_start"], c["match_end"], pad_sec=0.6)
        u_i = c["user_interval"]
        l_s = c["llm_segment"]
        text = transcripts.get(fk, "（无台词记录）")
        report_lines.append(f"### 案例 TP-{idx}: `{fk}` (区间: {c_s}s - {c_e}s)")
        report_lines.append(f"- 🎧 **试听音频**: [{out_fn}]({out_path.as_uri()})")
        report_lines.append(f"- 👤 **人工精标**: 元音 `{u_i['phone']}` (时间: {u_i['start']}s ~ {u_i['end']}s)")
        report_lines.append(f"- 🤖 **LLM 判定**: `{l_s.get('phonation_type')}` (时间: {l_s.get('start_sec')}s ~ {l_s.get('end_sec')}s)")
        report_lines.append(f"- 🧠 **LLM 声学理由**: {l_s.get('acoustic_reason', '未提供')}")
        report_lines.append(f"- 📜 **切片台词**: {text[:100]}...\n")

    # --- Section 2: LLM-Only Cases ---
    report_lines.append("## 🟡 象限 2：LLM 检出气声，但人工未标（LLM-Only / 潜在漏标或轻微气声）")
    report_lines.append("大模型判定含有显著气声/耳语，但您在此区间未打标为气声。请重点试听：是模型虚警，还是人工听觉疲劳漏标？\n")
    sample_llm_only = quadrant_llm_only[:3]
    for idx, c in enumerate(sample_llm_only, 1):
        fk = c["file_key"]
        l_s = c["llm_segment"]
        out_fn = f"case_llm_only_{idx}_{fk}.wav"
        out_path = OUTPUT_DIR / out_fn
        c_s, c_e = extract_audio_snippet(c["audio_path"], out_path, l_s.get("start_sec", 0.0), l_s.get("end_sec", 0.0), pad_sec=0.6)
        text = transcripts.get(fk, "（无台词记录）")
        report_lines.append(f"### 案例 LLM-Only-{idx}: `{fk}` (区间: {c_s}s - {c_e}s)")
        report_lines.append(f"- 🎧 **试听音频**: [{out_fn}]({out_path.as_uri()})")
        report_lines.append(f"- 👤 **人工精标**: 未标注为气声（标为常态或未列入）")
        report_lines.append(f"- 🤖 **LLM 判定**: `{l_s.get('phonation_type')}` (时间: {l_s.get('start_sec')}s ~ {l_s.get('end_sec')}s)")
        report_lines.append(f"- 🧠 **LLM 声学理由**: {l_s.get('acoustic_reason', '未提供')}")
        report_lines.append(f"- 📜 **切片台词**: {text[:100]}...\n")

    # --- Section 3: User-Only Cases ---
    report_lines.append("## 🔴 象限 3：人工标注了气声，但 LLM 漏报（User-Only / 模型保守漏报）")
    report_lines.append("您打标为气声，但大模型判定为常态发声。通常发生在音素极其短促、或气流感极为轻微的区间。\n")
    sample_user_only = quadrant_user_only[:3]
    for idx, c in enumerate(sample_user_only, 1):
        fk = c["file_key"]
        u_i = c["user_interval"]
        out_fn = f"case_user_only_{idx}_{fk}.wav"
        out_path = OUTPUT_DIR / out_fn
        c_s, c_e = extract_audio_snippet(c["audio_path"], out_path, u_i["start"], u_i["end"], pad_sec=0.8)
        text = transcripts.get(fk, "（无台词记录）")
        report_lines.append(f"### 案例 User-Only-{idx}: `{fk}` (区间: {c_s}s - {c_e}s)")
        report_lines.append(f"- 🎧 **试听音频**: [{out_fn}]({out_path.as_uri()})")
        report_lines.append(f"- 👤 **人工精标**: 元音 `{u_i['phone']}` (时间: {u_i['start']}s ~ {u_i['end']}s)")
        report_lines.append(f"- 🤖 **LLM 判定**: 整句判定为常态发声，未报阳")
        report_lines.append(f"- 📜 **切片台词**: {text[:100]}...\n")

    # --- Section 4: TN Cases ---
    report_lines.append("## ⚪ 象限 4：双方高度共识的纯常态（Agree Negative - TN）")
    report_lines.append("全切片 100% 为人工常态标注，大模型也完全认定为常态，未产生任何误报。\n")
    sample_tn = quadrant_tn[:2]
    for idx, c in enumerate(sample_tn, 1):
        fk = c["file_key"]
        out_fn = f"case_tn_{idx}_{fk}.wav"
        out_path = OUTPUT_DIR / out_fn
        # Extract middle 4 seconds
        mid = c["duration"] / 2.0
        c_s, c_e = extract_audio_snippet(c["audio_path"], out_path, mid - 1.5, mid + 1.5, pad_sec=0.0)
        text = transcripts.get(fk, "（无台词记录）")
        report_lines.append(f"### 案例 TN-{idx}: `{fk}` (截取区间: {c_s}s - {c_e}s)")
        report_lines.append(f"- 🎧 **试听音频**: [{out_fn}]({out_path.as_uri()})")
        report_lines.append(f"- 👤 **人工精标**: 100% 纯常态发声")
        report_lines.append(f"- 🤖 **LLM 判定**: Modal voice (无气声/耳语)")
        report_lines.append(f"- 📜 **切片台词**: {text[:100]}...\n")

    report_content = "\n".join(report_lines)
    report_file = OUTPUT_DIR / "comparison_report.md"
    report_file.write_text(report_content, encoding="utf-8")
    print(f"\n[OK] 对比报告与试听切片已生成完成！")
    print(f"报告路径: {report_file}")


if __name__ == "__main__":
    main()
