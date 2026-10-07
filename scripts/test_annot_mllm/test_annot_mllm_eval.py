#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation script for MLLM breathy/whispery voice detection (testing phase).

Strict Execution Rules:
- Strictly uses the specified model (default: gemini-3.8-flash-high on https://aaawinn.xyz).
- NO fallback to any other model!
- Any API error (e.g., 503, 400, timeout) terminates the script immediately with an error.
- Resumable per-sample JSON cache.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from tqdm import tqdm

from test_annot_mllm_prompts import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE

# Reconfigure stdout for UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_ENDPOINT = "https://aaawinn.xyz"
DEFAULT_KEY = "ENV_EVAL_KEY_PLACEHOLDER"
DEFAULT_MODEL = "gemini-3.8-flash-high"
DEFAULT_MANIFEST = Path("manifests/test_annot_mllm/benchmark_manifest.json")
CACHE_DIR = Path("manifests/test_annot_mllm/api_cache")
REPORT_PATH = Path("manifests/test_annot_mllm/eval_report.json")


def encode_audio_base64(path: Path) -> Tuple[str, str]:
    """Read audio file and return base64 string and mime format ('wav' or 'flac')."""
    ext = path.suffix.lower().lstrip(".")
    if ext == "flac":
        fmt = "flac"
    elif ext in ("wav", "wave"):
        fmt = "wav"
    else:
        fmt = "wav"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, fmt


def call_gemini_api(
    audio_path: Path,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: int = 60
) -> Dict[str, Any]:
    """Call Gemini v1beta endpoint on the target service.
    
    Strict policy: Errors immediately if anything fails. NO FALLBACK.
    """
    b64_audio, fmt = encode_audio_base64(audio_path)
    url = f"{endpoint.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": USER_PROMPT_TEMPLATE},
                    {
                        "inline_data": {
                            "mime_type": f"audio/{fmt}",
                            "data": b64_audio
                        }
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.1
        }
    }

    try:
        r = requests.post(url, json=payload, timeout=timeout)
    except Exception as e:
        print(f"\n[FATAL ERROR] Network exception when calling model '{model}': {e}", file=sys.stderr)
        sys.exit(1)

    if r.status_code != 200:
        print(f"\n[FATAL ERROR] Model '{model}' request failed with HTTP {r.status_code}!", file=sys.stderr)
        print(f"Error Response Body:\n{r.text}", file=sys.stderr)
        sys.exit(1)

    res_data = r.json()
    candidates = res_data.get("candidates", [])
    if not candidates:
        print(f"\n[FATAL ERROR] Model '{model}' returned 200 OK but candidates list is empty!", file=sys.stderr)
        print(f"Full response:\n{json.dumps(res_data, indent=2, ensure_ascii=False)}", file=sys.stderr)
        sys.exit(1)

    content = candidates[0]["content"]["parts"][0]["text"]
    return {
        "status": "success",
        "model_used": model,
        "raw_content": content,
        "usage": res_data.get("usageMetadata", {})
    }


def parse_response_json(raw_text: str) -> Dict[str, Any]:
    """Extract JSON object from the response text."""
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text, re.IGNORECASE)
    if match:
        json_str = match.group(1)
        try:
            return json.loads(json_str)
        except Exception:
            pass

    brace_match = re.search(r"(\{[\s\S]*\})", raw_text)
    if brace_match:
        json_str = brace_match.group(1)
        try:
            return json.loads(json_str)
        except Exception:
            pass

    return {
        "has_breathy": "uncertain",
        "has_whispery": "uncertain",
        "has_creaky": "uncertain",
        "overall_confidence": "low",
        "segments": [],
        "parse_error": True
    }


def run_evaluation(
    manifest_path: Path,
    endpoint: str,
    api_key: str,
    model: str,
    limit: Optional[int] = None
):
    """Run evaluation on manifest strictly with the target model."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    samples = manifest_data["samples"]
    if limit is not None:
        samples = samples[:limit]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"Starting MLLM evaluation on {len(samples)} samples...")
    print(f"Endpoint: {endpoint}")
    print(f"Target Model: {model} (Strict mode: NO FALLBACK, errors abort immediately)\n")

    for idx, s in enumerate(tqdm(samples, desc="Evaluating")):
        sid = s["sample_id"]
        cache_file = CACHE_DIR / f"{sid}.json"

        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                res_record = json.load(f)
        else:
            audio_p = Path(s["audio_path"])
            api_res = call_gemini_api(audio_p, endpoint, api_key, model)

            parsed = parse_response_json(api_res["raw_content"])
            res_record = {
                "sample_id": sid,
                "source": s.get("source"),
                "gt_has_breathy": s.get("gt_has_breathy"),
                "gt_type": s.get("gt_type"),
                "gt_segments": s.get("gt_segments", []),
                "duration_sec": s.get("duration_sec"),
                "api_status": api_res["status"],
                "model_used": api_res.get("model_used"),
                "raw_reasoning": api_res.get("raw_content", ""),
                "prediction": parsed
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(res_record, f, indent=2, ensure_ascii=False)

            # Polite delay between requests
            time.sleep(1.0)

        results.append(res_record)

    compute_metrics(results)


def compute_metrics(results: List[Dict[str, Any]]):
    """Compute and display accuracy, precision, recall, and confusion matrix."""
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    uncertain_count = 0
    parse_error_count = 0

    trans_boundary_errors = []

    for r in results:
        gt = r["gt_has_breathy"]
        pred_obj = r["prediction"]
        pred_b = pred_obj.get("has_breathy")
        pred_w = pred_obj.get("has_whispery")

        if pred_b is True or pred_w is True:
            pred_pos = True
        elif pred_b == "uncertain" or pred_w == "uncertain":
            pred_pos = "uncertain"
        else:
            pred_pos = False

        if pred_obj.get("parse_error"):
            parse_error_count += 1

        if pred_pos == "uncertain":
            uncertain_count += 1
            continue

        if gt is True:
            if pred_pos is True:
                tp += 1
            else:
                fn += 1
        else:
            if pred_pos is False:
                tn += 1
            else:
                fp += 1

        if r.get("gt_type") == "transition_modal_to_breathy" and pred_pos is True:
            gt_segs = r.get("gt_segments", [])
            pred_segs = pred_obj.get("segments", [])
            if gt_segs and pred_segs:
                gt_start = gt_segs[0]["start_sec"]
                pred_starts = [seg.get("start_sec", 0.0) for seg in pred_segs if seg.get("phonation_type") in ("breathy", "whispery")]
                if pred_starts:
                    err = min(abs(ps - gt_start) for ps in pred_starts)
                    trans_boundary_errors.append(err)

    total_decided = tp + fp + tn + fn
    accuracy = (tp + tn) / total_decided if total_decided > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 55)
    print("           MLLM 发声态初筛测试评估报告")
    print("=" * 55)
    print(f"总样本数: {len(results)}")
    print(f"有效做出判定样本数: {total_decided} | 不确定(Uncertain)样本数: {uncertain_count} | 解析异常: {parse_error_count}")
    print("-" * 55)
    print(f"真阳性 (TP): {tp:3d}  | 假阳性 (FP): {fp:3d}")
    print(f"假阴性 (FN): {fn:3d}  | 真阴性 (TN): {tn:3d}")
    print("-" * 55)
    print(f"准确率 (Accuracy):    {accuracy * 100:6.2f}%")
    print(f"精确率 (Precision):   {precision * 100:6.2f}% (保守倾向判定)")
    print(f"召回率 (Recall):      {recall * 100:6.2f}%")
    print(f"特异度 (Specificity): {specificity * 100:6.2f}% (纯常态负样本不误报率)")
    print(f"F1-Score:             {f1:6.4f}")

    if trans_boundary_errors:
        mean_err = sum(trans_boundary_errors) / len(trans_boundary_errors)
        print(f"转折样本边界平均绝对误差 (MAE): {mean_err:.2f} 秒 (N={len(trans_boundary_errors)})")
    print("=" * 55)

    report = {
        "total_samples": len(results),
        "decided_samples": total_decided,
        "uncertain_count": uncertain_count,
        "parse_error_count": parse_error_count,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "specificity": round(specificity, 4),
            "f1": round(f1, 4),
            "transition_boundary_mae_sec": round(sum(trans_boundary_errors)/len(trans_boundary_errors), 3) if trans_boundary_errors else None
        },
        "confusion_matrix": {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn
        }
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"详细指标已保存至: {REPORT_PATH}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MLLM breathy detection")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST))
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT)
    parser.add_argument("--key", type=str, default=DEFAULT_KEY)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    run_evaluation(
        manifest_path=Path(args.manifest),
        endpoint=args.endpoint,
        api_key=args.key,
        model=args.model,
        limit=args.limit
    )


if __name__ == "__main__":
    main()
