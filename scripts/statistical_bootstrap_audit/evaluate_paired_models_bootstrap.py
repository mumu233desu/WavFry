#!/usr/bin/env python
"""
WavFry Paired Model Cluster Bootstrap Significance Audit.

Performs rigorous non-parametric speaker-level cluster bootstrap (B=1000)
to evaluate paired differences between two models (e.g. Weak-Supervision On vs Off)
across unseen speakers.

Methodology:
- Cluster Bootstrap Resampling (B=1000 iterations over N speakers with replacement)
- Eliminates sliding window temporal pseudo-replication
- Computes Paired Delta Delta = Metric(Model A) - Metric(Model B)
- 95% Percentile Confidence Intervals [2.5%, 97.5%]
- Davison & Hinkley (1997) continuity-smoothed two-sided empirical p-values
- Speaker-level Win/Loss breakdown

Author: WavFry ML Infrastructure
Date: October 2026
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from wavfry.data import splits as splits_mod
from wavfry.eval.acoustic_degradation import get_acoustic_degradation_engine, parse_condition_str
from wavfry.model.wavlm_classifier import WavFryClassifier


def resolve_strict_checkpoint(path_str: Optional[str], label: str) -> Path:
    """Resolve checkpoint with strict zero-guessing policy."""
    if not path_str:
        raise ValueError(f"Explicit path for {label} is REQUIRED under Zero-Guessing policy.")
    p = Path(path_str)
    if not p.is_file():
        raise FileNotFoundError(f"Specified checkpoint for {label} not found: {p}")
    return p


def load_model(ckpt_path: Path, device: torch.device) -> tuple[WavFryClassifier, Dict[str, Any]]:
    print(f"[Loading] Loading model from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = json.loads(json.dumps(ckpt.get("config", {})))

    model = WavFryClassifier(config)
    sd = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    cleaned_sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    missing_keys, unexpected_keys = model.load_state_dict(cleaned_sd, strict=False)
    critical_missing = [k for k in missing_keys if not k.startswith("wavlm.")]
    if critical_missing:
        raise RuntimeError(f"Fail-Fast: Required classification weights missing from {ckpt_path}: {critical_missing}")

    model.to(device)
    model.eval()

    for p in model.wavlm.parameters():
        p.requires_grad = False

    return model, config


def run_speaker_cluster_bootstrap_paired(
    preds_a_by_spk: Dict[str, np.ndarray],
    preds_b_by_spk: Dict[str, np.ndarray],
    labels_by_spk: Dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform non-parametric speaker-level cluster bootstrap for paired comparison.
    """
    np.random.seed(seed)
    unique_speakers = sorted(list(labels_by_spk.keys()))
    n_spk = len(unique_speakers)

    boot_pr_a = []
    boot_pr_b = []
    boot_delta_pr = []

    boot_roc_a = []
    boot_roc_b = []
    boot_delta_roc = []

    for _ in range(n_bootstrap):
        sampled_spks = np.random.choice(unique_speakers, size=n_spk, replace=True)

        b_labels = np.concatenate([labels_by_spk[s] for s in sampled_spks])
        valid_m = ~np.isnan(b_labels)
        b_labels = b_labels[valid_m].astype(int)

        if len(np.unique(b_labels)) < 2:
            continue

        b_logits_a = np.concatenate([preds_a_by_spk[s] for s in sampled_spks])[valid_m]
        b_logits_b = np.concatenate([preds_b_by_spk[s] for s in sampled_spks])[valid_m]

        b_probs_a = 1.0 / (1.0 + np.exp(-b_logits_a))
        b_probs_b = 1.0 / (1.0 + np.exp(-b_logits_b))

        pr_a = average_precision_score(b_labels, b_probs_a)
        pr_b = average_precision_score(b_labels, b_probs_b)
        delta_pr = pr_a - pr_b

        roc_a = roc_auc_score(b_labels, b_probs_a)
        roc_b = roc_auc_score(b_labels, b_probs_b)
        delta_roc = roc_a - roc_b

        boot_pr_a.append(pr_a)
        boot_pr_b.append(pr_b)
        boot_delta_pr.append(delta_pr)

        boot_roc_a.append(roc_a)
        boot_roc_b.append(roc_b)
        boot_delta_roc.append(delta_roc)

    boot_delta_pr = np.array(boot_delta_pr)
    boot_delta_roc = np.array(boot_delta_roc)
    n_valid = len(boot_delta_pr)

    # Davison & Hinkley (1997) continuity-smoothed empirical p-value for H0: delta = 0
    count_pos = np.sum(boot_delta_pr >= 0)
    count_neg = np.sum(boot_delta_pr <= 0)
    p_val_pr = float(2.0 * min((1 + count_pos) / (n_valid + 1), (1 + count_neg) / (n_valid + 1)))
    p_val_pr = min(1.0, p_val_pr)

    count_pos_roc = np.sum(boot_delta_roc >= 0)
    count_neg_roc = np.sum(boot_delta_roc <= 0)
    p_val_roc = float(2.0 * min((1 + count_pos_roc) / (n_valid + 1), (1 + count_neg_roc) / (n_valid + 1)))
    p_val_roc = min(1.0, p_val_roc)

    # Speaker win / tie / loss breakdown
    spk_wins_a = 0
    spk_wins_b = 0
    spk_ties = 0
    spk_deltas = []

    for s in unique_speakers:
        s_y = labels_by_spk[s]
        valid_s = ~np.isnan(s_y)
        s_y = s_y[valid_s].astype(int)
        if len(np.unique(s_y)) >= 2:
            p_a = 1.0 / (1.0 + np.exp(-preds_a_by_spk[s][valid_s]))
            p_b = 1.0 / (1.0 + np.exp(-preds_b_by_spk[s][valid_s]))
            s_pr_a = average_precision_score(s_y, p_a)
            s_pr_b = average_precision_score(s_y, p_b)
            diff = s_pr_a - s_pr_b
            spk_deltas.append(diff)
            if diff > 1e-4:
                spk_wins_a += 1
            elif diff < -1e-4:
                spk_wins_b += 1
            else:
                spk_ties += 1

    return {
        "n_speakers": n_spk,
        "n_bootstrap": n_valid,
        "pr_a_mean": float(np.mean(boot_pr_a)),
        "pr_a_ci": [float(np.percentile(boot_pr_a, 2.5)), float(np.percentile(boot_pr_a, 97.5))],
        "pr_b_mean": float(np.mean(boot_pr_b)),
        "pr_b_ci": [float(np.percentile(boot_pr_b, 2.5)), float(np.percentile(boot_pr_b, 97.5))],
        "delta_pr_mean": float(np.mean(boot_delta_pr)),
        "delta_pr_ci": [float(np.percentile(boot_delta_pr, 2.5)), float(np.percentile(boot_delta_pr, 97.5))],
        "p_val_pr": p_val_pr,
        "roc_a_mean": float(np.mean(boot_roc_a)),
        "roc_a_ci": [float(np.percentile(boot_roc_a, 2.5)), float(np.percentile(boot_roc_a, 97.5))],
        "roc_b_mean": float(np.mean(boot_roc_b)),
        "roc_b_ci": [float(np.percentile(boot_roc_b, 2.5)), float(np.percentile(boot_roc_b, 97.5))],
        "delta_roc_mean": float(np.mean(boot_delta_roc)),
        "delta_roc_ci": [float(np.percentile(boot_delta_roc, 2.5)), float(np.percentile(boot_delta_roc, 97.5))],
        "p_val_roc": p_val_roc,
        "speaker_breakdown": {
            "evaluated_speakers": len(spk_deltas),
            "wins_model_a": spk_wins_a,
            "wins_model_b": spk_wins_b,
            "ties": spk_ties,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="WavFry Paired Model Cluster Bootstrap Significance Audit")
    parser.add_argument("--model-a", type=str, required=True, help="Path to Model A checkpoint (e.g. Weak-On)")
    parser.add_argument("--model-b", type=str, required=True, help="Path to Model B checkpoint (e.g. Weak-Off)")
    parser.add_argument("--label-a", type=str, default="Weak-On (VSDC)", help="Human-readable label for Model A")
    parser.add_argument("--label-b", type=str, default="Weak-Off (Baseline)", help="Human-readable label for Model B")
    parser.add_argument("--noise-condition", type=str, default="clean", help="Acoustic degradation condition")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamplings")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for block evaluation")
    parser.add_argument("--output-dir", type=str, default="data/experiments/statistical_bootstrap_audit")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_a = resolve_strict_checkpoint(args.model_a, "Model A")
    ckpt_b = resolve_strict_checkpoint(args.model_b, "Model B")

    print(f"\n{'='*80}")
    print(f"WavFry Paired Speaker-Level Cluster Bootstrap Audit")
    print(f"Model A: {args.label_a} -> {ckpt_a.name}")
    print(f"Model B: {args.label_b} -> {ckpt_b.name}")
    print(f"Degradation Condition: {args.noise_condition}")
    print(f"Bootstrap Iterations: B={args.n_bootstrap}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    model_a, cfg_a = load_model(ckpt_a, device)
    model_b, cfg_b = load_model(ckpt_b, device)

    tasks = [t for t in ["creaky", "breathy"] if t in model_a.tasks and t in model_b.tasks]
    print(f"[Tasks] Evaluating shared tasks: {tasks}")

    # Prepare validation split in Block mode (exact training protocol)
    val_manifest = PROJECT_ROOT / "manifests" / "segments_multi.csv"
    val_df = pd.read_csv(val_manifest, dtype={"speaker_id": str, "phone": str, "file_key": str})
    _, val_split_df, _ = splits_mod.make_batch_split(val_df, isolate=True)

    from wavfry.data.paradigm_block import ContinuousAudioBlockDataset, collate_block_fn
    block_cfg = cfg_a.get("training", {}).get("block", {})
    ds = ContinuousAudioBlockDataset(
        val_split_df,
        duration_ms=block_cfg.get("duration_ms", 2000),
        hop_ms=block_cfg.get("hop_ms", 1000),
        max_intervals=block_cfg.get("max_intervals", 8),
        seed=42,
    )
    val_loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_block_fn)
    print(f"[Dataset] Validation set: {len(val_split_df)} segments, {len(val_loader)} blocks, {val_split_df['speaker_id'].nunique()} speakers.")

    deg_engine = get_acoustic_degradation_engine() if args.noise_condition != "clean" else None

    # Storage: task -> spk -> interval_id -> {"label": float, "logit_a": [sum, count], "logit_b": [sum, count]}
    raw_spk_data: Dict[str, Dict[str, Dict[str, Any]]] = {t: {} for t in tasks}

    print("\n[Inference] Running synchronized single-pass evaluation across models...")
    start_time = time.time()

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            if (b_idx + 1) % 50 == 0 or (b_idx + 1) == len(val_loader):
                print(f"  Processed {b_idx + 1}/{len(val_loader)} blocks...", flush=True)

            waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch

            if deg_engine is not None:
                waveforms = deg_engine.apply_tensor(waveforms, args.noise_condition, seed=42 + b_idx * 1009)

            waveforms = waveforms.to(device)
            lengths = lengths.to(device)
            bounds = bounds.to(device)
            valid = valid.to(device)

            # Forward Model A
            out_a = model_a(waveforms, lengths=lengths, interval_bounds=bounds, interval_valid=valid)
            # Forward Model B
            out_b = model_b(waveforms, lengths=lengths, interval_bounds=bounds, interval_valid=valid)

            for t_idx, task in enumerate(tasks):
                t_logits_a = out_a["interval_logits"][task]
                t_logits_b = out_b["interval_logits"][task]
                t_labels = labels[..., t_idx] if (labels.ndim > 2 and labels.shape[-1] > t_idx) else labels

                for log_a_row, log_b_row, lab_row, v_row, i_row, spk in zip(
                    t_logits_a, t_logits_b, t_labels, valid, interval_ids, speakers
                ):
                    spk_str = str(spk)
                    if spk_str not in raw_spk_data[task]:
                        raw_spk_data[task][spk_str] = {}

                    for la, lb, lab, v, i_id in zip(log_a_row, log_b_row, lab_row, v_row, i_row):
                        if v and i_id is not None:
                            i_key = str(i_id.item() if isinstance(i_id, torch.Tensor) else i_id)
                            if i_key not in raw_spk_data[task][spk_str]:
                                raw_spk_data[task][spk_str][i_key] = {
                                    "label": lab.item(),
                                    "logit_a": [la.item(), 1],
                                    "logit_b": [lb.item(), 1],
                                }
                            else:
                                raw_spk_data[task][spk_str][i_key]["logit_a"][0] += la.item()
                                raw_spk_data[task][spk_str][i_key]["logit_a"][1] += 1
                                raw_spk_data[task][spk_str][i_key]["logit_b"][0] += lb.item()
                                raw_spk_data[task][spk_str][i_key]["logit_b"][1] += 1

    elapsed = time.time() - start_time
    print(f"[Inference] Completed in {elapsed:.1f}s.")

    # Format per-speaker deduplicated arrays
    task_audit_results = {}
    print(f"\n[Bootstrap] Running Speaker-Level Cluster Bootstrap (B={args.n_bootstrap})...\n")

    for task in tasks:
        spk_preds_a: Dict[str, np.ndarray] = {}
        spk_preds_b: Dict[str, np.ndarray] = {}
        spk_labels: Dict[str, np.ndarray] = {}

        for spk, items in raw_spk_data[task].items():
            l_list = []
            a_list = []
            b_list = []
            for i_key, data in items.items():
                if np.isnan(data["label"]):
                    continue
                l_list.append(int(data["label"]))
                a_list.append(data["logit_a"][0] / data["logit_a"][1])
                b_list.append(data["logit_b"][0] / data["logit_b"][1])

            if len(l_list) > 0 and len(np.unique(l_list)) >= 1:
                spk_labels[spk] = np.array(l_list, dtype=np.int32)
                spk_preds_a[spk] = np.array(a_list, dtype=np.float32)
                spk_preds_b[spk] = np.array(b_list, dtype=np.float32)

        res = run_speaker_cluster_bootstrap_paired(
            preds_a_by_spk=spk_preds_a,
            preds_b_by_spk=spk_preds_b,
            labels_by_spk=spk_labels,
            n_bootstrap=args.n_bootstrap,
            seed=42,
        )
        task_audit_results[task] = res

        print(f"─" * 80)
        print(f"Task: {task.upper()} (Condition: {args.noise_condition})")
        print(f"  {args.label_a:22s}: PR-AUC = {res['pr_a_mean']:.4f}  95% CI [{res['pr_a_ci'][0]:.4f}, {res['pr_a_ci'][1]:.4f}]")
        print(f"  {args.label_b:22s}: PR-AUC = {res['pr_b_mean']:.4f}  95% CI [{res['pr_b_ci'][0]:.4f}, {res['pr_b_ci'][1]:.4f}]")
        p_str = f"< 0.001" if res['p_val_pr'] < 0.001 else f"= {res['p_val_pr']:.4f}"
        print(f"  Paired Delta (A - B)  : Delta PR-AUC = {res['delta_pr_mean']:+.4f}  95% CI [{res['delta_pr_ci'][0]:+.4f}, {res['delta_pr_ci'][1]:+.4f}] (p {p_str})")
        print(f"  ROC-AUC Comparison    : {args.label_a} {res['roc_a_mean']:.4f} vs {args.label_b} {res['roc_b_mean']:.4f} (Delta: {res['delta_roc_mean']:+.4f}, p={res['p_val_roc']:.4f})")
        sb = res["speaker_breakdown"]
        print(f"  Speaker Win/Loss Ratio: {args.label_a} Won {sb['wins_model_a']}/{sb['evaluated_speakers']} speakers, {args.label_b} Won {sb['wins_model_b']}, Tied {sb['ties']}")
        if res['delta_pr_ci'][0] > 0 and res['p_val_pr'] <= 0.05:
            print(f"  >> Statistical Verdict: Statistically Significant Gain (p <= 0.05, 95% CI strictly > 0) <<")
        elif res['delta_pr_ci'][1] < 0 and res['p_val_pr'] <= 0.05:
            print(f"  >> Statistical Verdict: Statistically Significant Degradation (p <= 0.05) <<")
        else:
            print(f"  >> Statistical Verdict: Null Hypothesis Cannot Be Rejected (Not Statistically Significant, 95% CI crosses 0) <<")
        print(f"─" * 80 + "\n")

    # Export markdown report
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_md = out_dir / f"paired_bootstrap_audit_{args.noise_condition}.md"

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# WavFry 发音人聚类非参数配对 Bootstrap 统计检验报告\n\n")
        f.write(f"- **对比模型 A**: `{args.label_a}` ({ckpt_a.name})\n")
        f.write(f"- **对比模型 B**: `{args.label_b}` ({ckpt_b.name})\n")
        f.write(f"- **声学环境**: `{args.noise_condition}`\n")
        f.write(f"- **重采样规模**: $B = {args.n_bootstrap}$ 轮，发音人整群抽样 ($N = {len(val_split_df['speaker_id'].unique())}$)\n\n")
        f.write("## 1. 核心统计学裁定表\n\n")
        f.write("| 评估任务 | 模型 A 指标 (95% CI) | 模型 B 指标 (95% CI) | 配对差值 $\\Delta$ (95% CI) | 经验 $p$ 值 | 发音人胜/负/平 | 统计学结论 |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|:---|\n")

        for task in tasks:
            res = task_audit_results[task]
            p_disp = "< 0.001" if res["p_val_pr"] < 0.001 else f"{res['p_val_pr']:.4f}"
            sb = res["speaker_breakdown"]
            if res["delta_pr_ci"][0] > 0 and res["p_val_pr"] <= 0.05:
                sig_text = "**显著正向增益 (p <= 0.05)**"
            elif res["delta_pr_ci"][1] < 0 and res["p_val_pr"] <= 0.05:
                sig_text = "**显著负向衰退 (p <= 0.05)**"
            else:
                sig_text = "差异不显著 (95% CI 跨越 0)"

            f.write(
                f"| **{task.capitalize()} (PR-AUC)** | {res['pr_a_mean']:.4f} `[{res['pr_a_ci'][0]:.4f}, {res['pr_a_ci'][1]:.4f}]` | "
                f"{res['pr_b_mean']:.4f} `[{res['pr_b_ci'][0]:.4f}, {res['pr_b_ci'][1]:.4f}]` | "
                f"**{res['delta_pr_mean']:+.4f}** `[{res['delta_pr_ci'][0]:+.4f}, {res['delta_pr_ci'][1]:+.4f}]` | "
                f"`{p_disp}` | {sb['wins_model_a']} 胜 / {sb['wins_model_b']} 负 / {sb['ties']} 平 | {sig_text} |\n"
            )

        f.write("\n## 2. 统计学方法说明\n\n")
        f.write("1. **发音人级整群抽样 (Speaker-Level Cluster Bootstrap)**：单次重采样以未见发音人为基本单元，消除滑动窗口（2000ms 窗长，1000ms 步长）切片间的声学生理自相关伪重复。\n")
        f.write("2. **配对差值设计 (Paired Delta)**：每次 Bootstrap 抽样同时在同一发音人群上推演 Model A 与 Model B，相减得到 $\\Delta$，对冲发音人固有识别难度的变异。\n")
        f.write("3. **连续性平滑 $p$ 值 (Davison & Hinkley, 1997)**：规避有限蒙特卡洛抽样下的非法 $p=0.0000$ 漏洞。\n")

    print(f"[Done] Report exported to {report_md}")


if __name__ == "__main__":
    main()
