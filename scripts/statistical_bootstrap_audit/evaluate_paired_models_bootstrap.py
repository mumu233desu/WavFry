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
from wavfry.eval.acoustic_degradation import (
    CANONICAL_EVAL_CONDITIONS,
    get_acoustic_degradation_engine,
    get_condition_metadata,
    parse_condition_str,
)
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


def evaluate_condition_bootstrap(
    cond: str,
    model_a: WavFryClassifier,
    model_b: WavFryClassifier,
    val_loader: DataLoader,
    device: torch.device,
    tasks: List[str],
    deg_engine: Optional[Any],
    n_bootstrap: int,
    label_a: str,
    label_b: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Evaluate paired bootstrap significance for a single acoustic condition."""
    raw_spk_data: Dict[str, Dict[str, Dict[str, Any]]] = {t: {} for t in tasks}

    with torch.no_grad():
        for b_idx, batch in enumerate(val_loader):
            waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch

            if deg_engine is not None and cond != "clean":
                waveforms = deg_engine.apply_tensor(waveforms, cond, seed=42 + b_idx * 1009)

            waveforms = waveforms.to(device)
            lengths = lengths.to(device)
            bounds = bounds.to(device)
            valid = valid.to(device)

            out_a = model_a(waveforms, lengths=lengths, interval_bounds=bounds, interval_valid=valid)
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

    task_results: Dict[str, Any] = {}
    csv_rows: List[Dict[str, Any]] = []
    meta = get_condition_metadata(cond)

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
            n_bootstrap=n_bootstrap,
            seed=42,
        )
        task_results[task] = res

        ci_low = res["delta_pr_ci"][0]
        ci_high = res["delta_pr_ci"][1]
        p_val = res["p_val_pr"]
        sb = res["speaker_breakdown"]
        tot_spk = max(1, sb["evaluated_speakers"])
        win_rate = sb["wins_model_a"] / tot_spk

        if ci_low > 0 and p_val <= 0.05:
            verdict = f"Statistically Significant Gain (p<={p_val:.3f})"
        elif ci_high < 0 and p_val <= 0.05:
            verdict = f"Statistically Significant Degradation (p<={p_val:.3f})"
        else:
            verdict = "Null Hypothesis Retained (Difference Not Significant)"

        csv_rows.append({
            "Model-A": label_a,
            "Model-B": label_b,
            "Task": task,
            "Condition": cond,
            "Scenario": meta["Scenario"],
            "SNR_dB": meta["SNR_dB"],
            "Tier": meta["Tier"],
            "Has_RIR": meta["Has_RIR"],
            "PR-AUC-A": round(float(res["pr_a_mean"]), 4),
            "PR-AUC-B": round(float(res["pr_b_mean"]), 4),
            "Delta-PR-AUC-Mean": round(float(res["delta_pr_mean"]), 4),
            "Delta-PR-AUC-CI-Low": round(float(ci_low), 4),
            "Delta-PR-AUC-CI-High": round(float(ci_high), 4),
            "p-Value": round(float(p_val), 4),
            "Win-Rate": round(float(win_rate), 4),
            "ROC-AUC-A": round(float(res["roc_a_mean"]), 4),
            "ROC-AUC-B": round(float(res["roc_b_mean"]), 4),
            "Delta-ROC-AUC-Mean": round(float(res["delta_roc_mean"]), 4),
            "Statistical-Verdict": verdict,
        })

    return csv_rows, task_results


def main():
    parser = argparse.ArgumentParser(description="WavFry Paired Model Cluster Bootstrap Significance Audit")
    parser.add_argument("--model-a", type=str, required=True, help="Path to Model A checkpoint (e.g. Weak-On)")
    parser.add_argument("--model-b", type=str, required=True, help="Path to Model B checkpoint (e.g. Weak-Off)")
    parser.add_argument("--label-a", type=str, default="Weak-On (VSDC)", help="Human-readable label for Model A")
    parser.add_argument("--label-b", type=str, default="Weak-Off (Baseline)", help="Human-readable label for Model B")
    parser.add_argument("--conditions", nargs="+", default=None, help="List of degradation conditions (default: all 18 canonical conditions).")
    parser.add_argument("--noise-condition", type=str, default=None, help="Legacy alias to evaluate a single condition.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of bootstrap resamplings")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for block evaluation")
    parser.add_argument("--output-dir", type=str, default="data/experiments/statistical_bootstrap_audit")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_a = resolve_strict_checkpoint(args.model_a, "Model A")
    ckpt_b = resolve_strict_checkpoint(args.model_b, "Model B")

    if args.conditions is not None:
        eval_conditions = args.conditions
    elif args.noise_condition is not None:
        eval_conditions = [args.noise_condition]
    else:
        eval_conditions = list(CANONICAL_EVAL_CONDITIONS)

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"WavFry Paired Speaker-Level Cluster Bootstrap Audit Suite")
    print(f"Model A: {args.label_a} -> {ckpt_a.name}")
    print(f"Model B: {args.label_b} -> {ckpt_b.name}")
    print(f"Evaluating {len(eval_conditions)} Conditions: {', '.join(eval_conditions)}")
    print(f"Bootstrap Iterations: B={args.n_bootstrap}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")

    model_a, cfg_a = load_model(ckpt_a, device)
    model_b, cfg_b = load_model(ckpt_b, device)

    tasks = [t for t in ["creaky", "breathy"] if t in model_a.tasks and t in model_b.tasks]
    print(f"[Tasks] Evaluating shared tasks: {tasks}")

    # Prepare validation split in Block mode
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

    deg_engine = get_acoustic_degradation_engine()

    all_csv_rows: List[Dict[str, Any]] = []
    all_json_results: Dict[str, Any] = {
        "model_a": args.label_a,
        "checkpoint_a": ckpt_a.name,
        "model_b": args.label_b,
        "checkpoint_b": ckpt_b.name,
        "n_bootstrap": args.n_bootstrap,
        "evaluated_speakers": int(val_split_df["speaker_id"].nunique()),
        "conditions": {},
    }

    start_suite = time.time()
    for c_idx, cond in enumerate(eval_conditions):
        print(f"[{c_idx + 1}/{len(eval_conditions)}] Evaluating condition: {cond}...", flush=True)
        cond_rows, cond_res = evaluate_condition_bootstrap(
            cond=cond,
            model_a=model_a,
            model_b=model_b,
            val_loader=val_loader,
            device=device,
            tasks=tasks,
            deg_engine=deg_engine,
            n_bootstrap=args.n_bootstrap,
            label_a=args.label_a,
            label_b=args.label_b,
        )
        all_csv_rows.extend(cond_rows)
        all_json_results["conditions"][cond] = {
            task: {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in t_res.items()
            }
            for task, t_res in cond_res.items()
        }

        # Print brief line
        for r in cond_rows:
            p_str = "<0.001" if r["p-Value"] < 0.001 else f"={r['p-Value']:.3f}"
            print(f"    [{r['Task']}] Delta PR-AUC: {r['Delta-PR-AUC-Mean']:+.4f} (95% CI [{r['Delta-PR-AUC-CI-Low']:+.4f}, {r['Delta-PR-AUC-CI-High']:+.4f}], p {p_str}) | {r['Statistical-Verdict']}", flush=True)

    elapsed_suite = time.time() - start_suite
    print(f"\n[Completed] Full suite evaluated in {elapsed_suite:.1f}s.")

    # 1. Primary Tidy CSV with TitleCase LaTeX-aligned headers
    csv_path = out_dir / "paired_bootstrap_suite_report.csv"
    pd.DataFrame(all_csv_rows).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"[Saved] Standardized Tidy CSV saved to {csv_path}")

    # 2. Hierarchical Machine-Readable JSON
    json_path = out_dir / "paired_bootstrap_suite.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_json_results, f, indent=2, ensure_ascii=False)
    print(f"[Saved] Hierarchical JSON saved to {json_path}")

    # 3. Human-Friendly Markdown Table Byproduct
    md_path = out_dir / "paired_bootstrap_suite_report.md"
    md_lines = [
        "# WavFry 发音人聚类非参数配对 Bootstrap 统计检验全景套件报告\n",
        f"- **对比模型 A**: `{args.label_a}` ({ckpt_a.name})",
        f"- **对比模型 B**: `{args.label_b}` ({ckpt_b.name})",
        f"- **重采样规模**: $B = {args.n_bootstrap}$ 轮，发音人整群抽样 ($N = {val_split_df['speaker_id'].nunique()}$)",
        f"- **覆盖声学条件**: {len(eval_conditions)} 个标准条件",
        "",
        "## 1. 核心统计学裁定表 (Canonical Suite Summary)",
        "",
        "| 条件 | 任务 | 模型 A PR (95% CI) | 模型 B PR (95% CI) | 配对差值 $\\Delta$ (95% CI) | $p$ 值 | 胜率 | 统计学结论 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]
    for r in all_csv_rows:
        p_str = "< 0.001" if r["p-Value"] < 0.001 else f"{r['p-Value']:.4f}"
        pr_a_str = f"{r['PR-AUC-A']:.4f}"
        pr_b_str = f"{r['PR-AUC-B']:.4f}"
        delta_str = f"**{r['Delta-PR-AUC-Mean']:+.4f}** `[{r['Delta-PR-AUC-CI-Low']:+.4f}, {r['Delta-PR-AUC-CI-High']:+.4f}]`"
        win_str = f"{r['Win-Rate']*100:.1f}%"
        md_lines.append(
            f"| **{r['Condition']}** | `{r['Task']}` | {pr_a_str} | {pr_b_str} | {delta_str} | `{p_str}` | {win_str} | {r['Statistical-Verdict']} |"
        )

    md_lines.extend([
        "",
        "## 2. 统计学方法说明",
        "",
        "1. **发音人级整群抽样 (Speaker-Level Cluster Bootstrap)**：单次重采样以未见发音人为基本单元，消除滑动窗口（2000ms 窗长，1000ms 步长）切片间的声学生理自相关伪重复。",
        "2. **配对差值设计 (Paired Delta)**：每次 Bootstrap 抽样同时在同一发音人群上推演 Model A 与 Model B，相减得到 $\\Delta$，对冲发音人固有识别难度的变异。",
        "3. **连续性平滑 $p$ 值 (Davison & Hinkley, 1997)**：规避有限蒙特卡洛抽样下的非法 $p=0.0000$ 漏洞。",
        "",
    ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"[Saved] Markdown summary report saved to {md_path}")


if __name__ == "__main__":
    main()
