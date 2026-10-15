"""
Master Paper Evaluation Pipeline Orchestrator.

Canonical Entry Point:
- Dispatches and orchestrates all paper evaluation modules:
  * Tiered Prediction Dump (WavFry across 5 degradation tiers)
  * Table 1 & Figure 2: Baseline Degradation Comparison (evaluate_qut_baselines.py)
  * Table 2: Paired Speaker Cluster Bootstrap Audit (evaluate_paired_models_bootstrap.py)
  * Table 4: Speaker Gender Fairness Audit (evaluate_gender_fairness.py)
  * Table 8b & Figure 8: Acoustic-Physical Correlation Validation (evaluate_acoustic_correlations.py)
  * Table 5 & Figure 4: LOLO Micro Layer Ablation (evaluate_lolo_ablation.py)
  * Table 6 & Figure 5: Top-k Depth Truncation Pareto (evaluate_topk_layer_truncation.py)
  * Table 8 & Figure 7: CKA Representation Geometry & Fidelity (evaluate_representation_cka.py)
  * Table 9: Edge ONNX Export & RTF Benchmark (export_deployment.py, test_onnx_export.py)
- Modular & Pluggable: Select modules via `--modules all` or `--modules baselines,gender,...`
- Automated Visualization Dispatch: Triggers offline plotting in data/visualizations/.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_paper_evaluations")

AVAILABLE_MODULES = [
    "dump",
    "baselines",
    "bootstrap",
    "gender",
    "acoustic_corr",
    "lolo",
    "topk",
    "cka",
    "export",
]

MODULE_DESCRIPTIONS = {
    "dump": "Model Canonical Prediction Dump (5 Tiers)",
    "baselines": "Table 1 & Fig 2: Baseline Comparison under Degradation",
    "bootstrap": "Table 2: Paired Speaker Cluster Bootstrap Audit",
    "gender": "Table 4: Speaker Gender Fairness Audit",
    "acoustic_corr": "Table 8b & Fig 8: Acoustic-Physical Correlation Audit",
    "lolo": "Table 5 & Fig 4: LOLO Micro Layer Ablation Sensitivity",
    "topk": "Table 6 & Fig 5: Top-k Depth Truncation Pareto",
    "cka": "Table 8 & Fig 7: CKA Representation Geometry & Fidelity",
    "export": "Table 9: Edge ONNX Export & Latency Benchmark",
}


def run_command_logged(cmd: List[str], cwd: Path, desc: str, extra_env: Optional[Dict[str, str]] = None) -> bool:
    """Run a subprocess command with flushed I/O and real-time status logging."""
    logger.info(f"\n{'='*75}\n>>> [RUNNING] {desc}\nCommand: {' '.join(cmd)}\n{'='*75}")
    t0 = time.time()
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            check=False,
            text=True,
            capture_output=False,
            env=env,
        )
        elapsed = time.time() - t0
        if proc.returncode == 0:
            logger.info(f">>> [SUCCESS] {desc} finished in {elapsed:.1f}s.")
            return True
        else:
            logger.error(f">>> [FAILED] {desc} exited with code {proc.returncode} ({elapsed:.1f}s).")
            return False
    except Exception as e:
        logger.error(f">>> [ERROR] Failed to execute {desc}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Master Paper Evaluation Pipeline Orchestrator.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to WavFry model checkpoint.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="manifests/segments_multi.csv",
        help="Path to evaluation test manifest.",
    )
    parser.add_argument(
        "--preds-csv",
        type=str,
        default=None,
        help="Path to pre-computed tiered predictions CSV (skips dump step).",
    )
    parser.add_argument(
        "--modules",
        type=str,
        default="all",
        help=f"Comma-separated list of modules to execute, or 'all'. Options: {','.join(AVAILABLE_MODULES)}",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip triggering plotting scripts in data/visualizations.",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Limit number of test samples (for rapid verification/smoke testing).",
    )
    parser.add_argument(
        "--output-base",
        type=str,
        default="data/experiments",
        help="Base directory for experiment outputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu",
        help="Compute device (cuda or cpu).",
    )

    args = parser.parse_args()
    proj_root = Path(__file__).resolve().parents[1]
    vis_dir = Path(r"E:\Coding\WavFry\data\visualizations")
    if not vis_dir.exists():
        vis_dir = proj_root / "data" / "visualizations"

    # 1. Parse modules
    if args.modules.strip().lower() == "all":
        selected_modules = AVAILABLE_MODULES.copy()
    else:
        req_mods = [m.strip().lower() for m in args.modules.split(",") if m.strip()]
        selected_modules = []
        for m in req_mods:
            if m not in AVAILABLE_MODULES:
                logger.error(f"Unknown module '{m}'. Choose from: {AVAILABLE_MODULES}")
                sys.exit(1)
            selected_modules.append(m)

    logger.info("=" * 75)
    logger.info("       WavFry Master Paper Evaluation Pipeline")
    logger.info(f"  Project Root: {proj_root}")
    logger.info(f"  Checkpoint:   {args.checkpoint}")
    logger.info(f"  Manifest:     {args.manifest}")
    logger.info(f"  Active Modules: {selected_modules}")
    logger.info("=" * 75)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    manifest_path = Path(args.manifest) if Path(args.manifest).is_absolute() else (proj_root / args.manifest)
    out_base = Path(args.output_base) if Path(args.output_base).is_absolute() else (proj_root / args.output_base)
    cache_dir = out_base / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    preds_csv_path = Path(args.preds_csv) if args.preds_csv else None

    statuses: Dict[str, str] = {}
    total_start = time.time()

    # Step 0: Ensure prediction dump if required
    needs_dump = any(m in selected_modules for m in ["dump", "baselines", "gender", "acoustic_corr"])
    if needs_dump and (preds_csv_path is None or not preds_csv_path.exists()):
        if ckpt_path is None or not ckpt_path.exists():
            logger.error("A valid --checkpoint is required to produce prediction dumps.")
            sys.exit(1)

        dump_target = cache_dir / f"preds_{ckpt_path.stem}_tiered.csv"
        dump_cmd = [
            sys.executable,
            "-c",
            f"""
from pathlib import Path
import torch
from wavfry.eval.cache_manager import dump_model_predictions
from wavfry.eval.acoustic_degradation import AcousticDegradationEngine
from wavfry.eval.unified import build_test_manifest, load_model_from_checkpoint
from wavfry.model.wavlm_classifier import WavFryClassifier
import pandas as pd

full_df = pd.read_csv(r'{manifest_path}')
test_df = build_test_manifest(full_df, task='creaky')
limit_val = {args.limit_samples} if {args.limit_samples is not None} else None
if limit_val is not None:
    test_df = test_df.iloc[:limit_val]

model, _ = load_model_from_checkpoint(Path(r'{ckpt_path}'), torch.device('{args.device}'), WavFryClassifier)
deg_engine = AcousticDegradationEngine()
dump_model_predictions(
    model=model,
    manifest_df=test_df,
    degradation_engine=deg_engine,
    out_csv_path=Path(r'{dump_target}'),
    device=torch.device('{args.device}'),
)
print('Dump complete.')
""",
        ]
        ok = run_command_logged(dump_cmd, proj_root, "Model Prediction Dump (5 Tiers)")
        if ok and dump_target.exists():
            preds_csv_path = dump_target
            statuses["dump"] = "SUCCESS"
        else:
            statuses["dump"] = "FAILED"
            logger.error("Failed to generate model prediction dump. Aborting downstream dependencies.")

    # Execute selected modules
    for mod in selected_modules:
        if mod == "dump":
            continue

        if mod == "baselines":
            cmd = [
                sys.executable,
                "scripts/baseline_comparison/evaluate_qut_baselines.py",
                "--manifest", str(manifest_path),
                "--output-dir", str(out_base / "baselines"),
            ]
            if preds_csv_path and preds_csv_path.exists():
                cmd.extend(["--preds-csv", str(preds_csv_path)])
            elif ckpt_path:
                cmd.extend(["--checkpoint", str(ckpt_path)])
            if args.limit_samples:
                cmd.extend(["--limit-samples", str(args.limit_samples)])

            ok = run_command_logged(cmd, proj_root, MODULE_DESCRIPTIONS["baselines"])
            statuses["baselines"] = "SUCCESS" if ok else "FAILED"

        elif mod == "bootstrap":
            cmd = [
                sys.executable,
                "scripts/statistical_bootstrap_audit/evaluate_paired_models_bootstrap.py",
                "--checkpoint-a", str(ckpt_path) if ckpt_path else "",
                "--checkpoint-b", str(ckpt_path) if ckpt_path else "",
                "--manifest", str(manifest_path),
                "--output-dir", str(out_base / "bootstrap"),
            ]
            if ckpt_path and ckpt_path.exists():
                ok = run_command_logged(cmd, proj_root, MODULE_DESCRIPTIONS["bootstrap"])
                statuses["bootstrap"] = "SUCCESS" if ok else "FAILED"
            else:
                statuses["bootstrap"] = "SKIPPED (Checkpoint Required)"

        elif mod == "gender":
            cmd = [
                sys.executable,
                "scripts/statistical_bootstrap_audit/evaluate_gender_fairness.py",
                "--manifest", str(manifest_path),
                "--output-dir", str(out_base / "fairness"),
            ]
            if preds_csv_path and preds_csv_path.exists():
                cmd.extend(["--preds-csv", str(preds_csv_path)])
            elif ckpt_path:
                cmd.extend(["--checkpoint", str(ckpt_path)])

            ok = run_command_logged(cmd, proj_root, MODULE_DESCRIPTIONS["gender"])
            statuses["gender"] = "SUCCESS" if ok else "FAILED"

        elif mod == "acoustic_corr":
            cmd = [
                sys.executable,
                "scripts/acoustic_interpretability_audit/evaluate_acoustic_correlations.py",
                "--manifest", str(manifest_path),
                "--output-dir", str(out_base / "interpretability"),
            ]
            if preds_csv_path and preds_csv_path.exists():
                cmd.extend(["--preds-csv", str(preds_csv_path)])
            elif ckpt_path:
                cmd.extend(["--checkpoint", str(ckpt_path)])
            if args.limit_samples:
                cmd.extend(["--limit-samples", str(args.limit_samples)])

            ok = run_command_logged(cmd, proj_root, MODULE_DESCRIPTIONS["acoustic_corr"])
            statuses["acoustic_corr"] = "SUCCESS" if ok else "FAILED"

        elif mod == "lolo":
            if ckpt_path and ckpt_path.exists():
                cmd = [
                    sys.executable,
                    "scripts/layer_importance_and_truncation_audit/evaluate_lolo_ablation.py",
                    "--checkpoint", str(ckpt_path),
                    "--manifest", str(manifest_path),
                    "--output", str(out_base / "lolo_ablation_results.csv"),
                ]
                if args.limit_samples:
                    cmd.extend(["--max-batches", "2"])
                ok = run_command_logged(cmd, proj_root, MODULE_DESCRIPTIONS["lolo"])
                statuses["lolo"] = "SUCCESS" if ok else "FAILED"
            else:
                statuses["lolo"] = "SKIPPED (Checkpoint Required)"

        elif mod == "topk":
            if ckpt_path and ckpt_path.exists():
                cmd = [
                    sys.executable,
                    "scripts/layer_importance_and_truncation_audit/evaluate_topk_layer_truncation.py",
                    "--checkpoint", str(ckpt_path),
                    "--manifest", str(manifest_path),
                    "--output-dir", str(out_base / "topk_layer_truncation_evaluation"),
                ]
                if args.limit_samples:
                    cmd.extend(["--max-batches", "2"])
                ok = run_command_logged(cmd, proj_root, MODULE_DESCRIPTIONS["topk"])
                statuses["topk"] = "SUCCESS" if ok else "FAILED"
            else:
                statuses["topk"] = "SKIPPED (Checkpoint Required)"

        elif mod == "cka":
            if ckpt_path and ckpt_path.exists():
                cmd = [
                    sys.executable,
                    "scripts/representation_and_calibration_audit/evaluate_representation_cka.py",
                    "--checkpoint", str(ckpt_path),
                    "--manifest", str(manifest_path),
                    "--output-dir", str(out_base / "representation_cka_audit"),
                ]
                if args.limit_samples:
                    cmd.extend(["--max-samples", str(args.limit_samples)])
                ok = run_command_logged(cmd, proj_root, MODULE_DESCRIPTIONS["cka"])
                statuses["cka"] = "SUCCESS" if ok else "FAILED"
            else:
                statuses["cka"] = "SKIPPED (Checkpoint Required)"

        elif mod == "export":
            if ckpt_path and ckpt_path.exists():
                deploy_dir = proj_root / "deployments"
                deploy_dir.mkdir(parents=True, exist_ok=True)
                cmd1 = [
                    sys.executable,
                    "scripts/export_deployment.py",
                    "--checkpoint", str(ckpt_path),
                    "--output-dir", str(deploy_dir),
                    "--task", "all",
                ]
                ok1 = run_command_logged(cmd1, proj_root, "ONNX Export")
                cmd2 = [sys.executable, "scripts/test_onnx_export.py"]
                ok2 = run_command_logged(cmd2, proj_root, "ONNX Runtime Verification")
                statuses["export"] = "SUCCESS" if (ok1 and ok2) else "FAILED"
            else:
                statuses["export"] = "SKIPPED (Checkpoint Required)"

    # Step 4: Automated Offline Visualization
    if not args.skip_plots and vis_dir.exists():
        logger.info("\n" + "=" * 75)
        logger.info(">>> [VISUALIZATION] Rendering Publication-Ready Figures in data/visualizations")
        logger.info("=" * 75)

        plot_scripts = [
            ("plot_fig2_noise_robustness.py", "Figure 2: Noise Robustness", out_base / "baselines" / "vowel_robustness_metrics.json"),
            ("plot_fig4_lolo_ablation.py", "Figure 4: LOLO Ablation Sensitivity", out_base / "lolo_ablation_results.csv"),
            ("plot_fig5_topk_truncation.py", "Figure 5: Top-k Truncation Pareto", out_base / "topk_layer_truncation_evaluation" / "topk_layer_truncation_report.csv"),
            ("plot_fig7_cka_fidelity.py", "Figure 7: CKA Representation Fidelity", out_base / "representation_cka_audit" / "cka_matrices.json"),
            ("plot_fig8_acoustic_correlations.py", "Figure 8: Acoustic Correlations", out_base / "interpretability" / "acoustic_correlations.csv"),
            ("plot_gender_fairness_distribution.py", "Table 4: Demographic Parity Distribution", out_base / "fairness" / "gender_fairness_distribution.csv"),
        ]

        for p_script, p_desc, req_file in plot_scripts:
            script_fpath = vis_dir / p_script
            # Check if input file exists before triggering plot
            alt_req = Path(r"E:\Coding\WavFry") / req_file.relative_to(proj_root) if req_file.is_relative_to(proj_root) else req_file
            if script_fpath.exists() and (req_file.exists() or alt_req.exists()):
                run_command_logged([sys.executable, str(script_fpath)], vis_dir, p_desc, extra_env={"PYTHONPATH": f"{proj_root};{vis_dir}"})
            else:
                logger.info(f"Skipping {p_desc} (data file not generated in this run).")

    # Executive Summary Table
    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("                     EVALUATION PIPELINE EXECUTIVE SUMMARY")
    print("=" * 80)
    print(f"{'Module ID':<15} {'Status':<15} {'Description':<50}")
    print("-" * 80)
    for mod_id, status in statuses.items():
        desc = MODULE_DESCRIPTIONS.get(mod_id, mod_id)
        status_tag = f"[{status}]"
        print(f"{mod_id:<15} {status_tag:<15} {desc:<50}")
    print("-" * 80)
    print(f"Total Execution Time: {total_elapsed:.1f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
