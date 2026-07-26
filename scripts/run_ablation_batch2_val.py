import subprocess
import json
import sys
from pathlib import Path
import argparse
import yaml

# Add project root to sys.path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

def run_command(cmd_args):
    import os
    print(f"\nRunning command: {' '.join(cmd_args)}")
    env = os.environ.copy()
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    result = subprocess.run(cmd_args, capture_output=True, text=True, encoding='utf-8', errors='ignore', env=env)
    if result.returncode != 0:
        print(f"Error executing command. Exit code: {result.returncode}")
        print("Stdout:")
        if result.stdout:
            try:
                print(result.stdout)
            except UnicodeEncodeError:
                sys.stdout.buffer.write(result.stdout.encode('utf-8', errors='replace'))
                print()
        print("Stderr:")
        if result.stderr:
            try:
                print(result.stderr)
            except UnicodeEncodeError:
                sys.stdout.buffer.write(result.stderr.encode('utf-8', errors='replace'))
                print()
        return False
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Run complete ablation study using Batch 2 (second annotator) as the validation set."
    )
    parser.add_argument("--config", type=str, default="config/_defaults.yaml", help="Path to base configuration file")
    parser.add_argument("--name", type=str, default="batch2_ablation", help="Ablation study name / output folder")
    parser.add_argument("--resume", action="store_true", help="Resume ablation study from existing results")
    
    args = parser.parse_args()
    
    config_path = args.config
    name_dir = args.name
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        
    epochs = config_data.get('training', {}).get('epochs', 25)
    
    # ── Validation split: Batch 2 (second annotator, ~100 samples) ──────────
    val_spk_str = "[batch2]"

    # ── Specific overrides per ablation group ────────────────────────────────
    ablations_specific = {
        "Full (All Enabled)": [
            f"tensorboard.log_dir=R:/runs/{name_dir}/full",
            f"checkpoint.dir=R:/checkpoints/{name_dir}/full"
        ],
        "W/o Sampler (M1 Ablated)": [
            "data.sampler.enabled=false",
            f"tensorboard.log_dir=R:/runs/{name_dir}/no_sampler",
            f"checkpoint.dir=R:/checkpoints/{name_dir}/no_sampler"
        ],
        "W/o Positional Encoding (M3 Ablated)": [
            "training.ablation.use_positional_encoding=false",
            f"tensorboard.log_dir=R:/runs/{name_dir}/no_pos",
            f"checkpoint.dir=R:/checkpoints/{name_dir}/no_pos"
        ],
        "32-bit AdamW (M4 Ablated)": [
            "training.optimizer=adamw_fp32",
            f"tensorboard.log_dir=R:/runs/{name_dir}/adamw_32bit",
            f"checkpoint.dir=R:/checkpoints/{name_dir}/adamw_32bit"
        ],
        "White Noise (M5 Ablated)": [
            "augmentation.noise.type=white",
            f"tensorboard.log_dir=R:/runs/{name_dir}/white_noise",
            f"checkpoint.dir=R:/checkpoints/{name_dir}/white_noise"
        ],
        "Mean Pooling Only (M6 Ablated)": [
            "training.pooling_mode=mean",
            f"tensorboard.log_dir=R:/runs/{name_dir}/mean_pooling",
            f"checkpoint.dir=R:/checkpoints/{name_dir}/mean_pooling"
        ],
        "Top-K Pooling (M7 Ablated)": [
            "training.pooling_mode=mean_topk",
            f"tensorboard.log_dir=R:/runs/{name_dir}/topk_pooling",
            f"checkpoint.dir=R:/checkpoints/{name_dir}/topk_pooling"
        ],
        "All Disabled": [
            "data.sampler.enabled=false",
            "training.ablation.use_cosine_restart=false",
            "training.ablation.use_positional_encoding=false",
            f"tensorboard.log_dir=R:/runs/{name_dir}/baseline",
            f"checkpoint.dir=R:/checkpoints/{name_dir}/baseline"
        ]
    }
    
    common_overrides = [
        f"data.val_speakers={val_spk_str}",
        "eval.early_stopping_patience=4"
    ]
    
    print(f"\n" + "="*80)
    print(f"BATCH-2-VAL ABLATION STUDY: RUNNING 8 GROUPS — Val = Batch 2 ({val_spk_str})")
    print(f"Train = Batch 1 (101 samples) + 8 fully-annotated speakers")
    print("="*80 + "\n")
    
    # Store results for report
    results = {}
    
    for name, specific_overrides in ablations_specific.items():
        overrides = common_overrides + specific_overrides
        print("\n" + "="*60)
        print(f"Starting Ablation: {name}")
        print("="*60)
        
        # Parse results path safely
        log_dir_sub = f"R:/runs/{name_dir}/full"
        for ov in specific_overrides:
            if ov.startswith("tensorboard.log_dir="):
                log_dir_sub = ov.split('=', 1)[1]
                break
        
        log_dir_path = Path(log_dir_sub)
        if log_dir_path.is_absolute() or log_dir_sub.startswith("R:"):
            results_json_path = Path(log_dir_sub) / "fixed_results.json"
        else:
            results_json_path = root / log_dir_sub / "fixed_results.json"
            
        # Parse checkpoint path safely
        ckpt_dir_sub = None
        for ov in specific_overrides:
            if ov.startswith("checkpoint.dir="):
                ckpt_dir_sub = ov.split('=', 1)[1]
                break
        
        if ckpt_dir_sub:
            if ckpt_dir_sub.startswith("R:") or Path(ckpt_dir_sub).is_absolute():
                ckpt_dir = Path(ckpt_dir_sub)
            else:
                ckpt_dir = root / ckpt_dir_sub
        else:
            ckpt_dir = None

        skip_group = False
        if args.resume and results_json_path.exists():
            print(f"Ablation group '{name}' already completed. Skipping training and ensuring checkpoints are cleared.")
            skip_group = True
            if ckpt_dir and ckpt_dir.exists():
                for pt_file in ckpt_dir.glob("*.pt"):
                    try:
                        pt_file.unlink()
                        print(f"  [CLEANUP] Deleted checkpoint: {pt_file}")
                    except Exception as e:
                        print(f"  [CLEANUP] Failed to delete checkpoint {pt_file}: {e}")
        
        if not skip_group:
            # Single-fold run (no --loocv)
            cmd = ["uv", "run", "python", "scripts/train.py", "--config", config_path, "--override"] + overrides
            if args.resume:
                cmd.append("--resume")
                
            success = run_command(cmd)
            if not success:
                print(f"Ablation {name} failed. Skipping report generation for this group.")
                continue
                
            # train.py writes `loocv_results.json` in normal (non-LOOCV) mode
            train_results_path = log_dir_path / "loocv_results.json"
            if train_results_path.exists():
                train_results_path.rename(results_json_path)
                
            if results_json_path.exists():
                print(f"Ablation group '{name}' completed successfully. Cleaning up checkpoints to save space.")
                if ckpt_dir and ckpt_dir.exists():
                    for pt_file in ckpt_dir.glob("*.pt"):
                        try:
                            pt_file.unlink()
                            print(f"  [CLEANUP] Deleted checkpoint: {pt_file}")
                        except Exception as e:
                            print(f"  [CLEANUP] Failed to delete checkpoint {pt_file}: {e}")
                            
        # Read results
        if results_json_path.exists():
            try:
                with open(results_json_path, 'r', encoding='utf-8') as f:
                    results[name] = json.load(f)
            except Exception as e:
                print(f"Failed to read results JSON for {name}: {e}")
        else:
            print(f"Results JSON not found at {results_json_path} for {name}")
            
    # ── Generate report ───────────────────────────────────────────────────────
    if not results:
        print("No ablation results collected.")
        return

    full_auc_pr = None
    full_default_f1 = None
    if "Full (All Enabled)" in results:
        full_data = results["Full (All Enabled)"]
        full_auc_pr = full_data.get('auc_pr', full_data.get('mean_auc_pr'))
        full_default_f1 = full_data.get('f1', full_data.get('mean_f1'))

    print("\n" + "="*80)
    print(f"BATCH-2-VAL ABLATION COMPARISON TABLE  (Val = Batch 2, {epochs} EPOCHS)")
    print("="*80)
    
    header = "| Configuration | PR-AUC | Default-t F1 | Accuracy | Precision | Recall | Peak VRAM |"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(divider)
    
    md_rows = [header, divider]
    
    for name in ablations_specific.keys():
        if name not in results:
            continue
        data = results[name]
        
        auc_pr = data.get('auc_pr', data.get('mean_auc_pr', 0.0))
        f1     = data.get('f1',     data.get('mean_f1',     0.0))
        acc    = data.get('accuracy',  data.get('mean_accuracy',  0.0))
        prec   = data.get('precision', data.get('mean_precision', 0.0))
        rec    = data.get('recall',    data.get('mean_recall',    0.0))
        vram   = data.get('peak_vram_mb', data.get('mean_peak_vram_mb', 0.0))
        
        auc_pr_diff_str    = ""
        default_f1_diff_str = ""
        
        if name == "Full (All Enabled)":
            auc_pr_diff_str    = " (ref)"
            default_f1_diff_str = " (ref)"
        else:
            if full_auc_pr is not None:
                diff = auc_pr - full_auc_pr
                auc_pr_diff_str = f" ({diff:+.4f})"
            if full_default_f1 is not None:
                diff = f1 - full_default_f1
                default_f1_diff_str = f" ({diff:+.4f})"
                
        row = (
            f"| {name:<35} | {auc_pr:.4f}{auc_pr_diff_str} | {f1:.4f}{default_f1_diff_str} | "
            f"{acc:.4f} | {prec:.4f} | {rec:.4f} | "
            f"{vram:.1f} MB |"
        )
        print(row)
        md_rows.append(row)
        
    print("="*80)
    
    # Save report
    report_path_e = root / "runs" / name_dir / "ablation_batch2_report.md"
    report_path_r = Path(f"R:/runs/{name_dir}/ablation_batch2_report.md")
    
    for report_path in [report_path_e, report_path_r]:
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Ablation Study Report (Batch 2 Validation)\n\n")
                f.write("Validation set: **Batch 2** — independently annotated by a second annotator (~100 samples). ")
                f.write("Train set: **Batch 1** (101 samples) + **8 fully-annotated speakers**.\n\n")
                f.write(f"Training performed for **{epochs} epochs** per configuration.\n\n")
                f.write("## Performance Comparison\n\n")
                for row in md_rows:
                    f.write(row + "\n")
                f.write("\n## Metric Interpretations\n\n")
                f.write("- **PR-AUC**: Primary training metric (Area Under Precision-Recall Curve).\n")
                f.write("- **Default-t F1**: F1 score evaluated at the default threshold of `0.5`.\n")
                f.write("- **Peak VRAM**: Average peak GPU memory consumed during training.\n\n")
            print(f"Ablation report successfully written to: {report_path}")
        except Exception as e:
            print(f"Failed to write report to {report_path}: {e}")

if __name__ == "__main__":
    main()
