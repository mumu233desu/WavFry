import subprocess
import json
import sys
from pathlib import Path

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
        print(result.stdout)
        print("Stderr:")
        print(result.stderr)
        return False
    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run smoketest ablation study pipeline over LOOCV folds.")
    parser.add_argument("--resume", action="store_true", help="Resume ablation study from existing results and checkpoints")
    args = parser.parse_args()

    import yaml
    config_path = "config/smoketest.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    epochs = config_data.get('training', {}).get('epochs', 15)
    speakers = ["U1001", "U1089"]
    
    # Helper to check if a group has completed training for all specified speakers
    def is_group_completed(results_json_path, target_speakers):
        if not results_json_path.exists():
            return False
        try:
            with open(results_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            completed_spks = {fold['speaker'] for fold in data.get('folds', [])}
            return all(spk in completed_spks for spk in target_speakers)
        except Exception as e:
            print(f"Error checking completion status for {results_json_path}: {e}")
            return False

    ablations_specific = {
        "Full (All Enabled)": [
            "tensorboard.log_dir=R:/runs/smoketest_ablation/full",
            "checkpoint.dir=R:/checkpoints/smoketest_ablation/full"
        ],
        "W/o Sampler (M1 Ablated)": [
            "data.sampler.enabled=false",
            "tensorboard.log_dir=R:/runs/smoketest_ablation/no_sampler",
            "checkpoint.dir=R:/checkpoints/smoketest_ablation/no_sampler"
        ],
        "W/o Positional Encoding (M3 Ablated)": [
            "training.ablation.use_positional_encoding=false",
            "tensorboard.log_dir=R:/runs/smoketest_ablation/no_pos",
            "checkpoint.dir=R:/checkpoints/smoketest_ablation/no_pos"
        ],
        "32-bit AdamW (M4 Ablated)": [
            "training.optimizer=adamw_fp32",
            "tensorboard.log_dir=R:/runs/smoketest_ablation/adamw_32bit",
            "checkpoint.dir=R:/checkpoints/smoketest_ablation/adamw_32bit"
        ],
        "White Noise (M5 Ablated)": [
            "augmentation.noise.type=white",
            "tensorboard.log_dir=R:/runs/smoketest_ablation/white_noise",
            "checkpoint.dir=R:/checkpoints/smoketest_ablation/white_noise"
        ],
        "Mean Pooling Only (M6 Ablated)": [
            "training.pooling_mode=mean",
            "tensorboard.log_dir=R:/runs/smoketest_ablation/mean_pooling",
            "checkpoint.dir=R:/checkpoints/smoketest_ablation/mean_pooling"
        ],
        "Top-K Pooling (M7 Ablated)": [
            "training.pooling_mode=mean_topk",
            "tensorboard.log_dir=R:/runs/smoketest_ablation/topk_pooling",
            "checkpoint.dir=R:/checkpoints/smoketest_ablation/topk_pooling"
        ],
        "All Disabled": [
            "data.sampler.enabled=false",
            "training.ablation.use_cosine_restart=false",
            "training.ablation.use_positional_encoding=false",
            "tensorboard.log_dir=R:/runs/smoketest_ablation/baseline",
            "checkpoint.dir=R:/checkpoints/smoketest_ablation/baseline"
        ]
    }
    
    # We run the folds incrementally (first all groups on Fold 1, then all groups on Fold 1 + Fold 2, etc.)
    for fold_step in range(1, len(speakers) + 1):
        active_speakers = speakers[:fold_step]
        spk_str = "[" + ",".join(active_speakers) + "]"
        common_overrides = [
            "eval.early_stopping_patience=0",
            "data.val_speakers=loocv",
            f"data.annotated_speakers={spk_str}"
        ]
        
        print(f"\n" + "="*80)
        print(f"ABLATION PHASE {fold_step}/{len(speakers)}: RUNNING GROUPS ON SPEAKERS: {active_speakers}")
        print("="*80 + "\n")
        
        for name, specific_overrides in ablations_specific.items():
            overrides = common_overrides + specific_overrides
            print("="*60)
            print(f"Starting Ablation (Phase {fold_step}): {name}")
            print("="*60)
            
            # Parse results path safely
            log_dir_sub = "runs/smoketest_ablation/full"
            for ov in specific_overrides:
                if ov.startswith("tensorboard.log_dir="):
                    log_dir_sub = ov.split('=', 1)[1]
                    break
            
            log_dir_path = Path(log_dir_sub)
            if log_dir_path.is_absolute() or log_dir_sub.startswith("R:"):
                results_json_path = Path(log_dir_sub) / "loocv_results.json"
            else:
                results_json_path = root / log_dir_sub / "loocv_results.json"
            
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

            # Check if this group has completed the current phase (all active speakers trained)
            skip_group = False
            if args.resume and is_group_completed(results_json_path, active_speakers):
                print(f"Ablation group '{name}' already completed for active speakers {active_speakers}. Skipping training and ensuring checkpoints are cleared.")
                skip_group = True
                
                # Clean up checkpoints for this fold step to save space
                if ckpt_dir and ckpt_dir.exists():
                    for pt_file in ckpt_dir.glob("*.pt"):
                        try:
                            pt_file.unlink()
                            print(f"  [CLEANUP] Deleted checkpoint: {pt_file}")
                        except Exception as e:
                            print(f"  [CLEANUP] Failed to delete checkpoint {pt_file}: {e}")
            
            if not skip_group:
                # Remove pre-existing results to force re-run if not resuming and on step 1
                if not args.resume and fold_step == 1 and results_json_path.exists():
                    try: results_json_path.unlink()
                    except: pass
                
                # Build command with --loocv flag
                cmd = ["uv", "run", "python", "scripts/train.py", "--config", config_path, "--loocv", "--override"] + overrides
                # Pass --resume to train.py if user requested resume OR if we are beyond the first fold step (to resume from previous fold)
                if args.resume or fold_step > 1:
                    cmd.append("--resume")
                
                success = run_command(cmd)
                if not success:
                    print(f"Ablation {name} failed at fold step {fold_step}. Skipping current run.")
                    continue
                
                # Clean up checkpoints on successful completion of the current fold step to save space
                if results_json_path.exists():
                    print(f"Ablation group '{name}' completed fold step {fold_step} successfully. Cleaning up checkpoints to save space.")
                    if ckpt_dir and ckpt_dir.exists():
                        for pt_file in ckpt_dir.glob("*.pt"):
                            try:
                                pt_file.unlink()
                                print(f"  [CLEANUP] Deleted checkpoint: {pt_file}")
                            except Exception as e:
                                print(f"  [CLEANUP] Failed to delete checkpoint {pt_file}: {e}")
            
    # Collect final results for all completed groups
    results = {}
    for name, specific_overrides in ablations_specific.items():
        log_dir_sub = "runs/smoketest_ablation/full"
        for ov in specific_overrides:
            if ov.startswith("tensorboard.log_dir="):
                log_dir_sub = ov.split('=', 1)[1]
                break
        
        log_dir_path = Path(log_dir_sub)
        if log_dir_path.is_absolute() or log_dir_sub.startswith("R:"):
            results_json_path = Path(log_dir_sub) / "loocv_results.json"
        else:
            results_json_path = root / log_dir_sub / "loocv_results.json"
            
        if results_json_path.exists():
            try:
                with open(results_json_path, 'r', encoding='utf-8') as f:
                    results[name] = json.load(f)
                print(f"Successfully collected results for comparison: {name}")
            except Exception as e:
                print(f"Failed to read results JSON for {name}: {e}")
        else:
            print(f"Results JSON not found at {results_json_path} for {name}")
            
    # Generate report
    if not results:
        print("No ablation results collected.")
        return
    val_spk_detected = "Unknown"
    for name in ablations_specific.keys():
        if name in results:
            spks = [f.get('speaker', 'Unknown') for f in results[name]['folds']]
            val_spk_detected = ", ".join(spks)
            break

    # Extract Full configuration metrics as base for comparison
    full_auc_pr = None
    full_best_f1 = None
    full_default_f1 = None
    if "Full (All Enabled)" in results:
        full_data = results["Full (All Enabled)"]
        full_auc_pr = full_data['mean_auc_pr']
        full_default_f1 = full_data['mean_f1']
        full_folds = full_data['folds']
        full_best_f1 = sum(f.get('threshold_f1', 0.0) for f in full_folds) / len(full_folds)

    print("\n" + "="*80)
    print(f"ABLATION STUDY COMPARISON TABLE (SMOKETEST, Validation Split: {val_spk_detected}, 15 EPOCHS)")
    print("="*80)
    
    header = "| Configuration | PR-AUC | Default-t F1 | Best-t (Grid) | Best-t F1 | Accuracy | Precision | Recall |"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(divider)
    
    # Store for Markdown report
    md_rows = []
    md_rows.append(header)
    md_rows.append(divider)
    
    for name in ablations_specific.keys():
        if name not in results:
            continue
        data = results[name]
        folds = data['folds']
        mean_best_t = sum(f.get('best_threshold', 0.5) for f in folds) / len(folds)
        mean_best_f1 = sum(f.get('threshold_f1', 0.0) for f in folds) / len(folds)
        
        auc_pr_diff_str = ""
        best_f1_diff_str = ""
        default_f1_diff_str = ""
        
        if name == "Full (All Enabled)":
            auc_pr_diff_str = " (ref)"
            best_f1_diff_str = " (ref)"
            default_f1_diff_str = " (ref)"
        else:
            if full_auc_pr is not None:
                diff = data['mean_auc_pr'] - full_auc_pr
                auc_pr_diff_str = f" ({diff:+.4f})"
            if full_best_f1 is not None:
                diff = mean_best_f1 - full_best_f1
                best_f1_diff_str = f" ({diff:+.4f})"
            if full_default_f1 is not None:
                diff = data['mean_f1'] - full_default_f1
                default_f1_diff_str = f" ({diff:+.4f})"
                
        row = (
            f"| {name:<35} | {data['mean_auc_pr']:.4f}{auc_pr_diff_str} | {data['mean_f1']:.4f}{default_f1_diff_str} | "
            f"{mean_best_t:.3f} | {mean_best_f1:.4f}{best_f1_diff_str} | "
            f"{data['mean_accuracy']:.4f} | {data['mean_precision']:.4f} | {data['mean_recall']:.4f} |"
        )
        print(row)
        md_rows.append(row)
        
    print("="*80)
    
    # Save Report to file
    report_path = root / "runs" / "smoketest_ablation" / "ablation_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Ablation Study Report (Smoketest)\n\n")
        f.write("This report presents the ablation study results for the newly implemented modules. ")
        f.write(f"The training was performed on the restricted smoketest dataset (100 train, 32 val segments, validating on validation split `{val_spk_detected}`).\n\n")
        f.write("## Performance Comparison\n\n")
        for row in md_rows:
            f.write(row + "\n")
        f.write("\n## Metric Interpretations\n\n")
        f.write("- **PR-AUC**: Primary training metric (Area Under Precision-Recall Curve). Parentheses display the difference compared to `Full (All Enabled)`.\n")
        f.write("- **Default-t F1**: F1 score evaluated at the default threshold of `0.5`.\n")
        f.write("- **Best-t**: Best threshold discovered during the brute-force validation grid search.\n")
        f.write("- **Best-t F1**: Maximum F1 score achieved at the best threshold.\n\n")
        f.write("## Observations & Key Takeaways\n\n")
        f.write("1. **Sampler Influence**: Check if training without `GroupedBatchSampler` displays larger class imbalance bias.\n")
        f.write("2. **Cosine Warm Restart Impact**: Compare learning curve or performance stability.\n")
        f.write("3. **Positional Encoding Role**: Check if relative location embeddings improve detection accuracy on this fold.\n")
        
    print(f"\nAblation report successfully written to: {report_path}")

if __name__ == "__main__":
    main()
