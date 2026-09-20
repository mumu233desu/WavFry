import json
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1]
    runs_dir = Path("R:/runs/full_ablation")
    
    ablations_specific = {
        "Full (All Enabled)": "full",
        "W/o Sampler (M1 Ablated)": "no_sampler",
        "W/o Positional Encoding (M3 Ablated)": "no_pos",
        "32-bit AdamW (M4 Ablated)": "adamw_32bit",
        "White Noise (M5 Ablated)": "white_noise",
        "Mean Pooling Only (M6 Ablated)": "mean_pooling",
        "Top-K Pooling (M7 Ablated)": "topk_pooling",
        "All Disabled": "baseline"
    }

    results = {}
    for name, log_dir_sub in ablations_specific.items():
        results_json_path = runs_dir / log_dir_sub / "loocv_results.json"
        if results_json_path.exists():
            try:
                with open(results_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Filter down to exactly the first 2 folds if it somehow has more
                    data['folds'] = data['folds'][:2]
                    
                    if len(data['folds']) > 0:
                        import numpy as np
                        data['mean_auc_pr'] = float(np.mean([f.get('auc_pr', 0) for f in data['folds']]))
                        data['mean_f1'] = float(np.mean([f['f1'] for f in data['folds']]))
                        data['mean_accuracy'] = float(np.mean([f.get('accuracy', 0) for f in data['folds']]))
                        data['mean_precision'] = float(np.mean([f.get('precision', 0) for f in data['folds']]))
                        data['mean_recall'] = float(np.mean([f.get('recall', 0) for f in data['folds']]))
                        results[name] = data
            except Exception as e:
                print(f"Failed to read results JSON for {name}: {e}")

    if not results:
        print("No ablation results collected.")
        return

    full_auc_pr = None
    full_best_f1 = None
    full_default_f1 = None
    if "Full (All Enabled)" in results and len(results["Full (All Enabled)"]['folds']) == 2:
        full_data = results["Full (All Enabled)"]
        full_auc_pr = full_data['mean_auc_pr']
        full_default_f1 = full_data['mean_f1']
        full_folds = full_data['folds']
        full_best_f1 = sum(f.get('threshold_f1', 0.0) for f in full_folds) / len(full_folds)
        
    print("\n" + "="*80)
    print(f"ABLATION STUDY COMPARISON TABLE (2 Folds: U1001, U0046)")
    print("="*80)
    
    header = "| Configuration | PR-AUC | Accuracy | Precision | Recall | Default-t F1 | Best-t (Grid) | Best-t F1 |"
    divider = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(header)
    print(divider)
    
    md_rows = [header, divider]
    
    for name in ablations_specific.keys():
        if name not in results:
            continue
        data = results[name]
        folds = data['folds']
        if len(folds) < 2:
            row = f"| {name:<35} | INCOMPLETE ({len(folds)}/2 folds) | - | - | - | - | - | - |"
            print(row)
            md_rows.append(row)
            continue
            
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
            f"| {name:<35} | {data['mean_auc_pr']:.4f}{auc_pr_diff_str} | "
            f"{data['mean_accuracy']:.4f} | {data['mean_precision']:.4f} | {data['mean_recall']:.4f} | "
            f"{data['mean_f1']:.4f}{default_f1_diff_str} | "
            f"{mean_best_t:.3f} | {mean_best_f1:.4f}{best_f1_diff_str} |"
        )
        print(row)
        md_rows.append(row)
        
    print("="*80)
    
    report_path_e = root / "exp_results" / "runs" / "full_ablation" / "ablation_2folds_report.md"
    report_path_e.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path_e, 'w', encoding='utf-8') as f:
        f.write("# Ablation Study Report (2 Folds)\n\n")
        f.write("Validation splits: **U1001, U0046**\n\n")
        for row in md_rows:
            f.write(row + "\n")
    print(f"Report written to {report_path_e}")

if __name__ == '__main__':
    main()
