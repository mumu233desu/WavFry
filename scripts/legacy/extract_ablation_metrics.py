import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

runs = [
    ('ablation_weak_off', 'Baseline (Weak OFF)'),
    ('ablation_weak_mil_positive_only', 'Direction 1 (MIL Positive-Only)'),
    ('ablation_weak_mil_soft_weight', 'Direction 2 (MIL Soft Weight)'),
]

print("=" * 80)
print(f"{'Experiment':<35} | {'Creaky PR-AUC':<13} | {'Creaky F1':<10} | {'Breathy PR-AUC':<14} | {'Breathy F1':<10}")
print("-" * 80)

for run_id, label in runs:
    path = ROOT / f"exp_results/runs/{run_id}/epoch_history.json"
    if not path.exists():
        path = Path(f"exp_results/runs/{run_id}/epoch_history.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            hist = json.load(f)
    except Exception as e:
        print(f"Failed to read {path}: {e}")
        continue

    # Best epoch based on primary metric: Creaky PR-AUC
    best_ep = max(hist, key=lambda x: x.get("val_auc_pr_creaky", x.get("val_auc_pr", 0)))
    c_auc = best_ep.get("val_auc_pr_creaky", best_ep.get("val_auc_pr", 0))
    c_f1 = best_ep.get("threshold_f1_creaky", best_ep.get("threshold_f1", best_ep.get("val_f1", 0)))
    b_auc = best_ep.get("val_auc_pr_breathy", 0)
    b_f1 = best_ep.get("threshold_f1_breathy", 0)
    ep_num = best_ep.get("epoch", 0)

    print(f"{label} (Ep {ep_num:02d}) | {c_auc:<13.4f} | {c_f1:<10.4f} | {b_auc:<14.4f} | {b_f1:<10.4f}")

print("=" * 80)

# Now show detailed breakdown across all epochs for each run
for run_id, label in runs:
    path = ROOT / f"exp_results/runs/{run_id}/epoch_history.json"
    if not path.exists():
        path = Path(f"exp_results/runs/{run_id}/epoch_history.json")
    if not path.exists():
        continue
    with open(path, "r", encoding="utf-8") as f:
        hist = json.load(f)
    print(f"\n--- Detailed History: {label} ({run_id}) ---")
    print(f"{'Ep':<3} | {'Tr Loss':<7} | {'Val Loss':<8} | {'C-AUC':<7} | {'C-F1':<6} | {'C-Thresh':<8} | {'B-AUC':<7} | {'B-F1':<6} | {'B-Thresh':<8} | {'WeakLoss':<8}")
    print("-" * 85)
    for row in hist:
        ep = row.get("epoch", 0)
        tr_loss = row.get("train_loss", 0)
        val_loss = row.get("val_loss", 0)
        c_auc = row.get("val_auc_pr_creaky", row.get("val_auc_pr", 0))
        c_f1 = row.get("threshold_f1_creaky", row.get("threshold_f1", row.get("val_f1", 0)))
        c_th = row.get("best_threshold_creaky", row.get("best_threshold", 0.5))
        b_auc = row.get("val_auc_pr_breathy", 0)
        b_f1 = row.get("threshold_f1_breathy", 0)
        b_th = row.get("best_threshold_breathy", 0.5)
        w_loss = row.get("loss_weak_breathy", 0)
        is_best = "*" if row.get("is_best", False) else " "
        print(f"{ep:<2}{is_best} | {tr_loss:<7.4f} | {val_loss:<8.4f} | {c_auc:<7.4f} | {c_f1:<6.4f} | {c_th:<8.3f} | {b_auc:<7.4f} | {b_f1:<6.4f} | {b_th:<8.3f} | {w_loss:<8.4f}")
