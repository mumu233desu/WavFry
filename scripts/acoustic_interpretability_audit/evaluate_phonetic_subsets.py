import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
df = pd.read_csv(ROOT / "manifests" / "segments_multi.csv", dtype={"speaker_id": str, "phone": str, "file_key": str})
test_df = df[df["batch_split"] == "test"].copy()

test_df["interval_id"] = (
    test_df["file_key"].astype(str)
    + "|"
    + test_df["start_s"].apply(lambda x: f"{float(x):.4f}")
    + "|"
    + test_df["end_s"].apply(lambda x: f"{float(x):.4f}")
    + "|"
    + test_df["phone"].fillna("UNK").astype(str)
)

pt_path = ROOT / "exp_results" / "checkpoints" / "ablation_backbone_wavlm_baseplus" / "test_predictions_wavlm_plus_clean.pt"
d = torch.load(pt_path, map_location="cpu", weights_only=False)

for task, thresh in [("creaky", 0.43), ("breathy", 0.44)]:
    t_preds = d["predictions"][task]
    iids = t_preds["interval_ids"]
    logits = t_preds["logits"].numpy()
    labels = t_preds["labels"].numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    pred_map = pd.DataFrame({"interval_id": iids, "prob": probs, "pred_label": labels})

    sub_df = test_df[test_df[f"label_{task}"].notna()].copy()
    merged = pd.merge(sub_df, pred_map, on="interval_id", how="inner")

    out_msg = f"\n==================== TASK: {task.upper()} (Merged N={len(merged)} / Total={len(sub_df)}) ====================\n"
    sys.stdout.buffer.write(out_msg.encode("utf-8"))

    subsets = [
        ("Pure Vowel", merged[merged["phone_type"] == "vowel"]),
        ("Sonorant only", merged[merged["phone_type"] == "sonorant"]),
        ("Vowel + Sonorant", merged[merged["phone_type"].isin(["vowel", "sonorant"])]),
        ("Voiceless Obs", merged[merged["phone_type"] == "voiceless_obs"]),
        ("Voiced Obs", merged[merged["phone_type"] == "voiced_obs"]),
        ("Silence", merged[merged["phone_type"] == "silence"]),
        ("Unknown", merged[merged["phone_type"] == "unknown"]),
        ("All Extended Phonemes", merged),
    ]

    for name, sub in subsets:
        n = len(sub)
        if n == 0:
            continue
        yt = sub[f"label_{task}"].values.astype(int)
        yp = sub["prob"].values
        pos = int(np.sum(yt))
        pr = average_precision_score(yt, yp) if pos > 0 and pos < n else 0.0
        roc = roc_auc_score(yt, yp) if pos > 0 and pos < n else 0.0
        yp_bin = (yp >= thresh).astype(int)
        f1 = f1_score(yt, yp_bin, zero_division=0.0) if pos > 0 else 0.0
        prec = precision_score(yt, yp_bin, zero_division=0.0) if pos > 0 else 0.0
        rec = recall_score(yt, yp_bin, zero_division=0.0) if pos > 0 else 0.0

        line = (
            f"{name:22s} | N={n:5d} ({n/len(merged)*100:5.2f}%) | "
            f"Pos={pos:4d} ({pos/n*100:5.2f}%) | "
            f"PR-AUC={pr:.4f} | ROC-AUC={roc:.4f} | "
            f"F1@{thresh}={f1:.4f} (P={prec:.4f}, R={rec:.4f})\n"
        )
        sys.stdout.buffer.write(line.encode("utf-8"))
