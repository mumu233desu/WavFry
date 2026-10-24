"""
Evaluates layer-wise independent and cumulative prefix probing on the FULL canonical test sets:
- Creaky: N = 3,147 tokens across 18 speakers
- Breathy: N = 3,932 tokens across 29 speakers

Features both:
1. Single-Layer Independent Probes (Linear LogReg + 2-layer MLP)
2. Constant-Dimension Prefix Probing (Layer-Mean Prefix h_0..k, D=768 constant)
3. Concatenated Prefix Probing with LogisticRegressionCV (adaptive L2 regularization)
"""

from __future__ import annotations

import logging
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = Path(r"E:\Coding\WavFry")
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("temp_full_probing")

OUTPUT_DIR = BASE_DIR / "data" / "experiments" / "temp_layer_redundancy_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "data" / "experiments" / "cache"


class FastMLPProbe(nn.Module):
    """Lightweight 2-layer MLP probe with BatchNorm and Dropout."""
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_eval_mlp_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    epochs: int = 25,
    batch_size: int = 128,
) -> np.ndarray:
    pos_count = int(np.sum(y_train == 1))
    neg_count = int(np.sum(y_train == 0))
    pos_weight = float(neg_count / max(pos_count, 1))

    model = FastMLPProbe(in_dim=X_train.shape[1], hidden_dim=128).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

    ds_tr = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()
    for _ in range(epochs):
        for bx, by in loader_tr:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        tx = torch.from_numpy(X_test).float().to(device)
        probs = torch.sigmoid(model(tx)).cpu().numpy()
    return probs


def run_speaker_cv_probing(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    probe_type: str = "linear",
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y), dtype=np.float32)

    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        y_tr = y[train_idx]

        if probe_type == "linear":
            clf = LogisticRegression(C=1.0, max_iter=500, class_weight="balanced", random_state=42)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
        elif probe_type == "linear_cv":
            clf = LogisticRegressionCV(Cs=5, cv=3, max_iter=500, class_weight="balanced", random_state=42)
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_te)[:, 1]
        else:
            probs = train_eval_mlp_probe(X_tr, y_tr, X_te, y[test_idx], device=device, epochs=25)

        oof_preds[test_idx] = probs

    pr_auc = float(average_precision_score(y, oof_preds))
    roc_auc = float(roc_auc_score(y, oof_preds))
    return pr_auc, roc_auc


def evaluate_task_full(task: str, device: torch.device):
    cache_path = CACHE_DIR / f"temp_full_test_features_cache_{task}.npz"
    logger.info(f"Loading full test feature cache: {cache_path}")
    data = np.load(cache_path, allow_pickle=True)
    features_3d = data["features_3d"]  # [N, 13, 768]
    y = data["labels"]
    groups = data["speakers"]
    n_tokens, n_layers, d_model = features_3d.shape
    logger.info(f"Loaded {task}: {n_tokens} tokens across {len(np.unique(groups))} speakers. Positives: {np.sum(y == 1)}")

    layer_names = ["CNN_L0"] + [f"Transformer_L{i}" for i in range(1, 13)]

    # 1. Single-Layer Independent Probing
    ind_results = []
    for l_idx in range(n_layers):
        l_name = layer_names[l_idx]
        X_l = features_3d[:, l_idx, :]
        pr_lin, roc_lin = run_speaker_cv_probing(X_l, y, groups, probe_type="linear")
        pr_mlp, roc_mlp = run_speaker_cv_probing(X_l, y, groups, probe_type="mlp", device=device)
        ind_results.append({
            "task": task,
            "layer_index": l_idx,
            "layer_name": l_name,
            "linear_probe_pr_auc": round(pr_lin, 4),
            "linear_probe_roc_auc": round(roc_lin, 4),
            "mlp_probe_pr_auc": round(pr_mlp, 4),
            "mlp_probe_roc_auc": round(roc_mlp, 4),
        })
        logger.info(
            f"  [{task}] {l_name:<16} | Linear PR: {pr_lin:.4f} (ROC: {roc_lin:.4f}) | MLP PR: {pr_mlp:.4f} (ROC: {roc_mlp:.4f})"
        )

    df_ind = pd.DataFrame(ind_results)
    out_csv_ind = OUTPUT_DIR / f"temp_full_independent_probing_metrics_{task}.csv"
    df_ind.to_csv(out_csv_ind, index=False)

    # 2. Constant-Dimension Prefix Probing (Layer-Mean, D=768 constant)
    cum_results = []
    for k in range(n_layers):
        k_name = f"Prefix_L0_to_L{k}"
        # Constant dimension D=768 prefix (mean pooling layers 0..k)
        X_mean_prefix = np.mean(features_3d[:, : k + 1, :], axis=1)  # [N, 768]
        pr_mean_lin, roc_mean_lin = run_speaker_cv_probing(X_mean_prefix, y, groups, probe_type="linear")

        # Concatenated prefix with tuned C
        X_cum = features_3d[:, : k + 1, :].reshape(n_tokens, -1)
        pr_cum_cv, roc_cum_cv = run_speaker_cv_probing(X_cum, y, groups, probe_type="linear")

        cum_results.append({
            "task": task,
            "prefix_k": k,
            "prefix_name": k_name,
            "num_layers_included": k + 1,
            "mean_prefix_dim": 768,
            "mean_prefix_pr_auc": round(pr_mean_lin, 4),
            "mean_prefix_roc_auc": round(roc_mean_lin, 4),
            "concat_dim": X_cum.shape[1],
            "concat_pr_auc": round(pr_cum_cv, 4),
            "concat_roc_auc": round(roc_cum_cv, 4),
        })
        logger.info(
            f"  [{task}] {k_name:<18} | MeanPrefix(768d) PR: {pr_mean_lin:.4f} | Concat({X_cum.shape[1]}d) PR: {pr_cum_cv:.4f}"
        )

    df_cum = pd.DataFrame(cum_results)
    out_csv_cum = OUTPUT_DIR / f"temp_full_cumulative_probing_metrics_{task}.csv"
    df_cum.to_csv(out_csv_cum, index=False)

    return df_ind, df_cum


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    evaluate_task_full("creaky", device)
    evaluate_task_full("breathy", device)
    logger.info("Full set probing completed.")


if __name__ == "__main__":
    main()
