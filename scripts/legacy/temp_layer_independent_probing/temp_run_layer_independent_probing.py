"""
Temporary Layer-wise Independent & Cumulative Probing Evaluator.

Extracts layer-wise hidden representations (L0 to L12) from frozen WavLM Base+
for the canonical test vowel intervals (1,371 tokens, 18 speakers).
Evaluates:
1. Single-Layer Independent Linear Probe (Logistic Regression) across L0-L12
2. Single-Layer Non-Linear MLP Probe (2-layer MLP) across L0-L12
3. Cumulative Prefix Re-trained Probe (L0 to Lk) across k=0..12

Uses 5-Fold Speaker-Grouped Cross-Validation to guarantee strictly unseen-speaker evaluation.

Outputs:
    data/experiments/cache/temp_layer_features_cache.npz
    data/experiments/temp_layer_redundancy_audit/temp_independent_probing_metrics.csv
    data/experiments/temp_layer_redundancy_audit/temp_cumulative_probing_metrics.csv
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set base directories
BASE_DIR = Path(r"E:\Coding\WavFry")
CODE_DIR = BASE_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from wavfry.eval.unified import build_test_manifest, load_model_from_checkpoint
from wavfry.model.wavlm_classifier import WavFryClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("temp_probing")

OUTPUT_DIR = BASE_DIR / "data" / "experiments" / "temp_layer_redundancy_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = BASE_DIR / "data" / "experiments" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_CACHE_PATH = CACHE_DIR / "temp_layer_features_cache.npz"


def extract_or_load_layer_features(
    test_manifest: pd.DataFrame,
    device: torch.device,
    task: str = "creaky",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts or loads pooled hidden states [N_samples, 13, 768].
    Returns (features_3d, labels, speakers, interval_keys).
    """
    cache_path = CACHE_DIR / f"temp_layer_features_cache_{task}.npz"
    if not cache_path.exists() and task == "creaky" and FEATURE_CACHE_PATH.exists():
        cache_path = FEATURE_CACHE_PATH

    if cache_path.exists():
        logger.info(f"Loading cached layer features ({task}) from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        return data["features_3d"], data["labels"], data["speakers"], data["interval_keys"]

    logger.info(f"Extracting WavLM 13-layer features for task: {task}...")
    ckpt_path = CODE_DIR / "exp_results" / "checkpoints" / "ablation_backbone_wavlm_baseplus" / "best_model.pt"
    model, _ = load_model_from_checkpoint(ckpt_path, device, WavFryClassifier)
    model.eval()

    staged_clean_dir = CODE_DIR / "data" / "experiments" / "cache" / "staged_audio" / "clean"
    unique_fkeys = test_manifest["file_key"].unique()
    file_hidden_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    # Build file key to path mapping (prefer staged clean, fallback to manifest audio_path)
    fkey_to_path: Dict[str, Path] = {}
    for _, row in test_manifest.iterrows():
        fkey = str(row["file_key"])
        if fkey not in fkey_to_path:
            staged_p = staged_clean_dir / f"{fkey}.wav"
            if staged_p.exists():
                fkey_to_path[fkey] = staged_p
            elif "audio_path" in row and Path(str(row["audio_path"])).exists():
                fkey_to_path[fkey] = Path(str(row["audio_path"]))

    t0 = time.time()
    logger.info(f"Running WavLM forward pass for {len(unique_fkeys)} unique audio files ({task})...")
    with torch.no_grad():
        for i, fkey in enumerate(unique_fkeys):
            if fkey not in fkey_to_path:
                logger.warning(f"Audio missing for {fkey}")
                continue
            wav_p = fkey_to_path[fkey]
            audio, sr = sf.read(str(wav_p), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            if sr != 16000:
                import torchaudio.functional as AF
                audio_t = AF.resample(torch.from_numpy(audio), orig_freq=sr, new_freq=16000)
                audio = audio_t.numpy()

            x = torch.from_numpy(audio).unsqueeze(0).to(device)
            res = model.wavlm(x, output_hidden_states=True)
            # Tuple of 13 tensors, each [1, T, 768]
            hs_stack = torch.stack(res.hidden_states, dim=1).squeeze(0).cpu().numpy()  # [13, T, 768]
            T = hs_stack.shape[1]
            timestamps = (np.arange(T, dtype=np.float32) * 320.0 + 160.0) / 16000.0
            file_hidden_cache[fkey] = (timestamps, hs_stack)
            if (i + 1) % 15 == 0 or (i + 1) == len(unique_fkeys):
                logger.info(f"  Processed {i+1}/{len(unique_fkeys)} files ({time.time()-t0:.1f}s)...")

    # Pool interval features across all tokens
    all_feats = []
    labels = []
    speakers = []
    interval_keys = []
    label_col = f"label_{task}"

    for _, row in test_manifest.iterrows():
        fkey = str(row["file_key"])
        s_time = float(row["start_s"])
        e_time = float(row["end_s"])
        y_val = int(row[label_col])
        spk = str(row["speaker_id"])
        int_key = f"{row['dataset_id']}::{fkey}::{s_time:.4f}_{e_time:.4f}"

        if fkey not in file_hidden_cache:
            continue
        timestamps, hs_stack = file_hidden_cache[fkey]
        mask = (timestamps >= s_time) & (timestamps <= e_time)
        if np.any(mask):
            feat_pooled = np.mean(hs_stack[:, mask, :], axis=1)  # [13, 768]
        else:
            idx = int(np.argmin(np.abs(timestamps - (s_time + e_time) / 2.0)))
            feat_pooled = hs_stack[:, idx, :]  # [13, 768]

        all_feats.append(feat_pooled)
        labels.append(y_val)
        speakers.append(spk)
        interval_keys.append(int_key)

    features_3d = np.array(all_feats, dtype=np.float32)  # [N, 13, 768]
    labels_arr = np.array(labels, dtype=np.int32)
    speakers_arr = np.array(speakers, dtype=object)
    keys_arr = np.array(interval_keys, dtype=object)

    target_cache = CACHE_DIR / f"temp_layer_features_cache_{task}.npz"
    np.savez_compressed(
        target_cache,
        features_3d=features_3d,
        labels=labels_arr,
        speakers=speakers_arr,
        interval_keys=keys_arr,
    )
    logger.info(f"Saved feature cache: {target_cache} (Shape: {features_3d.shape})")
    return features_3d, labels_arr, speakers_arr, keys_arr


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
) -> np.ndarray:
    """Trains a 2-layer MLP probe and returns test probabilities."""
    in_dim = X_train.shape[1]
    probe = FastMLPProbe(in_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    ds_tr = TensorDataset(torch.from_numpy(X_train).to(device), torch.from_numpy(y_train).float().to(device))
    loader = DataLoader(ds_tr, batch_size=64, shuffle=True)

    probe.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = probe(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()

    probe.eval()
    with torch.no_grad():
        x_te = torch.from_numpy(X_test).to(device)
        logits = probe(x_te)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def run_speaker_cv_probing(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    probe_type: str = "linear",
    device: torch.device = torch.device("cpu"),
) -> Tuple[float, float]:
    """Runs 5-fold speaker-grouped cross validation."""
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
        else:
            probs = train_eval_mlp_probe(X_tr, y_tr, X_te, y[test_idx], device=device, epochs=25)

        oof_preds[test_idx] = probs

    pr_auc = float(average_precision_score(y, oof_preds))
    roc_auc = float(roc_auc_score(y, oof_preds))
    return pr_auc, roc_auc


def run_probing_for_task(task: str, full_manifest: pd.DataFrame, device: torch.device):
    logger.info(f"=== Running Layer Probing Audit for Task: {task.upper()} ===")
    test_manifest = build_test_manifest(full_manifest, task=task)
    logger.info(f"Loaded test set ({task}): {len(test_manifest)} tokens across {test_manifest['speaker_id'].nunique()} speakers.")

    features_3d, y, groups, _ = extract_or_load_layer_features(test_manifest, device, task=task)
    n_tokens, n_layers, d_model = features_3d.shape
    layer_names = ["CNN_L0"] + [f"Transformer_L{i}" for i in range(1, 13)]

    # 1. Single-Layer Independent Probing across L0 to L12
    ind_results = []
    for l_idx in range(n_layers):
        l_name = layer_names[l_idx]
        X_l = features_3d[:, l_idx, :]  # [N, 768]

        # Linear Probe
        pr_lin, roc_lin = run_speaker_cv_probing(X_l, y, groups, probe_type="linear")
        # MLP Probe
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
            f"  [{task}] {l_name:<16} | Linear PR-AUC: {pr_lin:.4f} (ROC: {roc_lin:.4f}) | MLP PR-AUC: {pr_mlp:.4f} (ROC: {roc_mlp:.4f})"
        )

    df_ind = pd.DataFrame(ind_results)
    out_csv_ind = OUTPUT_DIR / f"temp_independent_probing_metrics_{task}.csv"
    df_ind.to_csv(out_csv_ind, index=False)
    if task == "creaky":
        df_ind.to_csv(OUTPUT_DIR / "temp_independent_probing_metrics.csv", index=False)
    logger.info(f"Saved independent probing metrics for {task}: {out_csv_ind}")

    # 2. Cumulative Prefix Re-trained Probing (L0..Lk for k=0..12)
    logger.info(f"=== Running Cumulative Prefix Probes ({task}) ===")
    cum_results = []

    for k in range(n_layers):
        k_name = f"Prefix_L0_to_L{k}"
        X_cum = features_3d[:, : k + 1, :].reshape(n_tokens, -1)  # [N, (k+1)*768]

        # Linear probe on cumulative prefix
        pr_cum_lin, roc_cum_lin = run_speaker_cv_probing(X_cum, y, groups, probe_type="linear")

        cum_results.append({
            "task": task,
            "prefix_k": k,
            "prefix_name": k_name,
            "num_layers_included": k + 1,
            "feature_dim": X_cum.shape[1],
            "cumulative_linear_pr_auc": round(pr_cum_lin, 4),
            "cumulative_linear_roc_auc": round(roc_cum_lin, 4),
        })
        logger.info(
            f"  [{task}] {k_name:<18} (Dim: {X_cum.shape[1]:<5}) | Cumulative PR-AUC: {pr_cum_lin:.4f} (ROC: {roc_cum_lin:.4f})"
        )

    df_cum = pd.DataFrame(cum_results)
    out_csv_cum = OUTPUT_DIR / f"temp_cumulative_probing_metrics_{task}.csv"
    df_cum.to_csv(out_csv_cum, index=False)
    if task == "creaky":
        df_cum.to_csv(OUTPUT_DIR / "temp_cumulative_probing_metrics.csv", index=False)
    logger.info(f"Saved cumulative prefix probing metrics for {task}: {out_csv_cum}")

    return df_ind, df_cum


def main():
    logger.info("Initializing Layer-wise Independent & Cumulative Probing Audit (Dual-Task: Creaky & Breathy)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load canonical test manifest
    manifest_p = CODE_DIR / "manifests" / "segments_multi.csv"
    full_manifest = pd.read_csv(manifest_p, low_memory=False)

    # Run for Creaky Voice
    run_probing_for_task("creaky", full_manifest, device)

    # Run for Breathy Voice
    run_probing_for_task("breathy", full_manifest, device)

    logger.info("All dual-task layer probing evaluations completed successfully.")


if __name__ == "__main__":
    main()
