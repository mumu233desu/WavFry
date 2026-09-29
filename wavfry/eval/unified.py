"""Shared evaluation utilities.

Single source of truth for:
- unified test-set construction (dataset / split / phone-type / NaN-label filters)
- checkpoint model loading (delta + full)
- frozen decision-threshold resolution (fail-fast, no silent constants)
- NaN-safe threshold grid scan

Every evaluation script must build its test set and resolve its threshold
through this module so that numbers reported in different tables are computed
on the same samples under the same protocol.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def build_test_manifest(
    manifest_df: pd.DataFrame,
    task: str = "creaky",
    require_label: bool = True,
) -> pd.DataFrame:
    """Construct the unified test set from a multi-dataset manifest.

    Filters, applied in order:
    1. ``dataset_id == "cs_dialogue"`` (when the column exists)
    2. ``batch_split == "test"`` (with legacy fallbacks to ``split`` / batch_id)
    3. ``phone_type == "vowel"`` (when the column exists)
    4. non-NaN labels for ``label_{task}`` — NaN rows are unlabeled for this
       task and must never be silently counted as negatives.

    Returns a copy with a reset index; callers may attach derived columns.
    """
    df = manifest_df.copy()

    if "dataset_id" in df.columns:
        df = df[df["dataset_id"] == "cs_dialogue"]

    if "batch_split" in df.columns:
        df = df[df["batch_split"] == "test"]
    elif "split" in df.columns:
        df = df[df["split"] == "test"]
    elif "batch_id" in df.columns:
        b0_spks = set(df[df["batch_id"] == 0]["speaker_id"].unique())
        df = df[df["batch_id"].isin([2, 3]) & (~df["speaker_id"].isin(b0_spks))]
    else:
        raise KeyError(
            "build_test_manifest: manifest has no batch_split/split/batch_id column; "
            "cannot identify the test partition."
        )

    if "phone_type" in df.columns:
        df = df[df["phone_type"] == "vowel"]

    label_col = f"label_{task}" if f"label_{task}" in df.columns else "label"
    labels = pd.to_numeric(df[label_col], errors="coerce")
    n_before = len(df)
    if require_label:
        df = df[~labels.isna()]
    if n_before - len(df) > 0:
        logger.warning(
            f"build_test_manifest: dropped {n_before - len(df)} rows without a "
            f"'{label_col}' label (unlabeled for this task)."
        )

    return df.copy().reset_index(drop=True)


def resolve_frozen_threshold(ckpt: Dict[str, Any], checkpoint_label: str = "") -> float:
    """Return the validation-frozen decision threshold stored in a checkpoint.

    Raises if the field is missing — a silent numeric fallback would compute
    the paper's P/R/F1 at a threshold that never came from the validation set.
    """
    if "best_threshold" not in ckpt or ckpt["best_threshold"] is None:
        raise RuntimeError(
            f"Checkpoint '{checkpoint_label}' has no frozen 'best_threshold'. "
            "This field is injected by the trainer after validation-set threshold "
            "grid search. Refusing to fall back to an arbitrary constant — "
            "evaluate with a checkpoint produced by the current training pipeline."
        )
    return float(ckpt["best_threshold"])


def load_model_from_checkpoint(
    ckpt_path: Path,
    device: torch.device,
    model_mod: Any,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load a WavFryClassifier (delta or full state dict) from a checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, Exception):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = model_mod.WavFryClassifier(config)
    save_mode = config.get("checkpoint", {}).get("save_mode", "delta")
    if save_mode == "delta":
        result = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if getattr(result, "unexpected_keys", None):
            logger.warning(f"Unexpected keys in delta checkpoint: {result.unexpected_keys}")
    else:
        model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, config


def predict_segment_logits(
    model: torch.nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    dataset_mod: Any,
    task: str = "creaky",
    context_padding_ms: float = 30.0,
    batch_size: int = 64,
    num_workers: int = 0,
) -> torch.Tensor:
    """Run segment-paradigm inference and return one logit per manifest row.

    The returned tensor is aligned with ``df`` row order. Raises if the
    dataset/loader yields a different number of predictions than input rows
    (e.g. audio files silently missing), which would misalign logits with
    labels in any downstream metric.
    """
    ds = dataset_mod.CreakyVowelDataset(
        df,
        context_padding_ms=context_padding_ms,
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=dataset_mod.collate_fn,
        pin_memory=device.type == "cuda",
    )

    all_logits = []
    with torch.no_grad():
        for waveforms, _, lengths, _, _ in loader:
            waveforms = waveforms.to(device)
            lengths = lengths.to(device)
            outputs = model(waveforms, lengths)
            if isinstance(outputs, dict):
                if task not in outputs:
                    raise KeyError(
                        f"Model has no head for task '{task}' (heads: {list(outputs)})."
                    )
                logits = outputs[task]
            else:
                logits = outputs
            all_logits.append(logits.detach().cpu().float().flatten())

    logits_tensor = torch.cat(all_logits)
    if len(logits_tensor) != len(df):
        raise RuntimeError(
            f"predict_segment_logits: got {len(logits_tensor)} logits for {len(df)} rows — "
            "predictions and labels are misaligned. Check for missing audio or "
            "dataset-level row filtering."
        )
    return logits_tensor


def f1_scan_best_threshold(
    logits: torch.Tensor,
    labels: torch.Tensor,
    step: float = 0.005,
) -> Tuple[float, float]:
    """NaN-safe F1 grid scan: returns (best_threshold, best_f1).

    Rows with NaN labels are excluded before scanning; if a caller feeds
    unfiltered labels the scan would otherwise count NaN-cast garbage as
    negatives. The threshold is expected to be scanned on validation data —
    never report a test-scanned threshold as a frozen operating point.
    """
    from sklearn.metrics import f1_score

    labels_t = labels.detach().cpu().float()
    valid_m = ~torch.isnan(labels_t)
    if not bool(valid_m.all()):
        logger.warning(
            f"f1_scan_best_threshold: excluded {int((~valid_m).sum())} NaN-label rows."
        )
    if not bool(valid_m.any()):
        raise ValueError("f1_scan_best_threshold: all labels are NaN.")
    logits_t = logits.detach().cpu().float()[valid_m]
    labels_t = labels_t[valid_m]

    probs = torch.sigmoid(logits_t).numpy()
    labels_np = labels_t.numpy().astype(np.int32)

    thresholds = [round(t, 3) for t in np.arange(0.01, 1.00, step).tolist()]
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(np.int32)
        score = float(f1_score(labels_np, preds, average="binary", zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_t = t
    return best_t, best_f1
