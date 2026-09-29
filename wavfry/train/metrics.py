import logging

import torch
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
)

logger = logging.getLogger(__name__)


def compute_metrics(logits, labels, threshold=0.5):
    """
    Compute classification metrics.
    logits: torch.Tensor of shape [B]
    labels: torch.Tensor of shape [B]

    Labels containing NaN (unlabeled rows, e.g. a task this dataset does not
    annotate) are filtered out before any metric computation — numpy silently
    casts NaN to -2147483648 on int conversion, which would otherwise poison
    TN/FN counts and every thresholded metric.
    """
    labels_t = labels.detach().cpu().float()
    valid_m = ~torch.isnan(labels_t)
    if not bool(valid_m.all()):
        n_dropped = int((~valid_m).sum().item())
        logger.warning(
            f"compute_metrics: dropped {n_dropped}/{labels_t.numel()} rows with NaN labels "
            "before metric computation."
        )
        if not bool(valid_m.any()):
            return {
                'accuracy': 0.0,
                'f1': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'auc_pr': float("nan"),
                'roc_auc': float("nan"),
                'cm': [[0, 0], [0, 0]],
            }
        labels_t = labels_t[valid_m]
        logits = torch.as_tensor(logits)[valid_m] if not isinstance(logits, torch.Tensor) else logits.detach().cpu().float()[valid_m]

    probs = torch.sigmoid(logits).detach().cpu().float().numpy()
    preds = (probs >= threshold).astype(np.int32)
    labels_np = labels_t.numpy().astype(np.int32)

    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds, average='binary', zero_division=0)

    # Handle case with only one class in labels (e.g. all non-creaky)
    try:
        auc_pr = float(average_precision_score(labels_np, probs))
    except ValueError:
        auc_pr = float("nan")
    try:
        roc_auc = float(roc_auc_score(labels_np, probs))
    except ValueError:
        roc_auc = float("nan")

    # Put Creaky (1) first to match common diagnostic practice
    cm = confusion_matrix(labels_np, preds, labels=[1, 0])

    accuracy = accuracy_score(labels_np, preds)

    return {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auc_pr': float(auc_pr),
        'roc_auc': float(roc_auc),
        'cm': cm.tolist()
    }
