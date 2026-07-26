import torch
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, average_precision_score, confusion_matrix, accuracy_score

def compute_metrics(logits, labels, threshold=0.5):
    """
    Compute classification metrics.
    logits: torch.Tensor of shape [B]
    labels: torch.Tensor of shape [B]
    """
    probs = torch.sigmoid(logits).detach().cpu().float().numpy()
    preds = (probs >= threshold).astype(np.int32)
    labels_np = labels.detach().cpu().numpy().astype(np.int32)
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds, average='binary', zero_division=0)
    
    # Handle case with only one class in labels (e.g. all non-creaky)
    try:
        auc_pr = average_precision_score(labels_np, probs)
    except ValueError:
        auc_pr = 0.0
        
    # Put Creaky (1) first to match common diagnostic practice
    cm = confusion_matrix(labels_np, preds, labels=[1, 0])
    
    accuracy = accuracy_score(labels_np, preds)
    
    return {
        'accuracy': float(accuracy),
        'f1': float(f1),
        'precision': float(precision),
        'recall': float(recall),
        'auc_pr': float(auc_pr),
        'cm': cm.tolist()
    }
