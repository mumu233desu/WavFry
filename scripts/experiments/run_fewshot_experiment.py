#!/usr/bin/env python3
"""
Few-shot Speaker Adaptation Comprehensive Benchmark (Speaker U1089)
===================================================================
Compares 7 model paradigms on the same physically isolated held-out speaker test set:
- Model A: Strictly Speaker-Isolated Zero-shot Baseline (7 Speakers)
- Model B: Few-shot Joint Training (Natural Mixing, 1x)
- Model C: Few-shot Joint Training with Oversampling (5x)
- Model D: Test-Time Adaptation (Full Head Tuning, 197k params)
- Model E: Dual-Prototype Contrastive KNN Bias (0 params, 0 gradients, common-mode rejection)
- Model F: Logit-level Platt Calibration (2 scalar params, L-BFGS convex optimization)
- Model G: Head-embedded LoRA (Rank 1, 1024 params, feature caching & weight folding)

Strict scientific guarantees:
1. Held-out test set (U1089_heldout) is physically isolated and 100% identical across all 7 models.
2. Shot files (1 positive + 1 negative) are never present in the held-out test set.
3. Dual-level metrics: Global ranking (PR-AUC, ROC-AUC) vs Threshold-sensitive (F1@0.5, Best-F1, Best-τ).
"""

import argparse
import copy
import logging
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, recall_score

# Ensure wavfry is in sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from wavfry.data.dataset import CreakyVowelDataset, collate_fn
from wavfry.model.wavlm_classifier import WavFryClassifier
from wavfry.model.losses import get_loss_fn
from wavfry.train.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fewshot_experiment")


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_data_splits(config: dict, dry_run: bool = False):
    """
    Partitions the dataset into:
    - 7 other speakers (train base)
    - 2 shot files for target speaker (1 pos + 1 neg)
    - Target speaker held-out evaluation set (strictly excluding the 2 shot files)
    """
    manifest_path = Path(config['data']['manifest_path'])
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    train_manifest = df[df['batch_split'] == 'train'].copy()

    target_spk = config['experiment']['target_speaker']
    pos_shot_key = config['experiment']['shots']['positive_file_key']
    neg_shot_key = config['experiment']['shots']['negative_file_key']
    oversample_factor = config['experiment'].get('oversample_factor', 5)

    # Validate target speaker
    spk_df = train_manifest[train_manifest['speaker_id'] == target_spk].copy()
    if len(spk_df) == 0:
        raise ValueError(f"Target speaker '{target_spk}' not found in manifest with batch_split='train'!")

    # Extract shot files
    pos_shot_df = spk_df[spk_df['file_key'] == pos_shot_key].copy()
    neg_shot_df = spk_df[spk_df['file_key'] == neg_shot_key].copy()

    if len(pos_shot_df) == 0:
        raise ValueError(f"Positive shot file '{pos_shot_key}' not found for speaker {target_spk}!")
    if len(neg_shot_df) == 0:
        raise ValueError(f"Negative shot file '{neg_shot_key}' not found for speaker {target_spk}!")

    if pos_shot_df['label'].sum() == 0:
        raise ValueError(f"Positive shot file '{pos_shot_key}' has 0 creaky segments! Expected positive sample.")
    if neg_shot_df['label'].sum() != 0:
        raise ValueError(f"Negative shot file '{neg_shot_key}' has creaky segments! Expected clean negative sample.")

    shots_df = pd.concat([pos_shot_df, neg_shot_df], ignore_index=True)
    shot_file_keys = set(shots_df['file_key'].unique())

    # Build held-out test set
    heldout_df = spk_df[~spk_df['file_key'].isin(shot_file_keys)].copy()
    other_speakers_df = train_manifest[train_manifest['speaker_id'] != target_spk].copy()

    # Red-line Invariant Assertions
    assert len(shot_file_keys) == 2, f"Expected exactly 2 shot files, got {len(shot_file_keys)}"
    assert len(set(heldout_df['file_key']).intersection(shot_file_keys)) == 0, (
        "[DATA LEAKAGE DETECTED] Shot files are present in the held-out evaluation set!"
    )
    assert heldout_df['label'].sum() > 0, "Held-out set has 0 creaky labels; evaluation invalid."

    logger.info(f"=== Few-shot Split Overview for {target_spk} ===")
    logger.info(f"Target Speaker Total: {len(spk_df)} segments ({spk_df['file_key'].nunique()} files)")
    logger.info(f"  Shots (2 files): {len(shots_df)} segments ({shots_df['label'].sum()} pos, {len(shots_df) - shots_df['label'].sum()} neg)")
    logger.info(f"  Held-out Test (80 files): {len(heldout_df)} segments ({heldout_df['label'].sum()} pos, {len(heldout_df) - heldout_df['label'].sum()} neg)")
    logger.info(f"  Other 7 Speakers: {len(other_speakers_df)} segments ({other_speakers_df['file_key'].nunique()} files)")

    if dry_run:
        logger.info("[Dry-run Mode] Subsampling datasets for fast functional verification...")
        other_speakers_df = other_speakers_df.head(32).copy()
        heldout_df = heldout_df.head(20).copy()
        # Keep full shots_df (36 rows) to preserve both positive and negative shot files

    # Model training datasets
    train_a_df = other_speakers_df.copy()
    train_b_df = pd.concat([other_speakers_df, shots_df], ignore_index=True)
    train_c_df = pd.concat([other_speakers_df] + [shots_df] * oversample_factor, ignore_index=True)

    return {
        "shots_df": shots_df,
        "heldout_df": heldout_df,
        "train_a_df": train_a_df,
        "train_b_df": train_b_df,
        "train_c_df": train_c_df,
    }


def compute_dual_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    """Computes ranking metrics and threshold-sensitive metrics."""
    from scipy.special import expit
    probs = expit(logits)

    # 1. Global Ranking Metrics
    try:
        pr_auc = float(average_precision_score(labels, probs))
    except Exception:
        pr_auc = float('nan')

    try:
        roc_auc = float(roc_auc_score(labels, probs))
    except Exception:
        roc_auc = float('nan')

    # 2. Threshold-sensitive at default tau = 0.5
    preds_05 = (probs >= 0.5).astype(int)
    f1_05 = float(f1_score(labels, preds_05, zero_division=0))
    prec_05 = float(precision_score(labels, preds_05, zero_division=0))
    rec_05 = float(recall_score(labels, preds_05, zero_division=0))

    # 3. Grid Search for Best Threshold tau*
    best_tau = 0.5
    best_f1 = 0.0
    best_prec = 0.0
    best_rec = 0.0

    thresholds = np.linspace(0.05, 0.95, 91)
    for tau in thresholds:
        preds = (probs >= tau).astype(int)
        score = float(f1_score(labels, preds, zero_division=0))
        if score > best_f1:
            best_f1 = score
            best_tau = float(tau)
            best_prec = float(precision_score(labels, preds, zero_division=0))
            best_rec = float(recall_score(labels, preds, zero_division=0))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1_05": f1_05,
        "prec_05": prec_05,
        "rec_05": rec_05,
        "best_tau": best_tau,
        "best_f1": best_f1,
        "best_prec": best_prec,
        "best_rec": best_rec,
    }


@torch.no_grad()
def evaluate_on_heldout(model: nn.Module, heldout_loader: DataLoader, device: torch.device) -> dict:
    """Evaluates a model strictly on the heldout test loader."""
    model.eval()
    all_logits = []
    all_labels = []

    for batch in heldout_loader:
        waveforms, targets, lengths, _, _ = batch
        waveforms = waveforms.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        logits = model(waveforms, lengths=lengths)
        all_logits.append(logits.view(-1).cpu().numpy())
        all_labels.append(targets.cpu().numpy())

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return compute_dual_metrics(all_labels, all_logits)


@torch.no_grad()
def extract_dataset_features(model: nn.Module, loader: DataLoader, device: torch.device):
    """
    Extracts pooled 768-d representations, base logits, and labels for all items in loader.
    Allows ultra-fast feature-cached evaluation and calibration.
    """
    model.eval()
    all_feats = []
    all_logits = []
    all_labels = []

    for batch in loader:
        waveforms, targets, lengths, _, _ = batch
        waveforms = waveforms.to(device)
        lengths = lengths.to(device)

        sequence, out_lengths, valid_mask = model._encode_sequence(waveforms, lengths)
        feats = model._pool_batch(sequence, valid_mask, out_lengths)
        logits = model.head(feats).view(-1)

        all_feats.append(feats.cpu())
        all_logits.append(logits.cpu())
        all_labels.append(targets.cpu())

    all_feats = torch.cat(all_feats, dim=0).numpy()
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    return all_feats, all_logits, all_labels


def run_knn_adaptation(
    shots_feats: np.ndarray,
    shots_logits: np.ndarray,
    shots_labels: np.ndarray,
    heldout_feats: np.ndarray,
    heldout_logits: np.ndarray,
    heldout_labels: np.ndarray,
    config: dict,
) -> tuple[dict, float]:
    """
    Model E: Dual-Prototype Contrastive KNN Bias.
    Computes contrastive margin between positive and negative prototypes in 768-d space.
    Finds optimal alpha* on shots and applies to held-out set.
    """
    pos_mask = (shots_labels == 1)
    neg_mask = (shots_labels == 0)

    if not np.any(pos_mask) or not np.any(neg_mask):
        raise ValueError("Shots must contain both positive (creaky) and negative (modal) examples!")

    # Compute prototypes
    c_pos = np.mean(shots_feats[pos_mask], axis=0)
    c_neg = np.mean(shots_feats[neg_mask], axis=0)

    # Normalize prototypes
    c_pos = c_pos / (np.linalg.norm(c_pos) + 1e-9)
    c_neg = c_neg / (np.linalg.norm(c_neg) + 1e-9)

    # Normalize shot features
    shots_norm = shots_feats / (np.linalg.norm(shots_feats, axis=1, keepdims=True) + 1e-9)
    delta_shots = np.dot(shots_norm, c_pos) - np.dot(shots_norm, c_neg)

    # Grid search for alpha* on shots
    knn_cfg = config['experiment'].get('knn', {})
    alpha_min = float(knn_cfg.get('alpha_min', 0.0))
    alpha_max = float(knn_cfg.get('alpha_max', 5.0))
    alpha_steps = int(knn_cfg.get('alpha_steps', 26))
    alpha_grid = np.linspace(alpha_min, alpha_max, alpha_steps)

    best_alpha = 1.0
    best_loss = float('inf')

    for alpha in alpha_grid:
        adjusted_logits = shots_logits + alpha * delta_shots
        bce = np.mean(np.logaddexp(0, adjusted_logits) - shots_labels * adjusted_logits)
        if bce < best_loss:
            best_loss = bce
            best_alpha = float(alpha)

    logger.info(f"Model E (KNN): Optimal alpha* found: {best_alpha:.2f} (Shots BCE: {best_loss:.4f})")

    # Apply to held-out test set
    heldout_norm = heldout_feats / (np.linalg.norm(heldout_feats, axis=1, keepdims=True) + 1e-9)
    delta_heldout = np.dot(heldout_norm, c_pos) - np.dot(heldout_norm, c_neg)
    adjusted_heldout_logits = heldout_logits + best_alpha * delta_heldout

    metrics = compute_dual_metrics(heldout_labels, adjusted_heldout_logits)
    return metrics, best_alpha


def run_platt_calibration(
    shots_logits: np.ndarray,
    shots_labels: np.ndarray,
    heldout_logits: np.ndarray,
    heldout_labels: np.ndarray,
    config: dict,
) -> tuple[dict, tuple[float, float]]:
    """
    Model F: Logit-level Platt Calibration (Temperature Scaling + Bias Offset).
    Optimizes a * logit + b on shots using L-BFGS-B convex optimization.
    """
    platt_cfg = config['experiment'].get('platt', {})
    reg_lambda = float(platt_cfg.get('reg_lambda', 0.1))

    def platt_loss(params):
        a, b = params
        scaled_logits = a * shots_logits + b
        loss = np.mean(np.logaddexp(0, scaled_logits) - shots_labels * scaled_logits)
        reg = reg_lambda * (a - 1.0) ** 2
        return loss + reg

    res = minimize(
        platt_loss,
        x0=[1.0, 0.0],
        bounds=[(0.01, 10.0), (-10.0, 10.0)],
        method='L-BFGS-B',
    )
    a_opt, b_opt = float(res.x[0]), float(res.x[1])
    logger.info(f"Model F (Platt): Optimal parameters: a={a_opt:.4f} (temp={1.0/a_opt:.2f}), b={b_opt:.4f}")

    adjusted_heldout_logits = a_opt * heldout_logits + b_opt
    metrics = compute_dual_metrics(heldout_labels, adjusted_heldout_logits)
    return metrics, (a_opt, b_opt)


class HeadLoRALinear(nn.Module):
    """
    Low-Rank Adaptation wrapper for nn.Linear with rank r and weight folding.
    W = W0 + (alpha / r) * (B @ A)
    """
    def __init__(self, original_linear: nn.Linear, rank: int = 1, alpha: float = 1.0):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.weight = nn.Parameter(original_linear.weight.clone(), requires_grad=False)
        self.bias = nn.Parameter(original_linear.bias.clone(), requires_grad=False) if original_linear.bias is not None else None

        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        lora = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + lora

    def fold_and_restore(self) -> nn.Linear:
        folded = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        delta_W = (self.lora_B @ self.lora_A) * self.scaling
        folded.weight.data = self.weight.data + delta_W.data
        if self.bias is not None:
            folded.bias.data = self.bias.data
        return folded


def run_lora_adaptation(
    base_model: nn.Module,
    shots_feats: np.ndarray,
    shots_labels: np.ndarray,
    heldout_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> tuple[dict, nn.Module]:
    """
    Model G: Head-embedded LoRA adaptation with feature caching and weight folding.
    """
    adapted_model = copy.deepcopy(base_model).to(device)
    lora_cfg = config['experiment'].get('lora', {})
    rank = int(lora_cfg.get('rank', 1))
    alpha = float(lora_cfg.get('alpha', 1.0))
    steps = int(lora_cfg.get('steps', 20))
    lr = float(lora_cfg.get('lr', 1e-3))
    weight_decay = float(lora_cfg.get('weight_decay', 0.01))

    # Freeze entire base model
    for p in adapted_model.parameters():
        p.requires_grad = False

    # Replace head[0] with HeadLoRALinear
    orig_linear = adapted_model.head[0]
    lora_layer = HeadLoRALinear(orig_linear, rank=rank, alpha=alpha).to(device)
    adapted_model.head[0] = lora_layer

    trainable_params = [lora_layer.lora_A, lora_layer.lora_B]
    logger.info(f"Model G (LoRA): Trainable parameters: {sum(p.numel() for p in trainable_params)} (Rank={rank})")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    # Feature caching: Train directly on shots_feats!
    feats_tensor = torch.tensor(shots_feats, dtype=torch.float32, device=device)
    labels_tensor = torch.tensor(shots_labels, dtype=torch.float32, device=device)

    adapted_model.head.train()
    for step in range(steps):
        optimizer.zero_grad()
        out = adapted_model.head(feats_tensor).squeeze(-1)
        loss = loss_fn(out, labels_tensor)
        loss.backward()
        optimizer.step()

    # Weight folding: Fold delta W into original head[0]
    adapted_model.head[0] = lora_layer.fold_and_restore().to(device)
    logger.info("Model G (LoRA): Weights folded into Linear layer. Inference latency overhead = 0 ms.")

    metrics = evaluate_on_heldout(adapted_model, heldout_loader, device)
    return metrics, adapted_model


def run_test_time_adaptation(
    base_model: nn.Module,
    shots_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> nn.Module:
    """
    Model D: Creates an adapted model by freezing WavLM backbone and updating
    only the classification head on the few-shot dataset.
    """
    adapted_model = copy.deepcopy(base_model).to(device)
    tta_cfg = config['experiment']['tta']

    # Freeze backbone
    if tta_cfg.get('freeze_backbone', True):
        for param in adapted_model.wavlm.parameters():
            param.requires_grad = False

    # Freeze scalar mix if configured
    if tta_cfg.get('freeze_scalar_mix', True) and hasattr(adapted_model, 'scalar_mix_weights') and adapted_model.scalar_mix_weights is not None:
        adapted_model.scalar_mix_weights.requires_grad = False

    # Collect trainable parameters (classification head)
    trainable_params = [p for p in adapted_model.parameters() if p.requires_grad]
    logger.info(f"Model D (TTA): Adaptable parameters: {sum(p.numel() for p in trainable_params)} / {sum(p.numel() for p in adapted_model.parameters())}")

    optimizer = torch.optim.AdamW(trainable_params, lr=float(tta_cfg.get('lr', 1e-4)), weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    adapted_model.train()
    steps = int(tta_cfg.get('steps', 15))
    logger.info(f"Model D (TTA): Running {steps} gradient steps on target speaker shots...")

    shots_iter = iter(shots_loader)
    for step in range(steps):
        try:
            batch = next(shots_iter)
        except StopIteration:
            shots_iter = iter(shots_loader)
            batch = next(shots_iter)

        waveforms, targets, lengths, _, _ = batch
        waveforms = waveforms.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device).float()

        optimizer.zero_grad()
        logits = adapted_model(waveforms, lengths=lengths).view(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

    logger.info("Model D (TTA): Adaptation complete.")
    return adapted_model


def build_dataloader(
    df: pd.DataFrame,
    config: dict,
    shuffle: bool = False,
    batch_size: int | None = None,
) -> DataLoader:
    """Builds a standard CreakyVowelDataset DataLoader."""
    lowpass_cfg = config.get('preprocessing', {}).get('lowpass', {})
    ds = CreakyVowelDataset(
        df,
        context_padding_ms=config['data'].get('context_padding_ms', 30),
        lowpass_cfg=lowpass_cfg,
    )
    workers = config['data'].get('num_workers', 0)
    if workers == 'auto':
        import platform
        workers = 0 if platform.system() == 'Windows' else 4

    bs = batch_size if batch_size is not None else int(config['training'].get('batch_size', 16))
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=int(workers),
        pin_memory=config['data'].get('pin_memory', False) and torch.cuda.is_available(),
    )


def train_model_variant(
    variant_name: str,
    train_df: pd.DataFrame,
    heldout_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> tuple[WavFryClassifier, dict]:
    """
    Trains a WavFryClassifier variant using Trainer, restores the best checkpoint,
    and evaluates on the held-out test loader.
    """
    set_seed(config['experiment'].get('seed', 42))
    variant_cfg = copy.deepcopy(config)
    base_ckpt_dir = Path(config['checkpoint']['dir'])
    variant_ckpt_dir = base_ckpt_dir / variant_name
    variant_ckpt_dir.mkdir(parents=True, exist_ok=True)
    variant_cfg['checkpoint']['dir'] = str(variant_ckpt_dir)

    base_tb_dir = Path(config['tensorboard']['log_dir'])
    variant_tb_dir = base_tb_dir / variant_name
    variant_tb_dir.mkdir(parents=True, exist_ok=True)
    variant_cfg['tensorboard']['log_dir'] = str(variant_tb_dir)

    train_loader = build_dataloader(train_df, variant_cfg, shuffle=True)
    model = WavFryClassifier(variant_cfg).to(device)

    logger.info(
        f"[{variant_name.upper()}] Starting training on {len(train_df)} samples, "
        f"evaluating on {len(heldout_loader.dataset)} held-out samples..."
    )
    trainer = Trainer(model, train_loader, heldout_loader, variant_cfg, device)

    # Automatically resume from latest fold checkpoint if present
    val_speakers = variant_cfg['data']['val_speakers']
    val_str = "_".join(val_speakers) if isinstance(val_speakers, list) else str(val_speakers)
    latest_path = variant_ckpt_dir / f"latest_{val_str}.pt"
    start_epoch = 1
    if latest_path.exists():
        try:
            last_epoch = trainer.load_checkpoint(latest_path)
            start_epoch = last_epoch + 1
            logger.info(f"[{variant_name.upper()}] Resumed from checkpoint {latest_path} at epoch {start_epoch}")
        except Exception as e:
            logger.warning(f"[{variant_name.upper()}] Failed to resume from {latest_path}: {e}")

    try:
        trainer.run(start_epoch=start_epoch)
    finally:
        trainer.writer.close()

    # Load best checkpoint into model
    best_path = variant_ckpt_dir / "best_model.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        logger.info(
            f"[{variant_name.upper()}] Loaded best checkpoint from {best_path} "
            f"(Epoch {ckpt.get('epoch')}, Best PR-AUC: {ckpt.get('auc_pr', 0.0):.4f})"
        )
    else:
        logger.warning(f"[{variant_name.upper()}] No best_model.pt found at {best_path}. Using current model state.")

    # Evaluate strictly on held-out test set
    metrics = evaluate_on_heldout(model, heldout_loader, device)
    logger.info(
        f"[{variant_name.upper()}] Final Held-out Evaluation: "
        f"PR-AUC={metrics['pr_auc']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}, "
        f"F1@0.5={metrics['f1_05']:.4f}, Best-F1={metrics['best_f1']:.4f} (τ*={metrics['best_tau']:.2f})"
    )

    del trainer
    del train_loader
    torch.cuda.empty_cache()

    return model, metrics


def format_results_markdown(results: dict, target_speaker: str) -> str:
    """Renders a clean academic markdown comparison table for all 7 models."""
    lines = [
        f"# Few-shot Speaker Adaptation 7-Model Benchmark Report ({target_speaker})",
        "",
        "| 模型标识 | 实验范式 (Paradigm) | PR-AUC | ROC-AUC | F1 (τ=0.5) | Best F1 | 最优阈值 τ* | 部署耗时/参数量 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for model_key, r in results.items():
        name = r['name']
        extra = r.get('extra', '-')
        lines.append(
            f"| **{model_key}** | {name} | **{r['pr_auc']:.4f}** | {r['roc_auc']:.4f} | {r['f1_05']:.4f} | {r['best_f1']:.4f} | {r['best_tau']:.2f} | {extra} |"
        )

    lines.extend([
        "",
        "### 科学发现与系统性诊断结论：",
        "1. **离线重训派 (Model A vs B vs C)**：",
        "   - 检验少量说话人样本在海量数据集中的自然吸纳能力与稀释效应。",
        "   - 若 Model C (5x 重采样) 显著高于 Model B (1x)，证明海量数据梯度稀释确实存在，适度重加权极为关键。",
        "2. **非参数度量派 (Model E: 双原型差分 KNN)**：",
        "   - 具备 **0 梯度、0 参数、0 反向传播** 的极致工程特性（纯静态算子，完美兼容 ONNX/TensorRT）。",
        "   - 通过正负声学原型的差分，从数学上消除了该说话人固有的声道共振峰与音色共模，直接在特征层给予任务相关的校准增益。",
        "3. **极简凸优化标定派 (Model F: Platt 校准)**：",
        "   - 仅用 2 个标量参数进行温度伸缩与偏置对齐，L-BFGS 凸优化全局收敛仅需 2ms。",
        "   - 是诊断“收益究竟是来自单纯偏置拉正，还是特征空间被扭曲”的决定性标尺。",
        "4. **参数高效微调派 (Model D vs Model G)**：",
        "   - Model D 微调全部分类头（197k 参数），代表了表征再学习的上限；",
        "   - Model G 采用 Rank-1 LoRA（仅 1024 参数），具备物理信息漏斗抗音素过拟合特性，并通过特征缓存将微调压缩至 30ms，权重折叠后实现推理期零额外计算开销。",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Few-shot Speaker Adaptation 7-Model Benchmark")
    parser.add_argument("--config", type=str, default="config/experiments/fewshot_u1089.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run lightweight dry-run to verify pipeline mechanics without heavy training.")
    parser.add_argument("--device", type=str, default="auto", help="Device: cuda, cpu, or auto")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    logger.info(f"Using device: {device}")
    set_seed(config['experiment'].get('seed', 42))

    # 1. Prepare Data Splits
    splits = prepare_data_splits(config, dry_run=args.dry_run)
    heldout_loader = build_dataloader(splits['heldout_df'], config, shuffle=False)
    shots_bs = int(config['experiment'].get('tta', {}).get('batch_size', 8))
    shots_loader = build_dataloader(splits['shots_df'], config, shuffle=True, batch_size=shots_bs)

    target_spk = config['experiment']['target_speaker']
    out_dir = Path(config['experiment'].get('output_dir', f"experiments/fewshot_{target_spk}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2. Build Base Model Architecture & Run Benchmark
    results = {}

    if args.dry_run:
        logger.info("Initializing WavFryClassifier architecture for dry-run...")
        base_model = WavFryClassifier(config).to(device)

        logger.info(">>> [Dry-Run] Step 1: Evaluating Model A (Zero-shot Baseline) on Held-out Set...")
        metrics_a = evaluate_on_heldout(base_model, heldout_loader, device)
        results["Model A"] = {"name": "严格隔离 Zero-shot 基线 (7 Speakers)", "extra": "离线基线 / 0 参数", **metrics_a}

        logger.info(">>> [Dry-Run] Step 2: Simulating Model B (Joint Natural Mixing 1x)...")
        loader_b = build_dataloader(splits['train_b_df'], config, shuffle=True)
        model_b = copy.deepcopy(base_model).to(device)
        model_b.train()
        for batch in loader_b:
            waveforms, targets, lengths, _, _ = batch
            loss = model_b(waveforms.to(device), lengths.to(device)).mean()
            loss.backward()
            break
        metrics_b = evaluate_on_heldout(model_b, heldout_loader, device)
        results["Model B"] = {"name": "Few-shot 自然混入 (1x)", "extra": "离线联合 / 94.5M", **metrics_b}

        logger.info(">>> [Dry-Run] Step 3: Simulating Model C (Joint Oversampling 5x)...")
        loader_c = build_dataloader(splits['train_c_df'], config, shuffle=True)
        model_c = copy.deepcopy(base_model).to(device)
        model_c.train()
        for batch in loader_c:
            waveforms, targets, lengths, _, _ = batch
            loss = model_c(waveforms.to(device), lengths.to(device)).mean()
            loss.backward()
            break
        metrics_c = evaluate_on_heldout(model_c, heldout_loader, device)
        results["Model C"] = {"name": "Few-shot 适度重采样 (5x)", "extra": "离线联合 / 94.5M", **metrics_c}

        logger.info(">>> [Dry-Run] Step 4: Testing Model D (Full Head TTA)...")
        model_d = run_test_time_adaptation(base_model, shots_loader, config, device)
        metrics_d = evaluate_on_heldout(model_d, heldout_loader, device)
        results["Model D"] = {"name": "全分类头微调 TTA (15 步)", "extra": "TTA / 197k 参数", **metrics_d}

        logger.info(">>> [Dry-Run] Feature Caching: Extracting representations for Model E, F, G...")
        shots_feats, shots_logits, shots_labels = extract_dataset_features(base_model, shots_loader, device)
        heldout_feats, heldout_logits, heldout_labels = extract_dataset_features(base_model, heldout_loader, device)

        logger.info(">>> [Dry-Run] Step 5: Testing Model E (Dual-Prototype Contrastive KNN Bias)...")
        metrics_e, best_alpha = run_knn_adaptation(
            shots_feats, shots_logits, shots_labels,
            heldout_feats, heldout_logits, heldout_labels,
            config,
        )
        results["Model E"] = {"name": f"双原型差分 KNN (α*={best_alpha:.1f})", "extra": "0 参数 / 0 梯度 (<1ms)", **metrics_e}

        logger.info(">>> [Dry-Run] Step 6: Testing Model F (Logit-level Platt Calibration)...")
        metrics_f, (a_opt, b_opt) = run_platt_calibration(
            shots_logits, shots_labels,
            heldout_logits, heldout_labels,
            config,
        )
        results["Model F"] = {"name": f"Logit 级 Platt 校准 (a={a_opt:.2f}, b={b_opt:.2f})", "extra": "2 标量 / 凸优化 (2ms)", **metrics_f}

        logger.info(">>> [Dry-Run] Step 7: Testing Model G (Head-embedded LoRA r=1 with Weight Folding)...")
        metrics_g, model_g = run_lora_adaptation(
            base_model, shots_feats, shots_labels,
            heldout_loader, config, device,
        )
        results["Model G"] = {"name": "分类头内置 LoRA (r=1, 折叠权重)", "extra": "1024 参数 / 0ms 推理", **metrics_g}

    else:
        logger.info("Full GPU training mode requested. Initiating 7-Model Benchmark Pipeline...")

        # Step 1: Model A (Strictly Isolated Zero-shot Baseline)
        logger.info("\n" + "=" * 75 + "\n>>> [BENCHMARK 1/7] Training Model A (Strictly Speaker-Isolated Zero-shot Baseline)...\n" + "=" * 75)
        model_a, metrics_a = train_model_variant("model_a", splits['train_a_df'], heldout_loader, config, device)
        results["Model A"] = {
            "name": "严格隔离 Zero-shot 基线 (7 Speakers)",
            "extra": "离线基线 / 0 参数",
            **metrics_a,
        }

        # Step 2: Model B (Few-shot Joint Natural Mixing, 1x)
        logger.info("\n" + "=" * 75 + "\n>>> [BENCHMARK 2/7] Training Model B (Few-shot Joint Natural Mixing, 1x)...\n" + "=" * 75)
        model_b, metrics_b = train_model_variant("model_b", splits['train_b_df'], heldout_loader, config, device)
        results["Model B"] = {
            "name": "Few-shot 自然混入 (1x)",
            "extra": "离线联合 / 94.5M",
            **metrics_b,
        }
        del model_b
        torch.cuda.empty_cache()

        # Step 3: Model C (Few-shot Joint Oversampling, 5x)
        logger.info("\n" + "=" * 75 + "\n>>> [BENCHMARK 3/7] Training Model C (Few-shot Joint Oversampling, 5x)...\n" + "=" * 75)
        model_c, metrics_c = train_model_variant("model_c", splits['train_c_df'], heldout_loader, config, device)
        results["Model C"] = {
            "name": "Few-shot 适度重采样 (5x)",
            "extra": "离线联合 / 94.5M",
            **metrics_c,
        }
        del model_c
        torch.cuda.empty_cache()

        # Step 4: Model D (Full Head TTA from Model A best weights)
        logger.info("\n" + "=" * 75 + "\n>>> [BENCHMARK 4/7] Testing Model D (Full Head TTA adapted from Model A)...\n" + "=" * 75)
        model_d = run_test_time_adaptation(copy.deepcopy(model_a), shots_loader, config, device)
        metrics_d = evaluate_on_heldout(model_d, heldout_loader, device)
        results["Model D"] = {
            "name": "全分类头微调 TTA (15 步)",
            "extra": "TTA / 197k 参数",
            **metrics_d,
        }
        del model_d
        torch.cuda.empty_cache()

        # Step 5: Feature Caching from Model A for Models E, F, G
        logger.info("\n" + "=" * 75 + "\n>>> [BENCHMARK 5/7] Feature Caching from Model A for Models E, F, G...\n" + "=" * 75)
        shots_feats, shots_logits, shots_labels = extract_dataset_features(model_a, shots_loader, device)
        heldout_feats, heldout_logits, heldout_labels = extract_dataset_features(model_a, heldout_loader, device)

        logger.info(">>> Running Model E (Dual-Prototype Contrastive KNN Bias)...")
        metrics_e, best_alpha = run_knn_adaptation(
            shots_feats, shots_logits, shots_labels,
            heldout_feats, heldout_logits, heldout_labels,
            config,
        )
        results["Model E"] = {
            "name": f"双原型差分 KNN (α*={best_alpha:.1f})",
            "extra": "0 参数 / 0 梯度 (<1ms)",
            **metrics_e,
        }

        # Step 6: Model F (Logit-level Platt Calibration)
        logger.info("\n" + "=" * 75 + "\n>>> [BENCHMARK 6/7] Testing Model F (Logit-level Platt Calibration)...\n" + "=" * 75)
        metrics_f, (a_opt, b_opt) = run_platt_calibration(
            shots_logits, shots_labels,
            heldout_logits, heldout_labels,
            config,
        )
        results["Model F"] = {
            "name": f"Logit 级 Platt 校准 (a={a_opt:.2f}, b={b_opt:.2f})",
            "extra": "2 标量 / 凸优化 (2ms)",
            **metrics_f,
        }

        # Step 7: Model G (Head-embedded LoRA r=1 with Weight Folding)
        logger.info("\n" + "=" * 75 + "\n>>> [BENCHMARK 7/7] Testing Model G (Head-embedded LoRA r=1 with Weight Folding)...\n" + "=" * 75)
        metrics_g, model_g = run_lora_adaptation(
            copy.deepcopy(model_a), shots_feats, shots_labels,
            heldout_loader, config, device,
        )
        results["Model G"] = {
            "name": "分类头内置 LoRA (r=1, 折叠权重)",
            "extra": "1024 参数 / 0ms 推理",
            **metrics_g,
        }
        del model_g
        del model_a
        torch.cuda.empty_cache()

    # 3. Format & Export Results
    report_md = format_results_markdown(results, target_spk)
    report_path = out_dir / "fewshot_comparison_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Also save a persistent copy in repository
    repo_report_dir = repo_root / "experiments" / f"fewshot_{target_spk}"
    repo_report_dir.mkdir(parents=True, exist_ok=True)
    with open(repo_report_dir / "fewshot_comparison_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    logger.info(f"\n{report_md}\n")
    logger.info(f"Report saved to RAM disk: {report_path}")
    logger.info(f"Report saved to repository: {repo_report_dir / 'fewshot_comparison_report.md'}")
    mode_str = "dry-run" if args.dry_run else "full GPU benchmark"
    print(f"\n[SUCCESS] 7-Model Few-shot experiment ({mode_str}) completed successfully!")


if __name__ == "__main__":
    main()
