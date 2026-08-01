import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
import io
import numpy as np
from pathlib import Path

from ..model import losses as losses_mod
get_loss_fn = losses_mod.get_loss_fn

from . import metrics as metrics_mod
compute_metrics = metrics_mod.compute_metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, augmentor=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.augmentor = augmentor
        self.paradigm = config['training'].get('paradigm', 'segment')
        self.frame_cfg = config['training'].get('frame_head', {})
        self._current_frame_proxy = None

        self.epochs = config['training']['epochs']
        grad_acc = config['training'].get('gradient_accumulation_steps', 1)
        import math
        steps_per_epoch = math.ceil(len(self.train_loader) / grad_acc)
        self.total_steps = self.epochs * steps_per_epoch
        
        # Optimizer with differential learning rate
        self.optimizer = self.create_optimizer()
        
        # Scheduler
        self.scheduler = self.create_scheduler()
        
        # Dataset construction already materializes epoch-1 generated windows, so
        # weighted BCE reflects the effective initial training distribution.
        if self.paradigm == 'block':
            interval_rows = [row for block in train_loader.dataset.blocks for row in block['intervals']]
            n_pos = sum(int(row['label']) for row in interval_rows)
            n_neg = len(interval_rows) - n_pos
        else:
            train_df = train_loader.dataset.df
            n_pos = train_df['label'].sum()
            n_neg = len(train_df) - n_pos
        pos_weight = torch.tensor([n_neg / (n_pos + 1e-9)]).to(device)
        self.loss_fn = get_loss_fn(config, pos_weight=pos_weight)
        
        # Mixed precision setup
        self.mixed_prec = config['training'].get('mixed_precision', 'no')
        self.use_amp = self.mixed_prec != 'no' and 'cuda' in str(device)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.autocast_dtype = torch.bfloat16 if self.mixed_prec == 'bf16' else torch.float16
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=config['tensorboard']['log_dir'])
        
        # Primary metric for early-stopping — driven by config (supports 'auc_pr' | 'f1')
        _valid_metrics = {'auc_pr', 'f1'}
        self.primary_metric = self.config['eval'].get('primary_metric', 'auc_pr')
        if self.primary_metric not in _valid_metrics:
            raise ValueError(
                f"eval.primary_metric must be one of {_valid_metrics}, "
                f"got '{self.primary_metric}'"
            )
        self.best_primary_metric_val = 0.0
        self.patience_counter = 0
        # Holds (logits, labels) from the best validation epoch for threshold search
        self._best_val_logits: torch.Tensor | None = None
        self._best_val_labels: torch.Tensor | None = None
        
        # Checkpoint dir
        self.ckpt_dir = Path(config['checkpoint']['dir'])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def create_optimizer(self):
        # Issue 4: AdamW weight decay optimization
        weight_decay = self.config['training'].get('weight_decay', 0.01)
        no_decay = ["bias", "LayerNorm.weight"]
        
        optimizer_grouped_parameters = [
            # 1. Backbone (WavLM) - No weight decay to protect pre-trained features
            {
                "params": [p for n, p in self.model.named_parameters() if "wavlm" in n and p.requires_grad],
                "weight_decay": 0.0,
                "lr": self.config['training']['backbone_lr']
            },
            # 2. Head (Weights) - Normal weight decay
            {
                "params": [p for n, p in self.model.named_parameters() if "wavlm" not in n and p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": self.config['training']['head_lr']
            },
            # 3. Head (Bias/LayerNorm) - No weight decay
            {
                "params": [p for n, p in self.model.named_parameters() if "wavlm" not in n and p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.config['training']['head_lr']
            },
        ]
        
        optimizer_type = self.config['training'].get('optimizer', 'adamw_fp32')
        if optimizer_type == 'adamw_8bit':
            import bitsandbytes as bnb
            print("[INFO] Using 8-bit AdamW optimizer (bitsandbytes)")
            return bnb.optim.AdamW8bit(optimizer_grouped_parameters)
        elif optimizer_type == 'adamw_fp32':
            return optim.AdamW(optimizer_grouped_parameters)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
    def create_scheduler(self):
        import math
        warmup_steps = self.config['training']['warmup_steps']
        total_steps = self.total_steps
        min_lr_ratio = self.config['training'].get('min_lr_ratio', 0.05)
        lr_schedule = self.config['training'].get('lr_schedule', 'cosine')

        # Ablation check
        use_cosine_restart = self.config['training'].get('ablation', {}).get('use_cosine_restart', True)
        if lr_schedule == 'cosine_restart' and not use_cosine_restart:
            lr_schedule = 'cosine'

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            
            if lr_schedule == 'none':
                return 1.0
                
            if lr_schedule == 'linear':
                progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))
                
            if lr_schedule == 'cosine_restart':
                aug_cfg = self.config.get('augmentation', {})
                if not aug_cfg:
                    aug_cfg = self.config.get('data', {}).get('augmentation', {})
                aug_schedule_power = aug_cfg.get('schedule_power', 2.0)
                restart_aug_threshold = self.config['training'].get('restart_aug_threshold', 0.11)
                restart_lr_ratio = self.config['training'].get('restart_lr_ratio', 0.45)
                restart_warmup_steps = self.config['training'].get('restart_warmup_steps', 100)
                
                restart_progress = restart_aug_threshold ** (1.0 / aug_schedule_power)
                restart_step = round(restart_progress * total_steps)
                restart_step = min(restart_step, total_steps - 1)
                
                if current_step < restart_step:
                    progress = (current_step - warmup_steps) / float(max(1, restart_step - warmup_steps))
                    cosine_val = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                    return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val)
                
                phase2_step = current_step - restart_step
                phase2_total = total_steps - restart_step
                
                if phase2_step < restart_warmup_steps:
                    return min_lr_ratio + (restart_lr_ratio - min_lr_ratio) * phase2_step / restart_warmup_steps
                
                phase2_progress = (phase2_step - restart_warmup_steps) / float(max(1, phase2_total - restart_warmup_steps))
                cosine_val = 0.5 * (1.0 + math.cos(math.pi * min(1.0, phase2_progress)))
                return max(min_lr_ratio, min_lr_ratio + (restart_lr_ratio - min_lr_ratio) * cosine_val)
            
            else:  # 'cosine' (default)
                progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                cosine_val = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
                return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val)

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _compute_block_loss(self, outputs, labels, slot_valid, training=True):
        self._current_frame_proxy = None
        slot_valid = outputs.get('interval_valid', slot_valid)
        logits = outputs['interval_logits'][slot_valid]
        targets = labels[slot_valid]
        interval_loss = self.loss_fn(logits, targets)
        if not self.frame_cfg.get('enabled', False):
            return interval_loss, logits, targets

        from ..model.losses import pool_frame_logits
        frame_logits = outputs['frame_logits']
        interval_masks = outputs['interval_masks']
        frame_masks = interval_masks[slot_valid]
        expanded = frame_logits.unsqueeze(1).expand_as(interval_masks)[slot_valid]
        mode = self.frame_cfg.get('mil_mode', 'lse')
        if mode == 'lse_mean':
            lse = pool_frame_logits(
                expanded, frame_masks, mode='lse',
                temperature=self.frame_cfg.get('temperature', 1.0),
            )
            mean = (expanded.masked_fill(~frame_masks, 0.0).sum(-1) /
                    frame_masks.sum(-1).clamp_min(1))
            mil_logits = self.model.frame_mil_fusion(
                torch.stack([lse, mean], dim=-1)
            ).squeeze(-1)
        else:
            mil_logits = pool_frame_logits(
                expanded,
                frame_masks,
                mode=mode,
                topk_ratio=self.frame_cfg.get('topk_ratio', 0.2),
                temperature=self.frame_cfg.get('temperature', 1.0),
            )
        if not training:
            try:
                from sklearn.metrics import roc_auc_score
                proxy_targets = targets.unsqueeze(-1).expand_as(frame_masks)[frame_masks].detach().cpu().numpy()
                proxy_scores = torch.sigmoid(expanded[frame_masks]).detach().cpu().float().numpy()
                if len(set(proxy_targets.tolist())) > 1:
                    self._current_frame_proxy = float(roc_auc_score(proxy_targets, proxy_scores))
            except (ValueError, ImportError):
                self._current_frame_proxy = None

        positive = targets > 0.5
        frame_terms = []
        if positive.any():
            frame_terms.append(torch.nn.functional.binary_cross_entropy_with_logits(
                mil_logits[positive], torch.ones_like(mil_logits[positive])
            ))
        negative = ~positive
        if negative.any():
            negative_logits = expanded[negative][frame_masks[negative]]
            frame_terms.append(torch.nn.functional.binary_cross_entropy_with_logits(
                negative_logits, torch.zeros_like(negative_logits)
            ))
        if frame_terms:
            # Sum class-specific means so abundant negative frames do not silently
            # erase the sparse positive-bag supervision.
            frame_loss = torch.stack(frame_terms).sum()
            interval_loss = interval_loss + float(self.frame_cfg.get('loss_weight', 0.1)) * frame_loss
        return interval_loss, logits, targets

    def _forward_batch(self, batch, epoch, training):
        if self.paradigm == 'block':
            waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
            waveforms = waveforms.to(self.device)
            bounds = bounds.to(self.device)
            labels = labels.to(self.device)
            valid = valid.to(self.device)
            lengths = lengths.to(self.device)
            if training and self.augmentor is not None:
                waveforms = self.augmentor.apply(waveforms, epoch, self.device)
            outputs = self.model(waveforms, lengths, interval_bounds=bounds, interval_valid=valid)
            effective_valid = outputs.get('interval_valid', valid)
            if not effective_valid.any():
                raise RuntimeError("Block batch contains no interval with valid frames")
            loss, logits, targets = self._compute_block_loss(
                outputs, labels, effective_valid, training=training
            )
            flat_speakers = []
            flat_ids = []
            for b in range(effective_valid.size(0)):
                for s in range(effective_valid.size(1)):
                    if bool(effective_valid[b, s]):
                        flat_speakers.append(speakers[b])
                        flat_ids.append(interval_ids[b][s])
            return loss, logits, targets, flat_speakers, flat_ids, None

        waveforms, labels, lengths, speakers, meta = batch
        waveforms = waveforms.to(self.device)
        labels = labels.to(self.device)
        lengths = lengths.to(self.device)
        if training and self.augmentor is not None:
            waveforms = self.augmentor.apply(waveforms, epoch, self.device)
        logits = self.model(waveforms, lengths)
        loss = self.loss_fn(logits, labels)
        return loss, logits, labels, list(speakers), list(meta), (waveforms, lengths)

    def train_epoch(self, epoch):
        # Rebuild deterministic epoch-specific samples before the loader iterates.
        if hasattr(self.train_loader.dataset, 'set_epoch'):
            self.train_loader.dataset.set_epoch(epoch)
        batch_sampler = getattr(self.train_loader, 'batch_sampler', None)
        if hasattr(batch_sampler, 'set_epoch'):
            batch_sampler.set_epoch(epoch)
        sampler = getattr(self.train_loader, 'sampler', None)
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        self.model.train()
        total_loss = 0
        all_logits = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for i, batch in enumerate(pbar):
            device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
            with torch.amp.autocast(device_type, enabled=self.use_amp, dtype=self.autocast_dtype):
                loss, logits, labels, speaker_ids, _meta, _waveforms = self._forward_batch(batch, epoch, training=True)
                
            loss = loss / self.config['training']['gradient_accumulation_steps']
            
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                if self.scaler:
                    scale_before = self.scaler.get_scale()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    skip_lr_sched = (self.scaler.get_scale() < scale_before)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                    self.optimizer.step()
                    skip_lr_sched = False
                
                self.optimizer.zero_grad()
                if not skip_lr_sched:
                    self.scheduler.step()
                
            total_loss += loss.item() * self.config['training']['gradient_accumulation_steps']
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            
            pbar.set_postfix({'loss': loss.item()})
            
        # Flush remaining gradients at the end of epoch to prevent leakage
        if (i + 1) % self.config['training']['gradient_accumulation_steps'] != 0:
            if self.scaler:
                scale_before = self.scaler.get_scale()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                skip_lr_sched = (self.scaler.get_scale() < scale_before)
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                self.optimizer.step()
                skip_lr_sched = False
            self.optimizer.zero_grad()
            if not skip_lr_sched:
                self.scheduler.step()
            
        metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_labels))
        # Log train PR-AUC and F1 as secondary
        return total_loss / len(self.train_loader), metrics

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        all_logits = []
        all_labels = []
        all_speaker_ids = []
        frame_proxy_values = []

        if not self.val_loader or len(self.val_loader) == 0:
            return 0.0, {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'speaker_metrics': {}, 'cm': [[0, 0], [0, 0]]}

        # Issue 8: Collect samples during validation to avoid duplicate traversal
        samples_to_log = {"TP": [], "FP": [], "FN": [], "TN": []}
        count_per_type = self.config['tensorboard'].get('audio_sample_count', 8) // 4
        if count_per_type < 1: count_per_type = 1
        
        interval_predictions = {}
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]"):
                device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
                with torch.amp.autocast(device_type, enabled=self.use_amp, dtype=self.autocast_dtype):
                    loss, logits, labels, speaker_ids, item_ids, waveforms = self._forward_batch(batch, epoch, training=False)

                total_loss += loss.item()
                if self._current_frame_proxy is not None:
                    frame_proxy_values.append(self._current_frame_proxy)
                if self.paradigm == 'block':
                    for logit, label, speaker, interval_id in zip(logits, labels, speaker_ids, item_ids):
                        value = logit.detach().cpu()
                        if interval_id not in interval_predictions:
                            interval_predictions[interval_id] = [value, label.detach().cpu(), speaker, 1]
                        else:
                            interval_predictions[interval_id][0] += value
                            interval_predictions[interval_id][3] += 1
                    continue

                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())
                all_speaker_ids.extend(speaker_ids)
                segment_waveforms, segment_lengths = waveforms
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                for i in range(segment_waveforms.size(0)):
                    gt = int(labels[i].item())
                    pr = int(preds[i].item())
                    cat = "TP" if gt==1 and pr==1 else "TN" if gt==0 and pr==0 else "FP" if gt==0 and pr==1 else "FN"
                    if len(samples_to_log[cat]) < count_per_type:
                        samples_to_log[cat].append({
                            'waveform': segment_waveforms[i, :segment_lengths[i]].detach().cpu(),
                            'prob': probs[i].item(),
                            'meta': item_ids[i]
                        })

        if self.paradigm == 'block':
            if not interval_predictions:
                raise RuntimeError("Block validation produced no valid intervals")
            all_logits = [(item[0] / item[3]).reshape(1) for item in interval_predictions.values()]
            all_labels = [item[1].reshape(1) for item in interval_predictions.values()]
            all_speaker_ids = [item[2] for item in interval_predictions.values()]
                
        logits_cat = torch.cat(all_logits)
        labels_cat = torch.cat(all_labels)
        metrics = compute_metrics(logits_cat, labels_cat)
        
        # Per-speaker metrics
        speaker_metrics = {}
        unique_speakers = sorted(list(set(all_speaker_ids)))
        for spk in unique_speakers:
            indices = [i for i, s in enumerate(all_speaker_ids) if s == spk]
            if indices:
                spk_logits = logits_cat[indices]
                spk_labels = labels_cat[indices]
                speaker_metrics[spk] = compute_metrics(spk_logits, spk_labels)
        
        metrics['speaker_metrics'] = speaker_metrics
        metrics['logits'] = logits_cat
        metrics['labels'] = labels_cat
        metrics['samples'] = samples_to_log
        if frame_proxy_values:
            metrics['frame_proxy_auroc'] = float(np.mean(frame_proxy_values))

        return total_loss / len(self.val_loader), metrics

    def run(self, start_epoch=1):
        for epoch in range(start_epoch, self.epochs + 1):
            train_loss, train_metrics = self.train_epoch(epoch)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('PR_AUC/train', train_metrics['auc_pr'], epoch)
            self.writer.add_scalar('F1/train', train_metrics['f1'], epoch)  # secondary
            
            # Learning Rates
            self.writer.add_scalar('LR/backbone', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar('LR/head', self.optimizer.param_groups[1]['lr'], epoch)

            has_val = self.val_loader is not None and len(self.val_loader) > 0
            
            if has_val:
                val_loss, val_metrics = self.validate(epoch)
                
                # Logging — PR-AUC is primary, F1 is secondary
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('PR_AUC/val', val_metrics['auc_pr'], epoch)  # primary
                self.writer.add_scalar('F1/val', val_metrics['f1'], epoch)           # secondary
                self.writer.add_scalar('Precision/val', val_metrics['precision'], epoch)
                self.writer.add_scalar('Recall/val', val_metrics['recall'], epoch)
                if 'frame_proxy_auroc' in val_metrics:
                    self.writer.add_scalar('Frame_Proxy_AUROC/val', val_metrics['frame_proxy_auroc'], epoch)

                # PR Curve
                if self.config['tensorboard'].get('log_pr_curve', True):
                    self.writer.add_pr_curve('PR_Curve/val', val_metrics['labels'], torch.sigmoid(val_metrics['logits']), epoch)
                
                # Per-speaker metrics
                for spk, spk_m in val_metrics['speaker_metrics'].items():
                    self.writer.add_scalar(f'Speaker_PR_AUC/{spk}', spk_m['auc_pr'], epoch)
                    self.writer.add_scalar(f'Speaker_F1/{spk}', spk_m['f1'], epoch)
                
                # Confusion Matrix
                self.log_confusion_matrix(val_metrics['cm'], epoch)
                
                # Audio Samples (Error Analysis)
                if self.config['tensorboard'].get('log_audio_samples', True):
                    self.log_audio_samples(epoch, val_metrics.get('samples', {}))
                
                # Early Stopping and Checkpointing — based on configured primary metric
                current_val = val_metrics[self.primary_metric]
                if current_val > self.best_primary_metric_val:
                    self.best_primary_metric_val = current_val
                    self.patience_counter = 0
                    # Cache raw outputs for post-training threshold grid search
                    self._best_val_logits = val_metrics['logits'].clone()
                    self._best_val_labels = val_metrics['labels'].clone()
                    self.save_checkpoint(epoch, val_metrics, is_best=True)
                else:
                    self.patience_counter += 1
                    # Save latest anyway even if not best, to support robust resume
                    self.save_checkpoint(epoch, val_metrics, is_best=False, only_latest=True)
                    
                patience = self.config['eval'].get('early_stopping_patience', 0)
                if patience > 0 and self.patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"({self.primary_metric} did not improve for {patience} epochs)"
                    )
                    break
            else:
                # No validation: just save the model every epoch
                self.save_checkpoint(epoch, train_metrics, is_best=False)
                print(f"Epoch {epoch} complete (No validation).")

        # ── Post-training: threshold grid search on best validation epoch ──
        if self._best_val_logits is not None:
            best_threshold, threshold_f1 = self.find_best_threshold(
                self._best_val_logits, self._best_val_labels
            )
            print(f"Best threshold (grid search): {best_threshold:.3f}  →  F1={threshold_f1:.4f}")
            # Patch it into the best checkpoint on disk
            self._update_checkpoint_threshold(best_threshold, threshold_f1)

    def log_confusion_matrix(self, cm, epoch):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Creaky', 'Non-Creaky'], yticklabels=['Creaky', 'Non-Creaky'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        from PIL import Image
        image = Image.open(buf).convert('RGB')
        image_np = np.array(image)
        self.writer.add_image('Confusion_Matrix', image_np, epoch, dataformats='HWC')

    def log_audio_samples(self, epoch, samples_dict):
        # Issue 8: No longer needs to iterate over val_loader
        for category, samples in samples_dict.items():
            for i, sample in enumerate(samples):
                wf = sample['waveform']
                prob = sample['prob']
                meta = sample['meta']
                
                # Tag: Samples/FP/File|Time|Phone|p=0.8
                tag = f"Samples/{category}/{meta}|p={prob:.3f}"
                self.writer.add_audio(tag, wf, epoch, sample_rate=16000)

    # ──────────────────────────────────────────────────────────────────────
    # Threshold grid search
    # ──────────────────────────────────────────────────────────────────────
    def find_best_threshold(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        thresholds: list[float] | None = None,
    ) -> tuple[float, float]:
        """Brute-force grid search over candidate thresholds.

        Returns the threshold that maximises F1 on the provided logits/labels,
        along with the corresponding F1 score.
        """
        import numpy as np
        from sklearn.metrics import f1_score

        if thresholds is None:
            # 201 evenly-spaced thresholds from 0.01 to 0.99
            thresholds = [round(t, 3) for t in np.linspace(0.01, 0.99, 201).tolist()]

        probs = torch.sigmoid(logits).cpu().float().numpy()
        labels_np = labels.cpu().numpy().astype(np.int32)

        best_t, best_f1 = 0.5, 0.0
        for t in thresholds:
            preds = (probs >= t).astype(np.int32)
            score = float(f1_score(labels_np, preds, average='binary', zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_t = t

        return best_t, best_f1

    def _update_checkpoint_threshold(
        self, best_threshold: float, threshold_f1: float
    ) -> None:
        """Load the best checkpoint from disk, inject threshold fields, and re-save."""
        val_speakers = self.config['data']['val_speakers']
        val_str = "_".join(val_speakers) if isinstance(val_speakers, list) else str(val_speakers)
        best_ckpt_path = self.ckpt_dir / "best_model.pt"
        if not best_ckpt_path.exists():
            # Try speaker-specific path that the LOOCV loop may have renamed it to
            candidate = self.ckpt_dir / f"best_model_{val_str}.pt"
            if candidate.exists():
                best_ckpt_path = candidate
            else:
                print("[WARNING] Could not locate best checkpoint to inject threshold — skipping.")
                return
        try:
            ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
            ckpt['best_threshold'] = best_threshold
            ckpt['threshold_f1'] = threshold_f1
            if 'metrics' in ckpt:
                ckpt['metrics']['best_threshold'] = best_threshold
                ckpt['metrics']['threshold_f1'] = threshold_f1
            torch.save(ckpt, best_ckpt_path)
            print(f"Threshold {best_threshold:.3f} saved to {best_ckpt_path}")
        except Exception as exc:
            print(f"[WARNING] Failed to update checkpoint with threshold: {exc}")

    # ──────────────────────────────────────────────────────────────────────
    # Checkpoint I/O
    # ──────────────────────────────────────────────────────────────────────
    def save_checkpoint(self, epoch, metrics, is_best=False, only_latest=False):
        save_mode = self.config['checkpoint']['save_mode']
        f1 = metrics['f1']
        
        if save_mode == 'delta':
            # Save all parameters that were being trained (requires_grad=True)
            state_dict = {
                name: param.detach().cpu().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
        else:
            state_dict = self.model.state_dict()
            
        # Clean up raw tensors/arrays to reduce checkpoint size
        metrics_to_save = {k: v for k, v in metrics.items() if k not in ('logits', 'labels', 'samples', 'cm', 'speaker_metrics')}
        
        auc_pr = metrics.get('auc_pr', 0.0)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'f1': f1,
            'auc_pr': auc_pr,
            # best_threshold / threshold_f1 are injected later by _update_checkpoint_threshold
            'metrics': metrics_to_save,
            'config': self.config,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        
        # For non-delta mode, we always want the optimizer to allow resuming from any checkpoint
        if save_mode != 'delta':
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Construct descriptive filename
        val_speakers = self.config['data']['val_speakers']
        val_str = "_".join(val_speakers) if isinstance(val_speakers, list) else str(val_speakers)
        
        if is_best:
            auc_pr_for_name = metrics.get('auc_pr', 0.0)
            name = f"best_ep{epoch}_prauc{auc_pr_for_name:.4f}_f1{f1:.4f}_{val_str}.pt"
        else:
            name = f"model_ep{epoch}_{val_str}.pt"
            
        if not only_latest:
            path = self.ckpt_dir / name
            torch.save(checkpoint, path)
        else:
            path = self.ckpt_dir / name # Just for the print message below
        
        # Also maintain a fixed name 'best_model.pt' for script automation
        if is_best:
            import shutil
            shutil.copy(path, self.ckpt_dir / "best_model.pt")
            
            # Cleanup old best checkpoints (keep the N best by PR-AUC)
            keep_count = self.config['checkpoint'].get('keep_best_count', 3)
            best_files = list(self.ckpt_dir.glob(f"best_ep*_prauc*_{val_str}.pt"))
            if len(best_files) > keep_count:
                def extract_prauc(p):
                    try: return float(p.name.split('_prauc')[1].split('_')[0])
                    except: return 0.0
                best_files.sort(key=extract_prauc)
                for f in best_files[:-keep_count]:
                    try: f.unlink()
                    except: pass
        
        # Save a 'latest.pt' for the current fold to support resuming - ALWAYS with full state
        if 'optimizer_state_dict' not in checkpoint:
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        latest_path = self.ckpt_dir / f"latest_{val_str}.pt"
        torch.save(checkpoint, latest_path)
            
        if only_latest:
            print(f"Updated latest checkpoint: {latest_path} (mode: {save_mode})")
        else:
            print(f"Checkpoint saved to {path} (mode: {save_mode})")
        return path

    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path} for resuming...")
        # We need weights_only=False because of optimizer/scheduler states and numpy in metrics
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        # Load model weights (handle delta or full)
        result = self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if result.missing_keys:
            print(f"[INFO] Missing keys (will use initialization): {result.missing_keys}")
        
        # Load optimizer and scheduler
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        rng_state = checkpoint.get('rng_state')
        if rng_state:
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if torch.cuda.is_available() and rng_state.get('cuda') is not None:
                torch.cuda.set_rng_state_all(rng_state['cuda'])

        return checkpoint['epoch']
