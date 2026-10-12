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
import logging
import math
import pandas as pd
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from ..model import losses as losses_mod
get_loss_fn = losses_mod.get_loss_fn

from . import metrics as metrics_mod
compute_metrics = metrics_mod.compute_metrics

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device, augmentor=None, heldout_df=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        if self.val_loader is None or len(self.val_loader) == 0:
            raise RuntimeError(
                "[TRAINER CONFIG ERROR] Trainer requires a non-empty validation DataLoader. "
                "Cannot train blindly without validation."
            )
        self.config = config
        self.device = device
        self.augmentor = augmentor
        self.paradigm = config['training'].get('paradigm', 'segment')
        self.frame_cfg = config['training'].get('block', {}).get('frame_head', {})
        self._current_frame_proxy = None
        self._freeze_disabled_task_params()

        self.epochs = config['training']['epochs']
        self.quiet = config['training'].get('quiet', False)
        grad_acc = config['training'].get('gradient_accumulation_steps', 1)
        import math
        steps_per_epoch = math.ceil(len(self.train_loader) / grad_acc)
        self.total_steps = self.epochs * steps_per_epoch
        from .weak_supervision import create_weak_loader, dataset_rows, epoch_update_counts, training_provenance
        self.weak_cfg = config['training'].get('weak_supervision', {})
        manual_train = dataset_rows(train_loader.dataset)
        manual_val = dataset_rows(val_loader.dataset)
        if heldout_df is not None:
            manual_val = pd.concat([manual_val, heldout_df], ignore_index=True) if manual_val is not None else heldout_df
        self.weak_loader, self.weak_fingerprint = create_weak_loader(config, model, manual_train, manual_val)
        self.training_provenance = training_provenance(
            manual_train, self.weak_loader.dataset.df if self.weak_loader is not None else None
        )
        if self.weak_loader is not None:
            self.total_steps = sum(epoch_update_counts(train_loader, self.weak_loader, self.epochs,
                                                      int(self.weak_cfg.get('start_epoch', 2)), grad_acc))
        
        # Optimizer with differential learning rate
        self.optimizer = self.create_optimizer()
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Scheduler
        self.scheduler = self.create_scheduler()
        
        # Primary task drives top-level metrics, early stopping and threshold search.
        # It must track train_heads gating: a disabled head must never arbitrate
        # early stopping, or best-checkpoint selection degenerates into noise.
        self.tasks = getattr(model, "tasks", ["creaky"])
        self.primary_task = self._resolve_primary_task(config, self.tasks)

        # Dataset construction already materializes epoch-1 generated windows, so
        # weighted BCE reflects the effective initial training distribution.
        # Per-task class statistics: pos_weight of one task must never bleed
        # into another (creaky and breathy have very different imbalance).
        self.loss_fns = {task: get_loss_fn(config, pos_weight=self._task_pos_weight(train_loader, task, device))
                         for task in self.tasks}
        # Frame-level MIL BCE per task with the same pos_weight proxy (frame-level
        # labels do not exist by design; interval imbalance is the best available prior).
        self.frame_loss_fns = {
            task: torch.nn.BCEWithLogitsLoss(
                pos_weight=self._task_pos_weight(train_loader, task, device)
            )
            for task in self.tasks
        }
        # Backwards-compatible alias for the primary task's loss function.
        self.loss_fn = self.loss_fns[self.primary_task]
        
        # Mixed precision setup
        self.mixed_prec = config['training'].get('mixed_precision', 'no')
        self.use_amp = self.mixed_prec != 'no' and 'cuda' in str(device)
        self.scaler = torch.amp.GradScaler('cuda') if (self.use_amp and self.mixed_prec == 'fp16') else None
        self.autocast_dtype = torch.bfloat16 if self.mixed_prec == 'bf16' else torch.float16
        
        # TensorBoard and log dir
        self.log_dir = Path(config['tensorboard']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))
        
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

        # Epoch-by-epoch metric history
        self.history: list[dict[str, Any]] = []

    @staticmethod
    def _resolve_primary_task(config, tasks) -> str:
        """Pick the task that arbitrates early stopping / top-level metrics.

        Priority: explicit ``eval.primary_task`` (must be an active task) >
        creaky if trainable > first trainable task > first task. Disabled
        (train_heads=false) heads are never eligible.
        """
        train_heads_cfg = config.get('training', {}).get('train_heads', {})
        configured_primary = config.get('eval', {}).get('primary_task')
        if configured_primary is not None:
            if configured_primary not in tasks:
                raise ValueError(
                    f"eval.primary_task '{configured_primary}' is not an active task "
                    f"{list(tasks)}"
                )
            return configured_primary
        trainable_tasks = [t for t in tasks if train_heads_cfg.get(t, True)]
        if "creaky" in trainable_tasks:
            return "creaky"
        if trainable_tasks:
            return trainable_tasks[0]
        return tasks[0]

    def _task_pos_weight(self, train_loader, task: str, device) -> torch.Tensor:
        """Count positives/negatives of one task's labels and return n_neg/n_pos."""
        if self.paradigm == 'block':
            interval_rows = [row for block in train_loader.dataset.blocks for row in block['intervals']]
            vals = []
            for row in interval_rows:
                val = row.get(f'label_{task}', row.get('label', float('nan')))
                try:
                    fval = float(val)
                    if not math.isnan(fval):
                        vals.append(fval)
                except (ValueError, TypeError):
                    pass
        else:
            train_df = train_loader.dataset.df
            col = f'label_{task}' if f'label_{task}' in train_df.columns else 'label'
            valid_series = pd.to_numeric(train_df[col], errors='coerce').dropna()
            vals = valid_series.tolist()
        n_pos = sum(1 for v in vals if v > 0.5)
        n_neg = len(vals) - n_pos
        if n_pos == 0 or n_neg == 0:
            logger.warning(
                f"Task '{task}' has a single-class training distribution "
                f"(pos={n_pos}, neg={n_neg}); pos_weight falls back to 1.0."
            )
        return torch.tensor([max(n_neg, 1) / max(n_pos, 1)]).to(device)

    def rebuild_writer(self) -> None:
        """Replace the TensorBoard writer with a fresh instance.

        Used by the watchdog recovery path: the failed run may have terminated
        mid-epoch, and writing to a closed or half-written event file would
        either raise or silently drop scalars for the rest of training.
        """
        try:
            self.writer.close()
        except Exception:
            pass
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def _freeze_disabled_task_params(self) -> None:
        """Freeze every per-task parameter of train_heads-disabled tasks.

        Loss gating alone leaves the disabled task's scalar mix / attention /
        head parameters with requires_grad=True: they never receive a real
        gradient, but the dummy-grad hook attaches zero-grad tensors to them,
        and AdamW's decoupled weight decay then applies to those tensors and
        slowly shrinks the disabled head. Freezing removes them from the
        optimizer, gradient clipping and the dummy-grad hook in one place.
        """
        train_heads_cfg = self.config.get('training', {}).get('train_heads', {})
        for task, trainable in train_heads_cfg.items():
            if trainable:
                continue
            for container_name in ("heads", "attention_heads", "frame_heads", "frame_mil_fusions"):
                container = getattr(self.model, container_name, None)
                if container is not None and task in container:
                    for param in container[task].parameters():
                        param.requires_grad_(False)
            scalar_mix = getattr(self.model, "scalar_mix_weights", None)
            if scalar_mix is not None and task in scalar_mix:
                scalar_mix[task].requires_grad_(False)

    def create_optimizer(self):
        # Issue 4: AdamW weight decay optimization
        weight_decay = self.config['training'].get('weight_decay', 0.01)
        no_decay = ["bias", "LayerNorm.weight"]
        
        is_backbone = lambda n: n.startswith("wavlm.") or n.startswith("backbone.")
        is_scalar_mix = lambda n: "scalar_mix" in n
        scalar_mix_lr = float(self.config['training'].get('scalar_mix_lr', 1.0e-3))

        optimizer_grouped_parameters = [
            # 1. Backbone (WavLM / generic backbone) - No weight decay to protect pre-trained features
            {
                "name": "backbone",
                "params": [p for n, p in self.model.named_parameters() if is_backbone(n) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": self.config['training']['backbone_lr']
            },
            # 2. Scalar Mix - No weight decay to prevent entropy collapse, dedicated moderate LR (default 1e-3)
            {
                "name": "scalar_mix",
                "params": [p for n, p in self.model.named_parameters() if is_scalar_mix(n) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": scalar_mix_lr
            },
            # 3. Head (Weights) - Normal weight decay
            {
                "name": "head",
                "params": [p for n, p in self.model.named_parameters() if not is_backbone(n) and not is_scalar_mix(n) and p.requires_grad and not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
                "lr": self.config['training']['head_lr']
            },
            # 4. Head (Bias/LayerNorm) - No weight decay
            {
                "name": "head_no_decay",
                "params": [p for n, p in self.model.named_parameters() if not is_backbone(n) and not is_scalar_mix(n) and p.requires_grad and any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": self.config['training']['head_lr']
            },
        ]
        # Filter out empty parameter groups (e.g. when backbone is 100% frozen)
        optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
        
        optimizer_type = self.config['training'].get('optimizer', 'adamw_fp32')
        if optimizer_type == 'adamw_8bit':
            try:
                import bitsandbytes as bnb
                print("[INFO] Using 8-bit AdamW optimizer (bitsandbytes)")
                return bnb.optim.AdamW8bit(optimizer_grouped_parameters)
            except (ImportError, ModuleNotFoundError):
                print("[WARNING] 'bitsandbytes' not found or failed to load. Falling back to native torch.optim.AdamW.")
                return optim.AdamW(optimizer_grouped_parameters)
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

    CANONICAL_TASKS = ("creaky", "breathy")

    def _get_task_target(self, labels: torch.Tensor, task: str) -> torch.Tensor:
        tasks = getattr(self.model, "tasks", ["creaky"])
        # If labels tensor has multi-task dimension matching CANONICAL_TASKS
        if labels.ndim > 1 and labels.shape[-1] == len(self.CANONICAL_TASKS):
            if task in self.CANONICAL_TASKS:
                return labels[..., self.CANONICAL_TASKS.index(task)]
        if labels.ndim > 1 and labels.shape[-1] == len(tasks):
            if task in tasks:
                return labels[..., tasks.index(task)]
        if labels.ndim > 1 and labels.shape[-1] == 1:
            return labels.squeeze(-1)
        return labels

    def _compute_block_loss(self, outputs, labels, slot_valid, training=True):
        self._current_frame_proxy = None
        slot_valid = outputs.get('interval_valid', slot_valid)
        train_heads_cfg = self.config.get('training', {}).get('train_heads', {})
        tasks = getattr(self.model, "tasks", ["creaky"])

        raw_interval_logits = outputs['interval_logits']
        if not isinstance(raw_interval_logits, dict):
            raw_interval_logits = {tasks[0]: raw_interval_logits}

        total_loss = torch.tensor(0.0, device=slot_valid.device)
        logits_by_task = {}
        targets_by_task = {}
        current_losses = {}

        for task in tasks:
            task_logits = raw_interval_logits[task][slot_valid]
            task_targets = self._get_task_target(labels, task)[slot_valid]
            logits_by_task[task] = task_logits
            targets_by_task[task] = task_targets

            valid_mask = ~torch.isnan(task_targets)
            if valid_mask.any():
                loss = self.loss_fns[task](task_logits[valid_mask], task_targets[valid_mask])
                current_losses[task] = loss.detach().item()
                is_trainable = train_heads_cfg.get(task, True)
                if not (training and not is_trainable):
                    total_loss = total_loss + loss

        if not total_loss.requires_grad and any(p.requires_grad for p in self.model.parameters()):
            dummy_grad = sum(p.sum() * 0.0 for p in self.model.parameters() if p.requires_grad)
            total_loss = total_loss + dummy_grad

        if self.frame_cfg.get('enabled', False) and 'frame_logits' in outputs:
            from ..model.losses import pool_frame_logits
            raw_frame_logits = outputs['frame_logits']
            interval_masks = outputs['interval_masks']
            frame_masks = interval_masks[slot_valid]
            mode = self.frame_cfg.get('mil_mode', 'lse')
            lw_cfg = self.frame_cfg.get('loss_weight', 0.1)

            for task in tasks:
                if isinstance(lw_cfg, dict):
                    loss_weight = float(lw_cfg.get(task, 0.1))
                else:
                    loss_weight = float(lw_cfg)

                if isinstance(raw_frame_logits, dict):
                    if task not in raw_frame_logits:
                        continue
                    t_frame_logits = raw_frame_logits[task]
                else:
                    if task != ("creaky" if "creaky" in tasks else tasks[0]):
                        continue
                    t_frame_logits = raw_frame_logits

                expanded = t_frame_logits.unsqueeze(1).expand_as(interval_masks)[slot_valid]
                if mode == 'lse_mean':
                    lse = pool_frame_logits(
                        expanded, frame_masks, mode='lse',
                        temperature=self.frame_cfg.get('temperature', 1.0),
                    )
                    mean = (expanded.masked_fill(~frame_masks, 0.0).sum(-1) /
                            frame_masks.sum(-1).clamp_min(1))
                    fusion_layer = (
                        self.model.frame_mil_fusions[task]
                        if hasattr(self.model, "frame_mil_fusions") and task in self.model.frame_mil_fusions
                        else getattr(self.model, "frame_mil_fusion", None)
                    )
                    mil_logits = fusion_layer(
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

                task_targets = targets_by_task[task]
                positive = task_targets > 0.5
                frame_terms = []
                if positive.any():
                    frame_terms.append(self.frame_loss_fns[task](
                        mil_logits[positive], torch.ones_like(mil_logits[positive])
                    ))
                negative = (~positive) & (~torch.isnan(task_targets))
                if negative.any():
                    negative_logits = expanded[negative][frame_masks[negative]]
                    frame_terms.append(self.frame_loss_fns[task](
                        negative_logits, torch.zeros_like(negative_logits)
                    ))
                if frame_terms:
                    task_frame_loss = torch.stack(frame_terms).sum()
                    w_task_frame_loss = loss_weight * task_frame_loss
                    is_trainable = train_heads_cfg.get(task, True)
                    if not (training and not is_trainable):
                        total_loss = total_loss + w_task_frame_loss
                    current_losses[f'{task}_frame'] = w_task_frame_loss.detach().item()

        self._current_losses = current_losses
        return total_loss, logits_by_task, targets_by_task

    def _forward_batch(self, batch, epoch, training):
        if self.paradigm == 'block':
            waveforms, bounds, labels, valid, lengths, speakers, block_ids, interval_ids = batch
            waveforms = waveforms.to(self.device, non_blocking=True)
            bounds = bounds.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            valid = valid.to(self.device, non_blocking=True)
            lengths = lengths.to(self.device, non_blocking=True)
            if training and self.augmentor is not None:
                waveforms = self.augmentor.apply(waveforms, epoch, self.device)
            outputs = self.model(waveforms, lengths, interval_bounds=bounds, interval_valid=valid)
            effective_valid = outputs.get('interval_valid', valid)
            if not effective_valid.any():
                raise RuntimeError("Block batch contains no interval with valid frames")
            loss, logits_by_task, targets_by_task = self._compute_block_loss(
                outputs, labels, effective_valid, training=training
            )
            if not training:
                valid_cpu = effective_valid.detach().cpu().numpy()
                flat_speakers = []
                flat_ids = []
                for b in range(valid_cpu.shape[0]):
                    for s in range(valid_cpu.shape[1]):
                        if valid_cpu[b, s]:
                            flat_speakers.append(speakers[b])
                            flat_ids.append(interval_ids[b][s])
                return loss, logits_by_task, targets_by_task, flat_speakers, flat_ids, None
            return loss, logits_by_task, targets_by_task, None, None, None

        # Segment mode
        waveforms, labels, lengths, speakers, meta = batch
        waveforms = waveforms.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        lengths = lengths.to(self.device, non_blocking=True)
        if training and self.augmentor is not None:
            waveforms = self.augmentor.apply(waveforms, epoch, self.device)
        logits_dict = self.model(waveforms, lengths)
        if not isinstance(logits_dict, dict):
            tasks = getattr(self.model, "tasks", ["creaky"])
            logits_dict = {tasks[0]: logits_dict}

        train_heads_cfg = self.config.get('training', {}).get('train_heads', {})
        tasks = getattr(self.model, "tasks", ["creaky"])
        total_loss = torch.tensor(0.0, device=self.device)
        logits_by_task = {}
        targets_by_task = {}
        current_losses = {}

        for task in tasks:
            task_logits = logits_dict[task]
            task_targets = self._get_task_target(labels, task)
            logits_by_task[task] = task_logits
            targets_by_task[task] = task_targets

            valid_mask = ~torch.isnan(task_targets)
            if valid_mask.any():
                loss = self.loss_fns[task](task_logits[valid_mask], task_targets[valid_mask])
                current_losses[task] = loss.detach().item()
                is_trainable = train_heads_cfg.get(task, True)
                if not (training and not is_trainable):
                    total_loss = total_loss + loss

        if not total_loss.requires_grad and any(p.requires_grad for p in self.model.parameters()):
            dummy_grad = sum(p.sum() * 0.0 for p in self.model.parameters() if p.requires_grad)
            total_loss = total_loss + dummy_grad

        self._current_losses = current_losses
        return total_loss, logits_by_task, targets_by_task, list(speakers), list(meta), (waveforms, lengths)

    @staticmethod
    def _merge_interval_prediction(
        store: dict, interval_id, logit: torch.Tensor, label: torch.Tensor
    ) -> None:
        """Accumulate one (block, slot) prediction into the per-interval store."""
        logit_cpu = logit if logit.device.type == "cpu" else logit.detach().cpu()
        label_cpu = label if label.device.type == "cpu" else label.detach().cpu()
        entry = store.get(interval_id)
        if entry is None:
            store[interval_id] = [logit_cpu, label_cpu, 1]
            return
        entry[0] += logit_cpu
        entry[2] += 1
        # A first-seen NaN label (interval unlabeled in an earlier block) must
        # not mask a valid label observed in a later block of the same interval.
        if torch.isnan(entry[1]) and not torch.isnan(label_cpu):
            entry[1] = label_cpu

    def _forward_weak_batch(self, batch):
        from ..data.utterance_bag import utterance_logits
        tensors = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = self.model(tensors['waveforms'], tensors['lengths'], return_frames=True)
        logits = utterance_logits(outputs, tensors, temperature=float(self.frame_cfg.get('temperature', 1.0)))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits.float(), torch.ones_like(logits).float())
        return float(self.weak_cfg.get('loss_weight', .05)) * loss, len(logits)

    def train_epoch(self, epoch):
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
        tasks = getattr(self.model, "tasks", ["creaky"])
        task_loss_sums = {task: 0.0 for task in tasks}
        task_loss_counts = {task: 0 for task in tasks}
        frame_loss_sums = {task: 0.0 for task in tasks}
        frame_loss_counts = {task: 0 for task in tasks}
        all_logits = {task: [] for task in tasks}
        all_labels = {task: [] for task in tasks}

        from .weak_supervision import interleave_batches
        weak_loader = getattr(self, 'weak_loader', None)
        if weak_loader is not None and epoch < int(self.weak_cfg.get('start_epoch', 2)):
            weak_loader = None
        weak_count, weak_population = 0, 0
        if weak_loader is not None:
            weak_loader.batch_sampler.set_epoch(epoch)
            weak_schedule = weak_loader.batch_sampler.batches()
            weak_count = len(weak_schedule)
            weak_population = sum(len(batch) for batch in weak_schedule)
        batch_count = len(self.train_loader) + weak_count
        weak_loss_sum, weak_examples, weak_batches = 0.0, 0, 0
        stream = interleave_batches(self.train_loader, weak_loader)
        pbar = stream if self.quiet else tqdm(stream, total=batch_count, desc=f"Epoch {epoch} [Train]")
        for i, (source, batch) in enumerate(pbar):
            device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
            with torch.amp.autocast(device_type, enabled=self.use_amp, dtype=self.autocast_dtype):
                if source == 'weak':
                    loss, examples = self._forward_weak_batch(batch)
                    weak_loss_sum += loss.detach().item() * examples
                    loss = loss * (examples * weak_count / weak_population)
                    weak_examples += examples
                    weak_batches += 1
                    logits_dict, labels_dict = {}, {}
                    self._current_losses = {}
                else:
                    loss, logits_dict, labels_dict, speaker_ids, _meta, _waveforms = self._forward_batch(batch, epoch, training=True)

            accumulation = self.config['training']['gradient_accumulation_steps']
            group_size = accumulation
            if getattr(self, 'weak_loader', None) is not None:
                group_start = (i // accumulation) * accumulation
                group_size = min(accumulation, batch_count - group_start)
            loss = loss / group_size

            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                if self.scaler:
                    scale_before = self.scaler.get_scale()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config['training']['grad_clip'])
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    skip_lr_sched = (self.scaler.get_scale() < scale_before)
                else:
                    torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config['training']['grad_clip'])
                    self.optimizer.step()
                    skip_lr_sched = False

                self.optimizer.zero_grad()
                if not skip_lr_sched:
                    self.scheduler.step()

            total_loss += loss.item() * group_size
            if hasattr(self, "_current_losses") and self._current_losses:
                for task in tasks:
                    if task in self._current_losses:
                        task_loss_sums[task] += self._current_losses[task]
                        task_loss_counts[task] += 1
                for task in tasks:
                    if f'{task}_frame' in self._current_losses:
                        frame_loss_sums[task] += self._current_losses[f'{task}_frame']
                        frame_loss_counts[task] += 1

            for task in tasks:
                if task in logits_dict:
                    all_logits[task].append(logits_dict[task].detach())
                    all_labels[task].append(labels_dict[task].detach())

            if not self.quiet:
                pbar.set_postfix({'loss': loss.item()})
            elif (i + 1) % 50 == 0 or (i + 1) == batch_count:
                print(f"  [Epoch {epoch:02d} Train] Step {i+1}/{batch_count} - Batch Loss: {loss.item():.4f}", flush=True)

        # Flush remaining gradients
        if (i + 1) % self.config['training']['gradient_accumulation_steps'] != 0:
            if self.scaler:
                scale_before = self.scaler.get_scale()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config['training']['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.trainable_params, self.config['training']['grad_clip'])
                self.optimizer.step()
            self.optimizer.zero_grad()
            if not self.scaler or self.scaler.get_scale() >= scale_before:
                self.scheduler.step()

        primary_task = self.primary_task
        metrics = {}
        if getattr(self, 'weak_loader', None) is not None:
            metrics['loss_weak_breathy'] = weak_loss_sum / max(weak_examples, 1)
            metrics['weak_utterances'] = weak_examples
            metrics['manual_batches'] = len(self.train_loader)
            metrics['weak_batches'] = weak_batches
        for task in tasks:
            if task_loss_counts[task] > 0:
                metrics[f"loss_{task}"] = task_loss_sums[task] / task_loss_counts[task]
        for task in tasks:
            if frame_loss_counts[task] > 0:
                metrics[f"loss_{task}_frame"] = frame_loss_sums[task] / frame_loss_counts[task]
        if "loss_creaky_frame" in metrics:
            metrics["loss_frame"] = metrics["loss_creaky_frame"]

        for task in tasks:
            if all_logits[task]:
                t_logits = torch.cat(all_logits[task]).cpu()
                t_labels = torch.cat(all_labels[task]).cpu()
                valid_m = ~torch.isnan(t_labels)
                if valid_m.sum() > 0:
                    task_m = compute_metrics(t_logits[valid_m], t_labels[valid_m])
                    for k, v in task_m.items():
                        metrics[f"{k}_{task}"] = v
                    if task == primary_task:
                        metrics.update(task_m)
                else:
                    metrics[f"auc_pr_{task}"] = 0.0
                    metrics[f"f1_{task}"] = 0.0
                    if task == primary_task:
                        metrics["auc_pr"] = 0.0
                        metrics["f1"] = 0.0

        return total_loss / batch_count, metrics

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        tasks = getattr(self.model, "tasks", ["creaky"])
        val_task_loss_sums = {task: 0.0 for task in tasks}
        val_task_loss_counts = {task: 0 for task in tasks}
        val_frame_loss_sums = {task: 0.0 for task in tasks}
        val_frame_loss_counts = {task: 0 for task in tasks}
        all_logits = {task: [] for task in tasks}
        all_labels = {task: [] for task in tasks}
        frame_proxy_values = []

        if not self.val_loader or len(self.val_loader) == 0:
            raise RuntimeError("val_loader is empty or None in Trainer.validate(). Cannot evaluate without validation data.")

        samples_to_log = {"TP": [], "FP": [], "FN": [], "TN": []}
        count_per_type = self.config['tensorboard'].get('audio_sample_count', 8) // 4
        if count_per_type < 1: count_per_type = 1

        interval_predictions = {task: {} for task in tasks}
        val_iter = self.val_loader if self.quiet else tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        with torch.inference_mode():
            for i, batch in enumerate(val_iter):
                device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'
                with torch.amp.autocast(device_type, enabled=self.use_amp, dtype=self.autocast_dtype):
                    loss, logits_dict, labels_dict, speaker_ids, item_ids, waveforms = self._forward_batch(batch, epoch, training=False)

                total_loss += loss.item()
                if hasattr(self, "_current_losses") and self._current_losses:
                    for task in tasks:
                        if task in self._current_losses:
                            val_task_loss_sums[task] += self._current_losses[task]
                            val_task_loss_counts[task] += 1
                    for task in tasks:
                        if f'{task}_frame' in self._current_losses:
                            val_frame_loss_sums[task] += self._current_losses[f'{task}_frame']
                            val_frame_loss_counts[task] += 1

                if self.quiet and ((i + 1) % 50 == 0 or (i + 1) == len(self.val_loader)):
                    print(f"  [Epoch {epoch:02d} Val] Step {i+1}/{len(self.val_loader)}", flush=True)

                if self._current_frame_proxy is not None:
                    frame_proxy_values.append(self._current_frame_proxy)

                if self.paradigm == 'block':
                    for task in tasks:
                        t_logits = logits_dict[task].detach().cpu()
                        t_labels = labels_dict[task].detach().cpu()
                        for logit, label, interval_id in zip(t_logits, t_labels, item_ids):
                            self._merge_interval_prediction(
                                interval_predictions[task], interval_id, logit, label
                            )
                    continue

                for task in tasks:
                    all_logits[task].append(logits_dict[task].detach().cpu())
                    all_labels[task].append(labels_dict[task].detach().cpu())

        if self.paradigm == 'block':
            for task in tasks:
                t_dict = interval_predictions[task]
                if t_dict:
                    all_logits[task] = [(item[0] / item[2]).reshape(1) for item in t_dict.values()]
                    all_labels[task] = [item[1].reshape(1) for item in t_dict.values()]

        primary_task = self.primary_task
        metrics = {}
        for task in tasks:
            if val_task_loss_counts[task] > 0:
                metrics[f"loss_{task}"] = val_task_loss_sums[task] / val_task_loss_counts[task]
        for task in tasks:
            if val_frame_loss_counts[task] > 0:
                metrics[f"loss_{task}_frame"] = val_frame_loss_sums[task] / val_frame_loss_counts[task]
        if "loss_creaky_frame" in metrics:
            metrics["loss_frame"] = metrics["loss_creaky_frame"]

        for task in tasks:
            if all_logits[task]:
                t_logits_cat = torch.cat(all_logits[task])
                t_labels_cat = torch.cat(all_labels[task])
                valid_m = ~torch.isnan(t_labels_cat)
                if valid_m.sum() > 0:
                    task_m = compute_metrics(t_logits_cat[valid_m], t_labels_cat[valid_m])
                    for k, v in task_m.items():
                        metrics[f"{k}_{task}"] = v
                    metrics[f'logits_{task}'] = t_logits_cat[valid_m]
                    metrics[f'labels_{task}'] = t_labels_cat[valid_m]
                    if task == primary_task:
                        metrics.update(task_m)
                        metrics['logits'] = t_logits_cat[valid_m]
                        metrics['labels'] = t_labels_cat[valid_m]
                else:
                    metrics[f"auc_pr_{task}"] = 0.0
                    metrics[f"f1_{task}"] = 0.0
                    metrics[f'logits_{task}'] = torch.zeros(1)
                    metrics[f'labels_{task}'] = torch.zeros(1)
                    if task == primary_task:
                        metrics["auc_pr"] = 0.0
                        metrics["f1"] = 0.0
                        metrics['logits'] = torch.zeros(1)
                        metrics['labels'] = torch.zeros(1)

        metrics['samples'] = samples_to_log
        if frame_proxy_values:
            metrics['frame_proxy_auroc'] = float(np.mean(frame_proxy_values))

        return total_loss / len(self.val_loader), metrics

    def save_history(self):
        """Save detailed epoch-by-epoch history to JSON and CSV atomically in both log_dir and ckpt_dir."""
        if not self.history:
            return
        import json
        import csv
        
        target_dirs = {self.log_dir, self.ckpt_dir}
        for d in target_dirs:
            try:
                d.mkdir(parents=True, exist_ok=True)
                json_path = d / "epoch_history.json"
                tmp_json = json_path.with_suffix(".json.tmp")
                with open(tmp_json, "w", encoding="utf-8") as f:
                    json.dump(self.history, f, indent=2)
                if tmp_json.exists():
                    os.replace(tmp_json, json_path)

                csv_path = d / "epoch_history.csv"
                tmp_csv = csv_path.with_suffix(".csv.tmp")
                fieldnames = []
                for record in self.history:
                    for k in record.keys():
                        if k not in fieldnames:
                            fieldnames.append(k)
                with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.history)
                if tmp_csv.exists():
                    os.replace(tmp_csv, csv_path)
            except Exception as e:
                logger.warning(f"Failed to save epoch history to {d}: {e}")

    def run(self, start_epoch=1):
        import time
        tasks = getattr(self.model, "tasks", ["creaky"])
        primary_task = self.primary_task
        for epoch in range(start_epoch, self.epochs + 1):
            epoch_start_time = time.time()
            train_loss, train_metrics = self.train_epoch(epoch)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('total_loss/train', train_loss, epoch)
            for key in ('loss_weak_breathy', 'weak_utterances', 'manual_batches', 'weak_batches'):
                if key in train_metrics:
                    self.writer.add_scalar(f'Weak/{key}', train_metrics[key], epoch)
            if len(tasks) == 1:
                self.writer.add_scalar('PR_AUC/train', train_metrics.get('auc_pr', 0.0), epoch)
                self.writer.add_scalar('F1/train', train_metrics.get('f1', 0.0), epoch)
            for task in tasks:
                if f'auc_pr_{task}' in train_metrics:
                    self.writer.add_scalar(f'PR_AUC_{task}/train', train_metrics[f'auc_pr_{task}'], epoch)
                if f'f1_{task}' in train_metrics:
                    self.writer.add_scalar(f'F1_{task}/train', train_metrics[f'f1_{task}'], epoch)
                if f'loss_{task}' in train_metrics:
                    self.writer.add_scalar(f'{task}_loss/train', train_metrics[f'loss_{task}'], epoch)
                    self.writer.add_scalar(f'Loss/{task}_train', train_metrics[f'loss_{task}'], epoch)

            for task in tasks:
                if f'loss_{task}_frame' in train_metrics:
                    self.writer.add_scalar(f'{task}_frame_loss/train', train_metrics[f'loss_{task}_frame'], epoch)
                    self.writer.add_scalar(f'Loss/{task}_frame_train', train_metrics[f'loss_{task}_frame'], epoch)
                if f'loss_{task}' in train_metrics:
                    t_train_tot = train_metrics[f'loss_{task}'] + train_metrics.get(f'loss_{task}_frame', 0.0)
                    self.writer.add_scalar(f'{task}_total_loss/train', t_train_tot, epoch)
                    self.writer.add_scalar(f'Loss/{task}_total_train', t_train_tot, epoch)
            
            # Learning Rates
            for g in self.optimizer.param_groups:
                g_name = g.get('name', 'group')
                self.writer.add_scalar(f'LR/{g_name}', g['lr'], epoch)

            val_loss, val_metrics = self.validate(epoch)

            # Scalar Mix layer distribution logging
            if getattr(self.model, "use_scalar_mix", False) and hasattr(self.model, "scalar_mix_weights"):
                with torch.no_grad():
                    for task in tasks:
                        if task in self.model.scalar_mix_weights:
                            w = torch.softmax(self.model.scalar_mix_weights[task].detach().cpu(), dim=0)
                            w_dict = {f"L{i}": float(v) for i, v in enumerate(w)}
                            if self.writer:
                                self.writer.add_scalars(f"ScalarMix/{task}", w_dict, epoch)
                            top3_idx = torch.topk(w, k=min(3, len(w))).indices.tolist()
                            logger.info(f"[{task}] Scalar Mix: " + ", ".join(f"L{i}:{w[i]:.3f}" for i in range(len(w))) + f" (Top3: {[f'L{i}' for i in top3_idx]})")
            
            extra_info = []
            for task in tasks:
                t_tr_l = train_metrics.get(f'loss_{task}')
                t_val_l = val_metrics.get(f'loss_{task}')
                t_val_pr = val_metrics.get(f'auc_pr_{task}')
                parts = []
                if t_tr_l is not None and t_val_l is not None:
                    parts.append(f"L: {t_tr_l:.4f}/{t_val_l:.4f}")
                if t_val_pr is not None:
                    parts.append(f"PR: {t_val_pr:.4f}")
                if parts:
                    extra_info.append(f"{task.capitalize()} ({', '.join(parts)})")
            extra_str = f" | {', '.join(extra_info)}" if extra_info else ""

            print(
                f"[Epoch {epoch:02d}/{self.epochs:02d}] "
                f"Train Loss: {train_loss:.4f}, PR-AUC: {train_metrics.get('auc_pr', 0.0):.4f}, F1: {train_metrics.get('f1', 0.0):.4f} | "
                f"Val Loss: {val_loss:.4f}, PR-AUC: {val_metrics.get('auc_pr', 0.0):.4f}, F1: {val_metrics.get('f1', 0.0):.4f}"
                f"{extra_str}",
                flush=True
            )
            
            # Logging
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('total_loss/val', val_loss, epoch)
            if len(tasks) == 1:
                self.writer.add_scalar('PR_AUC/val', val_metrics.get('auc_pr', 0.0), epoch)
                self.writer.add_scalar('F1/val', val_metrics.get('f1', 0.0), epoch)
            for task in tasks:
                if f'auc_pr_{task}' in val_metrics:
                    self.writer.add_scalar(f'PR_AUC_{task}/val', val_metrics[f'auc_pr_{task}'], epoch)
                if f'f1_{task}' in val_metrics:
                    self.writer.add_scalar(f'F1_{task}/val', val_metrics[f'f1_{task}'], epoch)
                if f'loss_{task}' in val_metrics:
                    self.writer.add_scalar(f'{task}_loss/val', val_metrics[f'loss_{task}'], epoch)
                    self.writer.add_scalar(f'Loss/{task}_val', val_metrics[f'loss_{task}'], epoch)

            for task in tasks:
                if f'loss_{task}_frame' in val_metrics:
                    self.writer.add_scalar(f'{task}_frame_loss/val', val_metrics[f'loss_{task}_frame'], epoch)
                    self.writer.add_scalar(f'Loss/{task}_frame_val', val_metrics[f'loss_{task}_frame'], epoch)
                if f'loss_{task}' in val_metrics:
                    t_val_tot = val_metrics[f'loss_{task}'] + val_metrics.get(f'loss_{task}_frame', 0.0)
                    self.writer.add_scalar(f'{task}_total_loss/val', t_val_tot, epoch)
                    self.writer.add_scalar(f'Loss/{task}_total_val', t_val_tot, epoch)
            if 'frame_proxy_auroc' in val_metrics:
                self.writer.add_scalar('Frame_Proxy_AUROC/val', val_metrics['frame_proxy_auroc'], epoch)

            # PR Curves per task
            if self.config['tensorboard'].get('log_pr_curve', True):
                for task in tasks:
                    t_labels = val_metrics.get(f'labels_{task}', val_metrics.get('labels') if task == primary_task else None)
                    t_logits = val_metrics.get(f'logits_{task}', val_metrics.get('logits') if task == primary_task else None)
                    if t_labels is not None and t_logits is not None and len(t_labels) > 1:
                        if (t_labels > 0.5).any() and (t_labels <= 0.5).any():
                            curve_tag = f'PR_Curve_{task}/val' if len(tasks) > 1 else 'PR_Curve/val'
                            self.writer.add_pr_curve(curve_tag, t_labels, torch.sigmoid(t_logits), epoch)
            
            # Confusion Matrix
            if 'cm' in val_metrics:
                self.log_confusion_matrix(val_metrics['cm'], epoch)
            
            # Audio Samples (Error Analysis)
            if self.config['tensorboard'].get('log_audio_samples', True):
                self.log_audio_samples(epoch, val_metrics.get('samples', {}))
            
            # Early Stopping and Checkpointing — based on configured primary metric
            is_best = False
            current_val = val_metrics.get(self.primary_metric, float('nan'))
            if math.isnan(current_val):
                logger.warning(
                    f"Epoch {epoch}: {self.primary_metric} is NaN (likely single-class validation fold). "
                    "Freezing patience counter."
                )
                self.save_checkpoint(epoch, val_metrics, is_best=False, only_latest=True)
            elif current_val > self.best_primary_metric_val:
                self.best_primary_metric_val = current_val
                self.patience_counter = 0
                # Cache raw outputs for post-training threshold grid search
                self._best_val_logits = val_metrics['logits'].clone()
                self._best_val_labels = val_metrics['labels'].clone()
                self.save_checkpoint(epoch, val_metrics, is_best=True)
                is_best = True
            else:
                self.patience_counter += 1
                # Save latest anyway even if not best, to support robust resume
                self.save_checkpoint(epoch, val_metrics, is_best=False, only_latest=True)

            epoch_time = time.time() - epoch_start_time
            lr_backbone = float(self.optimizer.param_groups[0]['lr']) if len(self.optimizer.param_groups) > 0 else 0.0
            lr_head = float(self.optimizer.param_groups[1]['lr']) if len(self.optimizer.param_groups) > 1 else lr_backbone

            epoch_record = {
                'epoch': int(epoch),
                'train_loss': float(train_loss),
                'train_auc_pr': float(train_metrics.get('auc_pr', 0.0)),
                'train_f1': float(train_metrics.get('f1', 0.0)),
                'val_loss': float(val_loss),
                'val_auc_pr': float(val_metrics.get('auc_pr', 0.0)),
                'val_f1': float(val_metrics.get('f1', 0.0)),
                'val_accuracy': float(val_metrics.get('accuracy', 0.0)),
                'val_precision': float(val_metrics.get('precision', 0.0)),
                'val_recall': float(val_metrics.get('recall', 0.0)),
                'lr_head': lr_head,
                'lr_backbone': lr_backbone,
                'is_best': bool(is_best),
                'elapsed_sec': round(epoch_time, 2),
            }
            for task in tasks:
                if f'loss_{task}' in train_metrics:
                    epoch_record[f'train_loss_{task}'] = float(train_metrics[f'loss_{task}'])
                if f'loss_{task}' in val_metrics:
                    epoch_record[f'val_loss_{task}'] = float(val_metrics[f'loss_{task}'])
                if f'auc_pr_{task}' in train_metrics:
                    epoch_record[f'train_auc_pr_{task}'] = float(train_metrics[f'auc_pr_{task}'])
                if f'auc_pr_{task}' in val_metrics:
                    epoch_record[f'val_auc_pr_{task}'] = float(val_metrics[f'auc_pr_{task}'])
                if f'loss_{task}_frame' in train_metrics:
                    epoch_record[f'train_loss_{task}_frame'] = float(train_metrics[f'loss_{task}_frame'])
                if f'loss_{task}_frame' in val_metrics:
                    epoch_record[f'val_loss_{task}_frame'] = float(val_metrics[f'loss_{task}_frame'])
            for key in ('loss_weak_breathy', 'weak_utterances', 'manual_batches', 'weak_batches'):
                if key in train_metrics:
                    epoch_record[key] = train_metrics[key]
            self.history.append(epoch_record)
            self.save_history()
                
            patience = self.config['eval'].get('early_stopping_patience', 0)
            if patience > 0 and self.patience_counter >= patience:
                print(
                    f"Early stopping at epoch {epoch} "
                    f"({self.primary_metric} did not improve for {patience} epochs)"
                )
                break

        # ── Post-training: threshold grid search on best validation epoch ──
        if self._best_val_logits is not None:
            best_threshold, threshold_f1 = self.find_best_threshold(
                self._best_val_logits, self._best_val_labels
            )
            print(f"Best threshold (grid search): {best_threshold:.3f}  →  F1={threshold_f1:.4f}")
            # Patch it into the best checkpoint on disk
            self._update_checkpoint_threshold(best_threshold, threshold_f1)
            for rec in self.history:
                if rec.get('is_best'):
                    rec['best_threshold'] = float(best_threshold)
                    rec['threshold_f1'] = float(threshold_f1)
            self.save_history()

        return self.history

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
        """Load the best checkpoints from disk, inject threshold fields, and re-save."""
        targets = [self.ckpt_dir / "best_model.pt", self.ckpt_dir / "latest.pt"]
        targets.extend(list(self.ckpt_dir.glob("best_ep*.pt")))
        
        updated_count = 0
        for ckpt_path in set(targets):
            if not ckpt_path.exists():
                continue
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                ckpt['best_threshold'] = best_threshold
                ckpt['threshold_f1'] = threshold_f1
                if 'metrics' in ckpt:
                    ckpt['metrics']['best_threshold'] = best_threshold
                    ckpt['metrics']['threshold_f1'] = threshold_f1
                tmp = ckpt_path.with_suffix('.pt.tmp')
                torch.save(ckpt, tmp)
                os.replace(tmp, ckpt_path)
                updated_count += 1
            except Exception as exc:
                print(f"[WARNING] Failed to update checkpoint with threshold ({ckpt_path}): {exc}")
        print(f"Synchronized optimal threshold {best_threshold:.3f} across {updated_count} checkpoints.")

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
        metrics_to_save = {
            k: v for k, v in metrics.items()
            if k not in ('logits', 'labels', 'samples', 'cm') and not k.startswith(('logits_', 'labels_'))
        }
        
        auc_pr = metrics.get('auc_pr', 0.0)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': state_dict,
            'f1': f1,
            'auc_pr': auc_pr,
            'patience_counter': self.patience_counter,
            'best_primary_metric_val': self.best_primary_metric_val,
            # Persist best-epoch validation outputs so a resumed run can still run
            # the threshold grid search even if it never beats the previous best.
            'best_val_logits': (
                self._best_val_logits.detach().cpu().clone()
                if self._best_val_logits is not None else None
            ),
            'best_val_labels': (
                self._best_val_labels.detach().cpu().clone()
                if self._best_val_labels is not None else None
            ),
            # best_threshold / threshold_f1 are injected later by _update_checkpoint_threshold
            'metrics': metrics_to_save,
            'config': self.config,
            'weak_supervision_state': {
                'fingerprint': getattr(self, 'weak_fingerprint', None),
                'configuration': getattr(self, 'weak_cfg', {}),
                'schedule': {
                    'epochs': self.config.get('training', {}).get('epochs'),
                    'accumulation': self.config.get('training', {}).get('gradient_accumulation_steps', 1),
                    'seed': self.config.get('training', {}).get('seed', 42),
                    'block': self.config.get('training', {}).get('block', {}),
                },
                'total_steps': getattr(self, 'total_steps', None),
                'completed_epoch': epoch,
            },
            'training_provenance': getattr(self, 'training_provenance', {'complete': False}),
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # For non-delta mode, we always want the optimizer to allow resuming from any checkpoint
        if save_mode != 'delta':
            checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
        # Construct descriptive filename
        if is_best:
            auc_pr_for_name = metrics.get('auc_pr', 0.0)
            name = f"best_ep{epoch}_prauc{auc_pr_for_name:.4f}_f1{f1:.4f}.pt"
        else:
            name = f"model_ep{epoch}.pt"
            
        if not only_latest:
            path = self.ckpt_dir / name
            import os
            tmp_path = path.with_suffix('.pt.tmp')
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, path)
        else:
            path = self.ckpt_dir / name # Just for the print message below
        
        # Also maintain a fixed name 'best_model.pt' for script automation using hardlinks (0 bytes overhead)
        if is_best:
            import os
            best_fixed_path = self.ckpt_dir / "best_model.pt"
            if best_fixed_path.exists():
                try:
                    best_fixed_path.unlink()
                except Exception as e:
                    print(f"Warning: could not unlink {best_fixed_path}: {e}")
            try:
                os.link(path, best_fixed_path)
            except Exception as e:
                print(f"Warning: could not create hardlink for {best_fixed_path}: {e}. Falling back to copy.")
                import shutil
                shutil.copy(path, best_fixed_path)
            
            # Cleanup old best checkpoints (keep the N best by PR-AUC)
            keep_count = self.config['checkpoint'].get('keep_best_count', 3)
            best_files = list(self.ckpt_dir.glob("best_ep*_prauc*.pt"))
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
            
        latest_path = self.ckpt_dir / "latest.pt"
        import os
        tmp_latest = latest_path.with_suffix('.pt.tmp')
        torch.save(checkpoint, tmp_latest)
        os.replace(tmp_latest, latest_path)
            
        if only_latest:
            print(f"Updated latest checkpoint: {latest_path} (mode: {save_mode})")
        else:
            print(f"Checkpoint saved to {path} (mode: {save_mode})")
        return path

    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path} for resuming...")
        try:
            checkpoint = torch.load(path, map_location=self.device, mmap=True, weights_only=False)
        except (TypeError, Exception):
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        saved_weak = checkpoint.get('weak_supervision_state', {})
        current_fingerprint = getattr(self, 'weak_fingerprint', None)
        if saved_weak.get('fingerprint') != current_fingerprint:
            raise ValueError("Weak supervision manifest changed on resume; start a new experiment")
        if current_fingerprint is not None:
            current_schedule = {
                'epochs': self.config['training']['epochs'],
                'accumulation': self.config['training'].get('gradient_accumulation_steps', 1),
                'seed': self.config['training'].get('seed', 42),
                'block': self.config['training'].get('block', {}),
            }
            if (saved_weak.get('configuration') != self.weak_cfg
                    or saved_weak.get('schedule') != current_schedule
                    or saved_weak.get('total_steps') != self.total_steps):
                raise ValueError("Weak supervision schedule/configuration changed on resume")
        self.optimizer.zero_grad(set_to_none=True)
        inherited = checkpoint.get('training_provenance', {})
        current = getattr(self, 'training_provenance', {'complete': False})
        current['complete'] = bool(current.get('complete') and inherited.get('complete'))
        for key in ('training_files', 'training_speakers'):
            current[key] = sorted(set(map(tuple, current.get(key, []))) | set(map(tuple, inherited.get(key, []))))
        self.training_provenance = current

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
            torch.set_rng_state(rng_state['torch'].cpu())
            if torch.cuda.is_available() and rng_state.get('cuda') is not None:
                torch.cuda.set_rng_state_all([s.cpu() if isinstance(s, torch.Tensor) else s for s in rng_state['cuda']])

        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.best_primary_metric_val = checkpoint.get('best_primary_metric_val', 0.0)
        self._best_val_logits = checkpoint.get('best_val_logits')
        self._best_val_labels = checkpoint.get('best_val_labels')
        if self._best_val_labels is not None:
            self._best_val_labels = self._best_val_labels.cpu()
        if self._best_val_logits is not None:
            self._best_val_logits = self._best_val_logits.cpu()
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # Restore epoch history up to checkpoint['epoch']
        for hist_cand in (self.ckpt_dir / "epoch_history.json", self.log_dir / "epoch_history.json"):
            if hist_cand.exists():
                try:
                    import json
                    with open(hist_cand, 'r', encoding='utf-8') as f:
                        old_hist = json.load(f)
                    self.history = [h for h in old_hist if h.get('epoch', 0) <= checkpoint['epoch']]
                    break
                except Exception as e:
                    logger.warning(f"Could not reload history from {hist_cand}: {e}")

        return checkpoint['epoch']
