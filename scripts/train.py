import argparse
import yaml
import torch
import sys
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import copy
import json
import warnings

# Suppress upstream PyTorch 2.6+ deprecation warning regarding WavLM's boolean key_padding_mask
warnings.filterwarnings("ignore", message=".*Support for mismatched key_padding_mask and attn_mask is deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.AudioMetaData.*")
warnings.filterwarnings("ignore", message=".*No positive class found in y_true.*")

# Add parent of 06_wavfry to sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Add the 06_wavfry directory to sys.path to allow 'import wavfry'
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from wavfry.data import dataset as dataset_mod
from wavfry.data import splits as splits_mod
from wavfry.model import wavlm_classifier as model_mod
from wavfry.train import trainer as trainer_mod

def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_training(config, df, device, dataset_mod, splits_mod, model_mod, trainer_mod, fold_idx=0, preloaded_wavlm=None, resume=False, warm_start=None):
    # Set seed for this fold/run (seed + fold_idx for variety in LOOCV)
    set_seed(config['training'].get('seed', 42) + fold_idx)
    if 'tasks' not in config.get('model', {}):
        print("[WARNING] 'model.tasks' not configured in config, defaulting to ['creaky']")
        config.setdefault('model', {})['tasks'] = ['creaky']
    
    # Split: Always use manifest batch_split as single source of truth
    isolate_speakers = bool(config['data'].get('isolate_split_speakers', True))
    train_df, val_df, test_df = splits_mod.make_batch_split(df, isolate=isolate_speakers)

    # Defense in depth: with speaker isolation enabled, verify it actually
    # removed every overlapping speaker. Silent isolation failures previously
    # relied solely on hand-maintained sampling_state split fields.
    if isolate_speakers and bool(config['data'].get('assert_split_exclusivity', True)):
        train_spk = set(train_df['speaker_id'].dropna().unique())
        for part_name, part_df in (("val", val_df), ("test", test_df)):
            if part_df.empty:
                continue
            overlap = train_spk & set(part_df['speaker_id'].dropna().unique())
            if overlap:
                raise RuntimeError(
                    f"[SPLIT EXCLUSIVITY VIOLATION] train and {part_name} share "
                    f"{len(overlap)} speaker(s): {sorted(overlap)[:10]}. "
                    "Speaker isolation failed - inspect the manifest batch_split "
                    "assignment before training."
                )

    print(f"Data Partition: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)} segments")
    if len(val_df) == 0:
        raise ValueError(
            "[DATA SPLIT ERROR] Validation dataset is empty (0 segments)! "
            "Please check manifest batch_split column. "
            "Training without validation is strictly prohibited as early stopping and best checkpointing cannot function."
        )
    if len(train_df) == 0:
        raise ValueError(
            "[DATA SPLIT ERROR] Training dataset is empty (0 segments)! "
            "Please check manifest batch_split column."
        )

    if not test_df.empty:
        test_out_path = Path(config['checkpoint']['dir']) / f"batch_test_set_{fold_idx}.csv"
        test_out_path.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(test_out_path, index=False)
        print(f"[Test] Batch Test Set: {len(test_df)} segments saved to {test_out_path}")
    
    # Datasets
    paradigm = config['training'].get('paradigm', 'segment')
    if paradigm not in {'segment', 'block'}:
        raise ValueError(f"Unknown training paradigm: {paradigm}")
    augmentor = None
    aug_cfg = config.get('augmentation', {})
    if aug_cfg.get('enabled', False):
        from wavfry.data.augmentation import AudioAugmentor
        train_epochs = config.get('training', {}).get('epochs', 30)
        augmentor = AudioAugmentor(aug_cfg, max_epochs=train_epochs)
        print(f"Data augmentation enabled for training (scheduled across {train_epochs} epochs).")

    if paradigm == 'block':
        from wavfry.data.paradigm_block import ContinuousAudioBlockDataset, collate_block_fn
        block_cfg = config['training']['block']
        common_block = dict(
            duration_ms=block_cfg.get('duration_ms', 2000),
            hop_ms=block_cfg.get('hop_ms', 1000),
            max_intervals=block_cfg.get('max_intervals', 8),
            boundary_trim_frames=block_cfg.get('boundary_trim_frames', 0),
            seed=config['training'].get('seed', 42),
            fold_index=fold_idx,
        )
        train_ds = ContinuousAudioBlockDataset(train_df, **common_block)
        val_ds = ContinuousAudioBlockDataset(val_df, **common_block)
        collate = collate_block_fn
        train_batch_size = int(block_cfg['batch_size'])
        val_batch_size = train_batch_size
    else:
        segment_cfg = config['training']['segment']
        train_ds = dataset_mod.CreakyVowelDataset(
            train_df,
            context_padding_ms=segment_cfg['context_padding_ms'],
            continuous_non_creaky_cfg=segment_cfg.get('continuous_non_creaky_sampling', {}),
            seed=config['training'].get('seed', 42),
            fold_index=fold_idx,
        )
        val_ds = dataset_mod.CreakyVowelDataset(
            val_df,
            context_padding_ms=segment_cfg['context_padding_ms'],
        )
        collate = dataset_mod.collate_fn
        train_batch_size = int(segment_cfg['batch_size'])
        val_batch_size = train_batch_size

    num_workers = dataset_mod.resolve_num_workers(config['data']['num_workers'])
    prefetch_factor = config['data'].get('prefetch_factor', 2) if num_workers > 0 else None
    persistent_workers = bool(config['data'].get('persistent_workers', False)) if num_workers > 0 else False
    loader_timeout = float(config['data'].get('timeout', 60.0)) if num_workers > 0 else 0.0

    from wavfry.data.epoch_sampler import EpochShuffleSampler
    epoch_sampler = EpochShuffleSampler(
        train_ds,
        seed=config['training'].get('seed', 42),
        fold_index=fold_idx,
    )

    def build_loaders():
        # A fresh DataLoader spawns a fresh worker pool and re-pickles the dataset
        # with its current epoch state. This is also the recovery path used by the
        # watchdog: persistent worker pools cannot be revived in place.
        train_loader = DataLoader(
            train_ds,
            batch_size=train_batch_size,
            sampler=epoch_sampler,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=config['data']['pin_memory'],
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            timeout=loader_timeout,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=config['data']['pin_memory'],
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            timeout=loader_timeout,
        )
        return train_loader, val_loader

    train_loader, val_loader = build_loaders()
    
    # Model
    if preloaded_wavlm is not None:
        # Fast path: Deepcopy the preloaded backbone to avoid redundant loading
        model = model_mod.WavFryClassifier(config, wavlm_instance=copy.deepcopy(preloaded_wavlm)).to(device)
    else:
        model = model_mod.WavFryClassifier(config).to(device)
    
    if warm_start:
        try:
            checkpoint = torch.load(warm_start, map_location=device, mmap=True, weights_only=False)
        except (TypeError, Exception):
            checkpoint = torch.load(warm_start, map_location=device, weights_only=False)
        result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Warm-started model weights from {warm_start}")
        if result.missing_keys:
            print(f"[INFO] Newly initialized keys: {result.missing_keys}")

    # Trainer
    trainer = trainer_mod.Trainer(model, train_loader, val_loader, config, device, augmentor=augmentor)
    
    start_epoch = 1
    if resume:
        latest_path = Path(config['checkpoint']['dir']) / "latest.pt"
        if latest_path.exists():
            try:
                last_epoch = trainer.load_checkpoint(latest_path)
                start_epoch = last_epoch + 1
                print(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                raise RuntimeError(
                    f"[RESUME ERROR] Failed to load resume checkpoint from '{latest_path}': {e}. "
                    f"Academic training refuses silent fallback to training from scratch!"
                ) from e

    max_watchdog_retries = int(config.get('data', {}).get('max_watchdog_retries', 3))
    attempt = 0
    try:
        while True:
            try:
                history = trainer.run(start_epoch=start_epoch)
                return history
            except RuntimeError as e:
                err_msg = str(e).lower()
                is_timeout = ("timed out" in err_msg or "dataloader" in err_msg)
                if is_timeout and attempt < max_watchdog_retries:
                    attempt += 1
                    print(
                        f"\n[WATCHDOG RECOVERY ALERT] DataLoader worker timeout/deadlock detected: {e}\n"
                        f"Attempting self-healing restart ({attempt}/{max_watchdog_retries})...",
                        flush=True
                    )
                    latest_path = Path(config['checkpoint']['dir']) / "latest.pt"
                    if latest_path.exists():
                        last_epoch = trainer.load_checkpoint(latest_path)
                        start_epoch = last_epoch + 1
                    else:
                        print("[WATCHDOG RECOVERY] No latest.pt found; resuming from scratch state.", flush=True)
                    # A persistent worker pool cannot be revived in place: dead or
                    # deadlocked worker processes are reused by the next iteration.
                    # Rebuild the writer (the failed run may have mid-written events)
                    # and both DataLoaders so training continues on a healthy pool
                    # carrying the dataset's current epoch state.
                    trainer.rebuild_writer()
                    train_loader, val_loader = build_loaders()
                    trainer.train_loader = train_loader
                    trainer.val_loader = val_loader
                    print(
                        f"[WATCHDOG RECOVERY] Rebuilt TensorBoard writer and DataLoaders; "
                        f"resuming from epoch {start_epoch}...",
                        flush=True
                    )
                    continue
                raise
            except KeyboardInterrupt:
                print("\nTraining interrupted by user. Gracefully exiting...")
                sys.exit(1)
    finally:
        trainer.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(root / 'config' / '_defaults.yaml'))
    parser.add_argument('--override', type=str, nargs='*', help='Override config keys: key.subkey=value')
    parser.add_argument('--loocv', action='store_true', help='Run Leave-One-Speaker-Out Cross Validation')
    parser.add_argument('--speakers', type=str, default=None, help='Comma-separated list of speakers for LOOCV')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    parser.add_argument('--warm-start', type=str, default=None, help='Load compatible model weights only')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Apply overrides
    if args.override:
        for item in args.override:
            if '=' in item:
                key_path, v = item.split('=', 1)
                if key_path.startswith(('smoketest', 'is_smoketest', 'data.val_speakers', 'sampler', 'training.sampler', 'data.sampler')):
                    print(f"[DEPRECATED OVERRIDE IGNORED] {key_path} is deprecated and ignored (manifest batch_split is the unified source of truth).")
                    continue
                # Parse value
                if v.lower() == 'true': v = True
                elif v.lower() == 'false': v = False
                elif v.startswith('[') and v.endswith(']'):
                    # Support list override: data.val_speakers=[U1001,U2002]
                    v = [x.strip() for x in v[1:-1].split(',') if x.strip()]
                else:
                    try: v = int(v)
                    except ValueError:
                        try: v = float(v)
                        except ValueError: pass
                
                # Navigate and validate
                keys = key_path.split('.')
                curr = config
                for k in keys[:-1]:
                    if k not in curr or not isinstance(curr[k], dict):
                        raise KeyError(
                            f"[CONFIG OVERRIDE ERROR] Invalid configuration key '{key_path}'! "
                            f"Segment '{k}' does not exist in base configuration. Check for typos."
                        )
                    curr = curr[k]
                if keys[-1] not in curr:
                    raise KeyError(
                        f"[CONFIG OVERRIDE ERROR] Invalid configuration key '{key_path}'! "
                        f"Field '{keys[-1]}' does not exist in base configuration. Check for typos."
                    )
                curr[keys[-1]] = v
                print(f"Overriding {key_path} with {v}")
            
    # Device
    # Resolve relative paths in config to be relative to 06_wavfry root
    if not Path(config['tensorboard']['log_dir']).is_absolute():
        config['tensorboard']['log_dir'] = str(root / config['tensorboard']['log_dir'])
    if not Path(config['checkpoint']['dir']).is_absolute():
        config['checkpoint']['dir'] = str(root / config['checkpoint']['dir'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Auto-detect tasks from datasets.yaml if not explicitly configured in model
    datasets_cfg_path = root / "config" / "datasets.yaml"
    if 'tasks' not in config.get('model', {}):
        if datasets_cfg_path.exists():
            with open(datasets_cfg_path, 'r', encoding='utf-8') as f:
                ds_registry = yaml.safe_load(f)
            inferred_tasks = set()
            ds_items = ds_registry.get('datasets', [])
            if isinstance(ds_items, dict):
                ds_items = ds_items.values()
            for ds_info in ds_items:
                if isinstance(ds_info, dict) and ds_info.get('enabled', False):
                    for task, task_spec in ds_info.get('label_schema', {}).items():
                        task_active = (
                            task_spec.get('enabled', True) or task_spec.get('silence_as_negative', False)
                            if isinstance(task_spec, dict)
                            else bool(task_spec)
                        )
                        if task_active:
                            inferred_tasks.add(task)
            canonical_order = ["creaky", "breathy"]
            if inferred_tasks:
                config.setdefault('model', {})['tasks'] = [t for t in canonical_order if t in inferred_tasks] + [t for t in inferred_tasks if t not in canonical_order]
            else:
                config.setdefault('model', {})['tasks'] = ['creaky']
        else:
            config.setdefault('model', {})['tasks'] = ['creaky']
    print(f"Model tasks: {config['model']['tasks']}")

    # Load manifest (prefer segments_multi.csv if present)
    multi_manifest = root / "manifests" / "segments_multi.csv"
    legacy_manifest = root / "manifests" / "segments_all.csv"
    if multi_manifest.exists():
        manifest_path = multi_manifest
        print(f"Using multi-dataset manifest: {manifest_path}")
    elif legacy_manifest.exists():
        manifest_path = legacy_manifest
        print(f"Using legacy manifest: {manifest_path}")
    else:
        print(f"Manifest not found at {multi_manifest} or {legacy_manifest}. Please run build_manifest.py first.")
        sys.exit(1)
        
    import csv
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    pd.set_option("mode.string_storage", "python")
    df = pd.DataFrame(data)
    # Convert numeric columns
    for col in ['start_s', 'end_s', 'label', 'label_creaky', 'label_breathy']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Task-aware manifest filtering:
    # Retain only rows that have at least one valid (non-NaN) label for the model's active tasks.
    active_tasks = config.get('model', {}).get('tasks', ['creaky'])
    task_cols = [f"label_{t}" if f"label_{t}" in df.columns else "label" for t in active_tasks]
    existing_task_cols = [c for c in task_cols if c in df.columns]
    if existing_task_cols:
        has_valid_label = df[existing_task_cols].notna().any(axis=1)
        dropped_count = len(df) - int(has_valid_label.sum())
        if dropped_count > 0:
            print(
                f"[Task Filter] Active tasks: {active_tasks}. Retaining rows with valid annotations "
                f"across {existing_task_cols}. Filtered out {dropped_count} irrelevant segments "
                f"(remaining: {int(has_valid_label.sum())} segments)."
            )
            df = df[has_valid_label].copy()

    is_loocv = bool(args.loocv or (config.get('training', {}).get('mode') == 'loocv'))
    
    # Smoketest overrides
    is_smoketest = "smoketest" in str(args.config) or (args.override and any("smoketest" in str(x) for x in args.override))
    if is_smoketest:
        if is_loocv:
            print("[INFO] Smoketest detected: Forcing 2-fold LOOCV with limited subsets.")
            is_loocv = True
        else:
            print("[INFO] Smoketest detected in single-split mode.")
            is_loocv = False
            seed = config['training'].get('seed', 42)
            pos = df[df['label'] == 1]
            neg = df[df['label'] == 0]
            n_pos = min(len(pos), 64)
            n_neg = min(len(neg), 64)
            df = pd.concat([pos.sample(n=n_pos, random_state=seed), neg.sample(n=n_neg, random_state=seed)]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
            print(f"[INFO] Smoketest dynamic slice: {len(df)} segments total.")
        
    if is_loocv:
        # Exclude non-train batches (e.g. Batch 2 or test batches) from LOOCV folds
        if 'batch_split' in df.columns:
            original_len = len(df)
            df = df[df['batch_split'] == 'train'].copy()
            if len(df) < original_len:
                print(f"[INFO] LOOCV Mode: Excluded {original_len - len(df)} non-train batch segments (val/test/unassigned).")
        elif 'batch_id' in df.columns:
            batch_ids = pd.to_numeric(df['batch_id'], errors='coerce')
            original_len = len(df)
            df = df[batch_ids != 2].copy()
            if len(df) < original_len:
                print(f"[INFO] LOOCV Mode: Excluded {original_len - len(df)} Batch 2 segments to maintain dataset consistency.")

        # Get all speakers to iterate over (from CLI --speakers, or dynamically from dataset)
        if args.speakers:
            speakers = [s.strip() for s in args.speakers.split(',') if s.strip()]
        else:
            # Only speakers providing valid labels for the primary task may form
            # a fold: a fold whose primary-task labels are all NaN reports 0.0
            # metrics forever (bypassing the NaN guard) and silently drops out
            # of the fold results.
            primary_task = config.get('eval', {}).get('primary_task', 'creaky')
            label_col = f'label_{primary_task}' if f'label_{primary_task}' in df.columns else 'label'
            label_valid = pd.to_numeric(df[label_col], errors='coerce')
            speakers = sorted(df.loc[label_valid.notna(), 'speaker_id'].dropna().unique())
            excluded_spk = sorted(set(df['speaker_id'].dropna().unique()) - set(speakers))
            if excluded_spk:
                print(f"[INFO] LOOCV: excluding {len(excluded_spk)} speaker(s) with no valid "
                      f"'{label_col}' labels: {excluded_spk[:10]}")
        if is_smoketest and not args.speakers:
            print(f"[INFO] Limiting LOOCV to the first two folds (speakers: {speakers[:2]})")
            speakers = speakers[:2]
        print(f"Starting LOOCV for speakers: {speakers}")
        
        base_log_dir = Path(config['tensorboard']['log_dir'])
        base_ckpt_dir = Path(config['checkpoint']['dir'])
        
        global_best_f1 = 0.0
        global_best_speaker = None
        
        # Optimization: Pre-load the backbone model once to avoid redundant loading in each fold
        print(f"Pre-loading backbone: {config['model']['backbone']}...")
        from transformers import WavLMModel
        preloaded_wavlm = WavLMModel.from_pretrained(config['model']['backbone'])
        # Keep on CPU to save GPU memory during the full LOOCV process
        preloaded_wavlm.cpu() 
        
        fold_results = []
        result_path = base_log_dir / "loocv_results.json"
        if args.resume and result_path.exists():
            try:
                with open(result_path, 'r', encoding='utf-8') as f:
                    res_data = json.load(f)
                for fold_res in res_data.get('folds', []):
                    fold_results.append({
                        'speaker': fold_res.get('speaker'),
                        'accuracy': fold_res.get('accuracy', 0.0),
                        'precision': fold_res.get('precision', 0.0),
                        'recall': fold_res.get('recall', 0.0),
                        'f1': fold_res.get('f1', 0.0),
                        'auc_pr': fold_res.get('auc_pr', fold_res.get('auc_pr', 0.0)),
                        'best_threshold': fold_res.get('best_threshold', 0.5),
                        'threshold_f1': fold_res.get('threshold_f1', 0.0),
                        'epoch': fold_res.get('epoch', 0),
                        'peak_vram_mb': fold_res.get('peak_vram_mb', 0.0)
                    })
                # Re-calculate global best
                for r in fold_results:
                    if r['auc_pr'] > global_best_f1:
                        global_best_f1 = r['auc_pr']
                        global_best_speaker = r['speaker']
                print(f"[INFO] Loaded {len(fold_results)} pre-existing completed folds from JSON report.")
            except Exception as e:
                print(f"[WARNING] Failed to parse loocv_results.json: {e}")

        for i, val_speaker in enumerate(speakers):
            print(f"\n{'='*30}\nFold: {val_speaker}\n{'='*30}")
            
            # Check if this fold is already completed (for resume)
            fold_completed = False
            fold_metric_data = None
            
            # 1. First check if it is documented in the pre-loaded in-memory fold_results
            for fold_res in fold_results:
                if fold_res.get('speaker') == val_speaker:
                    fold_metric_data = fold_res
                    fold_completed = True
                    print(f"[INFO] Fold for {val_speaker} already completed in JSON report (loaded in-memory). Skipping training.")
                    break
            
            # 2. Fallback: check if the best_model_{speaker}.pt checkpoint file exists
            fold_final_path = base_ckpt_dir / f"best_model_{val_speaker}.pt"
            if not fold_completed and args.resume and fold_final_path.exists():
                try:
                    # Locally produced checkpoints include NumPy RNG state, which
                    # PyTorch 2.6 rejects under weights_only=True.
                    ckpt = torch.load(fold_final_path, map_location='cpu', weights_only=False)
                    metrics = ckpt.get('metrics', {'f1': ckpt.get('f1', 0)})
                    fold_metric_data = {
                        'speaker': val_speaker,
                        'accuracy': metrics.get('accuracy', 0),
                        'precision': metrics.get('precision', 0),
                        'recall': metrics.get('recall', 0),
                        'f1': metrics.get('f1', 0),
                        'auc_pr': metrics.get('auc_pr', ckpt.get('auc_pr', 0)),
                        'best_threshold': metrics.get('best_threshold', ckpt.get('best_threshold', 0.5)),
                        'threshold_f1': metrics.get('threshold_f1', ckpt.get('threshold_f1', 0)),
                        'epoch': ckpt.get('epoch', 0),
                        'peak_vram_mb': 0.0
                    }
                    fold_completed = True
                    print(f"[INFO] Fold for {val_speaker} already completed via checkpoint file. Skipping training.")
                except Exception as e:
                    print(f"[WARNING] Failed to load checkpoint for {val_speaker}: {e}")

            if fold_completed and fold_metric_data:
                if not any(r['speaker'] == val_speaker for r in fold_results):
                    fold_results.append(fold_metric_data)
                if fold_metric_data['auc_pr'] > global_best_f1:
                    global_best_f1 = fold_metric_data['auc_pr']
                    global_best_speaker = val_speaker
                continue
            
            fold_config = copy.deepcopy(config)
            # Logs are still best kept in subdirs for TB comparison
            fold_config['tensorboard']['log_dir'] = str(base_log_dir / f"fold_{val_speaker}")
            # Checkpoints will go to the same base dir, but with a speaker-specific name
            fold_config['checkpoint']['dir'] = str(base_ckpt_dir)
            
            # Clean up residual checkpoint from previous fold to avoid race conditions
            fold_best_path = base_ckpt_dir / "best_model.pt"
            if not args.resume and fold_best_path.exists():
                fold_best_path.unlink()
            
            if is_smoketest:
                # Dynamic dataset slicing for smoketest
                seed = config['training'].get('seed', 42) + i
                
                # Validation split: speaker == val_speaker (32 segments)
                val_df_full = df[df['speaker_id'] == val_speaker]
                val_pos = val_df_full[val_df_full['label'] == 1]
                val_neg = val_df_full[val_df_full['label'] == 0]
                
                n_val_pos = min(len(val_pos), 16)
                n_val_neg = 32 - n_val_pos
                if n_val_neg > len(val_neg):
                    n_val_neg = len(val_neg)
                    n_val_pos = min(len(val_pos), 32 - n_val_neg)
                
                val_pos_sampled = val_pos.sample(n=n_val_pos, random_state=seed) if n_val_pos > 0 else val_pos
                val_neg_sampled = val_neg.sample(n=n_val_neg, random_state=seed) if n_val_neg > 0 else val_neg
                fold_val_df = pd.concat([val_pos_sampled, val_neg_sampled]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
                fold_val_df['batch_split'] = 'val'
                
                # Train split: speaker != val_speaker (100 segments)
                train_df_full = df[df['speaker_id'] != val_speaker]
                train_pos = train_df_full[train_df_full['label'] == 1]
                train_neg = train_df_full[train_df_full['label'] == 0]
                
                n_train_pos = min(len(train_pos), 50)
                n_train_neg = 100 - n_train_pos
                if n_train_neg > len(train_neg):
                    n_train_neg = len(train_neg)
                    n_train_pos = min(len(train_pos), 100 - n_train_neg)
                
                train_pos_sampled = train_pos.sample(n=n_train_pos, random_state=seed) if n_train_pos > 0 else train_pos
                train_neg_sampled = train_neg.sample(n=n_train_neg, random_state=seed) if n_train_neg > 0 else train_neg
                fold_train_df = pd.concat([train_pos_sampled, train_neg_sampled]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
                fold_train_df['batch_split'] = 'train'
                
                fold_df = pd.concat([fold_train_df, fold_val_df]).reset_index(drop=True)
                print(f"[INFO] Smoketest dynamic slice | Train: {len(fold_train_df)} (Pos: {n_train_pos}, Neg: {n_train_neg}) | Val: {len(fold_val_df)} (Pos: {n_val_pos}, Neg: {n_val_neg})")
            else:
                import numpy as np
                fold_df = df.copy()
                fold_df['batch_split'] = np.where(fold_df['speaker_id'] == val_speaker, 'val', 'train')
            
            run_training(fold_config, fold_df, device, dataset_mod, splits_mod, model_mod, trainer_mod,
                         fold_idx=i, preloaded_wavlm=preloaded_wavlm, resume=args.resume,
                         warm_start=args.warm_start)
            
            # After training this fold, find the best model it saved
            if fold_best_path.exists():
                # Load to check F1
                ckpt = torch.load(fold_best_path, map_location='cpu', weights_only=False)
                metrics = ckpt.get('metrics', {'f1': ckpt.get('f1', 0)})
                f1 = metrics.get('f1', 0)
                auc_pr = metrics.get('auc_pr', ckpt.get('auc_pr', 0))
                best_threshold = metrics.get('best_threshold', ckpt.get('best_threshold', 0.5))
                threshold_f1 = metrics.get('threshold_f1', ckpt.get('threshold_f1', 0))
                
                # Capture peak VRAM allocated during the fold training
                peak_vram = 0.0
                if torch.cuda.is_available():
                    peak_vram = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    torch.cuda.reset_peak_memory_stats(device)
                
                fold_results.append({
                    'speaker': val_speaker,
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': f1,
                    'auc_pr': auc_pr,
                    'best_threshold': best_threshold,
                    'threshold_f1': threshold_f1,
                    'epoch': ckpt.get('epoch', 0),
                    'peak_vram_mb': peak_vram
                })
                
                new_path = base_ckpt_dir / f"best_model_{val_speaker}.pt"
                # Move (overwrite if exists)
                if new_path.exists(): new_path.unlink()
                fold_best_path.rename(new_path)
                
                print(f"Fold {val_speaker} | PR-AUC: {auc_pr:.4f} | Best-threshold F1: {threshold_f1:.4f} @ t={best_threshold:.3f}")
                
                if auc_pr > global_best_f1:  # variable name kept for compat; now tracks best PR-AUC
                    global_best_f1 = auc_pr
                    global_best_speaker = val_speaker
                    print(f"*** Best Fold so far: Speaker: {val_speaker}, PR-AUC: {auc_pr:.4f} ***")
                
                # Cleanup intermediate checkpoints (best_epX_...pt and latest_*.pt) to save disk/ramdisk space
                try:
                    latest_path = base_ckpt_dir / "latest.pt"
                    if latest_path.exists():
                        latest_path.unlink()
                        print(f"[INFO] Cleaned up intermediate resume checkpoint: {latest_path}")
                    for ep_file in base_ckpt_dir.glob(f"best_ep*_prauc*_{val_speaker}.pt"):
                        try:
                            ep_file.unlink()
                            print(f"[INFO] Cleaned up redundant epoch checkpoint: {ep_file}")
                        except:
                            pass
                except Exception as e:
                    print(f"Warning: Failed to clean up intermediate fold checkpoints: {e}")
            else:
                print(f"Warning: Fold {val_speaker} did not produce a best_model.pt (F1 did not improve beyond 0.0 or training failed).")
            
            # Resource cleanup between folds
            import gc
            gc.collect()
            torch.cuda.empty_cache()
                    
            # Save incremental report after each fold completes
            if fold_results:
                import numpy as np
                
                accs = [r['accuracy'] for r in fold_results]
                precs = [r['precision'] for r in fold_results]
                recs = [r['recall'] for r in fold_results]
                f1s = [r['f1'] for r in fold_results]
                auc_prs = [r.get('auc_pr', 0) for r in fold_results]
                vrams = [r.get('peak_vram_mb', 0.0) for r in fold_results]
                
                report = {
                    'mean_accuracy': float(np.mean(accs)),
                    'std_accuracy': float(np.std(accs)),
                    'mean_precision': float(np.mean(precs)),
                    'std_precision': float(np.std(precs)),
                    'mean_recall': float(np.mean(recs)),
                    'std_recall': float(np.std(recs)),
                    'mean_f1': float(np.mean(f1s)),
                    'std_f1': float(np.std(f1s)),
                    'mean_auc_pr': float(np.mean(auc_prs)),
                    'std_auc_pr': float(np.std(auc_prs)),
                    'mean_peak_vram_mb': float(np.mean(vrams)),
                    'folds': fold_results,
                    'best_fold': {'speaker': global_best_speaker, 'auc_pr': global_best_f1}
                }
                
                # Print intermediate summary to console
                print("\n" + "="*40)
                print(f"LOOCV PROGRESS REPORT (Completed Folds: {len(fold_results)}/{len(speakers)})")
                print("="*40)
                print(f"Mean Accuracy:  {report['mean_accuracy']:.4f} \u00b1 {report['std_accuracy']:.4f}")
                print(f"Mean Precision: {report['mean_precision']:.4f} \u00b1 {report['std_precision']:.4f}")
                print(f"Mean Recall:    {report['mean_recall']:.4f} \u00b1 {report['std_recall']:.4f}")
                print(f"Mean F1:        {report['mean_f1']:.4f} \u00b1 {report['std_f1']:.4f}")
                print(f"Mean PR-AUC:    {report['mean_auc_pr']:.4f} \u00b1 {report['std_auc_pr']:.4f}  [primary]")
                print(f"Mean Peak VRAM: {report['mean_peak_vram_mb']:.2f} MB")
                print("-" * 40)
                for r in fold_results:
                    print(
                        f"Speaker: {r['speaker']:<10} | Acc: {r['accuracy']:.4f} | P: {r['precision']:.4f}"
                        f" | R: {r['recall']:.4f} | F1: {r['f1']:.4f}"
                        f" | PR-AUC: {r.get('auc_pr', 0):.4f}"
                        f" | Best-t: {r.get('best_threshold', 0.5):.3f} (F1={r.get('threshold_f1', 0):.4f})"
                        f" | VRAM: {r.get('peak_vram_mb', 0.0):.2f} MB"
                    )
                print("="*40)
                
                # Save to JSON
                import os
                result_path = base_log_dir / "loocv_results.json"
                tmp_path = result_path.with_suffix('.json.tmp')
                with open(tmp_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=4)
                os.replace(tmp_path, result_path)
                    
                # Save to Markdown
                md_path = base_log_dir / "loocv_report.md"
                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write("# LOOCV Training Report\n\n")
                    f.write(f"Generated incrementally. Completed Folds: **{len(fold_results)}/{len(speakers)}**.\n\n")
                    f.write("## Summary Statistics\n")
                    f.write(f"- **Mean PR-AUC** *(primary)*: {report['mean_auc_pr']:.4f} \u00b1 {report['std_auc_pr']:.4f}\n")
                    f.write(f"- **Mean F1**: {report['mean_f1']:.4f} \u00b1 {report['std_f1']:.4f}\n")
                    f.write(f"- **Mean Accuracy**: {report['mean_accuracy']:.4f} \u00b1 {report['std_accuracy']:.4f}\n")
                    f.write(f"- **Mean Precision**: {report['mean_precision']:.4f} \u00b1 {report['std_precision']:.4f}\n")
                    f.write(f"- **Mean Recall**: {report['mean_recall']:.4f} \u00b1 {report['std_recall']:.4f}\n")
                    f.write(f"- **Mean Peak VRAM**: {report['mean_peak_vram_mb']:.2f} MB\n\n")
                    f.write("## Per-Fold Details\n\n")
                    f.write("| Speaker | PR-AUC | F1 | Best Threshold | Threshold F1 | Accuracy | Precision | Recall | Best Epoch | Peak VRAM |\n")
                    f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                    for r in fold_results:
                        f.write(
                            f"| {r['speaker']} | {r.get('auc_pr', 0):.4f} | {r['f1']:.4f}"
                            f" | {r.get('best_threshold', 0.5):.3f} | {r.get('threshold_f1', 0):.4f}"
                            f" | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f}"
                            f" | {r['epoch']} | {r.get('peak_vram_mb', 0.0):.2f} MB |\n"
                        )
                    f.write(f"\n**Best Fold so far** *(by PR-AUC)*: {global_best_speaker} ({global_best_f1:.4f})\n")
                
                print(f"[INFO] Incremental results saved to {result_path} and {md_path}")

    else:
        history = run_training(config, df, device, dataset_mod, splits_mod, model_mod, trainer_mod,
                               resume=args.resume, warm_start=args.warm_start)
        
        # Write single fold results to JSON for ablation script compatibility
        base_log_dir = Path(config['tensorboard']['log_dir'])
        base_ckpt_dir = Path(config['checkpoint']['dir'])
        best_ckpt_path = base_ckpt_dir / "best_model.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location='cpu', weights_only=False)
            metrics = ckpt.get('metrics', {'f1': ckpt.get('f1', 0)})
            f1 = metrics.get('f1', 0)
            auc_pr = metrics.get('auc_pr', ckpt.get('auc_pr', 0))
            best_threshold = metrics.get('best_threshold', ckpt.get('best_threshold', 0.5))
            threshold_f1 = metrics.get('threshold_f1', ckpt.get('threshold_f1', 0))
            
            val_spk_str = "batch_split"
            single_fold_res = {
                'speaker': val_spk_str,
                'accuracy': metrics.get('accuracy', 0),
                'precision': metrics.get('precision', 0),
                'recall': metrics.get('recall', 0),
                'f1': f1,
                'auc_pr': auc_pr,
                'best_threshold': best_threshold,
                'threshold_f1': threshold_f1,
                'epoch': ckpt.get('epoch', 0),
                'peak_vram_mb': 0.0
            }
            report = {
                'mean_accuracy': single_fold_res['accuracy'],
                'std_accuracy': 0.0,
                'mean_precision': single_fold_res['precision'],
                'std_precision': 0.0,
                'mean_recall': single_fold_res['recall'],
                'std_recall': 0.0,
                'mean_f1': single_fold_res['f1'],
                'std_f1': 0.0,
                'mean_auc_pr': single_fold_res['auc_pr'],
                'std_auc_pr': 0.0,
                'mean_peak_vram_mb': 0.0,
                'folds': [single_fold_res],
                'best_fold': {'speaker': val_spk_str, 'auc_pr': auc_pr},
                'epoch_history': history or [],
            }
            import tempfile
            import os
            result_path = base_log_dir / "loocv_results.json"
            base_log_dir.mkdir(parents=True, exist_ok=True)
            tmp_path = result_path.with_suffix('.json.tmp')
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4)
            os.replace(tmp_path, result_path)
            print(f"[INFO] Single fold results saved to {result_path} for ablation compatibility.")

if __name__ == "__main__":
    main()
