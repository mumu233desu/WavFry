import argparse
import yaml
import torch
import sys
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
import copy

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

def run_training(config, df, device, dataset_mod, splits_mod, model_mod, trainer_mod, fold_idx=0, preloaded_wavlm=None, resume=False):
    # Set seed for this fold/run (seed + fold_idx for variety in LOOCV)
    set_seed(config['training'].get('seed', 42) + fold_idx)
    
    # Split
    val_speakers = config['data']['val_speakers']
    train_df, val_df = splits_mod.make_speaker_split(df, val_speakers)
    print(f"Train: {len(train_df)} segments, Val: {len(val_df)} segments (Speaker: {val_speakers})")
    
    # Datasets
    augmentor = None
    aug_cfg = config.get('augmentation', {})
    if aug_cfg.get('enabled', False):
        from wavfry.data.augmentation import AudioAugmentor
        augmentor = AudioAugmentor(aug_cfg)
        print("Data augmentation enabled for training.")

    lowpass_cfg = config.get('preprocessing', {}).get('lowpass', {})
    train_ds = dataset_mod.CreakyVowelDataset(
        train_df, 
        context_padding_ms=config['data']['context_padding_ms'],
        lowpass_cfg=lowpass_cfg
    )
    val_ds = dataset_mod.CreakyVowelDataset(
        val_df, 
        context_padding_ms=config['data']['context_padding_ms'],
        lowpass_cfg=lowpass_cfg
    )
    
    num_workers = dataset_mod.resolve_num_workers(config['data']['num_workers'])
    prefetch_factor = config['data'].get('prefetch_factor', 2) if num_workers > 0 else None
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=True, 
        num_workers=num_workers,
        collate_fn=dataset_mod.collate_fn,
        pin_memory=config['data']['pin_memory'],
        prefetch_factor=prefetch_factor
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config['training']['batch_size'], 
        shuffle=False, 
        num_workers=num_workers,
        collate_fn=dataset_mod.collate_fn,
        pin_memory=config['data']['pin_memory'],
        prefetch_factor=prefetch_factor
    )
    
    # Model
    if preloaded_wavlm is not None:
        # Fast path: Deepcopy the preloaded backbone to avoid redundant loading
        model = model_mod.WavFryClassifier(config, wavlm_instance=copy.deepcopy(preloaded_wavlm)).to(device)
    else:
        model = model_mod.WavFryClassifier(config).to(device)
    
    # Trainer
    trainer = trainer_mod.Trainer(model, train_loader, val_loader, config, device, augmentor=augmentor)
    
    start_epoch = 1
    if resume:
        val_str = "_".join(val_speakers) if isinstance(val_speakers, list) else str(val_speakers)
        latest_path = Path(config['checkpoint']['dir']) / f"latest_{val_str}.pt"
        if latest_path.exists():
            try:
                last_epoch = trainer.load_checkpoint(latest_path)
                start_epoch = last_epoch + 1
                print(f"Resuming {val_speakers} from epoch {start_epoch}")
            except Exception as e:
                print(f"Failed to load resume checkpoint for {val_speakers}: {e}. Starting from scratch.")

    try:
        trainer.run(start_epoch=start_epoch)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Gracefully exiting...")
    finally:
        trainer.writer.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=str(root / 'config' / '_defaults.yaml'))
    parser.add_argument('--override', type=str, nargs='*', help='Override config keys: key.subkey=value')
    parser.add_argument('--loocv', action='store_true', help='Run Leave-One-Speaker-Out Cross Validation')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Apply overrides
    if args.override:
        for item in args.override:
            if '=' in item:
                key_path, v = item.split('=', 1)
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
                
                # Navigate and set
                keys = key_path.split('.')
                curr = config
                for k in keys[:-1]:
                    if k not in curr: curr[k] = {}
                    curr = curr[k]
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
    
    # Load manifest
    manifest_path = root / "manifests" / "segments_all.csv"
    if not manifest_path.exists():
        print(f"Manifest not found at {manifest_path}. Please run build_manifest.py first.")
        sys.exit(1)
        
    import csv
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        data = list(reader)
    pd.set_option("mode.string_storage", "python")
    df = pd.DataFrame(data)
    # Convert numeric columns
    for col in ['start_s', 'end_s', 'label']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col])
    
    val_speakers_cfg = config['data'].get('val_speakers', [])
    is_loocv = (val_speakers_cfg == 'loocv') or args.loocv
    
    if is_loocv:
        # Get all annotated speakers
        speakers = config['data']['annotated_speakers']
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
        for i, val_speaker in enumerate(speakers):
            print(f"\n{'='*30}\nFold: {val_speaker}\n{'='*30}")
            
            # Check if this fold is already completed (for resume)
            fold_final_path = base_ckpt_dir / f"best_model_{val_speaker}.pt"
            if args.resume and fold_final_path.exists():
                print(f"Fold for {val_speaker} already completed. Loading results and skipping.")
                # Use weights_only=True for security since it's just metrics
                ckpt = torch.load(fold_final_path, map_location='cpu', weights_only=True)
                metrics = ckpt.get('metrics', {'f1': ckpt.get('f1', 0)})
                fold_results.append({
                    'speaker': val_speaker,
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': metrics.get('f1', 0),
                    'auc_pr': metrics.get('auc_pr', ckpt.get('auc_pr', 0)),
                    'best_threshold': metrics.get('best_threshold', ckpt.get('best_threshold', 0.5)),
                    'threshold_f1': metrics.get('threshold_f1', ckpt.get('threshold_f1', 0)),
                    'epoch': ckpt.get('epoch', 0)
                })
                if metrics.get('auc_pr', ckpt.get('auc_pr', 0)) > global_best_f1:
                    global_best_f1 = metrics.get('auc_pr', ckpt.get('auc_pr', 0))
                    global_best_speaker = val_speaker
                continue
            
            fold_config = copy.deepcopy(config)
            fold_config['data']['val_speakers'] = [val_speaker]
            # Logs are still best kept in subdirs for TB comparison
            fold_config['tensorboard']['log_dir'] = str(base_log_dir / f"fold_{val_speaker}")
            # Checkpoints will go to the same base dir, but with a speaker-specific name
            fold_config['checkpoint']['dir'] = str(base_ckpt_dir)
            
            # Clean up residual checkpoint from previous fold to avoid race conditions
            fold_best_path = base_ckpt_dir / "best_model.pt"
            if not args.resume and fold_best_path.exists():
                fold_best_path.unlink()
            
            run_training(fold_config, df, device, dataset_mod, splits_mod, model_mod, trainer_mod, 
                         fold_idx=i, preloaded_wavlm=preloaded_wavlm, resume=args.resume)
            
            # After training this fold, find the best model it saved
            if fold_best_path.exists():
                # Load to check F1
                ckpt = torch.load(fold_best_path, map_location='cpu', weights_only=True)
                metrics = ckpt.get('metrics', {'f1': ckpt.get('f1', 0)})
                f1 = metrics.get('f1', 0)
                auc_pr = metrics.get('auc_pr', ckpt.get('auc_pr', 0))
                best_threshold = metrics.get('best_threshold', ckpt.get('best_threshold', 0.5))
                threshold_f1 = metrics.get('threshold_f1', ckpt.get('threshold_f1', 0))
                
                fold_results.append({
                    'speaker': val_speaker,
                    'accuracy': metrics.get('accuracy', 0),
                    'precision': metrics.get('precision', 0),
                    'recall': metrics.get('recall', 0),
                    'f1': f1,
                    'auc_pr': auc_pr,
                    'best_threshold': best_threshold,
                    'threshold_f1': threshold_f1,
                    'epoch': ckpt.get('epoch', 0)
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
            else:
                print(f"Warning: Fold {val_speaker} did not produce a best_model.pt (F1 did not improve beyond 0.0 or training failed).")
            
            # Resource cleanup between folds
            import gc
            gc.collect()
            torch.cuda.empty_cache()
                    
        # Summary Report
        if fold_results:
            import numpy as np
            import json
            
            accs = [r['accuracy'] for r in fold_results]
            precs = [r['precision'] for r in fold_results]
            recs = [r['recall'] for r in fold_results]
            f1s = [r['f1'] for r in fold_results]
            auc_prs = [r.get('auc_pr', 0) for r in fold_results]
            
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
                'folds': fold_results,
                'best_fold': {'speaker': global_best_speaker, 'auc_pr': global_best_f1}
            }
            
            # Print to console
            print("\n" + "="*40)
            print("LOOCV SUMMARY REPORT")
            print("="*40)
            print(f"Mean Accuracy:  {report['mean_accuracy']:.4f} \u00b1 {report['std_accuracy']:.4f}")
            print(f"Mean Precision: {report['mean_precision']:.4f} \u00b1 {report['std_precision']:.4f}")
            print(f"Mean Recall:    {report['mean_recall']:.4f} \u00b1 {report['std_recall']:.4f}")
            print(f"Mean F1:        {report['mean_f1']:.4f} \u00b1 {report['std_f1']:.4f}")
            print(f"Mean PR-AUC:    {report['mean_auc_pr']:.4f} \u00b1 {report['std_auc_pr']:.4f}  [primary]")
            print("-" * 40)
            for r in fold_results:
                print(
                    f"Speaker: {r['speaker']:<10} | Acc: {r['accuracy']:.4f} | P: {r['precision']:.4f}"
                    f" | R: {r['recall']:.4f} | F1: {r['f1']:.4f}"
                    f" | PR-AUC: {r.get('auc_pr', 0):.4f}"
                    f" | Best-t: {r.get('best_threshold', 0.5):.3f} (F1={r.get('threshold_f1', 0):.4f})"
                )
            print("="*40)
            
            # Save to JSON
            result_path = base_log_dir / "loocv_results.json"
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=4)
                
            # Save to Markdown
            md_path = base_log_dir / "loocv_report.md"
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write("# LOOCV Training Report\n\n")
                f.write("## Summary Statistics\n")
                f.write(f"- **Mean PR-AUC** *(primary)*: {report['mean_auc_pr']:.4f} \u00b1 {report['std_auc_pr']:.4f}\n")
                f.write(f"- **Mean F1**: {report['mean_f1']:.4f} \u00b1 {report['std_f1']:.4f}\n")
                f.write(f"- **Mean Accuracy**: {report['mean_accuracy']:.4f} \u00b1 {report['std_accuracy']:.4f}\n")
                f.write(f"- **Mean Precision**: {report['mean_precision']:.4f} \u00b1 {report['std_precision']:.4f}\n")
                f.write(f"- **Mean Recall**: {report['mean_recall']:.4f} \u00b1 {report['std_recall']:.4f}\n\n")
                f.write("## Per-Fold Details\n\n")
                f.write("| Speaker | PR-AUC | F1 | Best Threshold | Threshold F1 | Accuracy | Precision | Recall | Best Epoch |\n")
                f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
                for r in fold_results:
                    f.write(
                        f"| {r['speaker']} | {r.get('auc_pr', 0):.4f} | {r['f1']:.4f}"
                        f" | {r.get('best_threshold', 0.5):.3f} | {r.get('threshold_f1', 0):.4f}"
                        f" | {r['accuracy']:.4f} | {r['precision']:.4f} | {r['recall']:.4f}"
                        f" | {r['epoch']} |\n"
                    )
                f.write(f"\n**Best Fold** *(by PR-AUC)*: {global_best_speaker} ({global_best_f1:.4f})\n")
            
            print(f"Results saved to {result_path} and {md_path}")
        else:
            print("\nLOOCV Complete. No valid folds were trained.")

    else:
        run_training(config, df, device, dataset_mod, splits_mod, model_mod, trainer_mod)

if __name__ == "__main__":
    main()
