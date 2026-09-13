import argparse
import torch
import yaml
from pathlib import Path
import sys

# Add 06_wavfry to path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from wavfry.model import wavlm_classifier as model_mod

class ONNXExportWrapper(torch.nn.Module):
    def __init__(self, model, task="creaky"):
        super().__init__()
        self.model = model
        self.task = task
        
    def forward(self, waveform, lengths):
        out = self.model(waveform, lengths)
        if isinstance(out, dict):
            if self.task != "all" and self.task in out:
                return out[self.task]
            # Return tuple for all tasks in order
            return tuple(out[t] for t in self.model.tasks)
        return out

def export_to_onnx(model, output_path, task="creaky"):
    print(f"Exporting model to ONNX: {output_path} (task: {task})...")
    model.eval()
    
    wrapper = ONNXExportWrapper(model, task=task)
    wrapper.eval()
    
    tasks = getattr(model, 'tasks', ['creaky'])
    if task != "all" and task in tasks:
        output_names = ['logits']
    else:
        output_names = [f'logits_{t}' for t in tasks]

    dynamic_axes = {
        'waveform': {0: 'batch_size', 1: 'sequence_length'},
        'lengths': {0: 'batch_size'},
    }
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    dummy_waveform = torch.randn(1, 16000, device=next(model.parameters()).device)
    dummy_lengths = torch.tensor([16000], dtype=torch.long, device=dummy_waveform.device)
    
    try:
        torch.onnx.export(
            wrapper,
            (dummy_waveform, dummy_lengths),
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['waveform', 'lengths'],
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print(f"Successfully exported ONNX model to {output_path}")
        print(f"File size: {Path(output_path).stat().st_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

def main():
    parser = argparse.ArgumentParser(description="Export WavFry delta checkpoint to full ONNX model")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the delta .pt checkpoint")
    parser.add_argument('--output-dir', type=str, default=str(root / 'checkpoints'), help="Directory to save the exported models")
    parser.add_argument('--onnx-name', type=str, default="wavfry_model.onnx", help="Name of the exported ONNX file")
    parser.add_argument('--pt-name', type=str, default="wavfry_full.pt", help="Name of the exported full PyTorch file")
    parser.add_argument('--task', type=str, default="creaky", help="Task to export ('creaky', 'breathy', or 'all'). Default: 'creaky'")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except Exception:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = ckpt['config']
    
    print(f"Initializing WavFryClassifier with backbone: {config['model']['backbone']}...")
    # This will trigger a download of the base model from HF if not cached
    model = model_mod.WavFryClassifier(config)
    
    print("Applying delta weights...")
    if config['checkpoint'].get('save_mode', 'delta') == 'delta':
        missing_keys, unexpected_keys = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded delta weights. Expected {len(missing_keys)} missing keys (base model layers).")
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        print("Loaded full weights.")
        
    model.eval()

    # 1. Export Full PyTorch Model
    full_pt_path = out_dir / args.pt_name
    print(f"\nSaving full PyTorch model to: {full_pt_path}...")
    best_threshold = ckpt.get('best_threshold', ckpt.get('metrics', {}).get('best_threshold', 0.5))
    threshold_f1 = ckpt.get('threshold_f1', ckpt.get('metrics', {}).get('threshold_f1', None))
    full_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_threshold': float(best_threshold),
        'threshold_f1': float(threshold_f1) if threshold_f1 is not None else None,
        'info': 'Exported full model (base + delta)'
    }
    # Update config to indicate this is now a full model
    full_checkpoint['config']['checkpoint']['save_mode'] = 'full'
    if 'inference' not in full_checkpoint['config']:
        full_checkpoint['config']['inference'] = {}
    full_checkpoint['config']['inference']['threshold'] = float(best_threshold)
    torch.save(full_checkpoint, full_pt_path)
    print(f"File size: {full_pt_path.stat().st_size / (1024 * 1024):.2f} MB")

    # 2. Export ONNX Model
    onnx_path = out_dir / args.onnx_name
    export_to_onnx(model, onnx_path, task=args.task)

if __name__ == "__main__":
    main()
