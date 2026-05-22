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

def export_to_onnx(model, output_path):
    print(f"Exporting model to ONNX: {output_path}...")
    model.eval()
    
    # Dummy inputs for tracing
    # WavLM expects input of shape [Batch, Time]
    # We use a batch size of 1 and a dummy length of 16000 (1 second at 16kHz)
    dummy_waveform = torch.randn(1, 16000, device=next(model.parameters()).device)
    dummy_lengths = torch.tensor([16000], dtype=torch.long, device=dummy_waveform.device)
    
    try:
        torch.onnx.export(
            model,
            (dummy_waveform, dummy_lengths),
            output_path,
            export_params=True,
            opset_version=14,  # Opset 14 is generally well-supported and handles most transformer ops
            do_constant_folding=True,
            input_names=['waveform', 'lengths'],
            output_names=['logits'],
            dynamic_axes={
                'waveform': {0: 'batch_size', 1: 'sequence_length'},
                'lengths': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            }
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
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading checkpoint from: {args.checkpoint}")
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
    full_checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'info': 'Exported full model (base + delta)'
    }
    # Update config to indicate this is now a full model
    full_checkpoint['config']['checkpoint']['save_mode'] = 'full'
    torch.save(full_checkpoint, full_pt_path)
    print(f"File size: {full_pt_path.stat().st_size / (1024 * 1024):.2f} MB")

    # 2. Export ONNX Model
    onnx_path = out_dir / args.onnx_name
    export_to_onnx(model, onnx_path)

if __name__ == "__main__":
    main()
