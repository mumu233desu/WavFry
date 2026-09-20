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
    def __init__(self, model, task="all", export_frames=True):
        super().__init__()
        self.model = model
        self.task = task
        self.export_frames = export_frames
        
        tasks = getattr(model, 'tasks', ['creaky'])
        if task == "all" or task == "multi":
            self.active_tasks = list(tasks)
        elif task in tasks:
            self.active_tasks = [task]
        else:
            self.active_tasks = list(tasks)
            
        self.has_creaky_frame = False
        self.has_breathy_frame = False
        if export_frames:
            if hasattr(model, "frame_heads") and model.frame_heads is not None:
                self.has_creaky_frame = "creaky" in model.frame_heads and "creaky" in self.active_tasks
                self.has_breathy_frame = "breathy" in model.frame_heads and "breathy" in self.active_tasks
            elif hasattr(model, "frame_head") and model.frame_head is not None:
                self.has_creaky_frame = "creaky" in self.active_tasks

    def forward(self, waveform, lengths):
        # Pass attention_mask=None to wavlm to avoid static Where broadcast shape issues in ONNX runtime
        outputs = self.model.wavlm(waveform, attention_mask=None, output_hidden_states=True)
        hidden_by_task = {}
        if self.model.use_scalar_mix:
            task_weight_list = []
            for task in self.active_tasks:
                task_weight_list.append(torch.softmax(self.model.scalar_mix_weights[task], dim=0))
            task_weights = torch.stack(task_weight_list, dim=0)
            H = torch.stack(outputs.hidden_states, dim=0)
            num_layers, b_sz, t_sz, d_sz = H.shape
            flat_H = H.view(num_layers, -1)
            mixed = (task_weights.to(flat_H.dtype) @ flat_H).view(len(self.active_tasks), b_sz, t_sz, d_sz)
            for idx, task in enumerate(self.active_tasks):
                hidden_by_task[task] = mixed[idx]
        else:
            for task in self.active_tasks:
                hidden_by_task[task] = outputs.last_hidden_state

        first_seq = next(iter(hidden_by_task.values()))
        out_lengths = self.model.wavlm._get_feat_extract_output_lengths(lengths)
        out_lengths = out_lengths.clamp(min=1, max=first_seq.size(1))
        valid_mask = torch.arange(first_seq.size(1), device=waveform.device).unsqueeze(0) < out_lengths.unsqueeze(1)

        outputs_list = []
        for t in self.active_tasks:
            sequence = hidden_by_task[t]
            mask_f = valid_mask.unsqueeze(-1).to(sequence.dtype)
            if self.model.pooling_mode == "mean":
                counts = mask_f.sum(dim=1).clamp_min(1.0)
                pooled = (sequence * mask_f).sum(dim=1) / counts
            elif self.model.pooling_mode in ("mean_attn", "mean_topk"):
                counts = mask_f.sum(dim=1).clamp_min(1.0)
                mean_p = (sequence * mask_f).sum(dim=1) / counts
                attn_p = self.model._attn_pool(sequence, valid_mask, t)
                pooled = torch.cat([mean_p, attn_p], dim=-1)
            else:
                pooled = self.model._attn_pool(sequence, valid_mask, t)
            outputs_list.append(self.model.heads[t](pooled).squeeze(-1))

        if self.export_frames:
            if hasattr(self.model, "frame_heads") and self.model.frame_heads is not None:
                if self.has_creaky_frame:
                    outputs_list.append(self.model.frame_heads["creaky"](hidden_by_task["creaky"]).squeeze(-1))
                if self.has_breathy_frame:
                    outputs_list.append(self.model.frame_heads["breathy"](hidden_by_task["breathy"]).squeeze(-1))
            elif hasattr(self.model, "frame_head") and self.model.frame_head is not None:
                outputs_list.append(self.model.frame_head(hidden_by_task["creaky"]).squeeze(-1))

        if len(outputs_list) == 1:
            return outputs_list[0]
        return tuple(outputs_list)

def export_to_onnx(model, output_path, task="all", export_frames=True):
    print(f"Exporting model to ONNX: {output_path} (task: {task}, export_frames: {export_frames})...")
    model.eval()
    
    wrapper = ONNXExportWrapper(model, task=task, export_frames=export_frames).float()
    wrapper.eval()
    
    tasks = wrapper.active_tasks
    output_names = []
    if len(tasks) == 1 and not (wrapper.has_creaky_frame or wrapper.has_breathy_frame):
        output_names = ['logits']
    else:
        for t in tasks:
            output_names.append(f'{t}_logits')
        if wrapper.export_frames:
            if wrapper.has_creaky_frame:
                output_names.append('creaky_frame_logits')
            if wrapper.has_breathy_frame:
                output_names.append('breathy_frame_logits')

    dynamic_axes = {
        'waveform': {0: 'batch_size', 1: 'sequence_length'},
        'lengths': {0: 'batch_size'},
    }
    for name in output_names:
        if 'frame_logits' in name:
            dynamic_axes[name] = {0: 'batch_size', 1: 'num_frames'}
        else:
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
        print(f"Outputs: {output_names}")
        print(f"File size: {Path(output_path).stat().st_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Error during ONNX export: {e}")

def main():
    parser = argparse.ArgumentParser(description="Export WavFry delta checkpoint to full ONNX model")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the delta .pt checkpoint")
    parser.add_argument('--output-dir', type=str, default=str(root / 'exp_results' / 'checkpoints'), help="Directory to save the exported models")
    parser.add_argument('--onnx-name', type=str, default="wavfry_model.onnx", help="Name of the exported ONNX file")
    parser.add_argument('--pt-name', type=str, default="wavfry_full.pt", help="Name of the exported full PyTorch file")
    parser.add_argument('--task', type=str, default="all", help="Task to export ('all', 'creaky', or 'breathy'). Default: 'all'")
    parser.add_argument('--no-frames', action='store_true', help="Disable frame head export")
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
    model = model.float()
    if hasattr(model, "wavlm"):
        model.wavlm = model.wavlm.float()

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
    export_to_onnx(model, onnx_path, task=args.task, export_frames=not args.no_frames)

if __name__ == "__main__":
    main()
