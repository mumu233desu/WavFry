import sys
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
import time

# Add 06_wavfry to path
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from wavfry.model import wavlm_classifier as model_mod

def test_export():
    pt_path = root / 'checkpoints' / 'wavfry_full.pt'
    onnx_path = root / 'checkpoints' / 'wavfry_model.onnx'
    
    print("1. Loading PyTorch model...")
    ckpt = torch.load(pt_path, map_location='cpu', weights_only=True)
    config = ckpt['config']
    pt_model = model_mod.WavFryClassifier(config)
    pt_model.load_state_dict(ckpt['model_state_dict'])
    pt_model.eval()
    
    print("2. Loading ONNX model...")
    # Explicitly test CPU provider to ensure exact deterministic math
    sess = ort.InferenceSession(str(onnx_path), providers=['CPUExecutionProvider'])
    
    # 3. Create dummy input (e.g. 0.5 seconds at 16kHz)
    np.random.seed(42)
    # Scale to [-1, 1] like real audio
    dummy_audio_np = np.random.uniform(-1.0, 1.0, size=(1, 8000)).astype(np.float32)
    
    pt_audio = torch.from_numpy(dummy_audio_np)
    pt_lengths = torch.tensor([8000], dtype=torch.long)
    
    print("\nRunning inference...")
    
    # PyTorch inference
    t0 = time.time()
    with torch.no_grad():
        pt_logits = pt_model(pt_audio, pt_lengths)
        pt_probs = torch.sigmoid(pt_logits).numpy()
    pt_time = time.time() - t0
    
    # ONNX inference
    inputs = {
        'waveform': dummy_audio_np,
        'lengths': np.array([8000], dtype=np.int64)
    }
    t0 = time.time()
    onnx_outputs = sess.run(['logits'], inputs)
    onnx_logits = onnx_outputs[0]
    
    # Sigmoid
    onnx_probs = 1.0 / (1.0 + np.exp(-onnx_logits))
    onnx_time = time.time() - t0
    
    print(f"\nPyTorch Time: {pt_time*1000:.2f} ms")
    print(f"ONNX Time:    {onnx_time*1000:.2f} ms")
    
    print(f"\nPyTorch Logit: {pt_logits.item():.6f} | Prob: {pt_probs.item():.6f}")
    print(f"ONNX Logit:    {onnx_logits.item():.6f} | Prob: {onnx_probs.item():.6f}")
    
    # Calculate MSE
    mse = np.mean((pt_probs - onnx_probs) ** 2)
    max_diff = np.max(np.abs(pt_probs - onnx_probs))
    
    print(f"\nMean Squared Error (MSE): {mse:.2e}")
    print(f"Max Absolute Difference:  {max_diff:.2e}")
    
    if mse < 1e-5:
        print("\n[SUCCESS] ONNX export verified! The math matches PyTorch perfectly.")
    else:
        print("\n[FAILED] ONNX outputs differ significantly from PyTorch.")

if __name__ == "__main__":
    test_export()
