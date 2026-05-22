import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torchaudio
from wavfry.data.augmentation import AudioAugmentor
import os
from pathlib import Path

def test_augmentation():
    # 1. Create a dummy sine wave (16kHz, 1s)
    sr = 16000
    t = torch.linspace(0, 1, sr)
    waveform = torch.sin(2 * torch.pi * 440 * t)
    
    config = {
        'enabled': True,
        'aug_order': 'noise_first',
        'schedule_power': 2.0,
        'denoising': {
            'enabled': True,
            'prob': 1.0,
            'atten_lim_db_start': 0.0,
            'atten_lim_db_end': 20.0
        },
        'noise': {
            'enabled': True,
            'prob': 1.0,
            'snr_db_start': 40.0,
            'snr_db_end': 10.0
        }
    }
    
    augmentor = AudioAugmentor(config)
    
    # Test Epoch 1 (Should be almost clean)
    augmentor.set_epoch(1, 10)
    out_ep1 = augmentor.apply(waveform.clone())
    diff_ep1 = torch.abs(out_ep1 - waveform).mean()
    print(f"Epoch 1 Mean Diff: {diff_ep1:.6f} (Expect small if noise is low)")
    
    # Test Epoch 10 (Should be noisy)
    augmentor.set_epoch(10, 10)
    out_ep10 = augmentor.apply(waveform.clone())
    diff_ep10 = torch.abs(out_ep10 - waveform).mean()
    print(f"Epoch 10 Mean Diff: {diff_ep10:.6f} (Expect larger)")
    
    # Save for manual verification
    output_dir = Path("artifacts/aug_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save(output_dir / "original.wav", waveform.unsqueeze(0), sr)
    torchaudio.save(output_dir / "epoch_1.wav", out_ep1.unsqueeze(0), sr)
    torchaudio.save(output_dir / "epoch_10.wav", out_ep10.unsqueeze(0), sr)
    
    print(f"Test files saved to {output_dir}")

if __name__ == "__main__":
    test_augmentation()
