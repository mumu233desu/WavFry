import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import torchaudio
from wavfry.data.augmentation import AudioAugmentor, generate_pink_noise

def test_augmentation():
    sr = 16000
    duration = 2.0
    T = int(sr * duration)
    
    # 1. Create a dummy sine wave (16kHz, 2s, B=1)
    t = torch.linspace(0, duration, T)
    waveform = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # Shape [1, T]
    
    print("--- 1. Testing Pink Noise Utility directly ---")
    # Generate pink noise directly
    pink = generate_pink_noise(B=1, T=T, device='cpu', dtype=torch.float32, low_cutoff_hz=10.0, sample_rate=sr)
    
    # Check DC offset
    dc_offset = pink.mean().item()
    print(f"Pink Noise Mean (DC Offset): {dc_offset:.8f} (Expected: exactly 0.0)")
    assert abs(dc_offset) < 1e-7, f"DC offset is too high: {dc_offset}"
    
    # Check energy concentration (lower half of spectrum vs upper half)
    pink_fft = torch.fft.rfft(pink[0])
    pink_psd = torch.abs(pink_fft) ** 2
    
    num_freqs = len(pink_psd)
    lower_half_pwr = pink_psd[1:num_freqs//2].sum().item()
    upper_half_pwr = pink_psd[num_freqs//2:].sum().item()
    pwr_ratio = lower_half_pwr / (upper_half_pwr + 1e-9)
    print(f"Pink Noise Spectral Power Ratio (Lower/Upper): {pwr_ratio:.4f} (Expected: >> 1.0, e.g. > 5.0)")
    assert pwr_ratio > 5.0, f"Spectral slope check failed, ratio: {pwr_ratio}"
    
    # Compare with White Noise
    white = torch.randn(1, T)
    white_fft = torch.fft.rfft(white[0])
    white_psd = torch.abs(white_fft) ** 2
    white_ratio = white_psd[1:num_freqs//2].sum().item() / (white_psd[num_freqs//2:].sum().item() + 1e-9)
    print(f"White Noise Spectral Power Ratio (Lower/Upper): {white_ratio:.4f} (Expected: ~1.0)")
    assert 0.8 < white_ratio < 1.3, f"White noise ratio should be close to 1.0, got: {white_ratio}"

    print("\n--- 2. Testing AudioAugmentor Integration ---")
    # Configuration with White Noise
    config_white = {
        'enabled': True,
        'aug_order': 'noise_first',
        'schedule_power': 2.0,
        'max_epochs': 20,
        'denoising': {
            'enabled': False
        },
        'noise': {
            'enabled': True,
            'prob': 1.0,
            'type': 'white',
            'snr_db_start': 40.0,
            'snr_db_end': 10.0
        }
    }
    
    # Configuration with Pink Noise
    config_pink = {
        'enabled': True,
        'aug_order': 'noise_first',
        'schedule_power': 2.0,
        'max_epochs': 20,
        'denoising': {
            'enabled': False
        },
        'noise': {
            'enabled': True,
            'prob': 1.0,
            'type': 'pink',
            'pink_low_cutoff_hz': 10.0,
            'snr_db_start': 40.0,
            'snr_db_end': 10.0
        }
    }
    
    aug_white = AudioAugmentor(config_white)
    aug_pink = AudioAugmentor(config_pink)
    
    # Test Epoch 1 (Should be almost clean, SNR=40dB)
    out_white_ep1 = aug_white.apply(waveform.clone(), epoch=1, device='cpu')
    out_pink_ep1 = aug_pink.apply(waveform.clone(), epoch=1, device='cpu')
    
    diff_white_ep1 = torch.abs(out_white_ep1 - waveform).mean().item()
    diff_pink_ep1 = torch.abs(out_pink_ep1 - waveform).mean().item()
    print(f"Epoch 1 Mean Diff - White: {diff_white_ep1:.6f}, Pink: {diff_pink_ep1:.6f} (Expect very small)")
    
    # Test Epoch 20 (Should be noisy, SNR=10dB)
    out_white_ep20 = aug_white.apply(waveform.clone(), epoch=20, device='cpu')
    out_pink_ep20 = aug_pink.apply(waveform.clone(), epoch=20, device='cpu')
    
    diff_white_ep20 = torch.abs(out_white_ep20 - waveform).mean().item()
    diff_pink_ep20 = torch.abs(out_pink_ep20 - waveform).mean().item()
    print(f"Epoch 20 Mean Diff - White: {diff_white_ep20:.6f}, Pink: {diff_pink_ep20:.6f} (Expect much larger)")
    
    # Check that SNR logic is properly functioning: White and Pink Epoch 20 difference magnitudes should be comparable
    ratio_diff = diff_pink_ep20 / (diff_white_ep20 + 1e-9)
    print(f"Scale/SNR comparability ratio (Pink Diff / White Diff): {ratio_diff:.4f} (Expect ~ 1.0 ± 0.3)")
    assert 0.7 < ratio_diff < 1.4, f"SNR scaling mismatch: {ratio_diff}"
    
    # Save audio files
    output_dir = Path("artifacts/aug_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torchaudio.save(output_dir / "original.wav", waveform, sr)
    torchaudio.save(output_dir / "white_epoch_1.wav", out_white_ep1, sr)
    torchaudio.save(output_dir / "white_epoch_20.wav", out_white_ep20, sr)
    torchaudio.save(output_dir / "pink_epoch_1.wav", out_pink_ep1, sr)
    torchaudio.save(output_dir / "pink_epoch_20.wav", out_pink_ep20, sr)
    
    print(f"\nAll verification tests passed successfully!")
    print(f"Test wav files saved to {output_dir.resolve()}")

if __name__ == "__main__":
    test_augmentation()
