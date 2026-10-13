import math
import random
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import torchaudio.functional as F

from wavfry.data.augmentation import (
    AudioAugmentor,
    RealWorldNoisePool,
    ReverbAugmentor,
    generate_pink_noise,
)
from wavfry.data.paradigm_block import ContinuousAudioBlockDataset, collate_block_fn


def test_qut_metadata_and_files():
    """Verify that processed FLAC files and metadata exist and are valid."""
    meta_path = Path("E:/Corpus/Noise/QUT-NOISE_processed/metadata.json")
    if not meta_path.exists():
        pytest.skip("Processed QUT-NOISE metadata not found on this machine.")

    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["target_sample_rate"] == 16000
    assert "noise" in meta and len(meta["noise"]) == 5  # cafe, car, home, reverb, street
    assert "impulses" in meta and len(meta["impulses"]) == 2  # car, reverb

    total_noise_files = sum(len(files) for files in meta["noise"].values())
    assert total_noise_files == 20

    total_rirs = sum(len(files) for files in meta["impulses"].values())
    assert total_rirs == 8

    # Verify each RIR impulse on disk
    for scenario, rirs in meta["impulses"].items():
        for item in rirs:
            p = Path(item["path"])
            assert p.exists(), f"RIR file missing: {p}"
            info = sf.info(str(p))
            assert info.samplerate == 16000
            assert info.channels == 1
            data, _ = sf.read(str(p))
            energy = np.sum(data**2)
            assert abs(energy - 1.0) < 1e-3, f"Energy not normalized: {energy}"
            assert np.max(np.abs(data)) <= 1.0


def test_reverb_padded_slice_equivalence():
    """
    Verify mathematical identity:
    Pre-padded slice convolution is identical to whole-utterance convolution and slicing.
    """
    sr = 16000
    duration_s = 6.0
    full_wav = torch.randn(int(duration_s * sr)).float()

    # Create dummy RIR (0.5s = 8000 samples)
    rir = torch.randn(8000).float()
    rir = rir / torch.sqrt(torch.sum(rir**2))

    cfg = {"enabled": True, "prob": 1.0}
    augmentor = ReverbAugmentor(cfg)
    augmentor.rirs = [rir]

    # Method 1: Convolve whole utterance, then slice
    w_2d = full_wav.unsqueeze(0)
    r_2d = rir.unsqueeze(0)
    conv_full = F.fftconvolve(w_2d, r_2d, mode="full").squeeze(0)

    start_frame = int(2.0 * sr)
    end_frame = int(4.0 * sr)
    expected_slice = conv_full[start_frame:end_frame]

    # Method 2: ReverbAugmentor.apply_slice
    actual_slice = augmentor.apply_slice(full_wav, start_frame, end_frame, seed=123)

    diff = (expected_slice - actual_slice).abs().max().item()
    assert diff < 1e-5, f"Padded slice mismatch: max diff = {diff}"


def test_reverb_zero_padding_at_utterance_start():
    """Verify that slices starting at 0.0s handle left boundary zero-padding correctly."""
    sr = 16000
    full_wav = torch.randn(int(4.0 * sr)).float()
    rir = torch.randn(4000).float()
    rir = rir / torch.sqrt(torch.sum(rir**2))

    cfg = {"enabled": True, "prob": 1.0}
    augmentor = ReverbAugmentor(cfg)
    augmentor.rirs = [rir]

    conv_full = F.fftconvolve(full_wav.unsqueeze(0), rir.unsqueeze(0), mode="full").squeeze(0)
    expected_slice = conv_full[0 : int(2.0 * sr)]

    actual_slice = augmentor.apply_slice(full_wav, 0, int(2.0 * sr), seed=42)
    diff = (expected_slice - actual_slice).abs().max().item()
    assert diff < 1e-5


def test_reverb_determinism_and_prob():
    """Verify probabilistic trigger and determinism across seeds."""
    full_wav = torch.randn(32000).float()
    rir = torch.randn(2000).float()
    cfg = {"enabled": True, "prob": 0.5}
    augmentor = ReverbAugmentor(cfg, seed=100)
    augmentor.rirs = [rir]

    # Identical seed -> identical output
    out1 = augmentor.apply_slice(full_wav, 0, 16000, seed=999)
    out2 = augmentor.apply_slice(full_wav, 0, 16000, seed=999)
    assert torch.equal(out1, out2)

    # Prob = 0.0 -> clean waveform unchanged
    augmentor_off = ReverbAugmentor({"enabled": True, "prob": 0.0})
    augmentor_off.rirs = [rir]
    out_clean = augmentor_off.apply_slice(full_wav, 0, 16000, seed=999)
    assert torch.equal(out_clean, full_wav[0:16000])


def test_real_world_noise_pool_sampling():
    """Verify RealWorldNoisePool in-memory slicing and determinism."""
    noise_dir = Path("E:/Corpus/Noise/QUT-NOISE_processed/noise")
    if not noise_dir.exists():
        pytest.skip("Processed noise directory not found.")

    pool = RealWorldNoisePool(noise_dir=noise_dir, scenarios=["cafe", "car"])
    assert len(pool.buffers) > 0

    batch_size = 4
    length = 32000  # 2.0s
    device = torch.device("cpu")

    # Sample slices
    slices1 = pool.sample_slices(batch_size, length, device, torch.float32, seed=12345)
    assert slices1.shape == (batch_size, length)
    assert slices1.dtype == torch.float32
    assert slices1.min() >= -1.0 and slices1.max() <= 1.0

    # Test determinism
    slices2 = pool.sample_slices(batch_size, length, device, torch.float32, seed=12345)
    assert torch.equal(slices1, slices2)

    # Different seed should yield different slice
    slices3 = pool.sample_slices(batch_size, length, device, torch.float32, seed=54321)
    assert not torch.equal(slices1, slices3)


def test_audio_augmentor_multi_source_and_snr():
    """Verify AudioAugmentor dynamic SNR scaling with mixed noise sources."""
    cfg = {
        "enabled": True,
        "max_epochs": 10,
        "noise": {
            "enabled": True,
            "prob": 1.0,  # 100% add noise
            "sources": {"pink": 0.5, "real_world": 0.5},
            "real_world_dir": "E:/Corpus/Noise/QUT-NOISE_processed/noise",
            "scenarios": ["cafe", "car", "home", "reverb", "street"],
            "snr_db_start": 20.0,
            "snr_db_end": 10.0,
        },
    }
    augmentor = AudioAugmentor(cfg, max_epochs=10, seed=42)

    B, T = 8, 32000
    # Clean sine wave signal with known power
    t = torch.linspace(0, 2.0, T)
    clean_waveforms = torch.sin(2 * math.pi * 440 * t).unsqueeze(0).repeat(B, 1)

    epoch = 1  # Target SNR = 20 dB
    noisy = augmentor.apply(clean_waveforms, epoch=epoch, device=torch.device("cpu"), step=1)
    assert noisy.shape == (B, T)
    assert not torch.isnan(noisy).any()

    # Verify added noise power
    added_noise = noisy - clean_waveforms
    sig_power = torch.mean(clean_waveforms**2, dim=1)
    noise_power = torch.mean(added_noise**2, dim=1)
    measured_snr_db = 10 * torch.log10(sig_power / (noise_power + 1e-9))

    for b in range(B):
        # Measured SNR should be within 0.2 dB of target (20 dB)
        assert abs(measured_snr_db[b].item() - 20.0) < 0.2, (
            f"Sample {b} SNR mismatch: {measured_snr_db[b].item():.2f} dB vs 20.0 dB"
        )


def test_dataset_reverb_integration():
    """Verify ContinuousAudioBlockDataset integration with ReverbAugmentor."""
    import pandas as pd

    # Mock audio file
    dummy_wav_path = Path("test_dummy_speech.wav")
    dummy_data = (np.random.randn(48000) * 0.1).astype(np.float32)
    sf.write(str(dummy_wav_path), dummy_data, 16000)

    try:
        df = pd.DataFrame(
            [
                {
                    "file_key": "dummy_1",
                    "audio_path": str(dummy_wav_path),
                    "speaker_id": "SPK1",
                    "start_s": 0.5,
                    "end_s": 1.5,
                    "label_creaky": 1.0,
                    "label_breathy": 0.0,
                }
            ]
        )

        rir = torch.randn(1600).float()
        rir = rir / torch.sqrt(torch.sum(rir**2))
        reverb_aug = ReverbAugmentor({"enabled": True, "prob": 1.0})
        reverb_aug.rirs = [rir]

        ds = ContinuousAudioBlockDataset(
            df,
            duration_ms=2000.0,
            hop_ms=1000.0,
            reverb_augmentor=reverb_aug,
            seed=42,
        )

        assert len(ds) > 0
        batch_sample = ds[0]
        waveform = batch_sample[0]
        assert waveform.shape[0] == 32000
        assert not torch.isnan(waveform).any()
    finally:
        if dummy_wav_path.exists():
            dummy_wav_path.unlink()
