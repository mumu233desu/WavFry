"""
Unit tests for the unified acoustic degradation evaluation engine.
"""

import numpy as np
import pytest
import torch

from wavfry.eval.acoustic_degradation import (
    AcousticDegradationEngine,
    CANONICAL_EVAL_CONDITIONS,
    CANONICAL_NOISE_TYPES,
    CANONICAL_SNR_LADDER,
    PARADIGM_A_CONDITIONS,
    PARADIGM_A_NOISE_TYPES,
    PARADIGM_A_SNR_LADDER,
    parse_condition_str,
)


def test_parse_condition_str():
    # Clean
    c1 = parse_condition_str("clean")
    assert c1.is_clean is True
    assert c1.has_reverb is False

    # Standard noise
    c2 = parse_condition_str("cafe_15db")
    assert c2.is_clean is False
    assert c2.noise_type == "cafe"
    assert c2.snr_db == 15.0
    assert c2.has_reverb is False

    # Negative SNR
    c3 = parse_condition_str("car_-5db")
    assert c3.noise_type == "car"
    assert c3.snr_db == -5.0

    # Reverb only
    c4 = parse_condition_str("reverb")
    assert c4.has_reverb is True
    assert c4.noise_type is None

    # Authentic compound conditions (RIR strictly scenario-matched, default 15dB SNR)
    c_room = parse_condition_str("reverb_compound")
    assert c_room.has_reverb is True
    assert c_room.noise_type == "reverb"
    assert c_room.snr_db == 15.0

    c_car = parse_condition_str("car_compound")
    assert c_car.has_reverb is True
    assert c_car.noise_type == "car"
    assert c_car.snr_db == 15.0

    # Explicit compound SNR override
    c_car_5 = parse_condition_str("car_compound_5db")
    assert c_car_5.has_reverb is True
    assert c_car_5.noise_type == "car"
    assert c_car_5.snr_db == 5.0


def test_canonical_constants():
    assert len(CANONICAL_NOISE_TYPES) == 5
    for expected in ["cafe", "car", "home", "reverb", "street"]:
        assert expected in CANONICAL_NOISE_TYPES
    assert CANONICAL_SNR_LADDER == [15.0, 5.0, 0.0]
    assert len(CANONICAL_EVAL_CONDITIONS) == 18
    assert "clean" in CANONICAL_EVAL_CONDITIONS
    assert "car_compound" in CANONICAL_EVAL_CONDITIONS
    assert "reverb_compound" in CANONICAL_EVAL_CONDITIONS
    # Check backward compatibility
    assert len(PARADIGM_A_CONDITIONS) == 18


def test_rir_scenario_strict_matching():
    engine = AcousticDegradationEngine()
    x = torch.randn(2, 16000)

    # Valid scenario-matched combinations
    out_car = engine.apply_tensor(x, "car_compound", seed=42)
    assert out_car.shape == x.shape
    assert not torch.all(out_car == x)

    out_room = engine.apply_tensor(x, "room_compound", seed=42)
    assert out_room.shape == x.shape
    assert not torch.all(out_room == x)

    # Pure RIR conditions
    out_reverb = engine.apply_tensor(x, "reverb", seed=42)
    assert out_reverb.shape == x.shape
    out_car_rir = engine.apply_tensor(x, "car_rir", seed=42)
    assert out_car_rir.shape == x.shape

    # Unphysical cross-scenario RIR mixing must raise ValueError
    with pytest.raises(ValueError, match="Physical Acoustic Invariant Violation"):
        engine.apply_tensor(x, "reverb_cafe_5db", seed=42)


def test_apply_tensor_clean_noop():
    engine = AcousticDegradationEngine()
    x = torch.randn(4, 16000)
    out = engine.apply_tensor(x, "clean")
    assert torch.all(x == out)


def test_apply_tensor_snr_accuracy():
    engine = AcousticDegradationEngine()
    # 1 kHz sine wave
    t = torch.linspace(0, 1.0, 16000)
    signal = torch.sin(2 * np.pi * 1000 * t).unsqueeze(0)  # (1, 16000)
    sig_pwr = torch.mean(signal ** 2).item()

    target_snr = 10.0
    # Pink or white noise
    out = engine.apply_tensor(signal, f"pink_{target_snr}db", seed=42)
    noise_part = out - signal
    noise_pwr = torch.mean(noise_part ** 2).item()

    measured_snr = 10.0 * np.log10(sig_pwr / noise_pwr)
    assert pytest.approx(target_snr, abs=0.5) == measured_snr


def test_apply_tensor_reverb_shape_and_sound():
    engine = AcousticDegradationEngine()
    x = torch.randn(2, 32000)
    out = engine.apply_tensor(x, "reverb", seed=123)
    assert out.shape == x.shape
    # Must not be all zeros and must not be identical to original
    assert not torch.all(out == x)
    assert not torch.all(out == 0)


def test_apply_numpy_interface():
    engine = AcousticDegradationEngine()
    audio_np = np.random.randn(16000).astype(np.float32)
    out_np = engine.apply_numpy(audio_np, sr=16000, condition="cafe_5db", seed=99)

    assert isinstance(out_np, np.ndarray)
    assert out_np.shape == audio_np.shape
    assert out_np.dtype == np.float32
    assert not np.all(out_np == audio_np)


def test_deterministic_reproducibility():
    engine = AcousticDegradationEngine()
    x = torch.randn(2, 16000)

    out1 = engine.apply_tensor(x, "street_5db", seed=777)
    out2 = engine.apply_tensor(x, "street_5db", seed=777)

    assert torch.all(out1 == out2)


def test_causal_rir_convolution_impulse_response():
    engine = AcousticDegradationEngine()
    # Delta impulse signal
    delta = torch.zeros(1, 16000)
    delta[0, 0] = 1.0

    out = engine.apply_tensor(delta, "reverb", seed=42)
    # The output at t=0 should be non-zero (direct sound arrival)
    assert abs(out[0, 0].item()) > 1e-4

    # Early energy (first 1000 samples) must be significantly higher than late energy (last 1000 samples)
    # Under time-reversed convolution, late energy would be high and early energy low!
    early_energy = torch.sum(out[0, :1000] ** 2).item()
    late_energy = torch.sum(out[0, 10000:11000] ** 2).item()
    assert early_energy > late_energy * 2.0


def test_global_torch_seed_not_polluted():
    engine = AcousticDegradationEngine()
    x = torch.randn(1, 16000)

    # Calling degradation with a seed should not reset global torch.randn sequence
    torch.manual_seed(999)
    val_before = torch.randn(10)

    torch.manual_seed(999)
    # Call degradation
    _ = engine.apply_tensor(x, "pink_10db", seed=12345)
    val_after = torch.randn(10)

    # Should match because global PRNG was not mutated
    assert torch.allclose(val_before, val_after)
