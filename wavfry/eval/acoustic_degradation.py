"""
WavFry Unified Acoustic Degradation and Noise Evaluation Engine.

Provides centralized, reproducible, and standardized acoustic degradation
(Real-World QUT-NOISE, synthetic pink/white noise, and RIR reverberation)
for evaluation across models (WavFry, DeepFry, Creapy, Praat, rule-based baselines).

Supports both:
1. PyTorch Tensor batches (B, T) or (B, 1, T) for GPU/CPU deep learning models.
2. 1D NumPy arrays (T,) and temporary file generation for external baseline CLI tools.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Canonical Factorial Evaluation Protocol (5 Noise Scenarios x 3 SNRs + 2 Compounds + Clean)
CANONICAL_NOISE_TYPES: List[str] = ["cafe", "car", "home", "reverb", "street"]
CANONICAL_SNR_LADDER: List[float] = [15.0, 5.0, 0.0]

# Canonical 18 evaluation conditions:
# 1 Clean + 15 Factorial Noises (5 scenarios x 3 SNRs) + 2 Scenario-Matched Compounds @ 15dB
CANONICAL_EVAL_CONDITIONS: List[str] = [
    "clean",
    # 5 noise scenarios x 3 SNRs (15 dB, 5 dB, 0 dB)
    "cafe_15db",
    "cafe_5db",
    "cafe_0db",
    "car_15db",
    "car_5db",
    "car_0db",
    "home_15db",
    "home_5db",
    "home_0db",
    "reverb_15db",
    "reverb_5db",
    "reverb_0db",
    "street_15db",
    "street_5db",
    "street_0db",
    # 2 strictly scenario-matched physical compound environments @ 15dB
    "car_compound",
    "reverb_compound",
]

# Backward-compatible aliases
PARADIGM_A_NOISE_TYPES: List[str] = CANONICAL_NOISE_TYPES
PARADIGM_A_SNR_LADDER: List[float] = CANONICAL_SNR_LADDER
PARADIGM_A_CONDITIONS: List[str] = CANONICAL_EVAL_CONDITIONS

# Default directories for processed QUT-NOISE and RIRs
DEFAULT_NOISE_DIR = Path("E:/Corpus/Noise/QUT-NOISE_processed/noise")
DEFAULT_RIR_DIR = Path("E:/Corpus/Noise/QUT-NOISE_processed/impulses")


@dataclass
class ParsedCondition:
    raw_name: str
    is_clean: bool = False
    has_reverb: bool = False
    noise_type: Optional[str] = None  # 'cafe', 'car', 'street', 'home', 'reverb', 'pink', 'white'
    snr_db: Optional[float] = None


def parse_condition_str(condition: str) -> ParsedCondition:
    """
    Parse condition string into structured parameters following the Canonical Degradation Protocol.
    Examples:
        - 'clean' -> is_clean=True
        - 'cafe_15db' -> noise_type='cafe', snr_db=15.0
        - 'car_5db' -> noise_type='car', snr_db=5.0
        - 'home_0db' -> noise_type='home', snr_db=0.0
        - 'reverb_15db' -> noise_type='reverb', snr_db=15.0 (authentic swimming pool/carpark ambient noise)
        - 'street_5db' -> noise_type='street', snr_db=5.0
        - 'reverb' / 'room_rir' -> has_reverb=True (swimming pool & carpark RIR), noise_type=None
        - 'car_rir' -> has_reverb=True (car cabin RIR), noise_type='car', snr_db=None
        - 'car_compound' -> has_reverb=True (car cabin RIR), noise_type='car', snr_db=15.0 (or parsed if given)
        - 'reverb_compound' / 'room_compound' -> has_reverb=True (room RIR), noise_type='reverb', snr_db=15.0
    """
    norm = condition.strip().lower()
    if norm in ("clean", "none", "default"):
        return ParsedCondition(raw_name=condition, is_clean=True)

    # Extract SNR if present: e.g. '_15db', '15db', '-5db'
    snr_match = re.search(r"(-?\d+(?:\.\d+)?)db", norm)
    snr_db = float(snr_match.group(1)) if snr_match else None

    # 1. Canonical compound conditions (RIR + matched additive noise, default 15dB)
    if "compound" in norm:
        compound_snr = snr_db if snr_db is not None else 15.0
        if "car" in norm:
            return ParsedCondition(raw_name=condition, is_clean=False, has_reverb=True, noise_type="car", snr_db=compound_snr)
        elif any(k in norm for k in ("reverb", "room")):
            return ParsedCondition(raw_name=condition, is_clean=False, has_reverb=True, noise_type="reverb", snr_db=compound_snr)

    # 2. Pure RIR conditions (Spatial RIR only, no noise)
    if norm in ("reverb", "reverb_rir", "room_rir"):
        return ParsedCondition(raw_name=condition, is_clean=False, has_reverb=True, noise_type=None, snr_db=None)
    if norm == "car_rir":
        return ParsedCondition(raw_name=condition, is_clean=False, has_reverb=True, noise_type="car", snr_db=None)

    # 3. Identify additive noise type: cafe, car, street, home, reverb, pink, white
    noise_type = None
    for candidate in ("cafe", "car", "street", "home", "pink", "white", "reverb"):
        if candidate in norm:
            if candidate == "reverb" and any(other in norm for other in ("cafe", "car", "street", "home", "pink", "white")):
                continue
            noise_type = candidate
            break

    # If no noise type found but SNR found, default to pink noise
    if noise_type is None and snr_db is not None:
        noise_type = "pink"

    # has_reverb is only True if 'reverb' is explicitly in norm and it's not pure ambient noise
    has_reverb = False
    if "reverb" in norm and noise_type != "reverb":
        has_reverb = True

    return ParsedCondition(
        raw_name=condition,
        is_clean=False,
        has_reverb=has_reverb,
        noise_type=noise_type,
        snr_db=snr_db,
    )


def get_condition_metadata(condition: str) -> Dict[str, Any]:
    """
    Extract standardized LaTeX/TitleCase metadata dictionary for an acoustic condition.
    Returns:
        {
            "Condition": str,
            "Scenario": str,
            "SNR_dB": Optional[float],
            "Has_RIR": bool,
            "Tier": str,
        }
    """
    parsed = parse_condition_str(condition)
    if parsed.is_clean:
        return {
            "Condition": "clean",
            "Scenario": "clean",
            "SNR_dB": None,
            "Has_RIR": False,
            "Tier": "Clean",
        }

    scen = parsed.noise_type or ("car" if "car" in parsed.raw_name else "reverb")
    snr = parsed.snr_db
    has_rir = parsed.has_reverb

    if "compound" in parsed.raw_name or has_rir:
        tier = "Compound"
    elif snr is not None:
        if abs(snr - 15.0) < 1e-3:
            tier = "Mild"
        elif abs(snr - 5.0) < 1e-3:
            tier = "Moderate"
        elif abs(snr - 0.0) < 1e-3:
            tier = "Severe"
        else:
            tier = f"{snr:.0f}dB"
    else:
        tier = "Degraded"

    return {
        "Condition": parsed.raw_name,
        "Scenario": scen,
        "SNR_dB": snr,
        "Has_RIR": has_rir,
        "Tier": tier,
    }


class AcousticDegradationEngine:
    """
    Unified degradation engine managing noise pools, RIR convolution,
    and deterministic seeded audio perturbations.
    """

    def __init__(
        self,
        noise_dir: Optional[Union[str, Path]] = None,
        rir_dir: Optional[Union[str, Path]] = None,
        sample_rate: int = 16000,
        cache_in_memory: bool = True,
    ):
        self.sample_rate = sample_rate
        self.cache_in_memory = cache_in_memory

        self.noise_dir = Path(noise_dir) if noise_dir else DEFAULT_NOISE_DIR
        self.rir_dir = Path(rir_dir) if rir_dir else DEFAULT_RIR_DIR

        self._noise_pool: Dict[str, List[torch.Tensor]] = {}
        self._rir_pool_by_scenario: Dict[str, List[torch.Tensor]] = {}
        self._rir_pool: List[torch.Tensor] = []

        self._load_noise_sources()
        self._load_rir_sources()

    def _load_noise_sources(self):
        """Scan and pre-load real-world noise audio files grouped by scenario."""
        if not self.noise_dir.is_dir():
            print(f"[AcousticDegradation] Notice: Noise directory {self.noise_dir} not found. Fallback to synthetic.")
            return

        # QUT-NOISE consists of 5 canonical scenarios: cafe, car, home, reverb (pool/carpark), street
        scenarios = sorted([d.name.lower() for d in self.noise_dir.iterdir() if d.is_dir()])
        if not scenarios:
            scenarios = ["cafe", "car", "home", "reverb", "street"]

        for scenario in scenarios:
            scen_dir = self.noise_dir / scenario
            if not scen_dir.is_dir():
                continue

            audio_files = sorted(list(scen_dir.glob("*.flac")) + list(scen_dir.glob("*.wav")))
            self._noise_pool[scenario] = []

            for fpath in audio_files:
                try:
                    wf, sr = sf.read(str(fpath), always_2d=False, dtype="float32")
                    if wf.ndim > 1:
                        wf = np.mean(wf, axis=1)
                    if sr != self.sample_rate:
                        tensor_wf = torch.from_numpy(wf).unsqueeze(0)
                        tensor_wf = torchaudio.functional.resample(tensor_wf, sr, self.sample_rate)
                        wf = tensor_wf.squeeze(0).numpy()
                    self._noise_pool[scenario].append(torch.from_numpy(wf))
                except Exception as e:
                    print(f"[AcousticDegradation] Warning loading {fpath}: {e}")

        total_files = sum(len(v) for v in self._noise_pool.values())
        print(f"[AcousticDegradation] Loaded {total_files} real noise files across {list(self._noise_pool.keys())}.")

    def _load_rir_sources(self):
        """Scan and pre-load RIR impulse responses grouped by scenario."""
        if not self.rir_dir.is_dir():
            print(f"[AcousticDegradation] Notice: RIR directory {self.rir_dir} not found. Fallback to synthetic.")
            return

        self._rir_pool_by_scenario = {}
        self._rir_pool = []

        # QUT-NOISE impulses are structured by scenario: 'car' (cabin RIR) and 'reverb' (swimming pool & carpark RIR)
        scenario_dirs = [d for d in self.rir_dir.iterdir() if d.is_dir()]
        if not scenario_dirs:
            scenario_dirs = [self.rir_dir]

        for sdir in scenario_dirs:
            scen_name = sdir.name.lower() if sdir != self.rir_dir else "general"
            self._rir_pool_by_scenario[scen_name] = []
            rir_files = sorted(list(sdir.glob("*.flac")) + list(sdir.glob("*.wav")))
            for fpath in rir_files:
                try:
                    wf, sr = sf.read(str(fpath), always_2d=False, dtype="float32")
                    if wf.ndim > 1:
                        wf = np.mean(wf, axis=1)
                    if sr != self.sample_rate:
                        tensor_wf = torch.from_numpy(wf).unsqueeze(0)
                        tensor_wf = torchaudio.functional.resample(tensor_wf, sr, self.sample_rate)
                        wf = tensor_wf.squeeze(0).numpy()

                    # Normalize RIR to unit energy
                    rir_tensor = torch.from_numpy(wf)
                    energy = torch.sum(rir_tensor ** 2)
                    if energy > 1e-9:
                        rir_tensor = rir_tensor / torch.sqrt(energy)

                    self._rir_pool_by_scenario[scen_name].append(rir_tensor)
                    self._rir_pool.append(rir_tensor)
                except Exception as e:
                    print(f"[AcousticDegradation] Warning loading RIR {fpath}: {e}")

        counts_str = ", ".join(f"{k}: {len(v)}" for k, v in self._rir_pool_by_scenario.items())
        print(f"[AcousticDegradation] Loaded {len(self._rir_pool)} real RIR impulse responses ({counts_str}).")

    def _generate_synthetic_rir(self, rt60: float = 0.4, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        """Fallback synthetic room impulse response with exponential decay."""
        decay_samples = int(self.sample_rate * rt60)
        t = torch.linspace(0, 1, decay_samples, device=device)
        noise = torch.randn(decay_samples, device=device)
        envelope = torch.exp(-3.0 * np.log(10) * t)
        rir = noise * envelope
        rir = rir / torch.norm(rir).clamp_min(1e-9)
        return rir

    def _sample_noise_segment(
        self,
        noise_type: str,
        num_samples: int,
        seed: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Sample a deterministic or random noise slice of length `num_samples`."""
        seed_clean = int(seed % (2**32 - 1)) if seed is not None else None
        rng = np.random.RandomState(seed_clean) if seed_clean is not None else np.random

        # 1. Real-world scenario noise from pool (slice on CPU first to avoid PCIe overhead)
        if noise_type in self._noise_pool and len(self._noise_pool[noise_type]) > 0:
            pool = self._noise_pool[noise_type]
            track_idx = rng.randint(0, len(pool))
            track_cpu = pool[track_idx]

            if len(track_cpu) >= num_samples:
                max_start = len(track_cpu) - num_samples
                start = rng.randint(0, max_start + 1)
                return track_cpu[start : start + num_samples].clone().to(device)
            else:
                repeats = int(math.ceil(num_samples / len(track_cpu)))
                tiled = track_cpu.repeat(repeats)
                return tiled[:num_samples].clone().to(device)

        # Isolated generator to avoid mutating global PyTorch seed
        gen = torch.Generator(device="cpu")
        if seed_clean is not None:
            gen.manual_seed(seed_clean)

        # 2. White noise fallback
        if noise_type == "white":
            return torch.randn(num_samples, generator=gen, device="cpu").to(device)

        # 3. Pink noise (1/f) synthesis
        white = torch.randn(num_samples + 200, generator=gen, device="cpu").to(device)
        fft_len = num_samples + 200
        fft_coeffs = torch.fft.rfft(white)
        freqs = torch.fft.rfftfreq(fft_len, d=1.0 / self.sample_rate)
        # Avoid DC singularity
        weights = 1.0 / torch.sqrt(torch.clamp_min(freqs, 20.0))
        pink = torch.fft.irfft(fft_coeffs * weights.to(device), n=fft_len)
        return pink[100 : 100 + num_samples].clone()

    def apply_tensor(
        self,
        waveforms: torch.Tensor,
        condition: Union[str, ParsedCondition],
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply degradation to a PyTorch tensor batch.

        Args:
            waveforms: Tensor of shape (B, T) or (B, 1, T).
            condition: Condition string (e.g. 'cafe_5db', 'reverb', 'clean') or ParsedCondition.
            seed: Deterministic random seed for reproducibility.

        Returns:
            Degraded tensor of the same shape and dtype on the same device.
        """
        parsed = parse_condition_str(condition) if isinstance(condition, str) else condition
        if parsed.is_clean:
            return waveforms.clone()

        orig_shape = waveforms.shape
        orig_device = waveforms.device
        orig_dtype = waveforms.dtype

        # Flatten to (B, T)
        if waveforms.ndim == 3 and waveforms.shape[1] == 1:
            wf = waveforms.squeeze(1).float()
        elif waveforms.ndim == 2:
            wf = waveforms.float()
        else:
            raise ValueError(f"Expected (B, T) or (B, 1, T), got {waveforms.shape}")

        b_size, t_len = wf.shape

        # Step 1: Apply RIR reverberation if requested (causal linear convolution)
        if parsed.has_reverb:
            seed_clean = int(seed % (2**32 - 1)) if seed is not None else None
            # Physical Acoustic Invariant: QUT-NOISE RIRs are strictly scenario-bound.
            # - 'car' RIR (cabin impulse) can ONLY be applied to 'car' scenarios ('car_rir', 'car_compound').
            # - 'reverb' RIR (pool/carpark impulse) can ONLY be used for 'reverb'/'room' scenarios ('room_rir', 'reverb', 'room_compound', 'reverb_compound').
            # Cross-scenario RIR mixing (e.g. reverb RIR on cafe/street/home noise) is physically invalid and strictly forbidden.
            if parsed.noise_type == "car" or parsed.raw_name in ("car_rir", "car_compound"):
                rir_pool = self._rir_pool_by_scenario.get("car", [])
            elif (
                parsed.noise_type in ("reverb", None)
                or parsed.raw_name in ("reverb", "reverb_rir", "room_rir", "room_compound", "reverb_compound")
            ):
                rir_pool = self._rir_pool_by_scenario.get("reverb", [])
            else:
                raise ValueError(
                    f"Physical Acoustic Invariant Violation: QUT-NOISE RIRs are strictly scenario-matched. "
                    f"Measured impulses exist ONLY for 'car' (cabin) and 'reverb' (swimming pool/carpark). "
                    f"Condition '{parsed.raw_name}' attempts to apply RIR to '{parsed.noise_type}', which is unphysical and forbidden."
                )

            if len(rir_pool) > 0:
                rng = np.random.RandomState(seed_clean) if seed_clean is not None else np.random
                rir_idx = rng.randint(0, len(rir_pool))
                rir = rir_pool[rir_idx].to(orig_device).float()
            else:
                rir = self._generate_synthetic_rir(rt60=0.4, device=orig_device).float()

            rir_tensor = rir.unsqueeze(0)  # (1, L)
            # Mathematically exact linear causal convolution via FFT convolution, truncated to original length
            reverbed = torchaudio.functional.fftconvolve(wf, rir_tensor, mode="full")[:, :t_len]

            if len(rir_pool) > 0:
                # Real-world impulse responses already preserve physical direct sound + reflection ratio
                wf = reverbed
            else:
                # Synthetic exponential decay RIR: dry/wet mix
                wf = 0.7 * wf + 0.3 * reverbed

        # Step 2: Apply noise if requested
        if parsed.noise_type is not None and parsed.snr_db is not None:
            snr_db = parsed.snr_db
            noise_slices = []
            for i in range(b_size):
                sub_seed = (seed + i * 1009) if seed is not None else None
                noise_slice = self._sample_noise_segment(
                    noise_type=parsed.noise_type,
                    num_samples=t_len,
                    seed=sub_seed,
                    device=orig_device,
                )
                noise_slices.append(noise_slice)
            noise_batch = torch.stack(noise_slices, dim=0)

            # Accurate local RMS SNR scaling
            sig_pwr = torch.mean(wf ** 2, dim=-1, keepdim=True).clamp_min(1e-9)
            noise_pwr = torch.mean(noise_batch ** 2, dim=-1, keepdim=True).clamp_min(1e-9)
            scale = torch.sqrt(sig_pwr / (noise_pwr * (10.0 ** (snr_db / 10.0))))
            wf = wf + noise_batch * scale

        out = wf.view(orig_shape).to(orig_dtype)
        return out

    def apply_numpy(
        self,
        audio: np.ndarray,
        sr: int,
        condition: Union[str, ParsedCondition],
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply degradation to a 1D NumPy array for external baseline tools (Praat, DeepFry, Creapy).

        Args:
            audio: 1D numpy array of shape (T,).
            sr: Sampling rate (if != self.sample_rate, resampled internally).
            condition: Degradation condition string.
            seed: Deterministic random seed.

        Returns:
            Degraded float32 numpy array of shape (T,).
        """
        parsed = parse_condition_str(condition) if isinstance(condition, str) else condition
        if parsed.is_clean:
            return audio.astype(np.float32)

        tensor_in = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)  # (1, T)
        if sr != self.sample_rate:
            tensor_in = torchaudio.functional.resample(tensor_in, sr, self.sample_rate)

        tensor_out = self.apply_tensor(tensor_in, parsed, seed=seed)

        if sr != self.sample_rate:
            tensor_out = torchaudio.functional.resample(tensor_out, self.sample_rate, sr)

        out_np = tensor_out.squeeze(0).cpu().numpy().astype(np.float32)
        # Strict sample length preservation
        if len(out_np) != len(audio):
            if len(out_np) > len(audio):
                out_np = out_np[:len(audio)]
            else:
                out_np = np.pad(out_np, (0, len(audio) - len(out_np)))
        return out_np

    def create_degraded_wav_file(
        self,
        src_path: Union[str, Path],
        dst_path: Union[str, Path],
        condition: Union[str, ParsedCondition],
        seed: Optional[int] = None,
    ) -> Path:
        """
        Helper method to generate a degraded audio file on disk for tools requiring file paths.
        """
        src = Path(src_path)
        dst = Path(dst_path)
        dst.parent.mkdir(parents=True, exist_ok=True)

        audio, sr = sf.read(str(src), always_2d=False, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        degraded = self.apply_numpy(audio, sr, condition, seed=seed)

        # Normalize if peak exceeds 1.0 to prevent PCM_16 hard distortion
        max_val = float(np.max(np.abs(degraded)))
        if max_val > 1.0:
            degraded = degraded / (max_val + 1e-4) * 0.99

        ext = dst.suffix.lower()
        fmt = "WAV" if ext in (".wav", "") else ("FLAC" if ext == ".flac" else "WAV")
        sf.write(str(dst), degraded, sr, format=fmt, subtype="PCM_16")
        return dst


# Global singleton instance for easy cross-module sharing
_GLOBAL_ENGINE: Optional[AcousticDegradationEngine] = None


def get_acoustic_degradation_engine() -> AcousticDegradationEngine:
    """Retrieve or lazily initialize the shared global degradation engine."""
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = AcousticDegradationEngine()
    return _GLOBAL_ENGINE
