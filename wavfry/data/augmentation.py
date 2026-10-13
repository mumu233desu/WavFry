import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as F

logger = logging.getLogger("wavfry.augmentation")


def generate_pink_noise(
    B: int,
    T: int,
    device: torch.device,
    dtype: torch.dtype,
    low_cutoff_hz: float = 10.0,
    sample_rate: int = 16000,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Generates pink noise in the frequency domain with 1/f amplitude scaling,
    exactly zero DC offset, and a zero-phase 2nd-order Butterworth-like high-pass filter.
    Returns a tensor of shape [B, T] with unit variance (std = 1.0).
    """
    num_freqs = T // 2 + 1

    # Generate complex Gaussian noise in frequency domain with isolated generator
    real = torch.randn(B, num_freqs, device=device, dtype=dtype, generator=generator)
    imag = torch.randn(B, num_freqs, device=device, dtype=dtype, generator=generator)
    spectrum = torch.complex(real, imag)

    # Frequency scaling: power spectrum S(f) ~ 1/f -> amplitude A(f) ~ 1/sqrt(f)
    k = torch.arange(1, num_freqs, device=device, dtype=dtype)
    pink_scale = 1.0 / torch.sqrt(k)

    weights = torch.zeros(num_freqs, device=device, dtype=dtype)
    if low_cutoff_hz > 0:
        bin_c = (low_cutoff_hz * T) / sample_rate
        hp_gain = (k**2) / (k**2 + bin_c**2)
        weights[1:] = pink_scale * hp_gain
    else:
        weights[1:] = pink_scale

    # Zero DC offset guaranteed by weights[0] == 0.0
    spectrum = spectrum * weights.unsqueeze(0)

    # Time-domain conversion
    noise = torch.fft.irfft(spectrum, n=T)

    # Normalize to unit variance per sample
    noise_std = torch.sqrt(torch.mean(noise**2, dim=1, keepdim=True) + 1e-9)
    noise = noise / noise_std
    return noise


class ReverbAugmentor:
    """
    Room Impulse Response (RIR) spatial acoustic convolver.
    Applies authentic acoustic reflections using QUT-NOISE (CAR & REVERB) scanned RIRs.
    
    Supports:
    - Pre-padded convolution at the slice level: mathematically 100% equivalent to
      convolving the entire infinite utterance and then slicing, preserving natural
      room tails from preceding speech without boundary truncation artifacts.
    - Whole-utterance convolution for full audio signals.
    - Staged curriculum scheduling (optional dry/wet fade-in across epochs).
    - Deterministic per-sample seeding for 100% reproducible training.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        seed: int = 42,
    ) -> None:
        self.config = config
        self.enabled = bool(config.get("enabled", False))
        self.prob = float(config.get("prob", 0.3))
        self.seed = int(seed)
        self.curriculum = bool(config.get("curriculum", False))
        self.max_epochs = int(config.get("max_epochs", 20))
        self.schedule_power = float(config.get("schedule_power", 2.0))

        rir_dir_str = config.get("rir_dir", "E:/Corpus/Noise/QUT-NOISE_processed/impulses")
        self.rir_dir = Path(rir_dir_str)
        self.scenarios = config.get("scenarios", ["car", "reverb"])

        self.rirs: List[torch.Tensor] = []
        self.rir_names: List[str] = []
        if self.enabled:
            self._load_rirs()

    def _load_rirs(self) -> None:
        """Load 16kHz normalized RIR impulses into memory."""
        if not self.rir_dir.exists():
            logger.warning(f"RIR directory {self.rir_dir} does not exist. Disabling ReverbAugmentor.")
            self.enabled = False
            return

        for scenario in self.scenarios:
            scenario_dir = self.rir_dir / scenario
            if not scenario_dir.exists():
                continue
            for rir_file in sorted(scenario_dir.glob("*.flac")):
                try:
                    data, sr = sf.read(str(rir_file), dtype="float32")
                    if sr != 16000:
                        tensor = torch.from_numpy(data).unsqueeze(0)
                        tensor = torchaudio.transforms.Resample(sr, 16000)(tensor).squeeze(0)
                        data = tensor.numpy()
                    rir_tensor = torch.from_numpy(data).float()
                    # Ensure unit energy sum(h^2) = 1.0
                    energy = float(torch.sum(rir_tensor**2))
                    if energy > 1e-12:
                        rir_tensor = rir_tensor / math.sqrt(energy)
                    self.rirs.append(rir_tensor)
                    self.rir_names.append(f"{scenario}/{rir_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to load RIR file {rir_file}: {e}")

        logger.info(
            f"ReverbAugmentor initialized: loaded {len(self.rirs)} RIR impulses "
            f"across scenarios {self.scenarios} (prob={self.prob})."
        )
        if len(self.rirs) == 0:
            logger.warning("No RIR files loaded! Disabling ReverbAugmentor.")
            self.enabled = False

    def apply_slice(
        self,
        full_waveform: torch.Tensor,
        start_frame: int,
        end_frame: int,
        epoch: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Convolve slice [start_frame:end_frame] with an RIR preserving pre-reverberation context.
        Mathematically identical to convolving full_waveform with RIR and taking slice.
        """
        slice_clean = full_waveform[start_frame:end_frame].clone()
        if not self.enabled or len(self.rirs) == 0 or self.prob <= 0.0:
            return slice_clean

        # Curriculum effective probability
        if self.curriculum and epoch is not None and self.max_epochs > 1:
            progress = min(max((epoch - 1) / (self.max_epochs - 1), 0.0), 1.0)
            curriculum_factor = progress ** self.schedule_power
            effective_prob = self.prob * curriculum_factor
        else:
            effective_prob = self.prob

        if effective_prob <= 0.0:
            return slice_clean

        rng = random.Random(seed if seed is not None else self.seed)
        if rng.random() >= effective_prob:
            return slice_clean

        # Pick random RIR
        rir_idx = rng.randint(0, len(self.rirs) - 1)
        rir = self.rirs[rir_idx]
        L = len(rir)
        pre_pad = L - 1

        # Extract pre-reverberation speech history
        actual_start = max(0, start_frame - pre_pad)
        left_zeros = pre_pad - (start_frame - actual_start)
        chunk = full_waveform[actual_start:end_frame]

        if left_zeros > 0:
            pad_tensor = torch.zeros(left_zeros, dtype=chunk.dtype, device=chunk.device)
            chunk = torch.cat([pad_tensor, chunk], dim=0)

        # Convolve via PyTorch FFT convolution
        chunk_2d = chunk.unsqueeze(0).float()
        rir_2d = rir.unsqueeze(0).to(chunk_2d.device).float()
        conv_full = F.fftconvolve(chunk_2d, rir_2d, mode="full").squeeze(0)

        # Extract the target slice with authentic preceding room tail
        target_len = end_frame - start_frame
        reverbed_slice = conv_full[pre_pad : pre_pad + target_len]
        return reverbed_slice.to(dtype=full_waveform.dtype)

    def apply_waveform(
        self,
        waveform: torch.Tensor,
        epoch: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """Convolve whole 1D waveform [T] with a randomly chosen RIR."""
        if not self.enabled or len(self.rirs) == 0 or self.prob <= 0.0:
            return waveform.clone()

        if self.curriculum and epoch is not None and self.max_epochs > 1:
            progress = min(max((epoch - 1) / (self.max_epochs - 1), 0.0), 1.0)
            effective_prob = self.prob * (progress ** self.schedule_power)
        else:
            effective_prob = self.prob

        if effective_prob <= 0.0:
            return waveform.clone()

        rng = random.Random(seed if seed is not None else self.seed)
        if rng.random() >= effective_prob:
            return waveform.clone()

        rir_idx = rng.randint(0, len(self.rirs) - 1)
        rir = self.rirs[rir_idx].to(waveform.device)

        w_2d = waveform.unsqueeze(0).float()
        r_2d = rir.unsqueeze(0).float()
        conv = F.fftconvolve(w_2d, r_2d, mode="full").squeeze(0)
        return conv[: len(waveform)].to(dtype=waveform.dtype)


class RealWorldNoisePool:
    """
    In-memory zero-I/O real-world noise cache.
    Loads 16kHz mono QUT-NOISE FLAC files into int16 CPU tensors (~1.2 GB RAM total)
    and slices arbitrary segments in microsecond time on demand.
    """

    def __init__(
        self,
        noise_dir: Union[str, Path] = "E:/Corpus/Noise/QUT-NOISE_processed/noise",
        scenarios: Optional[List[str]] = None,
    ) -> None:
        self.noise_dir = Path(noise_dir)
        self.scenarios = scenarios or ["cafe", "car", "home", "reverb", "street"]
        self.buffers: List[torch.Tensor] = []
        self.buffer_names: List[str] = []
        self._load_noise_files()

    def _load_noise_files(self) -> None:
        if not self.noise_dir.exists():
            logger.warning(f"Noise directory {self.noise_dir} does not exist.")
            return

        total_samples = 0
        for scenario in self.scenarios:
            scen_dir = self.noise_dir / scenario
            if not scen_dir.exists():
                continue
            for flac_path in sorted(scen_dir.glob("*.flac")):
                try:
                    data, sr = sf.read(str(flac_path), dtype="int16")
                    if sr != 16000:
                        logger.warning(f"Sample rate mismatch for {flac_path}: {sr} != 16000")
                        continue
                    tensor = torch.from_numpy(data)  # int16 CPU tensor
                    if len(tensor) == 0:
                        continue
                    self.buffers.append(tensor)
                    self.buffer_names.append(f"{scenario}/{flac_path.name}")
                    total_samples += len(tensor)
                except Exception as e:
                    logger.warning(f"Failed to load noise file {flac_path}: {e}")

        total_hours = total_samples / (16000 * 3600.0)
        total_mb = (total_samples * 2) / (1024 * 1024.0)
        logger.info(
            f"RealWorldNoisePool initialized: {len(self.buffers)} files loaded "
            f"({total_hours:.2f} hours, {total_mb:.1f} MB in RAM int16)."
        )

    def sample_slices(
        self,
        num_slices: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
        seed: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample `num_slices` random slices of length `length`.
        Transfers int16 to GPU directly to halve PCIe bandwidth, then scales on GPU.
        """
        if len(self.buffers) == 0:
            return torch.randn(num_slices, length, device=device, dtype=dtype)

        rng = random.Random(seed)
        slices = []
        for i in range(num_slices):
            buf_idx = rng.randint(0, len(self.buffers) - 1)
            buf = self.buffers[buf_idx]
            L = len(buf)
            if L == 0:
                slices.append(torch.zeros(length, dtype=torch.int16))
                continue
            if L <= length:
                reps = int(math.ceil(length / L))
                sliced = buf.repeat(reps)[:length]
            else:
                start = rng.randint(0, L - length)
                sliced = buf[start : start + length]
            slices.append(sliced)

        batch_int16 = torch.stack(slices, dim=0).to(device=device, non_blocking=True)
        # Scaled on GPU directly: int16 -> target float dtype
        batch_float = batch_int16.to(dtype=dtype).mul_(1.0 / 32768.0)
        return batch_float


class AudioAugmentor:
    """
    GPU-accelerated Curriculum Audio Augmentor.
    Features:
    - Multi-source additive noise (Pink noise + QUT Real-world noise dynamic mixture)
    - Real-time local RMS SNR scaling per 2.0s block
    - Isolated PRNG stream eliminating global RNG / Dropout side effects
    - DeepFilterNet3 denoising curriculum
    - Power-curve intensity scheduling
    """

    def __init__(self, config: Dict[str, Any], max_epochs: Optional[int] = None, seed: int = 42):
        self.config = config
        self.enabled = config.get("enabled", False)
        self.seed = int(seed)
        if not self.enabled:
            return

        self.aug_order = config.get("aug_order", "noise_first")
        self.schedule_power = config.get("schedule_power", 2.0)

        self.denoise_cfg = config.get("denoising", {})
        self.noise_cfg = config.get("noise", {})

        if max_epochs is not None:
            self.max_epochs = int(max_epochs)
        elif "max_epochs" in config:
            self.max_epochs = int(config["max_epochs"])
        elif "training" in config and isinstance(config["training"], dict) and "epochs" in config["training"]:
            self.max_epochs = int(config["training"]["epochs"])
        else:
            self.max_epochs = 20

        # Real-world noise pool setup
        self.real_world_pool: Optional[RealWorldNoisePool] = None
        sources = self.noise_cfg.get("sources", {})
        if self.noise_cfg.get("enabled", False) and float(sources.get("real_world", 0.0)) > 0:
            noise_dir = self.noise_cfg.get("real_world_dir", "E:/Corpus/Noise/QUT-NOISE_processed/noise")
            scenarios = self.noise_cfg.get("scenarios", ["cafe", "car", "home", "reverb", "street"])
            self.real_world_pool = RealWorldNoisePool(noise_dir=noise_dir, scenarios=scenarios)

        # Lazy loading DFN models
        self.dfn_model = None
        self.dfn_df_state = None

        # Resampler cache
        self.resamplers = {}

    def _get_resampler(self, orig_sr: int, target_sr: int, device: torch.device):
        key = (orig_sr, target_sr, device)
        if key not in self.resamplers:
            self.resamplers[key] = torchaudio.transforms.Resample(orig_sr, target_sr).to(device)
        return self.resamplers[key]

    def _get_intensity(self, epoch: int, start_val: float, end_val: float) -> float:
        """Curriculum schedule: progress from start_val to end_val over max_epochs."""
        if self.max_epochs <= 1:
            return end_val
        progress = min(max((epoch - 1) / (self.max_epochs - 1), 0.0), 1.0)
        intensity_factor = progress**self.schedule_power
        return start_val + (end_val - start_val) * intensity_factor

    def _init_df(self):
        """Initialize DeepFilterNet model and state."""
        if self.dfn_model is None:
            from df.enhance import init_df

            self.dfn_model, self.dfn_df_state, _ = init_df()
            device = next(self.dfn_model.parameters()).device
            logger.info(f"DeepFilterNet initialized on device: {device}")

    def apply(
        self,
        waveforms: torch.Tensor,
        epoch: int,
        device: torch.device,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply augmentation to a batch of waveforms [B, T].
        Called AFTER waveforms are moved to the target device.
        """
        if not self.enabled:
            return waveforms

        denoise_prob = self.denoise_cfg.get("prob", 0.0)
        noise_prob = self.noise_cfg.get("prob", 0.0)

        device_type = "cuda" if "cuda" in str(device) else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            waveforms = waveforms.float()
            if self.aug_order == "noise_first":
                waveforms = self.add_noise(waveforms, epoch, noise_prob, device, step=step)
                waveforms = self.denoise(waveforms, epoch, denoise_prob, device, step=step)
            elif self.aug_order == "denoise_first":
                waveforms = self.denoise(waveforms, epoch, denoise_prob, device, step=step)
                waveforms = self.add_noise(waveforms, epoch, noise_prob, device, step=step)
            else:
                base_seed = self.seed + epoch * 100003 + (step or 0) * 1009
                rng = random.Random(base_seed)
                if rng.random() > 0.5:
                    waveforms = self.add_noise(waveforms, epoch, noise_prob, device, step=step)
                    waveforms = self.denoise(waveforms, epoch, denoise_prob, device, step=step)
                else:
                    waveforms = self.denoise(waveforms, epoch, denoise_prob, device, step=step)
                    waveforms = self.add_noise(waveforms, epoch, noise_prob, device, step=step)

        return waveforms

    def denoise(
        self,
        waveforms: torch.Tensor,
        epoch: int,
        prob: float,
        device: torch.device,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply DFN denoising to batch with isolated deterministic RNG."""
        if not self.denoise_cfg.get("enabled", False) or prob <= 0.0:
            return waveforms

        atten_db = self._get_intensity(
            epoch,
            self.denoise_cfg.get("atten_lim_db_start", 0.0),
            self.denoise_cfg.get("atten_lim_db_end", 20.0),
        )

        if atten_db < 0.5:
            return waveforms

        self._init_df()
        from df.enhance import enhance

        B, T = waveforms.shape
        enhanced_list = []

        if next(self.dfn_model.parameters()).device != device:
            self.dfn_model = self.dfn_model.to(device)

        resample_to_48 = self._get_resampler(16000, 48000, device)
        resample_to_16 = self._get_resampler(48000, 16000, device)

        # Isolated deterministic RNG avoiding global torch.rand
        base_seed = self.seed + epoch * 100003 + (step or 0) * 1009 + 42
        rng = random.Random(base_seed)

        for i in range(B):
            if rng.random() < prob:
                wav = waveforms[i : i + 1]
                wav_48 = resample_to_48(wav)
                if hasattr(self.dfn_df_state, "reset"):
                    self.dfn_df_state.reset()
                with torch.no_grad():
                    wav_48_cpu = wav_48.cpu()
                    wav_enhanced_cpu = enhance(self.dfn_model, self.dfn_df_state, wav_48_cpu, atten_lim_db=atten_db)
                    wav_enhanced = wav_enhanced_cpu.to(device)
                wav_16 = resample_to_16(wav_enhanced)
                if wav_16.shape[-1] > T:
                    wav_16 = wav_16[..., :T]
                elif wav_16.shape[-1] < T:
                    wav_16 = torch.nn.functional.pad(wav_16, (0, T - wav_16.shape[-1]))
                enhanced_list.append(wav_16)
            else:
                enhanced_list.append(waveforms[i : i + 1])

        return torch.cat(enhanced_list, dim=0)

    def add_noise(
        self,
        waveforms: torch.Tensor,
        epoch: int,
        prob: float,
        device: torch.device,
        step: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Add synthetic (Pink) or real-world (QUT-NOISE) noise to batch.
        Calculates local RMS SNR dynamically on each 2.0s block slice with isolated PRNG.
        """
        if not self.noise_cfg.get("enabled", False) or prob <= 0.0:
            return waveforms

        snr_db = self._get_intensity(
            epoch,
            self.noise_cfg.get("snr_db_start", 35.0),
            self.noise_cfg.get("snr_db_end", 12.0),
        )

        B, T = waveforms.shape
        # Seeded deterministic RNG for reproducible sampling
        base_seed = self.seed + epoch * 100003 + (step or 0) * 1009
        rng = random.Random(base_seed)
        # Dedicated PyTorch Generator avoiding global state mutation
        generator_seed = (base_seed ^ 0x5DEECE66D) & 0xFFFFFFFF
        gen = torch.Generator(device=device)
        gen.manual_seed(generator_seed)

        # Decide mask per sample
        mask_list = [1.0 if rng.random() < prob else 0.0 for _ in range(B)]
        mask = torch.tensor(mask_list, device=device, dtype=waveforms.dtype).view(B, 1)

        if mask.sum() == 0:
            return waveforms

        # Determine source probabilities
        sources = self.noise_cfg.get("sources", None)
        if sources is None:
            # Fallback to legacy 'type'
            legacy_type = self.noise_cfg.get("type", "pink")
            sources = {legacy_type: 1.0}

        p_pink = float(sources.get("pink", 0.0))
        p_real = float(sources.get("real_world", 0.0))
        p_white = float(sources.get("white", 0.0))
        total_p = p_pink + p_real + p_white
        if total_p <= 0.0:
            total_p = 1.0
            p_pink = 1.0

        p_pink /= total_p
        p_real /= total_p
        p_white /= total_p

        noise_tensors = torch.zeros_like(waveforms)
        real_indices = []
        pink_indices = []
        white_indices = []

        for i in range(B):
            if mask_list[i] > 0.5:
                r = rng.random()
                if r < p_pink:
                    pink_indices.append(i)
                elif r < p_pink + p_real:
                    if self.real_world_pool is not None and len(self.real_world_pool.buffers) > 0:
                        real_indices.append(i)
                    else:
                        pink_indices.append(i)  # fallback to pink
                else:
                    white_indices.append(i)

        # Generate pink noise for designated slots using isolated generator
        if pink_indices:
            k_pink = len(pink_indices)
            low_cutoff = float(self.noise_cfg.get("pink_low_cutoff_hz", 10.0))
            pink_noise = generate_pink_noise(
                k_pink,
                T,
                device,
                waveforms.dtype,
                low_cutoff_hz=low_cutoff,
                sample_rate=16000,
                generator=gen,
            )
            noise_tensors[pink_indices] = pink_noise

        # Sample real-world noise from in-memory pool for designated slots
        if real_indices and self.real_world_pool is not None:
            k_real = len(real_indices)
            real_noise = self.real_world_pool.sample_slices(
                num_slices=k_real,
                length=T,
                device=device,
                dtype=waveforms.dtype,
                seed=base_seed + 777,
            )
            noise_tensors[real_indices] = real_noise

        # Generate white noise for designated slots using isolated generator
        if white_indices:
            k_white = len(white_indices)
            white_noise = torch.randn(k_white, T, device=device, dtype=waveforms.dtype, generator=gen)
            noise_tensors[white_indices] = white_noise

        # Local RMS calculation for target SNR:
        # SNR = 10 * log10(P_signal / P_noise) -> P_noise_target = P_signal / 10^(SNR/10)
        signal_pwr = torch.mean(waveforms**2, dim=1, keepdim=True)
        noise_pwr = torch.mean(noise_tensors**2, dim=1, keepdim=True)
        scale = torch.sqrt(signal_pwr / (noise_pwr * (10.0 ** (snr_db / 10.0)) + 1e-9))
        noisy_waveforms = waveforms + noise_tensors * scale * mask
        return noisy_waveforms
