import torch
import torchaudio
import numpy as np
from pathlib import Path

def generate_pink_noise(B, T, device, dtype, low_cutoff_hz=10.0, sample_rate=16000):
    """
    Generates pink noise in the frequency domain with 1/f amplitude scaling,
    exactly zero DC offset, and a zero-phase 2nd-order Butterworth-like high-pass filter.
    Returns a tensor of shape [B, T] with unit variance (std = 1.0).
    """
    num_freqs = T // 2 + 1
    
    # Generate complex Gaussian noise in frequency domain
    real = torch.randn(B, num_freqs, device=device, dtype=dtype)
    imag = torch.randn(B, num_freqs, device=device, dtype=dtype)
    spectrum = torch.complex(real, imag)
    
    # Frequency scaling: power spectrum S(f) ~ 1/f -> amplitude A(f) ~ 1/sqrt(f)
    # The frequencies are 0, 1, 2, ..., num_freqs - 1
    k = torch.arange(1, num_freqs, device=device, dtype=dtype)
    
    pink_scale = 1.0 / torch.sqrt(k)
    
    # Define weights vector
    weights = torch.zeros(num_freqs, device=device, dtype=dtype)
    
    if low_cutoff_hz > 0:
        # Butterworth 2nd-order high-pass filter response in frequency domain:
        # H_gain = f^2 / (f^2 + f_c^2)
        bin_c = (low_cutoff_hz * T) / sample_rate
        hp_gain = (k ** 2) / (k ** 2 + bin_c ** 2)
        weights[1:] = pink_scale * hp_gain
    else:
        weights[1:] = pink_scale
        
    # Apply weights across the spectrum
    # weights[0] is exactly 0.0 -> guarantees absolute zero DC offset!
    spectrum = spectrum * weights.unsqueeze(0)
    
    # Convert back to time domain
    noise = torch.fft.irfft(spectrum, n=T)
    
    # Normalize noise to unit power (variance = 1.0) per sample
    # so it matches white noise characteristics and scales correctly with SNR db
    noise_std = torch.sqrt(torch.mean(noise ** 2, dim=1, keepdim=True) + 1e-9)
    noise = noise / noise_std
    
    return noise

class AudioAugmentor:
    """
    GPU-accelerated Curriculum Audio Augmentor.
    Features:
    - Batch processing on GPU
    - DeepFilterNet denoising with state reset (prevents leakage)
    - Gaussian noise addition
    - Power-curve intensity scheduling
    """
    def __init__(self, config):
        self.config = config
        self.enabled = config.get('enabled', False)
        if not self.enabled:
            return

        self.aug_order = config.get('aug_order', 'noise_first')
        self.schedule_power = config.get('schedule_power', 2.0)
        
        # Scheduling parameters
        self.denoise_cfg = config.get('denoising', {})
        self.noise_cfg = config.get('noise', {})
        
        self.max_epochs = config.get('max_epochs', 20) # Used for scheduling
        
        # Lazy loading DFN models
        self.dfn_model = None
        self.dfn_df_state = None
        
        # Resampler cache
        self.resamplers = {} # (orig_sr, target_sr, device) -> Resample object

    def _get_resampler(self, orig_sr, target_sr, device):
        key = (orig_sr, target_sr, device)
        if key not in self.resamplers:
            self.resamplers[key] = torchaudio.transforms.Resample(orig_sr, target_sr).to(device)
        return self.resamplers[key]

    def _get_intensity(self, epoch, start_val, end_val):
        """Curriculum schedule: progress from 0.0 to 1.0 over max_epochs."""
        progress = min(max((epoch - 1) / (self.max_epochs - 1), 0.0), 1.0)
        # Power curve for slow start
        intensity_factor = progress ** self.schedule_power
        return start_val + (end_val - start_val) * intensity_factor

    def _init_df(self):
        """Initialize DeepFilterNet model and state."""
        if self.dfn_model is None:
            from df.enhance import init_df
            # Use default DeepFilterNet3
            self.dfn_model, self.dfn_df_state, _ = init_df()
            # Move model to GPU if available
            device = next(self.dfn_model.parameters()).device
            print(f"DeepFilterNet initialized on device: {device}")

    def apply(self, waveforms, epoch, device):
        """
        Apply augmentation to a batch of waveforms [B, T].
        This should be called AFTER waveforms are moved to the target device.
        """
        if not self.enabled:
            return waveforms
        
        # Determine current intensity
        denoise_prob = self.denoise_cfg.get('prob', 0.0)
        noise_prob = self.noise_cfg.get('prob', 0.0)
        
        # Apply according to order
        if self.aug_order == 'noise_first':
            waveforms = self.add_noise(waveforms, epoch, noise_prob, device)
            waveforms = self.denoise(waveforms, epoch, denoise_prob, device)
        elif self.aug_order == 'denoise_first':
            waveforms = self.denoise(waveforms, epoch, denoise_prob, device)
            waveforms = self.add_noise(waveforms, epoch, noise_prob, device)
        else: # Random order
            if torch.rand(1).item() > 0.5:
                waveforms = self.add_noise(waveforms, epoch, noise_prob, device)
                waveforms = self.denoise(waveforms, epoch, denoise_prob, device)
            else:
                waveforms = self.denoise(waveforms, epoch, denoise_prob, device)
                waveforms = self.add_noise(waveforms, epoch, noise_prob, device)
                
        return waveforms

    def denoise(self, waveforms, epoch, prob, device):
        """Apply DFN denoising to batch."""
        if not self.denoise_cfg.get('enabled', False):
            return waveforms
            
        atten_db = self._get_intensity(
            epoch, 
            self.denoise_cfg.get('atten_lim_db_start', 0.0),
            self.denoise_cfg.get('atten_lim_db_end', 20.0)
        )
        
        if atten_db < 0.5: # Effectively off
            return waveforms
            
        self._init_df()
        from df.enhance import enhance
        
        # Process each sample in batch (DFN enhance doesn't support batching natively on GPU yet)
        # But resampling and other prep can be batched if needed
        B, T = waveforms.shape
        enhanced_list = []
        
        # Move model to same device as data
        if next(self.dfn_model.parameters()).device != device:
            self.dfn_model = self.dfn_model.to(device)
            
        resample_to_48 = self._get_resampler(16000, 48000, device)
        resample_to_16 = self._get_resampler(48000, 16000, device)
        
        for i in range(B):
            if torch.rand(1).item() < prob:
                wav = waveforms[i:i+1] # [1, T]
                
                # Resample to 48k for DFN
                wav_48 = resample_to_48(wav)
                
                # IMPORTANT: Reset state to prevent leakage between samples
                if hasattr(self.dfn_df_state, 'reset'):
                    self.dfn_df_state.reset()
                
                # Enhance
                # NOTE: DFN high-level 'enhance' needs CPU tensors for its internal analysis 
                # even if the model is on GPU.
                with torch.no_grad():
                    wav_48_cpu = wav_48.cpu()
                    wav_enhanced_cpu = enhance(self.dfn_model, self.dfn_df_state, wav_48_cpu, atten_lim_db=atten_db)
                    wav_enhanced = wav_enhanced_cpu.to(device)
                
                # Resample back to 16k
                wav_16 = resample_to_16(wav_enhanced)
                
                # Ensure length matches (resampling can slightly shift length)
                if wav_16.shape[-1] > T:
                    wav_16 = wav_16[..., :T]
                elif wav_16.shape[-1] < T:
                    wav_16 = torch.nn.functional.pad(wav_16, (0, T - wav_16.shape[-1]))
                    
                enhanced_list.append(wav_16)
            else:
                enhanced_list.append(waveforms[i:i+1])
                
        return torch.cat(enhanced_list, dim=0)

    def add_noise(self, waveforms, epoch, prob, device):
        """Add Gaussian or Pink noise to batch."""
        if not self.noise_cfg.get('enabled', False):
            return waveforms
            
        snr_db = self._get_intensity(
            epoch,
            self.noise_cfg.get('snr_db_start', 35.0),
            self.noise_cfg.get('snr_db_end', 18.0)
        )
        
        B, T = waveforms.shape
        mask = (torch.rand(B, device=device) < prob).float().view(B, 1)
        
        if mask.sum() == 0:
            return waveforms
            
        noise_type = self.noise_cfg.get('type', 'white')
        if noise_type == 'pink':
            low_cutoff = self.noise_cfg.get('pink_low_cutoff_hz', 10.0)
            noise = generate_pink_noise(B, T, device, waveforms.dtype, low_cutoff_hz=low_cutoff, sample_rate=16000)
        else:
            noise = torch.randn_like(waveforms)
        
        # Calculate power
        signal_pwr = torch.mean(waveforms**2, dim=1, keepdim=True)
        noise_pwr = torch.mean(noise**2, dim=1, keepdim=True)
        
        # Calculate scale factor for target SNR
        # SNR = 10 * log10(P_signal / P_noise)
        # P_noise_target = P_signal / 10**(SNR/10)
        scale = torch.sqrt(signal_pwr / (noise_pwr * 10**(snr_db / 10.0) + 1e-9))
        
        noisy_waveforms = waveforms + noise * scale * mask
        return noisy_waveforms
