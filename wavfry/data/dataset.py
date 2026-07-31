import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
import platform
from pathlib import Path

from .continuous_non_creaky_sampling import ContinuousNonCreakyWindowGenerator


class CreakyVowelDataset(Dataset):
    def __init__(
        self,
        df,
        context_padding_ms=50,
        target_sr=16000,
        lowpass_cfg=None,
        continuous_non_creaky_cfg=None,
        seed=42,
        fold_index=0,
    ):
        self.base_df = df.reset_index(drop=True)
        self.context_padding_ms = context_padding_ms
        self.target_sr = target_sr
        self.resamplers = {} # Cache resamplers by source sample rate
        self.current_epoch = 1
        self.continuous_non_creaky = ContinuousNonCreakyWindowGenerator(
            self.base_df,
            continuous_non_creaky_cfg,
            seed=seed,
            fold_index=fold_index,
        )
        self.df = self._build_epoch_df(self.current_epoch)
        
        # Lowpass filter config (deterministic, applied to ALL samples)
        self.lowpass_cfg = lowpass_cfg or {}
        self.lowpass_enabled = self.lowpass_cfg.get('enabled', False)
        self.lowpass_cutoff  = self.lowpass_cfg.get('cutoff_hz', 3000.0)
        self.lowpass_Q       = self.lowpass_cfg.get('Q', 0.707)

    def _build_epoch_df(self, epoch: int):
        generated_df = self.continuous_non_creaky.generate(epoch)
        if generated_df.empty:
            return self.base_df.copy()

        # Preserve all original rows and add deterministic epoch-specific views.
        combined = pd.concat([self.base_df, generated_df], ignore_index=True, sort=False)
        return combined.sample(
            frac=1.0,
            random_state=self.continuous_non_creaky.shuffle_seed_for_epoch(epoch),
        ).reset_index(drop=True)

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)
        self.df = self._build_epoch_df(self.current_epoch)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx, attempts=0):
        if attempts > 10:
            raise RuntimeError(f"Failed to load a valid audio sample after 10 attempts. Last attempted index: {idx}")
            
        row = self.df.iloc[idx]
        audio_path = row['audio_path']
        start_s = row['start_s']
        end_s = row['end_s']
        label = row['label']

        # Generated continuous windows are already expanded inside a verified
        # non-creaky run, so they intentionally receive no external context.
        row_padding_ms = row.get('context_padding_ms', self.context_padding_ms)
        if pd.isna(row_padding_ms):
            row_padding_ms = self.context_padding_ms
        padding_s = float(row_padding_ms) / 1000.0
        start_with_padding = max(0, start_s - padding_s)
        
        import time
        try:
            # We need the original sample rate to know how many samples to skip
            info = torchaudio.info(audio_path)
            orig_sr = info.sample_rate
            total_frames = info.num_frames
            
            # Bound the end_s + padding
            end_with_padding = min(end_s + padding_s, total_frames / orig_sr)
            
            start_frame = int(start_with_padding * orig_sr)
            num_frames = int((end_with_padding - start_with_padding) * orig_sr)
            
            # Load specific segment
            t0 = time.time()
            waveform, sr = torchaudio.load(audio_path, frame_offset=start_frame, num_frames=num_frames)
            load_time = time.time() - t0
            
            # Diagnostic: if loading a short segment takes > 2s, it's a "slow" file
            if load_time > 2.0:
                print(f"\n[Warning] Slow audio load ({load_time:.2f}s): {audio_path}")

            # Convert to mono if needed
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Resample to 16kHz
            if sr != self.target_sr:
                if sr not in self.resamplers:
                    self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
                waveform = self.resamplers[sr](waveform)
                
            # [Lowpass] Deterministic high-cut filter (applied to all sets)
            if self.lowpass_enabled:
                waveform = torchaudio.functional.lowpass_biquad(
                    waveform, # [1, T]
                    sample_rate=self.target_sr,
                    cutoff_freq=self.lowpass_cutoff,
                    Q=self.lowpass_Q
                )
                
            # Squeeze to [T]
            waveform = waveform.squeeze(0)
            
            # Metadata for provenance
            audio_stem = Path(audio_path).stem
            phone = row.get('phone', 'UNK')
            meta = f"{audio_stem}|{start_s:.3f}-{end_s:.3f}|{phone}"
            
            return waveform, torch.tensor(label, dtype=torch.float32), row['speaker_id'], meta
        except Exception as e:
            print(f"\n[Error] Failed to load {audio_path}: {e}")
            return self.__getitem__((idx + 1) % len(self.df), attempts + 1)

def collate_fn(batch):
    """
    Pad waveforms to the same length in a batch.
    """
    waveforms, labels, speaker_ids, meta_list = zip(*batch)
    lengths = [w.shape[0] for w in waveforms]
    max_len = max(lengths)
    
    padded_waveforms = torch.zeros(len(waveforms), max_len)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :w.shape[0]] = w
        
    return padded_waveforms, torch.stack(labels), torch.tensor(lengths), list(speaker_ids), list(meta_list)

def resolve_num_workers(cfg_value):
    if cfg_value == "auto":
        if platform.system() == "Windows":
            return 0
        return 4
    return int(cfg_value)
