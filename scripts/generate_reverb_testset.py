#!/usr/bin/env python
import json
import os
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from tqdm import tqdm

WORKSPACE_ROOT = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE")
AUDIO_DIR = WORKSPACE_ROOT / "01_en_dereverb"
SAMPLE_JSON = WORKSPACE_ROOT / "04_creaky_tools\_annotation_sampling\sampling_state.json"
OUT_DIR = WORKSPACE_ROOT / "01_en_dereverb_Reverb"
RT60 = 0.5  # Medium reverberation time in seconds

def generate_rir(fs: int, rt60: float) -> np.ndarray:
    """Generates a synthetic exponential decay Room Impulse Response."""
    t = np.arange(0, rt60, 1.0 / fs)
    # 60dB decay means amplitude decays by 1000 (10^(60/20))
    # decay envelope = exp(-t / tau), where at t=rt60, decay = 0.001
    # exp(-rt60 / tau) = 0.001  => tau = -rt60 / ln(0.001) = rt60 / (3 * ln(10))
    tau = rt60 / (3 * np.log(10))
    decay = np.exp(-t / tau)
    
    # Generate white noise and apply the decay envelope
    noise = np.random.randn(len(t))
    rir = noise * decay
    return rir / np.max(np.abs(rir))

def add_reverb(audio: np.ndarray, fs: int, rt60: float) -> np.ndarray:
    """Convolves audio with a synthetic RIR to add reverberation."""
    # Ensure audio is 1D
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
        
    rir = generate_rir(fs, rt60)
    
    # Convolve
    reverbed = fftconvolve(audio, rir, mode='full')
    
    # Truncate to original length to align properly with forced alignment timestamps
    reverbed = reverbed[:len(audio)]
    
    # Normalize to avoid clipping
    max_val = np.max(np.abs(reverbed))
    if max_val > 1.0:
        reverbed = reverbed / max_val
        
    return reverbed.astype(np.float32)

def main():
    if not SAMPLE_JSON.exists():
        print(f"Error: {SAMPLE_JSON} not found.")
        return

    # Parse which files are in the test set (batch 2)
    target_files = set()
    with open(SAMPLE_JSON, "r", encoding="utf-8") as f:
        st = json.load(f)
        for b in st.get("batches", []):
            if b.get("batch_id") == 2:
                for fh in b.get("files", []):
                    if fh.get("annotated", False):
                        target_files.add(fh["file_key"])
                break
                
    print(f"Found {len(target_files)} target files for Setup B evaluation.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_key in tqdm(list(target_files), desc="Generating Reverb Dataset"):
        src_wav = AUDIO_DIR / f"{file_key}.wav"
        src_flac = AUDIO_DIR / f"{file_key}.flac"
        
        src_path = None
        if src_wav.exists(): src_path = src_wav
        elif src_flac.exists(): src_path = src_flac
        else:
            print(f"Warning: Audio for {file_key} not found.")
            continue
            
        out_path = OUT_DIR / f"{file_key}.flac"
        if out_path.exists():
            continue
            
        audio, sr = sf.read(str(src_path), always_2d=False)
        reverbed_audio = add_reverb(audio, sr, RT60)
        sf.write(str(out_path), reverbed_audio, sr, format="FLAC", subtype="PCM_16")
            
    print("Reverberation injection complete!")

if __name__ == "__main__":
    main()
