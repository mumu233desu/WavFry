#!/usr/bin/env python
import json
import os
import shutil
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm

WORKSPACE_ROOT = Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE")
AUDIO_DIR = WORKSPACE_ROOT / "01_en_dereverb"
SAMPLE_JSON = WORKSPACE_ROOT / "04_creaky_tools/_annotation_sampling/sampling_state.json"
TARGET_SNRS = [20, 10, 0, -10, -20]

def add_white_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    """Adds white noise to a signal for a given SNR in dB."""
    signal_power = np.mean(audio ** 2)
    if signal_power == 0:
        return audio
    
    # Calculate noise power based on target SNR
    # SNR = 10 * log10(P_signal / P_noise)
    # P_noise = P_signal / (10 ** (SNR / 10))
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    
    # Generate white noise
    noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
    
    # Add noise to signal
    noisy_audio = audio + noise
    
    # Avoid clipping if possible, though float32 usually handles it.
    # We will clip to [-1.0, 1.0] if necessary, but soundfile handles float32 writing well.
    max_amp = np.max(np.abs(noisy_audio))
    if max_amp > 1.0:
        # Normalize to prevent clipping on save
        noisy_audio = noisy_audio / max_amp
        
    return noisy_audio.astype(np.float32)

def main():
    np.random.seed(42)
    if not SAMPLE_JSON.exists():
        print(f"Error: {SAMPLE_JSON} not found.")
        return

    # 1. Parse which files are in the test set (batch 2: Held-out Val)
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

    # 2. Create output directories
    out_dirs = {}
    for snr in TARGET_SNRS:
        d = WORKSPACE_ROOT / f"01_en_dereverb_SNR{snr}"
        d.mkdir(parents=True, exist_ok=True)
        out_dirs[snr] = d

    # 3. Process each file
    for file_key in tqdm(list(target_files), desc="Generating Noisy Datasets"):
        src_wav = AUDIO_DIR / f"{file_key}.wav"
        src_flac = AUDIO_DIR / f"{file_key}.flac"
        
        src_path = None
        if src_wav.exists(): src_path = src_wav
        elif src_flac.exists(): src_path = src_flac
        else:
            print(f"Warning: Audio for {file_key} not found.")
            continue
            
        audio, sr = sf.read(str(src_path), always_2d=False)
        if isinstance(audio, np.ndarray) and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
            
        for snr in TARGET_SNRS:
            out_path = out_dirs[snr] / f"{file_key}.flac"
            if out_path.exists():
                continue # Skip if already generated
                
            noisy_audio = add_white_noise(audio, snr)
            sf.write(str(out_path), noisy_audio, sr, format="FLAC", subtype="PCM_16")
            
    print("Noise injection complete!")

if __name__ == "__main__":
    main()
