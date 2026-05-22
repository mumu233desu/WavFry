print("Starting test_lowpass.py...")
import torch
import torchaudio
import pandas as pd
from pathlib import Path
import sys

print("Python path:", sys.path)
# Add root to path
root = Path(__file__).resolve().parents[1]
print("Adding to path:", root)
if str(root) not in sys.path:
    sys.path.append(str(root))

try:
    from wavfry.data import dataset as dataset_mod
    print("Import success")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_lowpass():
    # Create a dummy dataframe
    # We need a real audio file for testing
    audio_files = list(Path("E:/Corpus/CS-Dialogue/data/short_wav/WAVE/01_en_dereverb").glob("*.flac"))
    if not audio_files:
        # Try generic glob if specific path fails
        audio_files = list(Path(".").rglob("*.wav")) + list(Path(".").rglob("*.flac"))
        
    if not audio_files:
        print("No audio files found for testing.")
        return
        
    test_wav = audio_files[0]
    df = pd.DataFrame([{
        'audio_path': str(test_wav),
        'start_s': 0.0,
        'end_s': 1.0,
        'label': 0,
        'speaker_id': 'TEST',
        'phone': 'aa'
    }])
    
    lowpass_cfg = {
        'enabled': True,
        'cutoff_hz': 3000.0,
        'Q': 0.707
    }
    
    print("Testing dataset with lowpass enabled...")
    ds_enabled = dataset_mod.CreakyVowelDataset(df, lowpass_cfg=lowpass_cfg)
    wav_enabled, _, _, _ = ds_enabled[0]
    
    print("Testing dataset with lowpass disabled...")
    ds_disabled = dataset_mod.CreakyVowelDataset(df, lowpass_cfg={'enabled': False})
    wav_disabled, _, _, _ = ds_disabled[0]
    
    # Compare
    diff = torch.abs(wav_enabled - wav_disabled).mean().item()
    print(f"Mean difference: {diff:.6f}")
    
    if diff > 0:
        print("Success: Lowpass filter made a difference.")
    else:
        print("Failure: Lowpass filter had no effect.")

if __name__ == "__main__":
    test_lowpass()
