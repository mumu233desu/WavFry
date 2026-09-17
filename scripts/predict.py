import argparse
import torch
import torchaudio
import pandas as pd
from pathlib import Path
import sys
import importlib.util
from tqdm import tqdm
import tgt

# Add the 06_wavfry directory to sys.path to allow 'import wavfry'
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from wavfry.data import dataset as dataset_mod
from wavfry.model import wavlm_classifier as model_mod

# Comprehensive vowel set synced with 04_creaky_tools/src/creaky_vowel_analysis_core.py
VOWEL_SET = {
    # --- ARPABET (English) ---
    'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'eh', 'er', 'ey', 'ih', 'iy', 'ow', 'oy', 'uh', 'uw', 'ux',
    
    # --- IPA (English MFA) ---
    'a', 'aː', 'aj', 'aw', 'e', 'eː', 'ej', 'i', 'iː', 'o', 'oː', 'ow', 'u', 'uː', 'æ', 'ɑ', 'ɑː', 'ɐ', 'ɒ', 'ɒː', 'ɔ', 'ɔj', 'ə', 'əw', 'ɚ', 'ɛ', 'ɛː', 'ɜ', 'ɜː', 'ɝ', 'ɪ', 'ʉ', 'ʉː', 'ʊ',
    
    # --- Mandarin (HFA Pinyin-style) ---
    'a', 'e', 'i', 'o', 'u', 'v', 'ai', 'an', 'ao', 'ei', 'er', 'i0', 'ia', 'ie', 'in', 'io', 'ir', 'iu', 'ou', 'ua', 'ui', 'un', 'uo', 've', 'vn', 'ang', 'eng', 'ian', 'iao', 'ing', 'ong', 'uai', 'uan', 'van', 'iang', 'iong', 'uang', 'ueng',
    # Mandarin with Pinyin Tones (a1, e2, etc.)
    'a1', 'a2', 'a3', 'a4', 'a5', 'e1', 'e2', 'e3', 'e4', 'e5', 'i1', 'i2', 'i3', 'i4', 'i5', 
    'o1', 'o2', 'o3', 'o4', 'o5', 'u1', 'u2', 'u3', 'u4', 'u5', 'v1', 'v2', 'v3', 'v4', 'v5', 'y1', 'y2', 'y3', 'y4', 'y5',
    
    # --- Mandarin (MFA IPA with Tone Bars) ---
    'aj˥', 'aj˥˩', 'aj˧', 'aj˧˥', 'aj˨˩˦', 'aj˩', 'aw˥', 'aw˥˩', 'aw˧', 'aw˧˥', 'aw˨˩˦', 'aw˩', 
    'a˥', 'a˥˩', 'a˧', 'a˧˥', 'a˨˩˦', 'a˩', 'ej˥', 'ej˥˩', 'ej˧', 'ej˧˥', 'ej˨˩˦', 'ej˩', 
    'e˥', 'e˥˩', 'e˧', 'e˧˥', 'e˨˩˦', 'e˩', 'i˥', 'i˥˩', 'i˧', 'i˧˥', 'i˨˩˦', 'i˩', 
    'ow˥', 'ow˥˩', 'ow˧', 'ow˧˥', 'ow˨˩˦', 'ow˩', 'o˥', 'o˥˩', 'o˧', 'o˧˥', 'o˨˩˦', 'o˩', 
    'u˥', 'u˥˩', 'u˧', 'u˧˥', 'u˨˩˦', 'u˩', 'y˥', 'y˥˩', 'y˧', 'y˧˥', 'y˨˩˦', 'y˩', 
    'ə˥', 'ə˥˩', 'ə˧', 'ə˧˥', 'ə˨˩˦', 'ə˩', 'y',
    
    # --- General / Misc IPA ---
    'ɪ', 'ɛ', 'æ', 'ɑ', 'ɒ', 'ɔ', 'ʊ', 'ʌ', 'ɜ', 'ə', 'ɚ', 'ɤ', 'ʅ', 'ɿ',
    'aɪ', 'aʊ', 'eɪ', 'oʊ', 'ɔɪ', 'ɪə', 'ɛə', 'ʊə'
}

def get_vowels_from_tg(tg_path, phone_tier_name='phones'):
    tg = tgt.io.read_textgrid(str(tg_path))
    if phone_tier_name not in tg.get_tier_names():
        for name in ['phone', 'phones', 'MAU', 'words']:
            if name in tg.get_tier_names():
                phone_tier_name = name
                break
        else:
            return []
            
    tier = tg.get_tier_by_name(phone_tier_name)
    vowels = []
    for interval in tier:
        label = interval.text.strip().lower()
        # Clean label (remove tone numbers)
        clean_label = "".join([c for c in label if not c.isdigit()])
        if label in VOWEL_SET or clean_label in VOWEL_SET:
            vowels.append({
                'start_s': interval.start_time,
                'end_s': interval.end_time,
                'phone': interval.text
            })
    return vowels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--audio-dir', type=str, required=True)
    parser.add_argument('--textgrid-dir', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument(
        '--threshold', type=float, default=None,
        help=(
            'Decision threshold for creaky classification. '
            'Priority: this flag > checkpoint[best_threshold] > config.inference.threshold. '
            'Default: auto-resolved from the above chain.'
        )
    )
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--phone-tier', type=str, default='phones')
    parser.add_argument('--ext', type=str, default='wav', choices=['wav', 'flac'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument(
        '--task', type=str, default='creaky',
        help='Target task to predict (e.g. "creaky", "breathy"). Default: "creaky"'
    )
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
        
    print(f"Using device: {device}")
    
    # Load checkpoint
    try:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    except Exception:
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    config = ckpt['config']
    
    # Model
    model = model_mod.WavFryClassifier(config)
    
    # Load weights
    if config['checkpoint'].get('save_mode', 'delta') == 'delta':
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
    else:
        model.load_state_dict(ckpt['model_state_dict'])
        
    model.to(device)
    model.eval()
    
    # ── Threshold resolution (three-tier priority) ───────────────────────
    # 1. CLI --threshold (explicit user override) takes highest priority
    # 2. checkpoint['best_threshold'] from post-training grid search
    # 3. config.inference.threshold (static YAML fallback)
    if args.threshold is not None:
        threshold = args.threshold
        print(f"[Threshold] Using CLI value: {threshold:.3f}")
    elif 'best_threshold' in ckpt:
        threshold = float(ckpt['best_threshold'])
        threshold_f1 = ckpt.get('threshold_f1', None)
        msg = f"[Threshold] Using checkpoint best_threshold: {threshold:.3f}"
        if threshold_f1 is not None:
            msg += f"  (val F1={threshold_f1:.4f})"
        print(msg)
    else:
        threshold = float(config.get('inference', {}).get('threshold', 0.5))
        print(f"[Threshold] Using config.inference.threshold fallback: {threshold:.3f}")
    
    audio_dir = Path(args.audio_dir)
    tg_dir = Path(args.textgrid_dir)
    
    audio_files = list(audio_dir.glob(f"*.{args.ext}"))
    print(f"Found {len(audio_files)} {args.ext} files.")
    
    results = []
    
    # 1. Collect all segment metadata
    print("Collecting segment metadata from TextGrids...")
    all_segments = []
    for audio_path in tqdm(audio_files, desc="Parsing TextGrids"):
        file_key = audio_path.stem
        tg_path = tg_dir / f"{file_key}.TextGrid"
        if not tg_path.exists():
            continue
        try:
            vowels = get_vowels_from_tg(tg_path, args.phone_tier)
            for v in vowels:
                v['audio_path'] = audio_path
                v['file_key'] = file_key
                all_segments.append(v)
        except Exception as e:
            print(f"Error processing {tg_path}: {e}")
            
    if not all_segments:
        print("No segments found to predict.")
        return

    # 2. Dataset and Loader
    padding_ms = config.get('training', {}).get('segment', {}).get('context_padding_ms', 30)
    
    # Simple internal dataset for inference
    class InferenceDataset(torch.utils.data.Dataset):
        def __init__(self, segments, padding_ms, lowpass_cfg=None):
            self.segments = segments
            self.padding_ms = padding_ms
            self.target_sr = 16000
            self.resamplers = {}
            # Lowpass filter config (deterministic)
            self.lowpass_cfg = lowpass_cfg or {}
            self.lowpass_enabled = self.lowpass_cfg.get('enabled', False)
            self.lowpass_cutoff  = self.lowpass_cfg.get('cutoff_hz', 3000.0)
            self.lowpass_Q       = self.lowpass_cfg.get('Q', 0.707)
            
        def __len__(self):
            return len(self.segments)
            
        def __getitem__(self, idx):
            s = self.segments[idx]
            try:
                waveform, sr = torchaudio.load(s['audio_path'])
                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                if sr != self.target_sr:
                    if sr not in self.resamplers:
                        self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
                    waveform = self.resamplers[sr](waveform)
                
                # [Lowpass] Deterministic high-cut filter
                if self.lowpass_enabled:
                    waveform = torchaudio.functional.lowpass_biquad(
                        waveform, # [1, T]
                        sample_rate=self.target_sr,
                        cutoff_freq=self.lowpass_cutoff,
                        Q=self.lowpass_Q
                    )
                
                waveform = waveform.squeeze(0)
                total_len_s = waveform.shape[0] / self.target_sr
                
                s_pad = max(0, s['start_s'] - self.padding_ms / 1000.0)
                e_pad = min(s['end_s'] + self.padding_ms / 1000.0, total_len_s)
                
                start_frame = int(s_pad * self.target_sr)
                end_frame = int(e_pad * self.target_sr)
                segment = waveform[start_frame:end_frame]
                
                if segment.shape[0] == 0:
                    return None
                    
                return segment, idx
            except Exception as e:
                print(f"Error loading {s['audio_path']}: {e}")
                return None

    def collate_fn(batch):
        batch = [b for b in batch if b is not None]
        if not batch: return None
        waveforms, indices = zip(*batch)
        lengths = torch.tensor([w.shape[0] for w in waveforms])
        padded_waveforms = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True)
        return padded_waveforms, lengths, list(indices)

    lowpass_cfg = config.get('preprocessing', {}).get('lowpass', {})
    dataset = InferenceDataset(all_segments, padding_ms, lowpass_cfg=lowpass_cfg)
    loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        num_workers=dataset_mod.resolve_num_workers(args.num_workers),
        collate_fn=collate_fn
    )

    # 3. Batch Inference
    print(f"Starting prediction for {len(all_segments)} segments...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            if batch is None: continue
            waveforms, lengths, indices = batch
            waveforms, lengths = waveforms.to(device), lengths.to(device)
            
            outputs = model(waveforms, lengths)
            if isinstance(outputs, dict):
                if args.task in outputs:
                    task_logits = outputs[args.task]
                else:
                    first_task = next(iter(outputs.keys()))
                    print(f"[WARN] Task '{args.task}' not found in model outputs {list(outputs.keys())}. Falling back to '{first_task}'.")
                    task_logits = outputs[first_task]
            else:
                task_logits = outputs

            probs = torch.sigmoid(task_logits).cpu().float().numpy()
            
            for prob, idx in zip(probs, indices):
                seg_info = all_segments[idx]
                results.append({
                    'file_key': seg_info['file_key'],
                    'start_s': seg_info['start_s'],
                    'end_s': seg_info['end_s'],
                    'phone': seg_info['phone'],
                    'task': args.task,
                    'pred_label': 1 if prob >= threshold else 0,
                    'confidence': float(prob)
                })
                
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")

if __name__ == "__main__":
    main()
