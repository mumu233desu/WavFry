import os
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# Add project root directory to path to import 04_creaky_tools
sys.path.append(str(Path(__file__).resolve().parents[3]))

# Use __import__ for modules starting with a digit
manual_annotation = __import__('04_creaky_tools.src.manual_annotation', fromlist=['load_annotations', 'annotation_storage_path'])
load_annotations = manual_annotation.load_annotations
annotation_storage_path = manual_annotation.annotation_storage_path

def _build_batch_lookup(sampling_state_path: Path) -> dict:
    """Return a dict mapping file_key -> batch_id (int) from sampling_state.json."""
    lookup = {}
    if not sampling_state_path.exists():
        print(f"[manifest] WARNING: sampling_state.json not found at {sampling_state_path}. batch_id will be NaN for all sampled files.")
        return lookup
    try:
        import json as _json
        with open(sampling_state_path, 'r', encoding='utf-8') as f:
            state = _json.load(f)
        for batch in state.get('batches', []):
            batch_id = int(batch.get('batch_id', 0))
            for file_entry in batch.get('files', []):
                fk = file_entry.get('file_key')
                if fk:
                    lookup[fk] = batch_id
        print(f"[manifest] Loaded batch lookup: {len(lookup)} file_keys across {len(state.get('batches', []))} batches.")
    except Exception as e:
        print(f"[manifest] WARNING: Failed to load sampling_state.json: {e}")
    return lookup


def build_manifest(config):
    speakers = config['data']['annotated_speakers']
    audio_dirs = config['data']['audio_dirs']
    tg_dirs = config['data']['textgrid_dirs']
    
    # Load all annotations
    # Assume English MFA TextGrid dir for now as primary source for manual annotations
    anno_path = annotation_storage_path(Path(tg_dirs['en']))
    print(f"Loading annotations from {anno_path}...")
    all_annotations = load_annotations(anno_path)
    
    # Build batch_id lookup from sampling_state.json (sibling folder 04_creaky_tools)
    sampling_state_path = Path(__file__).resolve().parents[3] / '04_creaky_tools' / '_annotation_sampling' / 'sampling_state.json'
    batch_lookup = _build_batch_lookup(sampling_state_path)
    
    records = []
    files_with_annotations = all_annotations.get('files', {})
    
    for file_key, entry in tqdm(files_with_annotations.items(), desc="Processing annotations"):
        # Check if speaker is in our target list
        # Key format: Gender_Age_Region_Language_SpeakerID_SessionID_UtteranceID
        parts = file_key.split('_')
        if len(parts) < 5:
            continue
        
        # DEBUG
        # print(f"parts: {parts}")
        
        # If parts[4] is ZH-CN, speaker is parts[5]
        # Let's try to find speaker ID by looking for the one starting with U
        speaker_id = next((p for p in parts if p.startswith('U')), None)
        if not speaker_id:
            continue
            
        annotation_type = "full" if speaker_id in speakers else "sampled"
            
        language_code = parts[3] # EN or ZH
        lang_key = 'en' if language_code == 'EN' else 'zh'
        
        audio_dir = Path(audio_dirs[lang_key])
        tg_dir = Path(tg_dirs[lang_key])
        
        # Construct paths
        audio_path = audio_dir / f"{file_key}.wav"
        if not audio_path.exists():
            audio_path = audio_dir / f"{file_key}.flac"
        
        tg_path = tg_dir / f"{file_key}.TextGrid"
        
        if not audio_path.exists() or not tg_path.exists():
            continue
            
        annotations = entry.get('annotations', {})
        for interval_id, label in annotations.items():
            if label not in ['creaky', 'non_creaky']:
                continue
                
            # interval_id: start|end|phone
            iv_parts = interval_id.split('|')
            if len(iv_parts) != 3:
                continue
                
            start_s, end_s, phone = float(iv_parts[0]), float(iv_parts[1]), iv_parts[2]
            
            # Duration check
            if (end_s - start_s) * 1000 < config['data']['min_duration_ms']:
                continue
                
            # Determine batch_id: sampled files have a batch, full-speaker files do not
            batch_id = batch_lookup.get(file_key, None)  # None for full-speaker data
            
            records.append({
                'file_key': file_key,
                'audio_path': str(audio_path),
                'tg_path': str(tg_path),
                'start_s': start_s,
                'end_s': end_s,
                'phone': phone,
                'speaker_id': speaker_id,
                'language': lang_key,
                'label': 1 if label == 'creaky' else 0,
                'annotation_type': annotation_type,
                'batch_id': batch_id  # int (1 or 2) for sampled files, None for full-speaker files
            })
            
    df = pd.DataFrame(records)
    return df
