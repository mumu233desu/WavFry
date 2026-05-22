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

def build_manifest(config):
    speakers = config['data']['annotated_speakers']
    audio_dirs = config['data']['audio_dirs']
    tg_dirs = config['data']['textgrid_dirs']
    
    # Load all annotations
    # Assume English MFA TextGrid dir for now as primary source for manual annotations
    anno_path = annotation_storage_path(Path(tg_dirs['en']))
    print(f"Loading annotations from {anno_path}...")
    all_annotations = load_annotations(anno_path)
    
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
        if not speaker_id or speaker_id not in speakers:
            continue
            
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
                
            records.append({
                'file_key': file_key,
                'audio_path': str(audio_path),
                'tg_path': str(tg_path),
                'start_s': start_s,
                'end_s': end_s,
                'phone': phone,
                'speaker_id': speaker_id,
                'language': lang_key,
                'label': 1 if label == 'creaky' else 0
            })
            
    df = pd.DataFrame(records)
    return df
