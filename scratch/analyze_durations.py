import os
import sys
from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm

# Add project root directory to path to import 04_creaky_tools
root_dir = Path(r"e:\Corpus\CS-Dialogue\data\short_wav\WAVE")
sys.path.append(str(root_dir))

# Use __import__ for modules starting with a digit
manual_annotation = __import__('04_creaky_tools.src.manual_annotation', fromlist=['load_annotations', 'annotation_storage_path'])
load_annotations = manual_annotation.load_annotations
annotation_storage_path = manual_annotation.annotation_storage_path

def analyze_durations(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    tg_dirs = config['data']['textgrid_dirs']
    anno_path = annotation_storage_path(Path(tg_dirs['en']))
    all_annotations = load_annotations(anno_path)
    
    durations = []
    files_with_annotations = all_annotations.get('files', {})
    
    for file_key, entry in files_with_annotations.items():
        annotations = entry.get('annotations', {})
        for interval_id, label in annotations.items():
            if label not in ['creaky', 'non_creaky']:
                continue
            iv_parts = interval_id.split('|')
            if len(iv_parts) != 3:
                continue
            start_s, end_s = float(iv_parts[0]), float(iv_parts[1])
            durations.append((end_s - start_s) * 1000)
            
    df = pd.DataFrame(durations, columns=['dur_ms'])
    total = len(df)
    
    results = []
    for threshold in [0, 10, 20, 30, 40, 50]:
        count = len(df[df['dur_ms'] < threshold])
        percent = (count / total) * 100 if total > 0 else 0
        results.append({
            'Threshold (ms)': threshold,
            'Filtered Count': count,
            'Percentage': f"{percent:.2f}%"
        })
        
    print(f"Total Segments: {total}")
    print(pd.DataFrame(results).to_string(index=False))

if __name__ == "__main__":
    config_path = r"e:\Corpus\CS-Dialogue\data\short_wav\WAVE\06_wavfry\config\_defaults.yaml"
    analyze_durations(config_path)
