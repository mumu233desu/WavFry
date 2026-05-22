import sys
from pathlib import Path
import yaml

# Add the 06_wavfry directory to sys.path to allow 'import wavfry'
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.append(str(root))

from wavfry.data.manifest import build_manifest

def main():
    config_path = root / "config" / "_defaults.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    df = build_manifest(config)
    
    if df.empty:
        print("Warning: Manifest is empty!")
        sys.exit(1)
        
    output_dir = root / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "segments_all.csv"
    df.to_csv(output_path, index=False)
    print(f"Manifest saved to {output_path}")
    print(f"Total segments: {len(df)}")
    print(df['label'].value_counts())
    print(df.groupby('speaker_id')['label'].value_counts().unstack().fillna(0))

if __name__ == "__main__":
    main()
