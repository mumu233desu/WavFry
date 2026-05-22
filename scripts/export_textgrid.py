import argparse
import pandas as pd
import tgt
from pathlib import Path
from tqdm import tqdm

def export_to_textgrid(csv_path, tg_dir, output_dir, tier_name='wavfry'):
    df = pd.read_csv(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group by file_key
    grouped = df.groupby('file_key')
    
    for file_key, group in tqdm(grouped, desc="Exporting TextGrids"):
        tg_path = Path(tg_dir) / f"{file_key}.TextGrid"
        if not tg_path.exists():
            continue
            
        try:
            tg = tgt.io.read_textgrid(str(tg_path))
            
            # Remove existing tier if any
            if tier_name in tg.get_tier_names():
                tg.remove_tier(tg.get_tier_by_name(tier_name))
                
            new_tier = tgt.IntervalTier(start_time=0, end_time=tg.end_time, name=tier_name)
            
            for _, row in group.iterrows():
                # label could be 'creaky' or 'non_creaky'
                label = "creaky" if row['pred_label'] == 1 else "non_creaky"
                # Add confidence if available
                if 'confidence' in row:
                    label += f"({row['confidence']:.2f})"
                
                interval = tgt.Interval(start_time=row['start_s'], end_time=row['end_s'], text=label)
                # Ensure no overlap with existing intervals in the same tier
                # (though in our case vowels shouldn't overlap)
                new_tier.add_interval(interval)
                
            tg.add_tier(new_tier)
            
            out_path = output_dir / f"{file_key}.TextGrid"
            tgt.io.write_to_file(tg, str(out_path), format='long')
        except Exception as e:
            print(f"Error processing {file_key}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, required=True, help="Path to prediction CSV")
    parser.add_argument('--tg-dir', type=str, required=True, help="Directory containing original TextGrids")
    parser.add_argument('--output-dir', type=str, required=True, help="Directory to save modified TextGrids")
    parser.add_argument('--tier-name', type=str, default='wavfry', help="Name of the new tier")
    args = parser.parse_args()
    
    export_to_textgrid(args.csv, args.tg_dir, args.output_dir, args.tier_name)
