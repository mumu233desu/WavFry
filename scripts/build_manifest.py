import sys
from pathlib import Path
import yaml

# Add the project directory to sys.path to allow 'import wavfry'
root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from wavfry.data.manifest import build_manifest, build_manifest_multi


def main():
    config_path = root / "config" / "_defaults.yaml"
    datasets_path = root / "config" / "datasets.yaml"

    config = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    if datasets_path.exists():
        with open(datasets_path, "r", encoding="utf-8") as f:
            datasets_cfg = yaml.safe_load(f)
        print(f"[build_manifest] Loading multi-dataset registry: {datasets_path}")
        df = build_manifest_multi(datasets_cfg, global_cfg=config)
        out_filename = datasets_cfg.get("manifest", {}).get("output_filename", "segments_multi.csv")
    else:
        print(f"[build_manifest] Fallback to legacy single dataset mode using: {config_path}")
        df = build_manifest(config)
        out_filename = "segments_all.csv"

    if df.empty:
        print("Warning: Manifest is empty!")
        sys.exit(1)

    output_dir = root / "manifests"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / out_filename
    df.to_csv(output_path, index=False)
    print(f"[build_manifest] Primary manifest saved to {output_path} ({len(df)} rows)")

    # Also save per-dataset if configured
    if datasets_path.exists() and datasets_cfg.get("manifest", {}).get("save_per_dataset", False):
        for ds_id, sub_df in df.groupby("dataset_id"):
            sub_path = output_dir / f"segments_{ds_id}.csv"
            sub_df.to_csv(sub_path, index=False)
            print(f"  [{ds_id}] {len(sub_df)} rows -> {sub_path}")

    # Also keep segments_all.csv updated for backward compatibility
    legacy_output = output_dir / "segments_all.csv"
    df.to_csv(legacy_output, index=False)
    print(f"[build_manifest] Legacy compatibility manifest saved to {legacy_output}")


if __name__ == "__main__":
    main()
