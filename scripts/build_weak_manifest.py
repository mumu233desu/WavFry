"""Build positive-only utterance supervision from existing MLLM caches, offline.

Supported Modes:
1. Automated Registry Mode (Default, matching build_manifest.py):
   python scripts/build_weak_manifest.py [--dataset-id <id>] [--config <path>]
   - Reads config/datasets.yaml directly.
   - Automatically processes all enabled datasets providing breathy supervision.
   - Saves unified manifest to manifests/weak_breathy.csv (and manifests/weak_breathy_{dataset_id}.csv).

2. Manual CLI Mode (Backwards-compatible):
   python scripts/build_weak_manifest.py --dataset-id <id> --cache-dir <dir> --sampling-state <file> ...
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from wavfry.data.weak_manifest import build_weak_manifest, manifest_digest

logger = logging.getLogger("build_weak_manifest")


def _get_vowel_phones(language: str, custom_phones: str | None = None) -> set[str]:
    """Return default vowel inventory by language if not explicitly specified."""
    if custom_phones:
        return {p.strip().lower() for p in custom_phones.split(",") if p.strip()}
    lang = (language or "ja").lower()
    if lang in ("ja", "jp", "japanese"):
        return {"a", "i", "u", "e", "o", "um"}
    elif lang in ("en", "english"):
        return {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw"}
    return {"a", "e", "i", "o", "u"}


def process_dataset(
    *,
    dataset_id: str,
    cache_dir: Path,
    sampling_state: Path,
    audio_dir: Path,
    textgrid_dir: Path,
    vowel_phones: set[str],
    phone_tier: str = "phones",
    manual_manifest_path: Path | None = None,
    reviews_path: Path | None = None,
) -> pd.DataFrame:
    """Run build_weak_manifest for a single dataset with path validation."""
    manual_df = pd.read_csv(manual_manifest_path) if (manual_manifest_path and manual_manifest_path.exists()) else None
    reviews_df = pd.read_csv(reviews_path) if (reviews_path and reviews_path.exists()) else None

    return build_weak_manifest(
        dataset_id=dataset_id,
        cache_dir=cache_dir,
        sampling_state=sampling_state,
        audio_dir=audio_dir,
        textgrid_dir=textgrid_dir,
        vowel_phones=vowel_phones,
        phone_tier=phone_tier,
        manual_manifest=manual_df,
        reviews=reviews_df,
    )


def save_manifest_with_report(df: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Save dataframe and write accompanying json digest/report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    report: dict[str, Any] = {
        "fingerprint": manifest_digest(df),
        "rows": len(df),
        "status": df.import_status.value_counts().to_dict(),
        "partition_status": df.groupby(["batch_split", "import_status"]).size().to_dict(),
    }
    report["partition_status"] = {f"{a}/{b}": n for (a, b), n in report["partition_status"].items()}
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "datasets.yaml",
                        help="Path to datasets.yaml (default: config/datasets.yaml)")
    parser.add_argument("--dataset-id", help="Target dataset ID (default: all active breathy datasets in config)")
    parser.add_argument("--output", type=Path, help="Target output CSV (default: manifests/weak_breathy.csv)")

    # Manual override arguments (backwards-compatible with legacy CLI invocation)
    parser.add_argument("--cache-dir", type=Path, help="Explicit MLLM cache directory")
    parser.add_argument("--sampling-state", type=Path, help="Explicit sampling_state.json path")
    parser.add_argument("--audio-dir", type=Path, help="Explicit audio directory")
    parser.add_argument("--textgrid-dir", type=Path, help="Explicit TextGrid directory")
    parser.add_argument("--vowel-phones", help="Comma-separated vowel phones (e.g. 'a,i,u,e,o,um')")
    parser.add_argument("--phone-tier", default="phones", help="Phone tier name in TextGrid (default: phones)")
    parser.add_argument("--manual-manifest", type=Path, help="Optional manual manifest path")
    parser.add_argument("--reviews", type=Path, help="Explicit human clip reviews CSV")
    parser.add_argument("--no-clobber", action="store_true", help="Error if output files already exist")

    args = parser.parse_args()

    # Mode A: Legacy explicit CLI mode (when paths are supplied via CLI)
    if args.cache_dir and args.sampling_state and args.audio_dir and args.textgrid_dir:
        if not args.dataset_id:
            parser.error("--dataset-id is required when explicit directories are provided.")
        out_path = args.output or (ROOT / "manifests" / f"weak_{args.dataset_id}.csv")
        if args.no_clobber and (out_path.exists() or out_path.with_suffix(".json").exists()):
            raise FileExistsError(f"Output path already exists: {out_path}")

        vowels = _get_vowel_phones("ja", args.vowel_phones)
        df = process_dataset(
            dataset_id=args.dataset_id,
            cache_dir=args.cache_dir,
            sampling_state=args.sampling_state,
            audio_dir=args.audio_dir,
            textgrid_dir=args.textgrid_dir,
            vowel_phones=vowels,
            phone_tier=args.phone_tier,
            manual_manifest_path=args.manual_manifest,
            reviews_path=args.reviews,
        )
        report = save_manifest_with_report(df, out_path)
        print(f"[build_weak_manifest] Saved {len(df)} rows to {out_path}")
        print(json.dumps(report, indent=2))
        return

    # Mode B: Automated Registry Mode (Reads datasets.yaml)
    if not args.config.exists():
        print(f"Error: Config file not found at {args.config}")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        datasets_cfg = yaml.safe_load(f)

    manifest_meta = datasets_cfg.get("manifest", {})
    output_dir = ROOT / manifest_meta.get("output_dir", "manifests")
    default_out_filename = manifest_meta.get("weak_output_filename", "weak_breathy.csv")
    final_output_path = args.output if args.output else (output_dir / default_out_filename)
    save_per_dataset = manifest_meta.get("save_per_dataset", True)

    if args.no_clobber and (final_output_path.exists() or final_output_path.with_suffix(".json").exists()):
        raise FileExistsError(f"Output path already exists: {final_output_path}")

    all_dfs: list[pd.DataFrame] = []
    per_dataset_results: dict[str, pd.DataFrame] = {}

    for ds in datasets_cfg.get("datasets", []):
        if not ds.get("enabled", True):
            continue
        ds_id = ds["dataset_id"]
        if args.dataset_id and ds_id != args.dataset_id:
            continue

        schema = ds.get("label_schema", {})
        breathy_active = schema.get("breathy", False)
        if isinstance(breathy_active, dict):
            breathy_active = breathy_active.get("enabled", False)
        if not breathy_active:
            continue

        params = ds.get("adapter_params", {})
        from wavfry.data.path_resolver import resolve_corpus_path
        audio_dir = Path(resolve_corpus_path(params.get("audio_dir", "")) or "")
        tg_dir = Path(resolve_corpus_path(params.get("textgrid_dir", "")) or "")
        cache_base = Path(resolve_corpus_path(params.get("cache_dir", "")) or "") if params.get("cache_dir") else (tg_dir / "_breathy_cache")
        
        # Check mllm_cache directory
        mllm_cache = cache_base / "mllm_cache"
        if not mllm_cache.exists():
            mllm_cache = cache_base

        sampling_state = cache_base / "sampling_state.json"
        if not sampling_state.exists():
            sampling_state = cache_base.parent / "sampling_state.json"

        # Check if inference json files exist in mllm_cache
        has_inference_cache = False
        if mllm_cache.exists():
            for f in mllm_cache.glob("*.json"):
                if f.name not in ("sampling_state.json", "failed_files.json", "sampling_state_backup.json") and not f.name.startswith("wavfry_infer"):
                    has_inference_cache = True
                    break

        if not has_inference_cache or not sampling_state.exists() or not audio_dir.exists() or not tg_dir.exists():
            print(f"[build_weak_manifest] [{ds_id}] Notice: Missing MLLM cache or sampling state, skipping.")
            continue

        lang = params.get("language", "ja")
        vowels = _get_vowel_phones(lang, args.vowel_phones)
        phone_tier = params.get("phone_tier_name", args.phone_tier or "phones")

        # Optional manual manifest for cross-checking human annotations
        manual_manifest = output_dir / f"segments_{ds_id}.csv"
        if not manual_manifest.exists():
            manual_manifest = output_dir / "segments_multi.csv"

        print(f"[build_weak_manifest] [{ds_id}] Processing weak supervision from {mllm_cache}...")
        try:
            sub_df = process_dataset(
                dataset_id=ds_id,
                cache_dir=mllm_cache,
                sampling_state=sampling_state,
                audio_dir=audio_dir,
                textgrid_dir=tg_dir,
                vowel_phones=vowels,
                phone_tier=phone_tier,
                manual_manifest_path=manual_manifest if manual_manifest.exists() else None,
                reviews_path=args.reviews,
            )
            all_dfs.append(sub_df)
            per_dataset_results[ds_id] = sub_df
            print(f"  [{ds_id}] Extracted {len(sub_df)} weak clips ({sub_df.import_status.value_counts().to_dict()})")
        except Exception as e:
            print(f"[build_weak_manifest] [{ds_id}] Error processing dataset: {e}")

    if not all_dfs:
        print("[build_weak_manifest] Error: No eligible weak supervision data found.")
        sys.exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    report = save_manifest_with_report(combined_df, final_output_path)
    print(f"\n[build_weak_manifest] Primary weak manifest saved to {final_output_path} ({len(combined_df)} rows)")

    # Save per-dataset if requested
    if save_per_dataset:
        for ds_id, sub_df in per_dataset_results.items():
            sub_path = output_dir / f"weak_breathy_{ds_id}.csv"
            save_manifest_with_report(sub_df, sub_path)
            print(f"  [{ds_id}] {len(sub_df)} rows -> {sub_path}")

    print("\nSummary Report:")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
