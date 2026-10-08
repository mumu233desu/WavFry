"""Build positive-only utterance supervision from existing MLLM caches, offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wavfry.data.weak_manifest import build_weak_manifest, manifest_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-id", required=True)
    for name in ("cache-dir", "sampling-state", "audio-dir", "textgrid-dir", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--manual-manifest", type=Path)
    parser.add_argument("--reviews", type=Path, help="Explicit human clip reviews CSV")
    parser.add_argument("--vowel-phones", required=True, help="Comma-separated inventory; Japanese TIFA: a,i,u,e,o,um")
    parser.add_argument("--phone-tier", default="phones")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError("Choose a fresh output path; existing manifests are not overwritten")
    df = build_weak_manifest(
        dataset_id=args.dataset_id, cache_dir=args.cache_dir, sampling_state=args.sampling_state,
        audio_dir=args.audio_dir, textgrid_dir=args.textgrid_dir,
        vowel_phones={p.strip().lower() for p in args.vowel_phones.split(",") if p.strip()},
        phone_tier=args.phone_tier,
        manual_manifest=pd.read_csv(args.manual_manifest) if args.manual_manifest else None,
        reviews=pd.read_csv(args.reviews) if args.reviews else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    report = {"fingerprint": manifest_digest(df), "rows": len(df),
              "status": df.import_status.value_counts().to_dict(),
              "partition_status": df.groupby(["batch_split", "import_status"]).size().to_dict()}
    report["partition_status"] = {f"{a}/{b}": n for (a, b), n in report["partition_status"].items()}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
