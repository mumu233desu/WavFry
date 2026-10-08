"""Offline frame-head review queues. Scores are disagreement signals, not truth."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from wavfry.data.weak_manifest import manifest_digest, validate_source_stamps
from wavfry.eval.frame_quality import exposure, interval_mask, merge_frames, pooled_score, review_queues, window_starts
from wavfry.model.wavlm_classifier import WavFryClassifier


def score_file(model, rows, device, temperature=1.0, block_cfg=None):
    audio, sr = sf.read(rows[0]["audio_path"], dtype="float32", always_2d=True)
    waveform = torch.from_numpy(audio.mean(axis=1))
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    stride, receptive = 1, 1
    for kernel, step in zip(model.wavlm.config.conv_kernel, model.wavlm.config.conv_stride):
        receptive += (kernel - 1) * stride
        stride *= step
    block_cfg = block_cfg or {}
    block_samples = round(float(block_cfg.get('duration_ms', 2000)) * 16)
    hop_samples = round(float(block_cfg.get('hop_ms', 1000)) * 16)
    starts = window_starts(len(waveform), block=block_samples, hop=hop_samples, stride=stride)
    if len(waveform) < receptive:
        raise ValueError("Audio is shorter than the backbone receptive field")
    outputs, masks, class_scores = {}, [], {r["supervision_id"]: [] for r in rows}
    with torch.inference_mode():
        for start in starts:
            wave = waveform[start:start + block_samples]
            length = len(wave)
            wave = torch.nn.functional.pad(wave, (0, block_samples - length))
            contained = [r for r in rows if r["scope"] == "interval"
                         and r["start_s"] * 16000 >= start
                         and r["end_s"] * 16000 <= start + length]
            kwargs = {}
            if contained:
                bounds = torch.tensor([[[round(r["start_s"] * 16000) - start,
                                         round(r["end_s"] * 16000) - start] for r in contained]], device=device)
                kwargs = {"interval_bounds": bounds, "interval_valid": torch.ones(bounds.shape[:2], dtype=torch.bool, device=device)}
            out = model(wave.unsqueeze(0).to(device), torch.tensor([length], device=device), return_frames=True, **kwargs)
            masks.append(out["frame_valid"].cpu())
            for task, value in out["frame_logits"].items():
                outputs.setdefault(task, []).append(value.cpu())
            if contained:
                for slot, r in enumerate(contained):
                    if bool(out["interval_valid"][0, slot]):
                        class_scores[r["supervision_id"]].append(float(out["interval_logits"][r["task"]][0, slot]))
    merged = {task: merge_frames(torch.cat(values), torch.cat(masks), torch.tensor(starts),
                                torch.zeros(len(starts), dtype=torch.long), stride, receptive)[0]
              for task, values in outputs.items()}
    result = []
    for row in rows:
        item = merged[row["task"]]
        bounds = json.loads(row["vowel_intervals"]) if row["scope"] == "utterance" else [[row["start_s"], row["end_s"]]]
        mask = interval_mask(item["times"], bounds)
        record = dict(row)
        if not mask.any():
            record.update(frame_score=float("nan"), agreement=float("nan"), context_std=float("nan"), review_reason="no_candidate_frames")
            result.append(record)
            continue
        p = float(pooled_score(item["logits"], mask, temperature))
        label = float(row["label"])
        peak = item["logits"].masked_fill(~mask, -float("inf")).argmax()
        record.update(frame_score=p, agreement=(p if label == 1 else 1 - p) if np.isfinite(label) else float("nan"),
                      context_std=float(item["context_std"][mask].mean()), peak_time_s=float(item["times"][peak]),
                      interval_score=float(torch.tensor(class_scores[row["supervision_id"]]).mean().sigmoid()) if class_scores[row["supervision_id"]] else float("nan"),
                      review_reason="inspect_disagreement")
        raw_candidates = row.get("candidate_intervals", "[]")
        candidates = json.loads(raw_candidates) if isinstance(raw_candidates, str) else []
        candidate_bounds = [[c["start"], c["end"]] for c in candidates
                            if c.get("phonation_type", "breathy") == "breathy"]
        if row["scope"] == "utterance" and candidate_bounds:
            candidate_mask = interval_mask(item["times"], candidate_bounds)
            record["candidate_score"] = float(pooled_score(item["logits"], candidate_mask, temperature)) if candidate_mask.any() else float("nan")
            if p >= 0.5 and record["candidate_score"] < 0.5:
                record["review_reason"] = "location_only_review"
        record["frame_trace"] = json.dumps({"times": item["times"][mask].tolist(), "scores": item["logits"][mask].sigmoid().tolist()})
        result.append(record)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--weak-manifest", type=Path)
    parser.add_argument("--manual-manifest", type=Path)
    parser.add_argument("--fold-map", type=Path, help="JSON: dataset_id|speaker_id -> checkpoint path")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--task", default="breathy", choices=["breathy", "creaky"])
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh report directory; no old report is overwritten")
    inputs = []
    if args.weak_manifest:
        weak = pd.read_csv(args.weak_manifest)
        validate_source_stamps(weak)
        weak["scope"] = "utterance"
        inputs.extend(weak[weak.task == args.task].to_dict("records"))
    if args.manual_manifest:
        manual = pd.read_csv(args.manual_manifest)
        col = f"label_{args.task}"
        for row in manual[manual[col].notna()].to_dict("records"):
            row.update(task=args.task, scope="interval", source="human",
                       label=row[col], supervision_id=f"{row['dataset_id']}|{row['file_key']}|{row['start_s']:.4f}|{row['end_s']:.4f}|{args.task}")
            inputs.append(row)
    if not inputs:
        raise ValueError("No review inputs")
    fold_map = json.loads(args.fold_map.read_text(encoding="utf-8")) if args.fold_map else {}
    groups = {}
    for row in inputs:
        ckpt = fold_map.get(f"{row['dataset_id']}|{row['speaker_id']}", str(args.checkpoint))
        groups.setdefault(str(Path(ckpt).resolve()), []).append(row)
    results = []
    for path, rows in groups.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = WavFryClassifier(checkpoint["config"])
        strict = checkpoint["config"]["checkpoint"].get("save_mode", "delta") != "delta"
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if args.task not in model.tasks or not model.frame_head_enabled:
            raise ValueError(f"Checkpoint has no enabled {args.task} frame head")
        model.to(args.device).eval()
        frame_cfg = checkpoint["config"]["training"].get("block", {}).get("frame_head", {})
        if frame_cfg.get("mil_mode", "lse") != "lse":
            raise ValueError("First-version review supports LSE checkpoints only; do not silently change pooling")
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        for _, file_rows in pd.DataFrame(rows).groupby(["dataset_id", "file_key"], sort=True):
            records = score_file(model, file_rows.to_dict("records"), args.device, frame_cfg.get("temperature", 1.0),
                                 checkpoint['config']['training'].get('block', {}))
            for r in records:
                r.update(checkpoint=path, checkpoint_fingerprint=digest,
                         exposure=exposure(r, checkpoint.get("training_provenance")))
            results.extend(records)
        del model
    df = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True)
    df.to_csv(args.output_dir / "all_scores.csv", index=False)
    queues = review_queues(df, top_k=args.top_k)
    queues.to_csv(args.output_dir / "review_queues.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps({"rows": len(df), "exposure": df.exposure.value_counts().to_dict(),
        "weak_fingerprint": manifest_digest(weak) if args.weak_manifest else None,
        "notice": "Scores prioritize human review; they do not establish label correctness."}, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Annotation review\n\nQ is agreement, not a correctness probability.\n"
        "Use review_queues.csv for disagreement, instability, random and high-agreement controls.\n"
        "Held-out is asserted only from complete checkpoint training provenance.\n"
        "No labels have been changed. Frame traces and audio paths are in all_scores.csv.\n", encoding="utf-8")
    print(f"Wrote {len(df)} scores to {args.output_dir}")


if __name__ == "__main__":
    main()
