"""Inspect a WavFry creaky/breathy checkpoint and optionally score labeled rows.

This script is intended for a GPU workstation. It reports task-specific
scalar-mix weights without loading audio, and can evaluate a CSV of labeled
intervals on a selected device.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader


def load_checkpoint(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def report_scalar_mix(checkpoint: dict) -> dict:
    report = {}
    for name, value in checkpoint["model_state_dict"].items():
        if "scalar_mix_weights" not in name:
            continue
        weights = torch.softmax(value.detach().float(), dim=0)
        key = name.split(".")[-1]
        report[key] = {
            "weights": [float(x) for x in weights],
            "argmax_layer": int(weights.argmax()),
            "entropy": float(-(weights * weights.clamp_min(1e-12).log()).sum()),
            "effective_layers": float(1.0 / (weights.square().sum())),
        }
        print(f"{key}: " + " ".join(f"L{i}={float(w):.4f}" for i, w in enumerate(weights)))
        print(
            f"  argmax=L{int(weights.argmax())}, "
            f"entropy={report[key]['entropy']:.4f}, "
            f"effective_layers={report[key]['effective_layers']:.2f}"
        )
    return report


def evaluate_groups(checkpoint, csv_path: Path, device: torch.device, batch_size: int, out: Path | None):
    # Imports are delayed so weight-only inspection works without torchaudio.
    from wavfry.data.dataset import CreakyVowelDataset, collate_fn
    from wavfry.model.wavlm_classifier import WavFryClassifier

    data = pd.read_csv(csv_path)
    data = data[data["label_breathy"].notna()].copy().reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No labeled breathy rows in {csv_path}")

    config = checkpoint["config"]
    # The interval CSV is evaluated as independent segments, while the
    # checkpoint itself was trained with block batches.
    segment_config = json.loads(json.dumps(config))
    segment_config.setdefault("training", {})["paradigm"] = "segment"
    model = WavFryClassifier(segment_config)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if result.missing_keys:
        print(f"warning: missing keys={len(result.missing_keys)}")
    if result.unexpected_keys:
        print(f"warning: unexpected keys={len(result.unexpected_keys)}")
    model.to(device).eval()

    ds = CreakyVowelDataset(data, context_padding_ms=30)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    predictions = []
    with torch.inference_mode():
        for waveforms, _labels, lengths, _speakers, _meta in loader:
            waveforms = waveforms.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits = model(waveforms, lengths=lengths)["breathy"]
            predictions.extend(torch.sigmoid(logits).float().cpu().numpy().tolist())

    data["pred_breathy"] = predictions
    y = data["label_breathy"].astype(int)
    print(
        f"overall: n={len(data)} pos={int(y.sum())} "
        f"PR-AUC={average_precision_score(y, data.pred_breathy):.4f} "
        f"ROC-AUC={roc_auc_score(y, data.pred_breathy):.4f}"
    )
    for column in ["dataset_id", "language", "speaker_id", "phone", "phone_type"]:
        if column not in data:
            continue
        rows = []
        for value, group in data.groupby(column, dropna=False):
            if group["label_breathy"].nunique() < 2:
                continue
            rows.append({
                column: value,
                "n": len(group),
                "positive": int(group.label_breathy.sum()),
                "prauc": average_precision_score(group.label_breathy, group.pred_breathy),
                "roc_auc": roc_auc_score(group.label_breathy, group.pred_breathy),
                "mean_score": float(group.pred_breathy.mean()),
            })
        if rows:
            print(f"\n[{column}]\n{pd.DataFrame(rows).sort_values('n', ascending=False).to_string(index=False)}")
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(out, index=False)
        print(f"predictions written to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--csv", type=Path, help="Labeled interval CSV; omit for weight-only inspection")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--weights-json", type=Path, default=None)
    args = parser.parse_args()

    checkpoint = load_checkpoint(args.checkpoint)
    print(f"checkpoint={args.checkpoint}")
    print(f"epoch={checkpoint.get('epoch')} metrics={checkpoint.get('metrics', {})}")
    scalar_mix = report_scalar_mix(checkpoint)
    if args.weights_json:
        args.weights_json.parent.mkdir(parents=True, exist_ok=True)
        args.weights_json.write_text(json.dumps(scalar_mix, indent=2), encoding="utf-8")
    if args.csv:
        evaluate_groups(checkpoint, args.csv, torch.device(args.device), args.batch_size, args.out)


if __name__ == "__main__":
    main()
