"""Differentiable frame merging shared by weak training and quality review."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch

from wavfry.model.losses import logmeanexp_pool


def window_starts(length: int, block: int = 32000, hop: int = 16000, stride: int = 320) -> list[int]:
    if min(length, block, hop, stride) <= 0 or hop > block or block % stride or hop % stride:
        raise ValueError("Positive aligned windows with hop <= block are required")
    if length <= block:
        return [0]
    starts = list(range(0, length - block + 1, hop))
    tail = math.ceil((length - block) / stride) * stride
    if tail not in starts:
        starts.append(tail)
    return sorted(starts)


def merge_frames(logits, valid, starts, bag_index, stride, receptive, sample_rate=16000):
    """Average logits on a common absolute frame grid, counting each time once.

    starts must lie on the backbone's stride grid. Context disagreement is a
    diagnostic, not additional training weight. index_add keeps autograd.
    """
    results = []
    for bag in torch.unique(bag_index, sorted=True):
        selected = bag_index == bag
        rows = selected.nonzero(as_tuple=False).flatten()
        values, positions = [], []
        for row in rows:
            if int(starts[row]) % stride:
                raise ValueError("Window starts must be aligned to frame stride")
            idx = valid[row].nonzero(as_tuple=False).flatten()
            values.append(logits[row, idx].float())
            positions.append(idx + starts[row] // stride)
        values, positions = torch.cat(values), torch.cat(positions)
        bins, inverse = torch.unique(positions, sorted=True, return_inverse=True)
        counts = torch.zeros(len(bins), device=logits.device).index_add(0, inverse, torch.ones_like(values))
        means = torch.zeros_like(counts).index_add(0, inverse, values) / counts
        squares = torch.zeros_like(counts).index_add(0, inverse, values.square()) / counts
        std = (squares - means.square()).clamp_min(0).sqrt()
        times = (bins.float() * stride + receptive / 2) / sample_rate
        results.append({"bag_index": int(bag), "logits": means, "times": times,
                        "context_std": std, "counts": counts})
    return results


def interval_mask(times: torch.Tensor, intervals) -> torch.Tensor:
    mask = torch.zeros_like(times, dtype=torch.bool)
    for start, end in intervals:
        if end <= start:
            raise ValueError("Invalid candidate interval")
        contained = (times >= float(start)) & (times < float(end))
        if not contained.any() and times.numel():
            inside_clip = end > float(times.min()) and start <= float(times.max())
            if inside_clip:
                contained[(times - (float(start) + float(end)) / 2).abs().argmin()] = True
        mask |= contained
    return mask


def pooled_score(logits, mask, temperature=1.0):
    if not mask.any():
        raise ValueError("No usable candidate frames")
    return torch.sigmoid(logmeanexp_pool(logits.unsqueeze(0), mask.unsqueeze(0), temperature))[0]


def exposure(row, provenance=None):
    if not provenance:
        return "unknown"
    ident = [str(row["dataset_id"]), str(row["speaker_id"])]
    files = {tuple(v) for v in provenance.get("training_files", [])}
    speakers = {tuple(v) for v in provenance.get("training_speakers", [])}
    if (str(row["dataset_id"]), str(row["file_key"])) in files or tuple(ident) in speakers:
        return "training_seen"
    if provenance.get("complete") is True:
        return "held_out"
    return "unknown"


def review_queues(df: pd.DataFrame, top_k=20, random_k=5, seed=42) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    parts = []
    for _, group in df.groupby(["source", "task", "label"], dropna=False):
        rankings = {
            "low_agreement": group.dropna(subset=["agreement"]).nsmallest(top_k, "agreement"),
            "unstable_context": group.dropna(subset=['context_std']).nlargest(top_k, "context_std"),
            "high_agreement": group.dropna(subset=["agreement"]).nlargest(random_k, "agreement"),
            "random_control": group.sample(n=min(random_k, len(group)), random_state=seed),
        }
        if group['agreement'].isna().any():
            rankings['unscored_or_uncertain'] = group[group['agreement'].isna()]
        for kind, rows in rankings.items():
            subset = rows.copy()
            subset["queue"] = kind
            parts.append(subset)
    return pd.concat(parts, ignore_index=True)
