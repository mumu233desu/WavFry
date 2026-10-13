"""Complete utterances encoded as fixed blocks; supports unpaired and paired bags."""
from __future__ import annotations

from collections import defaultdict
import json
import math
import random

import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, Sampler

from wavfry.eval.frame_quality import interval_mask, merge_frames, window_starts
from wavfry.model.losses import logmeanexp_pool


class UtteranceBagDataset(Dataset):
    """
    Dataset of complete utterance bags sliced into fixed-length windows.
    Supports both positive-only bags (legacy) and paired positive/modal negative bags (VSDC-Net).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        block_ms: int = 2000,
        hop_ms: int = 1000,
        frame_stride: int = 320,
        receptive: int = 400,
        max_windows: int = 32,
        allow_modal: bool = False,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.block = round(block_ms * 16)
        self.hop = round(hop_ms * 16)
        self.stride = int(frame_stride)
        self.receptive = int(receptive)
        self.allow_modal = bool(allow_modal)
        self.starts = []
        self.costs = []
        self.bag_labels = []

        self.speaker_pos_indices = defaultdict(list)
        self.speaker_neg_indices = defaultdict(list)
        self.pos_indices = []
        self.neg_indices = []

        for idx, row in enumerate(self.df.to_dict("records")):
            is_pos = (row.get("task") == "breathy" and float(row.get("label", 0) or 0) == 1 and row.get("import_status") == "eligible")
            is_modal = (self.allow_modal and (row.get("reason") == "no_positive_evidence" or float(row.get("label", 0) or 0) == 0))

            if not is_pos and not is_modal:
                raise ValueError("Utterance training accepts eligible breathy positives (or modal negatives when enabled)")

            info = sf.info(row["audio_path"])
            length = math.ceil(info.frames * 16000 / info.samplerate)
            if abs(info.duration - float(row["end_s"])) > .05 or float(row["start_s"]) != 0:
                raise ValueError(f"Stale utterance length: {row['file_key']}")
            if length < self.receptive:
                raise ValueError(f"Clip shorter than receptive field: {row['file_key']}")
            bounds = json.loads(row["vowel_intervals"])
            if not bounds:
                raise ValueError(f"No vowel candidates: {row['file_key']}")
            starts = window_starts(length, self.block, self.hop, self.stride)
            if len(starts) > max_windows:
                raise ValueError(
                    f"Complete clip needs {len(starts)} windows, budget={max_windows}: {row['file_key']}; "
                    f"increase max_windows, never truncate"
                )

            self.starts.append(starts)
            self.costs.append(len(starts))
            spk = str(row.get("speaker_id", ""))

            if is_pos:
                self.bag_labels.append(1.0)
                self.pos_indices.append(idx)
                self.speaker_pos_indices[spk].append(idx)
            else:
                self.bag_labels.append(0.0)
                self.neg_indices.append(idx)
                self.speaker_neg_indices[spk].append(idx)

    def __len__(self):
        return len(self.df)

    def _get_single_item(self, index: int):
        row = self.df.iloc[index].to_dict()
        audio, sr = sf.read(row["audio_path"], dtype="float32", always_2d=True)
        wave = torch.from_numpy(audio.mean(axis=1))
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        starts = self.starts[index]
        windows, lengths = [], []
        for start in starts:
            block = wave[start:start + self.block]
            lengths.append(len(block))
            windows.append(torch.nn.functional.pad(block, (0, self.block - len(block))))
        return {
            "waveforms": torch.stack(windows),
            "lengths": torch.tensor(lengths),
            "starts": torch.tensor(starts),
            "vowels": json.loads(row["vowel_intervals"]),
            "bag_label": float(self.bag_labels[index]),
            "metadata": row,
        }

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
            pos_idx, neg_idx = index
            pos_item = self._get_single_item(pos_idx)
            neg_item = self._get_single_item(neg_idx) if (neg_idx is not None and neg_idx >= 0) else None
            return {"pos": pos_item, "neg": neg_item}
        return self._get_single_item(index)


class BagBatchSampler(Sampler):
    """Legacy sampler for single utterance bags."""
    def __init__(self, dataset: UtteranceBagDataset, max_windows=32, samples_per_epoch=64, seed=42):
        self.dataset = dataset
        self.max_windows = int(max_windows)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 1
        if min(self.max_windows, self.samples_per_epoch) < 1:
            raise ValueError("Positive batch/epoch budgets are required")

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def batches(self):
        order = list(range(len(self.dataset)))
        random.Random(self.seed + self.epoch * 1009).shuffle(order)
        order = order[:self.samples_per_epoch]
        result, batch, cost = [], [], 0
        for idx in order:
            count = self.dataset.costs[idx]
            if count > self.max_windows:
                raise ValueError("A complete utterance exceeds batch window budget")
            if batch and cost + count > self.max_windows:
                result.append(batch)
                batch, cost = [], 0
            batch.append(idx)
            cost += count
        if batch:
            result.append(batch)
        return result

    def __len__(self):
        return len(self.batches())

    def __iter__(self):
        return iter(self.batches())


class PairedBagBatchSampler(Sampler):
    """
    VSDC-Net paired sampler:
    For each positive bag, samples a same-speaker negative modal bag (if available).
    If the speaker has no modal clips (e.g. C0125), neg_idx is set to -1.
    """
    def __init__(
        self,
        dataset: UtteranceBagDataset,
        max_windows: int = 32,
        samples_per_epoch: int = 64,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.max_windows = int(max_windows)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 1

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def batches(self):
        rng = random.Random(self.seed + self.epoch * 1009)
        pos_order = list(self.dataset.pos_indices)
        rng.shuffle(pos_order)
        pos_order = pos_order[:self.samples_per_epoch]

        pairs = []
        for p_idx in pos_order:
            spk = str(self.dataset.df.iloc[p_idx].get("speaker_id", ""))
            neg_candidates = self.dataset.speaker_neg_indices.get(spk, [])
            if neg_candidates:
                n_idx = rng.choice(neg_candidates)
            else:
                n_idx = -1
            pairs.append((p_idx, n_idx))

        result, batch, cost = [], [], 0
        for p_idx, n_idx in pairs:
            p_cost = self.dataset.costs[p_idx]
            n_cost = self.dataset.costs[n_idx] if n_idx >= 0 else 0
            pair_cost = p_cost + n_cost
            if pair_cost > self.max_windows:
                # If positive + negative exceeds budget, drop negative to prevent budget overflow
                if p_cost <= self.max_windows:
                    n_idx = -1
                    n_cost = 0
                    pair_cost = p_cost
                else:
                    # Positive alone exceeds max_windows budget; skip
                    continue

            if batch and cost + pair_cost > self.max_windows:
                result.append(batch)
                batch, cost = [], 0
            batch.append((p_idx, n_idx))
            cost += pair_cost

        if batch:
            result.append(batch)
        return result

    def __len__(self):
        return len(self.batches())

    def __iter__(self):
        return iter(self.batches())


def collate_utterance_bags(items):
    """Collate a list of single utterance items."""
    return {
        "waveforms": torch.cat([item["waveforms"] for item in items]),
        "lengths": torch.cat([item["lengths"] for item in items]),
        "starts": torch.cat([item["starts"] for item in items]),
        "bag_index": torch.cat([
            torch.full((len(item["starts"]),), i, dtype=torch.long)
            for i, item in enumerate(items)
        ]),
        "vowels": [item["vowels"] for item in items],
        "bag_labels": torch.tensor([item["bag_label"] for item in items], dtype=torch.float32),
        "metadata": [item["metadata"] for item in items],
    }


def collate_paired_utterance_bags(items):
    """
    Collate paired positive/negative items.
    If items are legacy single bags, falls back to collate_utterance_bags.
    """
    if not items or "pos" not in items[0]:
        return collate_utterance_bags(items)

    pos_items = [it["pos"] for it in items]
    pos_batch = collate_utterance_bags(pos_items)

    neg_items = []
    has_same_speaker_neg = []
    pos_to_neg_idx = []

    for it in items:
        neg = it.get("neg")
        if neg is not None:
            neg_items.append(neg)
            has_same_speaker_neg.append(True)
            pos_to_neg_idx.append(len(neg_items) - 1)
        else:
            has_same_speaker_neg.append(False)
            pos_to_neg_idx.append(-1)

    neg_batch = collate_utterance_bags(neg_items) if neg_items else None

    return {
        "pos": pos_batch,
        "neg": neg_batch,
        "has_same_speaker_neg": torch.tensor(has_same_speaker_neg, dtype=torch.bool),
        "pos_to_neg_idx": torch.tensor(pos_to_neg_idx, dtype=torch.long),
        "metadata": [it["pos"]["metadata"] for it in items],
    }


def merge_frame_features(features, valid, starts, bag_index, stride, receptive, sample_rate=16000):
    """
    Vectorized & fully differentiable stitching of windowed frame features across an utterance.
    features: [N_windows, T_frames, D]
    valid: [N_windows, T_frames] (bool)
    starts: [N_windows] (sample index)
    bag_index: [N_windows] (utterance index)
    """
    results = []
    D = features.size(-1)
    unique_bags = torch.unique(bag_index, sorted=True)
    for bag in unique_bags:
        selected = bag_index == bag
        rows = selected.nonzero(as_tuple=False).flatten()
        values, positions = [], []
        for row in rows:
            idx = valid[row].nonzero(as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            values.append(features[row, idx])
            positions.append(idx + starts[row] // stride)
        if not values:
            raise ValueError(f"No valid frames in utterance {int(bag)}")
        values = torch.cat(values, dim=0)  # [total_valid, D]
        positions = torch.cat(positions, dim=0)
        bins, inverse = torch.unique(positions, sorted=True, return_inverse=True)
        counts = torch.zeros(len(bins), 1, device=features.device, dtype=features.dtype).index_add(
            0, inverse, torch.ones(len(inverse), 1, device=features.device, dtype=features.dtype)
        )
        means = torch.zeros(len(bins), D, device=features.device, dtype=features.dtype).index_add(
            0, inverse, values
        ) / counts.clamp_min(1.0)
        times = (bins.float() * stride + receptive / 2) / sample_rate
        results.append({"bag_index": int(bag), "features": means, "times": times})
    return results


def extract_vowel_instances(merged_features, vowels_per_bag):
    """
    Extracts a padded tensor of vowel representations h_vowels: [B, K_max, D] and
    vowel_mask: [B, K_max] from merged utterance features.
    """
    B = len(merged_features)
    device = merged_features[0]["features"].device
    dtype = merged_features[0]["features"].dtype
    D = merged_features[0]["features"].size(-1)

    all_vowel_vecs = []
    for item in merged_features:
        bag_idx = item["bag_index"]
        times = item["times"]
        feats = item["features"]
        vowels = vowels_per_bag[bag_idx]

        v_list = []
        for start_s, end_s in vowels:
            start_f = float(start_s)
            end_f = float(end_s)
            in_vowel = (times >= start_f) & (times < end_f)
            if in_vowel.any():
                v_repr = feats[in_vowel].mean(dim=0)
            elif times.numel() > 0:
                mid_t = (start_f + end_f) / 2.0
                nearest_idx = (times - mid_t).abs().argmin()
                v_repr = feats[nearest_idx]
            else:
                v_repr = torch.zeros(D, device=device, dtype=dtype)
            v_list.append(v_repr)
        if not v_list:
            v_list = [torch.zeros(D, device=device, dtype=dtype)]
        all_vowel_vecs.append(torch.stack(v_list))

    K_max = max(v.size(0) for v in all_vowel_vecs)
    h_vowels = torch.zeros(B, K_max, D, device=device, dtype=dtype)
    vowel_mask = torch.zeros(B, K_max, device=device, dtype=torch.bool)
    for b, v_tensor in enumerate(all_vowel_vecs):
        k = v_tensor.size(0)
        h_vowels[b, :k] = v_tensor
        vowel_mask[b, :k] = True

    return h_vowels, vowel_mask


def utterance_logits(outputs, batch, task="breathy", temperature=1.0):
    merged = merge_frames(outputs["frame_logits"][task], outputs["frame_valid"],
                          batch["starts"], batch["bag_index"], outputs["frame_stride_samples"],
                          outputs["frame_receptive_samples"])
    logits = []
    for item in merged:
        mask = interval_mask(item["times"], batch["vowels"][item["bag_index"]])
        if not mask.any():
            raise ValueError("No valid vowel frames in complete utterance")
        logits.append(logmeanexp_pool(item["logits"].unsqueeze(0), mask.unsqueeze(0), temperature)[0])
    return torch.stack(logits)
