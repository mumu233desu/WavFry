"""Complete utterances encoded as fixed blocks; no label is assigned per block."""
from __future__ import annotations

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
    def __init__(self, df: pd.DataFrame, block_ms=2000, hop_ms=1000, frame_stride=320,
                 receptive=400, max_windows=32):
        self.df = df.reset_index(drop=True).copy()
        self.block = round(block_ms * 16)
        self.hop = round(hop_ms * 16)
        self.stride = int(frame_stride)
        self.receptive = int(receptive)
        self.starts = []
        self.costs = []
        for row in self.df.to_dict("records"):
            if row["task"] != "breathy" or float(row["label"]) != 1 or row["import_status"] != "eligible":
                raise ValueError("Utterance training accepts eligible breathy positives only")
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
                raise ValueError(f"Complete clip needs {len(starts)} windows, budget={max_windows}: {row['file_key']}; increase the budget, never truncate")
            self.starts.append(starts)
            self.costs.append(len(starts))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
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
        return dict(waveforms=torch.stack(windows), lengths=torch.tensor(lengths),
                    starts=torch.tensor(starts), vowels=json.loads(row["vowel_intervals"]), metadata=row)


class BagBatchSampler(Sampler):
    def __init__(self, dataset, max_windows=32, samples_per_epoch=64, seed=42):
        self.dataset = dataset
        self.max_windows = int(max_windows)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.epoch = 1
        if min(self.max_windows, self.samples_per_epoch) < 1:
            raise ValueError("Positive batch/epoch budgets are required")

    def set_epoch(self, epoch):
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


def collate_utterance_bags(items):
    return {
        "waveforms": torch.cat([item["waveforms"] for item in items]),
        "lengths": torch.cat([item["lengths"] for item in items]),
        "starts": torch.cat([item["starts"] for item in items]),
        "bag_index": torch.cat([torch.full((len(item["starts"]),), i, dtype=torch.long) for i, item in enumerate(items)]),
        "vowels": [item["vowels"] for item in items],
        "metadata": [item["metadata"] for item in items],
    }


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
