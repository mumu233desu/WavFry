"""Continuous fixed-length audio blocks with interval-level supervision."""

from __future__ import annotations

import math
import random
import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
_WARNED_INTERVALS: set[str] = set()


class BlockIndex:
    """Build deterministic overlapping blocks without crossing audio files."""

    def __init__(
        self,
        df: pd.DataFrame,
        duration_ms: float = 2000.0,
        hop_ms: float = 1000.0,
        max_intervals: int = 8,
        seed: int = 42,
        fold_index: int = 0,
    ) -> None:
        if duration_ms <= 0 or hop_ms <= 0:
            raise ValueError("Block duration and hop must be positive")
        if hop_ms > duration_ms:
            raise ValueError("Block hop must not exceed block duration")
        if max_intervals < 1:
            raise ValueError("max_intervals must be at least 1")

        self.duration_s = float(duration_ms) / 1000.0
        self.hop_s = float(hop_ms) / 1000.0
        self.max_intervals = int(max_intervals)
        self.seed = int(seed)
        self.fold_index = int(fold_index)
        self.blocks = self._build(df.reset_index(drop=True))

    @staticmethod
    def _interval_id(row: dict[str, Any]) -> str:
        return (
            f"{row['file_key']}|{float(row['start_s']):.4f}|"
            f"{float(row['end_s']):.4f}|{row.get('phone', 'UNK')}"
        )

    def _select_intervals(self, rows: list[dict[str, Any]], primary: str | None, rng: random.Random):
        if len(rows) <= self.max_intervals:
            return rows
        primary_row = next((row for row in rows if row["interval_id"] == primary), None)
        others = [row for row in rows if row is not primary_row]
        rng.shuffle(others)
        selected = ([primary_row] if primary_row is not None else []) + others
        return sorted(selected[: self.max_intervals], key=lambda row: row["start_s"])

    def _build(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        if df.empty:
            return []
        required = {"file_key", "audio_path", "start_s", "end_s", "speaker_id"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Block index requires columns: {sorted(missing)}")

        blocks: dict[tuple[str, ...], dict[str, Any]] = {}
        for file_key, group in df.groupby("file_key", sort=True):
            file_hash = int(hashlib.md5(str(file_key).encode('utf-8')).hexdigest()[:8], 16)
            rows = sorted(group.to_dict("records"), key=lambda row: (row["start_s"], row["end_s"]))
            file_end = max(float(row["end_s"]) for row in rows)
            grid_end = max(0.0, file_end - self.duration_s)
            hop_ms = int(round(self.hop_s * 1000.0))
            starts = {
                (i * hop_ms) / 1000.0
                for i in range(math.floor(grid_end * 1000.0 / hop_ms) + 1)
            }
            starts.add(0.0)
            # Add interval-aligned starts so a boundary interval is complete in at least one block.
            for row in rows:
                starts.add(round(max(0.0, float(row["end_s"]) - self.duration_s), 6))

            for start in sorted(starts):
                end = start + self.duration_s
                contained = [
                    dict(row)
                    for row in rows
                    if float(row["start_s"]) >= start - 1e-7
                    and float(row["end_s"]) <= end + 1e-7
                ]
                if not contained:
                    continue
                for row in contained:
                    row["interval_id"] = self._interval_id(row)
                rng = random.Random(self.seed + self.fold_index * 100003 + file_hash + int(round(start * 1000)))
                selected = self._select_intervals(contained, None, rng)
                block_id = f"{file_key}|{start:.6f}|{end:.6f}"
                blocks[(str(file_key), int(round(start * 1_000_000)))] = {
                    "block_id": block_id,
                    "file_key": str(file_key),
                    "audio_path": rows[0]["audio_path"],
                    "speaker_id": rows[0]["speaker_id"],
                    "start_s": start,
                    "end_s": end,
                    "intervals": selected,
                }

        # Add an interval-aligned block only when the capped grid blocks did not
        # retain that interval. The interval ID is part of the key so multiple
        # missing intervals at the same start cannot overwrite one another.
        represented = {
            interval["interval_id"]
            for block in blocks.values()
            for interval in block["intervals"]
        }
        for file_key, group in df.groupby("file_key", sort=True):
            file_hash = int(hashlib.md5(str(file_key).encode('utf-8')).hexdigest()[:8], 16)
            rows = sorted(group.to_dict("records"), key=lambda row: (row["start_s"], row["end_s"]))
            for row in rows:
                primary = self._interval_id(row)
                if primary in represented:
                    continue
                interval_duration = float(row["end_s"]) - float(row["start_s"])
                if interval_duration > self.duration_s + 1e-7:
                    if primary not in _WARNED_INTERVALS:
                        _WARNED_INTERVALS.add(primary)
                        logger.warning(
                            f"Interval {primary} duration ({interval_duration:.3f}s) exceeds block capacity ({self.duration_s}s). "
                            f"Truncating to {self.duration_s}s."
                        )
                    start = float(row["start_s"])
                    end = start + self.duration_s
                    row_copy = dict(row)
                    row_copy["end_s"] = end
                    row_copy["interval_id"] = primary
                    contained = [row_copy]
                    for candidate in rows:
                        if candidate is row:
                            continue
                        if float(candidate["start_s"]) >= start - 1e-7 and float(candidate["end_s"]) <= end + 1e-7:
                            c = dict(candidate)
                            c["interval_id"] = self._interval_id(c)
                            contained.append(c)
                else:
                    start = max(0.0, float(row["end_s"]) - self.duration_s)
                    end = start + self.duration_s
                    contained = [
                        dict(candidate)
                        for candidate in rows
                        if float(candidate["start_s"]) >= start - 1e-7
                        and float(candidate["end_s"]) <= end + 1e-7
                    ]
                    for candidate in contained:
                        candidate["interval_id"] = self._interval_id(candidate)
                rng = random.Random(self.seed + self.fold_index * 100003 + file_hash + int(round(start * 1000)))
                selected = self._select_intervals(contained, primary, rng)
                key = (str(file_key), int(round(start * 1_000_000)), primary)
                blocks[key] = {
                    "block_id": f"{file_key}|{start:.6f}|{end:.6f}|{primary}",
                    "file_key": str(file_key),
                    "audio_path": row["audio_path"],
                    "speaker_id": row["speaker_id"],
                    "start_s": start,
                    "end_s": end,
                    "intervals": selected,
                }
                represented.update(interval["interval_id"] for interval in selected)

        return list(sorted(blocks.values(), key=lambda block: (block["file_key"], block["start_s"])))


class ContinuousAudioBlockDataset(Dataset):
    """Load fixed blocks and return interval bounds relative to each block."""

    _AUDIO_INFO_CACHE: dict[str, tuple[int, float]] = {}
    _AUDIO_WAVEFORM_CACHE: dict[str, tuple[torch.Tensor, float]] = {}

    def __init__(
        self,
        df: pd.DataFrame,
        duration_ms: float = 2000.0,
        hop_ms: float = 1000.0,
        max_intervals: int = 8,
        boundary_trim_frames: int = 0,
        target_sr: int = 16000,
        seed: int = 42,
        fold_index: int = 0,
        cache_waveforms: bool = True,
        max_cache_files: int = 1000,
    ) -> None:
        self.target_sr = int(target_sr)
        # Frame-space boundary trimming is performed by the model after WavLM
        # determines its output stride; the Dataset keeps sample-space bounds.
        self.boundary_trim_frames = max(0, int(boundary_trim_frames))
        self.base_df = df.reset_index(drop=True)
        self.duration_ms = float(duration_ms)
        self.hop_ms = float(hop_ms)
        self.max_intervals = int(max_intervals)
        self.seed = int(seed)
        self.fold_index = int(fold_index)
        self.cache_waveforms = bool(cache_waveforms)
        self.max_cache_files = int(max_cache_files)
        self.waveform_cache: OrderedDict[str, tuple[torch.Tensor, float]] = OrderedDict()
        self.current_epoch = 1
        self.index = self._make_index(self.current_epoch)
        self.blocks = self.index.blocks
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def clear_cache(self) -> None:
        """Clear instance waveform cache."""
        self.waveform_cache.clear()

    @classmethod
    def clear_waveform_cache(cls):
        cls._AUDIO_WAVEFORM_CACHE.clear()
        cls._AUDIO_INFO_CACHE.clear()

    def _get_audio_waveform(self, audio_path: str) -> tuple[torch.Tensor, float]:
        """Fetch resampled 16kHz mono audio waveform and duration, with RAM LRU caching."""
        if self.cache_waveforms and audio_path in self.waveform_cache:
            waveform, duration = self.waveform_cache[audio_path]
            self.waveform_cache.move_to_end(audio_path)
            return waveform, duration

        if audio_path not in self._AUDIO_INFO_CACHE:
            info = torchaudio.info(audio_path)
            self._AUDIO_INFO_CACHE[audio_path] = (info.sample_rate, float(info.num_frames) / float(info.sample_rate))
        orig_sr, total_duration = self._AUDIO_INFO_CACHE[audio_path]

        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = self.resamplers[sr](waveform)
        waveform = waveform.squeeze(0).contiguous()

        if self.cache_waveforms and self.max_cache_files > 0:
            if len(self.waveform_cache) >= self.max_cache_files:
                self.waveform_cache.popitem(last=False)
            self.waveform_cache[audio_path] = (waveform, total_duration)

        return waveform, total_duration

    def _make_index(self, epoch: int) -> BlockIndex:
        return BlockIndex(
            self.base_df,
            duration_ms=self.duration_ms,
            hop_ms=self.hop_ms,
            max_intervals=self.max_intervals,
            seed=self.seed + int(epoch) * 1009,
            fold_index=self.fold_index,
        )

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch == self.current_epoch:
            return
        self.current_epoch = epoch
        self.index = self._make_index(epoch)
        self.blocks = self.index.blocks

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int):
        block = self.blocks[idx]
        audio_path = block["audio_path"]
        full_waveform, total_duration = self._get_audio_waveform(audio_path)
        start_s = max(0.0, min(float(block["start_s"]), total_duration))
        end_s = max(start_s, min(float(block["end_s"]), total_duration))
        start_frame = max(0, int(round(start_s * self.target_sr)))
        end_frame = min(full_waveform.shape[0], int(round(end_s * self.target_sr)))
        # Clone waveform slice to prevent in-place modifications from polluting the RAM cache
        waveform = full_waveform[start_frame:end_frame].clone()
        if waveform.numel() == 0:
            target_samples = int(round(self.duration_ms * self.target_sr / 1000.0))
            waveform = torch.zeros(target_samples, dtype=full_waveform.dtype)

        bounds, labels, ids = [], [], []
        block_dur = max(0.0, end_s - start_s)
        for row in block["intervals"]:
            raw_s = float(row["start_s"])
            raw_e = float(row["end_s"])
            # Skip intervals with zero duration or outside the block
            if raw_e <= start_s or raw_s >= end_s:
                continue
            rel_start = max(0.0, min(raw_s - start_s, block_dur))
            rel_end = max(rel_start, min(raw_e - start_s, block_dur))
            if rel_end <= rel_start:
                continue
            bounds.append([int(round(rel_start * self.target_sr)), int(round(rel_end * self.target_sr))])
            
            # Multi-label support: [label_creaky, label_breathy]
            l_creaky = float(row.get("label_creaky", row.get("label", float("nan"))))
            l_breathy = float(row.get("label_breathy", float("nan")))
            labels.append([l_creaky, l_breathy])
            ids.append(row["interval_id"])

        bounds_tensor = torch.tensor(bounds, dtype=torch.long).reshape(-1, 2)
        labels_tensor = torch.tensor(labels, dtype=torch.float32).reshape(-1, 2)
        return (
            waveform,
            bounds_tensor,
            labels_tensor,
            block["speaker_id"],
            block["block_id"],
            ids,
        )


def collate_block_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return (
            torch.empty((0, 0), dtype=torch.float32),
            torch.empty((0, 0, 2), dtype=torch.long),
            torch.empty((0, 0, 1), dtype=torch.float32),
            torch.empty((0, 0), dtype=torch.bool),
            torch.empty((0,), dtype=torch.long),
            [],
            [],
            [],
        )

    waveforms, bounds, labels, speakers, block_ids, interval_ids = zip(*batch)
    lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
    max_audio = int(lengths.max())
    max_intervals = max((item.shape[0] for item in bounds), default=0)
    num_tasks = labels[0].shape[1] if (len(labels) > 0 and labels[0].ndim > 1) else 1

    padded = torch.zeros(len(batch), max_audio, dtype=waveforms[0].dtype)
    padded_bounds = torch.zeros(len(batch), max_intervals, 2, dtype=torch.long)
    if num_tasks > 1:
        padded_labels = torch.full((len(batch), max_intervals, num_tasks), float("nan"), dtype=torch.float32)
    else:
        padded_labels = torch.full((len(batch), max_intervals), float("nan"), dtype=torch.float32)
    valid = torch.zeros(len(batch), max_intervals, dtype=torch.bool)
    ids = [[None] * max_intervals for _ in batch]
    for i, waveform in enumerate(waveforms):
        padded[i, : waveform.numel()] = waveform
        count = bounds[i].shape[0]
        padded_bounds[i, :count] = bounds[i]
        padded_labels[i, :count] = labels[i]
        valid[i, :count] = True
        ids[i][:count] = interval_ids[i]
    return padded, padded_bounds, padded_labels, valid, lengths, list(speakers), list(block_ids), ids
