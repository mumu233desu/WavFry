"""Continuous fixed-length audio blocks with interval-level supervision."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset


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
        required = {"file_key", "audio_path", "start_s", "end_s", "label", "speaker_id"}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Block index requires columns: {sorted(missing)}")

        blocks: dict[tuple[str, ...], dict[str, Any]] = {}
        for file_key, group in df.groupby("file_key", sort=True):
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
                rng = random.Random(self.seed + self.fold_index * 100003 + int(round(start * 1000)))
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
            rows = sorted(group.to_dict("records"), key=lambda row: (row["start_s"], row["end_s"]))
            for row in rows:
                primary = self._interval_id(row)
                if primary in represented:
                    continue
                interval_duration = float(row["end_s"]) - float(row["start_s"])
                if interval_duration > self.duration_s + 1e-7:
                    raise ValueError(
                        f"Interval {primary} exceeds block duration "
                        f"({interval_duration:.3f}s > {self.duration_s:.3f}s)"
                    )
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
                rng = random.Random(self.seed + self.fold_index * 100003 + int(round(start * 1000)))
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

    def __init__(
        self,
        df: pd.DataFrame,
        duration_ms: float = 2000.0,
        hop_ms: float = 1000.0,
        max_intervals: int = 8,
        boundary_trim_frames: int = 0,
        target_sr: int = 16000,
        lowpass_cfg: dict[str, Any] | None = None,
        seed: int = 42,
        fold_index: int = 0,
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
        self.current_epoch = 1
        self.index = self._make_index(self.current_epoch)
        self.blocks = self.index.blocks
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}
        lowpass_cfg = lowpass_cfg or {}
        self.lowpass_enabled = bool(lowpass_cfg.get("enabled", False))
        self.lowpass_cutoff = float(lowpass_cfg.get("cutoff_hz", 3000.0))
        self.lowpass_q = float(lowpass_cfg.get("Q", 0.707))

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
        info = torchaudio.info(block["audio_path"])
        orig_sr = info.sample_rate
        total_duration = info.num_frames / orig_sr
        start_s = float(block["start_s"])
        end_s = min(float(block["end_s"]), total_duration)
        start_frame = max(0, int(round(start_s * orig_sr)))
        num_frames = max(1, int(round((end_s - start_s) * orig_sr)))
        waveform, sr = torchaudio.load(block["audio_path"], frame_offset=start_frame, num_frames=num_frames)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != self.target_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = self.resamplers[sr](waveform)
        if self.lowpass_enabled:
            waveform = torchaudio.functional.lowpass_biquad(
                waveform, self.target_sr, self.lowpass_cutoff, self.lowpass_q
            )
        waveform = waveform.squeeze(0)

        bounds, labels, ids = [], [], []
        for row in block["intervals"]:
            rel_start = max(0.0, float(row["start_s"]) - start_s)
            rel_end = min(end_s - start_s, float(row["end_s"]) - start_s)
            bounds.append([int(round(rel_start * self.target_sr)), int(round(rel_end * self.target_sr))])
            labels.append(float(row["label"]))
            ids.append(row["interval_id"])
        return (
            waveform,
            torch.tensor(bounds, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float32),
            block["speaker_id"],
            block["block_id"],
            ids,
        )


def collate_block_fn(batch):
    waveforms, bounds, labels, speakers, block_ids, interval_ids = zip(*batch)
    lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.long)
    max_audio = int(lengths.max())
    max_intervals = max(item.shape[0] for item in bounds)
    padded = torch.zeros(len(batch), max_audio, dtype=waveforms[0].dtype)
    padded_bounds = torch.zeros(len(batch), max_intervals, 2, dtype=torch.long)
    padded_labels = torch.zeros(len(batch), max_intervals, dtype=torch.float32)
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
