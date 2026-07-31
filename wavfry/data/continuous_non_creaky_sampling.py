"""Deterministic epoch-level sampling of continuous non-creaky windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_INTERNAL_BOUNDARY_TOLERANCE_S = 0.001


@dataclass(frozen=True)
class ContinuousNonCreakySamplingConfig:
    """Validated settings for continuous non-creaky window generation."""

    enabled: bool = False
    extra_sample_ratio: float = 0.08
    min_duration_ms: float = 150.0
    target_duration_ms: float = 300.0
    max_duration_ms: float = 800.0
    min_intervals: int = 1
    max_intervals: int = 4
    max_sp_duration_ms: float = 200.0
    max_sp_ratio: float = 0.35
    allow_sp_only: bool = False

    @classmethod
    def from_mapping(cls, values: dict[str, Any] | None) -> "ContinuousNonCreakySamplingConfig":
        values = values or {}
        config = cls(
            enabled=bool(values.get("enabled", False)),
            extra_sample_ratio=float(values.get("extra_sample_ratio", 0.08)),
            min_duration_ms=float(values.get("min_duration_ms", 150.0)),
            target_duration_ms=float(values.get("target_duration_ms", 300.0)),
            max_duration_ms=float(values.get("max_duration_ms", 800.0)),
            min_intervals=int(values.get("min_intervals", 1)),
            max_intervals=int(values.get("max_intervals", 4)),
            max_sp_duration_ms=float(values.get("max_sp_duration_ms", 200.0)),
            max_sp_ratio=float(values.get("max_sp_ratio", 0.35)),
            allow_sp_only=bool(values.get("allow_sp_only", False)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.extra_sample_ratio < 0.0:
            raise ValueError("extra_sample_ratio must be >= 0")
        if self.min_duration_ms <= 0.0:
            raise ValueError("min_duration_ms must be > 0")
        if not self.min_duration_ms <= self.target_duration_ms <= self.max_duration_ms:
            raise ValueError(
                "Expected min_duration_ms <= target_duration_ms <= max_duration_ms"
            )
        if self.min_intervals < 1:
            raise ValueError("min_intervals must be >= 1")
        if self.max_intervals < self.min_intervals:
            raise ValueError("max_intervals must be >= min_intervals")
        if self.max_sp_duration_ms < 0.0:
            raise ValueError("max_sp_duration_ms must be >= 0")
        if not 0.0 <= self.max_sp_ratio <= 1.0:
            raise ValueError("max_sp_ratio must be between 0 and 1")


class ContinuousNonCreakyWindowGenerator:
    """Build safe non-creaky runs once and sample deterministic windows per epoch."""

    def __init__(
        self,
        base_df: pd.DataFrame,
        config: dict[str, Any] | None,
        *,
        seed: int,
        fold_index: int = 0,
    ) -> None:
        self.base_df = base_df.reset_index(drop=True)
        self.config = ContinuousNonCreakySamplingConfig.from_mapping(config)
        self.seed = int(seed)
        self.fold_index = int(fold_index)
        self.runs = self._build_runs() if self.config.enabled else []
        self.last_generation_stats: dict[str, int] = {
            "requested": 0,
            "generated": 0,
            "failed_attempts": 0,
        }

    @property
    def extra_sample_count(self) -> int:
        if not self.config.enabled or not self.runs:
            return 0
        return int(round(len(self.base_df) * self.config.extra_sample_ratio))

    def _build_runs(self) -> list[list[dict[str, Any]]]:
        required = {"file_key", "audio_path", "start_s", "end_s", "label"}
        missing = required - set(self.base_df.columns)
        if missing:
            raise KeyError(f"Continuous non-creaky sampling requires columns: {sorted(missing)}")

        non_creaky = self.base_df[self.base_df["label"].astype(int) == 0].copy()
        non_creaky = non_creaky.sort_values(["file_key", "start_s", "end_s"])

        runs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for row in non_creaky.to_dict("records"):
            if current:
                previous = current[-1]
                same_file = row["file_key"] == previous["file_key"]
                gap_s = float(row["start_s"]) - float(previous["end_s"])
                is_contiguous = abs(gap_s) <= _INTERNAL_BOUNDARY_TOLERANCE_S
                if not same_file or not is_contiguous:
                    if len(current) >= self.config.min_intervals:
                        runs.append(current)
                    current = []
            current.append(row)

        if len(current) >= self.config.min_intervals:
            runs.append(current)
        return runs

    def rng_for_epoch(self, epoch: int) -> np.random.Generator:
        """Return the isolated RNG shared by all sampling choices in an epoch."""
        return np.random.default_rng(
            np.random.SeedSequence([self.seed, self.fold_index, int(epoch)])
        )

    def shuffle_seed_for_epoch(self, epoch: int) -> int:
        """Derive a stable pandas-compatible seed without fold/epoch collisions."""
        seed_sequence = np.random.SeedSequence(
            [self.seed, self.fold_index, int(epoch), 1]
        )
        return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])

    def generate(self, epoch: int) -> pd.DataFrame:
        """Generate the configured number of windows for one fold and epoch."""

        count = self.extra_sample_count
        self.last_generation_stats = {
            "requested": count,
            "generated": 0,
            "failed_attempts": 0,
        }
        if count <= 0:
            return self.base_df.iloc[0:0].copy()

        rng = self.rng_for_epoch(epoch)
        rows: list[dict[str, Any]] = []
        max_attempts = max(200, count * 50)
        attempts = 0
        while len(rows) < count and attempts < max_attempts:
            attempts += 1
            run = self.runs[int(rng.integers(0, len(self.runs)))]
            sampled = self._sample_window(run, rng)
            if sampled is not None:
                rows.append(sampled)

        self.last_generation_stats = {
            "requested": count,
            "generated": len(rows),
            "failed_attempts": attempts - len(rows),
        }
        message = (
            f"requested={count}, generated={len(rows)}, "
            f"failed_attempts={attempts - len(rows)}, eligible_runs={len(self.runs)}"
        )
        if len(rows) < count:
            raise RuntimeError(
                "Continuous non-creaky sampling could not satisfy the configured ratio: "
                + message
            )
        print(f"[ContinuousNonCreaky] {message}.")
        return pd.DataFrame(rows)

    def _sample_window(
        self,
        run: list[dict[str, Any]],
        rng: np.random.Generator,
    ) -> dict[str, Any] | None:
        if len(run) == 1:
            boundary_s = float(run[0]["start_s"])
        else:
            boundary_index = int(rng.integers(1, len(run)))
            boundary_s = float(run[boundary_index]["start_s"])
        run_start_s = float(run[0]["start_s"])
        run_end_s = float(run[-1]["end_s"])
        run_duration_s = run_end_s - run_start_s

        min_duration_s = self.config.min_duration_ms / 1000.0
        max_duration_s = min(self.config.max_duration_ms / 1000.0, run_duration_s)
        if max_duration_s < min_duration_s:
            return None

        # Draw a variable duration, then expand short draws toward the target.
        if max_duration_s <= min_duration_s + 1e-12:
            drawn_duration_s = min_duration_s
        else:
            drawn_duration_s = float(rng.uniform(min_duration_s, max_duration_s))
        window_duration_s = min(
            max(drawn_duration_s, self.config.target_duration_ms / 1000.0),
            max_duration_s,
        )
        if len(run) == 1:
            start_low = run_start_s
            start_high = run_end_s - window_duration_s
        else:
            epsilon = min(_INTERNAL_BOUNDARY_TOLERANCE_S, window_duration_s / 10.0)
            start_low = max(run_start_s, boundary_s - window_duration_s + epsilon)
            start_high = min(boundary_s - epsilon, run_end_s - window_duration_s)
        if start_high + _INTERNAL_BOUNDARY_TOLERANCE_S < start_low:
            return None

        if start_high <= start_low:
            start_s = start_low
        else:
            start_s = float(rng.uniform(start_low, start_high))
        end_s = start_s + window_duration_s

        overlapping = [
            interval
            for interval in run
            if min(end_s, float(interval["end_s"]))
            - max(start_s, float(interval["start_s"]))
            > _INTERNAL_BOUNDARY_TOLERANCE_S
        ]
        if not self.config.min_intervals <= len(overlapping) <= self.config.max_intervals:
            return None

        sp_duration_s = sum(
            max(
                0.0,
                min(end_s, float(interval["end_s"]))
                - max(start_s, float(interval["start_s"])),
            )
            for interval in overlapping
            if str(interval.get("phone_type", "")) == "silence"
            or str(interval.get("phone", "")).strip().upper() == "SP"
        )
        duration_s = end_s - start_s
        non_sp_duration_s = duration_s - sp_duration_s
        if sp_duration_s * 1000.0 > self.config.max_sp_duration_ms:
            return None
        if duration_s > 0.0 and sp_duration_s / duration_s > self.config.max_sp_ratio:
            return None
        if not self.config.allow_sp_only and non_sp_duration_s <= _INTERNAL_BOUNDARY_TOLERANCE_S:
            return None

        row = dict(overlapping[0])
        row.update(
            {
                "start_s": start_s,
                "end_s": end_s,
                "phone": "MULTI",
                "phone_type": "continuous_non_creaky",
                "label": 0,
                "sample_source": "continuous_non_creaky",
                "source_interval_count": len(overlapping),
                "sp_duration_ms": sp_duration_s * 1000.0,
                "context_padding_ms": 0.0,
            }
        )
        return row
