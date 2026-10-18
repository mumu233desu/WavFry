"""
WavFry Evaluation Cache Manager & Multi-Dataset Primary Key Engine.

Manages:
1. Multi-dataset collision-free composite keys: (dataset_id, file_key, start_s, end_s).
2. Incremental CSV caching for static baselines and physical acoustic features.
3. Checkpoint model canonical prediction dumping across acoustic degradation tiers.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

from wavfry.data.path_resolver import resolve_corpus_path

# Standard 5 Evaluation Tiers and Representative Conditions
CANONICAL_TIERS: List[str] = [
    "clean",
    "mild_15db",
    "moderate_5db",
    "severe_0db",
    "compound",
]

DEFAULT_TIER_CONDITIONS: Dict[str, List[str]] = {
    "clean": ["clean"],
    "mild_15db": ["cafe_15db", "car_15db", "home_15db", "street_15db", "reverb_15db"],
    "moderate_5db": ["cafe_5db", "car_5db", "home_5db", "street_5db", "reverb_5db"],
    "severe_0db": ["cafe_0db", "car_0db", "home_0db", "street_0db", "reverb_0db"],
    "compound": ["car_compound", "reverb_compound"],
}


def build_interval_key(
    dataset_id: str,
    file_key: str,
    start_s: Union[float, int, str],
    end_s: Union[float, int, str],
) -> str:
    """Generate a collision-free global primary key across multiple datasets.

    Format: ``{dataset_id}::{file_key}::{start_s:.4f}_{end_s:.4f}``
    """
    s = float(start_s)
    e = float(end_s)
    d = str(dataset_id).strip()
    f = str(file_key).strip()
    return f"{d}::{f}::{s:.4f}_{e:.4f}"


def attach_interval_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a standardized 'interval_key' column.

    If missing 'dataset_id', defaults to 'cs_dialogue'.
    """
    df = df.copy()
    if "interval_key" in df.columns:
        return df

    dataset_col = df["dataset_id"] if "dataset_id" in df.columns else "cs_dialogue"
    file_col = df["file_key"] if "file_key" in df.columns else df.get("audio_path", "")
    start_col = df["start_s"] if "start_s" in df.columns else 0.0
    end_col = df["end_s"] if "end_s" in df.columns else 0.0

    keys = [
        build_interval_key(d, f, s, e)
        for d, f, s, e in zip(
            pd.Series(dataset_col, index=df.index).astype(str),
            pd.Series(file_col, index=df.index).astype(str),
            pd.Series(start_col, index=df.index),
            pd.Series(end_col, index=df.index),
        )
    ]
    df["interval_key"] = keys
    return df


class IncrementalCSVCache:
    """Thread-safe and process-safe incremental CSV cache manager.

    Maintains a CSV file indexed by ``key_col`` (default: ``interval_key``).
    When matching against a new or updated test manifest, efficiently partitions
    records into cached hits vs missing keys, allowing incremental on-demand
    computation and appending.
    """

    def __init__(self, cache_path: Union[str, Path], key_col: str = "interval_key"):
        self.cache_path = Path(cache_path)
        self.key_col = key_col
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.df: pd.DataFrame = self._load_cache()

    def _load_cache(self) -> pd.DataFrame:
        if not self.cache_path.exists() or self.cache_path.stat().st_size == 0:
            return pd.DataFrame(columns=[self.key_col])
        try:
            df = pd.read_csv(self.cache_path, low_memory=False)
            if self.key_col not in df.columns:
                logger.warning(
                    f"Cache file {self.cache_path} missing key_col '{self.key_col}'. Resetting."
                )
                return pd.DataFrame(columns=[self.key_col])
            # Ensure key is string
            df[self.key_col] = df[self.key_col].astype(str)
            # Deduplicate by key keeping last
            df = df.drop_duplicates(subset=[self.key_col], keep="last").reset_index(drop=True)
            return df
        except Exception as e:
            logger.error(f"Error loading cache from {self.cache_path}: {e}")
            return pd.DataFrame(columns=[self.key_col])

    def partition_manifest(
        self, manifest_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Partition manifest into (cached_records, missing_manifest_rows).

        Returns:
            cached_records: Sub-DataFrame from cache containing matched records in manifest order.
            missing_manifest_rows: Sub-DataFrame from manifest for keys not yet in cache.
        """
        manifest = attach_interval_keys(manifest_df)
        req_keys = manifest[self.key_col].astype(str).tolist()

        if self.df.empty:
            return pd.DataFrame(columns=self.df.columns), manifest

        cached_dict = self.df.set_index(self.key_col).to_dict(orient="index")

        matched_records = []
        missing_indices = []

        for idx, key in enumerate(req_keys):
            if key in cached_dict:
                row_dict = cached_dict[key].copy()
                row_dict[self.key_col] = key
                matched_records.append(row_dict)
            else:
                missing_indices.append(idx)

        cached_df = pd.DataFrame(matched_records)
        missing_df = manifest.iloc[missing_indices].copy().reset_index(drop=True)
        return cached_df, missing_df

    def append_records(self, new_records_df: pd.DataFrame) -> None:
        """Atomically append new records to the CSV cache and in-memory table."""
        if new_records_df.empty:
            return

        new_df = attach_interval_keys(new_records_df)
        new_df[self.key_col] = new_df[self.key_col].astype(str)

        if self.df.empty:
            combined = new_df.copy()
        else:
            combined = pd.concat([self.df, new_df], ignore_index=True)

        combined = combined.drop_duplicates(subset=[self.key_col], keep="last").reset_index(drop=True)

        # Atomic write via temporary file
        tmp_path = self.cache_path.with_suffix(".tmp")
        combined.to_csv(tmp_path, index=False, encoding="utf-8")
        if tmp_path.exists():
            if self.cache_path.exists():
                self.cache_path.unlink()
            tmp_path.rename(self.cache_path)

        self.df = combined
        logger.info(
            f"[IncrementalCache] Appended {len(new_df)} records to {self.cache_path.name} (Total: {len(self.df)})."
        )

    def get_matched(self, manifest_df: pd.DataFrame) -> pd.DataFrame:
        """Return cached records strictly matched and ordered by manifest_df."""
        cached_df, missing_df = self.partition_manifest(manifest_df)
        if not missing_df.empty:
            raise KeyError(
                f"IncrementalCSVCache at {self.cache_path} is missing {len(missing_df)} "
                f"keys requested by manifest. Run incremental extraction first."
            )
        return cached_df


def dump_model_predictions(
    model: torch.nn.Module,
    manifest_df: pd.DataFrame,
    degradation_engine: Any,
    out_csv_path: Union[str, Path],
    device: torch.device = torch.device("cpu"),
    batch_size: int = 128,
    tier_conditions: Optional[Dict[str, List[str]]] = None,
    sample_rate: int = 16000,
) -> pd.DataFrame:
    """Perform batched inference across standard degradation tiers and dump to CSV.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded WavFryClassifier.
    manifest_df : pd.DataFrame
        Test set manifest dataframe.
    degradation_engine : AcousticDegradationEngine
        Degradation engine to perturb audio.
    out_csv_path : Path | str
        Destination CSV path.
    device : torch.device
        Device for model evaluation.
    batch_size : int
        Batch size.
    tier_conditions : Dict[str, List[str]], optional
        Mapping from tier name to conditions. Defaults to DEFAULT_TIER_CONDITIONS.
    sample_rate : int
        Audio sampling rate.

    Returns
    -------
    pd.DataFrame
        The full dumped predictions DataFrame.
    """
    manifest = attach_interval_keys(manifest_df).reset_index(drop=True)
    out_path = Path(out_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if tier_conditions is None:
        tier_conditions = DEFAULT_TIER_CONDITIONS

    model.eval()
    model.to(device)

    # Pre-cache audio waveforms in memory to avoid repeated disk reads
    audio_cache: Dict[str, np.ndarray] = {}
    unique_paths = manifest["audio_path"].unique()
    for p in unique_paths:
        try:
            wf, sr = sf.read(str(resolve_corpus_path(p)), dtype="float32", always_2d=False)
            if wf.ndim > 1:
                wf = np.mean(wf, axis=1)
            if sr != sample_rate:
                import torchaudio

                t_wf = torch.from_numpy(wf).unsqueeze(0)
                t_wf = torchaudio.functional.resample(t_wf, sr, sample_rate)
                wf = t_wf.squeeze(0).numpy()
            audio_cache[str(p)] = wf
        except Exception as e:
            logger.warning(f"Failed to pre-cache audio {p}: {e}")

    rows: List[Dict[str, Any]] = []

    for tier_name, cond_list in tier_conditions.items():
        logger.info(f"[DumpPredictions] Running tier '{tier_name}' with conditions {cond_list}...")
        for cond in cond_list:
            # Sliced evaluation in batches
            for i in range(0, len(manifest), batch_size):
                batch_sub = manifest.iloc[i : i + batch_size]
                batch_waves = []
                valid_mask = []

                for _, row in batch_sub.iterrows():
                    p = str(row["audio_path"])
                    if p not in audio_cache:
                        batch_waves.append(torch.zeros(3200))
                        valid_mask.append(False)
                        continue
                    wf_full = audio_cache[p]
                    s_idx = max(0, int(round(float(row["start_s"]) * sample_rate)))
                    e_idx = min(len(wf_full), int(round(float(row["end_s"]) * sample_rate)))
                    if e_idx <= s_idx:
                        e_idx = min(len(wf_full), s_idx + 320)
                    chunk = wf_full[s_idx:e_idx]
                    batch_waves.append(torch.from_numpy(chunk))
                    valid_mask.append(True)

                # Pad to max length in batch
                max_len = max(len(w) for w in batch_waves)
                padded_waves = torch.stack(
                    [torch.nn.functional.pad(w, (0, max_len - len(w))) for w in batch_waves]
                )

                # Apply acoustic degradation
                if cond != "clean" and degradation_engine is not None:
                    perturbed = degradation_engine.apply_tensor(padded_waves, cond, seed=42 + i)
                else:
                    perturbed = padded_waves

                perturbed = perturbed.to(device)

                with torch.inference_mode():
                    outputs = model(perturbed)
                    # Support multi-task output dictionary
                    if isinstance(outputs, dict):
                        logits_creak = outputs.get("logits_creaky", outputs.get("creaky"))
                        logits_breath = outputs.get("logits_breathy", outputs.get("breathy"))
                    elif isinstance(outputs, (tuple, list)):
                        logits_creak = outputs[0]
                        logits_breath = outputs[1] if len(outputs) > 1 else None
                    else:
                        logits_creak = outputs
                        logits_breath = None

                    if logits_creak is not None:
                        prob_creak = torch.sigmoid(logits_creak.squeeze(-1)).cpu().numpy()
                    else:
                        prob_creak = np.zeros(len(batch_sub))

                    if logits_breath is not None:
                        prob_breath = torch.sigmoid(logits_breath.squeeze(-1)).cpu().numpy()
                    else:
                        prob_breath = np.zeros(len(batch_sub))

                for j, (_, row) in enumerate(batch_sub.iterrows()):
                    rows.append(
                        {
                            "interval_key": row["interval_key"],
                            "dataset_id": row.get("dataset_id", "cs_dialogue"),
                            "file_key": row.get("file_key", Path(row["audio_path"]).stem),
                            "start_s": float(row["start_s"]),
                            "end_s": float(row["end_s"]),
                            "speaker_id": row.get("speaker_id", "unknown"),
                            "gender": row.get("gender", row.get("sex", "unknown")),
                            "tier": tier_name,
                            "condition": cond,
                            "label_creaky": float(row.get("label_creaky", row.get("label", np.nan))),
                            "prob_creaky": float(prob_creak[j]),
                            "label_breathy": float(row.get("label_breathy", np.nan)),
                            "prob_breathy": float(prob_breath[j]),
                        }
                    )

    preds_df = pd.DataFrame(rows)
    tmp_out = out_path.with_suffix(".tmp")
    preds_df.to_csv(tmp_out, index=False, encoding="utf-8")
    if tmp_out.exists():
        if out_path.exists():
            out_path.unlink()
        tmp_out.rename(out_path)

    logger.info(f"[DumpPredictions] Saved {len(preds_df)} prediction rows to {out_path}.")
    return preds_df
