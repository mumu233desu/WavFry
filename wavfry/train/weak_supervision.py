"""Positive-only weak training setup, leakage guards and reproducible batch schedule."""
from __future__ import annotations

from pathlib import Path
import math

import pandas as pd
from torch.utils.data import DataLoader

from wavfry.data.utterance_bag import BagBatchSampler, UtteranceBagDataset, collate_utterance_bags
from wavfry.data.weak_manifest import manifest_digest, validate_source_stamps


def dataset_rows(dataset):
    frame = getattr(dataset, "base_df", getattr(dataset, "df", None))
    return frame.copy() if isinstance(frame, pd.DataFrame) else None


def guard_weak_partitions(weak, manual_train, manual_val):
    """Check source assignments before filtering; reject weak train leakage."""
    key = ["dataset_id", "file_key"]
    if weak.duplicated("supervision_id").any():
        raise ValueError("Duplicate weak supervision IDs")
    if weak.groupby(key).batch_split.nunique().gt(1).any():
        raise ValueError("Conflicting weak file partitions")
    train = weak[(weak.batch_split == "train") & (weak.import_status == "eligible")]
    held = weak[weak.batch_split.isin(["val", "test"])]
    for frame in (held, manual_val):
        if frame is None or frame.empty:
            continue
        if set(key).issubset(frame):
            if set(map(tuple, train[key].values)) & set(map(tuple, frame[key].values)):
                raise ValueError("Weak train file overlaps held-out data")
        if {"dataset_id", "speaker_id"}.issubset(frame):
            cols = ["dataset_id", "speaker_id"]
            if set(map(tuple, train[cols].values)) & set(map(tuple, frame[cols].values)):
                raise ValueError("Weak train speaker overlaps held-out data")
        if "audio_path" in frame:
            paths = {str(Path(p).resolve()).casefold() for p in frame.audio_path.dropna()}
            if paths & {str(Path(p).resolve()).casefold() for p in train.audio_path}:
                raise ValueError("Weak train audio overlaps held-out data across dataset IDs")
    if manual_train is not None and not manual_train.empty and set(key).issubset(manual_train):
        common = train.merge(manual_train, on=key, suffixes=("_weak", "_manual"))
        if not common.empty and "speaker_id_manual" in common:
            if (common.speaker_id_weak.astype(str) != common.speaker_id_manual.astype(str)).any():
                raise ValueError("Manual/weak speaker identity mismatch")
        # A human positive already supplies the existence constraint; do not add it twice.
        if "label_breathy" in manual_train:
            human_positive = manual_train[pd.to_numeric(manual_train.label_breathy, errors="coerce").eq(1)]
            redundant = set(map(tuple, human_positive[key].values))
            train = train[[tuple(v) not in redundant for v in train[key].values]]
    return train.reset_index(drop=True)


def create_weak_loader(config, model, manual_train, manual_val):
    cfg = config["training"].get("weak_supervision", {})
    if not cfg.get("enabled", False):
        return None, None
    if cfg.get("task", "breathy") != "breathy" or "breathy" not in model.tasks:
        raise ValueError("Weak supervision requires an active breathy task")
    if not model.frame_head_enabled or not config["training"].get("train_heads", {}).get("breathy", True):
        raise ValueError("Weak supervision requires a trainable breathy frame head")
    if config["training"].get("paradigm", "segment") != "block":
        raise ValueError("Weak supervision currently requires the block paradigm")
    vsdc_cfg = config["training"].get("vsdc", {})
    vsdc_enabled = bool(vsdc_cfg.get("enabled", False))
    frame_cfg = config["training"].get("block", {}).get("frame_head", {})
    if not vsdc_enabled and (frame_cfg.get("mil_mode", "lse") != "lse" or cfg.get("candidate_scope", "vowels") != "vowels"):
        raise ValueError("First-version weak supervision uses LSE over all vowel candidates")
    weight = float(cfg.get("loss_weight", .05))
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("Positive finite weak loss_weight is required")
    if int(cfg.get("start_epoch", 2)) < 1 or int(cfg.get("start_epoch", 2)) > config["training"]["epochs"]:
        raise ValueError("weak start_epoch must fall within training epochs")
    path = Path(cfg["manifest"])
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    weak = pd.read_csv(path, dtype={"file_key": str, "speaker_id": str, "dataset_id": str})
    train = guard_weak_partitions(weak, manual_train, manual_val)
    if manual_train is not None and not manual_train.empty and 'audio_path' in manual_train:
        manual_paths = {str(Path(p).resolve()).casefold(): float(label)
                        for p, label in zip(manual_train.audio_path, manual_train.get('label_breathy', pd.Series(float('nan'), index=manual_train.index)))
                        if label == 1}
        train = train[[str(Path(p).resolve()).casefold() not in manual_paths for p in train.audio_path]].reset_index(drop=True)
    validate_source_stamps(train)
    if train.empty:
        raise ValueError("No non-redundant eligible weak training positives")

    if vsdc_enabled:
        # Also include modal negative clips from the same training partition
        modal_rows = weak[(weak.batch_split == "train") & (weak.reason == "no_positive_evidence")].copy()
        if not modal_rows.empty:
            train_combined = pd.concat([train, modal_rows], ignore_index=True)
        else:
            train_combined = train
    else:
        train_combined = train

    stride, receptive = 1, 1
    for kernel, step in zip(model.wavlm.config.conv_kernel, model.wavlm.config.conv_stride):
        receptive += (int(kernel) - 1) * stride
        stride *= int(step)
    block_cfg = config["training"]["block"]
    dataset = UtteranceBagDataset(train_combined, block_ms=block_cfg.get("duration_ms", 2000),
                                 hop_ms=block_cfg.get("hop_ms", 1000), frame_stride=stride,
                                 receptive=receptive, max_windows=int(cfg.get("max_windows", 64)),
                                 allow_modal=vsdc_enabled)
    if vsdc_enabled:
        from wavfry.data.utterance_bag import PairedBagBatchSampler, collate_paired_utterance_bags
        sampler = PairedBagBatchSampler(dataset, max_windows=int(cfg.get("max_windows", 64)),
                                      samples_per_epoch=int(cfg.get("samples_per_epoch", 64)),
                                      seed=config["training"].get("seed", 42))
        collate_fn = collate_paired_utterance_bags
    else:
        sampler = BagBatchSampler(dataset, max_windows=int(cfg.get("max_windows", 16)),
                                  samples_per_epoch=int(cfg.get("samples_per_epoch", 64)),
                                  seed=config["training"].get("seed", 42))
        collate_fn = collate_utterance_bags
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=collate_fn)
    return loader, manifest_digest(weak)


def epoch_update_counts(manual_loader, weak_loader, epochs, start_epoch, grad_acc):
    counts = []
    dataset = manual_loader.dataset
    original_epoch = getattr(dataset, 'current_epoch', 1)
    for epoch in range(1, epochs + 1):
        if hasattr(dataset, 'set_epoch'):
            dataset.set_epoch(epoch)
        weak_count = 0
        if weak_loader is not None and epoch >= start_epoch:
            weak_loader.batch_sampler.set_epoch(epoch)
            weak_count = len(weak_loader)
        counts.append(math.ceil((len(manual_loader) + weak_count) / grad_acc))
    if hasattr(dataset, 'set_epoch'):
        dataset.set_epoch(original_epoch)
    if weak_loader is not None:
        weak_loader.batch_sampler.set_epoch(1)
    return counts


def interleave_batches(manual_loader, weak_loader=None):
    """Distribute complete weak batches without reusing or dropping manual batches."""
    if weak_loader is None:
        for batch in manual_loader:
            yield "manual", batch
        return
    weak_iter = iter(weak_loader)
    n_manual, n_weak = len(manual_loader), len(weak_loader)
    done = 0
    for index, batch in enumerate(manual_loader, 1):
        yield "manual", batch
        target = index * n_weak // n_manual
        while done < target:
            yield "weak", next(weak_iter)
            done += 1


def training_provenance(manual_train, weak_train=None):
    complete = manual_train is not None and {"dataset_id", "speaker_id", "file_key"}.issubset(manual_train)
    files, speakers = set(), set()
    for frame in (manual_train, weak_train):
        if frame is None:
            continue
        if not {"dataset_id", "speaker_id", "file_key"}.issubset(frame):
            complete = False
            continue
        files.update(map(tuple, frame[["dataset_id", "file_key"]].astype(str).values))
        speakers.update(map(tuple, frame[["dataset_id", "speaker_id"]].astype(str).values))
    return {"complete": bool(complete), "training_files": sorted(files), "training_speakers": sorted(speakers)}
