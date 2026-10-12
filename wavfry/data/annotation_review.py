"""Read original annotation shards for review, without training-data filters."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import soundfile as sf


TASK_STATES = {
    "creaky": {"creaky": 1.0, "non_creaky": 0.0},
    "breathy": {"breathy": 1.0, "non_breathy": 0.0, "modal": 0.0},
}


def _json(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Unreadable annotation JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _index_audio(directories):
    result = {}
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {directory}")
        for path in directory.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".wav", ".flac"}:
                continue
            if any(part in {"_creaky_cache", "_breathy_cache"} for part in path.parts):
                continue
            key = path.stem
            if key in result and result[key].resolve() != path.resolve():
                raise ValueError(f"Ambiguous audio file_key {key!r}: {result[key]} and {path}")
            result[key] = path.resolve()
    return result


def _shards(batch_dir, task):
    names = [f"manual_{task}_annotations.json.d",
             "manual_breathy_annotations.json.d" if task == "creaky" else "manual_creaky_annotations.json.d"]
    for name in names:
        candidate = batch_dir / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"No {task} annotation shard directory in {batch_dir}")


def _bounds(value, origin):
    try:
        start, end = float(value[0]), float(value[1])
    except (ValueError, TypeError, IndexError, KeyError) as exc:
        raise ValueError(f"Malformed annotation bounds in {origin}: {value!r}") from exc
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        raise ValueError(f"Invalid annotation bounds in {origin}: {value!r}")
    return start, end


def load_raw_annotations(*, audio_dirs, task, cache_dir=None, shard_dir=None,
                         batches=None, sampling_state=None, dataset_id=None):
    """Return original interval records, including uncertain and span-only shards.

    Task-compatible filenames follow the annotation APP convention. The shard's
    file_key identifies audio; its hash filename does not. No annotations are
    written or normalized back into the cache.
    """
    if task not in TASK_STATES or (cache_dir is None) == (shard_dir is None):
        raise ValueError("Choose one raw cache/shard directory and creaky or breathy")
    if shard_dir is not None and batches is not None:
        raise ValueError("--batches applies only to --annotation-cache")
    audio = _index_audio(audio_dirs)
    directories = []
    assignments = {}
    batch_info = {}
    state_path = Path(sampling_state) if sampling_state else (Path(cache_dir) / "sampling_state.json" if cache_dir else None)
    if state_path is not None:
        state = _json(state_path)
        for batch in state.get("batches", []):
            bid = str(batch["batch_id"])
            if bid in batch_info:
                raise ValueError(f"Duplicate batch_id {bid} in {state_path}")
            batch_info[bid] = batch
            for info in batch.get("files", []):
                key = str(info["file_key"])
                previous = assignments.get(key)
                entry = (str(batch.get("split", "unassigned")), str(info.get("speaker_id") or ""))
                if previous and (previous[0] != entry[0] or (previous[1] and entry[1] and previous[1] != entry[1])):
                    raise ValueError(f"Conflicting sampling assignments for {key}")
                assignments[key] = entry
    if cache_dir is not None:
        name_to_id = {}
        for bid_key, b in batch_info.items():
            name_to_id[bid_key] = bid_key
            bname = str(b.get("name", "")).strip()
            if bname:
                name_to_id[bname] = bid_key
                name_to_id[bname.lower()] = bid_key
        requested = []
        if batches is not None:
            for b in batches:
                b_str = str(b).strip()
                resolved = name_to_id.get(b_str) or name_to_id.get(b_str.lower())
                if resolved is None:
                    raise ValueError(f"Batch {b} is not registered in {state_path}")
                requested.append(resolved)
        else:
            requested = list(batch_info)
        if not requested:
            raise ValueError("No batches selected or registered")
        for bid in requested:
            if bid not in batch_info:
                raise ValueError(f"Batch {bid} is not registered in {state_path}")
            directories.append((bid, _shards(Path(cache_dir) / "batches" / f"batch_{bid}", task)))
    else:
        path = Path(shard_dir)
        if not path.is_dir() or not path.name.endswith(".json.d"):
            raise FileNotFoundError(f"Expected annotation .json.d directory: {path}")
        bid = path.parent.name.removeprefix("batch_") if path.parent.name.startswith("batch_") else "unspecified"
        directories.append((bid, path))

    rows, labels_seen = [], {}
    durations = {}
    for bid, directory in directories:
        files = sorted(directory.glob("*.json"))
        if not files:
            raise ValueError(f"Empty annotation shard directory: {directory}")
        for path in files:
            entry = _json(path)
            key = entry.get("file_key")
            if not isinstance(key, str) or not key:
                raise ValueError(f"Missing file_key: {path}")
            if key not in audio:
                raise FileNotFoundError(f"No audio for annotated file_key {key!r}: {path}")
            if key not in durations:
                durations[key] = float(sf.info(str(audio[key])).duration)
            duration = durations[key]
            split, state_speaker = assignments.get(key, ("unknown", ""))
            shard_speaker = str(entry.get("speaker_id") or "")
            if state_speaker and shard_speaker and state_speaker != shard_speaker:
                raise ValueError(f"Speaker identity conflict: {key}")
            speaker = state_speaker or shard_speaker
            annotations = entry.get("annotations", {})
            if not isinstance(annotations, dict):
                raise ValueError(f"Invalid annotations object: {path}")
            auto_ids = entry.get("auto_sp_interval_ids", [])
            if not isinstance(auto_ids, list):
                raise ValueError(f"Invalid auto_sp_interval_ids: {path}")
            exact = entry.get("exact_creak_spans", [])
            uncertain = entry.get("uncertain_spans", [])
            if not isinstance(exact, list) or not isinstance(uncertain, list):
                raise ValueError(f"Invalid exact/uncertain spans: {path}")
            records = []
            for iid, label in annotations.items():
                pieces = str(iid).split("|", 2)
                if len(pieces) != 3:
                    raise ValueError(f"Malformed interval_id {iid!r}: {path}")
                start, end = _bounds(pieces, path)
                records.append((str(iid), start, end, pieces[2], label))
            if task == "creaky":
                for index, span in enumerate(exact):
                    start, end = _bounds(span, path)
                    records.append((f"exact_span:{index}", start, end, "", "exact_creaky"))
            for index, span in enumerate(uncertain):
                start, end = _bounds(span, path)
                records.append((f"uncertain_span:{index}", start, end, "", "uncertain"))
            for iid, start, end, phone, state in records:
                label = 1.0 if state == "exact_creaky" else TASK_STATES[task].get(state, float("nan"))
                source = "automatic_silence" if iid in auto_ids else ("mllm_prefill" if state == "mllm_breathy" else "human")
                status = "ready" if math.isfinite(label) else ("uncertain" if state == "uncertain" else "unlabeled_for_task")
                if start < 0 or end > duration:
                    status = "out_of_audio_bounds"
                if not iid.startswith(('exact_span:', 'uncertain_span:')):
                    identity = (key, task, iid)
                    if identity in labels_seen and labels_seen[identity] != state:
                        raise ValueError(f"Conflicting duplicate interval {identity}: {path}")
                    labels_seen[identity] = state
                rows.append(dict(
                    supervision_id=f"raw|{task}|{bid}|{key}|{iid}", dataset_id=dataset_id or "raw_annotations",
                    identity_verified=bool(dataset_id and speaker), file_key=key, speaker_id=speaker or "unknown",
                    audio_path=str(audio[key]), start_s=start, end_s=end, phone=phone, task=task,
                    scope="interval", label=label, source=source, batch_id=bid, batch_split=split,
                    interval_id=iid, original_state=state, shard_path=str(path.resolve()),
                    reviewed_at=entry.get("reviewed_at"), annotation_fingerprint=hashlib.sha256(path.read_bytes()).hexdigest(),
                    exact_creak_spans=json.dumps(exact), uncertain_spans=json.dumps(uncertain),
                    input_status=status, candidate_intervals="[]",
                ))
    if not rows:
        raise ValueError("Selected shards contain no intervals or ranges for review")
    return pd.DataFrame(rows)
