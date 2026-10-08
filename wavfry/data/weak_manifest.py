"""Offline import of positive-only MLLM supervision, separate from phone labels."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import soundfile as sf
import tgt

WEAK_COLUMNS = [
    "supervision_id", "dataset_id", "file_key", "speaker_id", "audio_path", "tg_path",
    "start_s", "end_s", "task", "label", "batch_split", "source", "source_model",
    "review_status", "import_status", "reason", "cache_path", "alignment_modes",
    "candidate_intervals", "vowel_intervals", "fingerprint",
]


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_digest(df: pd.DataFrame) -> str:
    normalized = df.fillna("").sort_values("supervision_id").to_csv(index=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def classify_prediction(prediction: dict) -> tuple[str, str]:
    value = prediction.get("has_breathy")
    if value is not True and value is not False and value != "uncertain":
        return "review", "invalid_has_breathy"
    segments = prediction.get("segments", [])
    if not isinstance(segments, list) or any(not isinstance(s, dict) for s in segments):
        return "review", "invalid_segments"
    types = {s.get("phonation_type") for s in segments}
    if value == "uncertain":
        return "review", "uncertain"
    if value is False:
        if "breathy" in types:
            return "review", "contradictory_prediction"
        if prediction.get("has_whispery") is True or "whispery" in types:
            return "review", "whispery_only"
        return "unknown", "no_positive_evidence"
    if "breathy" not in types or "uncertain" in types:
        return "review", "positive_without_clear_breathy_segment"
    return "positive", "clear_breathy"


def read_vowels(path: Path, phones: set[str], tier: str = "phones") -> list[list[float]]:
    grid = tgt.io.read_textgrid(str(path))
    if tier not in grid.get_tier_names():
        raise ValueError(f"Phone tier {tier!r} missing in {path}")
    return [[float(iv.start_time), float(iv.end_time)]
            for iv in grid.get_tier_by_name(tier).intervals
            if iv.text.strip().lower() in phones and iv.end_time > iv.start_time]


def build_weak_manifest(
    *, dataset_id: str, cache_dir: Path, sampling_state: Path, audio_dir: Path,
    textgrid_dir: Path, vowel_phones: set[str], phone_tier: str = "phones",
    manual_manifest: pd.DataFrame | None = None, reviews: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Preserve all import decisions; only import_status=eligible can train.

    Reviews are explicit human clip-level decisions (file_key, task, scope,
    label). Partial phone negatives never imply a negative clip.
    """
    state = json.loads(sampling_state.read_text(encoding="utf-8"))
    assignments = {}
    for batch in state.get("batches", []):
        split = batch.get("split", "unassigned")
        for info in batch.get("files", []):
            key = str(info["file_key"])
            if key in assignments and assignments[key][0] != split:
                raise ValueError(f"Conflicting file splits: {key}")
            assignments[key] = (split, str(info.get("speaker_id") or ""))
    if not assignments:
        raise ValueError(f"No file assignments in {sampling_state}")

    manual = manual_manifest if manual_manifest is not None else pd.DataFrame()
    if not manual.empty:
        manual = manual[manual.dataset_id == dataset_id] if "dataset_id" in manual else manual
    clip_reviews = {}
    if reviews is not None and not reviews.empty:
        required = {"file_key", "task", "scope", "label"}
        if not required.issubset(reviews.columns):
            raise ValueError(f"Reviews require {sorted(required)}")
        for row in reviews.to_dict("records"):
            if row["task"] == "breathy" and row["scope"] == "utterance":
                if row["label"] not in (0, 1):
                    raise ValueError("Human utterance review must be binary")
                key = str(row["file_key"])
                if key in clip_reviews and clip_reviews[key] != row["label"]:
                    raise ValueError(f"Conflicting human clip reviews: {key}")
                clip_reviews[key] = row["label"]

    audio_paths = {}
    for ext in ("*.wav", "*.flac"):
        for path in audio_dir.rglob(ext):
            if any(p in {"_breathy_cache", "_creaky_cache"} for p in path.parts):
                continue
            if path.stem in audio_paths:
                raise ValueError(f"Ambiguous audio file_key: {path.stem}")
            audio_paths[path.stem] = path
    grids = {}
    for path in textgrid_dir.rglob("*.TextGrid"):
        if any(p in {"_breathy_cache", "_creaky_cache"} for p in path.parts):
            continue
        if path.stem in grids:
            raise ValueError(f"Ambiguous TextGrid file_key: {path.stem}")
        grids[path.stem] = path

    rows = []
    for key, (split, speaker) in sorted(assignments.items()):
        cached_path = cache_dir / f"{key}.json"
        cached = json.loads(cached_path.read_text(encoding="utf-8")) if cached_path.exists() else {}
        if cached and cached.get("file_key", key) != key:
            raise ValueError(f"MLLM cache identity mismatch: {cached_path}")
        current_manual = manual[manual.file_key.astype(str) == key] if not manual.empty else manual
        if not current_manual.empty:
            if set(current_manual.batch_split.dropna()) - {split}:
                raise ValueError(f"Manual/weak split disagreement: {key}")
            speakers = current_manual.speaker_id.dropna().astype(str).unique()
            if len(speakers) != 1 or (speaker and speaker != speakers[0]):
                raise ValueError(f"Speaker identity disagreement: {key}")
            speaker = str(speakers[0])
        if not speaker:
            raise ValueError(f"Missing speaker_id in sampling state/manual manifest: {key}")
        if key not in audio_paths or key not in grids:
            raise FileNotFoundError(f"Missing audio/TextGrid for assigned file: {key}")
        audio, grid = audio_paths[key], grids[key]
        duration = float(sf.info(str(audio)).duration)
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"Invalid audio duration: {audio}")
        vowels = read_vowels(grid, vowel_phones, phone_tier)
        if any(a < 0 or b > duration + 0.05 for a, b in vowels):
            raise ValueError(f"TextGrid/audio duration mismatch: {key}")
        vowels = [[a, min(b, duration)] for a, b in vowels if a < duration]
        kind, reason = classify_prediction(cached.get("parsed_prediction", {})) if cached else ("unknown", "no_cache")
        status = "eligible" if kind == "positive" else kind
        human = clip_reviews.get(key)
        reviewed = "unreviewed"
        if human is not None:
            reviewed = "human_clip_positive" if human == 1 else "human_clip_negative"
            status, reason = "human_reviewed", "use_explicit_human_supervision"
        if not current_manual.empty and "label_breathy" in current_manual:
            if pd.to_numeric(current_manual.label_breathy, errors="coerce").eq(1).any():
                status, reason = "redundant", "supported_by_human_positive_interval"
        if human == 0 and reason == "supported_by_human_positive_interval":
            raise ValueError(f"Human clip negative conflicts with human phone positive: {key}")
        if not vowels and status == "eligible":
            status, reason = "review", "no_vowel_candidates"
        alignment = cached.get("alignment_result", {})
        candidates = alignment.get("candidate_vowels", [])
        modes = sorted({p.get("alignment_mode", "unknown") for p in alignment.get("detected_phrases", [])})
        stat = audio.stat()
        stamp = {
            "cache": file_digest(cached_path) if cached else "", "grid": file_digest(grid),
            "audio_size": stat.st_size, "audio_mtime_ns": stat.st_mtime_ns,
            "manual": current_manual.fillna("").to_dict("records"), "human_clip_review": human,
            "vowel_phones": sorted(vowel_phones), "phone_tier": phone_tier,
        }
        fingerprint = hashlib.sha256(json.dumps(stamp, sort_keys=True, default=str).encode()).hexdigest()
        rows.append(dict(
            supervision_id=f"{dataset_id}|{key}|breathy|utterance", dataset_id=dataset_id,
            file_key=key, speaker_id=speaker, audio_path=str(audio.resolve()), tg_path=str(grid.resolve()),
            start_s=0.0, end_s=duration, task="breathy", label=1.0 if kind == "positive" else float("nan"),
            batch_split=split, source="mllm", source_model=cached.get("model", "unknown"),
            review_status=reviewed, import_status=status, reason=reason, cache_path=str(cached_path.resolve()),
            alignment_modes=json.dumps(modes), candidate_intervals=json.dumps(candidates),
            vowel_intervals=json.dumps(vowels), fingerprint=fingerprint,
        ))
    return pd.DataFrame(rows, columns=WEAK_COLUMNS)
