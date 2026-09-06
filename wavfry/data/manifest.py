"""Build a safe, typed training manifest from sharded annotations."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


WAVE_ROOT = Path(__file__).resolve().parents[3]
if str(WAVE_ROOT) not in sys.path:
    sys.path.insert(0, str(WAVE_ROOT))

# The lightweight module is intentionally used instead of the librosa-heavy App core.
phone_inventory = __import__(
    "04_creaky_tools.src.phone_inventory",
    fromlist=[
        "IntervalType",
        "canonical_phone_for_id",
        "classify_phone_label",
        "get_phone_inventory",
        "make_interval_id",
    ],
)
IntervalType = phone_inventory.IntervalType
canonical_phone_for_id = phone_inventory.canonical_phone_for_id
classify_phone_label = phone_inventory.classify_phone_label
get_phone_inventory = phone_inventory.get_phone_inventory
make_interval_id = phone_inventory.make_interval_id

manual_annotation = __import__(
    "04_creaky_tools.src.manual_annotation",
    fromlist=["load_annotations", "annotation_storage_path"],
)
load_annotations = manual_annotation.load_annotations
annotation_storage_path = manual_annotation.annotation_storage_path


DEFAULT_TRAINING_PHONE_TYPES = {
    IntervalType.VOWEL.value,
    IntervalType.VOICELESS_OBS.value,
    IntervalType.SILENCE.value,
}


def _build_batch_lookup(sampling_state_path: Path) -> dict[str, int]:
    """Return ``file_key -> batch_id`` from sampling_state.json."""

    lookup: dict[str, int] = {}
    if not sampling_state_path.exists():
        print(
            "[manifest] WARNING: sampling_state.json not found at "
            f"{sampling_state_path}; sampled rows will have no batch_id."
        )
        return lookup
    try:
        state = json.loads(sampling_state_path.read_text(encoding="utf-8"))
        for batch in state.get("batches", []):
            batch_id = int(batch.get("batch_id", 0))
            for file_entry in batch.get("files", []):
                file_key = file_entry.get("file_key")
                if file_key:
                    lookup[str(file_key)] = batch_id
        print(
            f"[manifest] Loaded batch lookup: {len(lookup)} file_keys across "
            f"{len(state.get('batches', []))} batches."
        )
    except Exception as exc:
        print(f"[manifest] WARNING: Failed to load sampling_state.json: {exc}")
    return lookup


def _language_from_file_key(file_key: str) -> str | None:
    parts = str(file_key).split("_")
    if len(parts) < 4:
        return None
    language_code = parts[3].strip().upper()
    if language_code == "EN":
        return "en"
    if language_code in {"ZH", "CMN"}:
        return "zh"
    return None


def _speaker_id(file_key: str) -> str | None:
    return next(
        (part for part in str(file_key).split("_") if part.startswith("U")),
        None,
    )


def _annotation_records(
    config: dict[str, Any],
    *,
    include_languages: set[str],
    training_phone_types: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_config = config.get("data", {})
    speakers = {str(speaker) for speaker in data_config.get("annotated_speakers", [])}
    audio_dirs = data_config.get("audio_dirs", {})
    textgrid_dirs = data_config.get("textgrid_dirs", {})
    min_duration_ms = float(data_config.get("min_duration_ms", 30.0))

    for language in include_languages:
        if language not in audio_dirs or language not in textgrid_dirs:
            raise KeyError(
                f"Config must define audio_dirs and textgrid_dirs for language {language!r}"
            )
        audio_dir = Path(audio_dirs[language])
        textgrid_dir = Path(textgrid_dirs[language])
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory does not exist for {language}: {audio_dir}")
        if not textgrid_dir.is_dir():
            raise FileNotFoundError(
                f"TextGrid directory does not exist for {language}: {textgrid_dir}"
            )

    sampling_state_path = (
        WAVE_ROOT / "04_creaky_tools" / "_annotation_sampling" / "sampling_state.json"
    )
    batch_lookup = _build_batch_lookup(sampling_state_path)
    inventory_by_language = {
        language: get_phone_inventory(language) for language in include_languages
    }
    stats: Counter[str] = Counter()
    unknown_labels: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str]] = set()

    for store_language in sorted(include_languages):
        textgrid_dir = Path(textgrid_dirs[store_language])
        audio_dir = Path(audio_dirs[store_language])
        annotation_path = annotation_storage_path(textgrid_dir)
        print(f"[manifest] Loading annotations from {annotation_path}...")
        all_annotations = load_annotations(annotation_path)

        files_with_annotations = all_annotations.get("files", {})
        if not isinstance(files_with_annotations, dict):
            continue

        for file_key, entry in tqdm(
            files_with_annotations.items(),
            desc=f"Processing {store_language} annotations",
        ):
            file_key = str(file_key)
            language = _language_from_file_key(file_key)
            if language not in include_languages:
                stats["skipped_language"] += 1
                continue
            speaker_id = _speaker_id(file_key)
            if not speaker_id:
                stats["skipped_missing_speaker"] += 1
                continue
            if not isinstance(entry, dict) or not isinstance(entry.get("annotations"), dict):
                stats["skipped_malformed_entry"] += 1
                continue

            audio_dir_for_file = Path(audio_dirs[language])
            tg_dir_for_file = Path(textgrid_dirs[language])
            audio_path = next(
                (
                    audio_dir_for_file / f"{file_key}{suffix}"
                    for suffix in (".wav", ".flac")
                    if (audio_dir_for_file / f"{file_key}{suffix}").exists()
                ),
                None,
            )
            tg_path = tg_dir_for_file / f"{file_key}.TextGrid"
            if audio_path is None or not tg_path.exists():
                stats["skipped_missing_source"] += 1
                continue

            annotation_type = "full" if speaker_id in speakers else "sampled"
            inventory = inventory_by_language[language]
            for raw_interval_id, label in entry["annotations"].items():
                if label not in {"creaky", "non_creaky"}:
                    stats["skipped_state"] += 1
                    continue

                parts = str(raw_interval_id).split("|")
                if len(parts) != 3:
                    stats["skipped_invalid_id"] += 1
                    continue
                try:
                    start_s = float(parts[0])
                    end_s = float(parts[1])
                except (TypeError, ValueError):
                    stats["skipped_invalid_id"] += 1
                    continue
                phone = parts[2]
                interval_id = make_interval_id(start_s, end_s, phone)
                if (file_key, interval_id) in seen_rows:
                    stats["skipped_duplicate"] += 1
                    continue
                seen_rows.add((file_key, interval_id))

                if (end_s - start_s) * 1000.0 < min_duration_ms:
                    stats["skipped_too_short"] += 1
                    continue

                phone_type = classify_phone_label(phone, inventory)
                if phone_type == IntervalType.UNKNOWN:
                    unknown_labels[phone] += 1
                    stats["skipped_unknown"] += 1
                    continue
                if phone_type.value not in training_phone_types:
                    stats["skipped_phone_type"] += 1
                    continue

                batch_id = batch_lookup.get(file_key)
                is_golden = bool(entry.get("is_golden_file", False))
                records.append(
                    {
                        "file_key": file_key,
                        "audio_path": str(audio_path),
                        "tg_path": str(tg_path),
                        "start_s": start_s,
                        "end_s": end_s,
                        "phone": canonical_phone_for_id(phone),
                        "phone_type": phone_type.value,
                        "speaker_id": speaker_id,
                        "language": language,
                        "label": 1 if label == "creaky" else 0,
                        "annotation_type": annotation_type,
                        "batch_id": batch_id,
                        "is_golden": is_golden,
                    }
                )
                if is_golden:
                    stats["golden_files_included"] += 1
                stats["included"] += 1

    report: dict[str, Any] = {
        "rows": len(records),
        "filters": {
            "include_languages": sorted(include_languages),
            "training_phone_types": sorted(training_phone_types),
            "min_duration_ms": min_duration_ms,
        },
        "excluded": dict(sorted(stats.items())),
        "unknown_top_labels": dict(unknown_labels.most_common(20)),
    }
    return records, report


def _print_manifest_stats(df: pd.DataFrame, report: dict[str, Any]) -> None:
    print(f"[manifest] Included rows: {len(df)}")
    if df.empty:
        print("[manifest] No rows passed the language/type/state/duration filters.")
        return
    report["counts"] = {}
    for column in (
        "language",
        "phone_type",
        "label",
        "phone",
        "speaker_id",
        "annotation_type",
    ):
        counts = df.groupby(column, dropna=False).size().sort_values(ascending=False)
        report["counts"][column] = {
            str(key): int(value) for key, value in counts.items()
        }
        print(f"[manifest] Counts by {column}:")
        print(counts.to_string())
    class_counts = df["label"].value_counts().to_dict()
    positive_ratio = float(df["label"].mean())
    report["label_counts"] = {
        "non_creaky": int(class_counts.get(0, 0)),
        "creaky": int(class_counts.get(1, 0)),
    }
    report["positive_ratio"] = positive_ratio
    print(
        f"[manifest] Label counts: non_creaky={class_counts.get(0, 0)}, "
        f"creaky={class_counts.get(1, 0)}, positive_ratio={positive_ratio:.4f}"
    )
    if report.get("unknown_top_labels"):
        print(f"[manifest] UNKNOWN labels excluded: {report['unknown_top_labels']}")


def build_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Build and return the training DataFrame.

    The default configuration includes English vowel, voiceless-obstruent,
    silence, and sonorant intervals; callers may select any known phone types.
    """

    data_config = config.get("data", {})
    include_languages = {
        str(language).strip().lower()
        for language in data_config.get("include_languages", ["en"])
        if str(language).strip()
    }
    if not include_languages:
        raise ValueError("data.include_languages must contain at least one language")
    training_phone_types = {
        str(phone_type).strip().lower()
        for phone_type in data_config.get(
            "training_phone_types",
            sorted(DEFAULT_TRAINING_PHONE_TYPES),
        )
        if str(phone_type).strip()
    }
    known_phone_types = {item.value for item in IntervalType if item != IntervalType.UNKNOWN}
    unknown_config_types = training_phone_types - known_phone_types
    if unknown_config_types:
        raise ValueError(f"Unknown training phone types: {sorted(unknown_config_types)}")

    records, report = _annotation_records(
        config,
        include_languages=include_languages,
        training_phone_types=training_phone_types,
    )
    columns = [
        "file_key",
        "audio_path",
        "tg_path",
        "start_s",
        "end_s",
        "phone",
        "phone_type",
        "speaker_id",
        "language",
        "label",
        "annotation_type",
        "batch_id",
        "is_golden",
    ]
    df = pd.DataFrame(records, columns=columns)
    _print_manifest_stats(df, report)
    df.attrs["manifest_report"] = report
    return df
