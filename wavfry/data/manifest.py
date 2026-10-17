"""Build a safe, typed, multi-dataset training manifest with extensible adapters."""

from __future__ import annotations

import json
import logging
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


WAVE_ROOT = Path(__file__).resolve().parents[3]
if str(WAVE_ROOT) not in sys.path:
    sys.path.insert(0, str(WAVE_ROOT))

# Lightweight imports for phone inventory & manual annotation (graceful fallback if 04_creaky_tools missing)
try:
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
        fromlist=[
            "load_annotations",
            "annotation_storage_path",
            "ANNOTATION_FILENAME",
            "is_kappa_batch",
            "get_kappa_eval_dir",
            "load_kappa_consensus_annotations",
        ],
    )
    load_annotations = manual_annotation.load_annotations
    annotation_storage_path = manual_annotation.annotation_storage_path
    ANNOTATION_FILENAME = getattr(manual_annotation, "ANNOTATION_FILENAME", "manual_creaky_annotations.json")
    is_kappa_batch = getattr(manual_annotation, "is_kappa_batch", lambda b, cache_dir=None, task="creaky": False)
    get_kappa_eval_dir = getattr(manual_annotation, "get_kappa_eval_dir", lambda b, bid, task="creaky": None)
    load_kappa_consensus_annotations = getattr(manual_annotation, "load_kappa_consensus_annotations", None)
    _HAS_CREAKY_TOOLS = True
except (ImportError, ModuleNotFoundError):
    _HAS_CREAKY_TOOLS = False
    IntervalType = None
    canonical_phone_for_id = lambda p: p
    classify_phone_label = None
    get_phone_inventory = None
    make_interval_id = lambda s, e, p: f"{float(s):.3f}|{float(e):.3f}|{p}"
    load_annotations = None
    annotation_storage_path = None
    ANNOTATION_FILENAME = "manual_creaky_annotations.json"
    is_kappa_batch = lambda b, cache_dir=None, task="creaky": False
    get_kappa_eval_dir = lambda b, bid, task="creaky": None
    load_kappa_consensus_annotations = None

DEFAULT_TRAINING_PHONE_TYPES = {
    "vowel",
    "voiceless_obs",
    "silence",
}

MANIFEST_COLUMNS = [
    "dataset_id",
    "domain",
    "file_key",
    "audio_path",
    "tg_path",
    "start_s",
    "end_s",
    "speaker_id",
    "language",
    "phone",
    "phone_type",
    "annotation_granularity",
    "label_creaky",
    "label_breathy",
    "label",
    "batch_split",
    "annotation_type",
    "batch_id",
    "batch_name",
    "is_frame_verified",
]



def _language_from_file_key(file_key: str) -> str | None:
    parts = str(file_key).split("_")
    if len(parts) < 4:
        return None
    language_code = parts[3].strip().upper()
    if language_code == "EN":
        return "en"
    if language_code in {"ZH", "CMN"}:
        return "zh"
    if language_code in {"JA", "JPN"}:
        return "ja"
    return None


def _speaker_id(file_key: str) -> str | None:
    return next(
        (part for part in str(file_key).split("_") if part.startswith(("U", "C")) and part[1:].isdigit()),
        None,
    )


# ── 适配器函数实现 (Adapters) ────────────────────────────────────────────────

def adapter_cs_dialogue_textgrid(ds_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """CS-Dialogue MFA TextGrid + Sharded Annotation Adapter."""
    if not _HAS_CREAKY_TOOLS:
        raise ImportError(
            "[manifest] '04_creaky_tools' module is required for 'cs_dialogue_textgrid' adapter, "
            "but could not be imported. Please ensure '04_creaky_tools' is available."
        )
    params = ds_cfg.get("adapter_params", {})
    dataset_id = ds_cfg.get("dataset_id", "cs_dialogue")
    domain = ds_cfg.get("domain", "conversational_efl")
    granularity = params.get("annotation_granularity", "segment")

    label_schema = ds_cfg.get("label_schema", {})
    breathy_spec = label_schema.get("breathy", False)
    breathy_silence_as_neg = (
        isinstance(breathy_spec, dict) and bool(breathy_spec.get("silence_as_negative", False))
    )

    speakers = {str(s) for s in params.get("annotated_speakers", [])}
    audio_dirs = params.get("audio_dirs", {})
    textgrid_dirs = params.get("textgrid_dirs", {})
    min_duration_ms = float(params.get("min_duration_ms", 30.0))
    include_languages = {str(l).strip().lower() for l in params.get("include_languages", ["en"])}
    training_phone_types = {str(p).strip().lower() for p in params.get("training_phone_types", sorted(DEFAULT_TRAINING_PHONE_TYPES))}

    for language in include_languages:
        if language not in audio_dirs or language not in textgrid_dirs:
            raise KeyError(
                f"adapter_cs_dialogue_textgrid requires audio_dirs and textgrid_dirs for {language!r}"
            )
        if not Path(audio_dirs[language]).is_dir():
            raise FileNotFoundError(f"Audio dir not found for {language}: {audio_dirs[language]}")
        if not Path(textgrid_dirs[language]).is_dir():
            raise FileNotFoundError(f"TextGrid dir not found for {language}: {textgrid_dirs[language]}")

    inventory_by_language = {
        lang: get_phone_inventory(lang) for lang in include_languages
    }

    records: list[dict[str, Any]] = []
    seen_rows: set[tuple[str, str]] = set()
    seen_file_splits: dict[str, tuple[int | None, str]] = {}
    stats: Counter[str] = Counter()

    for store_language in sorted(include_languages):
        textgrid_dir = Path(textgrid_dirs[store_language])
        audio_dir = Path(audio_dirs[store_language])

        # Sampling state is co-located with TextGrid cache: <textgrid_dir>/_creaky_cache/sampling_state.json
        sampling_state_path = textgrid_dir / "_creaky_cache" / "sampling_state.json"
        if not sampling_state_path.exists():
            raise FileNotFoundError(
                f"[manifest] [{dataset_id}] Sampling state file not found at expected location: {sampling_state_path}. "
                "Co-located per-batch storage is required; legacy sampling paths are deprecated."
            )

        batches_dir = textgrid_dir / "_creaky_cache" / "batches"
        if not batches_dir.exists() or not any(batches_dir.glob("batch_*")):
            raise FileNotFoundError(
                f"[manifest] [{dataset_id}] Batches directory not found or empty at: {batches_dir}. "
                "Physical per-batch storage is required; unbatched/aggregated storage is no longer supported."
            )

        batch_items = []
        try:
            state = json.loads(sampling_state_path.read_text(encoding="utf-8"))
            for b in state.get("batches", []):
                b_id = b["batch_id"]
                b_name = b.get("name") or f"Batch {b_id}"
                b_split = b.get("split") or "unassigned"
                b_path = annotation_storage_path(textgrid_dir, batch_id=b_id)
                batch_items.append((b_id, b_name, b_split, b_path, b))
        except Exception as e:
            raise RuntimeError(
                f"[manifest] [{dataset_id}] Failed to read batches from {sampling_state_path}: {e}"
            ) from e

        if not batch_items:
            raise ValueError(
                f"[manifest] [{dataset_id}] No batches defined in {sampling_state_path}."
            )

        # Fail-Fast: physical batch directories absent from sampling_state are
        # invisible to the manifest — annotations there would silently vanish.
        def _dir_batch_id(name: str):
            try:
                return int(name.rsplit("_", 1)[-1])
            except ValueError:
                return None

        registered_ids = {b[0] for b in batch_items}
        unregistered_dirs = [
            p.name for p in batches_dir.glob("batch_*")
            if p.is_dir() and _dir_batch_id(p.name) is not None
            and _dir_batch_id(p.name) not in registered_ids
        ]

        missing_media: list[str] = []
        target_task = "breathy" if (breathy_spec and not label_schema.get("creaky", True)) else "creaky"

        for b_id, b_name, b_split, annotation_path, b_entry in batch_items:
            if b_split not in {"train", "val", "test"}:
                continue

            # Check if this batch is a Kappa Test Batch
            is_kappa = is_kappa_batch(b_entry, cache_dir=textgrid_dir, task=target_task)
            kappa_eval_dir = get_kappa_eval_dir(textgrid_dir, batch_id=b_id, task=target_task) if is_kappa else None

            if is_kappa and kappa_eval_dir and load_kappa_consensus_annotations is not None:
                print(f"[manifest] [{dataset_id}] Batch [{b_id}] '{b_name}' recognized as KAPPA TEST BATCH.")
                print(f"[manifest] [{dataset_id}] Loading multi-annotator consensus from: {kappa_eval_dir}...")
                batch_annotations = load_kappa_consensus_annotations(kappa_eval_dir, task=target_task)
                n_unanimous = batch_annotations.get("total_unanimous", 0)
                n_disputed = batch_annotations.get("total_disputed", 0)
                print(f"[manifest] [{dataset_id}] Kappa Consensus: {n_unanimous} unanimous intervals kept, {n_disputed} disputed/unverified intervals excluded.")
            else:
                if not annotation_path.exists() and not (annotation_path.with_name(annotation_path.name + ".d")).exists():
                    raise FileNotFoundError(
                        f"[manifest] [{dataset_id}] Annotation storage for Batch [{b_id}] '{b_name}' not found at: {annotation_path}"
                    )
                print(f"[manifest] [{dataset_id}] Loading Batch [{b_id}] '{b_name}' (split='{b_split}') from {annotation_path}...")
                batch_annotations = load_annotations(annotation_path)

            files_with_annotations = batch_annotations.get("files", {})
            if not isinstance(files_with_annotations, dict):
                continue

            for file_key, entry in tqdm(
                files_with_annotations.items(),
                desc=f"Processing {b_name} ({store_language})",
            ):
                file_key = str(file_key)
                file_b_id = b_id
                file_b_name = b_name
                file_b_split = b_split

                if file_b_split not in {"train", "val", "test"}:
                    continue

                if file_key in seen_file_splits:
                    prev_id, prev_split = seen_file_splits[file_key]
                    if prev_split != file_b_split:
                        raise ValueError(
                            f"[manifest] Split Exclusivity Violation: File '{file_key}' assigned to conflicting splits "
                            f"('{prev_split}' in Batch [{prev_id}] vs '{file_b_split}' in Batch [{file_b_id}])."
                        )
                seen_file_splits[file_key] = (file_b_id, file_b_split)

                language = _language_from_file_key(file_key)
                if language is None or language not in include_languages:
                    continue
                speaker_id = _speaker_id(file_key)
                if not speaker_id:
                    continue
                if not isinstance(entry, dict) or not isinstance(entry.get("annotations"), dict):
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
                    missing_media.append(f"{file_key} (batch {file_b_id}, split '{file_b_split}')")
                    continue

                annotation_type = "full" if speaker_id in speakers else "sampled"
                inventory = inventory_by_language[language]

                for raw_interval_id, label in entry["annotations"].items():
                    if label not in {"creaky", "non_creaky"}:
                        continue

                    parts = str(raw_interval_id).split("|")
                    if len(parts) != 3:
                        continue
                    try:
                        start_s = float(parts[0])
                        end_s = float(parts[1])
                    except (TypeError, ValueError):
                        continue
                    phone = parts[2]
                    interval_id = make_interval_id(start_s, end_s, phone)
                    if (file_key, interval_id) in seen_rows:
                        continue
                    seen_rows.add((file_key, interval_id))

                    if (end_s - start_s) * 1000.0 < min_duration_ms:
                        continue

                    phone_type = classify_phone_label(phone, inventory)
                    if phone_type == IntervalType.UNKNOWN or phone_type.value not in training_phone_types:
                        continue

                    is_frame_verified = bool(entry.get("is_frame_verified", entry.get("is_golden_file", False)))
                    creak_val = 1.0 if label == "creaky" else 0.0

                    records.append(
                        {
                            "dataset_id": dataset_id,
                            "domain": domain,
                            "file_key": file_key,
                            "audio_path": str(audio_path),
                            "tg_path": str(tg_path),
                            "start_s": start_s,
                            "end_s": end_s,
                            "speaker_id": speaker_id,
                            "language": language,
                            "phone": canonical_phone_for_id(phone),
                            "phone_type": phone_type.value,
                            "annotation_granularity": granularity,
                            "label_creaky": creak_val,
                            "label_breathy": (
                                0.0 if (phone_type == IntervalType.SILENCE and breathy_silence_as_neg) else float("nan")
                            ),
                            "label": creak_val,
                            "batch_split": file_b_split,
                            "annotation_type": annotation_type,
                            "batch_id": file_b_id,
                            "batch_name": file_b_name,
                            "is_frame_verified": is_frame_verified,
                        }
                    )

    lenient = bool(ds_cfg.get("adapter_params", {}).get("lenient_missing", False))
    problems = []
    if unregistered_dirs:
        problems.append(
            f"{len(unregistered_dirs)} batch directory(ies) on disk are NOT registered in "
            f"{sampling_state_path.name}: {unregistered_dirs[:10]} - their annotations are "
            "invisible to the manifest (stale/mis-synced sampling_state?)."
        )
    if missing_media:
        problems.append(
            f"{len(missing_media)} annotated file(s) have no audio/TextGrid on disk: "
            f"{missing_media[:10]}{'...' if len(missing_media) > 10 else ''} - the dataset "
            "would silently shrink, possibly removing an entire val/test speaker."
        )
    if problems:
        detail = "\n  - ".join(problems)
        if lenient:
            print(f"[manifest] [{dataset_id}] [WARN] Data integrity issues (lenient mode):\n  - {detail}")
        else:
            raise RuntimeError(
                f"[manifest] [{dataset_id}] Data integrity check failed.\n  - {detail}\n"
                "Set adapter_params.lenient_missing=true to downgrade to warnings."
            )

    return records


def adapter_grbas_csv(ds_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """GRBAS Voice Disorder CSV Adapter (utterance-level MIL)."""
    import torchaudio

    params = ds_cfg.get("adapter_params", {})
    dataset_id = ds_cfg.get("dataset_id", "femh_2018")
    domain = ds_cfg.get("domain", "pathological_voice")
    granularity = params.get("annotation_granularity", "utterance")

    audio_dir = Path(params.get("audio_dir", ""))
    csv_path = Path(params.get("annotation_csv", ""))
    col_map = params.get("csv_columns", {"file_id": "file_id", "grbas_b": "B", "speaker_id": "speaker_id"})
    threshold = float(params.get("breathy_threshold", 1.0))

    if not csv_path.exists() or not audio_dir.exists():
        print(f"[manifest] [{dataset_id}] Notice: CSV '{csv_path}' or audio dir '{audio_dir}' not found. Returning empty.")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[manifest] [{dataset_id}] Failed to read CSV {csv_path}: {e}")
        return []

    file_col = col_map.get("file_id", "file_id")
    b_col = col_map.get("grbas_b", "B")
    spk_col = col_map.get("speaker_id", "speaker_id")

    records = []
    for _, row in df.iterrows():
        file_id = str(row[file_col]).strip()
        audio_path = next(
            (audio_dir / f"{file_id}{ext}" for ext in (".wav", ".flac") if (audio_dir / f"{file_id}{ext}").exists()),
            None,
        )
        if audio_path is None:
            continue

        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
        except Exception:
            duration = 2.0  # fallback

        b_score = float(row[b_col]) if not pd.isna(row[b_col]) else float("nan")
        breathy_val = 1.0 if (not math.isnan(b_score) and b_score >= threshold) else (0.0 if not math.isnan(b_score) else float("nan"))
        speaker_id = str(row[spk_col]) if spk_col in row and not pd.isna(row[spk_col]) else file_id

        records.append({
            "dataset_id": dataset_id,
            "domain": domain,
            "file_key": file_id,
            "audio_path": str(audio_path),
            "tg_path": "",
            "start_s": 0.0,
            "end_s": round(duration, 4),
            "speaker_id": speaker_id,
            "language": "zh",
            "phone": "VOWEL",
            "phone_type": "vowel",
            "annotation_granularity": granularity,
            "label_creaky": float("nan"),
            "label_breathy": breathy_val,
            "label": breathy_val,
            "batch_split": "train",
            "annotation_type": "external",
            "batch_id": float("nan"),
            "batch_name": "external",
            "is_frame_verified": False,
        })
    return records


def adapter_capev_csv(ds_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """CAPE-V Ratings CSV Adapter (threshold or continuous soft label)."""
    import torchaudio

    params = ds_cfg.get("adapter_params", {})
    dataset_id = ds_cfg.get("dataset_id", "pvqd")
    domain = ds_cfg.get("domain", "pathological_voice")
    granularity = params.get("annotation_granularity", "utterance")

    audio_dir = Path(params.get("audio_dir", ""))
    csv_path = Path(params.get("annotation_csv", ""))
    col_map = params.get("csv_columns", {"file_id": "file_id", "capev_breathiness": "Breathiness", "speaker_id": "SpeakerID"})
    label_mode = params.get("breathy_label_mode", "threshold")
    threshold = float(params.get("breathy_threshold", 40.0))

    if not csv_path.exists() or not audio_dir.exists():
        print(f"[manifest] [{dataset_id}] Notice: CSV '{csv_path}' or audio dir '{audio_dir}' not found. Returning empty.")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[manifest] [{dataset_id}] Failed to read CSV {csv_path}: {e}")
        return []

    file_col = col_map.get("file_id", "file_id")
    b_col = col_map.get("capev_breathiness", "Breathiness")
    spk_col = col_map.get("speaker_id", "SpeakerID")

    records = []
    for _, row in df.iterrows():
        file_id = str(row[file_col]).strip()
        audio_path = next(
            (audio_dir / f"{file_id}{ext}" for ext in (".wav", ".flac") if (audio_dir / f"{file_id}{ext}").exists()),
            None,
        )
        if audio_path is None:
            continue

        try:
            info = torchaudio.info(str(audio_path))
            duration = info.num_frames / info.sample_rate
        except Exception:
            duration = 2.0

        raw_val = float(row[b_col]) if not pd.isna(row[b_col]) else float("nan")
        if math.isnan(raw_val):
            breathy_val = float("nan")
        elif label_mode == "threshold":
            breathy_val = 1.0 if raw_val >= threshold else 0.0
        else:
            breathy_val = max(0.0, min(1.0, raw_val / 100.0))

        speaker_id = str(row[spk_col]) if spk_col in row and not pd.isna(row[spk_col]) else file_id

        records.append({
            "dataset_id": dataset_id,
            "domain": domain,
            "file_key": file_id,
            "audio_path": str(audio_path),
            "tg_path": "",
            "start_s": 0.0,
            "end_s": round(duration, 4),
            "speaker_id": speaker_id,
            "language": "en",
            "phone": "VOWEL",
            "phone_type": "vowel",
            "annotation_granularity": granularity,
            "label_creaky": float("nan"),
            "label_breathy": breathy_val,
            "label": breathy_val,
            "batch_split": "train",
            "annotation_type": "external",
            "batch_id": float("nan"),
            "batch_name": "external",
            "is_frame_verified": False,
        })
    return records


def adapter_generic_csv(ds_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Generic CSV Adapter for arbitrary custom annotations."""
    params = ds_cfg.get("adapter_params", {})
    dataset_id = ds_cfg.get("dataset_id", "custom_csv")
    domain = ds_cfg.get("domain", "conversational_efl")
    granularity = params.get("annotation_granularity", "segment")

    csv_path = Path(params.get("csv_path", ""))
    audio_base = Path(params.get("audio_base_dir", ""))

    if not csv_path.exists():
        print(f"[manifest] [{dataset_id}] Notice: CSV '{csv_path}' not found. Returning empty.")
        return []

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[manifest] [{dataset_id}] Failed to read CSV {csv_path}: {e}")
        return []

    records = []
    for _, row in df.iterrows():
        raw_audio = str(row.get("audio_path", ""))
        audio_path = Path(raw_audio) if Path(raw_audio).is_absolute() else audio_base / raw_audio
        if not audio_path.exists():
            continue

        file_key = str(row.get("file_key", audio_path.stem))
        start_s = float(row.get("start_s", 0.0))
        end_s = float(row.get("end_s", 0.0))
        min_duration_ms = float(params.get("min_duration_ms", 0.0))
        if min_duration_ms > 0 and (end_s - start_s) * 1000.0 < min_duration_ms:
            continue
        speaker_id = str(row.get("speaker_id", "UNK"))
        lang = str(row.get("language", "en"))
        phone = str(row.get("phone", "VOWEL"))
        phone_type = str(row.get("phone_type", "vowel"))
        l_creaky = float(row["label_creaky"]) if "label_creaky" in row and not pd.isna(row["label_creaky"]) else float("nan")
        l_breathy = float(row["label_breathy"]) if "label_breathy" in row and not pd.isna(row["label_breathy"]) else float("nan")
        b_split = str(row.get("batch_split", "train"))

        records.append({
            "dataset_id": dataset_id,
            "domain": domain,
            "file_key": file_key,
            "audio_path": str(audio_path),
            "tg_path": str(row.get("tg_path", "")),
            "start_s": start_s,
            "end_s": end_s,
            "speaker_id": speaker_id,
            "language": lang,
            "phone": phone,
            "phone_type": phone_type,
            "annotation_granularity": granularity,
            "label_creaky": l_creaky,
            "label_breathy": l_breathy,
            "label": l_creaky if not math.isnan(l_creaky) else l_breathy,
            "batch_split": b_split,
            "annotation_type": str(row.get("annotation_type", "custom")),
            "batch_id": float(row.get("batch_id", float("nan"))),
            "batch_name": str(row.get("batch_name", "")),
            "is_frame_verified": bool(row.get("is_frame_verified", row.get("is_golden", False))),
        })
    return records


def _load_standalone_sharded_annotations(manifest_path: Path) -> dict[str, Any]:
    shard_dir = manifest_path.with_name(manifest_path.name + ".d")
    files: dict[str, Any] = {}
    if shard_dir.exists():
        for sf in shard_dir.glob("*.json"):
            try:
                data = json.loads(sf.read_text(encoding="utf-8"))
                f_key = data.get("file_key") or sf.stem
                files[f_key] = data
            except Exception:
                pass
    elif manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = data.get("files", {})
        except Exception:
            pass
    return {"version": 1, "schema": "direct_breathy_annotation", "files": files}


def adapter_breathy_cache(ds_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Load annotations from a standardized _breathy_cache directory.

    Expected adapter_params:
      audio_dir: Path to directory containing audio (.wav/.flac) files.
      cache_dir: (Optional) Path to _breathy_cache directory. Defaults to <audio_dir>/_breathy_cache.
      textgrid_dir: (Optional) Path to textgrid directory.
      language: (Optional) Language code, e.g. "maj", "guj", "hmn".
      annotation_granularity: (Optional) "segment" (default).
    """
    params = ds_cfg.get("adapter_params", {})
    dataset_id = ds_cfg.get("dataset_id", "ucla_breathy")
    domain = ds_cfg.get("domain", "phonetic_controlled")
    granularity = params.get("annotation_granularity", "segment")
    language = params.get("language", "und")
    min_duration_ms = float(params.get("min_duration_ms", 30.0))

    audio_dir = Path(params.get("audio_dir", ""))
    cache_dir = Path(params.get("cache_dir", "")) if params.get("cache_dir") else (audio_dir / "_breathy_cache")
    textgrid_dir = Path(params.get("textgrid_dir", "")) if params.get("textgrid_dir") else None

    manifest_file = cache_dir / "manual_breathy_annotations.json"
    if not manifest_file.exists() and not (manifest_file.with_name(manifest_file.name + ".d")).exists():
        manifest_file = cache_dir / "manual_creaky_annotations.json"
        if not manifest_file.exists() and not (manifest_file.with_name(manifest_file.name + ".d")).exists():
            print(f"[manifest] [{dataset_id}] Notice: Breathy cache manifest not found in {cache_dir}. Returning empty.")
            return []

    # Load annotations via load_annotations (supports sharded .d and monolith)
    if _HAS_CREAKY_TOOLS and load_annotations is not None:
        cache_data = load_annotations(manifest_file)
    else:
        cache_data = _load_standalone_sharded_annotations(manifest_file)

    files = cache_data.get("files", {})
    if not isinstance(files, dict) or not files:
        print(f"[manifest] [{dataset_id}] Notice: No files in breathy cache {manifest_file}.")
        return []

    # Map audio files
    audio_files: dict[str, Path] = {}
    if audio_dir.exists():
        for ext in ("*.wav", "*.flac"):
            for p in audio_dir.rglob(ext):
                if "_breathy_cache" in p.parts or "_creaky_cache" in p.parts:
                    continue
                audio_files[p.stem] = p

    # Map TextGrid files if textgrid_dir exists
    tg_files: dict[str, Path] = {}
    if textgrid_dir and textgrid_dir.exists():
        for p in textgrid_dir.rglob("*.TextGrid"):
            tg_files[p.stem] = p

    label_schema = ds_cfg.get("label_schema", {})
    has_breathy_schema = bool(label_schema.get("breathy", True))
    has_creaky_schema = bool(label_schema.get("creaky", False))

    records: list[dict[str, Any]] = []
    missing_audio: list[str] = []
    lenient = bool(params.get("lenient_missing", False))

    for file_key, entry in files.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("annotations"), dict):
            continue

        audio_path = audio_files.get(file_key)
        if audio_path is None:
            missing_audio.append(file_key)
            continue
        tg_path = tg_files.get(file_key, "")

        speaker_id = str(entry.get("speaker_id") or _speaker_id(file_key) or audio_path.parent.name)

        for raw_interval_id, state in entry["annotations"].items():
            parts = str(raw_interval_id).split("|")
            if len(parts) != 3:
                continue
            try:
                start_s = float(parts[0])
                end_s = float(parts[1])
            except (TypeError, ValueError):
                continue
            phone = parts[2]
            if (end_s - start_s) * 1000.0 < min_duration_ms:
                continue

            # Determine label_breathy
            if not has_breathy_schema:
                l_breathy = float("nan")
            elif state == "breathy":
                l_breathy = 1.0
            elif state in ("non_breathy", "modal", "creaky"):
                l_breathy = 0.0
            else:  # uncertain or unknown
                l_breathy = float("nan")

            # Determine label_creaky
            if not has_creaky_schema:
                l_creaky = float("nan")
            elif state == "creaky":
                l_creaky = 1.0
            elif state in ("non_breathy", "modal", "breathy"):
                l_creaky = 0.0
            else:
                l_creaky = float("nan")

            # Skip intervals where neither task has a valid label
            if math.isnan(l_breathy) and math.isnan(l_creaky):
                continue

            label_val = l_breathy if not math.isnan(l_breathy) else l_creaky

            records.append({
                "dataset_id": dataset_id,
                "domain": domain,
                "file_key": file_key,
                "audio_path": str(audio_path),
                "tg_path": str(tg_path) if tg_path else "",
                "start_s": start_s,
                "end_s": end_s,
                "speaker_id": speaker_id,
                "language": language,
                "phone": phone,
                "phone_type": "vowel",
                "annotation_granularity": granularity,
                "label_creaky": l_creaky,
                "label_breathy": l_breathy,
                "label": label_val,
                "batch_split": "unassigned",
                "annotation_type": "phonological_minimal_pair",
                "batch_id": float("nan"),
                "batch_name": "",
                "is_frame_verified": False,
            })

    if missing_audio:
        preview = sorted(missing_audio)[:10]
        msg = (
            f"{len(missing_audio)} annotated file(s) have no audio under {audio_dir}: "
            f"{preview}{'...' if len(missing_audio) > 10 else ''}"
        )
        if lenient:
            print(f"[manifest] [{dataset_id}] [WARN] {msg}")
        else:
            raise RuntimeError(
                f"[manifest] [{dataset_id}] {msg}\n"
                "Set adapter_params.lenient_missing=true to downgrade to a warning."
            )

    print(f"[manifest] [{dataset_id}] Extracted {len(records)} intervals from breathy cache ({cache_dir}).")
    return records


def adapter_sharded_textgrid_cache(ds_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Generic adapter for TextGrid corpora managed by the sharded cache protocol (_creaky_cache / _breathy_cache).

    Compatible with any dataset annotated with 04_creaky_tools (MELI, YSJ, BreathyTTS, ITA, etc.).

    Expected adapter_params:
      audio_dir: Path to directory containing audio (.flac/.wav) files.
      textgrid_dir: Path to directory containing .TextGrid files.
      cache_dir: (Optional) Path to cache directory. Defaults to <textgrid_dir>/_breathy_cache or _creaky_cache.
      manifest_csv: (Optional) Path to manifest_slices.csv.
      language: (Optional) Language code, default 'ja'.
      phone_tier_name: (Optional) default 'phones'.
      min_duration_ms: (Optional) default 30.0.
      min_smr_db: (Optional) Float SMR threshold in dB. Slices with smr_db < min_smr_db are skipped.
      exclude_gender_x: (Optional) If True, slices with gender == 'X' are excluded. Default False.
      lenient_missing: (Optional) default False.
    """
    params = ds_cfg.get("adapter_params", {})
    dataset_id = ds_cfg.get("dataset_id", "ysj_japanese")
    domain = ds_cfg.get("domain", "spontaneous_podcast")
    granularity = params.get("annotation_granularity", "segment")
    language = params.get("language", "ja")
    min_duration_ms = float(params.get("min_duration_ms", 30.0))
    min_smr_db = float(params["min_smr_db"]) if ("min_smr_db" in params and params["min_smr_db"] is not None) else None
    exclude_gender_x = bool(params.get("exclude_gender_x", False))
    lenient = bool(params.get("lenient_missing", False))
    allow_mllm_pseudo_labels = bool(params.get("allow_mllm_pseudo_labels", False))

    audio_dir = Path(params.get("audio_dir", ""))
    textgrid_dir = Path(params.get("textgrid_dir", ""))
    cache_dir = Path(params.get("cache_dir", "")) if params.get("cache_dir") else (textgrid_dir / "_breathy_cache")
    manifest_csv = Path(params.get("manifest_csv", "")) if params.get("manifest_csv") else None

    # Load slice metadata from manifest_csv if present
    slice_meta: dict[str, dict[str, Any]] = {}
    if manifest_csv and manifest_csv.exists():
        import csv
        try:
            with open(manifest_csv, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s_id = row.get("slice_id") or row.get("file_key") or row.get("\ufeffslice_id") or row.get("\ufefffile_key")
                    if not s_id:
                        continue
                    s_id = s_id.strip()
                    spk_uid = row.get("speaker_uid") or row.get("speaker_id") or ""
                    g = row.get("gender", "").strip().upper()
                    try:
                        smr = float(row.get("smr_db", 0.0))
                    except (ValueError, TypeError):
                        smr = 0.0
                    try:
                        mos = float(row.get("mos_score", 0.0))
                    except (ValueError, TypeError):
                        mos = 0.0
                    slice_meta[s_id] = {
                        "speaker_id": spk_uid.strip(),
                        "gender": g,
                        "smr_db": smr,
                        "mos_score": mos,
                    }
        except Exception as e:
            print(f"[manifest] [{dataset_id}] Notice: Error reading manifest_csv {manifest_csv}: {e}")

    # Map audio files
    audio_files: dict[str, Path] = {}
    if audio_dir.exists():
        for ext in ("*.wav", "*.flac"):
            for p in audio_dir.rglob(ext):
                if "_breathy_cache" in p.parts or "_creaky_cache" in p.parts:
                    continue
                audio_files[p.stem] = p

    # Map TextGrid files
    tg_files: dict[str, Path] = {}
    if textgrid_dir.exists():
        for p in textgrid_dir.rglob("*.TextGrid"):
            tg_files[p.stem] = p

    # Collect batches and/or monolithic cache files
    sampling_state_path = cache_dir / "sampling_state.json"
    batches_dir = cache_dir / "batches"
    batch_items: list[tuple[int | None, str, str, Path, dict[str, Any] | None]] = []

    if sampling_state_path.exists():
        try:
            state = json.loads(sampling_state_path.read_text(encoding="utf-8"))
            for b in state.get("batches", []):
                b_id = b["batch_id"]
                b_name = b.get("name") or f"Batch {b_id}"
                b_split = b.get("split") or "unassigned"
                b_anno_dir = batches_dir / f"batch_{b_id}"
                b_path = b_anno_dir / ANNOTATION_FILENAME
                if not b_path.exists() and not (b_anno_dir / f"{ANNOTATION_FILENAME}.d").exists():
                    alt_path = b_anno_dir / "manual_breathy_annotations.json"
                    if alt_path.exists() or (b_anno_dir / "manual_breathy_annotations.json.d").exists():
                        b_path = alt_path
                batch_items.append((b_id, b_name, b_split, b_path, b))
        except Exception as e:
            print(f"[manifest] [{dataset_id}] Warning: Failed to parse sampling_state.json: {e}")

    if not batch_items:
        for fname in ("manual_breathy_annotations.json", ANNOTATION_FILENAME):
            unbatched_path = cache_dir / fname
            if unbatched_path.exists() or (cache_dir / f"{fname}.d").exists():
                batch_items.append((None, "Default", "unassigned", unbatched_path, None))
                break

    if not batch_items:
        print(f"[manifest] [{dataset_id}] Notice: No annotations or batches found in {cache_dir}. Returning empty.")
        return []

    label_schema = ds_cfg.get("label_schema", {})
    if not label_schema:
        has_breathy_schema = True
        breathy_silence_as_neg = False
        has_creaky_schema = False
    else:
        breathy_spec = label_schema.get("breathy", False)
        has_breathy_schema = (
            bool(breathy_spec.get("enabled", True))
            if isinstance(breathy_spec, dict)
            else bool(breathy_spec)
        )
        breathy_silence_as_neg = (
            isinstance(breathy_spec, dict)
            and bool(breathy_spec.get("silence_as_negative", False))
        )
        creaky_spec = label_schema.get("creaky", False)
        has_creaky_schema = (
            bool(creaky_spec.get("enabled", True))
            if isinstance(creaky_spec, dict)
            else bool(creaky_spec)
        )
    anno_task = "creaky" if (has_creaky_schema and not has_breathy_schema) else "breathy"

    records: list[dict[str, Any]] = []
    missing_audio: list[str] = []
    seen_file_splits: dict[str, tuple[int | None, str]] = {}

    for b_id, b_name, b_split, annotation_path, b_entry in batch_items:
        is_kappa = (b_entry is not None) and is_kappa_batch(b_entry, cache_dir=cache_dir, task=anno_task)
        kappa_eval_dir = get_kappa_eval_dir(cache_dir, batch_id=b_id, task=anno_task) if (is_kappa and b_id is not None) else None

        if is_kappa and kappa_eval_dir and load_kappa_consensus_annotations is not None:
            print(f"[manifest] [{dataset_id}] Batch [{b_id}] '{b_name}' recognized as KAPPA TEST BATCH.")
            print(f"[manifest] [{dataset_id}] Loading multi-annotator consensus from: {kappa_eval_dir}...")
            batch_data = load_kappa_consensus_annotations(kappa_eval_dir, task=anno_task)
            n_unanimous = batch_data.get("total_unanimous", 0)
            n_disputed = batch_data.get("total_disputed", 0)
            print(f"[manifest] [{dataset_id}] Kappa Consensus: {n_unanimous} unanimous intervals kept, {n_disputed} disputed/unverified intervals excluded.")
        else:
            if not annotation_path.exists() and not (annotation_path.with_name(annotation_path.name + ".d")).exists():
                continue

            if _HAS_CREAKY_TOOLS and load_annotations is not None:
                batch_data = load_annotations(annotation_path)
            else:
                batch_data = _load_standalone_sharded_annotations(annotation_path)

        files = batch_data.get("files", {})
        if not isinstance(files, dict):
            continue

        for file_key, entry in files.items():
            if not isinstance(entry, dict) or not isinstance(entry.get("annotations"), dict):
                continue
            file_key = str(file_key)

            if b_split in {"train", "val", "test"}:
                if file_key in seen_file_splits:
                    prev_id, prev_split = seen_file_splits[file_key]
                    if prev_split != b_split:
                        raise ValueError(
                            f"[manifest] [{dataset_id}] Split Exclusivity Violation: File '{file_key}' assigned to conflicting splits "
                            f"('{prev_split}' in Batch [{prev_id}] vs '{b_split}' in Batch [{b_id}])."
                        )
                seen_file_splits[file_key] = (b_id, b_split)

            audio_path = audio_files.get(file_key)
            if audio_path is None:
                missing_audio.append(file_key)
                continue
            tg_path = tg_files.get(file_key, "")

            f_meta = slice_meta.get(file_key, {})
            speaker_id = f_meta.get("speaker_id") or _speaker_id(file_key) or entry.get("speaker_id") or audio_path.parent.name
            gender = f_meta.get("gender")
            if not gender:
                parts = file_key.split("_")
                gender = parts[0].upper() if parts and parts[0].upper() in {"M", "F", "X"} else "X"

            if min_smr_db is not None and "smr_db" in f_meta:
                if f_meta["smr_db"] < min_smr_db:
                    continue
            if exclude_gender_x and gender == "X":
                continue

            is_frame_verified = bool(entry.get("is_frame_verified") or entry.get("is_golden_file"))

            for raw_interval_id, state in entry["annotations"].items():
                parts = str(raw_interval_id).split("|")
                if len(parts) != 3:
                    continue
                try:
                    start_s = float(parts[0])
                    end_s = float(parts[1])
                except (TypeError, ValueError):
                    continue
                phone = parts[2]
                if (end_s - start_s) * 1000.0 < min_duration_ms:
                    continue

                phone_type = "vowel"
                if classify_phone_label is not None and get_phone_inventory is not None:
                    try:
                        inv = get_phone_inventory(language)
                        p_t = classify_phone_label(phone, inv)
                        if hasattr(p_t, "value"):
                            phone_type = p_t.value
                        elif p_t:
                            phone_type = str(p_t)
                    except Exception:
                        phone_type = "silence" if phone.upper() in {"SP", "SIL", "SILENCE", "AP", "EP"} else "vowel"
                elif phone.upper() in {"SP", "SIL", "SILENCE", "AP", "EP"}:
                    phone_type = "silence"

                is_silence = (phone_type == "silence") or (phone.upper() in {"SP", "SIL", "SILENCE", "AP", "EP"})

                if not has_breathy_schema:
                    if breathy_silence_as_neg and is_silence:
                        l_breathy = 0.0
                    else:
                        l_breathy = float("nan")
                elif state == "breathy":
                    l_breathy = 1.0
                elif allow_mllm_pseudo_labels and state == "mllm_breathy" and b_split == "train":
                    l_breathy = 1.0
                elif state in ("non_breathy", "modal", "creaky", "non_creaky"):
                    l_breathy = 0.0
                else:
                    l_breathy = float("nan")

                if not has_creaky_schema:
                    l_creaky = float("nan")
                elif state == "creaky":
                    l_creaky = 1.0
                elif state in ("non_creaky", "non_breathy", "modal", "breathy"):
                    l_creaky = 0.0
                else:
                    l_creaky = float("nan")

                if math.isnan(l_breathy) and math.isnan(l_creaky):
                    continue

                label_val = l_creaky if has_creaky_schema else l_breathy

                records.append({
                    "dataset_id": dataset_id,
                    "domain": domain,
                    "file_key": file_key,
                    "audio_path": str(audio_path),
                    "tg_path": str(tg_path) if tg_path else "",
                    "start_s": start_s,
                    "end_s": end_s,
                    "speaker_id": speaker_id,
                    "gender": gender,
                    "language": language,
                    "phone": phone,
                    "phone_type": phone_type,
                    "annotation_granularity": granularity,
                    "label_creaky": l_creaky,
                    "label_breathy": l_breathy,
                    "label": label_val,
                    "batch_split": b_split,
                    "annotation_type": "mllm_pseudo_label" if state == "mllm_breathy" else "manual_annotation",
                    "batch_id": b_id if b_id is not None else float("nan"),
                    "batch_name": b_name,
                    "is_frame_verified": is_frame_verified,
                })

    if missing_audio:
        preview = sorted(missing_audio)[:10]
        msg = (
            f"{len(missing_audio)} annotated file(s) have no audio under {audio_dir}: "
            f"{preview}{'...' if len(missing_audio) > 10 else ''}"
        )
        if lenient:
            print(f"[manifest] [{dataset_id}] [WARN] {msg}")
        else:
            raise RuntimeError(
                f"[manifest] [{dataset_id}] {msg}\n"
                "Set adapter_params.lenient_missing=true to downgrade to a warning."
            )

    print(f"[manifest] [{dataset_id}] Extracted {len(records)} intervals from sharded TextGrid cache ({cache_dir}).")
    return records


# ── 适配器注册表 ─────────────────────────────────────────────────────────────

ADAPTERS: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {
    "cs_dialogue_textgrid": adapter_cs_dialogue_textgrid,
    "grbas_csv": adapter_grbas_csv,
    "capev_csv": adapter_capev_csv,
    "generic_csv": adapter_generic_csv,
    "breathy_cache": adapter_breathy_cache,
    "sharded_textgrid_cache": adapter_sharded_textgrid_cache,
}


# ── 划分策略应用 (Split Strategies) ──────────────────────────────────────────

def _apply_split_strategies(df: pd.DataFrame, datasets_cfg: dict[str, Any]) -> None:
    """Apply dataset-specific partitioning strategies in-place."""
    split_defaults = datasets_cfg.get("split_defaults", {})
    seed = int(split_defaults.get("seed", 42))

    for ds in datasets_cfg.get("datasets", []):
        if not ds.get("enabled", True):
            continue
        ds_id = ds["dataset_id"]
        mask = df["dataset_id"] == ds_id
        if not mask.any():
            continue

        strategy = ds.get("split_strategy", "all_train")

        if strategy == "all_train":
            df.loc[mask, "batch_split"] = "train"

        elif strategy == "batch_split":
            # Already populated by the adapter (e.g. from sampling_state.json), keep as-is
            pass

        elif strategy == "speaker_list":
            val_spks = set(ds.get("val_speakers", []))
            test_spks = set(ds.get("test_speakers", []))
            df.loc[mask & df["speaker_id"].isin(val_spks), "batch_split"] = "val"
            if test_spks:
                df.loc[mask & df["speaker_id"].isin(test_spks), "batch_split"] = "test"
                df.loc[mask & ~df["speaker_id"].isin(val_spks | test_spks), "batch_split"] = "train"
            else:
                df.loc[mask & ~df["speaker_id"].isin(val_spks), "batch_split"] = "train"

        elif strategy == "random":
            spks = sorted(df.loc[mask, "speaker_id"].dropna().unique())
            rng = random.Random(seed)
            rng.shuffle(spks)
            n_val = max(1, int(round(len(spks) * float(split_defaults.get("val_fraction", 0.1)))))
            n_test = max(1, int(round(len(spks) * float(split_defaults.get("test_fraction", 0.1)))))
            val_set = set(spks[:n_val])
            test_set = set(spks[n_val : n_val + n_test])
            train_set = set(spks[n_val + n_test :])

            df.loc[mask & df["speaker_id"].isin(val_set), "batch_split"] = "val"
            df.loc[mask & df["speaker_id"].isin(test_set), "batch_split"] = "test"
            df.loc[mask & df["speaker_id"].isin(train_set), "batch_split"] = "train"


def _print_multi_manifest_stats(df: pd.DataFrame, tasks: list[str]) -> None:
    print(f"\n=======================================================")
    print(f"[Manifest] Multi-Dataset Manifest Built: {len(df)} total rows")
    print(f"[Manifest] Active Tasks ({len(tasks)}): {tasks}")
    print(f"=======================================================")
    if df.empty:
        return
    print("\nCounts by Dataset & Split:")
    print(pd.crosstab(df["dataset_id"], df["batch_split"], margins=True).to_string())

    for task in tasks:
        col = f"label_{task}"
        if col in df.columns:
            valid = df[col].dropna()
            pos = (valid > 0.5).sum()
            ratio = (pos / len(valid)) if len(valid) > 0 else 0.0
            print(f"\nTask [{task.upper()}]: {len(valid)} labeled ({pos} positive, {len(valid) - pos} negative, pos_ratio={ratio:.3f})")


# ── 主入口 (Main Multi-Dataset Manifest Builder) ──────────────────────────────

def build_manifest_multi(
    datasets_cfg: dict[str, Any],
    global_cfg: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Build unified multi-dataset manifest from datasets.yaml specifications."""
    all_records: list[dict[str, Any]] = []
    active_tasks: set[str] = set()

    for ds in datasets_cfg.get("datasets", []):
        if not ds.get("enabled", True):
            continue

        adapter_name = ds.get("annotation_adapter")
        if adapter_name not in ADAPTERS:
            raise ValueError(f"Unknown annotation_adapter: '{adapter_name}'. Registered: {list(ADAPTERS.keys())}")

        schema = ds.get("label_schema", {})
        for task_name, task_spec in schema.items():
            # Deliberate semantics (test_dict_label_schema_silence_as_negative):
            # silence_as_negative constitutes partial supervision for the task
            # (deterministic negatives), so it activates the task alongside
            # `enabled` even when vowel-level annotation is disabled.
            is_active = (
                bool(task_spec.get("enabled", True))
                or bool(task_spec.get("silence_as_negative", False))
                if isinstance(task_spec, dict)
                else bool(task_spec)
            )
            if is_active:
                active_tasks.add(task_name)

        adapter_fn = ADAPTERS[adapter_name]
        records = adapter_fn(ds)
        all_records.extend(records)

    df = pd.DataFrame(all_records, columns=MANIFEST_COLUMNS)
    _apply_split_strategies(df, datasets_cfg)

    # Sort tasks in the model's canonical order (creaky first), not alphabetically
    sorted_tasks = [t for t in ("creaky", "breathy") if t in active_tasks]
    _print_multi_manifest_stats(df, sorted_tasks)

    df.attrs["tasks"] = sorted_tasks
    return df


# ── 向后兼容接口 (Backward Compatibility Wrapper) ─────────────────────────────

def _annotation_records(
    config: dict[str, Any],
    *,
    include_languages: set[str],
    training_phone_types: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Legacy helper for backward compatibility."""
    ds_cfg = {
        "dataset_id": "cs_dialogue",
        "domain": "conversational_efl",
        "adapter_params": {
            "annotated_speakers": config.get("data", {}).get("annotated_speakers", []),
            "audio_dirs": config.get("data", {}).get("audio_dirs", {}),
            "textgrid_dirs": config.get("data", {}).get("textgrid_dirs", {}),
            "phone_tier_name": config.get("data", {}).get("phone_tier_name", "phones"),
            "creaky_tier_name": config.get("data", {}).get("creaky_tier_name", "creaky"),
            "min_duration_ms": config.get("data", {}).get("min_duration_ms", 30.0),
            "include_languages": list(include_languages),
            "training_phone_types": list(training_phone_types),
        },
    }
    records = adapter_cs_dialogue_textgrid(ds_cfg)
    report = {"rows": len(records)}
    return records, report


def _print_manifest_stats(df: pd.DataFrame, report: dict[str, Any]) -> None:
    """Legacy stats printer."""
    tasks = df.attrs.get("tasks", ["creaky"])
    _print_multi_manifest_stats(df, tasks)


def build_manifest(config: dict[str, Any]) -> pd.DataFrame:
    """Build manifest: loads datasets.yaml if present, or falls back to cs_dialogue."""
    if "datasets" in config:
        return build_manifest_multi(config, global_cfg=config)

    data_cfg = config.get("data", {})
    if "audio_dirs" in data_cfg or "textgrid_dirs" in data_cfg:
        ds_cfg = {
            "dataset_id": "cs_dialogue",
            "domain": "conversational_efl",
            "enabled": True,
            "annotation_adapter": "cs_dialogue_textgrid",
            "adapter_params": data_cfg,
            "label_schema": {"creaky": True, "breathy": False},
            "split_strategy": "batch_split",
        }
        datasets_cfg = {"datasets": [ds_cfg]}
        return build_manifest_multi(datasets_cfg, global_cfg=config)

    root = Path(__file__).resolve().parents[2]
    datasets_yaml = root / "config" / "datasets.yaml"
    if datasets_yaml.exists():
        import yaml
        with open(datasets_yaml, encoding="utf-8") as f:
            datasets_cfg = yaml.safe_load(f)
        return build_manifest_multi(datasets_cfg, global_cfg=config)

    # Fallback to single cs_dialogue dataset using existing config
    ds_cfg = {
        "dataset_id": "cs_dialogue",
        "domain": "conversational_efl",
        "enabled": True,
        "annotation_adapter": "cs_dialogue_textgrid",
        "adapter_params": data_cfg,
        "label_schema": {"creaky": True, "breathy": False},
        "split_strategy": "batch_split",
    }
    datasets_cfg = {"datasets": [ds_cfg]}
    return build_manifest_multi(datasets_cfg, global_cfg=config)


if __name__ == "__main__":
    from scripts.build_manifest import main
    main()
