#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert Praat TextGrid annotations into standardized manual_breathy_annotations cache.

Output format (1:1 symmetric to CS-Dialogue's manual_creaky_annotations):
- Manifest: <cache_dir>/manual_breathy_annotations.json
- Shards:   <cache_dir>/manual_breathy_annotations.json.d/<sha256(file_key)>.json

Usage:
    python scripts/convert_textgrid_to_breathy_cache.py \\
        --audio-dir E:/Corpus/Breathy/UCLA/Mazatec/Mazatec_Audio \\
        --textgrid-dir E:/Corpus/Breathy/UCLA/Mazatec/Mazatec_TextGrids \\
        --cache-dir E:/Corpus/Breathy/UCLA/Mazatec/Mazatec_Audio/_breathy_cache
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ANNOTATION_FILENAME = "manual_breathy_annotations.json"
ANNOTATION_STORAGE_MODE = "sharded_json_v2"
ANNOTATION_SHARD_DIR_SUFFIX = ".d"


def parse_textgrid_tiers(tg_path: Path) -> Dict[str, List[Tuple[float, float, str]]]:
    """Parse TextGrid into a dictionary of tier_name -> list of (xmin, xmax, text)."""
    text = None
    for enc in ["utf-8", "utf-16", "latin1", "cp1252"]:
        try:
            text = tg_path.read_text(encoding=enc)
            break
        except Exception:
            pass
    if text is None:
        raise ValueError(f"Unable to read {tg_path} with supported encodings")

    tiers: Dict[str, List[Tuple[float, float, str]]] = {}
    items = text.split("item [")
    for it in items[1:]:
        name_m = re.search(r'name = "(.*?)"', it)
        if not name_m:
            continue
        tname = name_m.group(1).strip()
        int_blocks = re.findall(
            r"xmin = ([\d\.]+)\s+xmax = ([\d\.]+)\s+text = \"(.*?)\"",
            it,
            re.DOTALL,
        )
        intervals = []
        for x1, x2, label in int_blocks:
            intervals.append((float(x1), float(x2), label.strip()))
        tiers[tname] = intervals
    return tiers


# ── Dataset Parser Strategies ──────────────────────────────────────────

def parse_ucla_mazatec(
    tiers: Dict[str, List[Tuple[float, float, str]]],
    file_stem: str,
    audio_path: Path,
    tg_path: Path,
) -> Dict[str, str]:
    """Mazatec: explicit 'phonation' tier ('breathy', 'modal', 'creaky')."""
    annotations: Dict[str, str] = {}
    phonation_intervals = tiers.get("phonation", [])
    vowel_intervals = tiers.get("vowel", []) or tiers.get("vowels", [])

    for p_start, p_end, p_text in phonation_intervals:
        if not p_text:
            continue
        norm_phon = p_text.strip().lower()
        if norm_phon == "breathy":
            state = "breathy"
        elif norm_phon == "modal":
            state = "non_breathy"
        elif norm_phon == "creaky":
            state = "creaky"
        else:
            state = "uncertain"

        phone = "V"
        for v_start, v_end, v_text in vowel_intervals:
            if abs(v_start - p_start) < 0.01 and abs(v_end - p_end) < 0.01:
                if v_text:
                    phone = v_text.strip()
                break

        interval_id = f"{p_start:.4f}|{p_end:.4f}|{phone}"
        annotations[interval_id] = state
    return annotations


def parse_ucla_gujarati(
    tiers: Dict[str, List[Tuple[float, float, str]]],
    file_stem: str,
    audio_path: Path,
    tg_path: Path,
) -> Dict[str, str]:
    """Gujarati: 'vowels' tier with '=' breathy suffix and 'h'/'H' aspiration."""
    annotations: Dict[str, str] = {}
    v_intervals = tiers.get("vowels", []) or tiers.get("vowel", [])

    for start, end, text in v_intervals:
        if not text:
            continue
        text = text.strip()
        if text.endswith("="):
            state = "breathy"
            phone = text[:-1]
        elif any(h in text for h in ["h", "H"]):
            state = "uncertain"
            phone = text
        else:
            state = "non_breathy"
            phone = text

        interval_id = f"{start:.4f}|{end:.4f}|{phone}"
        annotations[interval_id] = state
    return annotations


def parse_ucla_zapotec(
    tiers: Dict[str, List[Tuple[float, float, str]]],
    file_stem: str,
    audio_path: Path,
    tg_path: Path,
) -> Dict[str, str]:
    """Zapotec: 'vowels'/'phonation' tier with B (breathy), C (creaky), M (modal) codes."""
    annotations: Dict[str, str] = {}
    target_tier = None
    for tname in ["phonation", "vowels", "vowel", "Vowels"]:
        if tname in tiers:
            target_tier = tiers[tname]
            break

    if not target_tier:
        for tname, ivs in tiers.items():
            if re.search(r"vowel|phonation", tname, re.IGNORECASE):
                target_tier = ivs
                break

    if not target_tier:
        return annotations

    for start, end, text in target_tier:
        raw_text = text.strip()
        if not raw_text:
            continue

        # Look for phonation code in interval label
        code = None
        m_code = re.search(r"[a-zA-Z]([BCMbc])\d*", raw_text)
        if m_code:
            code = m_code.group(1).upper()
        else:
            # Fallback to filename (e.g. speaker2M_aB14 -> 'B')
            m_fn = re.search(r"_([a-zA-Z]+?)([BCMbc])\d+", file_stem)
            if m_fn:
                code = m_fn.group(2).upper()
            else:
                m_fn2 = re.search(r"([BCMbc])\d+", file_stem)
                if m_fn2:
                    code = m_fn2.group(1).upper()

        if code == "B":
            state = "breathy"
        elif code == "C":
            state = "creaky"
        elif code == "M":
            state = "non_breathy"
        else:
            state = "uncertain"

        # Extract vowel phone
        m_ph = re.search(r"^([a-zA-Z]+?)[BCMbc]", raw_text)
        phone = m_ph.group(1) if m_ph else (raw_text[:1] if raw_text.isalpha() else "V")

        interval_id = f"{start:.4f}|{end:.4f}|{phone}"
        annotations[interval_id] = state
    return annotations


def parse_ucla_hmong(
    tiers: Dict[str, List[Tuple[float, float, str]]],
    file_stem: str,
    audio_path: Path,
    tg_path: Path,
) -> Dict[str, str]:
    """Hmong (White Hmong): 'vowel' tier, word tone in filename: -g (breathy), -m (creaky), modal."""
    annotations: Dict[str, str] = {}
    v_intervals = tiers.get("vowel", []) or tiers.get("vowels", [])
    if not v_intervals and "vuv" in tiers:
        # Fallback to voiced intervals if vowel tier missing
        v_intervals = [iv for iv in tiers["vuv"] if iv[2].strip() == "V"]

    # Extract tone from word in filename
    # Format: Speaker-Word_(token)_-Variety_Audio
    norm_stem = file_stem.replace(" ", "_")
    m = re.search(r"^\w+-([a-zA-Z]+)(?:_\d+)?(?:_-?([gw]))?(?:_Audio)?$", norm_stem, re.IGNORECASE)
    if m:
        word = m.group(1).lower()
    else:
        parts = norm_stem.split("-")
        word = parts[1].split("_")[0].lower() if len(parts) > 1 else ""

    if not word:
        return annotations

    last_c = word[-1]
    if last_c == "g":
        state = "breathy"
    elif last_c == "m":
        state = "creaky"
    elif last_c in ["b", "j", "v", "s"] or last_c in ["a", "e", "i", "o", "u", "w"]:
        state = "non_breathy"
    else:
        state = "uncertain"

    for start, end, text in v_intervals:
        text = text.strip()
        if not text:
            continue
        phone = text
        interval_id = f"{start:.4f}|{end:.4f}|{phone}"
        annotations[interval_id] = state
    return annotations


def parse_ucla_miao(
    tiers: Dict[str, List[Tuple[float, float, str]]],
    file_stem: str,
    audio_path: Path,
    tg_path: Path,
) -> Dict[str, str]:
    """Black Miao: 'seg' tier, 2-digit tones: 33 (breathy), 11 (creaky), 55 (tense/uncertain), modal."""
    annotations: Dict[str, str] = {}
    seg_intervals = tiers.get("seg", [])

    for start, end, text in seg_intervals:
        raw_text = text.strip()
        if not raw_text:
            continue

        # Check if this interval carries a 2-digit tone number
        m_tone = re.search(r"(\d{2})", raw_text)
        if not m_tone:
            # Consonant segment (e.g. p, t, l, s, hm) -> skip
            continue

        tone = m_tone.group(1)
        if tone == "33":
            state = "breathy"
        elif tone == "11":
            state = "creaky"
        elif tone in ("44", "22", "45", "51", "13", "21"):
            state = "non_breathy"
        elif tone == "55":
            # Tense / BmT -> mark as uncertain to prevent contamination
            state = "uncertain"
        else:
            state = "uncertain"

        # Clean phone representation: remove token index and digits
        phone = re.sub(r"_\d+$", "", raw_text)
        phone = re.sub(r"\d+", "", phone) or "V"

        interval_id = f"{start:.4f}|{end:.4f}|{phone}"
        annotations[interval_id] = state
    return annotations


DATASET_PARSERS = {
    "ucla_mazatec": parse_ucla_mazatec,
    "ucla_gujarati": parse_ucla_gujarati,
    "ucla_zapotec": parse_ucla_zapotec,
    "ucla_hmong": parse_ucla_hmong,
    "ucla_miao": parse_ucla_miao,
}


def extract_speaker_id(dataset: str, audio_path: Path, file_stem: str) -> str:
    """Extract speaker ID consistently based on dataset convention."""
    if dataset == "ucla_zapotec":
        # Folder is Speaker1..Speaker6 or filename is speaker3M_...
        if "speaker" in audio_path.parent.name.lower():
            return audio_path.parent.name
        m = re.search(r"(speaker\d+[MF]?)", file_stem, re.IGNORECASE)
        return m.group(1) if m else "spk_zapotec"

    elif dataset == "ucla_hmong":
        # Filename starts with speaker prefix: '1-caag_1_-g_Audio' -> 'spk_1'
        prefix = file_stem.split("-")[0]
        return f"spk_{prefix}" if prefix else "spk_hmong"

    elif dataset == "ucla_miao":
        # Filename: Liushilong_m1_cji45 or liuhua_01 or zhangxueqi_m2_01
        m = re.search(r"^([a-zA-Z]+(?:_m\d+)?)", file_stem)
        return m.group(1) if m else "spk_miao"

    elif dataset == "ucla_mazatec":
        # Filename suffix: hwa2_F1 -> F1
        parts = file_stem.split("_")
        return parts[-1] if len(parts) > 1 else audio_path.parent.name

    elif dataset == "ucla_gujarati":
        # Audio folder or default
        p_name = audio_path.parent.name
        return p_name if p_name and p_name != "Gujarati_Audio" else "spk_gujarati"

    return "spk"


def convert_dataset(
    dataset: str,
    audio_dir: Path,
    textgrid_dir: Path,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    dataset_key = dataset.strip().lower()
    if dataset_key not in DATASET_PARSERS:
        raise ValueError(
            f"Unsupported dataset: '{dataset}'. Must be one of: {list(DATASET_PARSERS.keys())}"
        )
    parser_fn = DATASET_PARSERS[dataset_key]

    if cache_dir is None:
        cache_dir = audio_dir / "_breathy_cache"

    print("=" * 70)
    print("Converting TextGrids to Standardized manual_breathy_annotations Cache")
    print(f"Dataset:      {dataset_key}")
    print(f"Audio Dir:    {audio_dir}")
    print(f"TextGrid Dir: {textgrid_dir}")
    print(f"Cache Dir:    {cache_dir}")
    print("=" * 70)

    # 1. Index audio files (normalize stem by replacing spaces with underscores for robust matching)
    audio_files: Dict[str, Path] = {}
    audio_orig_stems: Dict[str, str] = {}
    for ext in ("*.wav", "*.flac"):
        for p in audio_dir.rglob(ext):
            if "_breathy_cache" in p.parts or "_creaky_cache" in p.parts:
                continue
            norm_stem = p.stem.replace(" ", "_")
            audio_files[norm_stem] = p
            audio_orig_stems[norm_stem] = p.stem

    # 2. Index TextGrids (normalize stem by replacing spaces with underscores)
    tg_files: Dict[str, Path] = {}
    for p in textgrid_dir.rglob("*.TextGrid"):
        norm_stem = p.stem.replace(" ", "_")
        tg_files[norm_stem] = p

    print(f"Found {len(audio_files)} audio files, {len(tg_files)} TextGrid files.")

    # Match by normalized stem
    matched_stems = sorted(set(audio_files.keys()) & set(tg_files.keys()))
    print(f"Matched {len(matched_stems)} audio-TextGrid pairs.")

    if not matched_stems:
        print("[WARNING] No matching audio and TextGrid stems found! Aborting.")
        return {}

    cache_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = cache_dir / (ANNOTATION_FILENAME + ANNOTATION_SHARD_DIR_SUFFIX)
    shard_dir.mkdir(parents=True, exist_ok=True)

    state_counter = Counter()
    files_payload: Dict[str, Any] = {}
    now_unix = time.time()

    for norm_stem in matched_stems:
        audio_path = audio_files[norm_stem]
        tg_path = tg_files[norm_stem]
        file_key = audio_orig_stems[norm_stem]  # use original audio stem as file_key

        speaker_id = extract_speaker_id(dataset_key, audio_path, file_key)

        try:
            tiers = parse_textgrid_tiers(tg_path)
            annotations = parser_fn(tiers, file_stem=file_key, audio_path=audio_path, tg_path=tg_path)
        except Exception as e:
            print(f"[ERROR] Failed to parse {tg_path.name}: {e}")
            continue

        if not annotations:
            continue

        for st in annotations.values():
            state_counter[st] += 1

        file_entry = {
            "version": 1,
            "schema": "direct_breathy_annotation",
            "file_key": file_key,
            "speaker_id": speaker_id,
            "annotations": annotations,
            "reviewed_at": now_unix,
        }

        # Write shard file: <digest>.json
        digest = hashlib.sha256(str(file_key).encode("utf-8")).hexdigest()
        shard_path = shard_dir / f"{digest}.json"
        shard_path.write_text(json.dumps(file_entry, indent=2, ensure_ascii=False), encoding="utf-8")

        files_payload[file_key] = file_entry

    # Write root manifest
    manifest_payload = {
        "version": 1,
        "schema": "direct_breathy_annotation",
        "dataset_id": dataset_key,
        "storage_mode": ANNOTATION_STORAGE_MODE,
        "updated_unix_time": now_unix,
    }
    manifest_path = cache_dir / ANNOTATION_FILENAME
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "-" * 70)
    print(f"Conversion of [{dataset_key}] Completed Successfully!")
    print(f"Total processed files: {len(files_payload)}")
    print(f"Manifest written to:   {manifest_path}")
    print(f"Shards written to:     {shard_dir} ({len(list(shard_dir.glob('*.json')))} files)")
    print("Annotation label counts:")
    for st, cnt in sorted(state_counter.items()):
        print(f"  {st:15s}: {cnt:6d}")
    print("-" * 70)

    return {
        "dataset": dataset_key,
        "manifest_path": manifest_path,
        "shard_dir": shard_dir,
        "total_files": len(files_payload),
        "label_counts": dict(state_counter),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert Praat TextGrid annotations into standardized manual_breathy_annotations cache."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_PARSERS.keys()),
        help=f"Dataset identifier. Must be one of: {list(DATASET_PARSERS.keys())}",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Path to the directory containing audio (.wav/.flac) files.",
    )
    parser.add_argument(
        "--textgrid-dir",
        type=Path,
        required=True,
        help="Path to the directory containing Praat .TextGrid files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Target _breathy_cache directory. Defaults to <audio-dir>/_breathy_cache.",
    )

    args = parser.parse_args()

    if not args.audio_dir.exists():
        sys.exit(f"Error: audio-dir does not exist: {args.audio_dir}")
    if not args.textgrid_dir.exists():
        sys.exit(f"Error: textgrid-dir does not exist: {args.textgrid_dir}")

    convert_dataset(
        dataset=args.dataset,
        audio_dir=args.audio_dir,
        textgrid_dir=args.textgrid_dir,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
