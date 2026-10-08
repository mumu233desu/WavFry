#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TextGrid labeler and projector for synthesized speech.

Parses TIFA TextGrids, projects structured modal/breathy script parts onto
phoneme and word tiers, and produces ground-truth interval segments.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("textgrid_labeler")

JAPANESE_VOWELS = {
    "a", "i", "u", "e", "o",
    "a:", "i:", "u:", "e:", "o:",
    "A", "I", "U", "E", "O",  # Devoiced vowels if present
}


def parse_praat_textgrid_intervals(tg_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Lightweight pure-python parser for Praat TextGrid interval tiers."""
    with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f]

    tiers: Dict[str, List[Dict[str, Any]]] = {}
    current_tier_name = None
    in_interval = False
    cur_xmin = 0.0
    cur_xmax = 0.0
    cur_text = ""

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('name = "'):
            current_tier_name = line.split('"')[1]
            tiers[current_tier_name] = []
        elif line.startswith("intervals [") or line.startswith("item ["):
            pass
        elif line.startswith("xmin ="):
            cur_xmin = float(line.split("=")[1].strip())
        elif line.startswith("xmax ="):
            cur_xmax = float(line.split("=")[1].strip())
        elif line.startswith('text = "'):
            # Text might span quotes
            cur_text = line.split('text = "')[1].rstrip('"')
            if current_tier_name is not None and cur_text != "":
                tiers[current_tier_name].append({
                    "xmin": cur_xmin,
                    "xmax": cur_xmax,
                    "text": cur_text,
                    "duration": round(cur_xmax - cur_xmin, 4),
                })
        i += 1

    return tiers


def extract_labeled_segments(
    textgrid_path: Path,
    script_meta: Dict[str, Any],
    dataset_id: str = "gemini_tts_smoke",
) -> List[Dict[str, Any]]:
    """Project script modal/breathy boundaries onto TextGrid vowels.

    Args:
        textgrid_path: Path to TIFA TextGrid.
        script_meta: Script metadata dict containing modal_text, breathy_text_clean, voice, etc.
        dataset_id: Dataset identifier.

    Returns:
        List of interval records formatted for WavFry manifest.
    """
    tiers = parse_praat_textgrid_intervals(textgrid_path)
    
    # Try phones/words tier variants
    phones_tier = tiers.get("phones") or tiers.get("phone") or tiers.get("tokens") or []
    words_tier = tiers.get("words") or tiers.get("word") or []

    if not phones_tier:
        logger.warning(f"No phones tier found in {textgrid_path.name}")
        return []

    sample_id = script_meta["sample_id"]
    voice = script_meta.get("voice", "unknown")
    modal_text = script_meta.get("modal_text", "")
    breathy_text = script_meta.get("breathy_text_clean", "")

    # Estimate character-based boundary split ratio
    total_len = max(1, len(modal_text) + len(breathy_text))
    modal_ratio = len(modal_text) / total_len

    # Identify audio midpoint dividing modal and breathy portions
    total_speech_start = phones_tier[0]["xmin"]
    total_speech_end = phones_tier[-1]["xmax"]
    total_speech_duration = max(0.1, total_speech_end - total_speech_start)

    # If words tier exists, use words to find the transition point
    transition_time_s = total_speech_start + (total_speech_duration * modal_ratio)
    if words_tier:
        # Match words sequentially
        accum_chars = 0
        for w in words_tier:
            accum_chars += len(w["text"])
            if accum_chars >= len(modal_text):
                transition_time_s = w["xmax"]
                break

    records: List[Dict[str, Any]] = []
    for p_idx, p in enumerate(phones_tier):
        phone_label = p["text"].strip().lower()
        # Clean special symbols
        norm_phone = re.sub(r"[0-9_#^]", "", phone_label)

        # Check if vowel
        is_vowel = norm_phone in JAPANESE_VOWELS or norm_phone.startswith(("a", "i", "u", "e", "o"))
        if not is_vowel:
            continue

        start_s = round(p["xmin"], 4)
        end_s = round(p["xmax"], 4)
        duration_s = round(end_s - start_s, 4)

        if duration_s < 0.03:  # Skip tiny transient intervals < 30ms
            continue

        # Classify breathy based on transition point
        mid_point = (start_s + end_s) / 2.0
        is_breathy = mid_point >= transition_time_s

        seg_record = {
            "dataset_id": dataset_id,
            "speaker_id": voice,
            "file_id": sample_id,
            "segment_id": f"{sample_id}_{start_s:.3f}_{end_s:.3f}",
            "start_s": start_s,
            "end_s": end_s,
            "duration_s": duration_s,
            "phone": norm_phone,
            "creaky": float("nan"),  # Not labeled for creaky
            "breathy": 1.0 if is_breathy else 0.0,
            "scope": "interval",
            "context_id": script_meta.get("context_id", ""),
            "section": "breathy" if is_breathy else "modal",
        }
        records.append(seg_record)

    logger.info(
        f"Extracted {len(records)} vowel intervals for {sample_id}: "
        f"{sum(1 for r in records if r['breathy'] == 1.0)} breathy, "
        f"{sum(1 for r in records if r['breathy'] == 0.0)} modal."
    )
    return records
