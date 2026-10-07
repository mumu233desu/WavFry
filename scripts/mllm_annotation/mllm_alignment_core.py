#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core Dual-Constraint Alignment and Acoustic Snapping Module.

Bridges Multimodal LLM (Gemini) coarse acoustic detections and Kanji/Kana text
with fine-grained Praat TextGrids:
1. Temporal Gating: Limits candidate search to [T_start - 0.5s, T_end + 0.5s].
2. Kanji/Kana Sequence Alignment: Matches Gemini's target_text against Tier 1 (texts).
3. Acoustic Snapping: Snaps coarse LLM window to exact Tier 1 acoustic word boundaries.
4. Phoneme Projection: Tags candidate vowels from Tier 3 (phones) inside matched words.
5. Produces standardized interval_ids ("start|end|phone") for direct cache shard injection.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

VOWEL_SET: Set[str] = {
    "a", "i", "u", "e", "o", "um",
    "ɯ", "aː", "iː", "uː", "eː", "oː", "ɯː", "i̥", "ɯ̥",
}


# ── TextGrid Parsing and Serializing ──────────────────────────────────────────

def parse_textgrid(tg_path: Path) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Parse TextGrid into (total_xmin, total_xmax, list of tiers).
    
    Each tier is a dict: {'name': str, 'intervals': List[Tuple[float, float, str]]}
    """
    text = tg_path.read_text(encoding="utf-8", errors="ignore")
    header_m = re.match(r"(.*?)(item\s*\[\s*1\s*\]:.*)", text, re.DOTALL)
    if not header_m:
        raise ValueError(f"Invalid TextGrid format: {tg_path}")
    header = header_m.group(1)

    total_xmin = float(re.search(r"xmin\s*=\s*([0-9.]+)", header).group(1))
    total_xmax = float(re.search(r"xmax\s*=\s*([0-9.]+)", header).group(1))

    items = re.split(r"item\s*\[\s*\d+\s*\]:", text)[1:]
    tiers: List[Dict[str, Any]] = []
    for it in items:
        name_m = re.search(r'name\s*=\s*"([^"]*)"', it)
        name = name_m.group(1) if name_m else "unnamed"
        iv_matches = re.findall(
            r'xmin\s*=\s*([0-9.]+)\s*xmax\s*=\s*([0-9.]+)\s*text\s*=\s*"([^"]*)"',
            it,
        )
        intervals = [(float(x1), float(x2), txt) for x1, x2, txt in iv_matches]
        tiers.append({"name": name, "intervals": intervals})

    return total_xmin, total_xmax, tiers


def write_textgrid(
    out_path: Path,
    total_xmin: float,
    total_xmax: float,
    tiers: List[Dict[str, Any]],
) -> None:
    """Serialize tiers to Praat TextGrid format."""
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        f"xmin = {total_xmin:.6f}",
        f"xmax = {total_xmax:.6f}",
        "tiers? <exists>",
        f"size = {len(tiers)}",
        "item []:",
    ]
    for idx, tier in enumerate(tiers, 1):
        lines.append(f"    item [{idx}]:")
        lines.append('        class = "IntervalTier"')
        lines.append(f'        name = "{tier["name"]}"')
        lines.append(f"        xmin = {total_xmin:.6f}")
        lines.append(f"        xmax = {total_xmax:.6f}")
        lines.append(f'        intervals: size = {len(tier["intervals"])}')
        for jdx, (x1, x2, txt) in enumerate(tier["intervals"], 1):
            lines.append(f"        intervals [{jdx}]:")
            lines.append(f"            xmin = {x1:.6f}")
            lines.append(f"            xmax = {x2:.6f}")
            safe_txt = str(txt).replace('"', '""')
            lines.append(f'            text = "{safe_txt}"')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_continuous_tier(
    total_xmin: float,
    total_xmax: float,
    active_intervals: List[Tuple[float, float, str]],
) -> List[Tuple[float, float, str]]:
    """Ensure intervals strictly partition [total_xmin, total_xmax] continuously."""
    sorted_active = sorted(active_intervals, key=lambda x: x[0])
    continuous: List[Tuple[float, float, str]] = []
    curr_t = total_xmin

    for x1, x2, lbl in sorted_active:
        x1 = max(total_xmin, min(total_xmax, x1))
        x2 = max(total_xmin, min(total_xmax, x2))
        if x2 <= x1:
            continue
        if x1 > curr_t + 1e-4:
            continuous.append((curr_t, x1, ""))
        continuous.append((x1, x2, lbl))
        curr_t = x2

    if curr_t < total_xmax - 1e-4:
        continuous.append((curr_t, total_xmax, ""))
    return continuous


# ── Text Normalization and Alignment ──────────────────────────────────────────

def normalize_japanese_text(text: str) -> str:
    """Normalize text by stripping punctuation, wave dashes (U+301C, U+FF5E), brackets, and prolonged marks."""
    if not text:
        return ""
    return re.sub(r'[、。！？\sー~〜～・「」『』【】〔〕\'"・\(\)（）\.\,\!\?]', "", str(text))


def match_contiguous_words(
    candidate_words: List[Tuple[float, float, str]],
    target_text: str,
) -> Optional[List[Tuple[float, float, str]]]:
    """Find the contiguous slice of Tier 1 words matching target_text.
    
    Uses multi-pass alignment:
    1. Exact contiguous substring match (100% exact character sequence).
    2. Tightest substring containment (finds the shortest contiguous word slice containing target_text).
    3. Longest substring containment (fallback when target is partial).
    """
    norm_target = normalize_japanese_text(target_text)
    if not norm_target:
        return None

    n = len(candidate_words)
    # Pass 1: Exact concatenated sequence match
    for i in range(n):
        concatenated = ""
        for j in range(i, n):
            concatenated += normalize_japanese_text(candidate_words[j][2])
            if concatenated == norm_target:
                return candidate_words[i : j + 1]

    # Pass 2: Tightest substring containment (avoids greedy left-window expansion)
    best_match = None
    min_span_len = float("inf")
    for i in range(n):
        concatenated = ""
        for j in range(i, n):
            concatenated += normalize_japanese_text(candidate_words[j][2])
            if norm_target in concatenated:
                span_len = len(concatenated)
                if span_len < min_span_len:
                    min_span_len = span_len
                    best_match = candidate_words[i : j + 1]
                break

    if best_match is not None:
        return best_match

    # Pass 3: Longest substring match (partial target overlap)
    best_len = 0
    for i in range(n):
        concatenated = ""
        for j in range(i, n):
            concatenated += normalize_japanese_text(candidate_words[j][2])
            if concatenated in norm_target and len(concatenated) > best_len:
                best_len = len(concatenated)
                best_match = candidate_words[i : j + 1]

    return best_match


def locate_tiers(
    tiers: List[Dict[str, Any]],
) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
    """Dynamically locate texts and phones tiers by pattern matching with positional fallbacks."""
    tier_texts: List[Tuple[float, float, str]] = []
    tier_phones: List[Tuple[float, float, str]] = []

    for tier in tiers:
        name = tier.get("name", "").strip().lower()
        if not tier_texts and any(k in name for k in ("text", "word", "phrase", "transcription")):
            tier_texts = tier.get("intervals", [])
        elif not tier_phones and any(k in name for k in ("phone", "phoneme")):
            tier_phones = tier.get("intervals", [])

    # Positional fallback if tier naming is non-standard
    if not tier_texts and len(tiers) >= 1:
        tier_texts = tiers[0].get("intervals", [])
    if not tier_phones and len(tiers) >= 2:
        tier_phones = tiers[-1].get("intervals", [])

    return tier_texts, tier_phones


# ── Projection and Snapping Core ──────────────────────────────────────────────

def project_mllm_prediction_to_vowels(
    tg_path: Path,
    prediction: Dict[str, Any],
    vowel_coverage_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Project MLLM prediction onto TextGrid vowels via dual-constraint alignment.
    
    Returns:
        {
            "has_breathy": bool,
            "detected_phrases": List[Dict[str, Any]],  # Macro phrases snapped
            "candidate_vowels": List[Dict[str, Any]],  # Micro vowels tagged
            "modal_only": bool,                        # True if completely modal
        }
    """
    total_xmin, total_xmax, tiers = parse_textgrid(tg_path)
    tier_texts, tier_phones = locate_tiers(tiers)

    segments = prediction.get("segments", [])
    if not segments or not (prediction.get("has_breathy") or prediction.get("has_whispery")):
        return {
            "has_breathy": False,
            "has_whispery": False,
            "detected_phrases": [],
            "candidate_vowels": [],
            "modal_only": True,
        }

    detected_phrases: List[Dict[str, Any]] = []
    candidate_vowels: List[Dict[str, Any]] = []
    seen_vowel_ids: Set[str] = set()

    for seg in segments:
        p_type = seg.get("phonation_type", "breathy")
        if p_type not in ("breathy", "whispery"):
            continue

        s_t = float(seg.get("start_sec", total_xmin))
        e_t = float(seg.get("end_sec", total_xmax))
        target_text = seg.get("target_text", "")

        matched_words = None
        if target_text:
            # Temporal gate: [s_t - 0.5s, e_t + 0.5s]
            gate_start = max(total_xmin, s_t - 0.5)
            gate_end = min(total_xmax, e_t + 0.5)
            candidates = [
                (x1, x2, txt)
                for x1, x2, txt in tier_texts
                if txt.strip() and not (x2 < gate_start or x1 > gate_end)
            ]
            matched_words = match_contiguous_words(candidates, target_text)

        if matched_words:
            # Acoustic boundary snapped from Tier 1 orthographic words!
            snap_start = matched_words[0][0]
            snap_end = matched_words[-1][1]
            alignment_mode = "text_matched"
        else:
            # Fallback to temporal boundary if target_text unmatched or empty
            snap_start = max(total_xmin, s_t)
            snap_end = min(total_xmax, e_t)
            alignment_mode = "temporal_fallback"

        phrase_info = {
            "start": round(snap_start, 4),
            "end": round(snap_end, 4),
            "phonation_type": p_type,
            "target_text": target_text,
            "alignment_mode": alignment_mode,
            "acoustic_reason": seg.get("acoustic_reason", ""),
        }
        detected_phrases.append(phrase_info)

        # Micro vowel projection on Tier 3 (phones)
        for x1, x2, ph in tier_phones:
            if ph.lower() in VOWEL_SET:
                # Submersion / Coverage calculation
                inter = max(0.0, min(x2, snap_end) - max(x1, snap_start))
                dur = x2 - x1
                if dur > 0 and (inter / dur) >= vowel_coverage_threshold:
                    interval_id = f"{x1:.4f}|{x2:.4f}|{ph}"
                    if interval_id not in seen_vowel_ids:
                        seen_vowel_ids.add(interval_id)
                        candidate_vowels.append({
                            "interval_id": interval_id,
                            "start": round(x1, 4),
                            "end": round(x2, 4),
                            "phone": ph,
                            "state": "mllm_breathy",
                            "phonation_type": p_type,
                        })

    return {
        "has_breathy": prediction.get("has_breathy", False),
        "has_whispery": prediction.get("has_whispery", False),
        "detected_phrases": detected_phrases,
        "candidate_vowels": candidate_vowels,
        "modal_only": len(candidate_vowels) == 0,
    }


def export_enriched_textgrid(
    tifa_tg_path: Path,
    out_tg_path: Path,
    detected_phrases: List[Dict[str, Any]],
    candidate_vowels: List[Dict[str, Any]],
) -> Path:
    """Export Praat TextGrid with Tier 4 (macro phrase) and Tier 5 (micro vowels)."""
    total_xmin, total_xmax, tiers = parse_textgrid(tifa_tg_path)

    macro_intervals = [
        (p["start"], p["end"], f"{p['phonation_type']} [{p.get('target_text', '')}]")
        for p in detected_phrases
    ]
    micro_intervals = [
        (v["start"], v["end"], f"{v['phone']}:{v['state']}")
        for v in candidate_vowels
    ]

    tier_macro = {
        "name": "mllm_breathy_phrase",
        "intervals": build_continuous_tier(total_xmin, total_xmax, macro_intervals),
    }
    tier_micro = {
        "name": "mllm_breathy_vowels",
        "intervals": build_continuous_tier(total_xmin, total_xmax, micro_intervals),
    }

    all_tiers = list(tiers) + [tier_macro, tier_micro]
    write_textgrid(out_tg_path, total_xmin, total_xmax, all_tiers)
    return out_tg_path
