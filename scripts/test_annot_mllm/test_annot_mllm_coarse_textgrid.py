#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dual-Constraint Projection for MLLM Coarse Annotation onto TIFA TextGrids.

Bridges Multimodal LLM (Gemini) coarse acoustic detections and Kanji/Kana text
with fine-grained TIFA Praat TextGrids:
1. Temporal Gating: Limits search to [T_start - 0.5s, T_end + 0.5s].
2. Kanji/Kana Sequence Alignment: Matches Gemini's quoted phrase against Tier 1 (texts).
3. Acoustic Snapping: Snaps coarse LLM window to exact Tier 1 word boundaries.
4. Phoneme Projection: Tags candidate vowels from Tier 3 (phones) inside matched words.
5. Exports enriched Praat TextGrid with:
   - Tier 4: 'mllm_breathy_phrase' (macro phrase & text)
   - Tier 5: 'mllm_breathy_vowels' (micro candidate vowels)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_textgrid(tg_path: Path) -> Tuple[float, float, List[Dict[str, Any]]]:
    """Parse TextGrid into (total_xmin, total_xmax, list of tiers)."""
    text = tg_path.read_text(encoding="utf-8", errors="ignore")
    header_m = re.match(r"(.*?)(item\s*\[\s*1\s*\]:.*)", text, re.DOTALL)
    if not header_m:
        raise ValueError(f"Invalid TextGrid format: {tg_path}")
    header = header_m.group(1)

    total_xmin = float(re.search(r"xmin\s*=\s*([0-9.]+)", header).group(1))
    total_xmax = float(re.search(r"xmax\s*=\s*([0-9.]+)", header).group(1))

    items = re.split(r"item\s*\[\s*\d+\s*\]:", text)[1:]
    tiers = []
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
            safe_txt = txt.replace('"', '""')
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


def normalize_japanese_text(text: str) -> str:
    """Normalize text by stripping punctuation, quotes, and prolonged marks."""
    return re.sub(r'[、。！？\sー~・「」\'"・\(\)（）]', "", text)


def extract_quoted_text_from_reason(reason: str) -> Optional[str]:
    """Extract quoted Japanese words from acoustic reasoning text."""
    matches = re.findall(r"[「『\'\"]([^「『\'\"」』]+)[」』\'\"]", reason)
    for m in matches:
        # Check if it contains Japanese characters
        if re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]", m):
            return m
    return None


def match_contiguous_words(
    candidate_words: List[Tuple[float, float, str]],
    target_text: str,
) -> Optional[List[Tuple[float, float, str]]]:
    """Find the contiguous slice of Tier 1 words matching target_text."""
    norm_target = normalize_japanese_text(target_text)
    if not norm_target:
        return None

    n = len(candidate_words)
    # 1. Exact match search
    for i in range(n):
        concatenated = ""
        for j in range(i, n):
            concatenated += normalize_japanese_text(candidate_words[j][2])
            if concatenated == norm_target:
                return candidate_words[i : j + 1]

    # 2. Substring containment search (longest overlap)
    best_match = None
    best_len = 0
    for i in range(n):
        concatenated = ""
        for j in range(i, n):
            concatenated += normalize_japanese_text(candidate_words[j][2])
            if concatenated in norm_target and len(concatenated) > best_len:
                best_len = len(concatenated)
                best_match = candidate_words[i : j + 1]
            elif norm_target in concatenated:
                return candidate_words[i : j + 1]

    return best_match


def project_mllm_to_textgrid(
    tifa_tg_path: Path,
    gemini_prediction: Dict[str, Any],
    out_tg_path: Path,
    vowel_coverage_threshold: float = 0.8,
) -> Dict[str, Any]:
    """Project Gemini predictions onto TIFA TextGrid via dual-constraint."""
    total_xmin, total_xmax, tiers = parse_textgrid(tifa_tg_path)

    # Locate tiers: texts (Tier 1), words (Tier 2), phones (Tier 3)
    tier_texts = tiers[0]["intervals"]
    tier_phones = tiers[2]["intervals"]

    segments = gemini_prediction.get("segments", [])
    macro_intervals: List[Tuple[float, float, str]] = []
    micro_vowel_intervals: List[Tuple[float, float, str]] = []

    vowel_set = {"a", "i", "u", "e", "o"}

    for seg in segments:
        p_type = seg.get("phonation_type", "breathy")
        if p_type not in ("breathy", "whispery"):
            continue

        s_t = float(seg.get("start_sec", 0.0))
        e_t = float(seg.get("end_sec", total_xmax))
        reason = seg.get("acoustic_reason", "")

        # Try to extract quoted text or use entire segment text
        target_text = seg.get("target_text") or extract_quoted_text_from_reason(reason)

        matched_words = None
        if target_text:
            # Temporal gating: [s_t - 0.5s, e_t + 0.5s]
            gate_start = max(total_xmin, s_t - 0.5)
            gate_end = min(total_xmax, e_t + 0.5)
            candidates = [
                (x1, x2, txt)
                for x1, x2, txt in tier_texts
                if txt.strip() and not (x2 < gate_start or x1 > gate_end)
            ]
            matched_words = match_contiguous_words(candidates, target_text)

        if matched_words:
            # Acoustic boundary snapped from Tier 1 words!
            snap_start = matched_words[0][0]
            snap_end = matched_words[-1][1]
            phrase_label = f"{p_type} [{target_text}]"
        else:
            # Fallback to temporal boundary if no text match
            snap_start = s_t
            snap_end = e_t
            phrase_label = f"{p_type} (time_gated)"

        macro_intervals.append((snap_start, snap_end, phrase_label))

        # Project to vowels in Tier 3
        for x1, x2, ph in tier_phones:
            if ph.lower() in vowel_set:
                # Calculate coverage inside [snap_start, snap_end]
                inter = max(0.0, min(x2, snap_end) - max(x1, snap_start))
                dur = x2 - x1
                if dur > 0 and (inter / dur) >= vowel_coverage_threshold:
                    micro_vowel_intervals.append((x1, x2, f"{ph}:{p_type}"))

    # Build continuous tiers
    tier_macro = {
        "name": "mllm_breathy_phrase",
        "intervals": build_continuous_tier(total_xmin, total_xmax, macro_intervals),
    }
    tier_micro = {
        "name": "mllm_breathy_vowels",
        "intervals": build_continuous_tier(total_xmin, total_xmax, micro_vowel_intervals),
    }

    all_tiers = list(tiers) + [tier_macro, tier_micro]
    write_textgrid(out_tg_path, total_xmin, total_xmax, all_tiers)

    return {
        "out_textgrid": str(out_tg_path),
        "macro_count": len(macro_intervals),
        "vowel_count": len(micro_vowel_intervals),
        "macro_intervals": macro_intervals,
    }


if __name__ == "__main__":
    # Self-test on F_00_JP_JA_JA-JP_C0105_S0_19
    sample_key = "F_00_JP_JA_JA-JP_C0105_S0_19"
    tg_p = Path(rf"E:\Corpus\Breathy\YSJ\textgrid_tifa\{sample_key}.TextGrid")
    cache_p = Path(rf"manifests\test_annot_mllm\ysj_cache\{sample_key}.json")
    out_tg_p = Path(rf"manifests\test_annot_mllm\ysj_listen_cases\{sample_key}_dual_constraint.TextGrid")

    if tg_p.exists() and cache_p.exists():
        with open(cache_p, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        res = project_mllm_to_textgrid(tg_p, cdata.get("prediction", {}), out_tg_p)
        print("Test execution complete:")
        print(json.dumps(res, indent=2, ensure_ascii=False))
