#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end smoke test orchestrator for Gemini Japanese Breathy TTS synthesis.

Runs the full pipeline:
1. LLM dialogue script generation (Carlos Ishi paralinguistic contexts)
2. Gemini 3.8 Flash TTS synthesis (multi-voice 24kHz WAVs)
3. TIFA Japanese forced alignment (openvpi ST model)
4. TextGrid vowel interval extraction and label projection
5. Objective acoustic validation (CPPS, voicing, spectral tilt)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from typing import Any, Dict, List
import pandas as pd

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.gemini_tts_synthesis.script_generator import (
    generate_smoke_test_scripts,
    clean_transcript_for_tifa,
)
from scripts.gemini_tts_synthesis.tts_client import GeminiTTSClient
from scripts.gemini_tts_synthesis.tifa_align_wrapper import TIFAAlignmentWrapper
from scripts.gemini_tts_synthesis.textgrid_labeler import extract_labeled_segments
from scripts.gemini_tts_synthesis.acoustic_validator import (
    validate_sample_acoustics,
    generate_validation_summary,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("smoke_test_orchestrator")


def run_pipeline(
    output_dir: Path,
    config_path: Path,
    skip_llm: bool = False,
    skip_tts: bool = False,
    skip_tifa: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    tg_dir = output_dir / "textgrids"
    audio_dir.mkdir(parents=True, exist_ok=True)
    tg_dir.mkdir(parents=True, exist_ok=True)

    scripts_file = output_dir / "smoke_scripts.json"

    # ── Stage 1: Script Generation ──
    if not skip_llm and not scripts_file.exists():
        logger.info("=== STAGE 1: Generating Dialogue Scripts via LLM ===")
        scripts = generate_smoke_test_scripts(config_path, scripts_file)
    else:
        logger.info(f"Loading existing scripts from {scripts_file}...")
        with open(scripts_file, "r", encoding="utf-8") as f:
            scripts = json.load(f)

    # ── Stage 2: Gemini TTS Synthesis ──
    tts_client = GeminiTTSClient()
    logger.info(f"=== STAGE 2: Synthesizing {len(scripts)} Audio Samples via Gemini TTS ===")

    synthesized_records = []
    for s_idx, item in enumerate(scripts, 1):
        sample_id = item["sample_id"]
        wav_path = audio_dir / f"{sample_id}.wav"
        txt_path = audio_dir / f"{sample_id}.txt"

        # Always ensure clean transcript .txt sidecar exists for TIFA
        txt_path.write_text(item["clean_full_text"], encoding="utf-8")

        if wav_path.exists() and wav_path.stat().st_size > 4000:
            logger.info(f"[{s_idx}/{len(scripts)}] Audio already exists: {wav_path.name}")
            synthesized_records.append({"sample_id": sample_id, "wav_path": wav_path, "meta": item})
            continue

        if skip_tts:
            logger.info(f"[{s_idx}/{len(scripts)}] Skipping TTS for {sample_id}")
            continue

        try:
            res = tts_client.synthesize(
                parts=item["tts_parts"],
                voice=item["voice"],
                output_wav_path=wav_path,
            )
            synthesized_records.append({"sample_id": sample_id, "wav_path": wav_path, "meta": item})
        except Exception as e:
            logger.error(f"Failed to synthesize {sample_id}: {e}")

    # ── Stage 3: TIFA Forced Alignment ──
    if not skip_tifa:
        logger.info("=== STAGE 3: Running TIFA Japanese Forced Alignment ===")
        aligner = TIFAAlignmentWrapper()
        tg_map = aligner.align_directory(input_dir=audio_dir, output_dir=tg_dir)
    else:
        logger.info("Skipping TIFA alignment execution, checking existing TextGrids...")
        tg_map = {f.stem: f for f in tg_dir.glob("*.TextGrid")}

    # ── Stage 4: Labeling and Segment Projection ──
    logger.info("=== STAGE 4: Projecting Labels to TextGrid Vowel Intervals ===")
    all_intervals: List[Dict[str, Any]] = []
    sample_dict = {s["sample_id"]: s for s in scripts}

    for sample_id, tg_path in tg_map.items():
        if sample_id not in sample_dict:
            continue
        meta = sample_dict[sample_id]
        intervals = extract_labeled_segments(tg_path, meta, dataset_id="gemini_tts_smoke")
        all_intervals.extend(intervals)

    csv_out = output_dir / "smoke_intervals.csv"
    if all_intervals:
        df = pd.DataFrame(all_intervals)
        df.to_csv(csv_out, index=False, encoding="utf-8")
        logger.info(f"Saved {len(all_intervals)} labeled intervals to {csv_out}")

    # ── Stage 5: Objective Acoustic Validation ──
    logger.info("=== STAGE 5: Acoustic Verification (CPPS, Voicing, Spectral Tilt) ===")
    acoustic_results = []
    for sample_id, tg_path in tg_map.items():
        wav_path = audio_dir / f"{sample_id}.wav"
        if not wav_path.exists():
            continue
        item_intervals = [r for r in all_intervals if r["file_id"] == sample_id]
        if not item_intervals:
            continue
        res = validate_sample_acoustics(wav_path, item_intervals)
        acoustic_results.append(res)

    report_md = generate_validation_summary(acoustic_results)
    report_file = output_dir / "acoustic_validation_report.md"
    report_file.write_text(report_md, encoding="utf-8")
    logger.info(f"Acoustic report saved to {report_file}")

    print("\n" + "=" * 60)
    print(f"SMOKE TEST PIPELINE COMPLETE!")
    print(f"- Scripts:      {scripts_file}")
    print(f"- Audio Dir:    {audio_dir}")
    print(f"- TextGrids:    {tg_dir}")
    print(f"- Intervals:    {csv_out} ({len(all_intervals)} vowels)")
    print(f"- Report:       {report_file}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("exp_results/tts_smoke_test"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "configs" / "ishi_contexts.yaml",
    )
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-tifa", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        output_dir=args.output_dir,
        config_path=args.config,
        skip_llm=args.skip_llm,
        skip_tts=args.skip_tts,
        skip_tifa=args.skip_tifa,
    )


if __name__ == "__main__":
    main()
