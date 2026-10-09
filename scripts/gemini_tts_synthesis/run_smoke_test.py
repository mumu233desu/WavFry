#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end orchestrator for BreathyTTS Japanese synthesis, alignment & slicing.

Executes the complete production pipeline:
1. Script Pool Management (17 ATR sub-categories, screenwriter persona, local LLM with Fail-Fast)
2. Speaker Registry & Monotonic Indexing (30 mixed voices mapped to U0001~U0030)
3. Gemini 3.8 Flash TTS multi-voice synthesis (lossless 24kHz archived in raw_dialogues/)
4. TIFA Japanese forced alignment (openvpi ST model)
5. Three-Stage Intelligent Pop-Free Slicing (TIFA macro gating + RMS minimum + zero-crossing snap)
6. FLAC Level 8 Export & TextGrid linear remapping (Part 1 Modal vs Part 2 Breathy)
7. Sonorant + Vowel direct 'breathy = 1.0' labeling and shard JSON export (Modal unreviewed)
8. Acoustic CPPS & Voicing validation with full audit trailing in synthesis_metadata.csv
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import soundfile as sf

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.gemini_tts_synthesis.script_generator import (
    generate_script_pool,
    clean_transcript_for_tifa,
)
from scripts.gemini_tts_synthesis.metadata_manager import (
    MetadataManager,
    VOICE_REGISTRY,
    VOICE_BY_NAME,
)
from scripts.gemini_tts_synthesis.tts_client import GeminiTTSClient
from scripts.gemini_tts_synthesis.tifa_align_wrapper import TIFAAlignmentWrapper
from scripts.gemini_tts_synthesis.audio_dialogue_slicer import slice_dialogue_pair
from scripts.gemini_tts_synthesis.textgrid_labeler import (
    extract_breathy_slice_intervals,
    write_breathy_shard,
)
from scripts.gemini_tts_synthesis.acoustic_validator import (
    validate_sample_acoustics,
    generate_validation_summary,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("breathy_tts_orchestrator")

DEFAULT_DATASET_ROOT = Path(r"E:\Corpus\Breathy\BreathyTTS")


def run_pipeline(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    config_path: Optional[Path] = None,
    script_cache_dir: Optional[Path] = None,
    num_dialogues: int = 1,
    voice_filter: Optional[str] = None,
    skip_llm: bool = False,
    skip_tts: bool = False,
    skip_tifa: bool = False,
    scripts_only: bool = False,
) -> None:
    dataset_root = dataset_root.resolve()
    config_path = config_path or (Path(__file__).parent / "configs" / "ishi_contexts.yaml")
    script_cache_dir = script_cache_dir or (dataset_root / "script_pool_cache")

    logger.info(f"=== Initializing BreathyTTS Pipeline at {dataset_root} ===")
    meta_mgr = MetadataManager(dataset_root)

    # ── Stage 1: Dialogue Script Pool ──
    logger.info(f"=== STAGE 1: Managing Script Pool ({script_cache_dir}) ===")
    if not skip_llm:
        scripts = generate_script_pool(
            config_path=config_path,
            cache_dir=script_cache_dir,
            scenarios_per_context=3,
        )
    else:
        logger.info(f"Skipping LLM calls; loading existing cached scripts from {script_cache_dir}...")
        scripts = []
        for sf_file in sorted(script_cache_dir.glob("*.json")):
            try:
                scripts.append(json.loads(sf_file.read_text(encoding="utf-8")))
            except Exception:
                pass

    if not scripts:
        logger.error(f"No scripts available in {script_cache_dir}! Please ensure scripts are generated.")
        return

    if scripts_only:
        logger.info(f"=== Scripts-Only Mode: {len(scripts)} scripts ready in {script_cache_dir}. Exiting successfully. ===")
        print("\n" + "=" * 60)
        print("SCRIPTS-ONLY GENERATION COMPLETED!")
        print(f"- Total Scripts:   {len(scripts)}")
        print(f"- Script Pool Dir: {script_cache_dir}")
        print("=" * 60 + "\n")
        return

    # Filter voices if requested
    if voice_filter:
        voices_to_run = [v for v in VOICE_REGISTRY if v["voice"].lower() == voice_filter.lower()]
        if not voices_to_run:
            raise ValueError(f"Unknown voice filter '{voice_filter}'.")
    else:
        voices_to_run = VOICE_REGISTRY

    logger.info(f"Available scripts: {len(scripts)} | Voices to schedule: {len(voices_to_run)}")

    # ── Stage 2: Synthesis, Alignment, Slicing & Annotation ──
    tts_client = GeminiTTSClient()
    aligner = TIFAAlignmentWrapper()

    total_to_process = min(num_dialogues, len(scripts))
    logger.info(f"Starting batch of {total_to_process} dialogues...")

    processed_count = 0
    all_manifest_records: List[Dict[str, Any]] = []
    acoustic_results: List[Dict[str, Any]] = []

    for d_idx in range(total_to_process):
        script_item = scripts[d_idx % len(scripts)]
        voice_info = voices_to_run[d_idx % len(voices_to_run)]
        voice_name = voice_info["voice"]
        speaker_uid = voice_info["uid"]

        dialogue_id = f"dialogue_{d_idx+1:04d}_{voice_name.lower()}_{script_item['sub_id']}"
        logger.info(f"\n--- Processing [{d_idx+1}/{total_to_process}] {dialogue_id} (Speaker: {speaker_uid} / {voice_name}) ---")

        # 1. Allocate next monotonic utterance index pair
        modal_key, breathy_key, spk_meta = meta_mgr.allocate_next_utterance_pair(voice_name)
        logger.info(f"Allocated keys: Modal={modal_key} | Breathy={breathy_key}")

        # 2. Raw dialogue paths
        raw_wav_path = meta_mgr.raw_dialogues_dir / f"{dialogue_id}.wav"
        raw_txt_path = meta_mgr.raw_dialogues_dir / f"{dialogue_id}.txt"
        raw_tg_path = meta_mgr.raw_dialogues_dir / f"{dialogue_id}.TextGrid"
        raw_json_path = meta_mgr.raw_dialogues_dir / f"{dialogue_id}.json"

        # Prepare clean transcript sidecar for TIFA
        raw_txt_path.write_text(script_item["clean_full_text"], encoding="utf-8")

        # 3. TTS Synthesis
        if not (raw_wav_path.exists() and raw_wav_path.stat().st_size > 4000) and not skip_tts:
            tts_parts = [
                {
                    "text": script_item["modal_text"],
                    "speech_metadata": {
                        "speaker": voice_name,
                        "style": script_item["modal_style"],
                    },
                },
                {
                    "text": script_item["breathy_text_with_tag"],
                    "speech_metadata": {
                        "speaker": voice_name,
                        "style": script_item["breathy_style"],
                    },
                },
            ]
            logger.info(f"Synthesizing dialogue via Gemini TTS ({voice_name})...")
            tts_res = tts_client.synthesize(
                parts=tts_parts,
                voice=voice_name,
                output_wav_path=raw_wav_path,
            )
            raw_json_path.write_text(
                json.dumps({"script": script_item, "tts_response": tts_res}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            time.sleep(2.0)
        else:
            logger.info(f"Using existing raw audio: {raw_wav_path.name}")

        if not raw_wav_path.exists():
            logger.warning(f"Audio file missing for {dialogue_id}, skipping.")
            continue

        # 4. TIFA Alignment (Isolated single-file alignment, O(1) per dialogue)
        if not raw_tg_path.exists() and not skip_tifa:
            logger.info(f"Running TIFA forced alignment on {dialogue_id}...")
            aligner.align_single_file(
                audio_path=raw_wav_path,
                text_path=raw_txt_path,
                output_tg_path=raw_tg_path,
            )
        else:
            logger.info(f"Using existing raw TextGrid: {raw_tg_path.name}")

        # 5. Three-Stage Pop-Free Slicing & FLAC Encoding (Level 8)
        out_modal_flac = meta_mgr.audio_dir / f"{modal_key}.flac"
        out_breathy_flac = meta_mgr.audio_dir / f"{breathy_key}.flac"
        out_modal_tg = meta_mgr.textgrid_dir / f"{modal_key}.TextGrid"
        out_breathy_tg = meta_mgr.textgrid_dir / f"{breathy_key}.TextGrid"

        logger.info(f"Executing Three-Stage Zero-Crossing Slicing...")
        slice_stats = slice_dialogue_pair(
            audio_path=raw_wav_path,
            textgrid_path=raw_tg_path,
            out_audio_modal=out_modal_flac,
            out_audio_breathy=out_breathy_flac,
            out_tg_modal=out_modal_tg,
            out_tg_breathy=out_breathy_tg,
            flac_compression_level=8,
            modal_text=script_item.get("modal_text"),
            breathy_text=script_item.get("breathy_text_clean"),
        )
        logger.info(f"Sliced at {slice_stats['cut_time_s']}s (Zero Crossing). Modal={slice_stats['modal_duration_s']}s | Breathy={slice_stats['breathy_duration_s']}s")

        # 6. Extract Vowel + Sonorant Ground Truth & Write Breathy Shard
        b_intervals, b_shard = extract_breathy_slice_intervals(
            breathy_tg_path=out_breathy_tg,
            file_key=breathy_key,
            speaker_id=speaker_uid,
            context_id=script_item["sub_id"],
        )
        shard_file = write_breathy_shard(dataset_root, breathy_key, b_shard)
        logger.info(f"Exported Breathy shard with {len(b_intervals)} labeled intervals to {shard_file.name}")
        # Note: Modal slice intentionally receives NO shard file (0% false-negative guarantee)

        all_manifest_records.extend(b_intervals)

        # 7. Acoustic CPPS & Voicing Validation
        val_res = validate_sample_acoustics(out_breathy_flac, b_intervals)
        acoustic_results.append(val_res)

        # 8. Append full audit record to synthesis_metadata.csv
        audit_record = {
            "dialogue_id": dialogue_id,
            "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            "voice": voice_name,
            "speaker_id": speaker_uid,
            "gender": voice_info["gender"],
            "macro_id": script_item.get("macro_id", ""),
            "sub_id": script_item.get("sub_id", ""),
            "scenario_hint": script_item.get("scenario_hint", ""),
            "modal_file_key": modal_key,
            "breathy_file_key": breathy_key,
            "modal_text": script_item.get("modal_text", ""),
            "breathy_text_clean": script_item.get("breathy_text_clean", ""),
            "clean_full_text": script_item.get("clean_full_text", ""),
            "modal_style": script_item.get("modal_style", ""),
            "breathy_style": script_item.get("breathy_style", ""),
            "cut_time_s": slice_stats["cut_time_s"],
            "pause_duration_s": slice_stats["pause_duration_s"],
            "modal_duration_s": slice_stats["modal_duration_s"],
            "breathy_duration_s": slice_stats["breathy_duration_s"],
            "breathy_voicing": val_res.get("breathy_avg_voicing", float("nan")),
            "breathy_cpps_db": val_res.get("breathy_avg_cpps", float("nan")),
            "raw_dialogue_file": raw_wav_path.name,
        }
        meta_mgr.append_synthesis_record(audit_record)

        # Manifest slice rows
        slice_rows = [
            {
                "file_key": modal_key,
                "audio_path": f"audio/{modal_key}.flac",
                "tg_path": f"textgrid/{modal_key}.TextGrid",
                "speaker_id": speaker_uid,
                "gender": voice_info["gender"],
                "section": "modal",
                "duration_s": slice_stats["modal_duration_s"],
                "text": script_item.get("modal_text", ""),
                "is_annotated": False,
            },
            {
                "file_key": breathy_key,
                "audio_path": f"audio/{breathy_key}.flac",
                "tg_path": f"textgrid/{breathy_key}.TextGrid",
                "speaker_id": speaker_uid,
                "gender": voice_info["gender"],
                "section": "breathy",
                "duration_s": slice_stats["breathy_duration_s"],
                "text": script_item.get("breathy_text_clean", ""),
                "is_annotated": True,
            },
        ]
        meta_mgr.append_slices_manifest_rows(slice_rows)
        processed_count += 1

    # ── Summary Report ──
    if acoustic_results:
        report_md = generate_validation_summary(acoustic_results)
        report_file = dataset_root / "acoustic_validation_report.md"
        report_file.write_text(report_md, encoding="utf-8")
        logger.info(f"Validation summary report written to {report_file}")

    print("\n" + "=" * 65)
    print(f"BREATHYTTS PIPELINE EXECUTION COMPLETED!")
    print(f"- Processed Dialogues: {processed_count}")
    print(f"- Dataset Root:        {dataset_root}")
    print(f"- Sliced FLAC Audio:   {meta_mgr.audio_dir}")
    print(f"- Sliced TextGrids:    {meta_mgr.textgrid_dir}")
    print(f"- Shard Annotations:   {dataset_root / '_breathy_cache' / 'batches' / 'batch_1'}")
    print(f"- Metadata Audit Log:  {meta_mgr.metadata_csv}")
    print(f"- Speaker Registry:    {meta_mgr.speaker_mapping_tsv}")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--script-cache-dir", type=Path, default=None)
    parser.add_argument("--num-dialogues", type=int, default=1)
    parser.add_argument("--voice", type=str, default=None)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-tts", action="store_true")
    parser.add_argument("--skip-tifa", action="store_true")
    parser.add_argument("--scripts-only", action="store_true", help="Generate/manage script pool only and exit without synthesis.")
    args = parser.parse_args()

    run_pipeline(
        dataset_root=args.dataset_root,
        config_path=args.config,
        script_cache_dir=args.script_cache_dir,
        num_dialogues=args.num_dialogues,
        voice_filter=args.voice,
        skip_llm=args.skip_llm,
        skip_tts=args.skip_tts,
        skip_tifa=args.skip_tifa,
        scripts_only=args.scripts_only,
    )


if __name__ == "__main__":
    main()
