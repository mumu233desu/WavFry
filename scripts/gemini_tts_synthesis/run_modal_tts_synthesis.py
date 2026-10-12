#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end orchestrator for ModalTTS Japanese counterfactual synthesis, alignment & slicing.

Executes the counterfactual modal benchmark pipeline:
1. Script Pool Loading (re-uses existing 540 BreathyTTS cached scripts directly, 0 LLM cost)
2. Canonical Modal Style Mapping (loads configs/modal_contexts.yaml, 100% consistent across items)
3. De-tagged Clean Text Synthesis (removes <breath> tags, enforces firm glottal closure & periodic vibration)
4. Batch TIFA Forced Alignment (one-time PyTorch model load, GPU batch alignment)
5. Local Three-Stage Slicing, Shard Export & Metadata Registration (zero-crossing snap, Level 8 FLAC)
6. Full Breakpoint Resumption: 100% idempotent by default, skips completed dialogues in 0ms.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime
import json
import logging
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Set
import soundfile as sf
import yaml

# Ensure repository root is in sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from scripts.gemini_tts_synthesis.metadata_manager import (
    MetadataManager,
    VOICE_REGISTRY,
    VOICE_BY_NAME,
)
from scripts.gemini_tts_synthesis.tts_client import (
    GeminiTTSClient,
    DEFAULT_TTS_ENDPOINT,
    DEFAULT_TTS_KEY,
    DEFAULT_TTS_MODEL,
)
from scripts.gemini_tts_synthesis.tifa_align_wrapper import TIFAAlignmentWrapper
from scripts.gemini_tts_synthesis.audio_dialogue_slicer import slice_dialogue_pair
from scripts.gemini_tts_synthesis.textgrid_labeler import (
    extract_modal_slice_intervals,
    write_breathy_shard,
)
from scripts.gemini_tts_synthesis.acoustic_validator import (
    validate_sample_acoustics,
    generate_validation_summary,
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
logger = logging.getLogger("modal_tts_orchestrator")

DEFAULT_DATASET_ROOT = Path(r"E:\Corpus\Breathy\ModalTTS")
DEFAULT_SCRIPT_CACHE_DIR = Path(r"E:\Corpus\Breathy\BreathyTTS\script_pool_cache")
CONFIG_MODAL_CONTEXTS = Path(__file__).resolve().parent / "configs" / "modal_contexts.yaml"


def load_modal_contexts_config(config_path: Path = CONFIG_MODAL_CONTEXTS) -> Dict[str, Any]:
    """Load canonical modal styles and global instructions template."""
    if not config_path.exists():
        raise FileNotFoundError(f"Modal contexts config not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    contexts_map = {c["sub_id"]: c for c in data.get("contexts", [])}
    template = data.get(
        "global_instructions_template",
        "Section 1 ({modal_text}): {part1_style}. Section 2 ({part2_text}): {part2_style}. CRITICAL: Maintain solid periodic vocal fold vibration and full glottal closure; strictly do NOT produce unvoiced whisper or breathy voice.",
    )
    return {
        "global_instructions_template": template,
        "contexts_map": contexts_map,
    }


def run_pipeline(
    dataset_root: Path = DEFAULT_DATASET_ROOT,
    script_cache_dir: Path = DEFAULT_SCRIPT_CACHE_DIR,
    num_dialogues: Optional[int] = None,
    voice_filter: Optional[str] = None,
    workers: int = 8,
    tifa_batch_size: int = 4,
    tts_endpoint: str = DEFAULT_TTS_ENDPOINT,
    tts_api_key: str = DEFAULT_TTS_KEY,
    tts_model: str = DEFAULT_TTS_MODEL,
    skip_tts: bool = False,
    skip_tifa: bool = False,
) -> None:
    dataset_root = dataset_root.resolve()
    script_cache_dir = script_cache_dir.resolve()

    logger.info(f"=== Initializing ModalTTS Counterfactual Pipeline at {dataset_root} ===")
    meta_mgr = MetadataManager(dataset_root)
    modal_config = load_modal_contexts_config()
    modal_contexts_map = modal_config["contexts_map"]
    global_instructions_template = modal_config["global_instructions_template"]
    logger.info(f"Loaded {len(modal_contexts_map)} standardized modal context styles and global instructions template from YAML.")

    # ── Stage 0: Load Script Pool ──
    logger.info(f"=== STAGE 0: Loading Script Pool from {script_cache_dir} ===")
    scripts: List[Dict[str, Any]] = []
    if script_cache_dir.exists():
        for sf_file in sorted(script_cache_dir.glob("*.json")):
            try:
                scripts.append(json.loads(sf_file.read_text(encoding="utf-8")))
            except Exception:
                pass

    if not scripts:
        logger.error(
            f"No scripts available in {script_cache_dir}! "
            f"Please ensure BreathyTTS script cache exists."
        )
        return

    # Filter voices if requested
    if voice_filter:
        voices_to_run = [v for v in VOICE_REGISTRY if v["voice"].lower() == voice_filter.lower()]
        if not voices_to_run:
            raise ValueError(f"Unknown voice filter '{voice_filter}'.")
    else:
        voices_to_run = VOICE_REGISTRY

    total_to_process = min(num_dialogues or len(scripts), len(scripts))
    logger.info(f"Available scripts: {len(scripts)} | Voices to schedule: {len(voices_to_run)} | Target processing: {total_to_process}")

    # Build planned dialogue execution list (maintains round-robin pairing)
    planned_items: List[Dict[str, Any]] = []
    for d_idx in range(total_to_process):
        script_item = scripts[d_idx % len(scripts)]
        voice_info = voices_to_run[d_idx % len(voices_to_run)]
        voice_name = voice_info["voice"]
        speaker_uid = voice_info["uid"]
        sub_id = script_item.get("sub_id", "")
        dialogue_id = f"dialogue_{d_idx+1:04d}_{voice_name.lower()}_{sub_id}"

        raw_wav = meta_mgr.raw_dialogues_dir / f"{dialogue_id}.wav"
        raw_txt = meta_mgr.raw_dialogues_dir / f"{dialogue_id}.txt"
        raw_tg = meta_mgr.raw_dialogues_dir / f"{dialogue_id}.TextGrid"
        raw_json = meta_mgr.raw_dialogues_dir / f"{dialogue_id}.json"

        planned_items.append({
            "d_idx": d_idx + 1,
            "dialogue_id": dialogue_id,
            "script": script_item,
            "sub_id": sub_id,
            "voice_info": voice_info,
            "voice_name": voice_name,
            "speaker_uid": speaker_uid,
            "raw_wav_path": raw_wav,
            "raw_txt_path": raw_txt,
            "raw_tg_path": raw_tg,
            "raw_json_path": raw_json,
        })

    # Detect already completed dialogues for seamless default breakpoint resume
    completed_dialogues = meta_mgr.get_completed_dialogues()
    logger.info(f"Breakpoint scanner: {len(completed_dialogues)} dialogues already fully sliced and recorded.")

    # ──────────────────────────────────────────────────────────────────────────
    # ── STAGE 1: Concurrent High-Speed TTS Synthesis (Phase 1: Multi-Threaded)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*65}\nSTAGE 1: Concurrent Modal TTS Synthesis (Workers: {workers})\n{'='*65}")
    tts_client = GeminiTTSClient(
        endpoint=tts_endpoint,
        api_key=tts_api_key,
        model=tts_model,
    )

    items_to_synthesize = []
    for item in planned_items:
        script_item = item["script"]
        full_text = script_item.get("clean_full_text", "")
        norm_concat = re.sub(r"\s+", "", script_item.get("modal_text", "") + script_item.get("breathy_text_clean", ""))
        norm_full = re.sub(r"\s+", "", full_text)
        if norm_concat != norm_full:
            raise ValueError(
                f"Transcript consistency error in {script_item.get('script_id')}: "
                f"modal+clean ('{norm_concat}') does not match clean_full_text ('{norm_full}')"
            )

        # Write clean transcript sidecar for TIFA
        if not item["raw_txt_path"].exists() or item["raw_txt_path"].read_text(encoding="utf-8").strip() != full_text:
            item["raw_txt_path"].write_text(full_text, encoding="utf-8")

        # Determine if TTS audio is needed
        if item["raw_wav_path"].exists() and item["raw_wav_path"].stat().st_size > 4000:
            continue
        if not skip_tts:
            items_to_synthesize.append(item)

    session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_synthesized_items: List[Dict[str, Any]] = []

    if items_to_synthesize:
        logger.info(f"[{session_id}] Initiating concurrent Modal TTS generation for {len(items_to_synthesize)} dialogues ({workers} threads)...")

        def _synthesize_task(task_item: Dict[str, Any]) -> Dict[str, Any]:
            sc = task_item["script"]
            vn = task_item["voice_name"]
            sub_id = task_item["sub_id"]
            out_wav = task_item["raw_wav_path"]
            out_json = task_item["raw_json_path"]

            modal_ctx = modal_contexts_map.get(sub_id, {})
            rec_styles = modal_ctx.get("recommended_styles", {})

            part1_style = rec_styles.get("part1_style") or sc.get("modal_style", "casual, clear speaking tone")
            part2_modal_style = rec_styles.get("part2_modal_style") or "firm, resonant modal speech, full glottal closure, strictly no breathiness"

            # Strict de-tagging: use breathy_text_clean without any tags
            clean_part2_text = sc["breathy_text_clean"]

            parts = [
                {
                    "text": sc["modal_text"],
                    "speech_metadata": {
                        "speaker": vn,
                        "style": part1_style,
                    },
                },
                {
                    "text": clean_part2_text,
                    "speech_metadata": {
                        "speaker": vn,
                        "style": part2_modal_style,
                    },
                },
            ]

            custom_instructions = global_instructions_template.format(
                modal_text=sc["modal_text"],
                part1_style=part1_style,
                part2_text=clean_part2_text,
                part2_style=part2_modal_style,
            )

            resp = tts_client.synthesize(
                parts=parts,
                voice=vn,
                output_wav_path=out_wav,
                instructions_override=custom_instructions,
            )
            out_json.write_text(
                json.dumps({
                    "script": sc,
                    "tts_response": resp,
                    "modal_part1_style": part1_style,
                    "modal_part2_style": part2_modal_style,
                    "instructions": custom_instructions,
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return {
                "dialogue_id": task_item["dialogue_id"],
                "generation_id": resp.get("generation_id"),
                "voice": vn,
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {executor.submit(_synthesize_task, it): it for it in items_to_synthesize}
            for fut in concurrent.futures.as_completed(future_to_item):
                it = future_to_item[fut]
                try:
                    res = fut.result()
                    session_synthesized_items.append(res)
                    logger.info(f"[TTS Done] {it['dialogue_id']} (GenID: {res.get('generation_id')})")
                except Exception as exc:
                    logger.error(f"[TTS Failed] {it['dialogue_id']}: {exc}")
                    raise exc
    else:
        logger.info(f"All {total_to_process} raw audio files already exist on disk. Skipping TTS synthesis.")

    # ──────────────────────────────────────────────────────────────────────────
    # ── STAGE 2: Batch TIFA Forced Alignment (Phase 2: One-Time Model Loading)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*65}\nSTAGE 2: Batch TIFA Forced Alignment (One-Time Model Load)\n{'='*65}")
    aligner = TIFAAlignmentWrapper()

    missing_tifa_items = [
        it for it in planned_items
        if not it["raw_tg_path"].exists() and it["raw_wav_path"].exists()
    ]

    if missing_tifa_items and not skip_tifa:
        logger.info(f"Found {len(missing_tifa_items)} pending alignments. Launching single-pass batch TIFA (batch_size={tifa_batch_size})...")
        audio_paths = [it["raw_wav_path"] for it in missing_tifa_items]
        text_paths = [it["raw_txt_path"] for it in missing_tifa_items]

        aligner.align_batch_files(
            audio_paths=audio_paths,
            text_paths=text_paths,
            output_dir=meta_mgr.raw_dialogues_dir,
            batch_size=tifa_batch_size,
        )
    else:
        logger.info(f"All {total_to_process} TextGrid alignment files already present. Skipping TIFA alignment.")

    # ──────────────────────────────────────────────────────────────────────────
    # ── STAGE 3: Local Slicing, Shard Export & Metadata Registration (Phase 3)
    # ──────────────────────────────────────────────────────────────────────────
    logger.info(f"\n{'='*65}\nSTAGE 3: Pop-Free Slicing, Modal Shard Export & Metadata Registration\n{'='*65}")
    processed_count = 0
    all_manifest_records: List[Dict[str, Any]] = []
    acoustic_results: List[Dict[str, Any]] = []

    for item in planned_items:
        d_id = item["dialogue_id"]
        d_idx = item["d_idx"]
        script_item = item["script"]
        voice_info = item["voice_info"]
        voice_name = item["voice_name"]
        speaker_uid = item["speaker_uid"]
        sub_id = item["sub_id"]

        # Default breakpoint skip
        if d_id in completed_dialogues:
            logger.info(f"[{d_idx}/{total_to_process}] Skipping already completed dialogue: {d_id}")
            continue

        if not item["raw_wav_path"].exists() or not item["raw_tg_path"].exists():
            logger.warning(f"Missing WAV or TextGrid for {d_id}, cannot slice. Skipping.")
            continue

        # 1. Monotonic utterance key allocation (thread-safe)
        modal_key, breathy_key, spk_meta = meta_mgr.allocate_next_utterance_pair(voice_name)

        out_modal_flac = meta_mgr.audio_dir / f"{modal_key}.flac"
        out_breathy_flac = meta_mgr.audio_dir / f"{breathy_key}.flac"
        out_modal_tg = meta_mgr.textgrid_dir / f"{modal_key}.TextGrid"
        out_breathy_tg = meta_mgr.textgrid_dir / f"{breathy_key}.TextGrid"

        logger.info(f"[{d_idx}/{total_to_process}] Slicing {d_id} -> Modal_1={modal_key} | Modal_2={breathy_key}")

        # 2. Three-Stage Zero-Crossing Pop-Free Slicing & FLAC Level 8 Export
        slice_stats = slice_dialogue_pair(
            audio_path=item["raw_wav_path"],
            textgrid_path=item["raw_tg_path"],
            out_audio_modal=out_modal_flac,
            out_audio_breathy=out_breathy_flac,
            out_tg_modal=out_modal_tg,
            out_tg_breathy=out_breathy_tg,
            flac_compression_level=8,
            modal_text=script_item.get("modal_text"),
            breathy_text=script_item.get("breathy_text_clean"),
        )

        # 3. Extract Vowel Intervals & Write Pure Modal Shards for Both Slices
        m1_intervals, m1_shard = extract_modal_slice_intervals(
            modal_tg_path=out_modal_tg,
            file_key=modal_key,
            speaker_id=speaker_uid,
            context_id=sub_id,
        )
        write_breathy_shard(dataset_root, modal_key, m1_shard)
        all_manifest_records.extend(m1_intervals)

        m2_intervals, m2_shard = extract_modal_slice_intervals(
            modal_tg_path=out_breathy_tg,
            file_key=breathy_key,
            speaker_id=speaker_uid,
            context_id=sub_id,
        )
        write_breathy_shard(dataset_root, breathy_key, m2_shard)
        all_manifest_records.extend(m2_intervals)

        # 4. Acoustic Validation
        combined_intervals = m1_intervals + m2_intervals
        val_res = validate_sample_acoustics(out_breathy_flac, combined_intervals)
        acoustic_results.append(val_res)

        # Extract generation ID and OpenRouter token stats
        gen_id = ""
        prompt_tokens = 0
        comp_tokens = 0
        cost_usd = 0.0
        if item["raw_json_path"].exists():
            try:
                raw_j = json.loads(item["raw_json_path"].read_text(encoding="utf-8"))
                tts_info = raw_j.get("tts_response", {})
                gen_id = tts_info.get("generation_id", "")
            except Exception:
                pass

        if gen_id and "openrouter.ai" in tts_endpoint.lower():
            stats = tts_client.fetch_generation_stats(gen_id)
            prompt_tokens = stats.get("tokens_prompt", 0)
            comp_tokens = stats.get("tokens_completion", 0)
            cost_usd = stats.get("total_cost", 0.0)

        modal_ctx = modal_contexts_map.get(sub_id, {})
        rec_styles = modal_ctx.get("recommended_styles", {})
        part1_style = rec_styles.get("part1_style") or script_item.get("modal_style", "")
        part2_modal_style = rec_styles.get("part2_modal_style") or ""

        # 5. Append Full Audit Record to synthesis_metadata.csv
        audit_record = {
            "dialogue_id": d_id,
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
            "modal_style": part1_style,
            "breathy_style": part2_modal_style,  # Contains counterfactual modal style
            "cut_time_s": slice_stats["cut_time_s"],
            "pause_duration_s": slice_stats["pause_duration_s"],
            "modal_duration_s": slice_stats["modal_duration_s"],
            "breathy_duration_s": slice_stats["breathy_duration_s"],
            "breathy_voicing": val_res.get("breathy_avg_voicing", float("nan")),
            "breathy_cpps_db": val_res.get("breathy_avg_cpps", float("nan")),
            "raw_dialogue_file": item["raw_wav_path"].name,
            "generation_id": gen_id,
            "tokens_prompt": prompt_tokens,
            "tokens_completion": comp_tokens,
            "cost_usd": cost_usd,
        }
        meta_mgr.append_synthesis_record(audit_record)

        # 6. Append Slices to manifest_slices.csv
        slice_rows = [
            {
                "file_key": modal_key,
                "audio_path": f"audio/{modal_key}.flac",
                "tg_path": f"textgrid/{modal_key}.TextGrid",
                "speaker_id": speaker_uid,
                "gender": voice_info["gender"],
                "section": "modal_part1",
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
                "section": "modal_part2",
                "duration_s": slice_stats["breathy_duration_s"],
                "text": script_item.get("breathy_text_clean", ""),
                "is_annotated": False,
            },
        ]
        meta_mgr.append_slices_manifest_rows(slice_rows)
        processed_count += 1

    # ── Summary Report & Token Usage Accounting ──
    if acoustic_results:
        report_md = generate_validation_summary(acoustic_results)
        report_file = dataset_root / "acoustic_validation_report.md"
        report_file.write_text(report_md, encoding="utf-8")
        logger.info(f"Validation summary report written to {report_file}")

    # ── Session Token Usage & Cumulative Accounting ──
    session_prompt_tokens = 0
    session_comp_tokens = 0
    session_cost_usd = 0.0

    if session_synthesized_items:
        logger.info(f"Fetching OpenRouter generation stats for {len(session_synthesized_items)} newly synthesized dialogues...")
        for it in session_synthesized_items:
            gid = it.get("generation_id")
            if gid:
                st = tts_client.fetch_generation_stats(gid)
                session_prompt_tokens += st.get("tokens_prompt", 0)
                session_comp_tokens += st.get("tokens_completion", 0)
                session_cost_usd += st.get("total_cost", 0.0)

    sessions_jsonl = dataset_root / "token_usage_sessions.jsonl"
    session_record = {
        "session_id": session_id,
        "timestamp": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "newly_synthesized_dialogues": len(session_synthesized_items),
        "prompt_tokens": session_prompt_tokens,
        "completion_tokens": session_comp_tokens,
        "cost_usd": round(session_cost_usd, 6),
        "dialogues": [it["dialogue_id"] for it in session_synthesized_items],
    }
    with open(sessions_jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(session_record, ensure_ascii=False) + "\n")

    all_sessions = []
    if sessions_jsonl.exists():
        with open(sessions_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        all_sessions.append(json.loads(line))
                    except Exception:
                        pass

    all_time_requests = sum(s.get("newly_synthesized_dialogues", 0) for s in all_sessions)
    all_time_prompt = sum(s.get("prompt_tokens", 0) for s in all_sessions)
    all_time_comp = sum(s.get("completion_tokens", 0) for s in all_sessions)
    all_time_cost = sum(s.get("cost_usd", 0.0) for s in all_sessions)

    effective_prompt_tokens = 0
    effective_comp_tokens = 0
    effective_cost_usd = 0.0
    effective_dialogue_count = 0
    if meta_mgr.metadata_csv.exists() and meta_mgr.metadata_csv.stat().st_size > 0:
        try:
            import pandas as pd
            df_meta = pd.read_csv(meta_mgr.metadata_csv)
            effective_dialogue_count = len(df_meta)
            if "tokens_prompt" in df_meta.columns:
                effective_prompt_tokens = int(df_meta["tokens_prompt"].fillna(0).sum())
            if "tokens_completion" in df_meta.columns:
                effective_comp_tokens = int(df_meta["tokens_completion"].fillna(0).sum())
            if "cost_usd" in df_meta.columns:
                effective_cost_usd = float(df_meta["cost_usd"].fillna(0.0).sum())
        except Exception:
            pass

    summary_file = dataset_root / "token_usage_summary.json"
    summary_data = {
        "dataset": "ModalTTS",
        "last_updated": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "last_session_id": session_id,
        "effective_dataset_usage": {
            "dialogue_count": effective_dialogue_count,
            "prompt_tokens": effective_prompt_tokens,
            "completion_tokens": effective_comp_tokens,
            "total_tokens": effective_prompt_tokens + effective_comp_tokens,
            "total_cost_usd": round(effective_cost_usd, 6),
        },
        "session_usage": {
            "dialogue_count": len(session_synthesized_items),
            "prompt_tokens": session_prompt_tokens,
            "completion_tokens": session_comp_tokens,
            "total_tokens": session_prompt_tokens + session_comp_tokens,
            "total_cost_usd": round(session_cost_usd, 6),
        },
        "all_time_usage": {
            "total_api_requests": all_time_requests,
            "prompt_tokens": all_time_prompt,
            "completion_tokens": all_time_comp,
            "total_tokens": all_time_prompt + all_time_comp,
            "total_cost_usd": round(all_time_cost, 6),
        },
    }
    summary_file.write_text(json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"\n{'='*65}\nModalTTS Orchestration Run Complete!\n{'='*65}")
    print(f"\n=== ModalTTS Financial & Token Accounting ===")
    print(f"- Session Newly Synthesized:    {len(session_synthesized_items)} dialogues")
    print(f"- Session Cost:                 ${session_cost_usd:.4f} USD")
    print(f"- Effective Dataset Cost:       ${effective_cost_usd:.4f} USD")
    print(f"- Total Cumulative Cost:        ${all_time_cost:.4f} USD")
    print("==================================================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="ModalTTS Japanese Counterfactual Speech Synthesis Pipeline")
    parser.add_argument("--dataset_root", type=Path, default=DEFAULT_DATASET_ROOT, help="Output root directory for ModalTTS")
    parser.add_argument("--script_cache_dir", type=Path, default=DEFAULT_SCRIPT_CACHE_DIR, help="Path to cached scripts from BreathyTTS")
    parser.add_argument("--num_dialogues", type=int, default=None, help="Limit number of dialogues to process")
    parser.add_argument("--voice", type=str, default=None, help="Filter by single voice name")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent TTS synthesis threads")
    parser.add_argument("--tifa_batch_size", type=int, default=4, help="TIFA GPU batch alignment size")
    parser.add_argument("--tts_endpoint", type=str, default=DEFAULT_TTS_ENDPOINT, help="TTS endpoint URL")
    parser.add_argument("--tts_api_key", type=str, default=DEFAULT_TTS_KEY, help="TTS API authorization key")
    parser.add_argument("--tts_model", type=str, default=DEFAULT_TTS_MODEL, help="TTS model name")
    parser.add_argument("--skip_tts", action="store_true", help="Skip Stage 1 TTS synthesis")
    parser.add_argument("--skip_tifa", action="store_true", help="Skip Stage 2 TIFA alignment")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        dataset_root=args.dataset_root,
        script_cache_dir=args.script_cache_dir,
        num_dialogues=args.num_dialogues,
        voice_filter=args.voice,
        workers=args.workers,
        tifa_batch_size=args.tifa_batch_size,
        tts_endpoint=args.tts_endpoint,
        tts_api_key=args.tts_api_key,
        tts_model=args.tts_model,
        skip_tts=args.skip_tts,
        skip_tifa=args.skip_tifa,
    )
