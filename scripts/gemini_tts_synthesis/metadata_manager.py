#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metadata manager and speaker registry for BreathyTTS dataset.

Maintains 30 mixed Gemini TTS voices mapped to U0001~U0030, tracks monotonic
utterance indexing per speaker, and updates synthesis_metadata.csv, speaker_mapping.tsv,
and manifest_slices.csv.
"""

from __future__ import annotations

import csv
import datetime
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

logger = logging.getLogger("metadata_manager")

# Universal Registry of 30 standard Gemini TTS voices
# Mixed and balanced across genders
VOICE_REGISTRY: List[Dict[str, str]] = [
    {"uid": "U0001", "voice": "Kore", "gender": "F", "desc": "Firm, neutral, gentle narration & conversation"},
    {"uid": "U0002", "voice": "Aoede", "gender": "F", "desc": "Breezy, natural, conversational expression"},
    {"uid": "U0003", "voice": "Leda", "gender": "F", "desc": "Soft, warm, storytelling and healing"},
    {"uid": "U0004", "voice": "Zephyr", "gender": "F", "desc": "Bright, youthful, energetic female tone"},
    {"uid": "U0005", "voice": "Autonoe", "gender": "F", "desc": "Articulate, crisp, confident young female"},
    {"uid": "U0006", "voice": "Puck", "gender": "M", "desc": "Upbeat, lively, friendly young male"},
    {"uid": "U0007", "voice": "Charon", "gender": "M", "desc": "Calm, deep, professional, steady male"},
    {"uid": "U0008", "voice": "Fenrir", "gender": "M", "desc": "Passionate, resonant, dynamic male"},
    {"uid": "U0009", "voice": "Enceladus", "gender": "M", "desc": "Gentle, calm narrative male"},
    {"uid": "U0010", "voice": "Orus", "gender": "M", "desc": "Clear, resonant, assertive adult male"},
    {"uid": "U0011", "voice": "Despina", "gender": "F", "desc": "Gentle, understated feminine tone"},
    {"uid": "U0012", "voice": "Callirrhoe", "gender": "F", "desc": "Melodic, graceful conversational tone"},
    {"uid": "U0013", "voice": "Erinome", "gender": "F", "desc": "Thoughtful, intimate female speaker"},
    {"uid": "U0014", "voice": "Laomedeia", "gender": "F", "desc": "Poised, elegant female timbre"},
    {"uid": "U0015", "voice": "Pulcherrima", "gender": "F", "desc": "Polished, expressive female delivery"},
    {"uid": "U0016", "voice": "Sulafat", "gender": "F", "desc": "Soothing, relaxed female storyteller"},
    {"uid": "U0017", "voice": "Umbriel", "gender": "F", "desc": "Quiet, reflective, delicate female voice"},
    {"uid": "U0018", "voice": "Vindemiatrix", "gender": "F", "desc": "Mature, calm, composed female tone"},
    {"uid": "U0019", "voice": "Gacrux", "gender": "F", "desc": "Rich, grounded, warm feminine voice"},
    {"uid": "U0020", "voice": "Achernar", "gender": "F", "desc": "High-pitched, soft, soothing storyteller"},
    {"uid": "U0021", "voice": "Achird", "gender": "M", "desc": "Approachable, warm, low-mid pitch peer"},
    {"uid": "U0022", "voice": "Algenib", "gender": "M", "desc": "Gravelly, grounded, dramatic male"},
    {"uid": "U0023", "voice": "Algieba", "gender": "M", "desc": "Smooth, relaxed, polished corporate male"},
    {"uid": "U0024", "voice": "Alnilam", "gender": "M", "desc": "Firm, direct, commanding male tone"},
    {"uid": "U0025", "voice": "Iapetus", "gender": "M", "desc": "Steady, thoughtful, measured male tone"},
    {"uid": "U0026", "voice": "Rasalgethi", "gender": "M", "desc": "Deep, authoritative, mature male timbre"},
    {"uid": "U0027", "voice": "Sadachbia", "gender": "M", "desc": "Conversational, casual, easygoing male"},
    {"uid": "U0028", "voice": "Sadaltager", "gender": "M", "desc": "Warm, articulate, friendly adult male"},
    {"uid": "U0029", "voice": "Schedar", "gender": "M", "desc": "Robust, confident, vibrant male narrator"},
    {"uid": "U0030", "voice": "Zubenelgenubi", "gender": "M", "desc": "Calm, understated, natural male voice"},
]

VOICE_BY_NAME = {v["voice"]: v for v in VOICE_REGISTRY}
VOICE_BY_UID = {v["uid"]: v for v in VOICE_REGISTRY}


class MetadataManager:
    """Manages BreathyTTS speaker registry and monotonic utterance indexing."""

    def __init__(self, dataset_root: Path):
        self.dataset_root = dataset_root.resolve()
        self.audio_dir = self.dataset_root / "audio"
        self.textgrid_dir = self.dataset_root / "textgrid"
        self.raw_dialogues_dir = self.dataset_root / "raw_dialogues"
        self.cache_dir = self.textgrid_dir / "_breathy_cache"

        # Ensure core directories exist
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.textgrid_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dialogues_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_csv = self.dataset_root / "synthesis_metadata.csv"
        self.speaker_mapping_tsv = self.dataset_root / "speaker_mapping.tsv"
        self.manifest_slices_csv = self.dataset_root / "manifest_slices.csv"

        # Thread safety lock for concurrent allocations and CSV writes
        self._lock = threading.Lock()

        # Scan highest existing utterance index per speaker
        self._utterance_counters: Dict[str, int] = {}
        self._init_utterance_counters()
        self.ensure_speaker_mapping_file()

    def get_completed_dialogues(self) -> Set[str]:
        """Return set of dialogue_ids that have already been fully sliced and registered."""
        with self._lock:
            if not self.metadata_csv.exists() or self.metadata_csv.stat().st_size == 0:
                return set()
            try:
                df = pd.read_csv(self.metadata_csv, usecols=["dialogue_id", "modal_file_key", "breathy_file_key"])
                completed = set()
                for _, row in df.iterrows():
                    d_id = str(row["dialogue_id"])
                    m_key = str(row["modal_file_key"])
                    b_key = str(row["breathy_file_key"])
                    m_flac = self.audio_dir / f"{m_key}.flac"
                    b_flac = self.audio_dir / f"{b_key}.flac"
                    if m_flac.exists() and b_flac.exists():
                        completed.add(d_id)
                return completed
            except Exception as e:
                logger.warning(f"Error reading completed dialogues from {self.metadata_csv}: {e}")
                return set()

    def _init_utterance_counters(self) -> None:
        """Scan audio directory to initialize monotonic counter per speaker."""
        for v in VOICE_REGISTRY:
            self._utterance_counters[v["uid"]] = 0

        for f in self.audio_dir.glob("*.flac"):
            # Pattern: {Gender}_00_JP_JA_JA-JP_{UID}_S0_{UtteranceIdx}.flac
            parts = f.stem.split("_")
            if len(parts) >= 8 and parts[5] in self._utterance_counters:
                uid = parts[5]
                try:
                    u_idx = int(parts[7])
                    if u_idx > self._utterance_counters[uid]:
                        self._utterance_counters[uid] = u_idx
                except ValueError:
                    pass

    def allocate_next_utterance_pair(self, voice_name: str) -> Tuple[str, str, Dict[str, Any]]:
        """Allocate next monotonic utterance index pair for given voice.

        Returns:
            (modal_file_key, breathy_file_key, speaker_info)
        """
        info = VOICE_BY_NAME.get(voice_name)
        if not info:
            raise ValueError(f"Unknown voice '{voice_name}'. Must be one of 30 registered voices.")

        uid = info["uid"]
        gender = info["gender"]

        with self._lock:
            cur_count = self._utterance_counters[uid]
            idx_modal = cur_count + 1
            idx_breathy = cur_count + 2
            self._utterance_counters[uid] = idx_breathy

        modal_key = f"{gender}_00_JP_JA_JA-JP_{uid}_S0_{idx_modal}"
        breathy_key = f"{gender}_00_JP_JA_JA-JP_{uid}_S0_{idx_breathy}"

        return modal_key, breathy_key, info

    def ensure_speaker_mapping_file(self) -> None:
        """Create or update speaker_mapping.tsv formatted like YSJ."""
        rows = []
        for idx, v in enumerate(VOICE_REGISTRY, 1):
            rows.append({
                "Speaker": f"SPK{idx:04d}",
                "UID": v["uid"],
                "Gender": v["gender"],
                "Age": "00",
                "Region": "JP",
                "Language": "JA",
                "Locale": "JA-JP",
                "VoiceName": v["voice"],
                "Description": v["desc"],
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.speaker_mapping_tsv, sep="\t", index=False, encoding="utf-8")

    def append_synthesis_record(self, record: Dict[str, Any]) -> None:
        """Upsert full audit record to synthesis_metadata.csv (keyed by dialogue_id)."""
        if "created_at" not in record:
            record["created_at"] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()

        fieldnames = [
            "dialogue_id",
            "created_at",
            "voice",
            "speaker_id",
            "gender",
            "macro_id",
            "sub_id",
            "scenario_hint",
            "modal_file_key",
            "breathy_file_key",
            "modal_text",
            "breathy_text_clean",
            "clean_full_text",
            "modal_style",
            "breathy_style",
            "cut_time_s",
            "pause_duration_s",
            "modal_duration_s",
            "breathy_duration_s",
            "breathy_voicing",
            "breathy_cpps_db",
            "raw_dialogue_file",
            "generation_id",
            "tokens_prompt",
            "tokens_completion",
            "cost_usd",
        ]

        with self._lock:
            existing_rows = []
            if self.metadata_csv.exists() and self.metadata_csv.stat().st_size > 0:
                with open(self.metadata_csv, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        if r.get("dialogue_id") != record.get("dialogue_id"):
                            existing_rows.append(r)
            existing_rows.append(record)

            with open(self.metadata_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for r in existing_rows:
                    writer.writerow(r)

    def append_slices_manifest_rows(self, rows: List[Dict[str, Any]]) -> None:
        """Upsert sliced audio items to manifest_slices.csv (keyed by file_key)."""
        new_keys = {r["file_key"] for r in rows if "file_key" in r}
        with self._lock:
            if self.manifest_slices_csv.exists() and self.manifest_slices_csv.stat().st_size > 0:
                try:
                    df_existing = pd.read_csv(self.manifest_slices_csv)
                    if "file_key" in df_existing.columns:
                        df_existing = df_existing[~df_existing["file_key"].isin(new_keys)]
                    df_new = pd.DataFrame(rows)
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(self.manifest_slices_csv, index=False, encoding="utf-8")
                    return
                except Exception as e:
                    logger.warning(f"Error reading existing manifest_slices.csv for upsert: {e}")
            df = pd.DataFrame(rows)
            df.to_csv(self.manifest_slices_csv, index=False, encoding="utf-8")


if __name__ == "__main__":
    target_dir = Path(r"E:\Corpus\Breathy\BreathyTTS")
    mgr = MetadataManager(target_dir)
    print(f"MetadataManager initialized for {target_dir}")
    print(f"Total registered voices: {len(VOICE_REGISTRY)}")
    m_key, b_key, spk = mgr.allocate_next_utterance_pair("Kore")
    print(f"Sample allocation for Kore -> {m_key}, {b_key}")
