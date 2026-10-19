#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Production MLLM Breathy Voice Batch Coarse-Annotation Prefill Driver.

Executes unattended batch inference with Gemini 3.8 Flash High, dual-constraint
acoustic alignment, and atomic prefilling into the Qt annotation app shard cache.

Key Features:
1. Zero Fallback: Exclusively targets gemini-3.8-flash-high.
2. Human Labor Protection: Skips any file containing manual review states.
3. Persistent Layered Cache: Stores raw MLLM responses and snaps on resume.
4. Unattended Resilience: 3x exponential backoff retries, failure logging.
5. Graceful Exit: Catches SIGINT / SIGTERM, finishes current file before shutdown.
6. Non-Spamming Logging: Periodic progress statistics, detailed disk log.
7. Dry-Run Verification: Simulates execution and inspects files without API calls.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import datetime
import hashlib
import json
import logging
import os
import re
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    from scripts.mllm_annotation.mllm_prompts import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE
    from scripts.mllm_annotation.mllm_alignment_core import project_mllm_prediction_to_vowels
except ImportError:
    from mllm_prompts import SYSTEM_INSTRUCTION, USER_PROMPT_TEMPLATE
    from mllm_alignment_core import project_mllm_prediction_to_vowels


# ── Defaults and Corpus Configuration ──────────────────────────────────────────

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DEFAULT_ENDPOINT = os.environ.get("LOCAL_LLM_ENDPOINT", "http://127.0.0.1:8045")
DEFAULT_KEY = os.environ.get("LOCAL_LLM_KEY", "")
DEFAULT_MODEL = os.environ.get("LOCAL_LLM_MODEL", "gemini-3.8-flash-high")
DEFAULT_VOWEL_COVERAGE = 0.8
DEFAULT_MAX_RETRIES = 3
DEFAULT_LOG_INTERVAL = 20

HUMAN_STATES: Set[str] = {
    "breathy", "non_breathy", "uncertain", "modal", "creaky", "non_creaky"
}

CORPUS_PRESETS: Dict[str, Dict[str, Any]] = {
    "ysj": {
        "name": "YSJ",
        "audio_dir": Path(r"E:\Corpus\Breathy\YSJ\audio"),
        "textgrid_dir": Path(r"E:\Corpus\Breathy\YSJ\textgrid_tifa"),
        "cache_dir": Path(r"E:\Corpus\Breathy\YSJ\textgrid_tifa\_breathy_cache"),
        "sampling_state": Path(r"E:\Corpus\Breathy\YSJ\textgrid_tifa\_breathy_cache\sampling_state.json"),
    },
    "ita": {
        "name": "ITA",
        "audio_dir": Path(r"E:\Corpus\Breathy\ita-corpus-main"),
        "textgrid_dir": Path(r"E:\Corpus\Breathy\ita-corpus-main"),
        "cache_dir": Path(r"E:\Corpus\Breathy\ita-corpus-main\_breathy_cache"),
        "sampling_state": Path(r"E:\Corpus\Breathy\ita-corpus-main\_breathy_cache\sampling_state.json"),
    },
}


# ── Global Shutdown Handler ───────────────────────────────────────────────────

_SHUTDOWN_REQUESTED = False


def _signal_handler(signum, frame):
    global _SHUTDOWN_REQUESTED
    if not _SHUTDOWN_REQUESTED:
        _SHUTDOWN_REQUESTED = True
        print("\n\n[WARNING] Cancellation signal received! Safely finishing current file before exit...\n", flush=True)
    else:
        print("\n[CRITICAL] Force termination requested by user. Exiting immediately.", flush=True)
        sys.exit(130)


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Logging Setup ─────────────────────────────────────────────────────────────

def setup_logger(log_dir: Path, corpus_name: str) -> logging.Logger:
    """Configure disk file logger and concise console logger."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"prefill_{corpus_name}_{timestamp}.log"

    logger = logging.getLogger("mllm_prefill")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler: verbose debug
    file_fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_fmt)
    logger.addHandler(fh)

    # Console handler: info level with concise format
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.propagate = False
    logger.info(f"[Logger Initialized] Detailed execution log -> {log_file}")
    return logger


# ── File and Shard Utilities ──────────────────────────────────────────────────

def compute_shard_path(batch_dir: Path, file_key: str) -> Path:
    """Return path to shard, checking both breathy and fallback creaky shard directories."""
    digest = hashlib.sha256(str(file_key).encode("utf-8")).hexdigest()
    breathy_path = batch_dir / "manual_breathy_annotations.json.d" / f"{digest}.json"
    if breathy_path.exists():
        return breathy_path
    creaky_path = batch_dir / "manual_creaky_annotations.json.d" / f"{digest}.json"
    if creaky_path.exists():
        return creaky_path
    return breathy_path


def find_audio_path(audio_dir: Path, file_key: str) -> Optional[Path]:
    """Find audio file with supported extensions (.flac, .wav, .mp3)."""
    for ext in (".flac", ".wav", ".mp3"):
        p = audio_dir / f"{file_key}{ext}"
        if p.exists():
            return p
    return None


def find_textgrid_path(tg_dir: Path, file_key: str) -> Optional[Path]:
    """Find TextGrid file (.TextGrid or .textgrid)."""
    for ext in (".TextGrid", ".textgrid"):
        p = tg_dir / f"{file_key}{ext}"
        if p.exists():
            return p
    return None


def is_human_annotated(shard_path: Path) -> bool:
    """Check if the shard file contains any manual human annotations.
    
    Returns True if:
    1. 'reviewed_at', 'is_frame_verified', or 'is_golden_file' is present.
    2. 'exact_creak_spans' or 'uncertain_spans' is present.
    3. Any interval state is in HUMAN_STATES ('breathy', 'non_breathy', 'uncertain', 'non_creaky', etc.).
    Returns False only if shard does not exist or only contains 'mllm_breathy'.
    """
    if not shard_path.exists():
        return False
    try:
        content = shard_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if not isinstance(data, dict):
            return False

        if "reviewed_at" in data or data.get("is_frame_verified") or data.get("is_golden_file"):
            return True
        if data.get("exact_creak_spans") or data.get("uncertain_spans"):
            return True

        annotations = data.get("annotations", {})
        if isinstance(annotations, dict):
            for state in annotations.values():
                if state in HUMAN_STATES:
                    return True
        return False
    except Exception:
        # Conservative safeguard: if corrupted, preserve file
        return True


def _safe_atomic_write_json(path: Path, payload: Dict[str, Any], max_retries: int = 5) -> None:
    """Atomically write JSON with flush, fsync, and Windows PermissionError lock retries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}_{threading.get_ident()}_{int(time.time() * 1000)}.tmp")
    text = json.dumps(payload, ensure_ascii=False, indent=2)

    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())

    for attempt in range(max_retries):
        try:
            os.replace(tmp_path, path)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(0.05 * (2 ** attempt))
            else:
                tmp_path.unlink(missing_ok=True)
                raise
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise


def write_prefill_shard(
    shard_path: Path,
    file_key: str,
    candidate_vowels: List[Dict[str, Any]],
    model_name: str,
) -> None:
    """Atomically write candidate vowels as 'mllm_breathy', purging stale prefilled vowels."""
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    # Preserve non-mllm annotations (e.g. auto_sp) if shard existed, purge old mllm_breathy
    preserved_annos: Dict[str, str] = {}
    if shard_path.exists():
        try:
            old_data = json.loads(shard_path.read_text(encoding="utf-8"))
            if isinstance(old_data, dict) and isinstance(old_data.get("annotations"), dict):
                for k, v in old_data["annotations"].items():
                    if v != "mllm_breathy":
                        preserved_annos[k] = v
        except Exception:
            pass

    new_annos = dict(preserved_annos)
    for v in candidate_vowels:
        iv_id = v["interval_id"]
        new_annos[iv_id] = "mllm_breathy"

    payload: Dict[str, Any] = {
        "version": 1,
        "schema": "direct_creaky_annotation",
        "file_key": file_key,
        "annotations": new_annos,
        "prefilled_by": model_name,
        "prefilled_at": time.time(),
    }
    _safe_atomic_write_json(shard_path, payload)


def clean_prefill_shard_if_modal(shard_path: Path) -> None:
    """Remove stale MLLM prefill shard if audio is inferred as purely modal."""
    if not shard_path.exists():
        return
    if not is_human_annotated(shard_path):
        try:
            shard_path.unlink(missing_ok=True)
        except Exception:
            pass


def update_batch_manifest(manifest_path: Path) -> None:
    """Update batch manifest manual_breathy_annotations.json updated timestamp."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "schema": "direct_creaky_annotation",
        "storage_mode": "sharded_json_v2",
        "updated_unix_time": time.time(),
    }
    _safe_atomic_write_json(manifest_path, payload)


def save_mllm_cache(cache_path: Path, record: Dict[str, Any]) -> None:
    """Atomically persist MLLM inference and alignment result."""
    _safe_atomic_write_json(cache_path, record)


def record_failure(
    failed_path: Path,
    file_key: str,
    batch_id: int,
    error_msg: str,
) -> None:
    """Append or update failed file record in failed_files.json."""
    failed_path.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    if failed_path.exists():
        try:
            data = json.loads(failed_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                records = data
        except Exception:
            records = []

    records = [r for r in records if r.get("file_key") != file_key]
    records.append({
        "file_key": file_key,
        "batch_id": batch_id,
        "error": error_msg,
        "timestamp": datetime.datetime.now().isoformat(),
    })
    _safe_atomic_write_json(failed_path, records)


# ── MLLM API and Parsing Core ──────────────────────────────────────────────────

class GeminiSafetyBlockedError(Exception):
    """Raised when Gemini content or safety filters block a prompt/audio."""
    pass


def encode_audio_base64(path: Path) -> Tuple[str, str]:
    """Base64 encode audio file and return (data, format)."""
    ext = path.suffix.lower()
    if ext == ".flac":
        fmt = "flac"
    elif ext == ".mp3":
        fmt = "mp3"
    elif ext == ".ogg":
        fmt = "ogg"
    else:
        fmt = "wav"

    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return b64, fmt


def call_gemini_api(
    audio_path: Path,
    endpoint: str,
    api_key: str,
    model: str,
    timeout: int = 90,
) -> Dict[str, Any]:
    """Call Google Gemini v1beta endpoint via local proxy.
    
    Strict Zero Fallback: Raises error on any HTTP error, empty candidates, or timeout.
    """
    b64_audio, fmt = encode_audio_base64(audio_path)
    url = f"{endpoint.rstrip('/')}/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_INSTRUCTION}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": USER_PROMPT_TEMPLATE},
                    {
                        "inline_data": {
                            "mime_type": f"audio/{fmt}",
                            "data": b64_audio,
                        }
                    },
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.1,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "x-goog-api-key": api_key,
    }

    r = requests.post(url, json=payload, headers=headers, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")

    res_data = r.json()
    pf = res_data.get("promptFeedback", {})
    if pf.get("blockReason"):
        raise GeminiSafetyBlockedError(f"Prompt blocked by Gemini: {pf.get('blockReason')}")

    candidates = res_data.get("candidates", [])
    if not candidates:
        raise ValueError(f"Empty candidates in response: {json.dumps(res_data, ensure_ascii=False)[:300]}")

    c0 = candidates[0]
    if c0.get("finishReason") in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"):
        raise GeminiSafetyBlockedError(f"Candidate blocked with finishReason: {c0.get('finishReason')}")

    parts = c0.get("content", {}).get("parts", [])
    thoughts = [p.get("text", "") for p in parts if p.get("thought", False)]
    non_thoughts = [p.get("text", "") for p in parts if not p.get("thought", False)]

    content = "".join(non_thoughts) if non_thoughts else "".join([p.get("text", "") for p in parts])
    raw_thought = "".join(thoughts)

    if "blocked by gemini's filters" in content.lower():
        raise GeminiSafetyBlockedError(f"Response text indicates filter block: {content[:200]}")

    return {
        "status": "success",
        "model_used": model,
        "raw_content": content,
        "raw_thought": raw_thought,
        "usage": res_data.get("usageMetadata", {}),
    }


def parse_response_json(raw_text: str) -> Dict[str, Any]:
    """Extract structured JSON from model response text."""
    # Pattern 1: Fenced json code block
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", raw_text, re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Pattern 2: Outer braces
    m_brace = re.search(r"(\{[\s\S]*\})", raw_text)
    if m_brace:
        try:
            return json.loads(m_brace.group(1))
        except Exception:
            pass

    raise ValueError(f"Unable to parse valid JSON from text: {raw_text[:200]}")


# ── Batch Runner Engine ───────────────────────────────────────────────────────

class BatchPrefillRunner:
    """Manages state, execution, retries, and persistence for batch prefill."""

    def __init__(
        self,
        corpus_key: str,
        target_batches: List[int],
        audio_dir: Path,
        textgrid_dir: Path,
        cache_dir: Path,
        sampling_state_path: Path,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key: str = DEFAULT_KEY,
        model: str = DEFAULT_MODEL,
        vowel_coverage: float = DEFAULT_VOWEL_COVERAGE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        log_interval: int = DEFAULT_LOG_INTERVAL,
        concurrency: int = 2,
        dry_run: bool = False,
        force: bool = False,
        cache_only: bool = False,
    ):
        self.corpus_key = corpus_key
        self.target_batches = target_batches
        self.audio_dir = audio_dir
        self.textgrid_dir = textgrid_dir
        self.cache_dir = cache_dir
        self.sampling_state_path = sampling_state_path

        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.vowel_coverage = vowel_coverage
        self.max_retries = max_retries
        self.log_interval = log_interval
        self.concurrency = max(1, concurrency)
        self.dry_run = dry_run
        self.force = force
        self.cache_only = cache_only
        self.lock = threading.Lock()

        # Derived paths
        self.mllm_cache_dir = self.cache_dir / "mllm_cache"
        self.failed_files_path = self.mllm_cache_dir / "failed_files.json"
        self.log_dir = self.mllm_cache_dir / "logs"

        self.logger = setup_logger(self.log_dir, corpus_key)

        # Statistics counters
        self.total_examined = 0
        self.skipped_human = 0
        self.skipped_missing_input = 0
        self.processed_count = 0
        self.cached_hits = 0
        self.api_calls = 0
        self.prefilled_breathy_files = 0
        self.prefilled_vowels_count = 0
        self.modal_files_count = 0
        self.failed_count = 0

    def load_target_files(self) -> List[Dict[str, Any]]:
        """Load list of files in the target batches from sampling_state.json."""
        if not self.sampling_state_path.exists():
            raise FileNotFoundError(f"Sampling state file not found: {self.sampling_state_path}")

        data = json.loads(self.sampling_state_path.read_text(encoding="utf-8"))
        batches = data.get("batches", [])

        target_set = set(self.target_batches)
        selected_files: List[Dict[str, Any]] = []

        for b in batches:
            b_id = b.get("batch_id")
            if b_id in target_set:
                b_name = b.get("name", f"Batch {b_id}")
                for f_info in b.get("files", []):
                    selected_files.append({
                        "file_key": f_info["file_key"],
                        "batch_id": b_id,
                        "batch_name": b_name,
                        "speaker_id": f_info.get("speaker_id", ""),
                        "gender": f_info.get("gender", ""),
                    })

        return selected_files

    def print_progress_box(self, current_idx: int, total_to_process: int, elapsed_sec: float) -> None:
        """Print structured anti-spam summary box."""
        rate = current_idx / elapsed_sec if elapsed_sec > 0 else 0
        eta_sec = (total_to_process - current_idx) / rate if rate > 0 else 0
        pct = (current_idx / total_to_process * 100) if total_to_process > 0 else 100

        def fmt_time(s: float) -> str:
            m, sec = divmod(int(s), 60)
            h, m = divmod(m, 60)
            return f"{h:02d}:{m:02d}:{sec:02d}"

        box = [
            "\n" + "=" * 65,
            f" [PROGRESS] {current_idx} / {total_to_process} files processed ({pct:.1f}%)",
            f"   Elapsed: {fmt_time(elapsed_sec)} | ETA: {fmt_time(eta_sec)} | Avg: {1.0 / rate:.1f}s/file" if rate > 0 else f"   Elapsed: {fmt_time(elapsed_sec)}",
            f"   Prefilled (Breathy): {self.prefilled_breathy_files} files ({self.prefilled_vowels_count} vowels)",
            f"   Pure Modal (Clean):  {self.modal_files_count} files",
            f"   Cache Hits:          {self.cached_hits} files",
            f"   API Calls Made:      {self.api_calls}",
            f"   Failures (Skipped):  {self.failed_count} files",
            "=" * 65 + "\n",
        ]
        self.logger.info("\n".join(box))

    def run(self) -> None:
        """Execute prefill pipeline across targeted files."""
        global _SHUTDOWN_REQUESTED

        self.logger.info("=" * 65)
        self.logger.info(f" MLLM BATCH PREFILL PIPELINE - Model: {self.model}")
        self.logger.info(f" Corpus: {self.corpus_key.upper()} | Batches: {self.target_batches}")
        self.logger.info(f" Endpoint: {self.endpoint}")
        self.logger.info(f" Dry Run Mode: {self.dry_run} | Force Re-infer: {self.force}")
        self.logger.info("=" * 65)

        all_files = self.load_target_files()
        self.logger.info(f"Total files in selected batches: {len(all_files)}")

        # Step 1: Pre-filter files to protect human annotations
        files_to_process: List[Dict[str, Any]] = []
        for item in all_files:
            self.total_examined += 1
            fk = item["file_key"]
            b_id = item["batch_id"]
            batch_dir = self.cache_dir / "batches" / f"batch_{b_id}"
            shard_path = compute_shard_path(batch_dir, fk)

            if is_human_annotated(shard_path):
                self.skipped_human += 1
                self.logger.debug(f"Skip human-annotated file: {fk} (batch {b_id})")
                continue

            audio_path = find_audio_path(self.audio_dir, fk)
            tg_path = find_textgrid_path(self.textgrid_dir, fk)

            if not audio_path or not tg_path:
                self.skipped_missing_input += 1
                self.logger.warning(f"Skip file with missing audio/TextGrid: {fk} (audio={audio_path}, tg={tg_path})")
                continue

            item["audio_path"] = audio_path
            item["textgrid_path"] = tg_path
            item["batch_dir"] = batch_dir
            item["shard_path"] = shard_path
            item["cache_file"] = self.mllm_cache_dir / f"{fk}.json"
            files_to_process.append(item)

        self.logger.info(
            f"\nHuman Protection Check:\n"
            f"  - Protected Human Files:  {self.skipped_human}\n"
            f"  - Missing Audio/TextGrid: {self.skipped_missing_input}\n"
            f"  - Eligible for Prefill:   {len(files_to_process)}\n"
        )

        if self.dry_run:
            self.logger.info("[DRY RUN] Verification completed successfully. No network calls or disk writes made.")
            return

        if not files_to_process:
            self.logger.info("No eligible files remaining to process. Exiting.")
            return

    def process_single_file(
        self,
        item: Dict[str, Any],
        total_files: int,
        start_time: float,
        touched_batches: Set[int],
    ) -> None:
        """Process a single audio file through inference, alignment, and shard writing."""
        global _SHUTDOWN_REQUESTED
        if _SHUTDOWN_REQUESTED:
            return

        fk = item["file_key"]
        b_id = item["batch_id"]
        audio_path: Path = item["audio_path"]
        tg_path: Path = item["textgrid_path"]
        shard_path: Path = item["shard_path"]
        cache_file: Path = item["cache_file"]

        file_start = time.time()
        prediction_record: Optional[Dict[str, Any]] = None

        # Check cache first unless --force
        if cache_file.exists() and not self.force:
            try:
                cached_data = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached_data, dict) and "parsed_prediction" in cached_data:
                    prediction_record = cached_data
                    with self.lock:
                        self.cached_hits += 1
            except Exception as e:
                self.logger.warning(f"Cache file corrupted for {fk}, will re-infer: {e}")

        # If not in cache, invoke Gemini API with 3x retry
        if prediction_record is None:
            if self.cache_only:
                self.logger.info(f"[{fk}] Skipped in cache-only mode (no cached inference).")
                return

            backoffs = [2.0, 4.0, 8.0]
            call_success = False

            for attempt in range(self.max_retries + 1):
                if _SHUTDOWN_REQUESTED:
                    break

                try:
                    with self.lock:
                        self.api_calls += 1
                    api_res = call_gemini_api(
                        audio_path,
                        self.endpoint,
                        self.api_key,
                        self.model,
                        timeout=90,
                    )
                    parsed_pred = parse_response_json(api_res["raw_content"])
                    latency = round(time.time() - file_start, 2)

                    prediction_record = {
                        "version": 1,
                        "file_key": fk,
                        "audio_path": str(audio_path),
                        "textgrid_path": str(tg_path),
                        "model": self.model,
                        "created_at": datetime.datetime.now().isoformat(),
                        "latency_sec": latency,
                        "raw_thought": api_res.get("raw_thought", ""),
                        "raw_content": api_res.get("raw_content", ""),
                        "parsed_prediction": parsed_pred,
                    }
                    call_success = True
                    break

                except Exception as err:
                    err_str = str(err).lower()
                    is_safety_block = isinstance(err, GeminiSafetyBlockedError) or any(
                        k in err_str for k in [
                            "blocked by gemini's filters",
                            "blockreason",
                            "finishreason: safety",
                            "finishreason: blocklist",
                            "prohibited_content",
                        ]
                    )
                    if is_safety_block:
                        with self.lock:
                            self.failed_count += 1
                            record_failure(self.failed_files_path, fk, b_id, f"Gemini filter blocked: {err}")
                        self.logger.warning(
                            f"[{fk}] Request blocked by Gemini filters. Skipping immediately to protect account."
                        )
                        break

                    if attempt < self.max_retries and not _SHUTDOWN_REQUESTED:
                        wait_sec = backoffs[attempt] if attempt < len(backoffs) else 8.0
                        self.logger.warning(
                            f"[{fk}] Attempt {attempt + 1}/{self.max_retries + 1} failed: {err}. "
                            f"Retrying in {wait_sec}s..."
                        )
                        time.sleep(wait_sec)
                    else:
                        with self.lock:
                            self.failed_count += 1
                            record_failure(self.failed_files_path, fk, b_id, str(err))
                        self.logger.error(
                            f"[{fk}] All {self.max_retries + 1} attempts failed! Error: {err}. Skipping file."
                        )

            if not call_success or prediction_record is None:
                return

        # Step 3: Dual-constraint alignment & micro vowel projection
        try:
            parsed_pred = prediction_record["parsed_prediction"]
            align_res = project_mllm_prediction_to_vowels(
                tg_path,
                parsed_pred,
                vowel_coverage_threshold=self.vowel_coverage,
            )
            prediction_record["alignment_result"] = align_res

            # Persist to MLLM cache
            save_mllm_cache(cache_file, prediction_record)

            candidate_vowels = align_res.get("candidate_vowels", [])
            is_uncertain = (
                parsed_pred.get("has_breathy") == "uncertain"
                or any(s.get("phonation_type") == "uncertain" for s in parsed_pred.get("segments", []))
            )

            if candidate_vowels:
                write_prefill_shard(shard_path, fk, candidate_vowels, self.model)
                status_str = f"PREFILLED ({len(candidate_vowels)} vowels)"
            else:
                clean_prefill_shard_if_modal(shard_path)
                status_str = "UNCERTAIN (clean)" if is_uncertain else "MODAL (clean)"

            dur = time.time() - file_start

            with self.lock:
                if candidate_vowels:
                    self.prefilled_breathy_files += 1
                    self.prefilled_vowels_count += len(candidate_vowels)
                else:
                    self.modal_files_count += 1

                touched_batches.add(b_id)
                self.processed_count += 1
                curr_cnt = self.processed_count

                self.logger.info(
                    f"[{curr_cnt}/{total_files}] {fk} (B{b_id}) -> {status_str} [{dur:.1f}s]"
                )
                if curr_cnt % self.log_interval == 0:
                    self.print_progress_box(curr_cnt, total_files, time.time() - start_time)

        except Exception as e:
            with self.lock:
                self.failed_count += 1
                record_failure(self.failed_files_path, fk, b_id, f"Alignment error: {e}")
            self.logger.error(f"[{fk}] Alignment/Storage error: {e}")

    def run(self) -> None:
        """Execute prefill pipeline across targeted files."""
        global _SHUTDOWN_REQUESTED

        self.logger.info("=" * 65)
        self.logger.info(f" MLLM BATCH PREFILL PIPELINE - Model: {self.model}")
        self.logger.info(f" Corpus: {self.corpus_key.upper()} | Batches: {self.target_batches}")
        self.logger.info(f" Endpoint: {self.endpoint}")
        self.logger.info(f" Concurrency: {self.concurrency} threads | Dry Run: {self.dry_run} | Force: {self.force}")
        self.logger.info("=" * 65)

        all_files = self.load_target_files()
        self.logger.info(f"Total files in selected batches: {len(all_files)}")

        # Step 1: Pre-filter files to protect human annotations
        files_to_process: List[Dict[str, Any]] = []
        for item in all_files:
            self.total_examined += 1
            fk = item["file_key"]
            b_id = item["batch_id"]
            batch_dir = self.cache_dir / "batches" / f"batch_{b_id}"
            shard_path = compute_shard_path(batch_dir, fk)

            if is_human_annotated(shard_path):
                self.skipped_human += 1
                self.logger.debug(f"Skip human-annotated file: {fk} (batch {b_id})")
                continue

            audio_path = find_audio_path(self.audio_dir, fk)
            tg_path = find_textgrid_path(self.textgrid_dir, fk)

            if not audio_path or not tg_path:
                self.skipped_missing_input += 1
                self.logger.warning(f"Skip file with missing audio/TextGrid: {fk} (audio={audio_path}, tg={tg_path})")
                continue

            item["audio_path"] = audio_path
            item["textgrid_path"] = tg_path
            item["batch_dir"] = batch_dir
            item["shard_path"] = shard_path
            item["cache_file"] = self.mllm_cache_dir / f"{fk}.json"
            files_to_process.append(item)

        self.logger.info(
            f"\nHuman Protection Check:\n"
            f"  - Protected Human Files:  {self.skipped_human}\n"
            f"  - Missing Audio/TextGrid: {self.skipped_missing_input}\n"
            f"  - Eligible for Prefill:   {len(files_to_process)}\n"
        )

        if self.dry_run:
            self.logger.info("[DRY RUN] Verification completed successfully. No network calls or disk writes made.")
            return

        if not files_to_process:
            self.logger.info("No eligible files remaining to process. Exiting.")
            return

        # Step 2: Main Processing Loop (Sequential or ThreadPool concurrent)
        start_time = time.time()
        touched_batches: Set[int] = set()
        total_to_process = len(files_to_process)

        try:
            if self.concurrency > 1:
                self.logger.info(f"[Concurrency] Starting {self.concurrency} concurrent worker threads...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                    futures = [
                        executor.submit(
                            self.process_single_file, item, total_to_process, start_time, touched_batches
                        )
                        for item in files_to_process
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        if _SHUTDOWN_REQUESTED:
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        try:
                            future.result()
                        except Exception as e:
                            self.logger.error(f"Worker exception: {e}")
            else:
                for item in files_to_process:
                    if _SHUTDOWN_REQUESTED:
                        break
                    self.process_single_file(item, total_to_process, start_time, touched_batches)

        finally:
            # Step 5: Always finalize touched batch manifests, even on interrupt or error
            for b_id in touched_batches:
                manifest_path = self.cache_dir / "batches" / f"batch_{b_id}" / "manual_breathy_annotations.json"
                try:
                    update_batch_manifest(manifest_path)
                except Exception as e:
                    self.logger.error(f"Failed to update manifest for batch {b_id}: {e}")

        total_elapsed = time.time() - start_time
        self.logger.info("\n" + "=" * 65)
        self.logger.info(" BATCH PREFILL RUN COMPLETED" if not _SHUTDOWN_REQUESTED else " BATCH PREFILL RUN PAUSED")
        self.print_progress_box(self.processed_count, total_to_process, total_elapsed)



# ── Interactive CLI Selection ──────────────────────────────────────────────────

def parse_batch_selection(batch_input: str, available_ids: List[int]) -> List[int]:
    """Parse comma-separated or keyword batch string."""
    batch_input = batch_input.strip().lower()
    if batch_input in ("all", "a"):
        return sorted(available_ids)

    selected: List[int] = []
    for part in batch_input.split(","):
        part = part.strip()
        part = re.sub(r"^batch_", "", part)
        if part.isdigit():
            val = int(part)
            if val in available_ids:
                selected.append(val)
    return sorted(list(set(selected)))


def prompt_user_for_batches(sampling_state_path: Path) -> List[int]:
    """Prompt user interactively to select which batches to process."""
    if not sampling_state_path.exists():
        return [3]

    try:
        data = json.loads(sampling_state_path.read_text(encoding="utf-8"))
        batches = data.get("batches", [])
        avail_ids = [b["batch_id"] for b in batches]

        print("\n" + "=" * 60)
        print(" Available Batches:")
        for b in batches:
            b_id = b["batch_id"]
            b_name = b.get("name", f"Batch {b_id}")
            f_count = len(b.get("files", []))
            print(f"   [{b_id}] {b_name} ({f_count} files)")
        print("   [A] All batches")
        print("=" * 60)

        raw = input(" Select batch ID(s) (e.g. 3 or 3,4 or A) [default: 3]: ").strip()
        if not raw:
            return [3]
        parsed = parse_batch_selection(raw, avail_ids)
        return parsed if parsed else [3]
    except Exception:
        return [3]


# ── Main Entrypoint ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Production MLLM Breathy Voice Batch Coarse-Annotation Prefill",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", choices=list(CORPUS_PRESETS.keys()), default="ysj", help="Corpus preset to use")
    parser.add_argument("--batch", type=str, default=None, help="Batch ID(s) to process (e.g. '3', '3,4', 'all')")
    parser.add_argument("--endpoint", type=str, default=DEFAULT_ENDPOINT, help="Gemini API base endpoint")
    parser.add_argument("--api-key", type=str, default=DEFAULT_KEY, help="API Key for endpoint")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model identifier")
    parser.add_argument("--vowel-coverage", type=float, default=DEFAULT_VOWEL_COVERAGE, help="Vowel overlap coverage threshold")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Maximum retry attempts on failure")
    parser.add_argument("--log-interval", type=int, default=DEFAULT_LOG_INTERVAL, help="Progress summary logging frequency")
    parser.add_argument("--concurrency", type=int, default=2, help="Number of concurrent worker threads (default: 2)")
    parser.add_argument("--dry-run", action="store_true", help="Inspect and simulate files without making API calls or shard modifications")
    parser.add_argument("--force", action="store_true", help="Force re-inference even if file already exists in MLLM cache")
    parser.add_argument("--cache-only", action="store_true", help="Only re-align and prefill files that already exist in MLLM cache (0 API calls, 0 token cost)")

    # Path overrides
    parser.add_argument("--audio-dir", type=Path, default=None, help="Custom audio directory")
    parser.add_argument("--textgrid-dir", type=Path, default=None, help="Custom TextGrid directory")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Custom cache directory")
    parser.add_argument("--sampling-state", type=Path, default=None, help="Custom sampling_state.json path")

    args = parser.parse_args()

    # Load corpus preset defaults
    preset = CORPUS_PRESETS[args.corpus]
    audio_dir = args.audio_dir or preset["audio_dir"]
    textgrid_dir = args.textgrid_dir or preset["textgrid_dir"]
    cache_dir = args.cache_dir or preset["cache_dir"]
    sampling_state_path = args.sampling_state or preset["sampling_state"]

    # Resolve target batches
    if args.batch is not None:
        if sampling_state_path.exists():
            data = json.loads(sampling_state_path.read_text(encoding="utf-8"))
            avail_ids = [b["batch_id"] for b in data.get("batches", [])]
        else:
            avail_ids = [1, 2, 3, 4, 5, 6]
        target_batches = parse_batch_selection(args.batch, avail_ids)
        if not target_batches:
            print(f"[ERROR] Invalid batch selection '{args.batch}'. Available: {avail_ids}", file=sys.stderr)
            sys.exit(1)
    else:
        # Prompt interactively if stdin is tty
        if sys.stdin.isatty():
            target_batches = prompt_user_for_batches(sampling_state_path)
        else:
            target_batches = [3]  # Default to active batch 3 in non-interactive mode

    runner = BatchPrefillRunner(
        corpus_key=args.corpus,
        target_batches=target_batches,
        audio_dir=audio_dir,
        textgrid_dir=textgrid_dir,
        cache_dir=cache_dir,
        sampling_state_path=sampling_state_path,
        endpoint=args.endpoint,
        api_key=args.api_key,
        model=args.model,
        vowel_coverage=args.vowel_coverage,
        max_retries=args.max_retries,
        log_interval=args.log_interval,
        concurrency=args.concurrency,
        dry_run=args.dry_run,
        force=args.force,
        cache_only=args.cache_only,
    )

    runner.run()


if __name__ == "__main__":
    main()
