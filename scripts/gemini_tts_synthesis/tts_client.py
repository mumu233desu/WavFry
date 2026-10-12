#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gemini TTS client for Japanese speech generation.

Interacts with gemini-3.8-flash-tts at https://ai.chuyel.top via Bearer auth,
handles exponential backoff on 429/upstream rate limits, and saves 24kHz WAV.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
from requests.adapters import HTTPAdapter
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger("gemini_tts_client")

DEFAULT_TTS_ENDPOINT = "https://openrouter.ai/api/v1"
DEFAULT_TTS_KEY = "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"
DEFAULT_TTS_MODEL = "google/gemini-3.8-flash-tts"


class GeminiTTSClient:
    """Robust client for Gemini 3.8 Flash TTS."""

    def __init__(
        self,
        endpoint: str = DEFAULT_TTS_ENDPOINT,
        api_key: str = DEFAULT_TTS_KEY,
        model: str = DEFAULT_TTS_MODEL,
        proxy: Optional[str] = None,
        max_retries: int = 10,
        base_backoff_sec: float = 5.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.session = requests.Session()

        # Optimize connection pooling for high-concurrency requests
        adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

        if proxy is not None:
            self.session.proxies = {"http": proxy, "https": proxy}
        elif not os.environ.get("HTTPS_PROXY") and not os.environ.get("HTTP_PROXY"):
            # If accessing Google AI Studio directly from mainland without env proxy, default to local proxy
            if "googleapis.com" in self.endpoint:
                self.session.proxies = {
                    "http": "http://127.0.0.1:10808",
                    "https": "http://127.0.0.1:10808",
                }

    def synthesize(
        self,
        parts: List[Dict[str, Any]],
        voice: str,
        output_wav_path: Path,
        instructions_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize multi-part styled speech into WAV file.

        Args:
            parts: List of dicts, each with 'text' and 'speech_metadata'.
            voice: Voice name (e.g. 'Kore', 'Aoede', 'Puck', 'Fenrir').
            output_wav_path: Destination .wav file path.
            instructions_override: Optional custom instructions for OpenRouter API.

        Returns:
            Metadata dict including audio duration, sample rate, etc.
        """
        is_openrouter = "openrouter.ai" in self.endpoint.lower()
        if is_openrouter:
            ep = self.endpoint.rstrip("/")
            url = f"{ep}/audio/speech" if ep.endswith("/v1") else f"{ep}/api/v1/audio/speech"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/mumu/WavFry",
            }
            modal_text = parts[0].get("text", "")
            modal_style = parts[0].get("speech_metadata", {}).get("style", "")
            breathy_text = parts[1].get("text", "") if len(parts) > 1 else ""
            breathy_style = parts[1].get("speech_metadata", {}).get("style", "") if len(parts) > 1 else ""

            full_input = f"{modal_text} {breathy_text}".strip()
            if instructions_override:
                instructions = instructions_override
            else:
                instructions = (
                    f"Section 1 ({modal_text}): {modal_style}. "
                    f"Section 2 ({breathy_text}): {breathy_style}. "
                    "CRITICAL: Maintain distinct pitch contour and periodic vocal fold vibration; strictly do NOT produce unvoiced whisper."
                )
            payload = {
                "model": self.model,
                "input": full_input,
                "voice": voice,
                "instructions": instructions,
                "response_format": "pcm",
            }
        else:
            url = f"{self.endpoint}/v1beta/models/{self.model}:generateContent"
            headers = {
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            }
            if self.api_key.startswith("sk-"):
                headers["Authorization"] = f"Bearer {self.api_key}"

            payload = {
                "contents": [
                    {
                        "role": "user",
                        "parts": parts,
                    }
                ],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": {
                        "voiceConfig": {
                            "voice": voice,
                        }
                    },
                },
            }

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"Invoking TTS API for voice '{voice}' (attempt {attempt}/{self.max_retries})...")
                resp = self.session.post(url, json=payload, headers=headers, timeout=60)

                if resp.status_code == 200:
                    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
                    if is_openrouter:
                        audio_bytes = resp.content
                        mime_type = "audio/pcm"
                    else:
                        data = resp.json()
                        candidates = data.get("candidates", [])
                        if not candidates:
                            raise ValueError(f"No candidates in Gemini response: {data}")

                        content_parts = candidates[0].get("content", {}).get("parts", [])
                        audio_b64 = None
                        mime_type = "audio/wav"

                        for p in content_parts:
                            if "inlineData" in p:
                                audio_b64 = p["inlineData"].get("data")
                                mime_type = p["inlineData"].get("mimeType", mime_type)
                                break
                            elif "inline_data" in p:
                                audio_b64 = p["inline_data"].get("data")
                                mime_type = p["inline_data"].get("mime_type", mime_type)
                                break

                        if not audio_b64:
                            raise ValueError(f"No inline audio data found in candidate parts: {content_parts}")

                        audio_bytes = base64.b64decode(audio_b64)

                    # Save and normalize audio to 24kHz mono WAV
                    duration_s, sr = self._save_audio(audio_bytes, output_wav_path)
                    gen_id = resp.headers.get("X-Generation-Id")
                    logger.info(f"Synthesized {output_wav_path.name}: {duration_s:.2f}s, {sr}Hz (GenID: {gen_id or 'none'})")

                    return {
                        "status": "success",
                        "audio_path": str(output_wav_path),
                        "duration_s": duration_s,
                        "sample_rate": sr,
                        "voice": voice,
                        "mime_type": mime_type,
                        "generation_id": gen_id,
                        "provider_headers": dict(resp.headers),
                    }

                elif resp.status_code in (429, 400, 502, 503, 504):
                    err_msg = resp.text[:150].replace("\n", " ")
                    sleep_time = min(35.0, self.base_backoff_sec * (1.5 ** (attempt - 1)) + random.uniform(1.0, 4.0))
                    logger.warning(f"HTTP {resp.status_code} ({err_msg}). Backing off for {sleep_time:.1f}s (attempt {attempt}/{self.max_retries})...")
                    time.sleep(sleep_time)
                else:
                    resp.raise_for_status()

            except Exception as e:
                last_error = e
                sleep_time = min(30.0, self.base_backoff_sec * (1.5 ** (attempt - 1)) + random.uniform(1.0, 3.0))
                logger.warning(f"TTS call error on attempt {attempt}: {e}. Retrying in {sleep_time:.1f}s...")
                time.sleep(sleep_time)

        raise RuntimeError(f"Failed to synthesize audio after {self.max_retries} attempts: {last_error}")

    def fetch_generation_stats(self, generation_id: str) -> Dict[str, Any]:
        """Fetch official prompt/completion tokens and cost from OpenRouter API."""
        if not generation_id or "openrouter.ai" not in self.endpoint.lower():
            return {}
        stats_url = f"https://openrouter.ai/api/v1/generation?id={generation_id}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
        }
        for attempt in range(4):
            try:
                resp = self.session.get(stats_url, headers=headers, timeout=10)
                if resp.status_code == 200:
                    d = resp.json().get("data", {})
                    return {
                        "tokens_prompt": d.get("tokens_prompt", 0),
                        "tokens_completion": d.get("tokens_completion", 0),
                        "total_cost": d.get("total_cost", 0.0),
                        "latency_ms": d.get("latency", 0),
                    }
                time.sleep(1.5)
            except Exception:
                time.sleep(1.5)
        return {}

    def _save_audio(self, raw_bytes: bytes, out_path: Path) -> tuple[float, int]:
        """Save raw bytes to 24000Hz mono WAV."""
        try:
            # Check if RIFF/WAV header is present
            if raw_bytes.startswith(b"RIFF"):
                wav_io = io.BytesIO(raw_bytes)
                data, sr = sf.read(wav_io)
            else:
                # Treat as raw 16-bit PCM at 24000Hz
                import numpy as np
                pcm16 = np.frombuffer(raw_bytes, dtype=np.int16)
                data = pcm16.astype(np.float32) / 32768.0
                sr = 24000

            # Ensure mono
            if data.ndim > 1:
                data = data.mean(axis=1)

            # Write standard WAV
            sf.write(str(out_path), data, sr)
            duration_s = len(data) / float(sr)
            return duration_s, sr
        except Exception as e:
            # Fallback direct binary write
            with open(out_path, "wb") as f:
                f.write(raw_bytes)
            info = sf.info(str(out_path))
            return info.duration, info.samplerate
