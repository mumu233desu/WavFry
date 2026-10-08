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
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger("gemini_tts_client")

DEFAULT_TTS_ENDPOINT = "https://ai.chuyel.top"
DEFAULT_TTS_KEY = "sk-6Wowlpd3veS2iz43DUfIm67DWQtP6teyMenjrb8sjJoxXO8h"
DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"


class GeminiTTSClient:
    """Robust client for Gemini 3.8 Flash TTS."""

    def __init__(
        self,
        endpoint: str = DEFAULT_TTS_ENDPOINT,
        api_key: str = DEFAULT_TTS_KEY,
        model: str = DEFAULT_TTS_MODEL,
        max_retries: int = 5,
        base_backoff_sec: float = 6.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.session = requests.Session()

    def synthesize(
        self,
        parts: List[Dict[str, Any]],
        voice: str,
        output_wav_path: Path,
    ) -> Dict[str, Any]:
        """Synthesize multi-part styled speech into WAV file.

        Args:
            parts: List of dicts, each with 'text' and 'speech_metadata'.
            voice: Voice name (e.g. 'Kore', 'Aoede', 'Puck', 'Fenrir').
            output_wav_path: Destination .wav file path.

        Returns:
            Metadata dict including audio duration, sample rate, etc.
        """
        url = f"{self.endpoint}/v1beta/models/{self.model}:generateContent"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

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
                    output_wav_path.parent.mkdir(parents=True, exist_ok=True)

                    # Save and normalize audio
                    duration_s, sr = self._save_audio(audio_bytes, output_wav_path)
                    logger.info(f"Synthesized {output_wav_path.name}: {duration_s:.2f}s, {sr}Hz")

                    return {
                        "status": "success",
                        "audio_path": str(output_wav_path),
                        "duration_s": duration_s,
                        "sample_rate": sr,
                        "voice": voice,
                        "mime_type": mime_type,
                    }

                elif resp.status_code in (429, 400, 502, 503, 504):
                    err_msg = resp.text[:150].replace("\n", " ")
                    logger.warning(f"HTTP {resp.status_code} ({err_msg}). Backing off...")
                    sleep_time = self.base_backoff_sec * (2 ** (attempt - 1)) + random.uniform(1.0, 4.0)
                    time.sleep(sleep_time)
                else:
                    resp.raise_for_status()

            except Exception as e:
                last_error = e
                logger.warning(f"TTS call error on attempt {attempt}: {e}")
                sleep_time = self.base_backoff_sec * (2 ** (attempt - 1)) + random.uniform(1.0, 3.0)
                time.sleep(sleep_time)

        raise RuntimeError(f"Failed to synthesize audio after {self.max_retries} attempts: {last_error}")

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
