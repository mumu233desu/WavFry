#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acoustic validator for synthesized Japanese breathy speech.

Measures F0 periodicity, CPPS (Cepstral Peak Prominence Smoothed), and
spectral tilt to quantitatively verify that breathy intervals exhibit real
glottal aspiration while maintaining voicing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

logger = logging.getLogger("acoustic_validator")


def compute_frame_cpps(signal: np.ndarray, sr: int, frame_len: int = 1024, hop_len: int = 256) -> float:
    """Compute smoothed Cepstral Peak Prominence (CPPS)."""
    if len(signal) < frame_len:
        return float("nan")

    # Framing with Hanning window
    num_frames = (len(signal) - frame_len) // hop_len + 1
    if num_frames <= 0:
        return float("nan")

    cpps_values = []
    window = np.hanning(frame_len)
    quef_min = int(sr / 500.0)  # max F0 ~ 500Hz
    quef_max = int(sr / 60.0)   # min F0 ~ 60Hz

    for i in range(num_frames):
        frame = signal[i * hop_len : i * hop_len + frame_len] * window
        # Magnitude spectrum
        mag = np.abs(np.fft.rfft(frame))
        log_mag = np.log(mag + 1e-12)
        # Real cepstrum
        cepstrum = np.fft.irfft(log_mag)

        if len(cepstrum) <= quef_max:
            continue

        search_region = cepstrum[quef_min:quef_max]
        if len(search_region) == 0:
            continue

        peak_val = np.max(search_region)
        # Linear regression baseline
        x = np.arange(quef_min, quef_max)
        poly = np.polyfit(x, search_region, deg=1)
        peak_idx = np.argmax(search_region) + quef_min
        baseline_val = np.polyval(poly, peak_idx)

        cpp = peak_val - baseline_val
        cpps_values.append(cpp)

    if not cpps_values:
        return float("nan")
    return float(np.mean(cpps_values))


def compute_spectral_tilt_and_voicing(signal: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute ratio of high-freq aspiration energy (>2kHz) to low-freq energy (<1kHz),
    and autocorrelation-based voicing fraction.
    """
    if len(signal) < 512:
        return {"high_low_ratio": float("nan"), "voicing_score": float("nan")}

    # FFT power spectrum
    fft_vals = np.abs(np.fft.rfft(signal))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)

    low_mask = (freqs >= 80) & (freqs <= 1000)
    high_mask = (freqs >= 2000) & (freqs <= 8000)

    low_energy = np.sum(fft_vals[low_mask] ** 2) + 1e-12
    high_energy = np.sum(fft_vals[high_mask] ** 2) + 1e-12

    high_low_ratio = 10.0 * np.log10(high_energy / low_energy)

    # Autocorrelation for voicing
    corr = np.correlate(signal, signal, mode="full")
    corr = corr[len(signal) - 1 :]
    if corr[0] > 0:
        corr_norm = corr / corr[0]
        # Search peak in human F0 range (60Hz to 400Hz)
        lag_min = int(sr / 400.0)
        lag_max = int(sr / 60.0)
        if len(corr_norm) > lag_max:
            voicing_peak = np.max(corr_norm[lag_min:lag_max])
        else:
            voicing_peak = float("nan")
    else:
        voicing_peak = float("nan")

    return {
        "high_low_ratio_db": float(high_low_ratio),
        "voicing_peak": float(voicing_peak),
    }


def validate_sample_acoustics(
    wav_path: Path,
    interval_records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze acoustic differences between modal and breathy vowel intervals."""
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    modal_cpps, breathy_cpps = [], []
    modal_hl, breathy_hl = [], []
    modal_v, breathy_v = [], []

    for r in interval_records:
        start_idx = int(r["start_s"] * sr)
        end_idx = int(r["end_s"] * sr)
        chunk = audio[start_idx:end_idx]

        if len(chunk) < 512:
            continue

        cpps = compute_frame_cpps(chunk, sr)
        spec_info = compute_spectral_tilt_and_voicing(chunk, sr)

        is_breathy = (r.get("breathy") == 1.0) or (r.get("label_breathy") == 1.0)
        if is_breathy:
            if np.isfinite(cpps):
                breathy_cpps.append(cpps)
            if np.isfinite(spec_info["high_low_ratio_db"]):
                breathy_hl.append(spec_info["high_low_ratio_db"])
            if np.isfinite(spec_info["voicing_peak"]):
                breathy_v.append(spec_info["voicing_peak"])
        else:
            if np.isfinite(cpps):
                modal_cpps.append(cpps)
            if np.isfinite(spec_info["high_low_ratio_db"]):
                modal_hl.append(spec_info["high_low_ratio_db"])
            if np.isfinite(spec_info["voicing_peak"]):
                modal_v.append(spec_info["voicing_peak"])

    return {
        "wav_file": wav_path.name,
        "modal_count": len(modal_cpps),
        "breathy_count": len(breathy_cpps),
        "modal_avg_cpps": round(float(np.mean(modal_cpps)), 3) if modal_cpps else None,
        "breathy_avg_cpps": round(float(np.mean(breathy_cpps)), 3) if breathy_cpps else None,
        "modal_avg_hl_db": round(float(np.mean(modal_hl)), 2) if modal_hl else None,
        "breathy_avg_hl_db": round(float(np.mean(breathy_hl)), 2) if breathy_hl else None,
        "modal_avg_voicing": round(float(np.mean(modal_v)), 3) if modal_v else None,
        "breathy_avg_voicing": round(float(np.mean(breathy_v)), 3) if breathy_v else None,
    }


def generate_validation_summary(results: List[Dict[str, Any]]) -> str:
    """Format acoustic verification report in Markdown."""
    lines = [
        "# Gemini TTS 气声声学特征验证汇总报告\n",
        "| 样本音频 | 模态 | 元音数 | 平均 CPPS (dB) | 高频/低频能量比 (dB) | 周期性峰值 (Voicing) | 物理判定状态 |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in results:
        w = r["wav_file"]
        bv = r.get("breathy_avg_voicing") or 0.0
        if bv < 0.55:
            verdict = "⚠️ **耳语化倾向 (Whispery Warning, 基频偏弱)**"
        else:
            verdict = "✅ **标准气声 (Valid Breathy, 声带周期振动完整)**"

        # Modal row
        lines.append(
            f"| `{w}` | **Modal** | {r['modal_count']} | {r['modal_avg_cpps']} | {r['modal_avg_hl_db']} | {r['modal_avg_voicing']} | 常态基线 |"
        )
        # Breathy row
        lines.append(
            f"| | **Breathy** | {r['breathy_count']} | {r['breathy_avg_cpps']} | {r['breathy_avg_hl_db']} | {r['breathy_avg_voicing']} | {verdict} |"
        )

    lines.append("\n### 声学学理解读准则：")
    lines.append("- **CPPS (倒谱峰值突出度)**：气声音频由于声门湍流噪声增加，其 CPPS 应**显著低于**常态音。")
    lines.append("- **高低频能量比 (High/Low Ratio)**：气声伴随高频漏气湍流噪声，高频能量占比较常态音通常**更高**。")
    lines.append("- **周期性峰值 (Voicing)**：气声必须维持明确的声带基频振动（周期性峰值 >= 0.55），若低于此阈值代表退化为完全耳语（Whispery Voice）。")

    return "\n".join(lines)
