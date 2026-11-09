r"""
Audit and Quantify CREPE F0 Voicing Confidence Distributions across Phonetic Domains.

Domains:
1. MFA Pure Vowels (Mandarin monophthongs, diphthongs, apical vowels)
2. MFA Voiced Intervals (Vowels + Sonorants /m, n, ng, l, r, etc./)
3. Full Silero VAD Speech (Baseline containing voiceless consonants)

Stratified by:
- Modal Voice (WavFry P_creak < 0.560)
- Creaky Voice (WavFry P_creak >= 0.560)
- Combined

Usage:
  uv run python scripts/audit_voicing_f0_confidence.py --num-samples 60
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import torchcrepe
from silero_vad import load_silero_vad, get_speech_timestamps

# Setup logging
sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("audit_f0_conf")

DEFAULT_DATA_DIR = Path(r"E:\Corpus\zt\v2")
DEFAULT_PROBS_DIR = Path(r"E:\Corpus\zt\v2\wavfry_probs")
DEFAULT_TG_DIR = Path(r"E:\Corpus\zt\v2_textgrids")
DEFAULT_OUT_DIR = Path(r"E:\Corpus\zt\v2\visualizations")
OPTIMAL_THRESHOLD = 0.560
TARGET_SR = 16000

MFA_VOWEL_BASES = ("a", "o", "e", "i", "u", "y", "ə", "z̩", "ʐ̩")
MFA_SONORANT_BASES = ("m", "n", "ŋ", "l", "ɻ", "ʐ", "j", "w", "ɥ")


def is_vowel(p: str) -> bool:
    p = p.strip()
    return bool(p and p not in ("sil", "spn", "<eps>") and any(v in p for v in MFA_VOWEL_BASES))


def is_voiced(p: str) -> bool:
    p = p.strip()
    return bool(
        p
        and p not in ("sil", "spn", "<eps>")
        and (any(v in p for v in MFA_VOWEL_BASES) or any(s in p for s in MFA_SONORANT_BASES))
    )


def parse_tg(tg_path: Path) -> List[Tuple[float, float, str]]:
    phones = []
    if not tg_path.exists():
        return phones
    with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [l.strip() for l in f]
    tier = None
    i = 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("name ="):
            tier = l.split("=")[1].strip().strip('"')
        elif l.startswith("intervals [") and tier == "phones":
            try:
                xmin = float(lines[i + 1].split("=")[1])
                xmax = float(lines[i + 2].split("=")[1])
                txt = lines[i + 3].split("=")[1].strip().strip('"')
                phones.append((xmin, xmax, txt))
                i += 3
            except Exception:
                pass
        i += 1
    return phones


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=60)
    parser.add_argument("--threshold", type=float, default=OPTIMAL_THRESHOLD)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    vad_model = load_silero_vad()
    resampler_22k = torchaudio.transforms.Resample(22050, TARGET_SR)

    # Pick balanced samples across female and male speakers, T3 and non-T3
    sample_npzs = []
    spks = sorted([p.name for p in DEFAULT_DATA_DIR.glob("BJf*")] + [p.name for p in DEFAULT_DATA_DIR.glob("BJm*")])
    for spk in spks:
        s_npzs = sorted(list((DEFAULT_PROBS_DIR / spk).glob("*.npz")))
        t3 = [p for p in s_npzs if "T3" in p.name]
        other = [p for p in s_npzs if "T3" not in p.name]
        sample_npzs.extend(t3[:2])
        sample_npzs.extend(other[:2])
        if len(sample_npzs) >= args.num_samples:
            break

    sample_npzs = sample_npzs[:args.num_samples]
    logger.info(f"Selected {len(sample_npzs)} files for distribution audit.")

    conf_vowel_modal = []
    conf_vowel_creak = []
    conf_voiced_modal = []
    conf_voiced_creak = []
    conf_vad_modal = []
    conf_vad_creak = []

    t0 = time.time()
    for idx, npz_p in enumerate(sample_npzs):
        spk = npz_p.parent.name
        stem = npz_p.stem
        wav_p = DEFAULT_DATA_DIR / spk / f"{stem}.wav"
        tg_p = DEFAULT_TG_DIR / spk / f"{stem}.TextGrid"
        if not wav_p.exists() or not tg_p.exists():
            continue

        audio_orig, sr = sf.read(str(wav_p), dtype="float32", always_2d=False)
        if audio_orig.ndim > 1:
            audio_orig = np.mean(audio_orig, axis=1)
        wav_t = torch.from_numpy(audio_orig)
        if sr == 22050:
            wav_16k = resampler_22k(wav_t)
        elif sr != TARGET_SR:
            wav_16k = torchaudio.transforms.Resample(sr, TARGET_SR)(wav_t)
        else:
            wav_16k = wav_t

        with np.load(npz_p) as d:
            t_arr = d["timestamps"]
            p_crk = d["prob_creak"]

        # VAD
        sp_spans = get_speech_timestamps(wav_16k, vad_model, sampling_rate=TARGET_SR, threshold=0.45)
        vad_mask = np.zeros(len(t_arr), dtype=bool)
        f_samps = (t_arr * TARGET_SR).astype(int)
        for span in sp_spans:
            vad_mask |= (f_samps >= span["start"]) & (f_samps <= span["end"])

        # MFA phones
        phones = parse_tg(tg_p)
        vowel_mask = np.zeros(len(t_arr), dtype=bool)
        voiced_mask = np.zeros(len(t_arr), dtype=bool)
        for xmin, xmax, txt in phones:
            if is_vowel(txt):
                vowel_mask |= (t_arr >= xmin) & (t_arr <= xmax)
            if is_voiced(txt):
                voiced_mask |= (t_arr >= xmin) & (t_arr <= xmax)

        # CREPE
        audio_gpu = wav_16k.unsqueeze(0).to(device)
        f0, conf = torchcrepe.predict(
            audio_gpu,
            TARGET_SR,
            hop_length=320,
            fmin=50.0,
            fmax=550.0,
            model="full",
            batch_size=2048,
            device=device,
            return_harmonicity=True,
        )
        c_arr = conf.squeeze(0).cpu().numpy().astype(np.float32)
        c_times = (np.arange(len(c_arr), dtype=np.float32) * 320) / TARGET_SR
        conf_interp = np.interp(t_arr, c_times, c_arr)

        is_crk = p_crk >= args.threshold
        is_mdl = ~is_crk

        conf_vowel_modal.extend(conf_interp[vowel_mask & is_mdl])
        conf_vowel_creak.extend(conf_interp[vowel_mask & is_crk])
        conf_voiced_modal.extend(conf_interp[voiced_mask & is_mdl])
        conf_voiced_creak.extend(conf_interp[voiced_mask & is_crk])
        conf_vad_modal.extend(conf_interp[vad_mask & is_mdl])
        conf_vad_creak.extend(conf_interp[vad_mask & is_crk])

        if (idx + 1) % 15 == 0 or (idx + 1) == len(sample_npzs):
            logger.info(f"Progress: [{idx+1}/{len(sample_npzs)}] files processed in {time.time() - t0:.1f}s")
            sys.stdout.flush()

    def summarize_dist(arr, name):
        arr = np.array(arr)
        n = len(arr)
        if n == 0:
            return {}
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
        counts, _ = np.histogram(arr, bins=bins)
        pcts = counts / n * 100
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        p10, p25, p50, p75, p90 = np.percentile(arr, [10, 25, 50, 75, 90])
        return {
            "Domain & Voice State": name,
            "Total Frames": n,
            "[0.0, 0.2)": f"{pcts[0]:.1f}% ({counts[0]})",
            "[0.2, 0.4)": f"{pcts[1]:.1f}% ({counts[1]})",
            "[0.4, 0.6)": f"{pcts[2]:.1f}% ({counts[2]})",
            "[0.6, 0.8)": f"{pcts[3]:.1f}% ({counts[3]})",
            "[0.8, 1.0]": f"{pcts[4]:.1f}% ({counts[4]})",
            "Conf ≥ 0.50 (%)": f"{(np.sum(arr >= 0.5) / n * 100):.2f}%",
            "Mean": f"{mean_val:.3f}",
            "Median": f"{p50:.3f}",
            "Std": f"{std_val:.3f}",
            "IQR [P25, P75]": f"[{p25:.3f}, {p75:.3f}]",
            "P10 - P90": f"[{p10:.3f}, {p90:.3f}]",
        }

    rows = [
        summarize_dist(conf_vowel_modal, "1. Pure Vowels - Modal Voice (Clean)"),
        summarize_dist(conf_vowel_creak, "1. Pure Vowels - Creaky Voice (Creak)"),
        summarize_dist(conf_vowel_modal + conf_vowel_creak, "1. Pure Vowels - All Frames"),
        summarize_dist(conf_voiced_modal, "2. All Voiced - Modal Voice (Clean)"),
        summarize_dist(conf_voiced_creak, "2. All Voiced - Creaky Voice (Creak)"),
        summarize_dist(conf_voiced_modal + conf_voiced_creak, "2. All Voiced - All Frames"),
        summarize_dist(conf_vad_modal, "3. Full VAD - Modal Voice (Baseline)"),
        summarize_dist(conf_vad_creak, "3. Full VAD - Creaky Voice (Baseline)"),
        summarize_dist(conf_vad_modal + conf_vad_creak, "3. Full VAD - All Speech Frames"),
    ]

    df = pd.DataFrame(rows)
    csv_path = DEFAULT_OUT_DIR / "voicing_f0_confidence_distribution.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"Saved audit metrics to: {csv_path}")

    print("\n" + "=" * 120)
    print("CREPE F0 CONFIDENCE DISTRIBUTION AUDIT ACROSS PHONETIC TIERS")
    print("=" * 120)
    print(df.to_string(index=False))
    print("=" * 120 + "\n")


if __name__ == "__main__":
    main()
