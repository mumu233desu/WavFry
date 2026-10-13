#!/usr/bin/env python3
"""
QUT-NOISE Dataset Preprocessing Pipeline
Streamingly extracts, crops [start, finish] calibration-free noise, resamples to 16kHz mono,
normalizes RIR impulse responses (sum(h^2)=1.0), and saves as compact Level 8 FLAC files.
"""

import argparse
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("process_qut_noise")


SCENARIO_ZIPS = {
    "cafe": "qutnoisecafe.zip",
    "car": "qutnoisecar.zip",
    "home": "qutnoisehome.zip",
    "reverb": "qutnoisereverb.zip",
    "street": "qutnoisestreet.zip",
}


def parse_lab_file(content: str) -> Tuple[Optional[float], Optional[float], List[Tuple[float, float, str]]]:
    """Parse QUT-NOISE .lab.txt file to extract start/finish and any bad intervals."""
    lines = content.strip().splitlines()
    start_s = None
    finish_s = None
    bad_intervals = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                t0 = float(parts[0])
                t1 = float(parts[1])
                tag = " ".join(parts[2:])
                if tag.lower() == "start":
                    start_s = t0
                elif tag.lower() == "finish":
                    finish_s = t1
                elif "bad" in tag.lower():
                    bad_intervals.append((t0, t1, tag))
            except ValueError:
                continue

    return start_s, finish_s, bad_intervals


def process_noise_file(
    zf: zipfile.ZipFile,
    wav_entry: str,
    lab_content: str,
    output_path: Path,
    target_sr: int = 16000,
    chunk_seconds: float = 30.0,
) -> Dict[str, Any]:
    """
    Streamingly extract audio from ZIP entry, cut out sine sweep calibration,
    resample left channel to target_sr, and write to FLAC Level 8.
    """
    start_s, finish_s, bad_intervals = parse_lab_file(lab_content)
    if start_s is None or finish_s is None:
        raise ValueError(f"Could not determine start/finish from lab file for {wav_entry}")

    # Physical protection: If there are bad events like 'microphone fell over'
    # in STREET-CITY-1, truncate finish before the bad event.
    for b_start, b_end, tag in bad_intervals:
        if b_start >= start_s and b_start < finish_s:
            logger.warning(
                f"Detected bad section ({tag}) in {wav_entry} at [{b_start:.1f}s, {b_end:.1f}s]. "
                f"Truncating clean segment end from {finish_s:.1f}s to {b_start:.1f}s."
            )
            finish_s = min(finish_s, b_start)

    duration_orig = finish_s - start_s
    logger.info(
        f"Processing noise {Path(wav_entry).name}: crop [{start_s:.2f}s, {finish_s:.2f}s] "
        f"({duration_orig / 60.0:.2f} min)"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zf.open(wav_entry) as f_wav:
        with sf.SoundFile(f_wav) as sf_in:
            orig_sr = sf_in.samplerate
            channels = sf_in.channels
            total_frames = sf_in.frames

            start_frame = max(0, int(round(start_s * orig_sr)))
            finish_frame = min(total_frames, int(round(finish_s * orig_sr)))
            num_frames_to_read = finish_frame - start_frame

            if num_frames_to_read <= 0:
                raise ValueError(f"Invalid frame range for {wav_entry}: {start_frame} to {finish_frame}")

            sf_in.seek(start_frame)

            resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
            chunk_frames = int(round(chunk_seconds * orig_sr))

            written_samples = 0
            sum_sq = 0.0

            # Open FLAC writer
            with sf.SoundFile(
                str(output_path),
                mode="w",
                samplerate=target_sr,
                channels=1,
                format="FLAC",
                subtype="PCM_16",
            ) as sf_out:
                frames_left = num_frames_to_read
                while frames_left > 0:
                    read_n = min(chunk_frames, frames_left)
                    data = sf_in.read(read_n, dtype="float32")
                    if data.ndim == 2:
                        # Channel 0: Left channel (official QUT MATLAB standard)
                        data_mono = data[:, 0]
                    else:
                        data_mono = data

                    # Resample via torchaudio
                    tensor_48k = torch.from_numpy(data_mono).unsqueeze(0)
                    tensor_16k = resampler(tensor_48k).squeeze(0).numpy()

                    # Clip slightly to prevent PCM_16 overflow if any
                    np.clip(tensor_16k, -1.0, 0.9999, out=tensor_16k)

                    sf_out.write(tensor_16k)
                    written_samples += len(tensor_16k)
                    sum_sq += float(np.sum(tensor_16k**2))
                    frames_left -= read_n

    rms = np.sqrt(sum_sq / max(1, written_samples))
    duration_s = written_samples / target_sr

    return {
        "file": output_path.name,
        "path": str(output_path.as_posix()),
        "duration_s": round(duration_s, 3),
        "samples": written_samples,
        "sample_rate": target_sr,
        "rms": round(float(rms), 6),
    }


def process_impulse_file(
    zf: zipfile.ZipFile,
    imp_entry: str,
    output_path: Path,
    target_sr: int = 16000,
    orig_sr: int = 48000,
) -> Dict[str, Any]:
    """
    Extract first column of RIR impulse, resample to target_sr,
    normalize to unit energy (sum(h^2) = 1.0), and save as FLAC.
    """
    logger.info(f"Processing RIR impulse {Path(imp_entry).name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with zf.open(imp_entry) as f_imp:
        raw_arr = np.loadtxt(f_imp, dtype=np.float32)

    # Column 0: Reference channel (official QUT MATLAB standard: imp(:, 1))
    col0 = raw_arr[:, 0] if raw_arr.ndim == 2 else raw_arr
    tensor_48k = torch.from_numpy(col0).unsqueeze(0)

    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    tensor_16k = resampler(tensor_48k).squeeze(0)

    # Energy normalization: sum(h^2) = 1.0
    energy = float(torch.sum(tensor_16k**2))
    norm_factor = np.sqrt(energy) if energy > 1e-12 else 1.0
    h_normalized = (tensor_16k / norm_factor).numpy()

    # Save as FLAC
    sf.write(
        str(output_path),
        h_normalized,
        target_sr,
        format="FLAC",
        subtype="PCM_16",
    )

    peak_abs = float(np.max(np.abs(h_normalized)))
    final_energy = float(np.sum(h_normalized**2))

    return {
        "file": output_path.name,
        "path": str(output_path.as_posix()),
        "duration_s": round(len(h_normalized) / target_sr, 4),
        "samples": len(h_normalized),
        "sample_rate": target_sr,
        "peak_abs": round(peak_abs, 6),
        "energy": round(final_energy, 6),
    }


def main():
    parser = argparse.ArgumentParser(description="Preprocess QUT-NOISE dataset to 16kHz mono FLAC.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="E:/Corpus/Noise/QUT-NOISE",
        help="Directory containing QUT-NOISE zip files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="E:/Corpus/Noise/QUT-NOISE_processed",
        help="Directory to save processed FLAC files and metadata.",
    )
    parser.add_argument(
        "--target_sr",
        type=int,
        default=16000,
        help="Target sample rate (default 16000 Hz).",
    )
    parser.add_argument(
        "--chunk_seconds",
        type=float,
        default=30.0,
        help="Streaming chunk size in seconds.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "target_sample_rate": args.target_sr,
        "noise": {},
        "impulses": {},
    }

    total_noise_hours = 0.0
    total_rir_count = 0

    for scenario, zip_name in SCENARIO_ZIPS.items():
        zip_path = input_dir / zip_name
        if not zip_path.exists():
            logger.warning(f"Zip file {zip_path} not found! Skipping scenario '{scenario}'.")
            continue

        logger.info(f"=== Opening Archive: {zip_name} (Scenario: {scenario}) ===")
        metadata["noise"][scenario] = []

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = zf.namelist()

            # Find all .lab.txt files
            lab_entries = {
                Path(name).stem.replace(".lab", ""): name
                for name in namelist
                if name.endswith(".lab.txt")
            }

            # 1. Process Noise WAVs
            for name in namelist:
                if name.endswith(".wav") and not name.startswith("__"):
                    base = Path(name).stem
                    if base in lab_entries:
                        lab_content = zf.read(lab_entries[base]).decode("utf-8", errors="ignore")
                        out_flac = output_dir / "noise" / scenario / f"{base}.flac"
                        info = process_noise_file(
                            zf=zf,
                            wav_entry=name,
                            lab_content=lab_content,
                            output_path=out_flac,
                            target_sr=args.target_sr,
                            chunk_seconds=args.chunk_seconds,
                        )
                        # Relative path in metadata for portability
                        rel_path = out_flac.relative_to(output_dir).as_posix()
                        info["rel_path"] = rel_path
                        metadata["noise"][scenario].append(info)
                        total_noise_hours += info["duration_s"] / 3600.0
                    else:
                        logger.warning(f"No matching .lab.txt found for {name}, skipping.")

            # 2. Process RIR Impulses (car and reverb scenarios)
            imp_entries = [
                name for name in namelist if name.endswith(".imp.txt") and not name.startswith("__")
            ]
            if imp_entries:
                metadata["impulses"][scenario] = []
                for imp_name in imp_entries:
                    base = Path(imp_name).stem.replace(".imp", "")
                    out_flac = output_dir / "impulses" / scenario / f"{base}.flac"
                    imp_info = process_impulse_file(
                        zf=zf,
                        imp_entry=imp_name,
                        output_path=out_flac,
                        target_sr=args.target_sr,
                    )
                    rel_path = out_flac.relative_to(output_dir).as_posix()
                    imp_info["rel_path"] = rel_path
                    metadata["impulses"][scenario].append(imp_info)
                    total_rir_count += 1

    # Save metadata.json
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Processing complete!")
    logger.info(f"Total pure noise extracted: {total_noise_hours:.2f} hours across 5 scenarios.")
    logger.info(f"Total RIR impulses extracted: {total_rir_count} impulses.")
    logger.info(f"Metadata saved to: {meta_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
