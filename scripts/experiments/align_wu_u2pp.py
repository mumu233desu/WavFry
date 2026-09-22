#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forced Alignment for Wu Chinese (吴语) using official Conformer-U2pp-Wu.
Generates Praat .TextGrid files with character-level intervals for 04_creaky_tools GUI.
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from textgrid import TextGrid, IntervalTier

# Ensure wenet is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WENET_DIR = PROJECT_ROOT / "pretrained" / "wenet"
if str(WENET_DIR) not in sys.path:
    sys.path.insert(0, str(WENET_DIR))

from wenet.utils.init_model import init_model
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.common import get_subsample


# Official Demo Page Transcriptions for C:\Users\mumu\Downloads\wu-test
DEMO_TRANSCRIPTS = {
    "223327119763_Bvjtk_37_7480.wav": "青春的舞龙唱出短暂的曲子的清风里后世",
    "223327122594_1JW6p_41_7720.wav": "这自然应该是像上海大都市这能介告诉伊虽然伊同样是外来的闲话",
    "223327123108_v7RhP_110_8039.wav": "呃大灰狼就跟山羊奶奶讲山羊奶奶侬一家头蹲阿拉决定拿这点物事侪送拨侬",
    "224023060419_CSz76_66_8220.wav": "这物事里头是有利益分配的讲好个埃种大生意难做一趟做两三年也做不出的",
}


def load_char_dict(dict_path: str | Path) -> dict[str, int]:
    """Load character-to-id dictionary."""
    char_to_id = {}
    id_to_char = {}
    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                char, idx = parts[0], int(parts[1])
                char_to_id[char] = idx
                id_to_char[idx] = char
    return char_to_id, id_to_char


def compute_fbank(wav_path: str | Path, target_sr: int = 16000) -> tuple[torch.Tensor, float]:
    """Compute 80-dim log-mel fbank features matching WeNet training config."""
    waveform, sr = torchaudio.load(str(wav_path))
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    duration = waveform.shape[1] / target_sr

    # 25ms frame_length, 10ms frame_shift, 80 mel bins
    # WeNet kaldi fbank standard: waveform * (1 << 15) for int16 scaling
    mat = kaldi.fbank(
        waveform * (1 << 15),
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=target_sr,
    )
    return mat, duration


def perform_alignment(
    model: torch.nn.Module,
    feats: torch.Tensor,
    text: str,
    char_to_id: dict[str, int],
    blank_id: int = 0,
    device: str = "cuda"
) -> tuple[list[int], list[str]]:
    """
    Forward Conformer CTC and perform CTC forced alignment.
    Returns (alignment_tensor, aligned_tokens)
    """
    feats = feats.unsqueeze(0).to(device)  # (1, T, 80)
    feats_length = torch.tensor([feats.shape[1]], dtype=torch.int32, device=device)

    with torch.no_grad():
        encoder_out, _ = model._forward_encoder(feats, feats_length)
        ctc_log_probs = model.ctc.log_softmax(encoder_out)  # (1, T_enc, vocab_size)

    # Convert text to token IDs
    tokens = []
    token_chars = []
    for c in text:
        if c in char_to_id:
            tokens.append(char_to_id[c])
            token_chars.append(c)
        else:
            # Fallback to <unk> if present, otherwise ignore
            unk_id = char_to_id.get("<unk>", 1)
            tokens.append(unk_id)
            token_chars.append(c)

    target = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).cpu()  # (1, U)
    ctc_probs_cpu = ctc_log_probs.cpu()  # (1, T_enc, V)

    # CTC forced alignment via PyTorch
    alignment, _ = torchaudio.functional.forced_align(
        ctc_probs_cpu, target, blank=blank_id
    )
    return alignment[0].tolist(), token_chars


def alignment_to_intervals(
    align_seq: list[int],
    token_chars: list[str],
    duration: float,
    frame_shift_s: float = 0.04,  # 10ms * 4 subsample = 40ms
    blank_id: int = 0,
    max_pause_s: float = 0.45,
    default_token_half_width: float = 0.18
) -> list[tuple[float, float, str]]:
    """
    Convert frame-level CTC alignments to syllable-duration intervals.
    Uses peak timestamps and midpoint boundaries between adjacent tokens,
    with pause detection for natural auditory segments.
    """
    # 1. Find the peak/emission frame for each token in token_chars
    # align_seq contains token IDs along the time frames
    peaks = []
    curr_token_idx = 0
    in_token = False
    token_frames = []

    for f_idx, val in enumerate(align_seq):
        if val != blank_id and curr_token_idx < len(token_chars):
            token_frames.append(f_idx)
            in_token = True
        else:
            if in_token:
                # Find the center frame for this token
                center_frame = token_frames[len(token_frames) // 2]
                peaks.append((center_frame * frame_shift_s, token_chars[curr_token_idx]))
                curr_token_idx += 1
                token_frames = []
                in_token = False

    if in_token and curr_token_idx < len(token_chars):
        center_frame = token_frames[len(token_frames) // 2]
        peaks.append((center_frame * frame_shift_s, token_chars[curr_token_idx]))
        curr_token_idx += 1

    # 2. Convert peaks to natural continuous intervals
    intervals = []
    last_end = 0.0

    for i in range(len(peaks)):
        t_peak, ch = peaks[i]

        # Calculate start time
        if i == 0:
            t_start = max(0.0, t_peak - default_token_half_width)
        else:
            prev_peak = peaks[i - 1][0]
            gap = t_peak - prev_peak
            if gap > max_pause_s:
                t_start = t_peak - default_token_half_width
            else:
                t_start = (prev_peak + t_peak) / 2.0

        # Calculate end time
        if i == len(peaks) - 1:
            t_end = min(duration, t_peak + default_token_half_width)
        else:
            next_peak = peaks[i + 1][0]
            gap = next_peak - t_peak
            if gap > max_pause_s:
                t_end = t_peak + default_token_half_width
            else:
                t_end = (t_peak + next_peak) / 2.0

        t_start = max(last_end, t_start)
        t_end = min(duration, max(t_start + frame_shift_s, t_end))

        if t_start > last_end + 0.001:
            intervals.append((last_end, t_start, ""))

        intervals.append((t_start, t_end, ch))
        last_end = t_end

    if last_end < duration:
        intervals.append((last_end, duration, ""))

    return intervals


def export_textgrid(
    intervals: list[tuple[float, float, str]],
    duration: float,
    output_path: str | Path,
):
    """Write intervals to Praat TextGrid format with both 'words' and 'phones' tiers."""
    tg = TextGrid(maxTime=duration)
    tier_words = IntervalTier(name="words", maxTime=duration)
    tier_phones = IntervalTier(name="phones", maxTime=duration)

    for s, e, mark in intervals:
        s_clamped = max(0.0, s)
        e_clamped = min(duration, e)
        if e_clamped > s_clamped:
            tier_words.add(minTime=s_clamped, maxTime=e_clamped, mark=mark)
            tier_phones.add(minTime=s_clamped, maxTime=e_clamped, mark=mark)

    tg.append(tier_words)
    tg.append(tier_phones)
    tg.write(str(output_path))
    print(f"  -> Generated TextGrid: {output_path} ({len(tier_words)} intervals, tiers: words, phones)")


def main():
    parser = argparse.ArgumentParser(description="Wu Chinese U2++ CTC Forced Aligner")
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=r"C:\Users\mumu\Downloads\wu-test",
        help="Directory containing test .wav files",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=str(PROJECT_ROOT / "pretrained" / "wenet_u2pp_wu"),
        help="Directory containing u2++.pt, train.yaml, and dict/lang_char.txt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device (cuda or cpu)",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config_path = model_dir / "train.yaml"
    dict_path = model_dir / "dict" / "lang_char.txt"
    ckpt_path = model_dir / "u2++.pt"

    print(f"Loading WeNet configuration from: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Patch dictionary path and dimensions
    configs["tokenizer_conf"]["symbol_table_path"] = str(dict_path)
    char_to_id, id_to_char = load_char_dict(dict_path)
    blank_id = configs.get("ctc_conf", {}).get("ctc_blank_id", 0)

    configs["input_dim"] = configs.get("dataset_conf", {}).get("fbank_conf", {}).get("num_mel_bins", 80)
    configs["output_dim"] = len(char_to_id)
    configs["cmvn"] = None  # cmvn is 'none' in train.yaml

    print(f"Vocabulary size: {len(char_to_id)}, Blank ID: {blank_id}, Input dim: {configs['input_dim']}")
    print(f"Initializing U2++ Conformer model on device: {args.device}...")

    # WeNet init_model expects args object or dict
    class DummyArgs:
        checkpoint = str(ckpt_path)
        enc_init = None
        device = args.device
        gpu = 0 if "cuda" in args.device else -1

    model, configs = init_model(DummyArgs(), configs)
    
    # Load checkpoint weights
    print(f"Loading checkpoint weights from: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=False)
    model = model.to(args.device)
    model.eval()

    subsample = get_subsample(configs)  # usually 4
    frame_shift_s = 0.01 * subsample    # 10ms * 4 = 40ms
    print(f"Subsampling factor: {subsample}, Effective frame shift: {frame_shift_s * 1000:.1f} ms")

    wav_dir = Path(args.wav_dir)
    wav_files = sorted(list(wav_dir.glob("*.wav")))
    print(f"\nFound {len(wav_files)} WAV files in {wav_dir}\n" + "=" * 60)

    for wav_path in wav_files:
        filename = wav_path.name
        transcript = DEMO_TRANSCRIPTS.get(filename)
        if not transcript:
            print(f"[SKIP] No known transcript for: {filename}")
            continue

        print(f"\nProcessing: {filename}")
        print(f"  Transcript: {transcript}")

        # 1. Feature extraction
        feats, duration = compute_fbank(wav_path)
        print(f"  Audio duration: {duration:.3f}s, Fbank frames: {feats.shape[0]}")

        # 2. Forced alignment
        align_seq, token_chars = perform_alignment(
            model=model,
            feats=feats,
            text=transcript,
            char_to_id=char_to_id,
            blank_id=blank_id,
            device=args.device
        )

        # 3. Intervals
        intervals = alignment_to_intervals(
            align_seq=align_seq,
            token_chars=token_chars,
            duration=duration,
            frame_shift_s=frame_shift_s,
            blank_id=blank_id
        )

        # 4. Export TextGrid
        tg_path = wav_dir / f"{wav_path.stem}.TextGrid"
        export_textgrid(intervals, duration, tg_path)

    print("\n" + "=" * 60)
    print("All alignments successfully generated!")


if __name__ == "__main__":
    main()
