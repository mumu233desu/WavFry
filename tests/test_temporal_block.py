import math

import pandas as pd
import torch

from wavfry.data.paradigm_block import BlockIndex, ContinuousAudioBlockDataset, collate_block_fn
from wavfry.model.losses import logmeanexp_pool, pool_frame_logits
from wavfry.model.wavlm_classifier import WavFryClassifier


def _row(start, end, label=0, file_key="file", phone="a"):
    return {
        "file_key": file_key,
        "audio_path": "test.wav",
        "start_s": start,
        "end_s": end,
        "label": label,
        "speaker_id": "U1001",
        "phone": phone,
    }


def test_block_index_keeps_every_interval_complete():
    df = pd.DataFrame([
        _row(0.0, 0.3),
        _row(0.9, 1.2, 1),
        _row(1.9, 2.2),
        _row(2.8, 3.1, 1),
    ])
    index = BlockIndex(df, duration_ms=2000, hop_ms=1000, max_intervals=2)
    represented = {
        interval["interval_id"]
        for block in index.blocks
        for interval in block["intervals"]
    }
    expected = {
        BlockIndex._interval_id(row)
        for row in df.to_dict("records")
    }
    assert represented == expected
    assert all(block["intervals"] for block in index.blocks)
    assert all(len(block["intervals"]) <= 2 for block in index.blocks)


def test_interval_longer_than_block_truncates_safely():
    df = pd.DataFrame([_row(0.0, 2.5)])
    index = BlockIndex(df, duration_ms=2000, hop_ms=1000)
    assert len(index.blocks) > 0
    block = index.blocks[0]
    interval = block["intervals"][0]
    assert interval["end_s"] <= block["end_s"] + 1e-7


def test_interval_cap_does_not_drop_same_start_intervals():
    df = pd.DataFrame([
        _row(0.0, 0.2, phone=f"p{i}") for i in range(5)
    ])
    index = BlockIndex(df, duration_ms=2000, hop_ms=1000, max_intervals=2)
    represented = {
        interval["interval_id"]
        for block in index.blocks
        for interval in block["intervals"]
    }
    assert len(represented) == 5


def test_block_dataset_epoch_rebuild_is_deterministic():
    df = pd.DataFrame([_row(i * 0.05, i * 0.05 + 0.04, i % 2, phone=f"p{i}") for i in range(12)])
    first = ContinuousAudioBlockDataset(df, duration_ms=1000, hop_ms=500, max_intervals=3, seed=42)
    second = ContinuousAudioBlockDataset(df, duration_ms=1000, hop_ms=500, max_intervals=3, seed=42)
    first.set_epoch(3)
    second.set_epoch(3)
    first_ids = [[row["interval_id"] for row in block["intervals"]] for block in first.blocks]
    second_ids = [[row["interval_id"] for row in block["intervals"]] for block in second.blocks]
    assert first_ids == second_ids


def test_block_collate_pads_audio_and_interval_slots():
    batch = [
        (torch.ones(10), torch.tensor([[0, 4], [4, 8]]), torch.tensor([0.0, 1.0]), "s1", "b1", ["i1", "i2"]),
        (torch.ones(6), torch.tensor([[1, 5]]), torch.tensor([0.0]), "s2", "b2", ["i3"]),
    ]
    waveform, bounds, labels, valid, lengths, speakers, block_ids, ids = collate_block_fn(batch)
    assert waveform.shape == (2, 10)
    assert bounds.shape == (2, 2, 2)
    assert labels.shape == (2, 2)
    assert valid.tolist() == [[True, True], [True, False]]
    assert lengths.tolist() == [10, 6]
    assert ids[1] == ["i3", None]


def test_logmeanexp_is_length_normalized():
    short = torch.full((1, 2), 0.7)
    long = torch.full((1, 8), 0.7)
    for temperature in (0.5, 1.0, 2.0):
        assert torch.allclose(
            logmeanexp_pool(short, torch.ones_like(short, dtype=torch.bool), temperature),
            torch.tensor([0.7]),
        )
        assert torch.allclose(
            logmeanexp_pool(long, torch.ones_like(long, dtype=torch.bool), temperature),
            torch.tensor([0.7]),
        )


def test_frame_pool_rejects_empty_interval():
    logits = torch.zeros(2, 3)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    try:
        pool_frame_logits(logits, mask)
    except ValueError as exc:
        assert "at least one valid frame" in str(exc)
    else:
        raise AssertionError("Expected empty interval to fail")


def test_frame_pool_modes_return_one_logit_per_interval():
    logits = torch.tensor([[0.1, 2.0, -1.0], [0.4, 0.5, 0.6]])
    mask = torch.tensor([[True, True, False], [True, True, True]])
    for mode in ("lse", "max", "top_k_mean", "linear_softmax"):
        pooled = pool_frame_logits(logits, mask, mode=mode)
        assert pooled.shape == (2,)
        assert torch.isfinite(pooled).all()


def test_interval_relative_positions_restart_per_span():
    masks = torch.tensor([
        [False, True, True, True, False],
        [True, True, False, False, False],
    ])
    rel = WavFryClassifier._relative_positions(masks)
    assert torch.allclose(rel[0, 1:4], torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(rel[1, :2], torch.tensor([0.0, 1.0]))
    assert rel.masked_select(~masks).eq(0).all()
