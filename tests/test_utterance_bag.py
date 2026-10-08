import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from wavfry.data.utterance_bag import BagBatchSampler, UtteranceBagDataset, collate_utterance_bags, utterance_logits


def rows(tmp_path, seconds=(1, 5)):
    output = []
    for i, duration in enumerate(seconds):
        path = tmp_path / f"{i}.wav"
        sf.write(path, np.zeros(duration * 16000), 16000)
        output.append(dict(task="breathy", label=1, import_status="eligible", audio_path=str(path),
                           start_s=0, end_s=duration, file_key=str(i), vowel_intervals=json.dumps([[.1, duration - .1]])))
    return pd.DataFrame(output)


def test_complete_bag_tail_padding_and_membership(tmp_path):
    ds = UtteranceBagDataset(rows(tmp_path))
    batch = collate_utterance_bags([ds[0], ds[1]])
    assert batch["waveforms"].shape == (5, 32000)
    assert batch["lengths"].tolist() == [16000, 32000, 32000, 32000, 32000]
    assert batch["bag_index"].tolist() == [0, 1, 1, 1, 1]
    assert ds[1]["starts"][-1].item() + 32000 >= 5 * 16000


def test_budget_cannot_truncate_clip(tmp_path):
    with pytest.raises(ValueError, match="never truncate"):
        UtteranceBagDataset(rows(tmp_path, (18,)), max_windows=2)


def test_budgeted_sampler_deterministic(tmp_path):
    ds = UtteranceBagDataset(rows(tmp_path, (1, 2, 3, 4)))
    sampler = BagBatchSampler(ds, max_windows=4, samples_per_epoch=4)
    assert list(sampler) == list(sampler)
    assert all(sum(ds.costs[i] for i in b) <= 4 for b in sampler)
    assert sorted(i for b in sampler for i in b) == list(range(4))


def test_late_event_is_supervised_once_per_utterance():
    frame_logits = torch.tensor([[-5., -5.], [-5., 8.]], requires_grad=True)
    outputs = dict(frame_logits={"breathy": frame_logits}, frame_valid=torch.ones(2, 2, dtype=torch.bool),
                   frame_stride_samples=16000, frame_receptive_samples=1)
    batch = dict(starts=torch.tensor([0, 32000]), bag_index=torch.tensor([0, 0]), vowels=[[[0, 4]]])
    logits = utterance_logits(outputs, batch)
    assert logits.shape == (1,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
    assert loss.item() < .01
    loss.backward()
    assert frame_logits.grad[1, 1] < 0
    assert abs(frame_logits.grad[0, 0]) < abs(frame_logits.grad[1, 1])


def test_all_nan_or_empty_candidates_not_used(tmp_path):
    df = rows(tmp_path, (1,))
    df.loc[0, "vowel_intervals"] = "[]"
    with pytest.raises(ValueError, match="No vowel"):
        UtteranceBagDataset(df)
