import pandas as pd
import pytest
import torch
from transformers import WavLMConfig, WavLMModel

from wavfry.eval.frame_quality import exposure, interval_mask, merge_frames, pooled_score, review_queues, window_starts
from wavfry.model.wavlm_classifier import WavFryClassifier


def test_window_grid_covers_tail_and_short_clips():
    assert window_starts(16000) == [0]
    starts = window_starts(16000 * 9 + 73)
    assert starts[-1] + 32000 >= 16000 * 9 + 73
    assert all(s % 320 == 0 for s in starts)


def test_overlap_counting_and_backward():
    values = torch.tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    merged = merge_frames(values, torch.ones_like(values, dtype=torch.bool),
                          torch.tensor([0, 2]), torch.tensor([0, 0]), stride=1, receptive=1)[0]
    assert merged["logits"].tolist() == [1., 2., 3.5, 5., 6.]
    assert merged["counts"].tolist() == [1., 1., 2., 1., 1.]
    merged["logits"].sum().backward()
    assert values.grad.tolist() == [[1., 1., .5], [.5, 1., 1.]]


def test_padding_and_bag_separation():
    logits = torch.tensor([[1., 100.], [-1., 100.]])
    result = merge_frames(logits, torch.tensor([[True, False], [True, False]]),
                          torch.tensor([0, 0]), torch.tensor([0, 1]), 1, 1)
    assert len(result) == 2
    assert result[0]["logits"].tolist() == [1.]
    assert result[1]["logits"].tolist() == [-1.]


def test_candidate_mask_and_score():
    times = torch.tensor([.1, .2, .3])
    mask = interval_mask(times, [[.09, .21]])
    assert mask.tolist() == [True, True, False]
    assert pooled_score(torch.tensor([1., 1., 100.]), mask).item() == pytest.approx(torch.sigmoid(torch.tensor(1.)).item())


def test_no_claim_of_heldout_without_provenance():
    row = dict(dataset_id="d", speaker_id="s", file_key="f")
    assert exposure(row) == "unknown"
    assert exposure(row, dict(complete=True, training_speakers=[["d", "other"]])) == "held_out"
    assert exposure(row, dict(complete=True, training_speakers=[["d", "s"]])) == "training_seen"


def test_review_queue_includes_controls():
    df = pd.DataFrame([dict(source="human", task="breathy", label=1, agreement=.1, context_std=.2),
                       dict(source="human", task="breathy", label=1, agreement=.9, context_std=.1)])
    assert set(review_queues(df, top_k=1, random_k=1).queue) == {
        "low_agreement", "unstable_context", "high_agreement", "random_control"}


def test_frame_forward_optin_keeps_default_contract():
    backbone = WavLMModel(WavLMConfig(hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, conv_dim=(8, 8, 8), conv_stride=(5, 4, 4), conv_kernel=(10, 8, 8),
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2))
    config = {"model": {"backbone": "dummy", "fine_tune_layers": "none", "tasks": ["breathy"], "head_hidden_dim": 8},
              "training": {"pooling_mode": "mean", "block": {"frame_head": {"enabled": True}}}}
    model = WavFryClassifier(config, wavlm_instance=backbone).eval()
    x = torch.randn(1, 1600)
    assert set(model(x, torch.tensor([1600]))) == {"breathy"}
    frames = model(x, torch.tensor([1600]), return_frames=True)
    assert frames["frame_logits"]["breathy"].shape == frames["frame_valid"].shape
    assert frames["frame_stride_samples"] == 80
