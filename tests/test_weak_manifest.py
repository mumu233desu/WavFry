import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import tgt

from wavfry.data.weak_manifest import build_weak_manifest, classify_prediction


@pytest.mark.parametrize("prediction,status", [
    ({"has_breathy": True, "segments": [{"phonation_type": "breathy"}]}, "positive"),
    ({"has_breathy": False, "segments": []}, "unknown"),
    ({"has_breathy": "uncertain", "segments": []}, "review"),
    ({"has_breathy": "true", "segments": []}, "review"),
    ({"has_breathy": 1, "segments": []}, "review"),
    ({"has_breathy": False, "segments": [{"phonation_type": "breathy"}]}, "review"),
    ({"has_breathy": False, "has_whispery": True, "segments": [{"phonation_type": "whispery"}]}, "review"),
])
def test_strict_prediction_semantics(prediction, status):
    assert classify_prediction(prediction)[0] == status


def fixture_paths(tmp_path):
    audio, grids, cache = [tmp_path / n for n in ("audio", "grids", "cache")]
    for p in (audio, grids, cache):
        p.mkdir()
    sf.write(audio / "clip.wav", np.zeros(16000), 16000)
    grid = tgt.TextGrid()
    tier = tgt.IntervalTier(0, 1, name="phones")
    tier.add_interval(tgt.Interval(0.1, 0.8, "a"))
    grid.add_tier(tier)
    tgt.io.write_to_file(grid, str(grids / "clip.TextGrid"), format="long")
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"batches": [{"split": "train", "files": [{"file_key": "clip", "speaker_id": "s1"}]}]}))
    (cache / "clip.json").write_text(json.dumps({"model": "test", "parsed_prediction": {
        "has_breathy": True, "segments": [{"phonation_type": "breathy"}]},
        "alignment_result": {"detected_phrases": [{"alignment_mode": "temporal_fallback"}]}}))
    return dict(dataset_id="test", audio_dir=audio, textgrid_dir=grids, cache_dir=cache,
                sampling_state=state, vowel_phones={"a"})


def test_uncertain_location_still_supports_utterance(tmp_path):
    df = build_weak_manifest(**fixture_paths(tmp_path))
    assert df.iloc[0].import_status == "eligible"
    assert df.iloc[0].label == 1
    assert json.loads(df.iloc[0].vowel_intervals) == [[0.1, 0.8]]


def test_partial_negative_does_not_negate_clip(tmp_path):
    args = fixture_paths(tmp_path)
    args["manual_manifest"] = pd.DataFrame([dict(dataset_id="test", file_key="clip", speaker_id="s1", batch_split="train", label_breathy=0)])
    assert build_weak_manifest(**args).iloc[0].import_status == "eligible"
    args["manual_manifest"].loc[0, "label_breathy"] = 1
    assert build_weak_manifest(**args).iloc[0].import_status == "redundant"


def test_explicit_human_clip_review_overrides_machine(tmp_path):
    args = fixture_paths(tmp_path)
    args["reviews"] = pd.DataFrame([dict(file_key="clip", task="breathy", scope="utterance", label=0)])
    df = build_weak_manifest(**args)
    assert df.iloc[0].import_status == "human_reviewed"
    assert df.iloc[0].review_status == "human_clip_negative"
    assert df.iloc[0].label == 0
    assert df.iloc[0].source == "human_clip_review"


def test_split_conflict_fails(tmp_path):
    args = fixture_paths(tmp_path)
    args["manual_manifest"] = pd.DataFrame([dict(dataset_id="test", file_key="clip", speaker_id="s1", batch_split="test", label_breathy=0)])
    with pytest.raises(ValueError, match="split disagreement"):
        build_weak_manifest(**args)
