import json
from pathlib import Path
import sys

WAVE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WAVE_ROOT / "04_creaky_tools"))

from wavfry.data.manifest import build_manifest
from src.manual_annotation import annotation_storage_path, save_annotations
from src.phone_inventory import make_interval_id


def test_manifest_filters_unknown_and_non_whitelisted_types(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    textgrid_dir = tmp_path / "textgrid"
    audio_dir.mkdir()
    textgrid_dir.mkdir()
    file_key = "F_23_Test_EN_ZH-CN_U1001_S0_1"
    (audio_dir / f"{file_key}.wav").write_bytes(b"")
    (textgrid_dir / f"{file_key}.TextGrid").write_text("", encoding="utf-8")

    # Co-located per-batch storage layout (required since the sampling_state
    # migration): the adapter refuses the legacy monolithic layout.
    cache_dir = textgrid_dir / "_creaky_cache"
    cache_dir.mkdir()
    (cache_dir / "sampling_state.json").write_text(
        json.dumps({"batches": [{"batch_id": 0, "name": "Batch 0", "split": "train"}]}),
        encoding="utf-8",
    )
    annotation_path = annotation_storage_path(textgrid_dir, batch_id=0)
    canonical = make_interval_id(0.0, 0.1, "a")
    unknown = make_interval_id(0.2, 0.3, "spn")
    save_annotations(annotation_path, {"files": {file_key: {"annotations": {
        canonical: "creaky", unknown: "non_creaky"
    }}}})
    config = {"data": {
        "annotated_speakers": ["U1001"],
        "include_languages": ["en"],
        "training_phone_types": ["vowel"],
        "audio_dirs": {"en": str(audio_dir)},
        "textgrid_dirs": {"en": str(textgrid_dir)},
        "min_duration_ms": 30,
    }}
    df = build_manifest(config)
    assert len(df) == 1
    assert df.iloc[0]["phone_type"] == "vowel"
    assert df.iloc[0]["label"] == 1
