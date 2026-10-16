"""Unit tests for unified Kappa batch detection and consensus manifest loading."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest

from wavfry.data.manifest import _HAS_CREAKY_TOOLS

if _HAS_CREAKY_TOOLS:
    _mod = importlib.import_module("04_creaky_tools.src.manual_annotation")
    is_kappa_batch = getattr(_mod, "is_kappa_batch")
    find_kappa_batches = getattr(_mod, "find_kappa_batches")
    get_kappa_eval_dir = getattr(_mod, "get_kappa_eval_dir")
    load_kappa_consensus_annotations = getattr(_mod, "load_kappa_consensus_annotations")


@pytest.mark.skipif(not _HAS_CREAKY_TOOLS, reason="04_creaky_tools not available")
class TestKappaUnification:
    """Test unified Kappa detection and consensus aggregation."""

    def test_is_kappa_batch_variants(self, tmp_path: Path):
        # 1. Explicit is_kappa flag
        assert is_kappa_batch({"batch_id": 1, "is_kappa": True})
        assert not is_kappa_batch({"batch_id": 1, "is_kappa": False})

        # 2. sampling_mode == "kappa"
        assert is_kappa_batch({"batch_id": 2, "sampling_mode": "kappa"})
        assert is_kappa_batch({"batch_id": 2, "sampling_mode": "KAPPA"})

        # 3. Batch name contains "kappa"
        assert is_kappa_batch({"batch_id": 3, "name": "Batch 3 (Kappa_Test)"})
        assert is_kappa_batch({"batch_id": 3, "name": "kappa evaluation"})
        assert not is_kappa_batch({"batch_id": 3, "name": "Regular Batch 3"})

        # 4. Filesystem fallback
        cache_dir = tmp_path / "_creaky_cache"
        k_batch_dir = cache_dir / "kappa_eval" / "batch_9"
        k_batch_dir.mkdir(parents=True)
        (k_batch_dir / "annotator_A").mkdir()
        (k_batch_dir / "annotator_B").mkdir()

        assert is_kappa_batch({"batch_id": 9, "name": "Plain"}, cache_dir=cache_dir)
        assert not is_kappa_batch({"batch_id": 8, "name": "Plain"}, cache_dir=cache_dir)

    def test_find_kappa_batches(self):
        state = {
            "batches": [
                {"batch_id": 0, "name": "Batch 0", "split": "train"},
                {"batch_id": 1, "name": "Batch 1", "split": "val"},
                {"batch_id": 2, "name": "Batch 2", "split": "test"},
                {"batch_id": 4, "name": "Batch 4 (Kappa_Test)", "split": "test", "is_kappa": True},
            ]
        }
        res = find_kappa_batches(state)
        assert len(res) == 1
        assert res[0]["batch_id"] == 4

    def test_load_kappa_consensus_annotations(self, tmp_path: Path):
        # Setup mock kappa_eval dir with 2 annotators
        k_dir = tmp_path / "kappa_eval" / "batch_4"
        a_dir = k_dir / "annotator_A" / "manual_creaky_annotations.json.d"
        b_dir = k_dir / "annotator_B" / "manual_creaky_annotations.json.d"
        a_dir.mkdir(parents=True)
        b_dir.mkdir(parents=True)

        shard_a = {
            "file_key": "sample_1",
            "annotations": {
                "0.00|0.50|a": "creaky",      # Unanimous positive -> keep
                "0.50|1.00|i": "non_creaky",  # Unanimous negative -> keep
                "1.00|1.50|u": "creaky",      # Conflict -> drop
                "1.50|2.00|e": "uncertain",   # Uncertain -> drop
                "2.00|2.50|o": "creaky",      # Missing in B -> drop
            }
        }
        shard_b = {
            "file_key": "sample_1",
            "annotations": {
                "0.00|0.50|a": "creaky",
                "0.50|1.00|i": "non_creaky",
                "1.00|1.50|u": "non_creaky",
                "1.50|2.00|e": "non_creaky",
            }
        }

        (a_dir / "shard.json").write_text(json.dumps(shard_a), encoding="utf-8")
        (b_dir / "shard.json").write_text(json.dumps(shard_b), encoding="utf-8")

        consensus = load_kappa_consensus_annotations(k_dir, task="creaky")
        annos = consensus["files"]["sample_1"]["annotations"]

        assert annos["0.00|0.50|a"] == "creaky"
        assert annos["0.50|1.00|i"] == "non_creaky"
        assert annos["1.00|1.50|u"] == "uncertain"
        assert annos["1.50|2.00|e"] == "uncertain"
        assert annos["2.00|2.50|o"] == "uncertain"
        assert consensus["total_unanimous"] == 2
        assert consensus["total_disputed"] == 3
