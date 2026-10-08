import json
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch

from test_weak_manifest import fixture_paths
from test_weak_training import setup
from wavfry.data.weak_manifest import build_weak_manifest, validate_source_stamps


def test_stale_cache_and_split_assignments_rejected(tmp_path):
    args = fixture_paths(tmp_path)
    df = build_weak_manifest(**args)
    validate_source_stamps(df)
    (args["cache_dir"] / "clip.json").write_text('{}')
    with pytest.raises(ValueError, match="MLLM cache"):
        validate_source_stamps(df)


def test_quality_scoring_mixed_manual_and_weak_rows(tmp_path):
    trainer, manifest = setup(tmp_path)
    script = Path(__file__).resolve().parents[1] / 'scripts/review_annotation_quality.py'
    spec = importlib.util.spec_from_file_location('review_annotation_quality', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    weak = pd.read_csv(manifest).iloc[0].to_dict()
    weak.update(scope='utterance', source='mllm', candidate_intervals='[]')
    manual = dict(weak)
    manual.update(scope='interval', start_s=.1, end_s=.3, label=0., source='human',
                  supervision_id='manual', candidate_intervals=float('nan'))
    records = module.score_file(trainer.model.eval(), [weak, manual], 'cpu')
    assert len(records) == 2
    assert all(0 <= r['frame_score'] <= 1 for r in records)
    assert records[1]['agreement'] == pytest.approx(1 - records[1]['frame_score'])
    assert records[1]['interval_score'] == records[1]['interval_score']
    trainer.writer.close()


def test_quality_cli_writes_real_reports(tmp_path, monkeypatch):
    trainer, _ = setup(tmp_path)
    # Save a local backbone to make checkpoint reconstruction fully offline.
    backbone_dir = tmp_path / 'backbone'
    trainer.model.wavlm.save_pretrained(backbone_dir)
    trainer.config['model']['backbone'] = str(backbone_dir)
    trainer._best_val_logits = None
    trainer._best_val_labels = None
    trainer.save_checkpoint(1, {'f1': .5, 'auc_pr': .5}, only_latest=True)
    manual = trainer.train_loader.dataset.base_df.copy()
    manual_path = tmp_path / 'manual.csv'
    manual.to_csv(manual_path, index=False)
    script = Path(__file__).resolve().parents[1] / 'scripts/review_annotation_quality.py'
    spec = importlib.util.spec_from_file_location('review_cli_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys
    output = tmp_path / 'review'
    monkeypatch.setattr(sys, 'argv', ['review', '--checkpoint', str(trainer.ckpt_dir / 'latest.pt'),
        '--manual-manifest', str(manual_path), '--output-dir', str(output), '--device', 'cpu'])
    module.main()
    report = json.loads((output / 'report.json').read_text())
    assert report['rows'] == 2
    assert (output / 'review_queues.csv').exists()
    trainer.writer.close()


def test_training_run_logs_weak_metrics_and_threshold(tmp_path):
    trainer, manifest = setup(tmp_path)
    history = trainer.run()
    assert len(history) == 2
    assert history[0]['weak_utterances'] == 0
    assert history[1]['weak_utterances'] == 1
    checkpoint = torch.load(trainer.ckpt_dir / 'best_model.pt', weights_only=False)
    assert 'best_threshold' in checkpoint
    assert checkpoint['training_provenance']['complete']
    trainer.writer.close()
