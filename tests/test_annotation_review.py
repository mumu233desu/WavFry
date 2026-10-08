import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

from wavfry.data.annotation_review import load_raw_annotations
from test_weak_training import setup


def raw_fixture(tmp_path, task='creaky', filename_task=None, annotations=None):
    audio = tmp_path / 'audio'
    audio.mkdir()
    sf.write(audio / 'clip.wav', np.zeros(16000), 16000)
    cache = tmp_path / f'_{task}_cache'
    shards = cache / 'batches' / 'batch_1' / f'manual_{filename_task or task}_annotations.json.d'
    shards.mkdir(parents=True)
    (cache / 'sampling_state.json').write_text(json.dumps({'batches': [{
        'batch_id': 1, 'split': 'val', 'files': [{'file_key': 'clip', 'speaker_id': 's1'}]}]}))
    path = shards / (hashlib.sha256(b'clip').hexdigest() + '.json')
    states = annotations or {'0.1|0.3|s': task, '0.4|0.6|a': f'non_{task}', '0.7|0.8|a': 'uncertain'}
    path.write_text(json.dumps(dict(file_key='clip', annotations=states, reviewed_at=123)))
    return dict(audio_dirs=[audio], task=task, cache_dir=cache, dataset_id='d'), path


@pytest.mark.parametrize('task', ['creaky', 'breathy'])
def test_original_shards_both_tasks_no_phone_filter(tmp_path, task):
    args, path = raw_fixture(tmp_path, task)
    df = load_raw_annotations(**args)
    assert len(df) == 3
    assert df.label.iloc[0] == 1 and df.label.iloc[1] == 0
    assert pd.isna(df.label.iloc[2])
    assert df.phone.iloc[0] == 's'
    assert df.original_state.iloc[2] == 'uncertain'
    assert df.shard_path.eq(str(path.resolve())).all()
    assert df.reviewed_at.eq(123).all()


def test_fallback_filename_and_sources(tmp_path):
    args, path = raw_fixture(tmp_path, 'breathy', filename_task='creaky', annotations={
        '0.1|0.3|SP': 'non_breathy', '0.4|0.6|a': 'mllm_breathy', '0.61|0.62|a': 'breathy'})
    data = json.loads(path.read_text())
    data['auto_sp_interval_ids'] = ['0.1|0.3|SP']
    path.write_text(json.dumps(data))
    df = load_raw_annotations(**args)
    assert list(df.source) == ['automatic_silence', 'mllm_prefill', 'human']
    assert pd.isna(df.label.iloc[1])
    assert len(df) == 3


def test_span_only_shard_and_direct_directory(tmp_path):
    args, path = raw_fixture(tmp_path)
    path.write_text(json.dumps(dict(file_key='clip', annotations={}, exact_creak_spans=[[.1,.3]], uncertain_spans=[[.4,.6]])))
    args.pop('cache_dir')
    args['shard_dir'] = path.parent
    df = load_raw_annotations(**args)
    assert list(df.original_state) == ['exact_creaky', 'uncertain']
    assert df.label.iloc[0] == 1 and pd.isna(df.label.iloc[1])
    assert not df.identity_verified.any()


def test_bad_shards_and_missing_batches_fail(tmp_path):
    args, path = raw_fixture(tmp_path)
    with pytest.raises(ValueError, match='not registered'):
        load_raw_annotations(**args, batches=['99'])
    path.write_text('{broken')
    with pytest.raises(ValueError, match='Unreadable'):
        load_raw_annotations(**args)
    path.write_text(json.dumps({'annotations': {}}))
    with pytest.raises(ValueError, match='file_key'):
        load_raw_annotations(**args)


def test_missing_audio_and_conflicting_labels_fail(tmp_path):
    args, path = raw_fixture(tmp_path)
    extra = path.with_name('extra.json')
    data = json.loads(path.read_text())
    data['annotations']['0.1|0.3|s'] = 'non_creaky'
    extra.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Conflicting duplicate'):
        load_raw_annotations(**args)
    extra.unlink()
    (args['audio_dirs'][0] / 'clip.wav').unlink()
    with pytest.raises(FileNotFoundError, match='No audio'):
        load_raw_annotations(**args)


def test_out_of_bounds_preserved_for_unscored_report(tmp_path):
    args, _ = raw_fixture(tmp_path, annotations={'0.9|1.5|a': 'creaky'})
    assert load_raw_annotations(**args).iloc[0].input_status == 'out_of_audio_bounds'


def review_module():
    path = Path(__file__).resolve().parents[1] / 'scripts/review_annotation_quality.py'
    spec = importlib.util.spec_from_file_location('raw_review_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('task', ['creaky', 'breathy'])
def test_raw_quality_cli_both_tasks(tmp_path, monkeypatch, task):
    trainer, _ = setup(tmp_path)
    backbone = tmp_path / 'backbone'
    trainer.model.wavlm.save_pretrained(backbone)
    trainer.config['model']['backbone'] = str(backbone)
    trainer._best_val_logits = trainer._best_val_labels = None
    trainer.save_checkpoint(1, {'f1': .5, 'auc_pr': .5}, only_latest=True)
    raw_root = tmp_path / 'raw'
    raw_root.mkdir()
    args, shard = raw_fixture(raw_root, task, annotations={
        '0.1|0.3|s': task, '0.31|0.311|a': task, '0.4|0.5|a': 'uncertain', '0.9|1.5|a': task})
    out = tmp_path / 'report'
    monkeypatch.setattr(sys, 'argv', ['review', '--checkpoint', str(trainer.ckpt_dir/'latest.pt'),
        '--annotation-cache', str(args['cache_dir']), '--audio-dir', str(args['audio_dirs'][0]),
        '--batches', '1', '--dataset-id', 'd', '--task', task, '--output-dir', str(out)])
    review_module().main()
    rows = pd.read_csv(out / 'all_scores.csv')
    assert len(rows) == 4
    assert rows.shard_path.eq(str(shard.resolve())).all()
    assert 'interval_too_short' in rows.review_reason.tolist()
    assert 'out_of_audio_bounds' in rows.review_reason.tolist()
    queues = pd.read_csv(out / 'review_queues.csv')
    assert 'unscored_or_uncertain' in queues.queue.tolist()
    trainer.writer.close()


def test_delta_missing_frame_weights_refused(tmp_path, monkeypatch):
    trainer, _ = setup(tmp_path)
    backbone = tmp_path / 'backbone'
    trainer.model.wavlm.save_pretrained(backbone)
    trainer.config['model']['backbone'] = str(backbone)
    trainer.config['checkpoint']['save_mode'] = 'delta'
    trainer._best_val_logits = trainer._best_val_labels = None
    trainer.save_checkpoint(1, {'f1': .5, 'auc_pr': .5}, only_latest=True)
    ckpt_path = trainer.ckpt_dir / 'latest.pt'
    ckpt = torch.load(ckpt_path, weights_only=False)
    ckpt['model_state_dict'] = {k:v for k,v in ckpt['model_state_dict'].items() if not k.startswith('frame_heads.creaky.')}
    torch.save(ckpt, ckpt_path)
    manual = trainer.train_loader.dataset.base_df
    csv = tmp_path / 'manual.csv'
    manual.to_csv(csv, index=False)
    monkeypatch.setattr(sys, 'argv', ['review', '--checkpoint', str(ckpt_path), '--manual-manifest', str(csv),
        '--task', 'creaky', '--output-dir', str(tmp_path/'review')])
    with pytest.raises(ValueError, match='lacks trained creaky'):
        review_module().main()
    trainer.writer.close()
