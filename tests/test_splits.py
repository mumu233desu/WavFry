import pytest
import pandas as pd
from wavfry.data.splits import isolate_split_speakers, make_batch_split, make_speaker_split


def _build_dummy_df():
    data = [
        # Train speakers: U001, U002, U003
        {'speaker_id': 'U001', 'batch_split': 'train', 'label': 0},
        {'speaker_id': 'U001', 'batch_split': 'train', 'label': 1},
        {'speaker_id': 'U002', 'batch_split': 'train', 'label': 0},
        {'speaker_id': 'U003', 'batch_split': 'train', 'label': 1},
        # Val speakers: U002 (overlaps), U004 (unseen)
        {'speaker_id': 'U002', 'batch_split': 'val', 'label': 0},
        {'speaker_id': 'U004', 'batch_split': 'val', 'label': 1},
        # Test speakers: U003 (overlaps), U005 (unseen)
        {'speaker_id': 'U003', 'batch_split': 'test', 'label': 0},
        {'speaker_id': 'U005', 'batch_split': 'test', 'label': 1},
    ]
    return pd.DataFrame(data)


def test_isolate_split_speakers_drop_from_eval():
    df = _build_dummy_df()
    train_df = df[df['batch_split'] == 'train'].copy()
    val_df = df[df['batch_split'] == 'val'].copy()
    test_df = df[df['batch_split'] == 'test'].copy()

    t_clean, v_clean, te_clean = isolate_split_speakers(
        train_df, val_df, test_df, enabled=True
    )

    # Train must keep all rows (U001, U002, U003)
    assert len(t_clean) == 4
    assert set(t_clean['speaker_id']) == {'U001', 'U002', 'U003'}

    # Val must drop U002, keeping only U004
    assert len(v_clean) == 1
    assert set(v_clean['speaker_id']) == {'U004'}

    # Test must drop U003, keeping only U005
    assert len(te_clean) == 1
    assert set(te_clean['speaker_id']) == {'U005'}

    # No overlap
    assert len(set(t_clean['speaker_id']) & set(v_clean['speaker_id'])) == 0
    assert len(set(t_clean['speaker_id']) & set(te_clean['speaker_id'])) == 0


def test_isolate_split_speakers_disabled():
    df = _build_dummy_df()
    train_df = df[df['batch_split'] == 'train'].copy()
    val_df = df[df['batch_split'] == 'val'].copy()

    t_clean, v_clean, _ = isolate_split_speakers(
        train_df, val_df, enabled=False
    )
    assert len(t_clean) == len(train_df)
    assert len(v_clean) == len(val_df)
    assert 'U002' in set(v_clean['speaker_id'])


def test_make_batch_split_integration():
    df = _build_dummy_df()
    t_clean, v_clean, te_clean = make_batch_split(df, isolate=True)
    assert set(t_clean['speaker_id']) == {'U001', 'U002', 'U003'}
    assert set(v_clean['speaker_id']) == {'U004'}
    assert set(te_clean['speaker_id']) == {'U005'}


def test_make_speaker_split_integration():
    df = _build_dummy_df()
    # Explicit speaker split: val speaker U002; train has U001, U002, U003
    t_clean, v_clean = make_speaker_split(df, val_speakers=['U002'], isolate=True)
    assert 'U002' not in set(t_clean['speaker_id'])  # make_speaker_split default filters out val_speakers from train
    assert set(v_clean['speaker_id']) == {'U002'}
    assert len(set(t_clean['speaker_id']) & set(v_clean['speaker_id'])) == 0

