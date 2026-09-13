from __future__ import annotations

from typing import Any
import pandas as pd


def isolate_split_speakers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    *,
    enabled: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure zero speaker overlap between train and evaluation splits (val/test).

    Policy: Contested speakers are removed from val and test, preserving full
    training size while guaranteeing pure zero-shot evaluation on unseen speakers.
    """
    if test_df is None:
        test_df = train_df.iloc[0:0].copy()

    if "speaker_id" not in train_df.columns or "speaker_id" not in val_df.columns:
        return train_df, val_df, test_df

    train_spks = set(train_df["speaker_id"].dropna().unique())
    val_spks = set(val_df["speaker_id"].dropna().unique())
    test_spks = set(test_df["speaker_id"].dropna().unique()) if not test_df.empty else set()

    overlap_val = train_spks & val_spks
    overlap_test = train_spks & test_spks
    total_overlap = overlap_val | overlap_test

    if not enabled:
        if total_overlap:
            print(
                f"[Split] [DATA LEAKAGE WARNING] Overlapping speakers detected between train "
                f"and val/test splits ({len(total_overlap)} speakers: {sorted(total_overlap)})! "
                f"Isolation is explicitly disabled."
            )
        return train_df, val_df, test_df

    if not total_overlap:
        print("[Split] Speaker isolation verified: Zero speaker overlap between train and val/test.")
        return train_df, val_df, test_df

    print(
        f"[Split] Speaker Isolation: Detected {len(total_overlap)} overlapping "
        f"speakers between train and val/test splits (purging from val/test to protect zero-shot eval):\n"
        f"        Overlapping Speakers: {sorted(total_overlap)}"
    )

    val_clean = val_df[~val_df["speaker_id"].isin(overlap_val)].copy()
    test_clean = test_df[~test_df["speaker_id"].isin(overlap_test)].copy() if not test_df.empty else test_df
    print(
        f"[Split] Dropped {len(val_df) - len(val_clean)} segments from Val (retained {len(val_clean)} segs, "
        f"{val_clean['speaker_id'].nunique()} unseen speakers)."
    )
    if not test_df.empty:
        print(
            f"[Split] Dropped {len(test_df) - len(test_clean)} segments from Test (retained {len(test_clean)} segs, "
            f"{test_clean['speaker_id'].nunique()} unseen speakers)."
        )
    return train_df.copy(), val_clean, test_clean


def make_speaker_split(
    df: pd.DataFrame,
    val_speakers: Any,
    *,
    isolate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the manifest DataFrame into train and validation sets based on speaker IDs.
    
    Special val_speakers values:
      - 'sampled'  : validate on all annotation_type=='sampled' rows
      - 'batch2'   : validate on all batch_id==2 rows (second annotator's batch)
    """
    if val_speakers is None:
        val_speakers = []
    elif isinstance(val_speakers, str):
        val_speakers = [val_speakers]

    # ── Special case: Batch 2 as standalone val set ───────────────────────────
    if 'batch2' in val_speakers:
        if 'batch_id' not in df.columns:
            print("[Split] WARNING: 'batch2' found in val_speakers, but 'batch_id' column is missing from manifest. "
                  "Please rebuild the manifest with: uv run python scripts/build_manifest.py")
            val_df   = df.iloc[0:0].copy()  # empty
            train_df = df.copy()
        else:
            # Safely cast to numeric, as csv.DictReader leaves it as string '2.0' or ''
            batch_ids = pd.to_numeric(df['batch_id'], errors='coerce')
            val_df   = df[batch_ids == 2].copy()
            train_df = df[batch_ids != 2].copy()  # batch_id=1 AND NaN (full speakers) both go to train
            print(f"[Split] Batch2 Split: Train {len(train_df)} segments "
                  f"(full speakers + batch1), Val {len(val_df)} segments (batch2 only).")
        val_df['split']   = 'val'
        train_df['split'] = 'train'
        train_df, val_df, _ = isolate_split_speakers(train_df, val_df, None, enabled=isolate)
        return train_df, val_df

    # ── Special case: 'sampled' keyword ──────────────────────────────────────
    if 'sampled' in val_speakers:
        if 'annotation_type' in df.columns:
            val_df = df[df['annotation_type'] == 'sampled'].copy()
            train_df = df[df['annotation_type'] == 'full'].copy()
            
            # Handle if any extra speakers are passed in addition to 'sampled'
            extra_val_speakers = [s for s in val_speakers if s != 'sampled']
            if extra_val_speakers:
                extra_val_df = train_df[train_df['speaker_id'].isin(extra_val_speakers)]
                val_df = pd.concat([val_df, extra_val_df])
                train_df = train_df[~train_df['speaker_id'].isin(extra_val_speakers)]
                
            num_train_spk = train_df['speaker_id'].nunique()
            print(f"[Split] Automated Split: Training on {num_train_spk} fully-annotated speakers ({len(train_df)} segments), "
                  f"Validating on randomly-sampled speakers + extra ({len(val_df)} segments).")
        else:
            print("[Split] WARNING: 'sampled' found in val_speakers, but 'annotation_type' column is missing from manifest. Falling back to normal speaker ID matching.")
            val_df = df[df['speaker_id'].isin(val_speakers)].copy()
            train_df = df[~df['speaker_id'].isin(val_speakers)].copy()
        val_df['split']   = 'val'
        train_df['split'] = 'train'
        train_df, val_df, _ = isolate_split_speakers(train_df, val_df, None, enabled=isolate)
        return train_df, val_df

    # ── Default: split by explicit speaker ID list ────────────────────────────
    val_df   = df[df['speaker_id'].isin(val_speakers)].copy()
    train_df = df[~df['speaker_id'].isin(val_speakers)].copy()
        
    val_df['split']   = 'val'
    train_df['split'] = 'train'
    train_df, val_df, _ = isolate_split_speakers(train_df, val_df, None, enabled=isolate)
    
    return train_df, val_df


def make_batch_split(
    df: pd.DataFrame,
    *,
    isolate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split manifest DataFrame into train, validation, and test sets based on 'batch_split'.

    Extracts:
      - train_df: rows with batch_split == 'train'
      - val_df:   rows with batch_split == 'val'
      - test_df:  rows with batch_split == 'test'
    """
    if "batch_split" not in df.columns:
        raise KeyError(
            "[DATA SPLIT ERROR] Required partition column 'batch_split' not found in dataframe! "
            "Refusing silent fallback to all-train mode. Please ensure manifest contains 'batch_split'."
        )
    train_df = df[df["batch_split"] == "train"].copy()
    val_df = df[df["batch_split"] == "val"].copy()
    test_df = df[df["batch_split"] == "test"].copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    print(
        f"[Split] Batch Split (Raw): Train {len(train_df)} segments, "
        f"Val {len(val_df)} segments, Test {len(test_df)} segments."
    )

    train_df, val_df, test_df = isolate_split_speakers(
        train_df, val_df, test_df, enabled=isolate
    )
    return train_df, val_df, test_df
