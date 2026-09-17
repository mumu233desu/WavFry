from __future__ import annotations

from typing import Any
import pandas as pd


def isolate_split_speakers(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    *,
    enabled: bool = True,
    isolation_scope: str = "same_dataset",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ensure zero speaker overlap between train and evaluation splits (val/test).

    Policy: Contested speakers are removed from val and test, preserving full
    training size while guaranteeing pure zero-shot evaluation on unseen speakers.
    If isolation_scope == 'same_dataset', isolation is enforced strictly within
    each dataset_id, allowing external datasets to remain independent.
    """
    if test_df is None:
        test_df = train_df.iloc[0:0].copy()

    if "speaker_id" not in train_df.columns or "speaker_id" not in val_df.columns:
        return train_df, val_df, test_df

    if not enabled:
        return train_df, val_df, test_df

    if isolation_scope == "same_dataset" and "dataset_id" in train_df.columns and "dataset_id" in val_df.columns:
        val_clean_parts = []
        test_clean_parts = []
        all_ds = set(train_df["dataset_id"].dropna().unique()) | set(val_df["dataset_id"].dropna().unique())
        if not test_df.empty and "dataset_id" in test_df.columns:
            all_ds |= set(test_df["dataset_id"].dropna().unique())

        for ds_id in sorted(all_ds):
            ds_train = train_df[train_df["dataset_id"] == ds_id]
            ds_val = val_df[val_df["dataset_id"] == ds_id]
            ds_test = test_df[test_df["dataset_id"] == ds_id] if not test_df.empty else test_df.iloc[0:0]

            train_spks = set(ds_train["speaker_id"].dropna().unique())
            val_spks = set(ds_val["speaker_id"].dropna().unique())
            test_spks = set(ds_test["speaker_id"].dropna().unique()) if not ds_test.empty else set()

            overlap_val = train_spks & val_spks
            overlap_test = train_spks & test_spks
            total_overlap = overlap_val | overlap_test

            if total_overlap:
                print(
                    f"[Split] [{ds_id}] Speaker Isolation: Dropping {len(total_overlap)} overlapping "
                    f"speakers from val/test ({sorted(total_overlap)})."
                )
            v_clean = ds_val[~ds_val["speaker_id"].isin(overlap_val)] if not ds_val.empty else ds_val
            t_clean = ds_test[~ds_test["speaker_id"].isin(overlap_test)] if not ds_test.empty else ds_test
            val_clean_parts.append(v_clean)
            test_clean_parts.append(t_clean)

        val_clean = pd.concat(val_clean_parts, ignore_index=True) if val_clean_parts else val_df.iloc[0:0]
        test_clean = pd.concat(test_clean_parts, ignore_index=True) if test_clean_parts else test_df.iloc[0:0]
        return train_df.copy(), val_clean, test_clean

    train_spks = set(train_df["speaker_id"].dropna().unique())
    val_spks = set(val_df["speaker_id"].dropna().unique())
    test_spks = set(test_df["speaker_id"].dropna().unique()) if not test_df.empty else set()

    overlap_val = train_spks & val_spks
    overlap_test = train_spks & test_spks
    total_overlap = overlap_val | overlap_test

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
    return train_df.copy(), val_clean, test_clean


def make_speaker_split(
    df: pd.DataFrame,
    val_speakers: Any,
    *,
    isolate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the manifest DataFrame into train and validation sets based on explicit speaker IDs."""
    if val_speakers is None:
        val_speakers = []
    elif isinstance(val_speakers, str):
        val_speakers = [val_speakers]

    val_df = df[df['speaker_id'].isin(val_speakers)].copy()
    train_df = df[~df['speaker_id'].isin(val_speakers)].copy()

    val_df['split'] = 'val'
    train_df['split'] = 'train'
    train_df, val_df, _ = isolate_split_speakers(train_df, val_df, None, enabled=isolate)

    return train_df, val_df


def make_batch_split(
    df: pd.DataFrame,
    *,
    isolate: bool = True,
    isolation_scope: str = "same_dataset",
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
        train_df, val_df, test_df, enabled=isolate, isolation_scope=isolation_scope
    )
    return train_df, val_df, test_df
