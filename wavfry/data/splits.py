import pandas as pd

def make_speaker_split(df, val_speakers):
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
        return train_df, val_df

    # ── Default: split by explicit speaker ID list ────────────────────────────
    val_df   = df[df['speaker_id'].isin(val_speakers)].copy()
    train_df = df[~df['speaker_id'].isin(val_speakers)].copy()
        
    val_df['split']   = 'val'
    train_df['split'] = 'train'
    
    return train_df, val_df
