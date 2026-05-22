import pandas as pd

def make_speaker_split(df, val_speakers):
    """
    Split the manifest DataFrame into train and validation sets based on speaker IDs.
    """
    if isinstance(val_speakers, str):
        val_speakers = [val_speakers]
        
    val_df = df[df['speaker_id'].isin(val_speakers)].copy()
    train_df = df[~df['speaker_id'].isin(val_speakers)].copy()
    
    val_df['split'] = 'val'
    train_df['split'] = 'train'
    
    return train_df, val_df
