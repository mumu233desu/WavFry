import math
import random
from typing import List, Iterator
import pandas as pd
import torch
from torch.utils.data import Sampler

class GroupedBatchSampler(Sampler[List[int]]):
    """
    每个 batch 混合 Anchor（同 speaker 的 creaky/non-creaky 对）
    和 Random（全局随机）片段。
    anchor_slots 根据数据集组成自动计算，确保每条数据 ~1 次/epoch。
    """
    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int,
        anchor_speakers: List[str],          # 全量标注的 speaker ID 列表
        anchor_speakers_per_batch: int = 2,  # 每个 batch 抽几个 anchor speaker
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.anchor_speakers = anchor_speakers
        self.anchor_speakers_per_batch = anchor_speakers_per_batch
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        
        # Calculate anchor_slots and random_slots
        n_anchor = int((self.df['speaker_id'].isin(anchor_speakers)).sum())
        n_total = len(self.df)
        
        computed_slots = round(batch_size * n_anchor / max(n_total, 1))
        # Lower bound: each anchor speaker needs at least 1 creaky + 1 non-creaky
        self.anchor_slots = max(
            anchor_speakers_per_batch * 2,
            computed_slots
        )
        # Ensure anchor_slots doesn't exceed batch_size
        self.anchor_slots = min(self.anchor_slots, batch_size)
        self.random_slots = batch_size - self.anchor_slots
        
        # Preprocessing: Group by speaker_id and label
        self.anchor_buckets = {}
        for spk in anchor_speakers:
            self.anchor_buckets[spk] = {0: [], 1: []}
            
        for idx, row in self.df.iterrows():
            spk = row['speaker_id']
            lbl = int(row['label'])
            if spk in anchor_speakers:
                if lbl in [0, 1]:
                    self.anchor_buckets[spk][lbl].append(idx)
                    
        # Filter active anchor speakers that actually have both creaky and non-creaky samples
        self.active_anchor_speakers = [
            spk for spk in anchor_speakers
            if len(self.anchor_buckets[spk][0]) > 0 and len(self.anchor_buckets[spk][1]) > 0
        ]
        
        # Determine random pool base
        sampled_indices = [
            idx for idx, row in self.df.iterrows()
            if row['speaker_id'] not in anchor_speakers
        ]
        
        # Calculate expected number of batches per epoch
        self.total_batches = len(self.df) // batch_size if drop_last else math.ceil(len(self.df) / batch_size)
        needed_random = self.total_batches * self.random_slots
        
        if len(sampled_indices) < needed_random:
            # Fallback to including all data if sampled data is scarce
            if len(sampled_indices) > 0:
                print(
                    f"[WARNING] Insufficient sampled data ({len(sampled_indices)} samples) "
                    f"for {needed_random} random slots. Populating random pool with all data."
                )
            self.random_pool_base = list(range(len(self.df)))
        else:
            self.random_pool_base = sampled_indices
            
    def _draw_anchor_samples(self, speaker: str, label: int, n: int, rng: random.Random) -> List[int]:
        bucket = self.anchor_buckets[speaker][label]
        if not bucket:
            return []
        
        samples = []
        if len(bucket) >= n:
            samples = rng.sample(bucket, n)
        else:
            samples = list(bucket)
            # Fill the rest with replacement
            while len(samples) < n:
                samples.append(rng.choice(bucket))
        return samples

    def __iter__(self) -> Iterator[List[int]]:
        # Initialize RNG with seed + epoch to ensure variety but reproducibility
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        
        # Shuffle the random pool base for this epoch
        random_pool = list(self.random_pool_base)
        rng.shuffle(random_pool)
        
        # Pointer for the random pool
        random_pointer = 0
        
        slots_per_speaker = self.anchor_slots // self.anchor_speakers_per_batch
        slots_per_label = slots_per_speaker // 2
        
        for _ in range(self.total_batches):
            batch = []
            
            # 1. Sample anchor speakers
            if self.active_anchor_speakers and self.anchor_speakers_per_batch > 0:
                # Randomly choose active anchor speakers for this batch
                chosen_speakers = rng.sample(
                    self.active_anchor_speakers,
                    min(self.anchor_speakers_per_batch, len(self.active_anchor_speakers))
                )
                
                # Draw balanced creaky/non-creaky pairs for each chosen speaker
                for spk in chosen_speakers:
                    creaky_samples = self._draw_anchor_samples(spk, 1, slots_per_label, rng)
                    non_creaky_samples = self._draw_anchor_samples(spk, 0, slots_per_label, rng)
                    batch.extend(creaky_samples)
                    batch.extend(non_creaky_samples)
            
            # 2. Fill the remaining slots from the random pool
            needed_random = self.batch_size - len(batch)
            if needed_random > 0:
                drawn_random = []
                while len(drawn_random) < needed_random:
                    if random_pointer >= len(random_pool):
                        # Reshuffle and reset pointer if exhausted
                        rng.shuffle(random_pool)
                        random_pointer = 0
                    
                    drawn_random.append(random_pool[random_pointer])
                    random_pointer += 1
                batch.extend(drawn_random)
                
            # Shuffle the batch to avoid all anchors being grouped at the beginning
            rng.shuffle(batch)
            yield batch
            
    def __len__(self) -> int:
        return self.total_batches
