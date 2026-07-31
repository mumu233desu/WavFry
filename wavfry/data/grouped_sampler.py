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
        dataset=None,
    ):
        self.df = df.reset_index(drop=True)
        self.dataset = dataset
        self.batch_size = batch_size
        self.anchor_speakers = anchor_speakers
        self.anchor_speakers_per_batch = anchor_speakers_per_batch
        if self.anchor_speakers_per_batch < 1:
            raise ValueError("anchor_speakers_per_batch must be >= 1")
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 1
        self._refresh_index()

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch explicitly so resumed runs reproduce the same batches."""
        self.epoch = int(epoch)

    def _refresh_index(self) -> None:
        """Rebuild index pools after epoch-specific dataset rows are generated."""
        if self.dataset is not None:
            self.df = self.dataset.df.reset_index(drop=True)

        n_anchor = int((self.df['speaker_id'].isin(self.anchor_speakers)).sum())
        computed_slots = round(self.batch_size * n_anchor / max(len(self.df), 1))
        self.anchor_slots = min(
            max(self.anchor_speakers_per_batch * 2, computed_slots),
            self.batch_size,
        )
        self.random_slots = self.batch_size - self.anchor_slots

        self.anchor_buckets = {
            speaker: {0: [], 1: []} for speaker in self.anchor_speakers
        }
        for idx, row in self.df.iterrows():
            speaker = row['speaker_id']
            label = int(row['label'])
            if speaker in self.anchor_buckets and label in {0, 1}:
                self.anchor_buckets[speaker][label].append(idx)

        self.active_anchor_speakers = [
            speaker
            for speaker in self.anchor_speakers
            if self.anchor_buckets[speaker][0] and self.anchor_buckets[speaker][1]
        ]
        if not self.active_anchor_speakers:
            # No balanced anchor pair can be formed; safely degrade to shuffled data.
            self.anchor_slots = 0
            self.random_slots = self.batch_size

        sampled_indices = [
            idx
            for idx, row in self.df.iterrows()
            if row['speaker_id'] not in self.anchor_speakers
        ]
        self.total_batches = (
            len(self.df) // self.batch_size
            if self.drop_last
            else math.ceil(len(self.df) / self.batch_size)
        )
        needed_random = self.total_batches * self.random_slots
        self.random_pool_base = (
            list(range(len(self.df)))
            if len(sampled_indices) < needed_random
            else sampled_indices
        )
        if self.total_batches > 0 and not self.random_pool_base and not self.active_anchor_speakers:
            raise ValueError("GroupedBatchSampler has no usable training samples")
            
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
        self._refresh_index()
        # Epoch is set by Trainer, making resume deterministic.
        rng = random.Random(self.seed + self.epoch)

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
                if not random_pool:
                    raise RuntimeError(
                        "GroupedBatchSampler cannot fill the batch: random pool is empty"
                    )
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
        if self.dataset is not None:
            sample_count = len(self.dataset)
            return sample_count // self.batch_size if self.drop_last else math.ceil(sample_count / self.batch_size)
        return self.total_batches
