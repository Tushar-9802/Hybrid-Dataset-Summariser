"""
dataset.py — HDF5 DataLoader + CurriculumSampler
Reads engineering.h5, constructs causal LM training sequences with prompt wrapping.

Handles:
  - Paper/video/cross-modal datasets from HDF5
  - Prompt template wrapping with proper label masking (-100 for prompt tokens)
  - Phase-aware curriculum sampling with replay buffer
  - BOS deduplication (HDF5 stores pre-tokenized with BOS)
"""

import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, Sampler, DataLoader

log = logging.getLogger(__name__)

# Prompt template matching evaluate.py
PROMPT_PREFIX = "Summarize the following text.\n\nText: "
PROMPT_SUFFIX = "\n\nSummary:"

# Split mapping
SPLIT_MAP = {"train": 0, "val": 1, "test": 2}


class HDF5SummarizationDataset(Dataset):
    """
    Single-modality dataset (papers OR videos) from HDF5.

    Each sample becomes a causal LM sequence:
        [BOS] prefix source suffix label [EOS]
    Labels = -100 for all prompt positions, real IDs for summary + EOS.
    """

    def __init__(
        self,
        hdf5_path: str,
        tokenizer,
        group: str = "papers",       # "papers" or "videos"
        split: str = "train",
        max_total_len: int = 1280,
        indices: Optional[list] = None,  # override index selection (for replay buffer)
    ):
        self.tokenizer = tokenizer
        self.max_total_len = max_total_len
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id
        self.group_name = group

        # Tokenize prompt template parts once
        self.prefix_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)

        # Load HDF5 into RAM (11 MB total — trivial)
        split_idx = SPLIT_MAP[split]
        with h5py.File(hdf5_path, "r") as hf:
            grp = hf[group]
            splits = grp["split"][:]
            mask = splits == split_idx

            if indices is not None:
                # Override: use specific indices (for replay buffer)
                sel = np.array(indices)
            else:
                sel = np.where(mask)[0]

            self.input_ids = grp["input_ids"][sel]          # (N, 1024)
            self.attention_mask = grp["attention_mask"][sel] # (N, 1024)
            self.labels_raw = grp["labels"][sel]             # (N, 256)
            self.modality = grp["modality"][sel]             # (N,)
            self.sample_ids = grp["sample_id"][sel]          # (N,) strings

        self.n = len(self.input_ids)
        log.info(f"Loaded {self.n} {group}/{split} samples")

    def __len__(self):
        return self.n

    def _strip_padding(self, ids: np.ndarray, mask: np.ndarray) -> list:
        """Extract real token IDs using attention mask."""
        real_len = int(mask.sum())
        return ids[:real_len].tolist()

    def _strip_label_padding(self, label_ids: np.ndarray) -> list:
        """Extract real label IDs (remove pad tokens)."""
        return [int(t) for t in label_ids if t != self.pad_id]

    def __getitem__(self, idx):
        # Get raw tokens
        source_ids = self._strip_padding(self.input_ids[idx], self.attention_mask[idx])
        label_ids = self._strip_label_padding(self.labels_raw[idx])

        # Strip leading BOS if present (HDF5 was tokenized with add_special_tokens=True)
        if source_ids and source_ids[0] == self.bos_id:
            source_ids = source_ids[1:]
        if label_ids and label_ids[0] == self.bos_id:
            label_ids = label_ids[1:]

        # Build causal LM sequence:
        # [BOS] + prefix + source + suffix + label + [EOS]
        prompt_ids = [self.bos_id] + self.prefix_ids + source_ids + self.suffix_ids
        full_ids = prompt_ids + label_ids + [self.eos_id]

        # Truncate if too long (sacrifice source end, keep label intact)
        if len(full_ids) > self.max_total_len:
            # How much prompt can we keep?
            label_len = len(label_ids) + 1  # +1 for EOS
            max_prompt = self.max_total_len - label_len
            if max_prompt < len(self.prefix_ids) + 20:
                # Emergency: truncate label too
                max_prompt = self.max_total_len // 2
                label_portion = self.max_total_len - max_prompt
                prompt_ids = prompt_ids[:max_prompt]
                label_ids = label_ids[:label_portion - 1]
                full_ids = prompt_ids + label_ids + [self.eos_id]
            else:
                prompt_ids = prompt_ids[:max_prompt]
                full_ids = prompt_ids + label_ids + [self.eos_id]

        prompt_len = len(prompt_ids)
        seq_len = len(full_ids)

        # Build labels: -100 for prompt positions, real IDs for summary
        labels = [-100] * prompt_len + label_ids + [self.eos_id]

        # Pad to max_total_len
        pad_len = self.max_total_len - seq_len
        input_ids_padded = full_ids + [self.pad_id] * pad_len
        labels_padded = labels + [-100] * pad_len
        attention_mask = [1] * seq_len + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels_padded, dtype=torch.long),
            "modality_id": int(self.modality[idx]),
        }


class CrossModalDataset(Dataset):
    """
    Cross-modal pair dataset for CrossCLR contrastive learning.
    Returns both paper and video sides of each pair.
    """

    def __init__(
        self,
        hdf5_path: str,
        tokenizer,
        split: str = "train",
        max_total_len: int = 1280,
    ):
        self.tokenizer = tokenizer
        self.max_total_len = max_total_len
        self.pad_id = tokenizer.pad_token_id
        self.bos_id = tokenizer.bos_token_id
        self.eos_id = tokenizer.eos_token_id

        self.prefix_ids = tokenizer.encode(PROMPT_PREFIX, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(PROMPT_SUFFIX, add_special_tokens=False)

        split_idx = SPLIT_MAP[split]
        with h5py.File(hdf5_path, "r") as hf:
            cm = hf["cross_modal"]
            splits = cm["split"][:]
            sel = np.where(splits == split_idx)[0]

            self.paper_input_ids = cm["paper_input_ids"][sel]
            self.paper_attn = cm["paper_attention_mask"][sel]
            self.paper_labels = cm["paper_labels"][sel]
            self.video_input_ids = cm["video_input_ids"][sel]
            self.video_attn = cm["video_attention_mask"][sel]
            self.video_labels = cm["video_labels"][sel]
            self.similarity = cm["similarity"][sel]
            self.pair_ids = cm["pair_id"][sel]

        self.n = len(self.paper_input_ids)
        log.info(f"Loaded {self.n} cross-modal/{split} pairs")

    def __len__(self):
        return self.n

    def _build_sequence(self, source_ids_raw, source_mask, label_ids_raw):
        """Build a single causal LM sequence from raw HDF5 arrays."""
        real_len = int(source_mask.sum())
        source_ids = source_ids_raw[:real_len].tolist()
        label_ids = [int(t) for t in label_ids_raw if t != self.pad_id]

        # Strip BOS
        if source_ids and source_ids[0] == self.bos_id:
            source_ids = source_ids[1:]
        if label_ids and label_ids[0] == self.bos_id:
            label_ids = label_ids[1:]

        prompt_ids = [self.bos_id] + self.prefix_ids + source_ids + self.suffix_ids
        full_ids = prompt_ids + label_ids + [self.eos_id]

        if len(full_ids) > self.max_total_len:
            label_len = len(label_ids) + 1
            max_prompt = max(self.max_total_len - label_len, len(self.prefix_ids) + 20)
            prompt_ids = prompt_ids[:max_prompt]
            avail = self.max_total_len - len(prompt_ids) - 1
            label_ids = label_ids[:avail]
            full_ids = prompt_ids + label_ids + [self.eos_id]

        prompt_len = len(prompt_ids)
        seq_len = len(full_ids)
        labels = [-100] * prompt_len + label_ids + [self.eos_id]

        pad_len = self.max_total_len - seq_len
        return {
            "input_ids": torch.tensor(full_ids + [self.pad_id] * pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1] * seq_len + [0] * pad_len, dtype=torch.long),
            "labels": torch.tensor(labels + [-100] * pad_len, dtype=torch.long),
        }

    def __getitem__(self, idx):
        paper = self._build_sequence(
            self.paper_input_ids[idx], self.paper_attn[idx], self.paper_labels[idx]
        )
        video = self._build_sequence(
            self.video_input_ids[idx], self.video_attn[idx], self.video_labels[idx]
        )
        return {
            "paper_input_ids": paper["input_ids"],
            "paper_attention_mask": paper["attention_mask"],
            "paper_labels": paper["labels"],
            "video_input_ids": video["input_ids"],
            "video_attention_mask": video["attention_mask"],
            "video_labels": video["labels"],
            "similarity": torch.tensor(float(self.similarity[idx]), dtype=torch.float32),
        }


class CurriculumSampler(Sampler):
    """
    Phase-aware batch sampler that draws from paper/video/pair pools
    according to curriculum ratios.

    Phase 1: 100% papers
    Phase 2: 50% papers + 40% videos + 10% pairs
    Phase 3: 30% papers + 60% videos + 10% pairs
    """

    def __init__(
        self,
        paper_dataset: HDF5SummarizationDataset,
        video_dataset: Optional[HDF5SummarizationDataset],
        pair_dataset: Optional[CrossModalDataset],
        batch_size: int,
        paper_ratio: float = 1.0,
        video_ratio: float = 0.0,
        pair_ratio: float = 0.0,
        replay_indices: Optional[list] = None,
        replay_ratio: float = 0.0,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.paper_ratio = paper_ratio
        self.video_ratio = video_ratio
        self.pair_ratio = pair_ratio
        self.replay_ratio = replay_ratio
        self.rng = random.Random(seed)

        # Build index pools with source tags
        self.paper_indices = list(range(len(paper_dataset)))
        self.video_indices = list(range(len(video_dataset))) if video_dataset else []
        self.pair_indices = list(range(len(pair_dataset))) if pair_dataset else []
        self.replay_indices = replay_indices or []

        # Total steps per epoch = enough to see all papers once
        n_paper = len(self.paper_indices)
        n_video = len(self.video_indices)
        effective_per_epoch = max(n_paper, n_video) if n_video > 0 else n_paper
        self.total_batches = effective_per_epoch // batch_size

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        """Yield (source_type, index) tuples for each batch."""
        # Shuffle pools
        paper_pool = self.paper_indices.copy()
        video_pool = self.video_indices.copy()
        pair_pool = self.pair_indices.copy()
        replay_pool = self.replay_indices.copy()
        self.rng.shuffle(paper_pool)
        self.rng.shuffle(video_pool)
        self.rng.shuffle(pair_pool)
        self.rng.shuffle(replay_pool)

        pp, vp, prp, rp = 0, 0, 0, 0  # pool pointers

        for _ in range(self.total_batches):
            batch = []
            for _ in range(self.batch_size):
                r = self.rng.random()

                # Replay carves out of paper ratio
                effective_paper = self.paper_ratio - self.replay_ratio

                if r < effective_paper:
                    if pp >= len(paper_pool):
                        self.rng.shuffle(paper_pool)
                        pp = 0
                    batch.append(("paper", paper_pool[pp]))
                    pp += 1
                elif r < effective_paper + self.replay_ratio:
                    if replay_pool:
                        if rp >= len(replay_pool):
                            self.rng.shuffle(replay_pool)
                            rp = 0
                        batch.append(("replay", replay_pool[rp]))
                        rp += 1
                    else:
                        # Fallback to paper if no replay buffer
                        if pp >= len(paper_pool):
                            self.rng.shuffle(paper_pool)
                            pp = 0
                        batch.append(("paper", paper_pool[pp]))
                        pp += 1
                elif r < effective_paper + self.replay_ratio + self.video_ratio:
                    if video_pool:
                        if vp >= len(video_pool):
                            self.rng.shuffle(video_pool)
                            vp = 0
                        batch.append(("video", video_pool[vp]))
                        vp += 1
                else:
                    if pair_pool:
                        if prp >= len(pair_pool):
                            self.rng.shuffle(pair_pool)
                            prp = 0
                        batch.append(("pair", pair_pool[prp]))
                        prp += 1

            yield batch


def build_dataloader(
    hdf5_path: str,
    tokenizer,
    phase: int,
    batch_size: int = 3,
    max_total_len: int = 1280,
    replay_indices: Optional[list] = None,
    num_workers: int = 0,
    seed: int = 42,
):
    """
    Build phase-appropriate DataLoader.

    Returns separate loaders for each modality to avoid collation complexity.
    The trainer interleaves them according to curriculum ratios.
    """
    paper_ds = HDF5SummarizationDataset(
        hdf5_path, tokenizer, group="papers", split="train",
        max_total_len=max_total_len,
    )

    video_ds = None
    pair_ds = None

    if phase >= 2:
        video_ds = HDF5SummarizationDataset(
            hdf5_path, tokenizer, group="videos", split="train",
            max_total_len=max_total_len,
        )
    if phase >= 2:
        pair_ds = CrossModalDataset(
            hdf5_path, tokenizer, split="train",
            max_total_len=max_total_len,
        )

    # Replay buffer dataset (subset of papers)
    replay_ds = None
    if replay_indices and phase >= 2:
        replay_ds = HDF5SummarizationDataset(
            hdf5_path, tokenizer, group="papers", split="train",
            max_total_len=max_total_len, indices=replay_indices,
        )

    # Curriculum ratios
    ratios = {
        1: {"paper": 1.0, "video": 0.0, "pair": 0.0, "replay": 0.0},
        2: {"paper": 0.50, "video": 0.40, "pair": 0.10, "replay": 0.10},
        3: {"paper": 0.30, "video": 0.60, "pair": 0.10, "replay": 0.10},
    }[phase]

    # Per-loader generators so shuffle order is a clean function of `seed`,
    # not coupled to how much global RNG state was consumed before iteration.
    def _gen(offset: int) -> torch.Generator:
        g = torch.Generator()
        g.manual_seed(seed + offset)
        return g

    return {
        "paper": DataLoader(paper_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True,
                            generator=_gen(0)),
        "video": DataLoader(video_ds, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True, drop_last=True,
                            generator=_gen(1))
                 if video_ds else None,
        "pair": DataLoader(pair_ds, batch_size=batch_size, shuffle=True,
                           num_workers=num_workers, pin_memory=True, drop_last=True,
                           generator=_gen(2))
                if pair_ds else None,
        "replay": DataLoader(replay_ds, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True, drop_last=True,
                             generator=_gen(3))
                  if replay_ds else None,
        "ratios": ratios,
        "paper_dataset": paper_ds,
    }


def build_val_loader(hdf5_path, tokenizer, batch_size=3, max_total_len=1280):
    """Build validation loaders for both modalities."""
    paper_val = HDF5SummarizationDataset(
        hdf5_path, tokenizer, group="papers", split="val", max_total_len=max_total_len,
    )
    video_val = HDF5SummarizationDataset(
        hdf5_path, tokenizer, group="videos", split="val", max_total_len=max_total_len,
    )
    return {
        "paper": DataLoader(paper_val, batch_size=batch_size, shuffle=False, pin_memory=True),
        "video": DataLoader(video_val, batch_size=batch_size, shuffle=False, pin_memory=True),
    }