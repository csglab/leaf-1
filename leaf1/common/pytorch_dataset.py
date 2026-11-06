"""Streaming training dataset.

`TrainingDNAFragmentDataset` is a torch `IterableDataset`. It reads a CSV manifest
of BED files (columns: sample_id, modality, bed_file, n_fragment, group) and emits
tokenized fragments in rounds. Each round selects up to
``n_fragments_per_iteration`` previously-unread line numbers per file, runs them
through `initialize_iterable_get_fragment`, prepends the per-file modality token
(``[MOD]``), and yields a dict with ``input_ids`` and ``attention_mask`` ready for
the HuggingFace Trainer.

Sharding has two levels: across DDP ranks, then across DataLoader workers within
each rank.
"""

import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from .get_fragment import check_ablation, initialize_iterable_get_fragment, modality_slot_token
from .modality import resolve_modality_token


class TrainingDNAFragmentDataset(IterableDataset):
    def __init__(self, tokenizer_file, ref_genome_file, n_fragment_file,
                 n_fragments_per_iteration=1000, max_token_length=1024,
                 min_flank_token_length=100, seed=None, n_shards=1,
                 return_token_type_ids=False, return_special_tokens_mask=False, ablation="none"):
        super().__init__()
        self.ablation = check_ablation(ablation)
        self.ref_genome_file = ref_genome_file
        self.total_length = max_token_length
        self.min_flank = min_flank_token_length
        self.set_seed(seed if seed is not None else random.randint(0, 2**32 - 1))
        self.n_fragments_per_iteration = n_fragments_per_iteration
        self.n_shards = n_shards
        self.return_token_type_ids = return_token_type_ids
        self.return_special_tokens_mask = return_special_tokens_mask

        if not os.path.isfile(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            model_max_length=self.total_length,
            padding_side='right',
            truncation_side='right',
            cls_token='[CLS]',
            bos_token='[CLS]',
            sep_token='[SEP]',
            eos_token='[SEP]',
            unk_token='[UNK]',
            mask_token='[MASK]',
            pad_token='[PAD]',
        )

        if not os.path.isfile(n_fragment_file):
            raise FileNotFoundError(f"n_fragment file not found: {n_fragment_file}")
        self.index_df = pd.read_csv(n_fragment_file, sep=",")
        shuffled_index = self.rng.permutation(len(self.index_df))
        self.index_df = self.index_df.iloc[shuffled_index].reset_index(drop=True)
        self.index_df['cumulative_sum'] = self.index_df['n_fragment'].cumsum()
        self.total_fragments = self._len()

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _get_new_seed(self, rng=None):
        return (self.rng if rng is None else rng).integers(low=0, high=2**32 - 1)

    def map_global_index_to_file_and_line(self, idx):
        if idx < 0 or idx >= self._len():
            raise IndexError("Index out of bounds.")
        file_row = self.index_df[self.index_df['cumulative_sum'] > idx].iloc[0]
        line_number = idx - (file_row['cumulative_sum'] - file_row['n_fragment'])
        return file_row['bed_file'], line_number

    def _len(self):
        return self.index_df['cumulative_sum'].iloc[-1]

    def __iter__(self):
        # DDP rank sharding.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            global_rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            global_rank = 0
            world_size = 1

        total_length = len(self.index_df)
        global_shard_size = total_length // world_size
        start_global = global_rank * global_shard_size
        end_global = (global_rank + 1) * global_shard_size if global_rank != world_size - 1 else total_length
        global_shard_df = self.index_df.iloc[start_global:end_global]

        # DataLoader-worker sharding within this rank.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            n_shards = 1
            shard_id = 0
            print(f"Process {global_rank}: Single process, so 1 shard.")
            rng = self.rng
        else:
            n_shards = worker_info.num_workers
            shard_id = worker_info.id
            rng = np.random.default_rng(worker_info.seed)
            print(f"Process {global_rank}, Worker {shard_id}: Started with {n_shards} shards.")

        shard_size = len(global_shard_df) // n_shards
        start_index = shard_id * shard_size
        end_index = (shard_id + 1) * shard_size if shard_id != n_shards - 1 else len(global_shard_df)
        shard_df = global_shard_df.iloc[start_index:end_index]

        position = shard_id if worker_info else 0

        fragments_read = {row['bed_file']: 0 for _, row in shard_df.iterrows()}
        file_seeds = {row['bed_file']: self._get_new_seed(rng) for _, row in shard_df.iterrows()}

        with tqdm(
            shard_df.iterrows(),
            total=len(shard_df),
            desc=f"Worker {shard_id} get frag init",
            position=position,
            leave=True,
            dynamic_ncols=True,
        ) as pbar:
            get_fragment_dict = {
                row['bed_file']: initialize_iterable_get_fragment(
                    self.tokenizer,
                    row['bed_file'],
                    self.ref_genome_file,
                    self.total_length,
                    self.min_flank,
                    seed=file_seeds[row['bed_file']],
                    ablation=self.ablation,
                )
                for _, row in pbar
            }
        print(f"Worker {shard_id}: fragment init completed.")

        all_files_exhausted = False
        while not all_files_exhausted:
            all_files_exhausted = True

            shuffled_index = rng.permutation(len(shard_df))
            shard_df = shard_df.iloc[shuffled_index].reset_index(drop=True)

            for _, row in shard_df.iterrows():
                bed_file = row['bed_file']
                total_fragments_in_file = row['n_fragment']
                modality = modality_slot_token(resolve_modality_token(self.tokenizer, row['modality']), self.ablation)

                read_count = fragments_read[bed_file]
                if read_count < total_fragments_in_file:
                    all_files_exhausted = False
                    file_rng = np.random.default_rng(file_seeds[bed_file])

                    all_indices = np.arange(total_fragments_in_file)
                    file_rng.shuffle(all_indices)

                    start_index = read_count
                    end_index = min(read_count + self.n_fragments_per_iteration, total_fragments_in_file)
                    fragment_indices = all_indices[start_index:end_index]

                    get_fragment = get_fragment_dict[bed_file]
                    for line_number in fragment_indices:
                        fragment = get_fragment(line_number)
                        tokenized_input = self.tokenizer(
                            modality + fragment['final_sequence'],
                            return_special_tokens_mask=True,
                            truncation=True,
                            padding='max_length',
                            max_length=self.total_length,
                            return_tensors='pt',
                        )
                        if not self.return_special_tokens_mask:
                            del tokenized_input['special_tokens_mask']
                        if not self.return_token_type_ids and 'token_type_ids' in tokenized_input:
                            del tokenized_input['token_type_ids']
                        yield {key: value.squeeze(0) for key, value in tokenized_input.items()}

                    fragments_read[bed_file] += end_index - start_index

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Reference Genome File: {self.ref_genome_file}\n"
            f"  N Fragment File: {self.index_df.iloc[0, 0]}\n"
            f"  N Fragment to Read in Each Pass: {self.n_fragments_per_iteration}\n"
            f"  Max Token Length: {self.total_length}\n"
            f"  Min Flank Token Length: {self.min_flank}\n"
            f"  Seed: {self.seed}\n"
            f"  Number of Files: {len(self.index_df)}\n"
            f"  Total Fragments: {self._len()}\n"
            f"  Tokenizer: {self.tokenizer.__class__.__name__}, max_len={self.tokenizer.model_max_length}\n)"
        )

    def __repr__(self):
        return self.__str__()
