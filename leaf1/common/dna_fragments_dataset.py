"""Random-access validation dataset.

`DNAFragmentDataset` samples a fixed number of line numbers per BED file at init
and serves them via ``__getitem__``. Used as the eval/validation dataset during
pretraining so HuggingFace Trainer can compute deterministic MLM metrics.

CSV manifest format (comma-separated): sample_id, modality, bed_file, n_fragment, group.
"""

import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

from .get_fragment import check_ablation, initialize_iterable_get_fragment, modality_slot_token
from .modality import resolve_modality_token


class DNAFragmentDataset:
    def __init__(self, tokenizer_file, ref_genome_file, n_fragment_file,
                 n_fragment_per_file=1000, max_token_length=1024,
                 min_flank_token_length=100, reshuffle=False, seed=None,
                 return_token_type_ids=False, return_special_tokens_mask=False, ablation="none"):
        self.ablation = check_ablation(ablation)
        self.ref_genome_file = ref_genome_file
        self.total_length = max_token_length
        self.min_flank = min_flank_token_length
        self.n_fragment_per_file = n_fragment_per_file
        self.reshuffle = reshuffle
        self.set_seed(seed if seed is not None else random.randint(0, 2**32 - 1))
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
        (self.file_to_sampled_indices,
         self.file_to_accessed_indices,
         self.file_to_modality,
         self.file_to_group) = self._initialize_sampling()

    def _initialize_sampling(self):
        file_to_sampled_indices = {}
        file_to_accessed_indices = {}
        file_to_modality = {}
        file_to_group = {}
        total_fragments = 0

        self.get_fragment = {}

        for _, row in tqdm(self.index_df.iterrows(), total=len(self.index_df), desc="Init get fragment"):
            bed_file = row['bed_file']
            n_fragments = min(row['n_fragment'], self.n_fragment_per_file)
            sampled_indices = self.rng.choice(row['n_fragment'], size=n_fragments, replace=False)

            file_to_sampled_indices[bed_file] = sampled_indices
            file_to_accessed_indices[bed_file] = set()
            file_to_modality[bed_file] = resolve_modality_token(self.tokenizer, row['modality'])
            file_to_group[bed_file] = row['group']
            total_fragments += n_fragments

            fragment_seed = self._get_new_seed()
            self.get_fragment[bed_file] = initialize_iterable_get_fragment(
                self.tokenizer, bed_file, self.ref_genome_file,
                self.total_length, self.min_flank, fragment_seed,
                ablation=self.ablation,
            )

        self.total_fragments = total_fragments
        return file_to_sampled_indices, file_to_accessed_indices, file_to_modality, file_to_group

    def set_seed(self, seed):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _get_new_seed(self):
        return self.rng.integers(low=0, high=2**32 - 1)

    def __len__(self):
        return self.total_fragments

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_fragments:
            raise IndexError("Index out of bounds.")

        global_idx = 0
        for bed_file, sampled_indices in self.file_to_sampled_indices.items():
            if global_idx + len(sampled_indices) > idx:
                local_idx = idx - global_idx
                line_number = sampled_indices[local_idx]
                self.file_to_accessed_indices[bed_file].add(line_number)

                if self.reshuffle and self._check_full_access():
                    (self.file_to_sampled_indices,
                     self.file_to_accessed_indices,
                     self.file_to_modality,
                     self.file_to_group) = self._initialize_sampling()

                fragment = self.get_fragment[bed_file](line_number)
                modality = self.file_to_modality[bed_file]
                group = self.file_to_group[bed_file]

                tokenized_input = self.tokenizer(
                    modality_slot_token(modality, self.ablation) + fragment['final_sequence'],
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

                data = {key: value.squeeze(0) for key, value in tokenized_input.items()}
                data['file'] = bed_file
                data['modality'] = modality
                data['group'] = group
                data['line_number'] = line_number
                data['chr'] = fragment['chromosome']
                data['n_token_frag'] = fragment['n_token_frag']
                data['n_token_left'] = fragment['n_token_left']
                data['n_token_right'] = fragment['n_token_right']
                data['n_freq'] = fragment['relative_frequency_of_n']
                data['reverse_complemented'] = fragment['reverse_complemented']
                return data

            global_idx += len(sampled_indices)

        raise IndexError("Index out of bounds.")

    def _check_full_access(self):
        for bed_file, accessed_indices in self.file_to_accessed_indices.items():
            if len(accessed_indices) < len(self.file_to_sampled_indices[bed_file]):
                return False
        return True

    def __str__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Reference Genome File: {self.ref_genome_file}\n"
            f"  N Fragment File: {self.index_df.iloc[0, 0]}\n"
            f"  N Fragment per File: {self.n_fragment_per_file}\n"
            f"  Max Token Length: {self.total_length}\n"
            f"  Min Flank Token Length: {self.min_flank}\n"
            f"  Seed: {self.seed}\n"
            f"  Number of Files: {len(self.index_df)}\n"
            f"  Total Sampled Fragments: {self.total_fragments}\n"
            f"  Reshuffle after a full read: {self.reshuffle}\n"
            f"  Tokenizer: {self.tokenizer.__class__.__name__}, max_len={self.tokenizer.model_max_length}\n)"
        )

    def __repr__(self):
        return self.__str__()
