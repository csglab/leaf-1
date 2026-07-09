"""Integration tests that exercise the full bedtools-backed fragment pipeline."""

import numpy as np
import pytest

from leaf1.common.dna_fragments_dataset import DNAFragmentDataset
from leaf1.common.get_fragment import (
    initialize_iterable_get_fragment,
    random_access_get_fragment,
)


# ---------- get_fragment end-to-end ---------------------------------------

def test_random_access_get_fragment_returns_expected_record(
    needs_bedtools, tokenizer, tiny_fa, tiny_bed
):
    """Run the full pipeline on a single BED line and check the returned dict."""
    record = random_access_get_fragment(
        tokenizer=tokenizer,
        bed_file=str(tiny_bed),
        ref_genome=str(tiny_fa),
        line_number=1,  # chr1\t1000\t1150 in the fixture
        max_token_length=128,
        min_flank_token_length=8,
        seed=0,
    )
    assert record["chromosome"] == "chr1"
    assert record["fragment_start"] == 1000
    assert record["fragment_end"] == 1150
    assert record["n_token_frag"] >= 1
    assert record["n_token_left"] >= 0
    assert record["n_token_right"] >= 0
    assert 0.0 <= record["relative_frequency_of_n"] <= 1.0
    assert isinstance(record["reverse_complemented"], (bool, np.bool_))
    assert isinstance(record["final_sequence"], str)
    assert len(record["final_sequence"]) > 0


def test_initialize_iterable_get_fragment_is_callable_repeatedly(
    needs_bedtools, tokenizer, tiny_fa, tiny_bed
):
    """The closure returned by initialize_iterable_get_fragment can serve many calls."""
    fn = initialize_iterable_get_fragment(
        tokenizer=tokenizer,
        bed_file=str(tiny_bed),
        ref_genome=str(tiny_fa),
        max_token_length=128,
        min_flank_token_length=8,
        seed=1,
    )
    seen_chrs = set()
    for line_number in range(6):
        record = fn(line_number)
        seen_chrs.add(record["chromosome"])
        assert "raw_fragment_dna" in record
        assert len(record["raw_fragment_dna"]) == record["fragment_end"] - record["fragment_start"]
    # Fixture spans both chroms.
    assert seen_chrs == {"chr1", "chr2"}


def test_reverse_complement_path_extracts_complementary_body(
    needs_bedtools, tokenizer, tiny_fa, tiny_bed
):
    """Forced reverse complement should yield a body that is the RC of the forward body."""
    from leaf1.common.get_fragment import reverse_complement

    # Forward run with a seed that yields reverse_complemented = False
    # then a run that yields True is hard to engineer because the seed governs
    # rng.uniform(); instead we run many seeds and grab one of each.
    fn_forward = None
    fn_reverse = None
    for seed in range(50):
        fn = initialize_iterable_get_fragment(
            tokenizer=tokenizer, bed_file=str(tiny_bed), ref_genome=str(tiny_fa),
            max_token_length=128, min_flank_token_length=8, seed=seed,
        )
        rec = fn(1)  # chr1\t1000\t1150
        if rec["reverse_complemented"] and fn_reverse is None:
            fn_reverse = rec
        elif not rec["reverse_complemented"] and fn_forward is None:
            fn_forward = rec
        if fn_forward and fn_reverse:
            break

    assert fn_forward is not None
    assert fn_reverse is not None
    # Same coordinates, complementary body sequence.
    assert fn_forward["fragment_start"] == fn_reverse["fragment_start"]
    assert fn_forward["fragment_end"] == fn_reverse["fragment_end"]
    assert fn_reverse["raw_fragment_dna"] == reverse_complement(fn_forward["raw_fragment_dna"])


# ---------- DNAFragmentDataset --------------------------------------------

def test_dna_fragments_dataset_iter_returns_expected_keys(
    needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest
):
    """Smoke test: build the dataset and pull a few items."""
    ds = DNAFragmentDataset(
        tokenizer_file=str(tokenizer_path),
        ref_genome_file=str(tiny_fa),
        n_fragment_file=str(tiny_manifest),
        n_fragment_per_file=4,
        max_token_length=128,
        min_flank_token_length=8,
        seed=0,
    )
    assert len(ds) == 4
    item = ds[0]
    expected_keys = {
        "input_ids", "attention_mask",
        "file", "modality", "group", "line_number", "chr",
        "n_token_frag", "n_token_left", "n_token_right",
        "n_freq", "reverse_complemented",
    }
    assert expected_keys.issubset(set(item.keys()))
    assert item["input_ids"].shape == (128,)
    assert item["modality"] == "[CFDNA]"
    assert item["group"] == "healthy"
    assert item["chr"] in {"chr1", "chr2"}


def test_dna_fragments_dataset_indices_within_total_fragments(
    needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest
):
    """Accessing every index of the sampled dataset should succeed."""
    ds = DNAFragmentDataset(
        tokenizer_file=str(tokenizer_path),
        ref_genome_file=str(tiny_fa),
        n_fragment_file=str(tiny_manifest),
        n_fragment_per_file=3,
        max_token_length=128,
        min_flank_token_length=8,
        seed=0,
    )
    assert len(ds) == 3
    for i in range(len(ds)):
        item = ds[i]
        assert item["line_number"] >= 0
        assert item["line_number"] < 6  # fixture has 6 BED records


def test_dna_fragments_dataset_out_of_bounds_raises(
    needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest
):
    ds = DNAFragmentDataset(
        tokenizer_file=str(tokenizer_path),
        ref_genome_file=str(tiny_fa),
        n_fragment_file=str(tiny_manifest),
        n_fragment_per_file=2,
        max_token_length=128,
        min_flank_token_length=8,
        seed=0,
    )
    with pytest.raises(IndexError):
        ds[len(ds)]
    with pytest.raises(IndexError):
        ds[-1]


# ---------- TrainingDNAFragmentDataset (streaming) -----------------------

def test_training_dataset_iterates_at_least_one_fragment(
    needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest
):
    """Pull a single fragment from the streaming dataset to confirm the iter path runs."""
    from leaf1.common.pytorch_dataset import TrainingDNAFragmentDataset

    ds = TrainingDNAFragmentDataset(
        tokenizer_file=str(tokenizer_path),
        ref_genome_file=str(tiny_fa),
        n_fragment_file=str(tiny_manifest),
        n_fragments_per_iteration=2,
        max_token_length=128,
        min_flank_token_length=8,
        seed=0,
        n_shards=1,
    )
    it = iter(ds)
    first = next(it)
    assert "input_ids" in first and "attention_mask" in first
    assert first["input_ids"].shape == (128,)
