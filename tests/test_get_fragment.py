"""Unit tests for leaf1.common.get_fragment."""

import pandas as pd
import pytest

from leaf1.common import get_fragment as gf


# ---------- read_fai_file --------------------------------------------------

def test_read_fai_file_parses_chr_lengths(tiny_fai, chr_lengths):
    df = gf.read_fai_file(str(tiny_fai))
    parsed = dict(zip(df["chr"], df["length"]))
    assert parsed == chr_lengths


# ---------- read_index_file -----------------------------------------------

def test_read_index_file_empty_returns_zero_zero():
    empty_df = pd.DataFrame(columns=["line", "offset"])
    assert gf.read_index_file(empty_df, line_number=10) == [0, 0]


def test_read_index_file_before_min_returns_zero_zero():
    df = pd.DataFrame({"line": [1000, 2000], "offset": [50000, 100000]})
    assert gf.read_index_file(df, line_number=500) == [0, 0]


def test_read_index_file_after_max_returns_last_row():
    df = pd.DataFrame({"line": [1000, 2000], "offset": [50000, 100000]})
    assert gf.read_index_file(df, line_number=5000) == [2000, 100000]


def test_read_index_file_in_range_returns_nearest_leq():
    df = pd.DataFrame({"line": [1000, 2000, 3000], "offset": [50000, 100000, 150000]})
    assert gf.read_index_file(df, line_number=2500) == [2000, 100000]
    assert gf.read_index_file(df, line_number=2000) == [2000, 100000]
    assert gf.read_index_file(df, line_number=1000) == [1000, 50000]


# ---------- get_bed_line --------------------------------------------------

def test_get_bed_line_first_line(tiny_bed):
    line = gf.get_bed_line(str(tiny_bed), offset=0, line_number=0)
    assert line == "chr1\t100\t250"


def test_get_bed_line_nth_line(tiny_bed):
    line = gf.get_bed_line(str(tiny_bed), offset=0, line_number=2)
    assert line == "chr1\t2500\t2680"


def test_get_bed_line_out_of_range_raises(tiny_bed):
    with pytest.raises(ValueError):
        gf.get_bed_line(str(tiny_bed), offset=0, line_number=999)


# ---------- reverse_complement --------------------------------------------

def test_reverse_complement_basic():
    assert gf.reverse_complement("ACGT") == "ACGT"
    assert gf.reverse_complement("AAAA") == "TTTT"
    assert gf.reverse_complement("ACCGGT") == "ACCGGT"  # palindrome


def test_reverse_complement_preserves_n():
    assert gf.reverse_complement("ANNA") == "TNNT"


def test_reverse_complement_passthrough_for_unknown_chars():
    # Unknown chars survive via .get(base, base).
    assert gf.reverse_complement("aXc") == "cXa"


# ---------- process_line --------------------------------------------------

@pytest.fixture
def chr_lengths_dict():
    return {"chr1": 5000, "chr2": 3000}


def test_process_line_middle_of_chrom(chr_lengths_dict):
    """Fragment with plenty of room on both flanks splits the budget evenly."""
    result = gf.process_line(
        line="chr1\t1000\t1150",
        chr_lengths=chr_lengths_dict,
        reverse_complemented=False,
        max_token_length=64,
        min_flank_token_length=8,
    )
    chr, start_adj, end_adj, frag_start, frag_end, adj_frag_start, adj_frag_end = result
    assert chr == "chr1"
    assert frag_start == 1000 and frag_end == 1150
    # Symmetric split: 93 extra each side, plus 56 min each side, around the 150bp frag.
    assert start_adj == 851
    assert end_adj == 1299
    assert adj_frag_start == 149  # 1000 - 851
    assert adj_frag_end == 299    # 1150 - 851


def test_process_line_near_chr_start_compensates_right(chr_lengths_dict):
    """When left flank hits 0, the residual budget is added to the right flank."""
    result = gf.process_line(
        line="chr1\t100\t250",
        chr_lengths=chr_lengths_dict,
        reverse_complemented=False,
        max_token_length=64,
        min_flank_token_length=8,
    )
    chr, start_adj, end_adj, frag_start, frag_end, adj_frag_start, adj_frag_end = result
    assert start_adj == 0  # clamped at chr start
    assert end_adj == 485  # right got 100bp extra to compensate
    assert adj_frag_start == 100
    assert adj_frag_end == 250


def test_process_line_near_chr_end_compensates_left(chr_lengths_dict):
    """When right flank hits the chrom end, residual budget shifts to the left flank."""
    result = gf.process_line(
        line="chr1\t4800\t4950",
        chr_lengths=chr_lengths_dict,
        reverse_complemented=False,
        max_token_length=64,
        min_flank_token_length=8,
    )
    chr, start_adj, end_adj, frag_start, frag_end, adj_frag_start, adj_frag_end = result
    assert end_adj == 5000
    assert start_adj == 4506
    assert adj_frag_start == 294
    assert adj_frag_end == 444


def test_process_line_reverse_complemented_flips_adj_frag(chr_lengths_dict):
    """When reverse-complemented, adj_frag positions are measured from end_adj."""
    result_fwd = gf.process_line(
        line="chr1\t100\t250",
        chr_lengths=chr_lengths_dict,
        reverse_complemented=False,
        max_token_length=64,
        min_flank_token_length=8,
    )
    result_rev = gf.process_line(
        line="chr1\t100\t250",
        chr_lengths=chr_lengths_dict,
        reverse_complemented=True,
        max_token_length=64,
        min_flank_token_length=8,
    )
    # Genomic coordinates are unchanged ...
    assert result_fwd[1:5] == result_rev[1:5]  # start_adj, end_adj, frag_start, frag_end
    # ... but adj_frag_start/end mirror around end_adj.
    end_adj = result_fwd[2]
    frag_start = result_fwd[3]
    frag_end = result_fwd[4]
    assert result_rev[5] == end_adj - frag_end    # 485 - 250 = 235
    assert result_rev[6] == end_adj - frag_start  # 485 - 100 = 385


def test_process_line_preserves_chr_name(chr_lengths_dict):
    result = gf.process_line(
        line="chr2\t1500\t1700",
        chr_lengths=chr_lengths_dict,
        reverse_complemented=False,
        max_token_length=64,
        min_flank_token_length=8,
    )
    assert result[0] == "chr2"


def test_process_line_raises_on_oversized_fragment(chr_lengths_dict):
    """A fragment longer than max_token_length * 7 bp can't fit; surface a clear error."""
    # 5000-bp fragment vs max_token_length=64 → budget 64*7=448 bp.
    with pytest.raises(ValueError, match="exceeds the sequence budget"):
        gf.process_line(
            line="chr1\t0\t5000",
            chr_lengths=chr_lengths_dict,
            reverse_complemented=False,
            max_token_length=64,
            min_flank_token_length=8,
        )


# ---------- process_sequence ----------------------------------------------

def test_process_sequence_pads_when_short(tokenizer):
    """Short fragments pad up to max_token_length - 2 (for CLS + MOD)."""
    import numpy as np

    seq = "ACGTACGTAC" * 5  # 50bp
    adj_frag_start, adj_frag_end = 20, 30
    rng = np.random.default_rng(0)
    processed, n_freq, n_left, n_frag, n_right = gf.process_sequence(
        seq, adj_frag_start, adj_frag_end, tokenizer,
        max_token_length=256, min_flank_token_length=8, rng=rng,
    )
    # Padding tokens appear in the processed string.
    assert "[PAD]" in processed
    # Re-tokenizing the processed string + the CLS + MOD prepended later == max.
    tokens = tokenizer.tokenize(processed)
    assert len(tokens) == 256 - 2  # leaves room for [CLS] and [MOD]
    # n_token_* are positive ints.
    assert n_left >= 0 and n_frag >= 1 and n_right >= 0


def test_process_sequence_truncates_when_long(tokenizer):
    """When the region exceeds the budget, flanks are trimmed."""
    import numpy as np

    # Long sequence so flanks get truncated.
    seq = "ACGT" * 500  # 2000bp
    adj_frag_start, adj_frag_end = 980, 1020  # ~40bp body
    rng = np.random.default_rng(0)
    processed, n_freq, n_left, n_frag, n_right = gf.process_sequence(
        seq, adj_frag_start, adj_frag_end, tokenizer,
        max_token_length=64, min_flank_token_length=8, rng=rng,
    )
    tokens = tokenizer.tokenize(processed)
    # After trimming, total tokens == max - 2 (room for CLS + MOD).
    assert len(tokens) == 64 - 2
    # No padding was added.
    assert "[PAD]" not in processed
    # Flank counts respect the minimum.
    assert n_left >= 8 and n_right >= 8


def test_process_sequence_raises_when_no_room(tokenizer):
    """If the body alone overflows the budget, process_sequence cannot recover."""
    import numpy as np

    # Tiny budget + a body that tokenizes to more than the budget.
    seq = "ACGT" * 500  # 2000bp
    # Force the body to consume all 2000bp so there's no flank to trim.
    adj_frag_start, adj_frag_end = 0, 2000
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="Insufficient room"):
        gf.process_sequence(
            seq, adj_frag_start, adj_frag_end, tokenizer,
            max_token_length=32, min_flank_token_length=8, rng=rng,
        )


def test_process_sequence_n_chars_become_pad_tokens(tokenizer):
    """N characters in the input become [PAD] tokens in the processed sequence."""
    import numpy as np

    seq = "ACGT" * 25 + "N" * 50 + "ACGT" * 25  # 200bp with N's in the middle
    adj_frag_start, adj_frag_end = 90, 110
    rng = np.random.default_rng(0)
    processed, n_freq, _, _, _ = gf.process_sequence(
        seq, adj_frag_start, adj_frag_end, tokenizer,
        max_token_length=256, min_flank_token_length=8, rng=rng,
    )
    # PAD tokens from N's + padding contribute to n_freq > 0.
    assert "[PAD]" in processed
    assert n_freq > 0.0


# ---------- initialize_iterable_get_fragment / missing files -------------

def test_initialize_iterable_missing_bed_raises(tokenizer, tiny_fa):
    with pytest.raises(FileNotFoundError):
        gf.initialize_iterable_get_fragment(
            tokenizer=tokenizer,
            bed_file="/nonexistent/path/foo.bed",
            ref_genome=str(tiny_fa),
        )


def test_initialize_iterable_missing_ref_raises(tokenizer, tiny_bed):
    with pytest.raises(FileNotFoundError):
        gf.initialize_iterable_get_fragment(
            tokenizer=tokenizer,
            bed_file=str(tiny_bed),
            ref_genome="/nonexistent/path/missing.fa",
        )


def test_get_sequence_raises_on_bedtools_failure(tiny_fa):
    """A malformed BED line should yield a clear error, not an opaque IndexError."""
    # Pointing bedtools at a nonexistent chromosome triggers a nonzero exit code.
    with pytest.raises(RuntimeError, match="bedtools getfasta"):
        gf.get_sequence(
            ref_genome=str(tiny_fa),
            bed_line="chrNonExistent\t0\t100\n",
            reverse_complemented=False,
        )
