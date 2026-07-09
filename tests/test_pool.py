"""Unit tests for leaf1.common.pool."""

import pytest
import torch

from leaf1.common.pool import (
    AdaptiveMultiWindowPooler,
    compute_window_embeddings,
    extract_regions,
)


# ---------- extract_regions -----------------------------------------------

def test_extract_regions_basic_masks():
    """Left = (CLS,MOD)..first CUT exclusive; frag = between CUTs exclusive;
    right = after second CUT to end."""
    batch, seq_len, embed_dim = 2, 20, 4
    embeddings = torch.zeros(batch, seq_len, embed_dim)
    start_pos = torch.tensor([5, 6])   # first [CUT]
    end_pos = torch.tensor([12, 15])   # second [CUT]

    left_mask, frag_mask, right_mask = extract_regions(embeddings, start_pos, end_pos)

    # Row 0: left = positions 2..4 (3 tokens), frag = 6..11 (6), right = 13..19 (7)
    assert left_mask[0].sum().item() == 3
    assert frag_mask[0].sum().item() == 6
    assert right_mask[0].sum().item() == 7
    # CLS (0), MOD (1), CUT1 (5), CUT2 (12) are never in any of the three masks for row 0.
    for excluded in (0, 1, 5, 12):
        assert not left_mask[0, excluded].item()
        assert not frag_mask[0, excluded].item()
        assert not right_mask[0, excluded].item()


def test_extract_regions_raises_when_cuts_are_adjacent():
    """A frag of zero length (start_pos + 1 == end_pos) is invalid."""
    embeddings = torch.zeros(1, 10, 4)
    start_pos = torch.tensor([3])
    end_pos = torch.tensor([4])
    with pytest.raises(ValueError):
        extract_regions(embeddings, start_pos, end_pos)


# ---------- AdaptiveMultiWindowPooler -------------------------------------

def test_pooler_no_mask_output_shape():
    pooler = AdaptiveMultiWindowPooler(num_windows=3)
    embeddings = torch.randn(4, 50, 16)
    out = pooler(embeddings)
    assert out.shape == (4, 3 * 16)


def test_pooler_with_mask_output_shape():
    pooler = AdaptiveMultiWindowPooler(num_windows=3)
    embeddings = torch.randn(2, 30, 8)
    mask = torch.zeros(2, 30, dtype=torch.bool)
    mask[:, 5:25] = True
    out = pooler(embeddings, mask)
    assert out.shape == (2, 3 * 8)


def test_pooler_with_empty_mask_returns_zeros():
    pooler = AdaptiveMultiWindowPooler(num_windows=3)
    embeddings = torch.randn(2, 20, 8)
    mask = torch.zeros(2, 20, dtype=torch.bool)  # nothing selected
    out = pooler(embeddings, mask)
    assert torch.allclose(out, torch.zeros_like(out))


def test_pooler_mask_correctness_matches_hand_computed_mean():
    """One window over a mask should equal the mean of the masked positions."""
    pooler = AdaptiveMultiWindowPooler(num_windows=1)
    embeddings = torch.arange(16, dtype=torch.float).reshape(1, 4, 4)
    mask = torch.tensor([[True, False, True, False]])
    out = pooler(embeddings, mask)  # (1, 1 * 4)
    expected = (embeddings[0, 0] + embeddings[0, 2]) / 2
    assert torch.allclose(out.squeeze(0), expected, atol=1e-5)


# ---------- compute_window_embeddings -------------------------------------

def test_compute_window_embeddings_output_shape():
    """13 * embed_dim = 3 segments * 3 windows + 4 special tokens."""
    batch, seq_len, embed_dim = 3, 30, 8
    num_windows = 3
    embeddings = torch.randn(batch, seq_len, embed_dim)
    start_pos = torch.tensor([5, 7, 6])
    end_pos = torch.tensor([20, 22, 25])
    out = compute_window_embeddings(embeddings, start_pos, end_pos, num_windows=num_windows)
    assert out.shape == (batch, (3 * num_windows + 4) * embed_dim)


def test_compute_window_embeddings_special_tokens_at_expected_offsets():
    """CUT1, CUT2, CLS, MOD slots come from positions start_pos, end_pos, 0, 1."""
    batch, seq_len, embed_dim = 2, 20, 4
    num_windows = 3
    # Place unique identifiable values at each special-token row position.
    embeddings = torch.zeros(batch, seq_len, embed_dim)
    embeddings[:, 0, :] = 100  # CLS
    embeddings[:, 1, :] = 200  # MOD
    start_pos = torch.tensor([5, 6])
    end_pos = torch.tensor([12, 15])
    embeddings[0, 5, :] = 300   # CUT1 row 0
    embeddings[1, 6, :] = 301   # CUT1 row 1
    embeddings[0, 12, :] = 400  # CUT2 row 0
    embeddings[1, 15, :] = 401  # CUT2 row 1

    out = compute_window_embeddings(embeddings, start_pos, end_pos, num_windows=num_windows)

    # Layout: [left(3*D), frag(3*D), right(3*D), CUT1(D), CUT2(D), CLS(D), MOD(D)]
    seg = 3 * num_windows * embed_dim
    cut1 = out[:, seg : seg + embed_dim]
    cut2 = out[:, seg + embed_dim : seg + 2 * embed_dim]
    cls_tok = out[:, seg + 2 * embed_dim : seg + 3 * embed_dim]
    mod_tok = out[:, seg + 3 * embed_dim : seg + 4 * embed_dim]

    assert torch.allclose(cut1[0], torch.full((embed_dim,), 300.0))
    assert torch.allclose(cut1[1], torch.full((embed_dim,), 301.0))
    assert torch.allclose(cut2[0], torch.full((embed_dim,), 400.0))
    assert torch.allclose(cut2[1], torch.full((embed_dim,), 401.0))
    assert torch.allclose(cls_tok, torch.full((batch, embed_dim), 100.0))
    assert torch.allclose(mod_tok, torch.full((batch, embed_dim), 200.0))


def test_compute_window_embeddings_left_window_excludes_cls_and_mod():
    """Left flank starts at pos>=2, so CLS (0) and MOD (1) must not leak into it."""
    batch, seq_len, embed_dim = 1, 20, 4
    embeddings = torch.zeros(batch, seq_len, embed_dim)
    embeddings[0, 0, :] = 999  # CLS - should NOT appear in left window
    embeddings[0, 1, :] = 999  # MOD - should NOT appear in left window
    embeddings[0, 2:5, :] = 1  # actual left flank positions
    start_pos = torch.tensor([5])
    end_pos = torch.tensor([15])
    out = compute_window_embeddings(embeddings, start_pos, end_pos, num_windows=1)
    # Left window pooled value is the mean over positions 2..4 = 1.0 (not contaminated by 999).
    left_window = out[0, :embed_dim]
    assert torch.allclose(left_window, torch.full((embed_dim,), 1.0), atol=1e-5)
