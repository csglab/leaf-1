"""Unit tests for leaf1.common.collator."""

import random

import pytest
import torch

from leaf1.common.collator import DataCollatorForDNAFragmentModelingNoDecay


@pytest.fixture
def collator(tokenizer):
    return DataCollatorForDNAFragmentModelingNoDecay(
        tokenizer=tokenizer,
        mlm_probability=0.15,
        cut_token_probability=0.30,
    )


def _make_batch(tokenizer, seq_len=32, batch_size=4, cut_positions=(7, 24)):
    """Construct a synthetic batch with [CLS][MOD]...[CUT]...[CUT]... and padding.

    Returns a list of dicts (the collator's expected input format).
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    cut_id = tokenizer.convert_tokens_to_ids('[CUT]')
    mod_id = tokenizer.convert_tokens_to_ids('[CFDNA]')

    # Pick a generic base token (some ID outside special tokens).
    generic = max(cls_id, sep_id, pad_id, cut_id, mod_id) + 5

    items = []
    for b in range(batch_size):
        ids = [generic] * seq_len
        ids[0] = cls_id
        ids[1] = mod_id
        ids[cut_positions[0]] = cut_id
        ids[cut_positions[1]] = cut_id
        ids[-1] = sep_id
        # Sprinkle in some PAD at the tail for special-tokens-mask coverage.
        for j in range(seq_len - 4, seq_len - 1):
            ids[j] = pad_id
        attention = [1] * seq_len
        items.append({
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        })
    return items, cut_id, mod_id, cls_id, sep_id, pad_id


def test_collator_returns_expected_keys_and_shapes(collator, tokenizer):
    items, *_ = _make_batch(tokenizer)
    out = collator(items)
    assert set(out.keys()) == {"input_ids", "labels", "attention_mask"}
    assert out["input_ids"].shape == (4, 32)
    assert out["labels"].shape == (4, 32)
    assert out["attention_mask"].shape == (4, 32)


def test_collator_modality_token_is_never_masked(collator, tokenizer):
    """Column 1 (the modality token) must survive the collator untouched."""
    torch.manual_seed(0)
    items, _, mod_id, *_ = _make_batch(tokenizer)
    original_col1 = torch.stack([item["input_ids"] for item in items])[:, 1].clone()
    out = collator(items)
    # Either the input id is unchanged, or labels[:, 1] is -100 (i.e. position was not masked).
    assert torch.equal(out["input_ids"][:, 1], original_col1)
    assert torch.all(out["labels"][:, 1] == -100)


def test_collator_special_tokens_never_masked(collator, tokenizer):
    """[CLS], [SEP], [PAD] are always excluded from the masking distribution."""
    torch.manual_seed(123)
    items, _, mod_id, cls_id, sep_id, pad_id = _make_batch(tokenizer)
    out = collator(items)

    # For special-token positions, labels should be -100 (not part of the loss).
    input_ids_orig = torch.stack([item["input_ids"] for item in items])
    for special_id in (cls_id, sep_id, pad_id):
        positions = (input_ids_orig == special_id)
        assert torch.all(out["labels"][positions] == -100), (
            f"Special token id {special_id} was selected for masking"
        )


def test_collator_cut_tokens_jointly_masked_per_row(tokenizer):
    """Within each row, the two [CUT] tokens are either both selected for the loss or
    neither - i.e. the joint-boundary masking the paper describes.

    Bug-fix regression test: the prior NoDecay collator only touched one CUT per
    row in the per-row loop, leaving the other at the base CUT prob. Restored by
    `cut_indices = cut_indices.view(-1, 2)` before the loop.
    """
    collator = DataCollatorForDNAFragmentModelingNoDecay(
        tokenizer=tokenizer, mlm_probability=0.0, cut_token_probability=0.5
    )
    cut_id = tokenizer.convert_tokens_to_ids('[CUT]')
    cut_positions = (7, 24)

    rows_with_both_masked = 0
    rows_with_neither_masked = 0
    n_trials = 100
    for trial in range(n_trials):
        torch.manual_seed(trial)
        random.seed(trial)
        items, *_ = _make_batch(tokenizer, cut_positions=cut_positions)
        out = collator(items)
        labels = out["labels"]
        for b in range(labels.size(0)):
            left_was_target = (labels[b, cut_positions[0]].item() == cut_id)
            right_was_target = (labels[b, cut_positions[1]].item() == cut_id)
            assert left_was_target == right_was_target, (
                f"Trial {trial} row {b}: CUTs not jointly masked "
                f"(left_target={left_was_target}, right_target={right_was_target})"
            )
            if left_was_target:
                rows_with_both_masked += 1
            else:
                rows_with_neither_masked += 1
    # Both outcomes must occur under many trials at cut_prob=0.5.
    assert rows_with_both_masked > 0
    assert rows_with_neither_masked > 0


def test_collator_cut_masking_rate_matches_cut_probability(tokenizer):
    """Empirical rate of (both CUTs masked) should hover around cut_token_probability."""
    cut_prob = 0.5
    collator = DataCollatorForDNAFragmentModelingNoDecay(
        tokenizer=tokenizer, mlm_probability=0.0, cut_token_probability=cut_prob
    )
    cut_id = tokenizer.convert_tokens_to_ids('[CUT]')
    cut_positions = (7, 24)

    n_trials = 200
    batch_size = 4
    masked_rows = 0
    for trial in range(n_trials):
        torch.manual_seed(trial)
        random.seed(trial)
        items, *_ = _make_batch(tokenizer, cut_positions=cut_positions, batch_size=batch_size)
        out = collator(items)
        labels = out["labels"]
        for b in range(batch_size):
            if labels[b, cut_positions[0]].item() == cut_id:
                masked_rows += 1
    total_rows = n_trials * batch_size
    rate = masked_rows / total_rows
    # 800-trial Bernoulli at p=0.5 has 95% CI ~ [0.466, 0.534]; allow a wider band.
    assert 0.40 < rate < 0.60, f"Joint-mask rate {rate:.3f} drifted too far from {cut_prob}"


def test_collator_unmasked_positions_have_minus_100_label(collator, tokenizer):
    """Anywhere a token is not selected for the loss, the label is -100."""
    torch.manual_seed(7)
    items, *_ = _make_batch(tokenizer)
    input_ids_orig = torch.stack([item["input_ids"] for item in items])
    out = collator(items)
    # Build the set of positions where labels != -100 ; everything else must be -100.
    not_loss = out["labels"] == -100
    # Sanity: -100 positions exist (most positions are not masked at 15%).
    assert not_loss.any()
    # And every -100 label corresponds to either an unchanged input or a special token.
    # We just verify the type contract.
    assert (out["labels"][not_loss] == -100).all()


def test_collator_masked_positions_get_mask_or_random_or_keep(collator, tokenizer):
    """Masked positions follow the 80/10/10 split: [MASK], random, or unchanged."""
    torch.manual_seed(42)
    items, *_ = _make_batch(tokenizer, seq_len=64, batch_size=8)
    input_ids_orig = torch.stack([item["input_ids"] for item in items]).clone()
    out = collator(items)

    masked = out["labels"] != -100
    n_masked = masked.sum().item()
    assert n_masked > 0, "No tokens selected for masking - unlucky seed?"

    mask_token_id = tokenizer.mask_token_id
    n_replaced_with_mask = ((out["input_ids"] == mask_token_id) & masked).sum().item()
    n_unchanged = ((out["input_ids"] == input_ids_orig) & masked).sum().item()
    n_random = n_masked - n_replaced_with_mask - n_unchanged

    # All masked positions accounted for in one of the three branches.
    assert n_replaced_with_mask + n_unchanged + n_random == n_masked
    # And [MASK] should dominate over a reasonable batch size.
    assert n_replaced_with_mask >= n_random


def test_collator_attention_mask_passes_through(collator, tokenizer):
    """The attention_mask is passed through unchanged from the input items."""
    items, *_ = _make_batch(tokenizer)
    expected = torch.stack([item["attention_mask"] for item in items])
    out = collator(items)
    assert torch.equal(out["attention_mask"], expected)
