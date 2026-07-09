"""Focused smoke tests for leaf1.downstream.mil.train.

The full nested-CV main() is too brittle and slow to exercise end-to-end in CI
(checkpoints, early stop, multi-fold). These tests instead cover the inline
model classes and the argparse layer; together with the data + dataloader test
suites, this gives reasonable coverage of the trainer's surface area.
"""

import sys

import numpy as np
import pytest
import torch

from leaf1.downstream.mil import train as mil_train


# ---------- parse_args ----------------------------------------------------

def test_parse_args_paper_canonical(monkeypatch, tmp_path):
    """Smoke: parse the same flag set the paper's tcga_100k launcher uses."""
    config = tmp_path / "cfdna.yaml"
    config.write_text("data:\n  segments: cut,mod,fragment1\n  frags_per_bag: 10\n")
    split = tmp_path / "split.json"
    split.write_text("{}")
    out = tmp_path / "out"

    argv = [
        "leaf1-mil",
        "--config", str(config),
        "--split",  str(split),
        "--fold",   "0",
        "--seed",   "42",
        "--inner_k", "5",
        "--inner_folds", "0,1,2,3,4",
        "--feat_proj", "linear",
        "--feat_proj_dim", "512",
        "--feat_proj_ln",
        "--att_fix", "hhi",
        "--base_lr", "5e-4",
        "--max_epochs", "200",
        "--batch_size", "32",
        "--output_dir", str(out),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    args = mil_train.parse_args()
    assert args.fold == 0
    assert args.inner_k == 5
    assert args.feat_proj == "linear"
    assert args.feat_proj_dim == 512
    assert args.feat_proj_ln is True
    assert args.att_fix == "hhi"
    assert args.base_lr == 5e-4
    assert args.max_epochs == 200
    assert args.batch_size == 32


def test_parse_args_requires_config_and_split(monkeypatch):
    """argparse should reject calls missing required args."""
    monkeypatch.setattr(sys, "argv", ["leaf1-mil"])
    with pytest.raises(SystemExit):
        mil_train.parse_args()


# ---------- FrozenStandardize1d ------------------------------------------

def test_frozen_standardize_normalises_to_zero_mean_unit_std():
    """A normaliser fed its own statistics produces zero mean / unit std output."""
    rng = torch.Generator().manual_seed(0)
    x = torch.randn(64, 16, generator=rng)
    mean = x.mean(dim=0)
    std = x.std(dim=0)
    norm = mil_train.FrozenStandardize1d(mean, std)
    y = norm(x.clone())
    # Tolerance is loose because we round-trip through fp16 inside the buffer.
    assert torch.allclose(y.mean(dim=0), torch.zeros(16), atol=5e-2)
    assert torch.allclose(y.std(dim=0), torch.ones(16),  atol=5e-2)


def test_frozen_standardize_outputs_fp32():
    norm = mil_train.FrozenStandardize1d(torch.zeros(4), torch.ones(4))
    y = norm(torch.randn(3, 4, dtype=torch.float32))
    assert y.dtype == torch.float32


# ---------- FeatureProjector ---------------------------------------------

def test_feature_projector_none_mode_is_identity():
    proj = mil_train.FeatureProjector(in_dim=8, mode="none")
    x = torch.randn(5, 8)
    y = proj(x)
    assert y.shape == x.shape
    assert torch.equal(y, x)
    assert proj.out_dim == 8


def test_feature_projector_linear_mode_changes_dim():
    proj = mil_train.FeatureProjector(in_dim=8, out_dim=4, mode="linear",
                                       use_ln=True, dropout=0.1)
    x = torch.randn(5, 8)
    y = proj(x)
    assert y.shape == (5, 4)
    assert proj.out_dim == 4


def test_feature_projector_mlp_mode_uses_activation():
    proj = mil_train.FeatureProjector(in_dim=4, out_dim=4, mode="mlp",
                                       act="gelu", use_ln=False, dropout=0.0)
    x = torch.zeros(3, 4)
    y = proj(x)
    assert y.shape == (3, 4)


# ---------- GatedAttentionPool -------------------------------------------

def test_gated_attention_pool_output_shape_and_stats():
    """Bag of B x T x D -> pooled B x D plus attention stats."""
    pool = mil_train.GatedAttentionPool(feat_dim=8, att_dim=4)
    bags = torch.randn(3, 10, 8)
    pooled, stats = pool(bags)
    assert pooled.shape == (3, 8)
    assert "hhi" in stats and "entropy" in stats and "perplexity" in stats
    assert 0.0 < stats["hhi"] <= 1.0
    assert stats["entropy"] >= 0.0


def test_gated_attention_pool_respects_mask():
    """Masked positions should not contribute (after softmax their weights ~= 0)."""
    pool = mil_train.GatedAttentionPool(feat_dim=4, att_dim=2)
    bags = torch.randn(1, 6, 4)
    mask = torch.tensor([[True, True, True, False, False, False]])
    pooled_masked, _, weights = pool(bags, mask=mask, return_weights=True)
    # The masked positions should have ~zero attention weight after softmax.
    assert torch.allclose(weights[0, 3:, 0], torch.zeros(3), atol=1e-3)


def test_gated_attention_pool_hhi_rescaling_increases_norm():
    """With hhi rescaling, pooled norm scales up by sqrt(n_eff) vs no rescaling."""
    pool = mil_train.GatedAttentionPool(feat_dim=4, att_dim=4)
    bags = torch.randn(2, 12, 4)
    pooled_hhi, _ = pool(bags, attention_fix="hhi")
    pooled_none, _ = pool(bags, attention_fix="none")
    # The hhi-rescaled vector should be at least as large in norm (since sqrt(n_eff) >= 1
    # when attention is non-degenerate).
    assert pooled_hhi.norm(dim=-1).mean() >= pooled_none.norm(dim=-1).mean()


def test_gated_attention_pool_return_weights_shape():
    pool = mil_train.GatedAttentionPool(feat_dim=3, att_dim=2)
    bags = torch.randn(2, 5, 3)
    pooled, stats, W = pool(bags, return_weights=True)
    assert W.shape == (2, 5, 1)
    # Softmax-normalised: each row sums to ~1.
    assert torch.allclose(W.sum(dim=1).squeeze(-1), torch.ones(2), atol=1e-5)


# ---------- set_all_seeds + main-level helpers ---------------------------

def test_set_all_seeds_makes_torch_deterministic():
    mil_train.set_all_seeds(42)
    a = torch.randn(3)
    mil_train.set_all_seeds(42)
    b = torch.randn(3)
    assert torch.equal(a, b)


def test_head_elasticnet_returns_finite_penalty():
    """The elastic-net regulariser returns a scalar penalty tensor."""
    head = torch.nn.Linear(8, 1)
    pen = mil_train.head_elasticnet(head, C=1.0, l1_ratio=0.5)
    assert pen.ndim == 0
    assert torch.isfinite(pen)
    assert pen.item() >= 0.0


def test_submodule_l2_penalty_zero_when_C_infinite():
    """Very large C makes the L2 penalty approach zero (C is inverse-strength).

    The function takes a list of modules (each must have a `.weight` attribute),
    not bare parameter tensors.
    """
    lin = torch.nn.Linear(8, 8)
    pen = mil_train.submodule_l2_penalty([lin], C=1e9)
    pen_val = pen.item() if torch.is_tensor(pen) else float(pen)
    assert pen_val < 1e-6


def test_submodule_l2_penalty_scales_with_inverse_C():
    """Penalty for the same weights scales as 1/C."""
    lin = torch.nn.Linear(4, 4)
    p_low_C = mil_train.submodule_l2_penalty([lin], C=0.5)
    p_hi_C = mil_train.submodule_l2_penalty([lin], C=5.0)
    p_low_val = p_low_C.item() if torch.is_tensor(p_low_C) else float(p_low_C)
    p_hi_val = p_hi_C.item() if torch.is_tensor(p_hi_C) else float(p_hi_C)
    # 1/0.5 = 2.0 vs 1/5.0 = 0.2 → low-C penalty is 10x higher.
    assert p_low_val > p_hi_val
    assert abs(p_low_val / p_hi_val - 10.0) < 0.01
