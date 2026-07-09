"""Tests for leaf1.downstream.mil.data."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from leaf1.downstream.mil import data as mil_data


# ---------- get_indices ---------------------------------------------------

def test_get_indices_paper_cfdna():
    """cfDNA: 'cut,mod,fragment1' -> fragment1 + cut1 + cut2 + mod = 4*768 indices.

    `fragment1` is the second window over the fragment body in the SEG_OFFSETS
    convention (zero-indexed), so it lives at columns 4*768..5*768.
    """
    idx = mil_data.get_indices("cut,mod,fragment1")
    assert idx.size == 4 * 768
    idx_set = set(idx.tolist())
    expected = (set(range(4 * 768, 5 * 768))       # fragment1 (body window 1)
                | set(range(9 * 768, 10 * 768))    # cut1
                | set(range(10 * 768, 11 * 768))   # cut2
                | set(range(12 * 768, 13 * 768)))  # mod
    assert idx_set == expected


def test_get_indices_paper_scatac():
    """scATAC: 'mod,fragment1' -> 2 * 768 columns."""
    idx = mil_data.get_indices("mod,fragment1")
    assert idx.size == 2 * 768


def test_get_indices_all():
    assert mil_data.get_indices("all").size == 9984


def test_get_indices_tokens_alias():
    """tokens = cut1 + cut2 + cls + mod = 4 chunks of 768."""
    idx = mil_data.get_indices("tokens")
    assert idx.size == 4 * 768


def test_get_indices_dedup_repeated_names():
    """Repeated segment names should not double-count."""
    a = mil_data.get_indices("cut,cut,cut1")
    b = mil_data.get_indices("cut,cut1")
    assert np.array_equal(a, b)
    assert a.size == 2 * 768  # cut1 + cut2 (cut covers both, cut1 redundant)


def test_get_indices_empty_string_returns_empty():
    assert mil_data.get_indices("").size == 0


def test_get_indices_handles_whitespace_and_case():
    """Whitespace and case should be normalised."""
    a = mil_data.get_indices(" CUT,Mod, fragment1 ")
    b = mil_data.get_indices("cut,mod,fragment1")
    assert np.array_equal(a, b)


# ---------- load_splits ---------------------------------------------------

def test_load_splits_cancer_vs_healthy(mil_split_json):
    """Fold 0 of the synthetic split should split 4-4 across train/eval."""
    train_df, val_df, meta = mil_data.load_splits(str(mil_split_json), fold=0)
    assert len(train_df) == 4
    assert len(val_df) == 4
    # Labels: cancer_vs_healthy assigns 1 to non-healthy
    assert set(train_df["label"].tolist()) == {0, 1}
    assert set(val_df["label"].tolist()) == {0, 1}
    assert meta["comparison_type"] == "cancer_vs_healthy"


def test_load_splits_fold_out_of_range_raises(mil_split_json):
    with pytest.raises(ValueError, match="not found"):
        mil_data.load_splits(str(mil_split_json), fold=99)


def test_load_splits_missing_metadata_raises(tmp_path):
    """When metadata_complete.csv is missing next to the split, raise."""
    fake_root = tmp_path / "missing_meta"
    cv = fake_root / "cv_splits"
    cv.mkdir(parents=True)
    j = cv / "x_k5.json"
    j.write_text(json.dumps({
        "metadata": {"comparison_type": "one_vs_all", "target": "x"},
        "splits": [{"train_idx": [0], "eval_idx": [1]}],
    }))
    with pytest.raises(FileNotFoundError, match="Metadata not found"):
        mil_data.load_splits(str(j), fold=0)


def test_load_splits_one_vs_all_comparison_type(tmp_path):
    """one_vs_all -> label == 1 where group matches target."""
    root = tmp_path / "ova"
    (root / "cv_splits").mkdir(parents=True)
    pd.DataFrame([
        {"sample_id": "s0", "group": "A"},
        {"sample_id": "s1", "group": "B"},
        {"sample_id": "s2", "group": "A"},
        {"sample_id": "s3", "group": "C"},
    ]).to_csv(root / "metadata_complete.csv", index=False)
    j = root / "cv_splits" / "task.json"
    j.write_text(json.dumps({
        "metadata": {"comparison_type": "one_vs_all", "target": "A"},
        "splits": [{"train_idx": [0, 1], "eval_idx": [2, 3]}],
    }))

    train_df, val_df, _ = mil_data.load_splits(str(j), fold=0)
    # Train: idx 0 (A->1), idx 1 (B->0)
    assert train_df["label"].tolist() == [1, 0]
    # Val: idx 2 (A->1), idx 3 (C->0)
    assert val_df["label"].tolist() == [1, 0]


# ---------- compute_fragment_stats_from_bags ------------------------------

def test_compute_fragment_stats_matches_numpy_reference():
    """Parallel mean/std should match a hand-computed reference."""
    rng = np.random.default_rng(0)
    D = 4
    bags = [
        rng.standard_normal((5, D), dtype=np.float32).astype(np.float16),
        rng.standard_normal((8, D), dtype=np.float32).astype(np.float16),
        rng.standard_normal((3, D), dtype=np.float32).astype(np.float16),
    ]
    mean_t, std_t = mil_data.compute_fragment_stats_from_bags(bags, n_workers=2)
    # Reference: stack all and compute mean+std in fp64.
    stacked = np.concatenate([b.astype(np.float64) for b in bags], axis=0)
    ref_mean = stacked.mean(axis=0)
    ref_std = stacked.std(axis=0, ddof=1)
    np.testing.assert_allclose(mean_t.numpy(), ref_mean, atol=1e-3)
    np.testing.assert_allclose(std_t.numpy(), ref_std, atol=1e-3)


def test_compute_fragment_stats_empty_bags_raises():
    with pytest.raises(ValueError, match="Empty bags list"):
        mil_data.compute_fragment_stats_from_bags([])


def test_compute_fragment_stats_returns_defaults_for_single_fragment():
    """A single fragment cannot estimate variance; std should default to 1.0."""
    bag = np.zeros((1, 3), dtype=np.float16)
    mean_t, std_t = mil_data.compute_fragment_stats_from_bags([bag], n_workers=1)
    assert torch.allclose(std_t, torch.ones(3))


# ---------- load_bags_parallel --------------------------------------------

def test_load_bags_parallel_single_worker(downstream_metadata_csv):
    """num_workers=1 path: 8 bags loaded, all 200-fragment shape, feature dim = 4*768."""
    df = pd.read_csv(downstream_metadata_csv)
    # Add label column required by load_bags_parallel.
    df["label"] = (df["group"] == "cancer").astype(int)
    bags, labels, sids, feat_dim = mil_data.load_bags_parallel(
        df, segments="cut,mod,fragment1",
        frags_per_bag=10, embed_key="embeddings/bert_window_embeddings",
        layer_idx=4, seed=0, num_workers=1,
    )
    assert len(bags) == 8
    assert labels.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert feat_dim == 4 * 768
    for b in bags:
        assert b.shape == (10, 4 * 768)
        assert b.dtype == np.float16


def test_load_bags_parallel_multi_worker(downstream_metadata_csv):
    """num_workers=2 path: identical output (modulo ordering)."""
    df = pd.read_csv(downstream_metadata_csv)
    df["label"] = (df["group"] == "cancer").astype(int)
    bags, labels, sids, feat_dim = mil_data.load_bags_parallel(
        df, segments="cut", frags_per_bag=10,
        embed_key="embeddings/bert_window_embeddings",
        layer_idx=4, seed=0, num_workers=2,
    )
    assert len(bags) == 8
    assert feat_dim == 2 * 768  # cut1 + cut2


def test_load_bags_parallel_skips_missing_h5(downstream_metadata_csv, tmp_path):
    """Rows whose h5_path is missing get skipped with a warning."""
    df = pd.read_csv(downstream_metadata_csv)
    df["label"] = (df["group"] == "cancer").astype(int)
    # Insert a row with a bogus path
    bogus = df.iloc[[0]].copy()
    bogus["sample_id"] = "missing"
    bogus["h5_path"] = "/nonexistent/path.h5"
    df = pd.concat([df, bogus], ignore_index=True)

    bags, labels, sids, _ = mil_data.load_bags_parallel(
        df, segments="cut",
        frags_per_bag=5, embed_key="embeddings/bert_window_embeddings",
        layer_idx=4, seed=0, num_workers=1,
    )
    assert len(bags) == 8  # original 8, missing one dropped


def test_load_bags_parallel_unknown_segment_raises(downstream_metadata_csv):
    """An empty seg_idx after parsing raises."""
    df = pd.read_csv(downstream_metadata_csv)
    df["label"] = (df["group"] == "cancer").astype(int)
    with pytest.raises(ValueError, match="No features selected"):
        mil_data.load_bags_parallel(
            df, segments="not_a_segment",
            frags_per_bag=5, embed_key="embeddings/bert_window_embeddings",
            layer_idx=4, seed=0, num_workers=1,
        )
