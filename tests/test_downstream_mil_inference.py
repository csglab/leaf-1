"""Tests for the sample-level MIL predictor (leaf1.downstream.mil.inference).

Builds tiny synthetic inner-fold ensembles + flat leaf1-extract-style H5s in a
temp dir (no dependency on the bundled checkpoints), then exercises loading,
segment resolution, bag reading, scoring, and the two on-disk fold layouts.
"""

import json

import h5py
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from leaf1.downstream.mil import inference as inf  # noqa: E402

SEGMENTS = "cut,mod,fragment1"   # cfDNA preset -> 4 segments * 768 = 3072
FEAT_DIM = 3072
PROJ_DIM = 16


def _save_inner_fold(fold_dir, *, feat_dim=FEAT_DIM, proj_dim=PROJ_DIM, use_ln=True, seed=0):
    """Write a checkpoint.pt + norm_stats.pt with the real key layout."""
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    proj = inf.FeatureProjector(feat_dim, proj_dim, mode="linear", use_ln=use_ln)
    pool = inf.GatedAttentionPool(proj_dim, att_dim=64)  # matches load_checkpoint's fixed att_dim
    head = nn.Linear(proj_dim, 2)
    torch.save({
        "proj_state_dict": proj.state_dict(),
        "pool_state_dict": pool.state_dict(),
        "head_state_dict": head.state_dict(),
        # NOTE: deliberately omit feat_proj_ln to mirror the real minimal config
        "config": {"feat_dim": feat_dim, "feat_proj": "linear", "feat_proj_dim": proj_dim},
    }, fold_dir / "checkpoint.pt")
    torch.save(
        {"mean": torch.zeros(feat_dim), "std": torch.ones(feat_dim)},
        fold_dir / "norm_stats.pt",
    )


def _write_config_json(fold_parent, segments=SEGMENTS):
    with open(fold_parent / "config.json", "w") as f:
        json.dump({"data_config": {"segments": segments}}, f)


def _make_ensemble(root, n_inner=3, layout="flat"):
    """Create a fitted-ensemble dir. layout: 'flat' (F.../inner_fold_*) forms."""
    if layout == "outer":                       # collapsed release layout
        for k in range(2):
            fdir = root / f"F{k}"
            for i in range(n_inner):
                _save_inner_fold(fdir / f"inner_fold_{i}", seed=k * 10 + i)
            _write_config_json(fdir)
    elif layout == "sweep":                      # raw /scratch layout (extra config dir)
        for k in range(2):
            cfgdir = root / f"F{k}" / "dropout_headC10_l1r0"
            for i in range(n_inner):
                _save_inner_fold(cfgdir / f"inner_fold_{i}", seed=k * 10 + i)
            _write_config_json(cfgdir)
    else:                                        # single outer fold
        for i in range(n_inner):
            _save_inner_fold(root / f"inner_fold_{i}", seed=i)
        _write_config_json(root)
    return root


def _make_h5(path, n_frags=40, group="Kidney_cancer", sample_id="KP1", seed=1):
    rng = np.random.RandomState(seed)
    emb = rng.randn(n_frags, inf.get_indices("all").size).astype(np.float16)  # (n, 9984)
    with h5py.File(path, "w") as f:
        g = f.create_group("embeddings")
        g.create_dataset("bert_window_embeddings", data=emb)
        m = f.create_group("metadata")
        m.attrs["group"] = group
        m.attrs["sample_id"] = sample_id
        m.attrs["modality"] = "CFDNA"


# ---------------------------------------------------------------------------

def test_get_indices_cfdna_feat_dim():
    idx = inf.get_indices(SEGMENTS)
    assert idx.size == FEAT_DIM
    assert np.all(np.diff(idx) > 0)  # ascending, deterministic order


def test_detect_use_ln_from_state_dict(tmp_path):
    _save_inner_fold(tmp_path / "inner_fold_0", use_ln=True)
    ck = torch.load(tmp_path / "inner_fold_0" / "checkpoint.pt", weights_only=False)
    assert inf._detect_use_ln(ck["proj_state_dict"]) is True
    # a no-LN projector must be detected as such
    proj = inf.FeatureProjector(FEAT_DIM, PROJ_DIM, mode="linear", use_ln=False)
    assert inf._detect_use_ln(proj.state_dict()) is False


def test_load_checkpoint_roundtrip(tmp_path):
    _save_inner_fold(tmp_path / "inner_fold_0", use_ln=True)
    m = inf.load_checkpoint(tmp_path / "inner_fold_0", device="cpu")
    assert m["feat_dim"] == FEAT_DIM
    assert m["att_fix"] == "hhi"
    # forward a tiny bag through the reconstructed model
    bag = torch.randn(1, 5, FEAT_DIM)
    z = m["proj"](m["norm"](bag))
    pooled = m["pool"](z, attention_fix=m["att_fix"])
    assert pooled.shape == (1, PROJ_DIM)


def test_discover_fold_dirs_layouts(tmp_path):
    outer = _make_ensemble(tmp_path / "outer", layout="outer")
    sweep = _make_ensemble(tmp_path / "sweep", layout="sweep")
    single = _make_ensemble(tmp_path / "single", layout="single")

    # single outer fold -> its inner folds
    assert len(inf.discover_fold_dirs(single)) == 3
    # task dir, all_folds -> every inner fold across F0,F1 (both layouts)
    assert len(inf.discover_fold_dirs(outer, all_folds=True)) == 6
    assert len(inf.discover_fold_dirs(sweep, all_folds=True)) == 6
    # task dir, no all_folds -> only F0's inner folds
    assert len(inf.discover_fold_dirs(outer, all_folds=False)) == 3


def test_resolve_segments_from_config(tmp_path):
    root = _make_ensemble(tmp_path / "e", layout="outer")
    folds = inf.discover_fold_dirs(root, all_folds=True)
    assert inf.resolve_segments(root, folds) == SEGMENTS
    assert inf.resolve_segments(root, folds, override="mod") == "mod"


def test_load_ensemble_feat_dim_mismatch_raises(tmp_path):
    root = _make_ensemble(tmp_path / "e", layout="outer")
    # 'mod' resolves to 768 != checkpoint feat_dim 3072 -> loud failure
    with pytest.raises(ValueError, match="mismatch"):
        inf.load_ensemble(root, all_folds=True, segments="mod", device="cpu")


def test_contiguous_runs():
    runs = inf._contiguous_runs(np.array([2, 3, 4, 10, 11, 20]))
    assert runs == [(2, 5), (10, 12), (20, 21)]


def test_load_bag_selects_segment_columns(tmp_path):
    h5_path = tmp_path / "s.h5"
    _make_h5(h5_path, n_frags=40)
    seg_idx = inf.get_indices(SEGMENTS)
    with h5py.File(h5_path, "r") as f:
        bag = inf.load_bag(f, seg_idx, n_frags=100000)          # no subsample (n<n_frags)
        assert bag.shape == (40, FEAT_DIM)
        # subsampling path
        bag2 = inf.load_bag(f, seg_idx, n_frags=10, seed=0)
        assert bag2.shape == (10, FEAT_DIM)
        # columns actually match a direct gather
        full = f["embeddings/bert_window_embeddings"][:]
    np.testing.assert_allclose(bag, full[:, seg_idx].astype(np.float32))


def test_run_inference_end_to_end(tmp_path):
    root = _make_ensemble(tmp_path / "ckpt", layout="outer")
    files = []
    for i, (grp, sid) in enumerate([("Kidney_cancer", "KP1"), ("Healthy", "HC1")]):
        p = tmp_path / f"CFDNA_{sid}.h5"
        _make_h5(p, group=grp, sample_id=sid, seed=i)
        files.append(str(p))

    out = tmp_path / "out"
    rows = inf.run_inference(root, files, out, all_folds=True,
                             frags_per_bag=100000, device="cpu")
    assert len(rows) == 2
    assert (out / "sample_summary.csv").exists()
    assert (out / "inference_config.json").exists()
    for r in rows:
        assert 0.0 <= r["pred_prob"] <= 1.0
        assert r["n_models"] == 6            # 2 outer x 3 inner
    # labels + sample ids carried through from H5 metadata
    assert {r["group"] for r in rows} == {"Kidney_cancer", "Healthy"}
    cfg = json.loads((out / "inference_config.json").read_text())
    assert cfg["segments"] == SEGMENTS and cfg["feat_dim"] == FEAT_DIM
