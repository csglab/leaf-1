"""Tests for leaf1.downstream.mean_pool.train."""

import numpy as np
import pandas as pd
import pytest

from leaf1.downstream.mean_pool import train as mp


# ---------- parse_segments ------------------------------------------------

def test_parse_segments_paper_cfdna():
    """cfDNA paper preset: 'cut,mod,fragment1' -> 4 segment indices."""
    idx = mp.parse_segments("cut,mod,fragment1")
    # cut = [cut1, cut2] = indices 9, 10
    # mod = index 12
    # fragment1 = index 4
    assert set(idx) == {4, 9, 10, 12}


def test_parse_segments_paper_scatac():
    """scATAC paper preset: 'mod,fragment1' -> 2 segment indices."""
    idx = mp.parse_segments("mod,fragment1")
    assert set(idx) == {4, 12}


def test_parse_segments_group_aliases():
    """'left', 'fragment', 'right' aliases expand to their 3-window indices."""
    assert set(mp.parse_segments("left")) == {0, 1, 2}
    assert set(mp.parse_segments("fragment")) == {3, 4, 5}
    assert set(mp.parse_segments("right")) == {6, 7, 8}
    assert set(mp.parse_segments("cut")) == {9, 10}
    assert set(mp.parse_segments("all")) == set(range(13))


def test_parse_segments_dedup():
    """Repeated names should not produce duplicate indices."""
    idx = mp.parse_segments("cut,cut1,cut")
    assert list(idx) == sorted(set(idx))


def test_parse_segments_raises_on_unknown():
    with pytest.raises(ValueError, match="Unknown segment"):
        mp.parse_segments("nonexistent_segment_name")


# ---------- load_data -----------------------------------------------------

def test_load_data_binary_one_vs_all(downstream_aggregated_h5, downstream_metadata_csv):
    """Binary {Group}_vs_all should yield (X, y) where positives = the named group."""
    X, y, meta, le = mp.load_data(
        h5_path=str(downstream_aggregated_h5),
        meta_csv=str(downstream_metadata_csv),
        segments="cut,mod,fragment1",
        target="cancer_vs_all",
    )
    # cut+mod+fragment1 = 4 segments * 768 dims, flattened by load_data when ndim=3.
    assert X.shape == (8, 4 * 768)
    assert y.tolist() == [0] * 4 + [1] * 4   # 4 healthy, 4 cancer
    assert le is None
    assert "sample_id" in meta.columns


def test_load_data_multi_class(downstream_aggregated_h5, downstream_metadata_csv):
    """target='cell_type' should label-encode groups and return a multi-class y."""
    X, y, meta, le = mp.load_data(
        h5_path=str(downstream_aggregated_h5),
        meta_csv=str(downstream_metadata_csv),
        segments="all",
        target="cell_type",
    )
    assert X.shape == (8, 13 * 768)
    assert le is not None
    assert set(le.classes_) == {"healthy", "cancer"}
    # y should have both classes represented.
    assert set(y.tolist()) == {0, 1}


def test_load_data_mean_suffix_averages_segments(downstream_aggregated_h5, downstream_metadata_csv):
    """The '_mean' suffix should average across selected segments instead of concatenating."""
    X_concat, _, _, _ = mp.load_data(
        h5_path=str(downstream_aggregated_h5),
        meta_csv=str(downstream_metadata_csv),
        segments="cut,mod,fragment1",
        target="cancer_vs_all",
    )
    X_mean, _, _, _ = mp.load_data(
        h5_path=str(downstream_aggregated_h5),
        meta_csv=str(downstream_metadata_csv),
        segments="cut,mod,fragment1_mean",
        target="cancer_vs_all",
    )
    assert X_concat.shape == (8, 4 * 768)
    assert X_mean.shape == (8, 768)


def test_load_data_drops_unknown_group(downstream_aggregated_h5, tmp_path):
    """Rows whose group == 'Unknown' should be filtered out."""
    csv = tmp_path / "meta_with_unknown.csv"
    pd.DataFrame([
        {"sample_id": f"sample_{i}",
         "group": "Unknown" if i < 2 else ("healthy" if i < 5 else "cancer")}
        for i in range(8)
    ]).to_csv(csv, index=False)

    X, y, meta, _ = mp.load_data(
        h5_path=str(downstream_aggregated_h5),
        meta_csv=str(csv),
        segments="all",
        target="cancer_vs_all",
    )
    assert X.shape[0] == 6
    assert (meta["group"] != "Unknown").all()


# ---------- run_cv smoke --------------------------------------------------

def test_run_cv_binary_smoke(downstream_aggregated_h5, downstream_metadata_csv):
    """Smoke-run nested CV with SVM, fast grid, 2 outer * 2 inner folds.

    With only 8 samples the CV is too noisy to assert AUC > 0.5 reliably, so this
    test only verifies the pipeline runs end-to-end and the output schema is right.
    """
    X, y, meta, _ = mp.load_data(
        h5_path=str(downstream_aggregated_h5),
        meta_csv=str(downstream_metadata_csv),
        segments="cut,mod,fragment1",
        target="cancer_vs_all",
    )
    res = mp.run_cv(
        X, y, clf_name="svm", fast=True,
        inner_k=2, outer_k=2, n_jobs=1,
        pca=None, multiclass=False,
    )
    # Binary output schema.
    assert {"y_true", "y_prob", "y_pred", "auc", "ap", "f1",
            "balanced_acc", "cm", "opt_thresh"}.issubset(res.keys())
    assert res["y_true"].shape == (8,)
    assert res["y_pred"].shape == (8,)
    assert 0.0 <= res["auc"] <= 1.0
    assert 0.0 <= res["ap"] <= 1.0
    assert res["cm"].shape == (2, 2)


def test_run_cv_lr_classifier_smoke(downstream_aggregated_h5, downstream_metadata_csv):
    """Same as above but with logistic-regression classifier; exercises the
    'lr' branch (which includes StandardScaler in the pipeline) and elastic-net
    GridSearchCV path."""
    X, y, meta, _ = mp.load_data(
        h5_path=str(downstream_aggregated_h5),
        meta_csv=str(downstream_metadata_csv),
        segments="cut",
        target="cancer_vs_all",
    )
    res = mp.run_cv(
        X, y, clf_name="lr", fast=True,
        inner_k=2, outer_k=2, n_jobs=1,
        pca=None, multiclass=False,
    )
    assert res["y_pred"].shape == (8,)
    assert 0.0 <= res["auc"] <= 1.0
