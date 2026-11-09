"""Optimized MIL pipeline: val/test predictions map back to the right samples.

Runs on CPU with pinned memory and cuda.synchronize stubbed out. .to("cpu") doesn't copy, so the
iterators get more buffers than batches (a recycled buffer would overwrite an earlier batch).
"""

import numpy as np
import pytest
import torch

from leaf1.downstream.mil import dataloader as dl
from leaf1.downstream.mil.train import _preds_by_sample


@pytest.fixture
def cpu_only(monkeypatch):
    real_empty = torch.empty

    def empty_no_pin(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty_no_pin)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)


def _storage(tmp_path, n_samples=11, batch_size=4, seed=42):
    """Each bag is filled with its sample index, so every yielded row identifies its sample."""
    bags = [np.full((3, 2), i, dtype=np.float16) for i in range(n_samples)]
    labels = np.array([i % 2 for i in range(n_samples)])
    st = dl.PrecollatedStorage(storage_path=str(tmp_path / "s"), n_frags=3, feat_dim=2,
                               batch_size=batch_size, n_precollate_epochs=1, n_workers=1)
    st.precollate_epochs(bags=bags, labels=labels, sample_indices=list(range(n_samples)),
                         seed=seed, show_progress=False)
    return st, labels


def _run(it):
    rows, labs = [], []
    for bags, mask, y in it:
        rows.append(bags[:, 0, 0].cpu().numpy())
        labs.append(y.cpu().numpy())
    return np.concatenate(rows).astype(int), np.concatenate(labs)


def test_fixed_order_rows_map_to_samples(cpu_only, tmp_path):
    st, labels = _storage(tmp_path)
    it = dl.OptimizedBatchIterator(st, "cpu", torch.nn.Identity(), n_readers=1, queue_depth=1,
                                   n_buffers=4, shuffle=False)
    for _ in range(3):  # every pass gives the same order
        rows, labs = _run(it)
        assert len(rows) == 12  # 11 samples + 1 wraparound repeat
        pos = st.row_positions()
        assert sorted(pos.tolist()) == list(range(11))
        np.testing.assert_array_equal(rows[:11], pos)
        np.testing.assert_array_equal(labs[:11], labels[pos])

    sids = [f"S{i}" for i in range(11)]
    probs = rows.astype(float) / 100
    out = _preds_by_sample(probs, labs, it, sids, "test")
    assert len(out["sample_id"]) == 11 and len(set(out["sample_id"])) == 11
    for sid, lab, p in zip(out["sample_id"], out["label"], out["prob"]):
        i = int(sid[1:])
        assert lab == labels[i] and p == pytest.approx(i / 100)


def test_shuffle_flag_does_not_change_global_rng(cpu_only, tmp_path):
    st, _ = _storage(tmp_path)
    draws = []
    for shuffle in (True, False):
        np.random.seed(0)
        it = dl.OptimizedBatchIterator(st, "cpu", torch.nn.Identity(), n_readers=1, queue_depth=1,
                                       n_buffers=4, shuffle=shuffle)
        _run(it)
        draws.append(np.random.rand(5))
    np.testing.assert_array_equal(draws[0], draws[1])


def test_fixed_order_requires_single_reader(cpu_only, tmp_path):
    st, _ = _storage(tmp_path)
    with pytest.raises(ValueError):
        dl.OptimizedBatchIterator(st, "cpu", torch.nn.Identity(), n_readers=2, shuffle=False)
