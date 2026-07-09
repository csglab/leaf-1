"""Tests for leaf1.downstream.mil.dataloader."""

import numpy as np
import pytest
import torch

from leaf1.downstream.mil.dataloader import (
    CPUBatchDataset,
    GPUResidentDataset,
    collate_bags_cpu,
    collate_bags_gpu,
)


def _make_bags_cpu(n_samples=4, n_frags=8, feat_dim=16, dtype=np.float16, seed=0):
    rng = np.random.default_rng(seed)
    return [
        rng.standard_normal((n_frags, feat_dim), dtype=np.float32).astype(dtype)
        for _ in range(n_samples)
    ]


def _make_bags_gpu(*args, **kwargs):
    """CPU-resident torch tensors mimicking GPU layout (no actual GPU needed)."""
    cpu_bags = _make_bags_cpu(*args, **kwargs)
    return [torch.from_numpy(b).float() for b in cpu_bags]


# ---------- CPUBatchDataset / GPUResidentDataset --------------------------

def test_gpu_resident_dataset_len_and_getitem():
    bags = _make_bags_gpu(n_samples=4)
    labels = torch.tensor([0, 1, 0, 1])
    sids = [f"s{i}" for i in range(4)]
    ds = GPUResidentDataset(bags, labels, sample_ids=sids)
    assert len(ds) == 4
    bag, lab, sid = ds[2]
    assert bag.shape == (8, 16)
    assert lab.item() == 0
    assert sid == "s2"


def test_gpu_resident_dataset_subset_indices():
    """Passing `indices=[a, b]` should restrict iteration to those positions."""
    bags = _make_bags_gpu(n_samples=5)
    labels = torch.tensor([0, 1, 0, 1, 0])
    ds = GPUResidentDataset(bags, labels, indices=[1, 3])
    assert len(ds) == 2
    _, lab0, _ = ds[0]
    _, lab1, _ = ds[1]
    assert lab0.item() == 1
    assert lab1.item() == 1


def test_cpu_batch_dataset_len_and_getitem():
    bags = _make_bags_cpu(n_samples=3, n_frags=5, feat_dim=8)
    labels = np.array([1, 0, 1])
    ds = CPUBatchDataset(bags, labels, sample_ids=["a", "b", "c"])
    assert len(ds) == 3
    bag, lab, sid = ds[1]
    assert bag.shape == (5, 8)
    assert bag.dtype == np.float16
    assert lab == 0
    assert sid == "b"


def test_cpu_batch_dataset_default_sample_ids():
    """When sample_ids is None, defaults to 'idx_<n>' strings."""
    bags = _make_bags_cpu(n_samples=2, n_frags=3, feat_dim=4)
    labels = np.array([0, 1])
    ds = CPUBatchDataset(bags, labels, sample_ids=None)
    _, _, sid = ds[0]
    assert sid == "idx_0"


# ---------- collate_bags_gpu ----------------------------------------------

def test_collate_bags_gpu_uniform_size():
    """All bags same size -> padded is (B, T, D) fp32, mask all True."""
    bags = _make_bags_gpu(n_samples=3, n_frags=6, feat_dim=4)
    labels = torch.tensor([0, 1, 0])
    batch = [(bags[i], labels[i], f"s{i}") for i in range(3)]
    padded, lab, mask, sids = collate_bags_gpu(batch)
    assert padded.shape == (3, 6, 4)
    assert padded.dtype == torch.float32
    assert mask.shape == (3, 6)
    assert mask.dtype == torch.bool
    assert mask.all().item()
    assert lab.tolist() == [0, 1, 0]
    assert sids == ["s0", "s1", "s2"]


def test_collate_bags_gpu_ragged_pads_with_mask():
    """Mixed bag sizes -> padded zeros at the tail, mask zeros at the same positions."""
    bag_a = torch.full((3, 4), 5.0)
    bag_b = torch.full((7, 4), 3.0)
    batch = [
        (bag_a, torch.tensor(0), "a"),
        (bag_b, torch.tensor(1), "b"),
    ]
    padded, lab, mask, _ = collate_bags_gpu(batch)
    assert padded.shape == (2, 7, 4)
    # Row 0: first 3 rows = 5.0, rest = 0.
    assert (padded[0, :3] == 5.0).all()
    assert (padded[0, 3:] == 0.0).all()
    # Row 1: all 7 rows = 3.0.
    assert (padded[1] == 3.0).all()
    # Mask: row 0 first 3 True, rest False. Row 1 all True.
    assert mask[0, :3].all() and not mask[0, 3:].any()
    assert mask[1].all()


def test_collate_bags_gpu_fp16_input_converts_to_fp32():
    """If bags are fp16, the output should be promoted to fp32."""
    bag = torch.from_numpy(np.zeros((4, 5), dtype=np.float16))
    batch = [(bag, torch.tensor(0), "x")]
    padded, *_ = collate_bags_gpu(batch)
    assert padded.dtype == torch.float32


# ---------- collate_bags_cpu ----------------------------------------------

def test_collate_bags_cpu_uniform_fast_path():
    """All bags same size: stacked via np.stack and shared with torch (fp16)."""
    bags = _make_bags_cpu(n_samples=4, n_frags=5, feat_dim=8)
    labels = [0, 1, 0, 1]
    batch = list(zip(bags, labels, [f"s{i}" for i in range(4)]))
    padded, lab, mask, sids = collate_bags_cpu(batch)
    assert padded.shape == (4, 5, 8)
    assert padded.dtype == torch.float16
    assert mask.shape == (4, 5)
    assert mask.all().item()
    assert lab.tolist() == [0, 1, 0, 1]
    assert sids == ["s0", "s1", "s2", "s3"]


def test_collate_bags_cpu_ragged_path():
    """Variable bag sizes triggers the per-bag copy path with proper masking."""
    bag_a = np.full((2, 4), 1, dtype=np.float16)
    bag_b = np.full((5, 4), 2, dtype=np.float16)
    batch = [(bag_a, 0, "a"), (bag_b, 1, "b")]
    padded, lab, mask, _ = collate_bags_cpu(batch)
    assert padded.shape == (2, 5, 4)
    assert (padded[0, :2] == 1).all()
    assert (padded[0, 2:] == 0).all()
    assert (padded[1] == 2).all()
    assert mask[0, :2].all() and not mask[0, 2:].any()
    assert mask[1].all()
    assert lab.tolist() == [0, 1]
