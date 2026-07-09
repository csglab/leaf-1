"""Bag loaders and split utilities for the LEAF-1 MIL trainer.

Trimmed from the upstream `mil_nested/data/loader.py`. Only the four functions
the trainer imports are kept (`get_indices`, `load_splits`, `load_bags_parallel`,
`compute_fragment_stats_from_bags`) plus their internal helpers
(`_worker_bag_sample`, `_worker_mean_sample`, `estimate_bags_bytes`,
`move_bags_to_device`) and the segment-index lookup tables.

The bag loader already supports leaf1-extract's flat `(n, 9984)` layout natively
(handled in `_worker_bag_sample`); no schema-alignment reshape is needed here.
"""

import gc
import json
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm

# =========================
# Segment definitions
# =========================
EMBED_DIM = 768
TOTAL_DIM = 13 * EMBED_DIM  # 9984
SEG_OFFSETS = {
    'left': (0, 3 * EMBED_DIM), 'fragment': (3 * EMBED_DIM, 6 * EMBED_DIM),
    'right': (6 * EMBED_DIM, 9 * EMBED_DIM),
    'cut1': (9 * EMBED_DIM, 10 * EMBED_DIM), 'cut2': (10 * EMBED_DIM, 11 * EMBED_DIM),
    'cls': (11 * EMBED_DIM, 12 * EMBED_DIM), 'mod': (12 * EMBED_DIM, 13 * EMBED_DIM),
}
for _base in ('left', 'fragment', 'right'):
    _start, _ = SEG_OFFSETS[_base]
    for _w in range(3):
        SEG_OFFSETS[f'{_base}{_w}'] = (_start + _w * EMBED_DIM, _start + (_w + 1) * EMBED_DIM)


def get_indices(segments: str) -> np.ndarray:
    """Map a segment-name string (e.g. 'cut,mod,fragment1') to flat column indices in [0, 9984)."""
    segments = segments.lower().replace(' ', '')
    if segments == 'all':
        return np.arange(TOTAL_DIM)
    mask = np.zeros(TOTAL_DIM, dtype=bool)
    for seg in segments.split(','):
        if seg in ['cut', 'cuts']:
            for s in ['cut1', 'cut2']:
                start, end = SEG_OFFSETS[s]
                mask[start:end] = True
        elif seg == 'tokens':
            for s in ['cut1', 'cut2', 'cls', 'mod']:
                start, end = SEG_OFFSETS[s]
                mask[start:end] = True
        elif seg in SEG_OFFSETS:
            start, end = SEG_OFFSETS[seg]
            mask[start:end] = True
    return np.where(mask)[0]


# =========================
# Split utilities
# =========================
def load_splits(split_file, fold=0):
    """Read a CV splits JSON and return (train_df, val_df, split_metadata).

    The split JSON references `<split_file>.parent.parent / metadata_complete.csv`
    as the master metadata; train/val are produced by index-selection.
    """
    with open(split_file) as f:
        split_data = json.load(f)
    meta_file = Path(split_file).parent.parent / 'metadata_complete.csv'
    if not meta_file.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_file}")

    full_df = pd.read_csv(meta_file)
    if fold >= len(split_data['splits']):
        raise ValueError(f"Fold {fold} not found. Only {len(split_data['splits'])} folds available.")

    fold_split = split_data['splits'][fold]
    train_idx = fold_split['train_idx']
    val_idx = fold_split.get('eval_idx', fold_split.get('val_idx', fold_split.get('test_idx', [])))

    train_df = full_df[full_df.index.isin(train_idx)].copy()
    val_df = full_df[full_df.index.isin(val_idx)].copy()

    comparison_type = split_data['metadata'].get('comparison_type', 'one_vs_all')
    target = split_data['metadata'].get('target', split_data['metadata'].get('case_label'))

    def assign_labels(df):
        if df.empty:
            return df
        if comparison_type == 'cancer_vs_healthy':
            df['label'] = (~df['group'].str.lower().isin(['healthy', 'normal'])).astype(int)
        elif comparison_type == 'stage':
            case_stages = split_data['metadata'].get('case_stage', 'III,IV').split(',')
            df['label'] = df['stage_clean'].isin(case_stages).astype(int)
        elif comparison_type == 'pairwise_same_tissue':
            case_label = split_data['metadata'].get('case_label')
            df['label'] = (df['group'] == case_label).astype(int)
        else:
            df['label'] = (df['group'] == target).astype(int)
        return df

    return assign_labels(train_df), assign_labels(val_df), split_data['metadata']


# =========================
# Memory helpers
# =========================
def estimate_bags_bytes(bags_list):
    """Bytes needed to store bags as fp16 tensors on device (with 10% overhead)."""
    return int(sum(b.shape[0] * b.shape[1] * 2 for b in bags_list) * 1.1)


def move_bags_to_device(bags_list, target_device):
    """Move fp16 numpy bags to `target_device` as torch tensors."""
    device_bags = []
    try:
        for i, bag in enumerate(bags_list):
            t = torch.from_numpy(bag).to(target_device, dtype=torch.float16, non_blocking=False)
            device_bags.append(t)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to move bag {i}/{len(bags_list)} to {target_device}: {e}")
    return device_bags


# =========================
# Normalization statistics
# =========================
def compute_fragment_stats_from_bags(bags_list, n_workers: int = 20):
    """Parallel mean / std over all fragments across all bags.

    Returns (mean, std) as torch.float32 tensors of shape (D,).
    """
    if not bags_list:
        raise ValueError("Empty bags list")

    D = int(bags_list[0].shape[1])

    def _bag_sums(bag):
        if bag is None or bag.shape[0] == 0:
            return None
        n = int(bag.shape[0])
        s1 = bag.sum(axis=0, dtype=np.float64)
        ss = np.einsum("ij,ij->j", bag, bag, dtype=np.float64, optimize=True)
        return n, s1, ss

    n_total = 0
    sum1 = np.zeros(D, dtype=np.float64)
    sumsq = np.zeros(D, dtype=np.float64)

    with threadpool_limits(limits=1):
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_bag_sums, b) for b in bags_list]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc="Computing mean/std (parallel)", leave=False):
                out = fut.result()
                if out is None:
                    continue
                n, s1, ss = out
                n_total += n
                sum1 += s1
                sumsq += ss

    if n_total <= 1:
        return torch.zeros(D, dtype=torch.float32), torch.ones(D, dtype=torch.float32)

    mean = sum1 / n_total
    var = (sumsq - n_total * mean * mean) / (n_total - 1)
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


# =========================
# Per-sample bag readers
# =========================
def _worker_mean_sample(args):
    """One-worker mean-pool path. Returns (task_idx, ok, payload, label, sample_id, n_used)."""
    (task_idx, sample_id, h5_path, label,
     seg_idx, n_frags, embed_key, layer_idx, seed) = args
    try:
        with h5py.File(h5_path, "r") as f:
            ds = f[embed_key]
            T = int(ds.shape[0])
            if T <= 0:
                return (task_idx, False, "No fragments available", label, sample_id, 0)

            if n_frags == -1:
                n_sample = T
            elif isinstance(n_frags, float):
                n_sample = max(1, int(T * n_frags))
            else:
                n_sample = min(int(n_frags), T)

            rng = np.random.default_rng(seed ^ (task_idx + 13))
            if n_sample < T:
                ridx = np.sort(rng.choice(T, n_sample, replace=False).astype(np.int64))
            else:
                ridx = np.arange(T, dtype=np.int64)

            sub = ds[ridx]

            if sub.ndim == 2:
                flat = sub
            elif sub.ndim == 3 and sub.shape[1] == 13 and sub.shape[2] == 768:
                flat = sub.reshape(sub.shape[0], -1)
            elif sub.ndim == 4 and sub.shape[2] == 13 and sub.shape[3] == 768:
                lyr = min(layer_idx, sub.shape[1] - 1)
                flat = sub[:, lyr].reshape(sub.shape[0], -1)
            elif sub.ndim == 3 and sub.shape[2] == TOTAL_DIM:
                lyr = min(layer_idx, sub.shape[1] - 1)
                flat = sub[:, lyr, :]
            else:
                return (task_idx, False, f"Unexpected slice shape {sub.shape}", label, sample_id, 0)

            seg = np.take(flat, seg_idx, axis=1)
            mean_pooled = seg.mean(axis=0, dtype=np.float32)
            np.nan_to_num(mean_pooled, copy=False)
            return (task_idx, True, mean_pooled, label, sample_id, int(len(ridx)))
    except Exception as e:
        return (task_idx, False, str(e), label, sample_id, 0)


def _worker_bag_sample(args):
    """One-worker bag sampler. Returns (task_idx, ok, bag_fp16, label, sample_id, feat_dim)."""
    (task_idx, sample_id, h5_path, label,
     seg_idx, frags_per_bag, embed_key, layer_idx, seed, mode) = args
    try:
        with h5py.File(h5_path, "r") as f:
            ds = f[embed_key]
            T = int(ds.shape[0])
            if T <= 0:
                return (task_idx, False, "No fragments available", label, sample_id, 0)

            if frags_per_bag == -1:
                n_sample = T
            elif isinstance(frags_per_bag, float):
                n_sample = max(1, int(T * frags_per_bag))
            else:
                n_sample = min(int(frags_per_bag), T)

            rng = np.random.default_rng(seed ^ (task_idx + 101))
            replace = n_sample > T
            ridx = rng.choice(T, n_sample, replace=replace).astype(np.int64)
            ridx = np.sort(ridx)

            sub = ds[ridx]

            if sub.ndim == 2:
                flat = sub
            elif sub.ndim == 3 and sub.shape[1] == 13 and sub.shape[2] == 768:
                flat = sub.reshape(sub.shape[0], -1)
            elif sub.ndim == 4 and sub.shape[2] == 13 and sub.shape[3] == 768:
                lyr = min(layer_idx, sub.shape[1] - 1)
                flat = sub[:, lyr].reshape(sub.shape[0], -1)
            elif sub.ndim == 3 and sub.shape[2] == TOTAL_DIM:
                lyr = min(layer_idx, sub.shape[1] - 1)
                flat = sub[:, lyr, :]
            else:
                return (task_idx, False, f"Unexpected slice shape {sub.shape}", label, sample_id, 0)

            seg = np.take(flat, seg_idx, axis=1)
            bag_f16 = np.ascontiguousarray(seg, dtype=np.float16)
            return (task_idx, True, bag_f16, label, sample_id, bag_f16.shape[1])
    except Exception as e:
        return (task_idx, False, str(e), label, sample_id, 0)


def load_bags_parallel(df, segments='fragment', frags_per_bag=4000,
                       embed_key='embeddings', layer_idx=4, seed=42,
                       num_workers=40, mode='train'):
    """Preload per-sample bags into memory.

    Returns:
        (bags_list, labels_np, sample_ids_list, feature_dim)
        where bags_list is a list of np.ndarray fp16 with shape (Ti, Dsel).
    """
    seg_idx = get_indices(segments)
    if seg_idx.size == 0:
        raise ValueError(f"No features selected for segments='{segments}'. Check names.")

    print(f"\n{'=' * 60}")
    print(f"Loading bags [{mode.upper()}]")
    print(f"{'=' * 60}")
    print(f"  Samples in df     : {len(df)}")
    print(f"  Segments          : {segments} -> {seg_idx.size} features")
    print(f"  Frags per bag     : {frags_per_bag}")
    print(f"  Embed key         : {embed_key}")
    print(f"  Layer index       : {layer_idx}")
    print(f"  Seed              : {seed}")
    print(f"  Num workers       : {num_workers}")

    est_bytes_per_bag = frags_per_bag * seg_idx.size * 2  # fp16
    est_total_gb = (est_bytes_per_bag * len(df)) / (1024 ** 3)
    print(f"  Est. memory       : ~{est_total_gb:.2f} GB (fp16)")
    print(f"{'=' * 60}\n")

    tasks = []
    skipped = 0
    for i, (_, row) in enumerate(df.iterrows()):
        p = row.get('h5_path')
        if pd.isna(p) or not Path(str(p)).exists():
            skipped += 1
            continue
        sid = str(row.get('sample_id', f'idx_{i}'))
        lab = int(row.get('label', 0))
        tasks.append((i, sid, str(p), lab, seg_idx, frags_per_bag, embed_key,
                       int(layer_idx), int(seed), str(mode)))

    if skipped > 0:
        print(f"Skipped {skipped} samples with missing/invalid h5_path")
    if not tasks:
        raise ValueError("No valid H5 paths in dataframe")

    n_tasks = len(tasks)
    print(f"Loading {n_tasks} samples...")

    out_bags = [None] * n_tasks
    out_labels = np.empty(n_tasks, dtype=np.int64)
    out_sids = [None] * n_tasks

    errors = []
    feat_dim = None
    loaded_frags = 0
    loaded_bytes = 0
    valid_mask = np.zeros(n_tasks, dtype=bool)

    t_start = time.time()

    if num_workers <= 1:
        pbar = tqdm(total=n_tasks, desc=f'Loading H5 bags [{mode}]',
                    unit='sample', leave=True, dynamic_ncols=True)
        for res in map(_worker_bag_sample, tasks):
            idx, ok, payload, lab, sid, fdim = res
            if ok and np.isfinite(payload).all():
                out_bags[idx] = payload
                out_labels[idx] = lab
                out_sids[idx] = sid
                valid_mask[idx] = True
                if feat_dim is None:
                    feat_dim = fdim
                loaded_frags += payload.shape[0]
                loaded_bytes += payload.nbytes
                pbar.set_postfix({'frags': f'{loaded_frags:,}',
                                  'mem': f'{loaded_bytes / (1024 ** 3):.2f}GB'})
            else:
                errors.append((sid, payload if not ok else "non-finite values"))
            pbar.update(1)
        pbar.close()
    else:
        ctx = get_context('fork') if os.name == 'posix' else get_context('spawn')
        pool_size = min(num_workers, n_tasks)
        chunksize = max(1, n_tasks // (4 * pool_size))

        pbar = tqdm(total=n_tasks, desc=f'Loading H5 bags [{mode}]',
                    unit='sample', leave=True, dynamic_ncols=True)
        with ctx.Pool(processes=pool_size) as pool:
            for idx, ok, payload, lab, sid, fdim in pool.imap_unordered(
                _worker_bag_sample, tasks, chunksize=chunksize
            ):
                if ok and np.isfinite(payload).all():
                    out_bags[idx] = payload
                    out_labels[idx] = lab
                    out_sids[idx] = sid
                    valid_mask[idx] = True
                    if feat_dim is None:
                        feat_dim = fdim
                    loaded_frags += payload.shape[0]
                    loaded_bytes += payload.nbytes
                    pbar.set_postfix({'frags': f'{loaded_frags:,}',
                                      'mem': f'{loaded_bytes / (1024 ** 3):.2f}GB'})
                else:
                    errors.append((sid, payload if not ok else "non-finite values"))
                pbar.update(1)
        pbar.close()

    del tasks
    gc.collect()

    t_elapsed = time.time() - t_start

    if errors:
        warnings.warn("Failed to load {} / {} samples (showing up to 5):\n{}".format(
            len(errors), n_tasks,
            "\n".join(f"  - {sid}: {msg}" for sid, msg in errors[:5])
        ))

    n_valid = int(valid_mask.sum())
    if n_valid == 0:
        raise ValueError("No bags loaded successfully")

    if n_valid < n_tasks:
        write_idx = 0
        for read_idx in range(n_tasks):
            if valid_mask[read_idx]:
                if write_idx != read_idx:
                    out_bags[write_idx] = out_bags[read_idx]
                    out_labels[write_idx] = out_labels[read_idx]
                    out_sids[write_idx] = out_sids[read_idx]
                    out_bags[read_idx] = None
                write_idx += 1
        del out_bags[n_valid:]
        out_labels = out_labels[:n_valid].copy()
        del out_sids[n_valid:]

    del valid_mask
    gc.collect()

    if feat_dim is None:
        feat_dim = out_bags[0].shape[1]

    actual_bytes = sum(b.nbytes for b in out_bags)
    actual_frags = sum(b.shape[0] for b in out_bags)
    label_counts = np.bincount(out_labels, minlength=2)

    print(f"\n{'=' * 60}")
    print(f"Loading complete [{mode.upper()}]")
    print(f"{'=' * 60}")
    print(f"  Samples loaded    : {n_valid}/{n_tasks} ({len(errors)} errors)")
    print(f"  Total fragments   : {actual_frags:,}")
    print(f"  Feature dim       : {feat_dim}")
    print(f"  Memory used       : {actual_bytes / (1024 ** 3):.2f} GB (fp16)")
    print(f"  Avg frags/sample  : {actual_frags / n_valid:,.0f}")
    print(f"  Class distribution: {dict(enumerate(label_counts))}")
    print(f"  Time elapsed      : {t_elapsed:.1f}s ({n_valid / t_elapsed:.1f} samples/s)")
    print(f"{'=' * 60}\n")

    return out_bags, out_labels, out_sids, int(feat_dim)
