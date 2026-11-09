#!/usr/bin/env python3
"""
================================================================================
train_gated_final_optimized.py

GPU-OPTIMIZED MIL Training with THREE modes:

1. GPU-RESIDENT (default):
   - All data on GPU, fastest for small datasets
   - Use when: < 50 samples with 100K fragments

2. LEGACY CPU-BATCH (--cpu_data):
   - Data on CPU, transfer per batch
   - ~1.0s per batch for 100K fragments
   - Use when: backward compatibility needed

3. OPTIMIZED PIPELINE (--cpu_data --optimized_pipeline):
   - Pre-collate to NVMe + multi-reader + pinned + GPU prefetch
   - ~0.1s per batch for 100K fragments (10x faster!)
   - Use when: large datasets, 100K fragments

================================================================================
USAGE EXAMPLES:
================================================================================

# Small dataset (GPU-resident mode)
python train_gated_final_optimized.py \\
    --config configs/config.yaml \\
    --split splits/task_k5.json \\
    --fold 0 \\
    --batch_size 32 \\
    --output_dir outputs/small_run

# Large dataset (OPTIMIZED - recommended!)
python train_gated_final_optimized.py \\
    --config configs/config.yaml \\
    --split splits/task_k5.json \\
    --fold 0 \\
    --frags_per_bag 100000 \\
    --cpu_data \\
    --optimized_pipeline \\
    --nvme_path /tmp/leaf1_mil_cache \\
    --batch_size 4 \\
    --amp \\
    --output_dir outputs/large_optimized

# Large dataset (legacy mode - for comparison)
python train_gated_final_optimized.py \\
    --config configs/config.yaml \\
    --split splits/task_k5.json \\
    --fold 0 \\
    --frags_per_bag 100000 \\
    --cpu_data \\
    --prefetch \\
    --batch_size 4 \\
    --output_dir outputs/large_legacy

================================================================================
"""

import os
import sys
import argparse
import random
import yaml
import math
import gc
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Local imports (relative within the leaf1.downstream.mil package)
from leaf1.downstream.mil.data import (
    load_splits, load_bags_parallel, compute_fragment_stats_from_bags, get_indices,
)
from leaf1.downstream.mil.dataloader import (
    # GPU-resident (small datasets)
    GPUResidentDataset,
    collate_bags_gpu,
    # Legacy CPU-batch
    CPUBatchDataset,
    collate_bags_cpu,
    PrefetchLoader,
    # Optimized pipeline (large datasets)
    PrecollatedStorage,
    OptimizedBatchIterator,
)


# =============================================================================
# UTILITIES
# =============================================================================

def set_all_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# NORMALIZATION
# =============================================================================

class FrozenStandardize1d(nn.Module):
    """MEMORY-OPTIMIZED: In-place normalization to eliminate copies."""
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8):
        super().__init__()
        # Store in fp16 to save memory
        self.register_buffer("mean", mean.half())
        self.register_buffer("std", std.half().clamp_min(eps))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """In-place normalization for memory efficiency."""
        if x.dtype == torch.float32:
            x = x.half()
        
        # IN-PLACE operations (critical for memory!)
        x = x.sub_(self.mean)
        x = x.div_(self.std)
        
        return x.float()


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class FeatureProjector(nn.Module):
    """Optional feature projection layer."""
    
    def __init__(self, in_dim: int, out_dim: int = None, *, mode: str = "none",
                 act: str = "gelu", use_ln: bool = False, dropout: float = 0.0):
        super().__init__()
        assert mode in {"none", "linear", "mlp"}
        
        if mode == "none":
            self.block = nn.Identity()
            self.out_dim = in_dim
            self.linear = None
            self.post_linear = nn.Identity()
            return
            
        out_dim = out_dim or in_dim
        self.out_dim = out_dim
        
        act_map = {"gelu": nn.GELU(), "relu": nn.ReLU(), "silu": nn.SiLU(), "tanh": nn.Tanh()}
        self.linear = nn.Linear(in_dim, out_dim)
        
        layers = []
        if mode == "mlp":
            layers.append(act_map[act])
        if use_ln:
            layers.append(nn.LayerNorm(out_dim))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.post_linear = nn.Sequential(*layers) if layers else nn.Identity()
        
        nn.init.xavier_uniform_(self.linear.weight, gain=0.5)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.linear is None:
            return x
        x = self.linear(x)
        x = self.post_linear(x)
        return x


class GatedAttentionPool(nn.Module):
    """Gated attention pooling for MIL."""
    
    def __init__(self, feat_dim: int, att_dim: int = 64, 
                 dropout: float = 0.0, temperature: float = 1.0):
        super().__init__()
        self.V = nn.Linear(feat_dim, att_dim)
        self.U = nn.Linear(feat_dim, att_dim)
        self.w = nn.Linear(att_dim, 1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.temperature = temperature
        self.attention_params = [self.V, self.U, self.w]

    def forward(self, bags: torch.Tensor, mask: torch.Tensor = None, 
                attention_fix: str = 'hhi', return_weights: bool = False):
        A_V = torch.tanh(self.V(bags))
        A_U = torch.sigmoid(self.U(bags))
        gated = self.drop(A_V * A_U)
        scores = self.w(gated)
        
        if self.temperature != 1.0:
            scores = scores / self.temperature
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -1e9)
            
        W = F.softmax(scores, dim=1)
        pooled = (bags * W).sum(dim=1)

        # Optional re-scaling
        w_sq = W.squeeze(-1)
        if attention_fix == 'hhi':
            n_eff = 1.0 / w_sq.pow(2).sum(dim=1, keepdim=True).clamp_min(1e-8)
            pooled = pooled * n_eff.sqrt()
        elif attention_fix == 'entropy':
            w_cl = w_sq.clamp_min(1e-8)
            H = -(w_cl * w_cl.log()).sum(dim=1, keepdim=True)
            n_eff = H.exp()
            pooled = pooled * n_eff.sqrt()

        # Stats
        w_cl = w_sq.clamp_min(1e-8)
        entropy = -(w_cl * w_cl.log()).sum(dim=1).mean().item()
        hhi = w_sq.pow(2).sum(dim=1).mean().item()
        stats = {'hhi': hhi, 'entropy': entropy, 'perplexity': float(np.exp(entropy))}
        
        if return_weights:
            return pooled, stats, W
        return pooled, stats


# =============================================================================
# REGULARIZATION
# =============================================================================

def head_elasticnet(head: nn.Linear, C: float, l1_ratio: float) -> torch.Tensor:
    """ElasticNet regularization for head."""
    l2 = 0.5 * (1.0/C) * head.weight.pow(2).sum()
    l1 = (l1_ratio/C) * head.weight.abs().sum()
    return l2 + l1


def submodule_l2_penalty(module_params: list, C: float) -> torch.Tensor:
    """L2 penalty for submodule parameters."""
    penalty = 0.0
    for param in module_params:
        if hasattr(param, "weight") and param.weight.requires_grad:
            penalty += 0.5 * (1.0/C) * param.weight.pow(2).sum()
    return penalty


# =============================================================================
# TRACKERS
# =============================================================================

class MovingAverageTracker:
    """Track moving averages for model selection."""
    
    def __init__(self, windows: list = [3, 5, 10], warmup_epochs: int = 0):
        self.windows = windows
        self.warmup_epochs = warmup_epochs
        self.post_warmup_count = 0
        self.history = []
        self.ema_state = {w: None for w in windows}
        self.ma_history = {w: [] for w in windows}
        self.ema_history = {w: [] for w in windows}

    def update(self, val_auc: float, test_auc: float, epoch: int) -> dict:
        results = {}
        if epoch < self.warmup_epochs:
            return results
            
        self.post_warmup_count += 1
        self.history.append((val_auc, test_auc, epoch))
        
        for w in self.windows:
            if self.post_warmup_count >= w:
                recent = self.history[-w:]
                ma_val = np.mean([h[0] for h in recent])
                results[f'ma{w}'] = ma_val
                self.ma_history[w].append((ma_val, test_auc, epoch))
                if len(self.ma_history[w]) == 1 or ma_val >= max(h[0] for h in self.ma_history[w][:-1]):
                    results[f'ma{w}_new_max'] = True
                    
            if self.post_warmup_count >= w:
                alpha = 2.0 / (w + 1)
                if self.ema_state[w] is None:
                    first_w = self.history[:w]
                    self.ema_state[w] = np.mean([h[0] for h in first_w])
                else:
                    self.ema_state[w] = alpha * val_auc + (1 - alpha) * self.ema_state[w]
                ema_val = self.ema_state[w]
                results[f'ema{w}'] = ema_val
                self.ema_history[w].append((ema_val, test_auc, epoch))
                if len(self.ema_history[w]) == 1 or ema_val >= max(h[0] for h in self.ema_history[w][:-1]):
                    results[f'ema{w}_new_max'] = True
        return results

    def get_metric_value(self, metric_name: str):
        if metric_name == 'val_auc' and self.history:
            return self.history[-1][0]
        for w in self.windows:
            if metric_name == f'ma{w}' and self.ma_history[w]:
                return self.ma_history[w][-1][0]
            if metric_name == f'ema{w}' and self.ema_history[w]:
                return self.ema_history[w][-1][0]
        return None


class ThreePhaseCosineScheduler:
    """Three-phase learning rate scheduler: warmup → flat → cosine decay."""
    
    def __init__(self, optimizer, warmup_epochs: int, flat_epochs: int, 
                 total_epochs: int, base_lr: float = 1e-3, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.flat = flat_epochs
        self.total = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
    
    def step(self, epoch: int) -> float:
        if self.warmup > 0 and epoch < self.warmup:
            lr = self.min_lr + (self.base_lr - self.min_lr) * (epoch + 1) / self.warmup
        elif epoch < self.warmup + self.flat:
            lr = self.base_lr
        else:
            cosine_epochs = max(0, self.total - self.warmup - self.flat)
            lr = (self.base_lr if cosine_epochs == 0 else
                  self.min_lr + 0.5 * (self.base_lr - self.min_lr) *
                  (1 + math.cos(math.pi * (epoch - self.warmup - self.flat) / max(1, cosine_epochs))))
        for g in self.optimizer.param_groups:
            g['lr'] = lr
        return lr


# =============================================================================
# TRAINING: GPU-RESIDENT MODE
# =============================================================================

def train_inner_fold_gpu(
    inner_fold: int,
    bags_tr_idx: list,
    bags_va_idx: list,
    all_bags_gpu: list,
    all_labels_gpu: torch.Tensor,
    all_sample_ids: list,
    test_bags_gpu: list,
    test_labels_gpu: torch.Tensor,
    test_sample_ids: list,
    feat_dim: int,
    device: str,
    args,
    output_dir: Path,
) -> dict:
    """Train one inner fold with GPU-resident data."""
    
    set_all_seeds(args.seed + inner_fold)
    fold_dir = output_dir / f"inner_fold_{inner_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"INNER FOLD {inner_fold} (GPU-resident mode)")
    print(f"  Train: {len(bags_tr_idx)} | Val: {len(bags_va_idx)} | Test: {len(test_bags_gpu)}")
    print(f"{'='*60}")

    # Create datasets
    train_dataset = GPUResidentDataset(all_bags_gpu, all_labels_gpu, all_sample_ids, bags_tr_idx)
    val_dataset = GPUResidentDataset(all_bags_gpu, all_labels_gpu, all_sample_ids, bags_va_idx)
    test_dataset = GPUResidentDataset(test_bags_gpu, test_labels_gpu, test_sample_ids)
    
    # Weighted sampler
    tr_labels = all_labels_gpu[torch.as_tensor(bags_tr_idx, device=device, dtype=torch.long)].cpu().numpy()
    tr_class_counts = np.bincount(tr_labels, minlength=2).astype(np.float64)
    
    if (tr_class_counts > 0).sum() < 2:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_bags_gpu, num_workers=0)
    else:
        class_weights = 1.0 / np.maximum(tr_class_counts, 1.0)
        sample_weights = class_weights[tr_labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                  collate_fn=collate_bags_gpu, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_bags_gpu, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             collate_fn=collate_bags_gpu, num_workers=0)

    return _train_inner_fold_common(
        inner_fold, train_loader, val_loader, test_loader,
        feat_dim, device, args, output_dir
    )


# =============================================================================
# TRAINING: LEGACY CPU-BATCH MODE
# =============================================================================

def train_inner_fold_cpu(
    inner_fold: int,
    bags_tr_idx: list,
    bags_va_idx: list,
    bags_outer_train: list,
    y_outer_train: np.ndarray,
    sids_outer_train: list,
    bags_outer_test: list,
    y_outer_test: np.ndarray,
    sids_outer_test: list,
    normalizer: nn.Module,
    feat_dim: int,
    device: str,
    args,
    output_dir: Path,
) -> dict:
    """Train one inner fold with LEGACY CPU-batch mode."""
    
    set_all_seeds(args.seed + inner_fold)
    fold_dir = output_dir / f"inner_fold_{inner_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    mode_str = "CPU-batch (legacy)"
    if args.prefetch:
        mode_str += f" + prefetch(q={args.prefetch_count})"
    if args.amp:
        mode_str += " + AMP"
    
    print(f"\n{'='*60}")
    print(f"INNER FOLD {inner_fold} ({mode_str})")
    print(f"  Train: {len(bags_tr_idx)} | Val: {len(bags_va_idx)} | Test: {len(bags_outer_test)}")
    print(f"{'='*60}")

    # Create datasets
    train_dataset = CPUBatchDataset(bags_outer_train, y_outer_train, sids_outer_train, bags_tr_idx)
    val_dataset = CPUBatchDataset(bags_outer_train, y_outer_train, sids_outer_train, bags_va_idx)
    test_dataset = CPUBatchDataset(bags_outer_test, y_outer_test, sids_outer_test)
    
    # Weighted sampler
    tr_labels = y_outer_train[np.array(bags_tr_idx)]
    tr_class_counts = np.bincount(tr_labels, minlength=2).astype(np.float64)
    
    num_workers = 0
    pin_memory = args.prefetch
    
    if (tr_class_counts > 0).sum() < 2:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=collate_bags_cpu, num_workers=num_workers, 
                                  pin_memory=pin_memory)
    else:
        class_weights = 1.0 / np.maximum(tr_class_counts, 1.0)
        sample_weights = class_weights[tr_labels]
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler,
                                  collate_fn=collate_bags_cpu, num_workers=num_workers,
                                  pin_memory=pin_memory)

    val_batch_size = args.batch_size * 2
    test_batch_size = args.batch_size * 2
    
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                            collate_fn=collate_bags_cpu, num_workers=num_workers,
                            pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False,
                             collate_fn=collate_bags_cpu, num_workers=num_workers,
                             pin_memory=pin_memory)

    # Wrap with prefetch if enabled
    if args.prefetch:
        train_loader = PrefetchLoader(train_loader, prefetch_count=args.prefetch_count)
        val_loader = PrefetchLoader(val_loader, prefetch_count=args.prefetch_count)
        test_loader = PrefetchLoader(test_loader, prefetch_count=args.prefetch_count)

    return _train_inner_fold_common(
        inner_fold, train_loader, val_loader, test_loader,
        feat_dim, device, args, output_dir,
        normalizer=normalizer
    )


# =============================================================================
# TRAINING: OPTIMIZED PIPELINE (10x faster!)
# =============================================================================

def train_inner_fold_cpu_optimized(
    inner_fold: int,
    bags_tr_idx: list,
    bags_va_idx: list,
    bags_outer_train: list,
    y_outer_train: np.ndarray,
    sids_outer_train: list,
    bags_outer_test: list,
    y_outer_test: np.ndarray,
    sids_outer_test: list,
    normalizer: nn.Module,
    feat_dim: int,
    device: str,
    args,
    output_dir: Path,
) -> dict:
    """Train one inner fold with OPTIMIZED pipeline (10x faster!).
    
    Uses:
    - Pre-collation to NVMe
    - Multi-reader threads (6)
    - Pinned memory for fast DMA
    - GPU prefetch thread
    """
    
    set_all_seeds(args.seed + inner_fold)
    fold_dir = output_dir / f"inner_fold_{inner_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    
    mode_str = "OPTIMIZED (NVMe + multi-reader + pinned + GPU prefetch)"
    if args.amp:
        mode_str += " + AMP"
    
    print(f"\n{'='*60}")
    print(f"INNER FOLD {inner_fold}")
    print(f"  Mode: {mode_str}")
    print(f"  Train: {len(bags_tr_idx)} | Val: {len(bags_va_idx)} | Test: {len(bags_outer_test)}")
    print(f"  Readers: {args.n_readers} | Queue: {args.queue_depth}")
    print(f"  Pre-collate epochs: {args.n_precollate_epochs}")
    print(f"{'='*60}")
    
    n_frags = bags_outer_train[0].shape[0]
    
    # ==========================================================================
    # PHASE 1: Pre-collate all splits to NVMe
    # ==========================================================================
    print(f"\n  Phase 1: Pre-collating to NVMe...")
    
    # Storage paths
    base_storage = Path(args.nvme_path) / f"fold_{inner_fold}"
    train_storage_path = base_storage / "train"
    val_storage_path = base_storage / "val"
    test_storage_path = base_storage / "test"
    
    # TRAIN: Multiple epochs for shuffle variety
    print(f"    Training data ({len(bags_tr_idx)} samples, {args.n_precollate_epochs} epochs)...")
    train_storage = PrecollatedStorage(
        storage_path=str(train_storage_path),
        n_frags=n_frags,
        feat_dim=feat_dim,
        batch_size=args.batch_size,
        n_precollate_epochs=args.n_precollate_epochs,
        n_workers=args.n_workers,
    )
    train_storage.precollate_epochs(
        bags=bags_outer_train,
        labels=y_outer_train,
        sample_indices=bags_tr_idx,
        seed=args.seed + inner_fold,
        show_progress=True,
    )
    
    # VAL: Single epoch (no shuffle needed, but use same optimized pipeline)
    print(f"    Validation data ({len(bags_va_idx)} samples, 1 epoch)...")
    val_batch_size = args.batch_size * 2  # Larger for inference
    val_storage = PrecollatedStorage(
        storage_path=str(val_storage_path),
        n_frags=n_frags,
        feat_dim=feat_dim,
        batch_size=val_batch_size,
        n_precollate_epochs=1,
        n_workers=args.n_workers,
    )
    val_storage.precollate_epochs(
        bags=bags_outer_train,
        labels=y_outer_train,
        sample_indices=bags_va_idx,
        seed=args.seed,
        show_progress=True,
    )
    
    # TEST: Single epoch
    print(f"    Test data ({len(bags_outer_test)} samples, 1 epoch)...")
    test_batch_size = args.batch_size * 2
    test_storage = PrecollatedStorage(
        storage_path=str(test_storage_path),
        n_frags=n_frags,
        feat_dim=feat_dim,
        batch_size=test_batch_size,
        n_precollate_epochs=1,
        n_workers=args.n_workers,
    )
    test_storage.precollate_epochs(
        bags=bags_outer_test,
        labels=y_outer_test,
        sample_indices=list(range(len(bags_outer_test))),
        seed=args.seed,
        show_progress=True,
    )
    
    # ==========================================================================
    # PHASE 2: Create optimized iterators
    # ==========================================================================
    print(f"\n  Phase 2: Creating optimized iterators...")
    
    train_iterator = OptimizedBatchIterator(
        storage=train_storage,
        device=device,
        normalizer=normalizer,
        n_readers=args.n_readers,
        queue_depth=args.queue_depth,
    )
    
    val_iterator = OptimizedBatchIterator(
        storage=val_storage,
        device=device,
        normalizer=normalizer,
        n_readers=1,
        queue_depth=1,
        n_buffers=2,
        shuffle=False,
    )
    
    test_iterator = OptimizedBatchIterator(
        storage=test_storage,
        device=device,
        normalizer=normalizer,
        n_readers=1,
        queue_depth=1,
        n_buffers=2,
        shuffle=False,
    )
    
    # ==========================================================================
    # PHASE 3: Train using optimized iterators
    # ==========================================================================
    print(f"\n  Phase 3: Training...")
    
    result = _train_inner_fold_optimized(
        inner_fold=inner_fold,
        train_iterator=train_iterator,
        val_iterator=val_iterator,
        test_iterator=test_iterator,
        feat_dim=feat_dim,
        device=device,
        args=args,
        output_dir=output_dir,
        val_sids=[sids_outer_train[i] for i in bags_va_idx],
        test_sids=list(sids_outer_test),
    )
    
    # ==========================================================================
    # PHASE 4: Cleanup
    # ==========================================================================
    print(f"\n  Phase 4: Cleaning up NVMe storage...")
    train_storage.cleanup()
    val_storage.cleanup()
    test_storage.cleanup()
    
    # Remove base folder
    if base_storage.exists():
        import shutil
        shutil.rmtree(base_storage, ignore_errors=True)
    
    return result


def _preds_by_sample(probs, labels, iterator: OptimizedBatchIterator, sids, prefix: str) -> dict:
    """One row per sample with its real ID; drops the wraparound repeats in the last batch."""
    pos = iterator.storage.row_positions()
    return {
        'sample_id': [sids[p] if sids is not None else f"{prefix}_{p}" for p in pos],
        'label': labels[:len(pos)].tolist(),
        'prob': probs[:len(pos)].tolist(),
    }


def _train_inner_fold_optimized(
    inner_fold: int,
    train_iterator: OptimizedBatchIterator,
    val_iterator: OptimizedBatchIterator,
    test_iterator: OptimizedBatchIterator,
    feat_dim: int,
    device: str,
    args,
    output_dir: Path,
    val_sids: list = None,
    test_sids: list = None,
) -> dict:
    """Training loop for OPTIMIZED pipeline (uses iterators instead of DataLoaders).

    Model selection still uses the val AUC over all val rows, including the wraparound repeats.
    Saved predictions and reported AUCs use one row per sample."""
    
    fold_dir = output_dir / f"inner_fold_{inner_fold}"
    
    # Configure regularization
    if args.reg_scheme == 'dropout':
        proj_dropout = args.scheme_value if args.feat_proj != 'none' else 0.0
        att_dropout = args.scheme_value
        proj_C = None
        att_C = None
    else:
        proj_dropout = 0.0
        att_dropout = 0.0
        proj_C = args.scheme_value if args.feat_proj != 'none' else None
        att_C = args.scheme_value

    # Build model
    if args.feat_proj == 'none':
        proj = FeatureProjector(feat_dim, mode='none').to(device)
        proj_dim = feat_dim
    else:
        proj = FeatureProjector(feat_dim, args.feat_proj_dim, mode=args.feat_proj,
                                use_ln=args.feat_proj_ln, dropout=proj_dropout).to(device)
        proj_dim = proj.out_dim

    pool = GatedAttentionPool(proj_dim, att_dim=64, dropout=att_dropout,
                              temperature=args.att_temperature).to(device)
    head = nn.Linear(proj_dim, 2).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)

    params = list(pool.parameters()) + list(head.parameters())
    if args.feat_proj != 'none':
        params.extend(proj.parameters())
    opt = torch.optim.Adam(params, lr=args.base_lr, weight_decay=0.0)
    scheduler = ThreePhaseCosineScheduler(opt, args.lr_warmup, args.lr_flat, args.max_epochs,
                                          args.base_lr, args.min_lr)
    crit = nn.CrossEntropyLoss()

    scaler = GradScaler() if args.amp else None
    accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

    # Tracking
    best_metric_value = None
    best_metric_epoch = -1
    best_state = None
    es_patience_counter = 0
    ma_tracker = MovingAverageTracker([3, 5, 10], warmup_epochs=args.warmup_patience)
    best_val_preds = None
    best_test_preds = None

    # NEW: only run test forward on "new best" (based on early_stop_primary)
    last_test_auc = None

    epoch_pbar = tqdm(range(args.max_epochs), desc=f"IF{inner_fold}", leave=False, position=0)

    for epoch in epoch_pbar:
        lr = scheduler.step(epoch)

        # ============ TRAINING ============
        proj.train(); pool.train(); head.train()
        train_loss_acc = 0.0
        n_train_samples = 0
        opt.zero_grad()
        accum_step = 0

        train_pbar = tqdm(
            train_iterator, 
            total=len(train_iterator),
            desc="  Train", 
            leave=False, 
            position=1,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
        )

        for bags, mask, labels in train_pbar:
            # Data is already normalized and on GPU!
            if args.amp:
                with autocast():
                    z = proj(bags)
                    pooled, _ = pool(z, mask, args.att_fix)
                    logits = head(pooled)
                    loss = crit(logits, labels) + head_elasticnet(head, args.head_C, args.head_l1_ratio)
                    if args.reg_scheme == 'ridgeC':
                        if args.feat_proj != 'none' and proj.linear is not None:
                            loss += submodule_l2_penalty([proj.linear], proj_C)
                        loss += submodule_l2_penalty(pool.attention_params, att_C)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                z = proj(bags)
                pooled, _ = pool(z, mask, args.att_fix)
                logits = head(pooled)
                loss = crit(logits, labels) + head_elasticnet(head, args.head_C, args.head_l1_ratio)
                if args.reg_scheme == 'ridgeC':
                    if args.feat_proj != 'none' and proj.linear is not None:
                        loss += submodule_l2_penalty([proj.linear], proj_C)
                    loss += submodule_l2_penalty(pool.attention_params, att_C)
                loss = loss / accumulation_steps
                loss.backward()
            
            train_loss_acc += loss.item() * len(labels) * accumulation_steps
            n_train_samples += len(labels)
            accum_step += 1
            
            if accum_step % accumulation_steps == 0:
                if args.amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    opt.step()
                opt.zero_grad()
            
            train_pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
            del bags, mask, labels

        train_pbar.close()
        train_loss = train_loss_acc / max(1, n_train_samples)

        # ============ VALIDATION ============
        proj.eval(); pool.eval(); head.eval()
        val_probs, val_labels_all = [], []
        
        val_pbar = tqdm(val_iterator, total=len(val_iterator), desc="  Val  ", 
                       leave=False, position=1)
        
        with torch.no_grad():
            for bags, mask, labels in val_pbar:
                if args.amp:
                    with autocast():
                        z = proj(bags)
                        pooled, _ = pool(z, mask, args.att_fix)
                        logits = head(pooled)
                        probs = F.softmax(logits, dim=1)[:, 1]
                else:
                    z = proj(bags)
                    pooled, _ = pool(z, mask, args.att_fix)
                    logits = head(pooled)
                    probs = F.softmax(logits, dim=1)[:, 1]
                
                val_probs.append(probs.cpu().numpy())
                val_labels_all.append(labels.cpu().numpy())
                del bags, mask, labels
        
        val_pbar.close()
        val_probs = np.concatenate(val_probs)
        val_labels_all = np.concatenate(val_labels_all)
        val_auc = roc_auc_score(val_labels_all, val_probs) if len(np.unique(val_labels_all)) > 1 else 0.5

        # ============ TRACKING (use val to decide; test forward only on new best) ============
        # We still call update() every epoch so EMA/MA states advance; test_auc here is a placeholder.
        test_auc_placeholder = float(last_test_auc) if last_test_auc is not None else 0.0
        ma_results = ma_tracker.update(val_auc, test_auc_placeholder, epoch)
        current_metric = ma_tracker.get_metric_value(args.early_stop_primary)

        # Decide if this epoch is a "new best" according to primary metric
        test_was_run = False
        test_auc = last_test_auc  # shown as stale unless we run it now

        if epoch >= args.warmup_patience and current_metric is not None:
            is_new_best = (best_metric_value is None) or (current_metric > best_metric_value + args.early_stop_delta)
            if is_new_best:
                # ============ TEST (ONLY when primary metric improves) ============
                test_probs, test_labels_all = [], []
                test_pbar = tqdm(test_iterator, total=len(test_iterator), desc="  Test ", 
                                leave=False, position=1)
                
                with torch.no_grad():
                    for bags, mask, labels in test_pbar:
                        if args.amp:
                            with autocast():
                                z = proj(bags)
                                pooled, _ = pool(z, mask, args.att_fix)
                                logits = head(pooled)
                                probs = F.softmax(logits, dim=1)[:, 1]
                        else:
                            z = proj(bags)
                            pooled, _ = pool(z, mask, args.att_fix)
                            logits = head(pooled)
                            probs = F.softmax(logits, dim=1)[:, 1]
                        
                        test_probs.append(probs.cpu().numpy())
                        test_labels_all.append(labels.cpu().numpy())
                        del bags, mask, labels
                
                test_pbar.close()
                test_probs = np.concatenate(test_probs)
                test_labels_all = np.concatenate(test_labels_all)
                test_preds_now = _preds_by_sample(test_probs, test_labels_all, test_iterator, test_sids, "test")
                test_auc = (roc_auc_score(test_preds_now['label'], test_preds_now['prob'])
                            if len(np.unique(test_preds_now['label'])) > 1 else 0.5)

                last_test_auc = test_auc
                test_was_run = True

                best_metric_value = current_metric
                best_metric_epoch = epoch
                es_patience_counter = 0
                
                best_state = {
                    'proj': {k: v.cpu().clone() for k, v in proj.state_dict().items()},
                    'pool': {k: v.cpu().clone() for k, v in pool.state_dict().items()},
                    'head': {k: v.cpu().clone() for k, v in head.state_dict().items()},
                }
                best_val_preds = _preds_by_sample(val_probs, val_labels_all, val_iterator, val_sids, "val")
                best_test_preds = test_preds_now
            else:
                es_patience_counter += 1
                if es_patience_counter >= args.early_stop_patience:
                    tqdm.write(f"  Early stop at epoch {epoch}")
                    break

        # Logging
        test_disp = "skip" if (not test_was_run and test_auc is None) else (f"{test_auc:.3f}" if test_auc is not None else "skip")
        epoch_pbar.set_postfix({
            'val': f'{val_auc:.3f}',
            'test': test_disp,
            'loss': f'{train_loss:.4f}'
        })

        star_bits = [tag.upper() for tag in ('ema3', 'ema5', 'ema10', 'ma3', 'ma5', 'ma10')
                     if ma_results.get(f"{tag}_new_max")]
        if (epoch % 10 == 0) or star_bits:
            msg = f"  [ep {epoch:03d}] val={val_auc:.4f}"
            if test_was_run:
                msg += f" test={test_auc:.4f}"
            elif test_auc is not None:
                msg += f" test(stale)={test_auc:.4f}"
            else:
                msg += f" test=SKIP"
            if star_bits:
                msg += f"  ⭐ {','.join(star_bits)}"
            tqdm.write(msg)

    epoch_pbar.close()

    # ============ SAVE ============
    if best_state is None:
        best_state = {
            'proj': {k: v.cpu().clone() for k, v in proj.state_dict().items()},
            'pool': {k: v.cpu().clone() for k, v in pool.state_dict().items()},
            'head': {k: v.cpu().clone() for k, v in head.state_dict().items()},
        }
        best_val_preds = _preds_by_sample(val_probs, val_labels_all, val_iterator, val_sids, "val")
        best_metric_epoch = epoch
        best_metric_value = val_auc

    # Load best and (if needed) compute test preds once for saving
    proj.load_state_dict({k: v.to(device) for k, v in best_state['proj'].items()})
    pool.load_state_dict({k: v.to(device) for k, v in best_state['pool'].items()})
    head.load_state_dict({k: v.to(device) for k, v in best_state['head'].items()})

    if best_test_preds is None:
        # Run test once at the end so outputs are always present
        proj.eval(); pool.eval(); head.eval()
        test_probs, test_labels_all = [], []
        test_pbar = tqdm(test_iterator, total=len(test_iterator), desc="  Test ", 
                        leave=False, position=1)
        with torch.no_grad():
            for bags, mask, labels in test_pbar:
                if args.amp:
                    with autocast():
                        z = proj(bags)
                        pooled, _ = pool(z, mask, args.att_fix)
                        logits = head(pooled)
                        probs = F.softmax(logits, dim=1)[:, 1]
                else:
                    z = proj(bags)
                    pooled, _ = pool(z, mask, args.att_fix)
                    logits = head(pooled)
                    probs = F.softmax(logits, dim=1)[:, 1]
                test_probs.append(probs.cpu().numpy())
                test_labels_all.append(labels.cpu().numpy())
                del bags, mask, labels
        test_pbar.close()
        test_probs = np.concatenate(test_probs)
        test_labels_all = np.concatenate(test_labels_all)
        best_test_preds = _preds_by_sample(test_probs, test_labels_all, test_iterator, test_sids, "test")

    checkpoint = {
        'proj_state_dict': best_state['proj'],
        'pool_state_dict': best_state['pool'],
        'head_state_dict': best_state['head'],
        'config': {
            'feat_dim': feat_dim,
            'feat_proj': args.feat_proj,
            'feat_proj_dim': args.feat_proj_dim,
        },
        'best_epoch': best_metric_epoch,
        'best_metric': best_metric_value,
    }
    torch.save(checkpoint, fold_dir / "checkpoint.pt")

    pd.DataFrame(best_val_preds).to_csv(fold_dir / "inner_val_preds.csv", index=False)
    pd.DataFrame(best_test_preds).to_csv(fold_dir / "outer_test_preds.csv", index=False)

    val_auc_best = roc_auc_score(best_val_preds['label'], best_val_preds['prob']) if len(np.unique(best_val_preds['label'])) > 1 else 0.5
    test_auc_best = roc_auc_score(best_test_preds['label'], best_test_preds['prob']) if len(np.unique(best_test_preds['label'])) > 1 else 0.5
    
    print(f"  Best epoch: {best_metric_epoch}")
    print(f"  Val AUC: {val_auc_best:.4f} | Test AUC: {test_auc_best:.4f}")

    del proj, pool, head
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'inner_fold': inner_fold,
        'best_epoch': best_metric_epoch,
        'best_metric': best_metric_value,
        'val_auc': val_auc_best,
        'test_auc': test_auc_best,
        'val_preds': best_val_preds,
        'test_preds': best_test_preds,
    }


# =============================================================================
# COMMON TRAINING LOOP (for GPU-resident and legacy CPU modes)
# =============================================================================

def _train_inner_fold_common(
    inner_fold: int,
    train_loader,
    val_loader,
    test_loader,
    feat_dim: int,
    device: str,
    args,
    output_dir: Path,
    normalizer: nn.Module = None,
) -> dict:
    """Common training loop for GPU-resident and legacy CPU-batch modes."""
    
    fold_dir = output_dir / f"inner_fold_{inner_fold}"
    
    # Configure regularization
    if args.reg_scheme == 'dropout':
        proj_dropout = args.scheme_value if args.feat_proj != 'none' else 0.0
        att_dropout = args.scheme_value
        proj_C = None
        att_C = None
    else:
        proj_dropout = 0.0
        att_dropout = 0.0
        proj_C = args.scheme_value if args.feat_proj != 'none' else None
        att_C = args.scheme_value

    # Build model
    if args.feat_proj == 'none':
        proj = FeatureProjector(feat_dim, mode='none').to(device)
        proj_dim = feat_dim
    else:
        proj = FeatureProjector(feat_dim, args.feat_proj_dim, mode=args.feat_proj,
                                use_ln=args.feat_proj_ln, dropout=proj_dropout).to(device)
        proj_dim = proj.out_dim

    pool = GatedAttentionPool(proj_dim, att_dim=64, dropout=att_dropout,
                              temperature=args.att_temperature).to(device)
    head = nn.Linear(proj_dim, 2).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)

    params = list(pool.parameters()) + list(head.parameters())
    if args.feat_proj != 'none':
        params.extend(proj.parameters())
    opt = torch.optim.Adam(params, lr=args.base_lr, weight_decay=0.0)
    scheduler = ThreePhaseCosineScheduler(opt, args.lr_warmup, args.lr_flat, args.max_epochs,
                                          args.base_lr, args.min_lr)
    crit = nn.CrossEntropyLoss()

    scaler = GradScaler() if args.amp else None
    accumulation_steps = getattr(args, 'gradient_accumulation_steps', 1)

    # Tracking
    best_metric_value = None
    best_metric_epoch = -1
    best_state = None
    es_patience_counter = 0
    ma_tracker = MovingAverageTracker([3, 5, 10], warmup_epochs=args.warmup_patience)
    best_val_preds = None
    best_test_preds = None

    # NEW: only run test forward on "new best" (based on early_stop_primary)
    last_test_auc = None

    epoch_pbar = tqdm(range(args.max_epochs), desc=f"IF{inner_fold}", leave=False, position=0)

    for epoch in epoch_pbar:
        lr = scheduler.step(epoch)

        # ============ TRAINING ============
        proj.train(); pool.train(); head.train()
        train_loss_acc = 0.0
        n_train_samples = 0
        opt.zero_grad()
        accum_step = 0

        train_pbar = tqdm(
            enumerate(train_loader), 
            total=len(train_loader),
            desc="  Train", 
            leave=False, 
            position=1,
        )

        for batch_idx, (bags, labels, mask, sids) in train_pbar:
            # Transfer to GPU if needed (CPU-batch mode)
            if normalizer is not None:
                bags = bags.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                bags = bags.float()
                bags = normalizer(bags)
            
            if args.amp:
                with autocast():
                    z = proj(bags)
                    pooled, _ = pool(z, mask, args.att_fix)
                    logits = head(pooled)
                    loss = crit(logits, labels) + head_elasticnet(head, args.head_C, args.head_l1_ratio)
                    if args.reg_scheme == 'ridgeC':
                        if args.feat_proj != 'none' and proj.linear is not None:
                            loss += submodule_l2_penalty([proj.linear], proj_C)
                        loss += submodule_l2_penalty(pool.attention_params, att_C)
                    loss = loss / accumulation_steps
                scaler.scale(loss).backward()
            else:
                z = proj(bags)
                pooled, _ = pool(z, mask, args.att_fix)
                logits = head(pooled)
                loss = crit(logits, labels) + head_elasticnet(head, args.head_C, args.head_l1_ratio)
                if args.reg_scheme == 'ridgeC':
                    if args.feat_proj != 'none' and proj.linear is not None:
                        loss += submodule_l2_penalty([proj.linear], proj_C)
                    loss += submodule_l2_penalty(pool.attention_params, att_C)
                loss = loss / accumulation_steps
                loss.backward()
            
            train_loss_acc += loss.item() * len(labels) * accumulation_steps
            n_train_samples += len(labels)
            accum_step += 1
            
            if accum_step % accumulation_steps == 0:
                if args.amp:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
                    opt.step()
                opt.zero_grad()
            
            train_pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
            del bags, labels, mask

        train_pbar.close()
        train_loss = train_loss_acc / max(1, n_train_samples)

        # ============ VALIDATION ============
        proj.eval(); pool.eval(); head.eval()
        val_probs, val_labels_all, val_sids_all = [], [], []
        
        val_pbar = tqdm(val_loader, total=len(val_loader), desc="  Val  ", leave=False, position=1)
        
        with torch.no_grad():
            for bags, labels, mask, sids in val_pbar:
                if normalizer is not None:
                    bags = bags.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    bags = bags.float()
                    bags = normalizer(bags)
                
                if args.amp:
                    with autocast():
                        z = proj(bags)
                        pooled, _ = pool(z, mask, args.att_fix)
                        logits = head(pooled)
                        probs = F.softmax(logits, dim=1)[:, 1]
                else:
                    z = proj(bags)
                    pooled, _ = pool(z, mask, args.att_fix)
                    logits = head(pooled)
                    probs = F.softmax(logits, dim=1)[:, 1]
                
                val_probs.append(probs.cpu().numpy())
                val_labels_all.append(labels.cpu().numpy())
                val_sids_all.extend(sids)
                del bags, labels, mask
        
        val_pbar.close()
        val_probs = np.concatenate(val_probs)
        val_labels_all = np.concatenate(val_labels_all)
        val_auc = roc_auc_score(val_labels_all, val_probs) if len(np.unique(val_labels_all)) > 1 else 0.5

        # ============ TRACKING (use val to decide; test forward only on new best) ============
        test_auc_placeholder = float(last_test_auc) if last_test_auc is not None else 0.0
        ma_results = ma_tracker.update(val_auc, test_auc_placeholder, epoch)
        current_metric = ma_tracker.get_metric_value(args.early_stop_primary)

        test_was_run = False
        test_auc = last_test_auc

        if epoch >= args.warmup_patience and current_metric is not None:
            is_new_best = (best_metric_value is None) or (current_metric > best_metric_value + args.early_stop_delta)
            if is_new_best:
                # ============ TEST (ONLY when primary metric improves) ============
                test_probs, test_labels_all, test_sids_all = [], [], []
                test_pbar = tqdm(test_loader, total=len(test_loader), desc="  Test ", leave=False, position=1)
                
                with torch.no_grad():
                    for bags, labels, mask, sids in test_pbar:
                        if normalizer is not None:
                            bags = bags.to(device, non_blocking=True)
                            labels = labels.to(device, non_blocking=True)
                            mask = mask.to(device, non_blocking=True)
                            bags = bags.float()
                            bags = normalizer(bags)
                        
                        if args.amp:
                            with autocast():
                                z = proj(bags)
                                pooled, _ = pool(z, mask, args.att_fix)
                                logits = head(pooled)
                                probs = F.softmax(logits, dim=1)[:, 1]
                        else:
                            z = proj(bags)
                            pooled, _ = pool(z, mask, args.att_fix)
                            logits = head(pooled)
                            probs = F.softmax(logits, dim=1)[:, 1]
                        
                        test_probs.append(probs.cpu().numpy())
                        test_labels_all.append(labels.cpu().numpy())
                        test_sids_all.extend(sids)
                        del bags, labels, mask
                
                test_pbar.close()
                test_probs = np.concatenate(test_probs)
                test_labels_all = np.concatenate(test_labels_all)
                test_auc = roc_auc_score(test_labels_all, test_probs) if len(np.unique(test_labels_all)) > 1 else 0.5

                last_test_auc = test_auc
                test_was_run = True

                best_metric_value = current_metric
                best_metric_epoch = epoch
                es_patience_counter = 0
                
                best_state = {
                    'proj': {k: v.cpu().clone() for k, v in proj.state_dict().items()},
                    'pool': {k: v.cpu().clone() for k, v in pool.state_dict().items()},
                    'head': {k: v.cpu().clone() for k, v in head.state_dict().items()},
                }
                best_val_preds = {
                    'sample_id': val_sids_all,
                    'label': val_labels_all.tolist(),
                    'prob': val_probs.tolist(),
                }
                best_test_preds = {
                    'sample_id': test_sids_all,
                    'label': test_labels_all.tolist(),
                    'prob': test_probs.tolist(),
                }
            else:
                es_patience_counter += 1
                if es_patience_counter >= args.early_stop_patience:
                    tqdm.write(f"  Early stop at epoch {epoch}")
                    break

        test_disp = "skip" if (not test_was_run and test_auc is None) else (f"{test_auc:.3f}" if test_auc is not None else "skip")
        epoch_pbar.set_postfix({
            'val': f'{val_auc:.3f}',
            'test': test_disp,
            'loss': f'{train_loss:.4f}'
        })

        star_bits = [tag.upper() for tag in ('ema3', 'ema5', 'ema10', 'ma3', 'ma5', 'ma10')
                     if ma_results.get(f"{tag}_new_max")]
        if (epoch % 10 == 0) or star_bits:
            msg = f"  [ep {epoch:03d}] val={val_auc:.4f}"
            if test_was_run:
                msg += f" test={test_auc:.4f}"
            elif test_auc is not None:
                msg += f" test(stale)={test_auc:.4f}"
            else:
                msg += f" test=SKIP"
            if star_bits:
                msg += f"  ⭐ {','.join(star_bits)}"
            tqdm.write(msg)

    epoch_pbar.close()

    # ============ SAVE ============
    if best_state is None:
        best_state = {
            'proj': {k: v.cpu().clone() for k, v in proj.state_dict().items()},
            'pool': {k: v.cpu().clone() for k, v in pool.state_dict().items()},
            'head': {k: v.cpu().clone() for k, v in head.state_dict().items()},
        }
        best_val_preds = {
            'sample_id': val_sids_all,
            'label': val_labels_all.tolist(),
            'prob': val_probs.tolist(),
        }
        best_metric_epoch = epoch
        best_metric_value = val_auc

    proj.load_state_dict({k: v.to(device) for k, v in best_state['proj'].items()})
    pool.load_state_dict({k: v.to(device) for k, v in best_state['pool'].items()})
    head.load_state_dict({k: v.to(device) for k, v in best_state['head'].items()})

    if best_test_preds is None:
        proj.eval(); pool.eval(); head.eval()
        test_probs, test_labels_all, test_sids_all = [], [], []
        test_pbar = tqdm(test_loader, total=len(test_loader), desc="  Test ", leave=False, position=1)
        with torch.no_grad():
            for bags, labels, mask, sids in test_pbar:
                if normalizer is not None:
                    bags = bags.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    bags = bags.float()
                    bags = normalizer(bags)
                
                if args.amp:
                    with autocast():
                        z = proj(bags)
                        pooled, _ = pool(z, mask, args.att_fix)
                        logits = head(pooled)
                        probs = F.softmax(logits, dim=1)[:, 1]
                else:
                    z = proj(bags)
                    pooled, _ = pool(z, mask, args.att_fix)
                    logits = head(pooled)
                    probs = F.softmax(logits, dim=1)[:, 1]
                
                test_probs.append(probs.cpu().numpy())
                test_labels_all.append(labels.cpu().numpy())
                test_sids_all.extend(sids)
                del bags, labels, mask
        test_pbar.close()
        test_probs = np.concatenate(test_probs)
        test_labels_all = np.concatenate(test_labels_all)
        best_test_preds = {
            'sample_id': test_sids_all,
            'label': test_labels_all.tolist(),
            'prob': test_probs.tolist(),
        }

    checkpoint = {
        'proj_state_dict': best_state['proj'],
        'pool_state_dict': best_state['pool'],
        'head_state_dict': best_state['head'],
        'config': {
            'feat_dim': feat_dim,
            'feat_proj': args.feat_proj,
            'feat_proj_dim': args.feat_proj_dim,
            'feat_proj_ln': args.feat_proj_ln,
            'att_temperature': args.att_temperature,
            'att_fix': args.att_fix,
            'head_C': args.head_C,
            'head_l1_ratio': args.head_l1_ratio,
            'reg_scheme': args.reg_scheme,
            'scheme_value': args.scheme_value,
        },
        'best_epoch': best_metric_epoch,
        'best_metric': best_metric_value,
    }
    torch.save(checkpoint, fold_dir / "checkpoint.pt")

    # Save attention weights
    attention_weights = {
        'V_weight': pool.V.weight.detach().cpu(),
        'V_bias': pool.V.bias.detach().cpu(),
        'U_weight': pool.U.weight.detach().cpu(),
        'U_bias': pool.U.bias.detach().cpu(),
        'w_weight': pool.w.weight.detach().cpu(),
        'w_bias': pool.w.bias.detach().cpu(),
    }
    torch.save(attention_weights, fold_dir / "attention_weights.pt")

    # Save head coefficients
    head_weight = head.weight.detach().cpu().numpy()
    head_bias = head.bias.detach().cpu().numpy()
    
    coef_df = pd.DataFrame({
        'feature_idx': range(head_weight.shape[1]),
        'weight_class0': head_weight[0, :],
        'weight_class1': head_weight[1, :],
        'weight_diff': head_weight[1, :] - head_weight[0, :],
    })
    coef_df.to_csv(fold_dir / "head_coefficients.csv", index=False)
    
    with open(fold_dir / "head_bias.json", 'w') as f:
        json.dump({'bias_class0': float(head_bias[0]), 'bias_class1': float(head_bias[1])}, f)

    # Save projector weights if used
    if args.feat_proj != 'none' and proj.linear is not None:
        proj_weights = {
            'weight': proj.linear.weight.detach().cpu(),
            'bias': proj.linear.bias.detach().cpu(),
        }
        torch.save(proj_weights, fold_dir / "projector_weights.pt")

    pd.DataFrame(best_val_preds).to_csv(fold_dir / "inner_val_preds.csv", index=False)
    pd.DataFrame(best_test_preds).to_csv(fold_dir / "outer_test_preds.csv", index=False)

    val_auc_best = roc_auc_score(best_val_preds['label'], best_val_preds['prob']) if len(np.unique(best_val_preds['label'])) > 1 else 0.5
    test_auc_best = roc_auc_score(best_test_preds['label'], best_test_preds['prob']) if len(np.unique(best_test_preds['label'])) > 1 else 0.5
    
    print(f"  Best epoch: {best_metric_epoch}")
    print(f"  Val AUC: {val_auc_best:.4f} | Test AUC: {test_auc_best:.4f}")

    del proj, pool, head
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'inner_fold': inner_fold,
        'best_epoch': best_metric_epoch,
        'best_metric': best_metric_value,
        'val_auc': val_auc_best,
        'test_auc': test_auc_best,
        'val_preds': best_val_preds,
        'test_preds': best_test_preds,
    }


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Train MIL model - GPU OPTIMIZED")
    
    # Data
    p.add_argument("--config", required=True, type=str, help="YAML config file")
    p.add_argument("--split", required=True, type=str, help="Split JSON file")
    p.add_argument("--fold", default=0, type=int, help="Outer fold")
    p.add_argument("--seed", default=42, type=int)
    
    # Data mode
    p.add_argument("--cpu_data", action="store_true", default=False,
                   help="Keep data on CPU, transfer to GPU per batch")
    p.add_argument("--gpu_fp16", action="store_true", default=False,
                   help="Store GPU data as fp16 (half memory)")
    p.add_argument("--frags_per_bag", default=None, type=int,
                   help="Override frags_per_bag from config")
    
    # OPTIMIZED PIPELINE (new!)
    p.add_argument("--optimized_pipeline", action="store_true", default=False,
                   help="Use optimized pipeline (NVMe + multi-reader + pinned + GPU prefetch)")
    p.add_argument("--nvme_path", type=str, default="/tmp/leaf1_mil_cache",
                   help="Path for NVMe storage (pre-collated epochs)")
    p.add_argument("--n_precollate_epochs", type=int, default=5,
                   help="Number of epochs to pre-collate for training")
    p.add_argument("--n_readers", type=int, default=6,
                   help="Number of reader threads for optimized pipeline")
    p.add_argument("--queue_depth", type=int, default=3,
                   help="Queue depth for prefetching")
    
    # Legacy prefetch (for backward compat)
    p.add_argument("--prefetch", action="store_true", default=False,
                   help="Enable legacy CPU prefetching")
    p.add_argument("--prefetch_count", default=5, type=int,
                   help="Number of batches to prefetch (legacy)")
    
    # AMP and gradient accumulation
    p.add_argument("--amp", action="store_true", default=False,
                   help="Use automatic mixed precision")
    p.add_argument("--gradient_accumulation_steps", default=1, type=int,
                   help="Accumulate gradients over N batches")
    
    # Inner CV
    p.add_argument("--inner_k", default=5, type=int)
    p.add_argument("--inner_folds", default="0,1,2,3,4", type=str)
    
    # Model
    p.add_argument("--feat_proj", default="linear", choices=["none", "linear"])
    p.add_argument("--feat_proj_dim", default=512, type=int)
    p.add_argument("--feat_proj_ln", action="store_true", default=True)
    p.add_argument("--att_temperature", default=1.0, type=float)
    p.add_argument("--att_fix", default="hhi", type=str)
    
    # Regularization
    p.add_argument("--head_C", default=3.16, type=float)
    p.add_argument("--head_l1_ratio", default=0.3, type=float)
    p.add_argument("--reg_scheme", default="ridgeC", choices=["dropout", "ridgeC"])
    p.add_argument("--scheme_value", default=3.16, type=float)
    
    # Training
    p.add_argument("--max_epochs", default=200, type=int)
    p.add_argument("--warmup_patience", default=10, type=int)
    p.add_argument("--lr_warmup", default=5, type=int)
    p.add_argument("--lr_flat", default=15, type=int)
    p.add_argument("--base_lr", default=5e-4, type=float)
    p.add_argument("--min_lr", default=1e-6, type=float)
    p.add_argument("--batch_size", default=32, type=int)
    
    # Early stopping
    p.add_argument("--early_stop_primary", default="ema5", type=str)
    p.add_argument("--early_stop_delta", default=0.005, type=float)
    p.add_argument("--early_stop_patience", default=20, type=int)
    
    # I/O
    p.add_argument("--load_workers", default=40, type=int)
    p.add_argument("--n_workers", default=8, type=int, help="Workers for pre-collation")
    p.add_argument("--output_dir", required=True, type=str)
    
    return p.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = parse_args()
    set_all_seeds(args.seed)
    # args.early_stop_patience = 10
    
    # Allocator config left to the user. Earlier versions hardcoded
    # 'expandable_segments:True,max_split_size_mb:128' here, but
    # 'expandable_segments:True' triggers a known torch 2.1.x internal-assert in
    # the MIL head's forward pass on some Hopper drivers
    # (RuntimeError: !block->expandable_segment_). Set PYTORCH_CUDA_ALLOC_CONF
    # in your launch environment if you want a specific allocator policy; a
    # sane default is 'max_split_size_mb:128' (no expandable_segments).

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    seg_str = cfg["data"]["segments"]
    
    if args.frags_per_bag is not None:
        n_frags = args.frags_per_bag
        print(f">>> Using command-line override: frags_per_bag={n_frags}")
    else:
        n_frags = cfg["data"]["frags_per_bag"]
    
    embed_key = cfg["data"].get("embed_key", "embeddings")
    layer_idx = cfg["data"].get("layer_idx", 4)
    
    # Determine data mode string
    mode_parts = []
    if args.cpu_data:
        if args.optimized_pipeline:
            mode_parts.append("OPTIMIZED_PIPELINE")
        else:
            mode_parts.append("cpu_batch_legacy")
    elif args.gpu_fp16:
        mode_parts.append("gpu_fp16")
    else:
        mode_parts.append("gpu_fp32")
    
    if args.prefetch and not args.optimized_pipeline:
        mode_parts.append(f"prefetch_q{args.prefetch_count}")
    if args.amp:
        mode_parts.append("amp")
    if args.gradient_accumulation_steps > 1:
        mode_parts.append(f"accum{args.gradient_accumulation_steps}")
    
    data_mode_str = "+".join(mode_parts)
    
    # Save config
    config_save = {
        'args': vars(args),
        'data_config': cfg['data'],
        'frags_per_bag_used': n_frags,
        'data_mode': data_mode_str,
        'optimizations': {
            'optimized_pipeline': args.optimized_pipeline,
            'n_precollate_epochs': args.n_precollate_epochs if args.optimized_pipeline else 0,
            'n_readers': args.n_readers if args.optimized_pipeline else 0,
            'queue_depth': args.queue_depth if args.optimized_pipeline else 0,
            'nvme_path': args.nvme_path if args.optimized_pipeline else None,
            'amp': args.amp,
            'gradient_accumulation': args.gradient_accumulation_steps,
        },
        'timestamp': datetime.now().isoformat(),
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_save, f, indent=2)
    
    # Load splits
    outer_train_df, outer_test_df, split_meta = load_splits(args.split, args.fold)
    dataset_name = Path(args.split).stem.split("_")[0]
    
    print(f"\n{'='*70}")
    print("MIL TRAINING - GPU OPTIMIZED")
    print(f"{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"Fragments per bag: {n_frags:,}")
    print(f"Data mode: {data_mode_str}")
    print(f"Batch size: {args.batch_size}")
    if args.gradient_accumulation_steps > 1:
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    if args.optimized_pipeline:
        print(f"\n🚀 OPTIMIZED PIPELINE:")
        print(f"   NVMe path: {args.nvme_path}")
        print(f"   Pre-collate epochs: {args.n_precollate_epochs}")
        print(f"   Reader threads: {args.n_readers}")
        print(f"   Queue depth: {args.queue_depth}")
    print(f"{'='*70}")
    
    # Load bags
    print("\n>>> Loading bags...")
    bags_outer_train, y_outer_train, sids_outer_train, feat_dim = load_bags_parallel(
        outer_train_df, seg_str, n_frags, embed_key, layer_idx, args.seed, args.load_workers
    )
    bags_outer_test, y_outer_test, sids_outer_test, _ = load_bags_parallel(
        outer_test_df, seg_str, n_frags, embed_key, layer_idx, args.seed, args.load_workers
    )
    
    print(f"  Outer train: {len(bags_outer_train)} samples")
    print(f"  Outer test: {len(bags_outer_test)} samples")
    print(f"  Feature dim: {feat_dim}")
    
    # Inner CV splits
    inner_folds_to_train = [int(x) for x in args.inner_folds.split(",")]
    y_labels = outer_train_df["label"].values
    skf = StratifiedKFold(n_splits=args.inner_k, shuffle=True, random_state=args.seed)
    all_splits = list(skf.split(np.zeros(len(y_labels)), y_labels))
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n>>> Using device: {device}")
    
    all_results = []
    all_oof_preds = []
    all_test_preds = []
    
    # ========================================================================
    # GPU-RESIDENT MODE
    # ========================================================================
    if not args.cpu_data:
        storage_dtype = "fp16" if args.gpu_fp16 else "fp32"
        print(f"\n>>> Moving data to GPU ({storage_dtype})...")
        
        for inner_fold in inner_folds_to_train:
            tr_idx, va_idx = all_splits[inner_fold]
            
            print(f"\n>>> Computing normalization for IF{inner_fold}...")
            mu, sigma = compute_fragment_stats_from_bags([bags_outer_train[i] for i in tr_idx])
            norm = FrozenStandardize1d(mu.to(device), sigma.to(device))
            
            if args.gpu_fp16:
                all_bags_gpu = [norm(torch.from_numpy(b).float().to(device)).half() for b in bags_outer_train]
                test_bags_gpu = [norm(torch.from_numpy(b).float().to(device)).half() for b in bags_outer_test]
            else:
                all_bags_gpu = [norm(torch.from_numpy(b).float().to(device)) for b in bags_outer_train]
                test_bags_gpu = [norm(torch.from_numpy(b).float().to(device)) for b in bags_outer_test]
            
            all_labels_gpu = torch.tensor(y_outer_train, dtype=torch.long, device=device)
            test_labels_gpu = torch.tensor(y_outer_test, dtype=torch.long, device=device)
            
            fold_dir = output_dir / f"inner_fold_{inner_fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': mu, 'std': sigma}, fold_dir / "norm_stats.pt")
            
            result = train_inner_fold_gpu(
                inner_fold=inner_fold,
                bags_tr_idx=list(tr_idx),
                bags_va_idx=list(va_idx),
                all_bags_gpu=all_bags_gpu,
                all_labels_gpu=all_labels_gpu,
                all_sample_ids=list(sids_outer_train),
                test_bags_gpu=test_bags_gpu,
                test_labels_gpu=test_labels_gpu,
                test_sample_ids=list(sids_outer_test),
                feat_dim=feat_dim,
                device=device,
                args=args,
                output_dir=output_dir,
            )
            
            all_results.append(result)
            all_oof_preds.append(result['val_preds'])
            all_test_preds.append(result['test_preds'])
            
            del all_bags_gpu, test_bags_gpu, all_labels_gpu, test_labels_gpu, norm
            gc.collect()
            torch.cuda.empty_cache()
    
    # ========================================================================
    # CPU-BATCH MODE (OPTIMIZED or LEGACY)
    # ========================================================================
    else:
        y_outer_train = np.array(y_outer_train)
        y_outer_test = np.array(y_outer_test)
        
        for inner_fold in inner_folds_to_train:
            tr_idx, va_idx = all_splits[inner_fold]
            
            print(f"\n>>> Computing normalization for IF{inner_fold}...")
            mu, sigma = compute_fragment_stats_from_bags([bags_outer_train[i] for i in tr_idx])
            normalizer = FrozenStandardize1d(mu.to(device), sigma.to(device))
            
            fold_dir = output_dir / f"inner_fold_{inner_fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': mu, 'std': sigma}, fold_dir / "norm_stats.pt")
            
            # Choose training function based on mode
            if args.optimized_pipeline:
                result = train_inner_fold_cpu_optimized(
                    inner_fold=inner_fold,
                    bags_tr_idx=list(tr_idx),
                    bags_va_idx=list(va_idx),
                    bags_outer_train=bags_outer_train,
                    y_outer_train=y_outer_train,
                    sids_outer_train=list(sids_outer_train),
                    bags_outer_test=bags_outer_test,
                    y_outer_test=y_outer_test,
                    sids_outer_test=list(sids_outer_test),
                    normalizer=normalizer,
                    feat_dim=feat_dim,
                    device=device,
                    args=args,
                    output_dir=output_dir,
                )
            else:
                result = train_inner_fold_cpu(
                    inner_fold=inner_fold,
                    bags_tr_idx=list(tr_idx),
                    bags_va_idx=list(va_idx),
                    bags_outer_train=bags_outer_train,
                    y_outer_train=y_outer_train,
                    sids_outer_train=list(sids_outer_train),
                    bags_outer_test=bags_outer_test,
                    y_outer_test=y_outer_test,
                    sids_outer_test=list(sids_outer_test),
                    normalizer=normalizer,
                    feat_dim=feat_dim,
                    device=device,
                    args=args,
                    output_dir=output_dir,
                )
            
            all_results.append(result)
            all_oof_preds.append(result['val_preds'])
            all_test_preds.append(result['test_preds'])
            
            del normalizer
            gc.collect()
            torch.cuda.empty_cache()
    
    # ========================================================================
    # AGGREGATE OUTPUTS
    # ========================================================================
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS")
    print(f"{'='*60}")
    
    oof_df = pd.concat([pd.DataFrame(p) for p in all_oof_preds], ignore_index=True)
    oof_df.to_csv(output_dir / "oof_predictions.csv", index=False)
    print(f"  OOF predictions: {len(oof_df)} samples")
    
    # align inner folds by sample ID before averaging
    ref_ids = list(all_test_preds[0]['sample_id'])
    aligned = []
    for p in all_test_preds:
        df_p = pd.DataFrame(p).set_index('sample_id')
        assert df_p.index.is_unique and set(df_p.index) == set(ref_ids), "inner folds scored different test samples"
        df_p = df_p.loc[ref_ids]
        assert (df_p['label'].to_numpy() == np.asarray(all_test_preds[0]['label'])).all(), "label mismatch across inner folds"
        aligned.append(df_p['prob'].to_numpy())
    ensemble_probs = np.mean(aligned, axis=0)
    
    ensemble_df = pd.DataFrame({
        'sample_id': ref_ids,
        'label': all_test_preds[0]['label'],
        'prob_ensemble': ensemble_probs.tolist(),
    })
    for i, a in enumerate(aligned):
        ensemble_df[f'prob_if{inner_folds_to_train[i]}'] = a
    ensemble_df.to_csv(output_dir / "ensemble_test_preds.csv", index=False)
    
    if len(np.unique(ensemble_df['label'])) > 1:
        ensemble_test_auc = roc_auc_score(ensemble_df['label'], ensemble_df['prob_ensemble'])
        ensemble_test_auprc = average_precision_score(ensemble_df['label'], ensemble_df['prob_ensemble'])
    else:
        ensemble_test_auc = 0.5
        ensemble_test_auprc = np.mean(ensemble_df['label'])
    
    mean_test_auc = np.mean([r['test_auc'] for r in all_results])
    
    print(f"\n  Ensemble Test AUC: {ensemble_test_auc:.4f}")
    print(f"  Mean of AUCs:      {mean_test_auc:.4f}")
    print(f"  Ensemble AUPRC:    {ensemble_test_auprc:.4f}")
    
    summary = {
        'dataset': dataset_name,
        'split': Path(args.split).name,
        'outer_fold': args.fold,
        'n_inner_folds': len(all_results),
        'frags_per_bag': n_frags,
        'data_mode': data_mode_str,
        'optimizations': config_save['optimizations'],
        'per_fold_results': [
            {
                'inner_fold': r['inner_fold'],
                'best_epoch': r['best_epoch'],
                'val_auc': r['val_auc'],
                'test_auc': r['test_auc'],
            }
            for r in all_results
        ],
        'ensemble_metrics': {
            'test_auc': ensemble_test_auc,
            'test_auprc': ensemble_test_auprc,
            'mean_of_aucs': mean_test_auc,
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    import multiprocessing as mp, os
    mp.set_start_method("fork" if os.name == "posix" else "spawn", force=True)
    main()
