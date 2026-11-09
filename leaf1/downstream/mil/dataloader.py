#!/usr/bin/env python3
"""
================================================================================
train_gated_final_dataloader.py

ALL data loading classes for MIL training:

1. GPU-RESIDENT (small datasets):
   - GPUResidentDataset
   - collate_bags_gpu

2. LEGACY CPU-BATCH (backward compatibility):
   - CPUBatchDataset
   - collate_bags_cpu (uses np.stack)
   - PrefetchLoader (single thread CPU prefetch)

3. OPTIMIZED PIPELINE (100K fragments, 10x faster):
   - PrecollatedStorage (NVMe storage management)
   - OptimizedBatchIterator (multi-reader + pinned memory + GPU prefetch)

================================================================================
PERFORMANCE COMPARISON (100K fragments, batch_size=4):
================================================================================

  Legacy CPU-batch:     ~1.0s per batch
  Optimized pipeline:   ~0.1s per batch  → 10x faster!

================================================================================
USAGE:
================================================================================

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

================================================================================
"""

import os
import gc
import shutil
import threading
import queue
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor  # NOTE: ThreadPoolExecutor used for precollation
import multiprocessing as mp

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# =============================================================================
# GPU-RESIDENT DATA LOADING (for small datasets)
# =============================================================================

class GPUResidentDataset(Dataset):
    """Dataset with all bags pre-loaded on GPU.

    Use this when:
    - Dataset fits in GPU memory
    - Typically < 50 samples with 100K fragments
    - Fastest option when memory permits
    """

    def __init__(self, bags_gpu: List[torch.Tensor], labels_gpu: torch.Tensor,
                 sample_ids: List[str] = None, indices: List[int] = None):
        """
        Args:
            bags_gpu: List of tensors on GPU, each (n_frags, feat_dim)
            labels_gpu: Tensor of labels on GPU
            sample_ids: Optional list of sample IDs
            indices: Optional subset indices
        """
        self.bags = bags_gpu
        self.labels = labels_gpu
        self.indices = indices if indices is not None else list(range(len(bags_gpu)))
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        real_idx = self.indices[idx]
        sid = self.sample_ids[real_idx] if self.sample_ids else f"idx_{real_idx}"
        return self.bags[real_idx], self.labels[real_idx], sid


def collate_bags_gpu(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """Collate for GPU-resident data. Handles both fp16 and fp32 input, outputs fp32.

    Args:
        batch: List of (bag, label, sample_id) tuples

    Returns:
        padded: (B, max_len, feat_dim) float32 tensor on GPU
        labels: (B,) int64 tensor on GPU
        mask: (B, max_len) bool tensor on GPU
        sids: List of sample IDs
    """
    bags, labels, sids = zip(*batch)
    max_len = max(b.shape[0] for b in bags)
    feat_dim = bags[0].shape[1]
    device = bags[0].device

    # Always output fp32 for model compatibility (input may be fp16 or fp32)
    padded = torch.zeros(len(bags), max_len, feat_dim, dtype=torch.float32, device=device)
    mask = torch.zeros(len(bags), max_len, dtype=torch.bool, device=device)

    for i, b in enumerate(bags):
        n = b.shape[0]
        padded[i, :n] = b.float()  # Convert to fp32 if needed
        mask[i, :n] = True

    return padded, torch.stack(labels), mask, list(sids)


# =============================================================================
# LEGACY CPU-BATCH DATA LOADING (backward compatibility)
# =============================================================================

class CPUBatchDataset(Dataset):
    """Dataset with bags on CPU, transferred to GPU per batch.

    Use this when:
    - Dataset doesn't fit in GPU memory
    - Using legacy mode (--cpu_data without --optimized_pipeline)
    """

    def __init__(self, bags_cpu: List[np.ndarray], labels: np.ndarray,
                 sample_ids: List[str] = None, indices: List[int] = None):
        """
        Args:
            bags_cpu: List of numpy arrays on CPU
            labels: numpy array of labels
            sample_ids: Optional list of sample IDs
            indices: Optional subset indices
        """
        self.bags = bags_cpu
        self.labels = labels
        self.indices = indices if indices is not None else list(range(len(bags_cpu)))
        self.sample_ids = sample_ids

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, str]:
        real_idx = self.indices[idx]
        bag = self.bags[real_idx]  # numpy array, stays on CPU
        label = self.labels[real_idx]
        sid = self.sample_ids[real_idx] if self.sample_ids else f"idx_{real_idx}"
        return bag, label, sid


def collate_bags_cpu(batch: List[Tuple]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
    """LEGACY CPU collate using np.stack (single C call) + torch.from_numpy (zero-copy).

    Performance for 4 × 100K × 3072 fp16:
    - np.stack: ~0.7s per batch (memory bandwidth limited)
    - Total with transfer: ~1.0s per batch

    Args:
        batch: List of (bag_np, label, sample_id) tuples

    Returns:
        padded: (B, max_len, feat_dim) float16 tensor on CPU
        labels: (B,) int64 tensor on CPU
        mask: (B, max_len) bool tensor on CPU
        sids: List of sample IDs
    """
    bags_np, labels, sids = zip(*batch)

    sizes = [b.shape[0] for b in bags_np]
    feat_dim = bags_np[0].shape[1]
    batch_size = len(bags_np)

    # Fast path: all bags same size (common with fixed frags_per_bag)
    if len(set(sizes)) == 1:
        stacked = np.stack(bags_np, axis=0)  # Single C call
        padded = torch.from_numpy(stacked)    # Zero-copy view
        mask = torch.ones(batch_size, sizes[0], dtype=torch.bool)
    else:
        # Variable sizes - need padding
        max_len = max(sizes)
        padded = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float16)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        padded_np = padded.numpy()
        for i, b in enumerate(bags_np):
            n = b.shape[0]
            np.copyto(padded_np[i, :n], b, casting='no')
            mask[i, :n] = True

    labels_t = torch.tensor(labels, dtype=torch.long)
    return padded, labels_t, mask, list(sids)


class PrefetchLoader:
    """Legacy prefetch loader - single thread CPU prefetching.

    Prefetches CPU batches in background thread (collation only).
    GPU transfer happens in main thread.

    Performance: Limited speedup (~1.5x) because single thread can't keep up
    with memory-bandwidth-limited collation.
    """

    def __init__(self, dataloader: DataLoader, prefetch_count: int = 5):
        """
        Args:
            dataloader: PyTorch DataLoader with CPU collation
            prefetch_count: Number of batches to prefetch on CPU
        """
        self.dataloader = dataloader
        self.prefetch_count = prefetch_count

    def __iter__(self) -> Iterator:
        batch_queue = queue.Queue(maxsize=self.prefetch_count)
        stop_event = threading.Event()

        def producer():
            """Background thread: CPU collation only."""
            try:
                for batch in self.dataloader:
                    if stop_event.is_set():
                        break
                    batch_queue.put(batch)
                batch_queue.put(None)  # Signal end
            except Exception as e:
                batch_queue.put(e)

        # Start background thread
        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        # Yield CPU batches (caller does GPU transfer)
        try:
            while True:
                batch = batch_queue.get()
                if batch is None:
                    break
                if isinstance(batch, Exception):
                    raise batch
                yield batch
        finally:
            stop_event.set()
            thread.join(timeout=1.0)

    def __len__(self) -> int:
        return len(self.dataloader)


# =============================================================================
# OPTIMIZED PIPELINE (10x faster for 100K fragments)
# =============================================================================

def _worker_precollate_epoch(args: Tuple) -> Tuple[int, int, float]:
    """Worker function: pre-collate one epoch's batches to storage.

    MEMORY OPTIMIZED:
    - Writes batch-by-batch to mmap'd file (no 69GB allocation per worker)

    Args:
        args: Tuple of (epoch_idx, bags_list, labels, n_frags, feat_dim,
                       batch_size, n_batches, output_dir, seed, sample_indices, progress_queue)

    Returns:
        (epoch_idx, n_batches, size_gb)
    """
    (epoch_idx, bags_list, labels, n_frags, feat_dim,
     batch_size, n_batches, output_dir, seed, sample_indices, progress_queue) = args

    # Set per-epoch shuffle RNG locally (thread-safe; avoids global np.random.seed)
    rng = np.random.default_rng(seed + epoch_idx)
    shuffled_indices = rng.permutation(sample_indices)

    # Create epoch directory
    epoch_dir = Path(output_dir) / f"epoch_{epoch_idx}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    # row k of the epoch holds sample shuffled_indices[k % n]; saved to map predictions back to samples
    np.save(epoch_dir / "order.npy", np.asarray(shuffled_indices, dtype=np.int64))

    # Create mmap'd output files - write directly to disk, no RAM allocation!
    bags_shape = (n_batches, batch_size, n_frags, feat_dim)
    masks_shape = (n_batches, batch_size, n_frags)
    labels_shape = (n_batches, batch_size)

    bags_path = epoch_dir / "bags.npy"
    masks_path = epoch_dir / "masks.npy"
    labels_path = epoch_dir / "labels.npy"

    # Open mmap'd files for writing (writes directly to disk!)
    all_bags = np.lib.format.open_memmap(
        str(bags_path), mode='w+', dtype=np.float16, shape=bags_shape
    )
    all_masks = np.lib.format.open_memmap(
        str(masks_path), mode='w+', dtype=np.bool_, shape=masks_shape
    )
    all_labels = np.lib.format.open_memmap(
        str(labels_path), mode='w+', dtype=np.int64, shape=labels_shape
    )

    # Fill masks with True (all valid)
    all_masks[:] = True

    # Collate each batch - writes directly to disk via mmap
    n_indices = len(shuffled_indices)
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        # Handle wraparound for last batch
        batch_indices = [shuffled_indices[i % n_indices]
                        for i in range(start_idx, start_idx + batch_size)]

        for i, idx in enumerate(batch_indices):
            all_bags[batch_idx, i] = bags_list[int(idx)]
            all_labels[batch_idx, i] = labels[int(idx)]

        # Flush every batch to ensure writes go to disk
        all_bags.flush()

        # Report progress (if queue provided)
        if progress_queue is not None:
            try:
                progress_queue.put_nowait(1)  # Signal one batch done
            except Exception:
                pass  # Queue full or closed, skip update

    # Final flush
    all_bags.flush()
    all_masks.flush()
    all_labels.flush()

    size_gb = n_batches * batch_size * n_frags * feat_dim * 2 / 1e9
    return epoch_idx, n_batches, size_gb


class PrecollatedStorage:
    """NVMe storage manager for pre-collated epochs.

    Pre-collates N epochs with different shuffles to NVMe storage.
    Each epoch is stored as memory-mappable .npy files for fast access.

    Storage structure:
        storage_path/
            epoch_0/
                bags.npy      # (n_batches, batch_size, n_frags, feat_dim) fp16
                masks.npy     # (n_batches, batch_size, n_frags) bool
                labels.npy    # (n_batches, batch_size) int64
            epoch_1/
                ...
    """

    def __init__(
        self,
        storage_path: str,
        n_frags: int,
        feat_dim: int,
        batch_size: int,
        n_precollate_epochs: int = 5,
        n_workers: int = 8,
    ):
        """
        Args:
            storage_path: Path for NVMe storage
            n_frags: Fragments per bag
            feat_dim: Feature dimension
            batch_size: Batch size
            n_precollate_epochs: Number of epochs to pre-collate
            n_workers: Number of workers for parallel precollation
        """
        self.storage_path = Path(storage_path)
        self.n_frags = n_frags
        self.feat_dim = feat_dim
        self.batch_size = batch_size
        self.n_precollate_epochs = n_precollate_epochs
        self.n_workers = n_workers
        self.n_batches = None  # Set after precollation
        self.n_samples = None  # Set after precollation

    def precollate_epochs(
        self,
        bags: List[np.ndarray],
        labels: np.ndarray,
        sample_indices: List[int],
        seed: int = 42,
        show_progress: bool = True,
    ) -> None:
        """Pre-collate N epochs to NVMe storage.

        Args:
            bags: List of numpy arrays (all samples, even if not all used)
            labels: numpy array of labels (all samples)
            sample_indices: Which sample indices to use (e.g., train indices)
            seed: Random seed for shuffling
            show_progress: Whether to show progress bar
        """
        self.n_samples = len(sample_indices)
        self.n_batches = (self.n_samples + self.batch_size - 1) // self.batch_size

        # Cleanup old storage
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Calculate storage needs
        batch_size_gb = self.batch_size * self.n_frags * self.feat_dim * 2 / 1e9
        total_storage_gb = batch_size_gb * self.n_batches * self.n_precollate_epochs
        total_batches = self.n_batches * self.n_precollate_epochs

        if show_progress:
            print(f"    Pre-collating {self.n_precollate_epochs} epochs × {self.n_batches} batches = {total_batches} total batches")
            print(f"    Samples: {self.n_samples}, Storage needed: {total_storage_gb:.1f} GB")

        # Create labels array for workers (only the needed indices)
        labels_for_workers = np.array([labels[i] for i in sample_indices])

        # Create subset of bags for workers to access
        # Threads share memory directly (no pickling / no spawn/fork issues)
        bags_subset = [bags[i] for i in sample_indices]

        # Create progress queue for batch-level updates (thread-safe)
        progress_queue = queue.Queue() if show_progress else None

        # Worker arguments - pass bags_subset directly (threads share memory)
        worker_args = [
            (epoch_idx, bags_subset, labels_for_workers, self.n_frags,
             self.feat_dim, self.batch_size, self.n_batches,
             str(self.storage_path), seed, list(range(self.n_samples)), progress_queue)
            for epoch_idx in range(self.n_precollate_epochs)
        ]

        # Progress tracking thread
        pbar = None
        stop_progress = threading.Event()

        def progress_updater():
            """Background thread to update progress bar from queue."""
            nonlocal pbar
            while not stop_progress.is_set():
                try:
                    # Non-blocking get with timeout
                    progress_queue.get(timeout=0.1)
                    if pbar is not None:
                        pbar.update(1)
                except queue.Empty:
                    continue
                except Exception:
                    break

        # Start progress thread if needed
        progress_thread = None
        if show_progress and progress_queue is not None:
            pbar = tqdm(total=total_batches, desc="    Batches", leave=False,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
            progress_thread = threading.Thread(target=progress_updater, daemon=True)
            progress_thread.start()

        # Use threads to avoid pickling huge numpy arrays (fix for OOM-kill)
        try:
            with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                results = list(pool.map(_worker_precollate_epoch, worker_args))
        finally:
            # Stop progress thread
            if progress_thread is not None:
                stop_progress.set()
                progress_thread.join(timeout=2.0)
            if pbar is not None:
                pbar.close()

        if show_progress:
            total_size = sum(r[2] for r in results)
            print(f"    Done: {total_size:.1f} GB stored")

    def get_epoch_path(self, epoch_idx: int) -> Path:
        """Get path to epoch directory."""
        return self.storage_path / f"epoch_{epoch_idx}"

    def row_positions(self, epoch_idx: int = 0) -> np.ndarray:
        """Sample position of each of the first n_samples rows of an epoch, in stored batch order.
        Rows after that are wraparound repeats filling the last batch."""
        return np.load(self.get_epoch_path(epoch_idx) / "order.npy")[:self.n_samples]

    def get_n_batches(self) -> int:
        """Get number of batches per epoch."""
        return self.n_batches

    def cleanup(self) -> None:
        """Remove all pre-collated data."""
        if self.storage_path.exists():
            shutil.rmtree(self.storage_path)


class OptimizedBatchIterator:
    """Multi-reader + pinned memory + GPU prefetch iterator.

    Architecture:
        Reader threads (N): [mmap → pinned buffer] → CPU Queue
        Transfer thread:    [pinned → GPU]         → GPU Queue
        Main thread:        [GPU tensor → Compute]

    Performance: ~0.1s per batch (10x faster than legacy ~1.0s)

    Usage:
        iterator = OptimizedBatchIterator(storage, device, normalizer)
        for epoch in range(n_epochs):
            for bags, mask, labels in iterator:
                # bags is (B, T, D) float32, normalized, on GPU
                logits = model(bags, mask)
    """

    def __init__(
        self,
        storage: PrecollatedStorage,
        device: str,
        normalizer: torch.nn.Module,
        n_readers: int = 6,
        queue_depth: int = 3,
        return_sids: bool = False,
        n_buffers = None,
        shuffle: bool = True,
    ):
        """
        Args:
            storage: PrecollatedStorage instance with pre-collated data
            device: GPU device (e.g., "cuda")
            normalizer: FrozenStandardize1d instance for normalization
            n_readers: Number of reader threads (default: 6)
            queue_depth: Queue depth for prefetching (default: 3)
            return_sids: If True, return sample IDs (not supported in optimized mode)
            shuffle: If False, yield batches in stored order (needs n_readers=1)
        """
        if not shuffle and n_readers != 1:
            raise ValueError("shuffle=False needs n_readers=1 to keep batch order")
        self.storage = storage
        self.device = device
        self.normalizer = normalizer
        self.n_readers = n_readers
        self.queue_depth = queue_depth
        self.return_sids = return_sids
        self.shuffle = shuffle

        # Pre-allocate PINNED buffers
        n_buffers = queue_depth + 2 if n_buffers is None else n_buffers # Extra for safety
        self.n_buffers = n_buffers

        self.bags_buffers = [
            torch.empty(storage.batch_size, storage.n_frags, storage.feat_dim,
                        dtype=torch.float16, pin_memory=True)
            for _ in tqdm(range(n_buffers), leave=False)
        ]
        self.bags_buffers_np = [b.numpy() for b in self.bags_buffers]

        self.mask_buffers = [
            torch.empty(storage.batch_size, storage.n_frags,
                        dtype=torch.bool, pin_memory=True)
            for _ in range(n_buffers)
        ]
        self.mask_buffers_np = [m.numpy() for m in self.mask_buffers]

        # Track current epoch info
        self._current_epoch_idx = None
        self._current_batch_order = None

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Iterate over batches for one epoch.

        Yields:
            bags_gpu: (batch_size, n_frags, feat_dim) float32, normalized, on GPU
            mask_gpu: (batch_size, n_frags) bool, on GPU
            labels_gpu: (batch_size,) int64, on GPU
        """
        # 1. Select random pre-collated epoch
        self._current_epoch_idx = np.random.randint(0, self.storage.n_precollate_epochs)
        epoch_dir = self.storage.get_epoch_path(self._current_epoch_idx)

        # 2. Memory-map files
        bags_mmap = np.load(epoch_dir / "bags.npy", mmap_mode='r')
        masks_mmap = np.load(epoch_dir / "masks.npy", mmap_mode='r')
        labels_mmap = np.load(epoch_dir / "labels.npy", mmap_mode='r')

        # 3. Shuffle batch order (always drawn, so the global RNG stream doesn't depend on shuffle)
        n_batches = self.storage.n_batches
        perm = np.random.permutation(n_batches).tolist()
        self._current_batch_order = perm if self.shuffle else list(range(n_batches))

        # 4. Setup queues
        buffer_pool = queue.Queue()
        for i in range(self.n_buffers):
            buffer_pool.put(i)

        cpu_queue = queue.Queue(maxsize=self.queue_depth)
        gpu_queue = queue.Queue(maxsize=self.queue_depth)

        batch_queue = queue.Queue()
        for batch_idx in self._current_batch_order:
            batch_queue.put(batch_idx)

        stop_event = threading.Event()
        reader_errors = []
        readers_done = [0]
        readers_lock = threading.Lock()
        transfer_error = [None]

        # 5. Reader workers
        def reader_worker(worker_id: int):
            try:
                while not stop_event.is_set():
                    try:
                        batch_idx = batch_queue.get_nowait()
                    except queue.Empty:
                        break

                    buf_idx = buffer_pool.get()

                    # Read from mmap into pinned buffer
                    np.copyto(self.bags_buffers_np[buf_idx], bags_mmap[batch_idx])
                    np.copyto(self.mask_buffers_np[buf_idx], masks_mmap[batch_idx])
                    labels = labels_mmap[batch_idx].copy()

                    cpu_queue.put((buf_idx, labels, batch_idx))

            except Exception as e:
                reader_errors.append(e)
            finally:
                with readers_lock:
                    readers_done[0] += 1
                    if readers_done[0] == self.n_readers:
                        cpu_queue.put(None)  # Signal done

        # 6. Transfer worker
        def transfer_worker():
            try:
                while not stop_event.is_set():
                    item = cpu_queue.get()
                    if item is None:
                        gpu_queue.put(None)
                        break

                    buf_idx, labels, batch_idx = item

                    # Transfer pinned → GPU (fast DMA!)
                    bags_gpu = self.bags_buffers[buf_idx].to(self.device, non_blocking=True)
                    mask_gpu = self.mask_buffers[buf_idx].to(self.device, non_blocking=True)
                    labels_gpu = torch.from_numpy(labels).to(self.device)

                    # CRITICAL: Sync before recycling buffer
                    torch.cuda.synchronize()
                    buffer_pool.put(buf_idx)

                    # Normalize on GPU (in-place)
                    bags_gpu = self.normalizer(bags_gpu)

                    gpu_queue.put((bags_gpu, mask_gpu, labels_gpu))

            except Exception as e:
                transfer_error[0] = e
                gpu_queue.put(None)

        # 7. Start threads
        readers = [threading.Thread(target=reader_worker, args=(i,), daemon=True)
                   for i in range(self.n_readers)]
        for r in readers:
            r.start()

        transfer_thread = threading.Thread(target=transfer_worker, daemon=True)
        transfer_thread.start()

        # 8. Yield from GPU queue
        try:
            for _ in range(n_batches):
                item = gpu_queue.get()

                if item is None:
                    if transfer_error[0]:
                        raise transfer_error[0]
                    if reader_errors:
                        raise reader_errors[0]
                    break

                bags_gpu, mask_gpu, labels_gpu = item
                yield bags_gpu, mask_gpu, labels_gpu

                # Help GC
                del bags_gpu, mask_gpu, labels_gpu

        finally:
            # 9. Cleanup
            stop_event.set()
            for r in readers:
                r.join(timeout=2.0)
            transfer_thread.join(timeout=2.0)

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return self.storage.n_batches

    def get_epoch_info(self) -> Dict[str, Any]:
        """Get info about current/last epoch."""
        return {
            'precollate_epoch_idx': self._current_epoch_idx,
            'batch_order': self._current_batch_order,
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_optimized_iterator(
    bags: List[np.ndarray],
    labels: np.ndarray,
    sample_indices: List[int],
    normalizer: torch.nn.Module,
    device: str,
    storage_path: str,
    n_frags: int,
    feat_dim: int,
    batch_size: int,
    n_precollate_epochs: int = 5,
    n_readers: int = 6,
    queue_depth: int = 3,
    n_workers: int = 8,
    seed: int = 42,
    show_progress: bool = True,
) -> Tuple[OptimizedBatchIterator, PrecollatedStorage]:
    """Convenience function to create optimized iterator with storage.

    Args:
        bags: All bags (will be indexed by sample_indices)
        labels: All labels
        sample_indices: Which samples to use
        normalizer: FrozenStandardize1d for normalization
        device: GPU device
        storage_path: Path for NVMe storage
        n_frags: Fragments per bag
        feat_dim: Feature dimension
        batch_size: Batch size
        n_precollate_epochs: Number of epochs to pre-collate
        n_readers: Number of reader threads
        queue_depth: Queue depth
        n_workers: Workers for precollation
        seed: Random seed
        show_progress: Show progress bar

    Returns:
        (iterator, storage) - Remember to call storage.cleanup() when done!
    """
    storage = PrecollatedStorage(
        storage_path=storage_path,
        n_frags=n_frags,
        feat_dim=feat_dim,
        batch_size=batch_size,
        n_precollate_epochs=n_precollate_epochs,
        n_workers=n_workers,
    )

    storage.precollate_epochs(
        bags=bags,
        labels=labels,
        sample_indices=sample_indices,
        seed=seed,
        show_progress=show_progress,
    )

    iterator = OptimizedBatchIterator(
        storage=storage,
        device=device,
        normalizer=normalizer,
        n_readers=n_readers,
        queue_depth=queue_depth,
    )

    return iterator, storage


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    """Quick test of the optimized pipeline."""
    import time

    print("=" * 70)
    print("Testing Optimized Data Pipeline")
    print("=" * 70)

    # Test parameters
    n_samples = 20
    n_frags = 100000
    feat_dim = 3072
    batch_size = 4
    n_epochs = 3

    print(f"\nConfig:")
    print(f"  Samples: {n_samples}")
    print(f"  Frags/bag: {n_frags:,}")
    print(f"  Feat dim: {feat_dim}")
    print(f"  Batch size: {batch_size}")

    # Create fake data
    print("\nCreating synthetic data...")
    bags = [np.random.randn(n_frags, feat_dim).astype(np.float16) for _ in range(n_samples)]
    labels = np.random.randint(0, 2, n_samples)
    sample_indices = list(range(n_samples))

    # Create normalizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    class DummyNormalizer(torch.nn.Module):
        def forward(self, x):
            return x.float()

    normalizer = DummyNormalizer().to(device)

    # Create storage and iterator
    print("\nCreating optimized iterator...")
    storage_path = "/tmp/mil_test_storage"

    iterator, storage = create_optimized_iterator(
        bags=bags,
        labels=labels,
        sample_indices=sample_indices,
        normalizer=normalizer,
        device=device,
        storage_path=storage_path,
        n_frags=n_frags,
        feat_dim=feat_dim,
        batch_size=batch_size,
        n_precollate_epochs=3,
        n_readers=6,
        queue_depth=3,
    )

    # Test iteration
    print(f"\nTesting {n_epochs} epochs...")
    all_times = []

    for epoch in range(n_epochs):
        epoch_start = time.perf_counter()
        batch_times = []

        for bags_gpu, mask_gpu, labels_gpu in iterator:
            t = time.perf_counter()
            batch_times.append(t)

            # Simulate compute
            torch.cuda.synchronize()

        epoch_time = time.perf_counter() - epoch_start
        n_batches = len(batch_times)
        avg_batch = epoch_time / n_batches

        print(f"  Epoch {epoch+1}: {epoch_time:.2f}s total, {avg_batch:.4f}s/batch")
        all_times.extend([avg_batch] * n_batches)

    # Summary
    print(f"\nOverall: {np.mean(all_times[2:]):.4f}s ± {np.std(all_times[2:]):.4f}s per batch")

    # Cleanup
    storage.cleanup()
    print("\n✅ Test complete!")
