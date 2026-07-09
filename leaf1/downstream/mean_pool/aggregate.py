#!/usr/bin/env python3
"""Aggregate per-sample fragment embeddings into a single per-modality H5.

For each per-sample H5 produced by `leaf1-extract` (keys
`embeddings/bert_window_embeddings`, `metadata/*`, `progress/*`), compute the
mean across fragments to produce one row per sample. Output schema is the one
`mean_pool/train.py` expects: top-level `embeddings`, `sample_ids`, `groups`.

leaf1-extract writes single-layer embeddings as a flat 9984-d row. The
downstream `train.py` selects segments by their 13-segment index, so this
aggregator reshapes (n, 9984) -> (n, 13, 768) before averaging and writes
(N, 13, 768). If the input was already 3D (all-layers extraction), the chosen
layer is selected first.
"""

import argparse
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

EMBED_DIM = 768          # paper-canonical hidden dim
NUM_SEGMENTS = 13        # 3 windows * 3 regions + cut1 + cut2 + cls + mod
FLAT_DIM = NUM_SEGMENTS * EMBED_DIM  # 9984


def _reshape_to_segments(arr: np.ndarray, layer_idx: int = 4) -> np.ndarray:
    """Return (n, 13, 768) regardless of input layout.

    Recognised shapes:
        (n, 9984)           -> reshape to (n, 13, 768)
        (n, 13, 768)        -> as-is
        (n, num_layers, 9984) -> pick layer_idx, reshape to (n, 13, 768)
        (n, num_layers, 13, 768) -> pick layer_idx
    """
    if arr.ndim == 2 and arr.shape[1] == FLAT_DIM:
        return arr.reshape(arr.shape[0], NUM_SEGMENTS, EMBED_DIM)
    if arr.ndim == 3 and arr.shape[1] == NUM_SEGMENTS and arr.shape[2] == EMBED_DIM:
        return arr
    if arr.ndim == 3 and arr.shape[2] == FLAT_DIM:
        lyr = min(layer_idx, arr.shape[1] - 1)
        return arr[:, lyr, :].reshape(arr.shape[0], NUM_SEGMENTS, EMBED_DIM)
    if arr.ndim == 4 and arr.shape[2] == NUM_SEGMENTS and arr.shape[3] == EMBED_DIM:
        lyr = min(layer_idx, arr.shape[1] - 1)
        return arr[:, lyr, :, :]
    raise ValueError(
        f"Unrecognised embedding shape {arr.shape}. Expected (n, 9984), "
        f"(n, 13, 768), (n, L, 9984), or (n, L, 13, 768)."
    )


def process_one_sample(args):
    """Read one per-sample H5 (either leaf1-extract or layer4_reshaped schema)
    and return the (13, 768) mean-pooled embedding plus sample metadata."""
    h5_path, layer_idx = args
    try:
        with h5py.File(h5_path, "r") as f:
            # Completed-marker check: leaf1-extract uses progress/completed group attr,
            # layer4_reshaped uses a top-level `completed` attr. Either is acceptable.
            if "progress" in f and isinstance(f["progress"], h5py.Group):
                completed = f["progress"].attrs.get("completed", False)
            else:
                completed = f.attrs.get("completed", True)  # lab schema defaults to True
            if not completed:
                return None

            # Locate the embeddings dataset. Two recognized schemas:
            #   leaf1-extract:        embeddings/bert_window_embeddings
            #   layer4_reshaped/...:  embeddings  (top-level dataset, segment-major)
            if "embeddings/bert_window_embeddings" in f:
                embeddings = f["embeddings/bert_window_embeddings"][:]
            elif isinstance(f.get("embeddings"), h5py.Dataset):
                embeddings = f["embeddings"][:]
            else:
                raise KeyError(
                    f"{h5_path}: no embeddings dataset (looked for "
                    f"embeddings/bert_window_embeddings and top-level embeddings)"
                )

            sample_id = f["metadata"].attrs.get("sample_id", "unknown")
            group = f["metadata"].attrs.get("group", "unknown")
            modality = f["metadata"].attrs.get("modality", "unknown")

        seg3d = _reshape_to_segments(embeddings.astype(np.float32), layer_idx=layer_idx)
        mean_emb = seg3d.mean(axis=0)  # (13, 768)
        return {
            "sample_id": str(sample_id),
            "group": str(group),
            "modality": str(modality),
            "embedding": mean_emb,
        }
    except Exception as e:
        print(f"Error processing {h5_path}: {e}")
        return None


def aggregate(input_dir: str, output: str, model_name: str,
              workers: int = 8, layer_idx: int = 4) -> int:
    h5_dir = Path(input_dir) / "h5_files"
    if not h5_dir.is_dir():
        h5_dir = Path(input_dir)

    h5_files = sorted(h5_dir.glob("*.h5"))
    print(f"Found {len(h5_files)} H5 files in {h5_dir}")
    if not h5_files:
        return 0

    work_items = [(str(p), int(layer_idx)) for p in h5_files]
    with Pool(workers) as pool:
        results = pool.map(process_one_sample, work_items)
    results = [r for r in results if r is not None]
    print(f"Successfully aggregated {len(results)} / {len(h5_files)} samples")
    if not results:
        return 0

    sample_ids = [r["sample_id"] for r in results]
    groups = [r["group"] for r in results]
    embeddings = np.stack([r["embedding"] for r in results])  # (N, 13, 768)
    print(f"Aggregated shape: {embeddings.shape}")

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, dtype=np.float32, compression="gzip")
        f.create_dataset("sample_ids", data=[s.encode() for s in sample_ids])
        f.create_dataset("groups", data=[g.encode() for g in groups])

        f.attrs["model_name"] = model_name
        f.attrs["embed_dim"] = embeddings.shape[-1]
        f.attrs["num_segments"] = embeddings.shape[1]
        f.attrs["num_samples"] = len(results)
        f.attrs["pooling"] = "mean_across_fragments"
        f.attrs["layer_idx"] = int(layer_idx)
        f.attrs["timestamp"] = datetime.now().isoformat()

    print(f"Saved aggregated embeddings to {out_path}")
    return len(results)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-sample LEAF-1 fragment embeddings into one summary H5"
    )
    parser.add_argument("--input_dir", required=True,
                        help="Dir containing per-sample H5s (or a 'h5_files' subdir of it)")
    parser.add_argument("--output", required=True, help="Output aggregated H5 path")
    parser.add_argument("--model_name", required=True, help="Model name written to H5 metadata")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--layer_idx", type=int, default=4,
                        help="Layer to select when input is multi-layer (paper default: 4)")
    args = parser.parse_args()

    aggregate(args.input_dir, args.output, args.model_name,
              workers=args.workers, layer_idx=args.layer_idx)


if __name__ == "__main__":
    main()
