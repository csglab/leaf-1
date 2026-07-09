"""Reshape a leaf1-extract H5 into the lab's published per-sample format.

Input: a single-layer H5 from `leaf1-extract --target_layer K`, with
`embeddings/bert_window_embeddings` shape (N, 9984), OR an all-layers H5 with
`embeddings/bert_window_embeddings` shape (N, 13, 9984).

Output (matches the published `layer4_reshaped/CFDNA_EE*.h5` schema):
  embeddings              (N, 13, 768) float16   # segment-major
  segment_names           (13,) S20              # left_window_0 ... mod
  segment_slices          (13, 2) int32          # [start, end) in flat 9984 layout
  fragment_info/          (copied from input verbatim)
  metadata/               (copied from input verbatim)
  top-level attrs:
    completed = True
    layer_extracted = <K>
    creation_time = ISO timestamp
    source_file = <input filename>

This is a port of the lab's `pool_layer_store.py` so leaf1-extract output can be
compared bit-for-bit against the published H5s.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

SEGMENT_NAMES = (
    "left_window_0", "left_window_1", "left_window_2",
    "fragment_window_0", "fragment_window_1", "fragment_window_2",
    "right_window_0", "right_window_1", "right_window_2",
    "cut1", "cut2", "cls", "mod",
)
NUM_SEGMENTS = 13
EMBED_DIM = 768
FLAT_DIM = NUM_SEGMENTS * EMBED_DIM  # 9984


def _segment_slices() -> np.ndarray:
    """[start, end) byte ranges of each 768-dim segment in the flat 9984 layout."""
    starts = np.arange(0, FLAT_DIM, EMBED_DIM, dtype=np.int32)
    ends = starts + EMBED_DIM
    return np.column_stack([starts, ends])


def reshape(src_path: str, dst_path: str, layer: int = 4) -> None:
    src = Path(src_path)
    dst = Path(dst_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source H5 not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        ds = fin["embeddings/bert_window_embeddings"]
        if ds.ndim == 2 and ds.shape[1] == FLAT_DIM:
            flat = ds[:].astype(np.float16)
        elif ds.ndim == 3 and ds.shape[2] == FLAT_DIM:
            # All-layers extraction: (N, n_layers, 9984) -> pick `layer`.
            if layer >= ds.shape[1]:
                raise ValueError(
                    f"--layer={layer} out of range; H5 has {ds.shape[1]} layers."
                )
            flat = ds[:, layer, :].astype(np.float16)
        else:
            raise ValueError(
                f"Unexpected embeddings shape {ds.shape}; expected (N, 9984) "
                f"or (N, num_layers, 9984)."
            )

        n = flat.shape[0]
        segmented = flat.reshape(n, NUM_SEGMENTS, EMBED_DIM)

        fout.create_dataset(
            "embeddings", data=segmented, dtype=np.float16,
            compression="gzip", compression_opts=4,
        )
        fout.create_dataset(
            "segment_names",
            data=np.asarray(SEGMENT_NAMES, dtype="S20"),
        )
        fout.create_dataset(
            "segment_slices", data=_segment_slices(), dtype=np.int32,
        )

        # Copy fragment_info and metadata groups verbatim.
        for g in ("fragment_info", "metadata"):
            if g in fin:
                fin.copy(g, fout)

        # Copy top-level attrs that existed on the source.
        for k, v in fin.attrs.items():
            fout.attrs[k] = v

        # Stamp the post-process metadata (matches lab's pool_layer_store.py).
        fout.attrs["layer_extracted"] = int(layer)
        fout.attrs["creation_time"] = datetime.utcnow().isoformat()
        fout.attrs["completed"] = True
        fout.attrs["source_file"] = src.name

    print(f"Wrote {dst}  shape=({n}, {NUM_SEGMENTS}, {EMBED_DIM})", flush=True)


def main():
    p = argparse.ArgumentParser(
        description="Reshape leaf1-extract H5 into the lab's published per-sample format"
    )
    p.add_argument("--input", required=True, help="Source H5 from leaf1-extract")
    p.add_argument("--output", required=True, help="Destination H5 path")
    p.add_argument("--layer", type=int, default=4,
                   help="Layer to pick (only used when input is all-layers; default 4)")
    args = p.parse_args()
    reshape(args.input, args.output, layer=args.layer)


if __name__ == "__main__":
    main()
