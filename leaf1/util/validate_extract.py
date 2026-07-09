"""Diff two per-sample H5s in the published lab schema.

Used to confirm `leaf1-extract` + `leaf1-reshape-layer` reproduces
reference `layer4_reshaped/CFDNA_EE*.h5` files bit-for-bit within fp16 ULP.

Both H5s are expected to have:
  embeddings              shape=(N, 13, 768) float16
  fragment_info/{chromosome, fragment_start, fragment_end, adjusted_start,
                  adjusted_end, line_number, n_token_left, n_token_frag,
                  n_token_right, n_freq, reverse_complemented}

Exit code:
  0 = match (max-abs-diff = 0 for embeddings; all fragment_info columns equal)
  1 = divergence (with a per-column report)
  2 = schema mismatch
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


FRAGMENT_INFO_INT_COLS = (
    "fragment_start", "fragment_end",
    "adjusted_start", "adjusted_end",
    "line_number",
    "n_token_left", "n_token_frag", "n_token_right",
)


def _load_pub(path: str):
    """Read both shape conventions:
       - top-level `embeddings` (lab's layer4_reshaped/ schema)
       - nested `embeddings/bert_window_embeddings` (raw leaf1-extract output;
         the caller is responsible for reshape-then-compare).
    """
    with h5py.File(path, "r") as f:
        if "embeddings" in f and isinstance(f["embeddings"], h5py.Dataset):
            emb = f["embeddings"][:]
        elif "embeddings/bert_window_embeddings" in f:
            ds = f["embeddings/bert_window_embeddings"]
            if ds.ndim == 2 and ds.shape[1] == 9984:
                emb = ds[:].reshape(ds.shape[0], 13, 768)
            elif ds.ndim == 3:
                emb = ds[:]
            else:
                raise ValueError(f"Unexpected embeddings shape {ds.shape}")
        else:
            raise ValueError(f"No embeddings dataset found in {path}")

        info = {}
        if "fragment_info" in f:
            for k in f["fragment_info"].keys():
                info[k] = f[f"fragment_info/{k}"][:]
    return emb, info


def diff(a_path: str, b_path: str, *,
          int_cols: tuple = FRAGMENT_INFO_INT_COLS,
          embedding_atol: float = 0.0) -> dict:
    """Return a report dict comparing two H5 files.

    `embedding_atol=0.0` enforces strict bitwise equality on the fp16 buffer.
    Set to something larger (e.g. 1e-3) when comparing across different fp paths.
    """
    a_emb, a_info = _load_pub(a_path)
    b_emb, b_info = _load_pub(b_path)

    report = {"path_a": a_path, "path_b": b_path, "errors": []}

    if a_emb.shape != b_emb.shape:
        report["errors"].append(
            f"Embedding shape mismatch: {a_emb.shape} vs {b_emb.shape}"
        )
        return report

    report["n_fragments"] = int(a_emb.shape[0])
    report["embedding_shape"] = tuple(int(x) for x in a_emb.shape)

    # Align by line_number - bag order in published H5s and our extract may differ
    # in row index if either side re-shuffles internally. Both pipelines sort
    # sampled indices in `np.sort(rng.choice(...))` so order should match; this
    # alignment is defensive.
    if "line_number" in a_info and "line_number" in b_info:
        a_ln = a_info["line_number"]
        b_ln = b_info["line_number"]
        if not np.array_equal(a_ln, b_ln):
            order_a = np.argsort(a_ln)
            order_b = np.argsort(b_ln)
            if np.array_equal(a_ln[order_a], b_ln[order_b]):
                a_emb = a_emb[order_a]
                b_emb = b_emb[order_b]
                for k in a_info:
                    if a_info[k].shape and a_info[k].shape[0] == len(order_a):
                        a_info[k] = a_info[k][order_a]
                for k in b_info:
                    if b_info[k].shape and b_info[k].shape[0] == len(order_b):
                        b_info[k] = b_info[k][order_b]
                report["realigned_by_line_number"] = True
            else:
                report["errors"].append(
                    "line_number sets differ between the two H5s; cannot align."
                )
                return report

    # Embedding diff.
    a_f32 = a_emb.astype(np.float32)
    b_f32 = b_emb.astype(np.float32)
    diffs = np.abs(a_f32 - b_f32)
    report["embedding_max_abs_diff"] = float(diffs.max())
    report["embedding_mean_abs_diff"] = float(diffs.mean())
    report["embedding_bitwise_equal"] = bool(np.array_equal(a_emb, b_emb))

    if not report["embedding_bitwise_equal"] and report["embedding_max_abs_diff"] > embedding_atol:
        report["errors"].append(
            f"Embedding diverged: max_abs={report['embedding_max_abs_diff']:.6g} "
            f"(tol {embedding_atol})"
        )

    # Per-column integer diffs.
    report["fragment_info"] = {}
    for col in int_cols:
        if col not in a_info or col not in b_info:
            continue
        n_diff = int((a_info[col] != b_info[col]).sum())
        report["fragment_info"][col] = n_diff
        if n_diff:
            report["errors"].append(f"{col}: {n_diff} disagreements")

    # Boolean reverse_complemented.
    if "reverse_complemented" in a_info and "reverse_complemented" in b_info:
        n_diff = int((a_info["reverse_complemented"] != b_info["reverse_complemented"]).sum())
        report["fragment_info"]["reverse_complemented"] = n_diff
        if n_diff:
            report["errors"].append(f"reverse_complemented: {n_diff} disagreements")

    # Chromosome strings.
    if "chromosome" in a_info and "chromosome" in b_info:
        a_chr = [s.decode() if isinstance(s, bytes) else s for s in a_info["chromosome"]]
        b_chr = [s.decode() if isinstance(s, bytes) else s for s in b_info["chromosome"]]
        n_diff = sum(1 for x, y in zip(a_chr, b_chr) if x != y)
        report["fragment_info"]["chromosome"] = n_diff
        if n_diff:
            report["errors"].append(f"chromosome: {n_diff} disagreements")

    # n_freq is float; compare with small tol.
    if "n_freq" in a_info and "n_freq" in b_info:
        ndiff = float(np.abs(a_info["n_freq"].astype(np.float64)
                             - b_info["n_freq"].astype(np.float64)).max())
        report["fragment_info"]["n_freq_max_abs_diff"] = ndiff
        if ndiff > 1e-6:
            report["errors"].append(f"n_freq: max-abs-diff = {ndiff:.6g}")

    return report


def main():
    p = argparse.ArgumentParser(
        description="Diff two per-sample H5s (validate leaf1-extract vs published)"
    )
    p.add_argument("--a", required=True, help="First H5 (e.g. our reshaped output)")
    p.add_argument("--b", required=True, help="Second H5 (e.g. published layer4_reshaped/CFDNA_*.h5)")
    p.add_argument("--embedding_atol", type=float, default=0.0,
                   help="Absolute tolerance on embedding diff (0 = strict bitwise)")
    args = p.parse_args()

    report = diff(args.a, args.b, embedding_atol=args.embedding_atol)

    print(f"Path A: {report['path_a']}")
    print(f"Path B: {report['path_b']}")
    if "n_fragments" in report:
        print(f"N fragments    : {report['n_fragments']}")
        print(f"Embedding shape: {report['embedding_shape']}")
        print(f"Bitwise equal  : {report['embedding_bitwise_equal']}")
        print(f"Max abs diff   : {report['embedding_max_abs_diff']:.6g}")
        print(f"Mean abs diff  : {report['embedding_mean_abs_diff']:.6g}")
    if "fragment_info" in report:
        for k, v in report["fragment_info"].items():
            print(f"  fragment_info.{k}: {v}")

    if report["errors"]:
        print()
        print("DIVERGENCE:")
        for e in report["errors"]:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print()
        print("MATCH.")
        sys.exit(0)


if __name__ == "__main__":
    main()
