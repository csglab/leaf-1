"""Sample-level inference for the gated-attention MIL classifier.

Loads a *fitted* MIL ensemble (the ``inner_fold_*/`` checkpoints written by
``leaf1-mil``) and scores new samples' per-fragment embedding H5s, emitting one
cancer probability per sample. This is the counterpart to training: where
``leaf1-mil`` fits + saves, ``leaf1-mil-predict`` loads + scores.

CLI: ``leaf1-mil-predict``.

Input H5 layout
---------------
The flat ``leaf1-extract`` schema, i.e. a dataset at
``embeddings/bert_window_embeddings`` of shape ``(n_fragments, 9984)`` (float16),
plus a ``metadata`` group whose ``group`` / ``sample_id`` attrs (when present)
are carried through to the output. Segments are selected from the flat 9984-d
row with the same :func:`leaf1.downstream.mil.data.get_indices` resolver the
trainer uses, so inference cannot diverge from how the model was fit.

Checkpoint layout
-----------------
A fitted model is an *ensemble* of inner-fold checkpoints::

    <checkpoint_dir>/
      inner_fold_0/{checkpoint.pt, norm_stats.pt}
      inner_fold_1/...
      ...

``checkpoint.pt`` holds ``{proj,pool,head}_state_dict`` + a ``config`` dict;
``norm_stats.pt`` holds the ``{mean, std}`` standardization vectors. Point
``--mil_checkpoint`` at a single outer-fold dir (its ``inner_fold_*`` are
averaged), or at a task dir containing ``F0..F4/`` with ``--all_folds`` to
average every inner fold across every outer fold.
"""

import argparse
import glob
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import get_indices

FLAT_EMBED_KEY = "embeddings/bert_window_embeddings"


# =============================================================================
# Model components (reference paper-inference variants)
#
# These mirror the trainer's modules but are kept self-contained here so the
# loader is faithful to the checkpoints produced by the original MIL trainer
# (fp32 standardization, no memory-optimized in-place ops). State-dict key names
# match leaf1-mil output too, so both checkpoint provenances load unchanged.
# =============================================================================

class FeatureProjector(nn.Module):
    def __init__(self, in_dim, out_dim=None, mode="none", use_ln=False):
        super().__init__()
        if mode == "none":
            self.linear = None
            self.post_linear = nn.Identity()
            self.out_dim = in_dim
        else:
            self.out_dim = out_dim or in_dim
            self.linear = nn.Linear(in_dim, self.out_dim)
            layers = [nn.LayerNorm(self.out_dim)] if use_ln else []
            self.post_linear = nn.Sequential(*layers) if layers else nn.Identity()

    def forward(self, x):
        if self.linear is None:
            return x
        return self.post_linear(self.linear(x))


class GatedAttentionPool(nn.Module):
    def __init__(self, feat_dim, att_dim=64, temperature=1.0):
        super().__init__()
        self.V = nn.Linear(feat_dim, att_dim)
        self.U = nn.Linear(feat_dim, att_dim)
        self.w = nn.Linear(att_dim, 1)
        self.temperature = temperature

    def forward(self, bags, mask=None, attention_fix="hhi"):
        gated = torch.tanh(self.V(bags)) * torch.sigmoid(self.U(bags))
        scores = self.w(gated).squeeze(-1)
        if self.temperature != 1.0:
            scores = scores / self.temperature
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        W = F.softmax(scores, dim=1)
        pooled = (bags * W.unsqueeze(-1)).sum(dim=1)
        if attention_fix == "hhi":
            g_W = (1.0 / (W ** 2).sum(dim=1, keepdim=True).clamp_min(1e-8)).sqrt()
        elif attention_fix == "entropy":
            W_safe = W.clamp_min(1e-8)
            g_W = (-(W_safe * W_safe.log()).sum(dim=1, keepdim=True)).exp().sqrt()
        else:
            g_W = torch.ones(bags.shape[0], 1, device=bags.device)
        return pooled * g_W


class FrozenStandardize1d(nn.Module):
    def __init__(self, mean, std, eps=1e-8):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(eps))

    def forward(self, x):
        return (x - self.mean) / self.std


# =============================================================================
# Checkpoint loading
# =============================================================================

def _detect_use_ln(proj_state_dict):
    """The saved ``config`` omits ``feat_proj_ln``; recover it from the weights."""
    return any(k.startswith("post_linear") for k in proj_state_dict)


def load_checkpoint(fold_dir, device="cpu"):
    """Load one inner-fold model (proj + pool + head + standardizer)."""
    fold_dir = Path(fold_dir)
    ckpt = torch.load(fold_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = ckpt["config"]
    feat_dim = config["feat_dim"]

    if config.get("feat_proj", "none") == "none":
        proj = FeatureProjector(feat_dim, mode="none")
        proj_dim = feat_dim
    else:
        proj = FeatureProjector(
            feat_dim,
            config["feat_proj_dim"],
            mode=config["feat_proj"],
            use_ln=_detect_use_ln(ckpt["proj_state_dict"]),
        )
        proj_dim = proj.out_dim

    pool = GatedAttentionPool(proj_dim, att_dim=64, temperature=config.get("att_temperature", 1.0))
    head = nn.Linear(proj_dim, 2)

    proj.load_state_dict(ckpt["proj_state_dict"])
    pool.load_state_dict(ckpt["pool_state_dict"])
    head.load_state_dict(ckpt["head_state_dict"])

    norm = torch.load(fold_dir / "norm_stats.pt", map_location="cpu", weights_only=False)
    normalizer = FrozenStandardize1d(
        torch.as_tensor(norm["mean"]).float(),
        torch.as_tensor(norm["std"]).float(),
    )

    return {
        "proj": proj.to(device).eval(),
        "pool": pool.to(device).eval(),
        "head": head.to(device).eval(),
        "norm": normalizer.to(device),
        "att_fix": config.get("att_fix", "hhi"),
        "feat_dim": feat_dim,
    }


def _inner_folds_under(d):
    """``inner_fold_*`` directly under ``d``, or under one intermediate dir.

    Handles both the collapsed release layout (``F0/inner_fold_*``) and the raw
    sweep layout (``F0/<best_config>/inner_fold_*``).
    """
    inner = sorted(d.glob("inner_fold_*"))
    return inner if inner else sorted(d.glob("*/inner_fold_*"))


def discover_fold_dirs(checkpoint_dir, all_folds=False):
    """Resolve ``--mil_checkpoint`` to the list of ``inner_fold_*`` dirs to ensemble.

    - Task dir containing ``F0..Fk/``: with ``all_folds`` gather every fold's
      inner checkpoints (full nested-CV ensemble); otherwise use ``F0`` only.
    - Otherwise ``checkpoint_dir`` is itself one outer fold, holding
      ``inner_fold_*`` directly or under a single config dir.
    """
    checkpoint_dir = Path(checkpoint_dir)
    outer = sorted(checkpoint_dir.glob("F*"))
    if outer:
        chosen = outer if all_folds else outer[:1]
        dirs = [d for f in chosen for d in _inner_folds_under(f)]
        if dirs:
            return dirs

    inner = _inner_folds_under(checkpoint_dir)
    if inner:
        return inner
    raise ValueError(
        f"No inner_fold_* checkpoints found under {checkpoint_dir} "
        f"(looked under F*/, one config dir, and directly)."
    )


def resolve_segments(checkpoint_dir, fold_dirs, override=None):
    """Determine the segment string the checkpoints were trained with.

    Priority: explicit ``override`` -> ``config.json`` bundled next to the folds
    (its ``data_config.segments``) -> raise. Keeping segments tied to the
    checkpoint prevents a silent feat_dim / column-order mismatch.
    """
    if override:
        return override
    # config.json sits at the outer-fold dir (parent of inner_fold_*), or the
    # checkpoint dir itself. Search both.
    candidates = [Path(checkpoint_dir)]
    candidates += [d.parent for d in fold_dirs]
    for base in candidates:
        cfg_path = base / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            seg = cfg.get("data_config", {}).get("segments")
            if seg:
                return seg
    raise ValueError(
        "Could not determine segments: no --segments given and no config.json "
        f"with data_config.segments found near {checkpoint_dir}."
    )


def load_ensemble(checkpoint_dir, all_folds=False, segments=None, device="cpu"):
    """Load every inner-fold model + resolve the shared segment column indices."""
    fold_dirs = discover_fold_dirs(checkpoint_dir, all_folds=all_folds)
    models = [load_checkpoint(d, device) for d in fold_dirs]

    seg_str = resolve_segments(checkpoint_dir, fold_dirs, override=segments)
    seg_idx = get_indices(seg_str)
    feat_dim = models[0]["feat_dim"]
    if len(seg_idx) != feat_dim:
        raise ValueError(
            f"Segment/feature mismatch: segments '{seg_str}' -> {len(seg_idx)} "
            f"columns, but checkpoint expects feat_dim={feat_dim}. The checkpoint "
            f"was trained on a different segment set."
        )
    return models, seg_idx, seg_str


# =============================================================================
# Bag loading + scoring
# =============================================================================

def _contiguous_runs(idx):
    """Group a sorted ascending index array into ``(start, stop)`` half-open runs."""
    idx = np.asarray(idx)
    breaks = np.where(np.diff(idx) != 1)[0] + 1
    return [(int(g[0]), int(g[-1]) + 1) for g in np.split(idx, breaks)]


def load_bag(h5, seg_idx, n_frags, seed=42):
    """Sample up to ``n_frags`` fragments and select the trained segment columns.

    Returns a ``(T, feat_dim)`` float32 bag from the flat ``(N, 9984)`` layout.
    Only the selected columns are read from disk (as contiguous slices), so a
    100k-fragment bag pulls ~feat_dim/9984 of the row bytes rather than the whole
    row.
    """
    ds = h5[FLAT_EMBED_KEY]
    n_available = ds.shape[0]
    if n_available > n_frags:
        rng = np.random.RandomState(seed)
        rows = np.sort(rng.choice(n_available, n_frags, replace=False))
    else:
        rows = slice(None)
    cols = [ds[rows, a:b] for a, b in _contiguous_runs(seg_idx)]  # each (T, width) fp16
    return np.concatenate(cols, axis=1).astype(np.float32)


@torch.no_grad()
def score_sample(bag, models, device="cpu", chunk=20000):
    """Ensemble-average the cancer probability + logit over the fold models.

    Fragments are projected in chunks of ``chunk`` so the full-precision bag is
    never materialized twice: only the projected ``z`` (T × proj_dim) is held per
    model. Result is identical to a single-shot forward (all ops are per-fragment
    up to the attention softmax); chunking just bounds peak memory, which matters
    for 100k-fragment bags on CPU. ``bag`` stays as-is (``from_numpy`` shares it);
    each chunk is upcast on the fly.
    """
    bag_t = torch.from_numpy(bag)  # shares the numpy buffer (no copy)
    T = bag_t.shape[0]
    probs, logits = [], []
    for m in models:
        z_parts = []
        for i in range(0, T, chunk):
            xc = bag_t[i:i + chunk].float().to(device)
            z_parts.append(m["proj"](m["norm"](xc)))
        z = torch.cat(z_parts, dim=0).unsqueeze(0)      # (1, T, proj_dim)
        pooled = m["pool"](z, attention_fix=m["att_fix"])
        out = m["head"](pooled)                          # (1, 2)
        logits.append((out[0, 1] - out[0, 0]).item())
        probs.append(F.softmax(out, dim=1)[0, 1].item())
    return {
        "pred_prob": float(np.mean(probs)),
        "pred_logit": float(np.mean(logits)),
        "n_models": len(models),
    }


def _read_group_attrs(h5):
    """Pull sample_id / group labels from the H5 metadata group when present."""
    out = {}
    if "metadata" in h5:
        for key in ("sample_id", "group", "modality"):
            if key in h5["metadata"].attrs:
                val = h5["metadata"].attrs[key]
                out[key] = val.decode() if isinstance(val, bytes) else str(val)
    return out


def run_inference(checkpoint_dir, hdf5_files, output_dir, *, all_folds=False,
                  segments=None, frags_per_bag=100000, seed=42, device=None):
    """Score every H5 in ``hdf5_files`` and write ``sample_summary.csv``."""
    import csv

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models, seg_idx, seg_str = load_ensemble(
        checkpoint_dir, all_folds=all_folds, segments=segments, device=device
    )
    print(f"Loaded {len(models)} fold model(s); segments='{seg_str}' -> {len(seg_idx)} features; device={device}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, path in enumerate(hdf5_files):
        with h5py.File(path, "r") as h5:
            if FLAT_EMBED_KEY not in h5:
                raise ValueError(
                    f"{path}: no '{FLAT_EMBED_KEY}' dataset (expected the flat "
                    f"leaf1-extract layout)."
                )
            attrs = _read_group_attrs(h5)
            bag = load_bag(h5, seg_idx, frags_per_bag, seed=seed)
            result = score_sample(bag, models, device=device)
        row = {
            "sample_id": attrs.get("sample_id", Path(path).stem),
            "group": attrs.get("group", ""),
            "n_fragments": bag.shape[0],
            "pred_prob": result["pred_prob"],
            "pred_logit": result["pred_logit"],
            "n_models": result["n_models"],
        }
        rows.append(row)
        print(f"  [{i + 1}/{len(hdf5_files)}] {row['sample_id']:<40s} "
              f"group={row['group'] or '?':<14s} pred_prob={row['pred_prob']:.4f}")

    summary_path = output_dir / "sample_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    config_path = output_dir / "inference_config.json"
    with open(config_path, "w") as f:
        json.dump({
            "checkpoint_dir": str(checkpoint_dir),
            "all_folds": all_folds,
            "n_models": len(models),
            "segments": seg_str,
            "feat_dim": int(len(seg_idx)),
            "frags_per_bag": frags_per_bag,
            "seed": seed,
            "n_samples": len(rows),
        }, f, indent=2)

    print(f"\nWrote {summary_path} ({len(rows)} samples) and {config_path}")
    return rows


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="leaf1-mil-predict",
        description="Score samples with a fitted gated-attention MIL ensemble "
                    "(sample-level cancer probability).",
    )
    p.add_argument("--mil_checkpoint", required=True,
                   help="Fitted MIL checkpoint dir: an outer-fold dir with "
                        "inner_fold_*/, or a task dir with F0..Fk/ (use --all_folds).")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdf5_file", help="Single per-sample H5 (flat leaf1-extract layout).")
    src.add_argument("--hdf5_pattern", help="Glob for many H5s, e.g. '/path/CFDNA_*.h5'.")
    p.add_argument("--output_dir", required=True, help="Where to write sample_summary.csv.")
    p.add_argument("--all_folds", action="store_true",
                   help="Ensemble every inner fold across all outer folds F*/ "
                        "(full nested-CV model). Use for external cohorts.")
    p.add_argument("--segments", default=None,
                   help="Override segment set. Default: read from the checkpoint's "
                        "config.json (data_config.segments).")
    p.add_argument("--frags_per_bag", type=int, default=100000,
                   help="Fragments sampled per sample (default: 100000).")
    p.add_argument("--seed", type=int, default=42, help="Fragment-sampling seed.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.hdf5_file:
        hdf5_files = [args.hdf5_file]
    else:
        hdf5_files = sorted(glob.glob(args.hdf5_pattern))
        if not hdf5_files:
            raise SystemExit(f"No files match pattern: {args.hdf5_pattern}")
    run_inference(
        args.mil_checkpoint, hdf5_files, args.output_dir,
        all_folds=args.all_folds, segments=args.segments,
        frags_per_bag=args.frags_per_bag, seed=args.seed,
    )


if __name__ == "__main__":
    main()
