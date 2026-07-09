#!/usr/bin/env python3
"""Benchmark DNA embedding models on scATAC cell type classification.

Handles two H5 formats automatically:
  - External models:       (N, embed_dim)  - use full embedding
  - Internal-style models: (N, 13, dim)    - select segments, flatten
"""
from __future__ import annotations
import argparse, contextlib, json, os, warnings
from datetime import datetime
from pathlib import Path

import h5py, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── optional libs ─────────────────────────────────────────────────────────────
_OPT = {"xgb": ("xgboost", "XGBClassifier"), "lgb": ("lightgbm", "LGBMClassifier")}
_LOADED: dict = {}
for _k, (_mod, _cls) in _OPT.items():
    try:
        _LOADED[_k] = getattr(__import__(_mod), _cls)
    except ImportError:
        pass

NEEDS_SCALING = {"lr", "svm"}

# ── internal model segment map ────────────────────────────────────────────────
_SEG_NAMES = [
    "left_window_0", "left_window_1", "left_window_2",
    "fragment_window_0", "fragment_window_1", "fragment_window_2",
    "right_window_0", "right_window_1", "right_window_2",
    "cut1", "cut2", "cls", "mod",
]
_SEG_ALIAS: dict[str, int | list[int]] = {
    "left0": 0, "left1": 1, "left2": 2,
    "fragment0": 3, "fragment1": 4, "fragment2": 5,
    "right0": 6, "right1": 7, "right2": 8,
    "cut1": 9, "cut2": 10, "cls": 11, "mod": 12,
    "left": [0, 1, 2], "fragment": [3, 4, 5], "right": [6, 7, 8],
    "cut": [9, 10], "all": list(range(13)),
}


def parse_segments(seg_str: str) -> list[int]:
    out, seen = [], set()
    for part in seg_str.lower().split(","):
        part = part.strip()
        val = _SEG_ALIAS.get(part)
        if val is None:
            raise ValueError(f"Unknown segment '{part}'. Valid: {sorted(_SEG_ALIAS)}")
        for i in (val if isinstance(val, list) else [val]):
            if i not in seen:
                out.append(i)
                seen.add(i)
    return out


# ── classifier definitions ────────────────────────────────────────────────────
_BASE = {
    "lr":  LogisticRegression(random_state=42, class_weight="balanced",
                               max_iter=10_000, solver="saga", tol=0.01),
    "rf":  RandomForestClassifier(random_state=42, n_jobs=1,
                                   class_weight="balanced"),
    "svm": SVC(random_state=42, probability=True, class_weight="balanced",
                cache_size=2000),
    **{k: v for k, v in _LOADED.items()},
}
_GRIDS = {
    "lr": {
        "fast": {"clf__C": [1e-3, 1e-2, .1, 1, 10, 100],
                 "clf__penalty": ["elasticnet"], "clf__l1_ratio": [0, .5, 1]},
        "full": {"clf__C": [1e-4, 1e-3, 1e-2, .1, 1, 10, 100, 1000],
                 "clf__penalty": ["elasticnet"],
                 "clf__l1_ratio": [0, .25, .5, .75, 1]},
    },
    "rf": {
        "fast": {"clf__n_estimators": [100, 200],
                 "clf__max_depth": [10, 20, None],
                 "clf__min_samples_split": [2, 5]},
        "full": {"clf__n_estimators": [100, 200, 300],
                 "clf__max_depth": [10, 20, None],
                 "clf__min_samples_split": [2, 5, 10],
                 "clf__max_features": ["sqrt", "log2"]},
    },
    "svm": {
        "fast": {"clf__C": [1e-3, 1e-2, .1, 1, 10],
                 "clf__kernel": ["linear"]},
        "full": {"clf__C": [1e-3, 1e-2, .1, 1, 10, 100],
                 "clf__kernel": ["linear", "rbf"],
                 "clf__gamma": ["scale", "auto"]},
    },
    "xgb": {
        "fast": {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5],
                 "clf__learning_rate": [.01, .1]},
        "full": {"clf__n_estimators": [100, 200, 300],
                 "clf__max_depth": [3, 5, 7],
                 "clf__learning_rate": [.01, .05, .1],
                 "clf__subsample": [.8, 1]},
    },
    "lgb": {
        "fast": {"clf__n_estimators": [100, 200], "clf__max_depth": [5, 10],
                 "clf__learning_rate": [.01, .1], "clf__num_leaves": [31, 63]},
        "full": {"clf__n_estimators": [100, 200, 300],
                 "clf__max_depth": [5, 10, -1],
                 "clf__learning_rate": [.01, .05, .1],
                 "clf__num_leaves": [31, 63, 127]},
    },
}


# ── data loading ──────────────────────────────────────────────────────────────
def load_data(
    h5_path: str,
    meta_csv: str,
    segments: str,
    target: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, LabelEncoder | None]:
    """Load embeddings and labels.

    Returns (X, y, meta, label_encoder).
    For multi-class: y is integer-encoded, label_encoder maps back to names.
    For binary (_vs_all): y is 0/1, label_encoder is None.
    """
    meta = pd.read_csv(meta_csv)
    meta = meta[meta["group"] != "Unknown"].reset_index(drop=True)

    if target == "cell_type":
        # Multi-class: all cell types
        le = LabelEncoder()
        y = le.fit_transform(meta["group"].values)
        mode = "multi"
        print(f"Multi-class: {len(le.classes_)} cell types, {len(y)} cells")
        for i, ct in enumerate(le.classes_):
            print(f"  {i}: {ct} (n={(y==i).sum()})")
    elif target.endswith("_vs_all"):
        # Binary one-vs-rest
        cell_type = target[:-len("_vs_all")]
        y = (meta["group"] == cell_type).astype(int).values
        le = None
        mode = "binary"
        print(f"Binary: {cell_type} vs all → pos={y.sum()} neg={(y==0).sum()}")
    else:
        raise ValueError(f"Unknown target: {target}. Use 'cell_type' or '{{CellType}}_vs_all'")

    # Match metadata rows to H5 rows by sample_id
    with h5py.File(h5_path, "r") as f:
        emb = f["embeddings"][:]
        if "sample_ids" in f:
            h5_ids = [s.decode() if isinstance(s, bytes) else s for s in f["sample_ids"][:]]
        else:
            h5_ids = None

    if h5_ids is not None:
        # Build sample_id → h5 row index mapping
        id_to_h5idx = {sid: i for i, sid in enumerate(h5_ids)}
        idx = []
        keep_mask = []
        for i, sid in enumerate(meta["sample_id"]):
            if sid in id_to_h5idx:
                idx.append(id_to_h5idx[sid])
                keep_mask.append(True)
            else:
                keep_mask.append(False)
        idx = np.array(idx)
        if not all(keep_mask):
            n_dropped = sum(1 for x in keep_mask if not x)
            print(f"Warning: {n_dropped} metadata rows not found in H5, dropping them")
            meta = meta[keep_mask].reset_index(drop=True)
            y = y[np.array(keep_mask)]
            if le is not None:
                # Re-encode in case classes changed
                pass
        print(f"Matched {len(idx)} samples by sample_id")
    elif "h5_index" in meta.columns:
        meta = meta.sort_values("h5_index").reset_index(drop=True)
        idx = meta["h5_index"].astype(int).values
    else:
        idx = np.arange(len(meta))

    shape = emb.shape
    if len(shape) == 2:
        X = emb[idx].astype(np.float32)
        print(f"External embedding: shape={X.shape}")
    elif len(shape) == 3:
        # Check for _mean suffix: average segments instead of concatenating
        if segments.endswith("_mean"):
            seg_str = segments[:-len("_mean")]
            seg_idx = parse_segments(seg_str)
            X = emb[idx][:, seg_idx, :].astype(np.float32).mean(axis=1)  # (N, 768)
            seg_names = [_SEG_NAMES[i] for i in seg_idx]
            print(f"Internal embedding (mean-pooled): segments={seg_names} → X={X.shape}")
        else:
            seg_idx = parse_segments(segments)
            X = emb[idx][:, seg_idx, :].reshape(len(idx), -1).astype(np.float32)
            seg_names = [_SEG_NAMES[i] for i in seg_idx]
            print(f"Internal embedding: segments={seg_names} → X={X.shape}")
    else:
        raise ValueError(f"Unexpected embedding shape: {shape}")

    return X, y, meta, le


# ── nested CV ─────────────────────────────────────────────────────────────────
def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    clf_name: str,
    fast: bool,
    inner_k: int,
    outer_k: int,
    n_jobs: int,
    pca: int | None = None,
    multiclass: bool = False,
) -> dict:

    mdl = _BASE[clf_name]
    if clf_name in ("xgb", "lgb"):
        mdl.set_params(n_jobs=1)
    grid = _GRIDS[clf_name]["fast" if fast else "full"].copy()
    if pca:
        grid["pca__n_components"] = [min(pca, n) for n in [50, 200, 500] if n <= pca]

    scoring = "roc_auc_ovr" if multiclass else "roc_auc"

    outer_jobs = min(n_jobs, outer_k)
    inner_jobs = max(1, n_jobs // outer_jobs)
    print(f"Jobs: {outer_jobs} outer × {inner_jobs} inner")

    n_classes = len(np.unique(y)) if multiclass else 2

    def _fit(tr, te):
        steps = []
        if clf_name in NEEDS_SCALING:
            steps.append(("scaler", StandardScaler()))
        if pca:
            steps.append(("pca", PCA(random_state=42)))
        steps.append(("clf", mdl))
        pipe = Pipeline(steps)
        gs = GridSearchCV(
            pipe, grid,
            cv=StratifiedKFold(inner_k, shuffle=True, random_state=42),
            scoring=scoring, n_jobs=inner_jobs, error_score=np.nan,
        )
        with contextlib.redirect_stdout(None), contextlib.redirect_stderr(None):
            gs.fit(X[tr], y[tr])
        proba = gs.best_estimator_.predict_proba(X[te])  # (n_test, n_classes)
        pred = gs.best_estimator_.predict(X[te])
        return y[te], proba, pred, te

    splitter = StratifiedKFold(outer_k, shuffle=True, random_state=42).split(X, y)

    raw = list(tqdm(
        Parallel(n_jobs=outer_jobs, return_as="generator")(
            delayed(_fit)(tr, te) for tr, te in splitter
        ),
        total=outer_k,
        desc=f"{outer_k}-fold",
        colour="cyan",
    ))

    yt_all, yp_all, pred_all, te_all = [], [], [], []
    for yt_chunk, yp_chunk, pred_chunk, te_chunk in raw:
        yt_all.append(yt_chunk)
        yp_all.append(yp_chunk)
        pred_all.append(pred_chunk)
        te_all.append(te_chunk)

    te_cat = np.concatenate(te_all)
    order = np.argsort(te_cat)

    yt = np.concatenate(yt_all)[order]
    yp = np.vstack(yp_all)[order]
    pred = np.concatenate(pred_all)[order]

    if multiclass:
        macro_f1 = f1_score(yt, pred, average="macro")
        bal_acc = balanced_accuracy_score(yt, pred)
        cm = confusion_matrix(yt, pred)
        # Per-class AUC (one-vs-rest). sklearn treats two classes as binary even
        # when multi_class="ovr" is supplied, so compute the binary edge case from
        # the same per-class quantities used for larger multiclass problems.
        per_class_auc = {}
        for c in range(n_classes):
            binary_yt = (yt == c).astype(int)
            if binary_yt.sum() > 0 and binary_yt.sum() < len(binary_yt):
                per_class_auc[c] = roc_auc_score(binary_yt, yp[:, c])
        if n_classes == 2:
            macro_auc = float(np.mean(list(per_class_auc.values())))
        else:
            macro_auc = roc_auc_score(yt, yp, multi_class="ovr", average="macro")
        return {
            "y_true": yt, "y_prob": yp, "y_pred": pred,
            "macro_auc": macro_auc,
            "macro_f1": macro_f1,
            "balanced_acc": bal_acc,
            "cm": cm,
            "per_class_auc": per_class_auc,
        }
    else:
        # Binary
        yp_pos = yp[:, 1] if yp.ndim == 2 else yp
        pr, rc, th = precision_recall_curve(yt, yp_pos)
        f1s = 2 * pr * rc / (pr + rc + 1e-10)
        best = np.argmax(f1s[:-1])
        pred_binary = (yp_pos >= th[best]).astype(int)
        return {
            "y_true": yt, "y_prob": yp_pos, "y_pred": pred_binary,
            "auc": roc_auc_score(yt, yp_pos),
            "ap": average_precision_score(yt, yp_pos),
            "f1": f1_score(yt, pred_binary),
            "balanced_acc": balanced_accuracy_score(yt, pred_binary),
            "cm": confusion_matrix(yt, pred_binary),
            "opt_thresh": float(th[best]),
        }


# ── plots ─────────────────────────────────────────────────────────────────────
def save_plots(res: dict, meta: pd.DataFrame, out: Path, target: str,
               le: LabelEncoder | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    yt, yp = res["y_true"], res["y_prob"]

    if target == "cell_type" and le is not None:
        # Multi-class confusion matrix
        class_names = le.classes_
        cm = res["cm"]
        plt.figure(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt="d", cbar=True, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.title(f"Confusion Matrix (Macro AUC={res['macro_auc']:.3f})")
        plt.xticks(rotation=45, ha="right", fontsize=7)
        plt.yticks(fontsize=7)
        plt.tight_layout()
        plt.savefig(out / "cm.png", dpi=150)
        plt.close()

        # Per-class ROC
        plt.figure(figsize=(8, 6))
        for c, name in enumerate(class_names):
            if c in res["per_class_auc"]:
                binary_yt = (yt == c).astype(int)
                fpr, tpr, _ = roc_curve(binary_yt, yp[:, c])
                plt.plot(fpr, tpr, lw=1.2,
                         label=f"{name} (AUC={res['per_class_auc'][c]:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("Per-Cell-Type ROC (One-vs-Rest)")
        plt.legend(fontsize=6, loc="lower right")
        plt.tight_layout()
        plt.savefig(out / "roc_per_class.png", dpi=150)
        plt.close()
    else:
        # Binary plots
        yp_pos = yp[:, 1] if yp.ndim == 2 else yp
        fpr, tpr, _ = roc_curve(yt, yp_pos)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, lw=2, label=f"AUC={res['auc']:.3f}")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
        plt.savefig(out / "roc.png", dpi=150)
        plt.close()

        plt.figure(figsize=(4, 4))
        sns.heatmap(res["cm"], annot=True, fmt="d", cbar=False, cmap="Blues")
        plt.tight_layout()
        plt.savefig(out / "cm.png", dpi=150)
        plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark DNA embedding models on scATAC cell type classification"
    )
    p.add_argument("--h5_path",        required=True,
                   help="Path to aggregated model H5")
    p.add_argument("--metadata_path",  required=True,
                   help="Metadata CSV with sample_id, group columns")
    p.add_argument("--dna_model",      required=True,
                   help="Name of the DNA model")
    p.add_argument("--segments",       default="all",
                   help="Internal model only: segment selection")
    p.add_argument("--target_group",   default="cell_type",
                   help="'cell_type' for multi-class or '{CellType}_vs_all' for binary")
    p.add_argument("--classifier",     default="svm",
                   choices=list(_GRIDS), help="ML classifier")
    p.add_argument("--fast_mode",      action="store_true",
                   help="Use fast hyperparameter grid")
    p.add_argument("--pca_components", type=int, default=None)
    p.add_argument("--inner_cv_folds", type=int, default=3)
    p.add_argument("--outer_cv_folds", type=int, default=5,
                   help="Stratified k-fold (default 5)")
    p.add_argument("--n_jobs",         type=int,
                   default=os.cpu_count() or 1)
    p.add_argument("--output_dir",     default="results")
    p.add_argument("--run_name",       default=None)
    return p.parse_args()


def main() -> None:
    a = get_args()
    seg_label = a.segments.replace(",", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = a.run_name or (
        f"{a.dna_model}_{a.target_group}_{a.classifier}_{seg_label}_{ts}"
    )
    out = Path(a.output_dir) / run
    out.mkdir(parents=True, exist_ok=True)

    multiclass = (a.target_group == "cell_type")

    print(f"\n{'='*60}")
    print(f"  scATAC DNA Embedding Benchmark")
    print(f"  Model:      {a.dna_model}")
    print(f"  Segments:   {a.segments}")
    print(f"  Target:     {a.target_group} ({'multi-class' if multiclass else 'binary'})")
    print(f"  Classifier: {a.classifier} ({'fast' if a.fast_mode else 'full'})")
    print(f"  Outer CV:   {a.outer_cv_folds}-fold")
    print(f"  PCA:        {a.pca_components or 'off'}")
    print(f"  Output:     {out}")
    print(f"{'='*60}\n")

    X, y, meta, le = load_data(a.h5_path, a.metadata_path, a.segments, a.target_group)
    res = run_cv(
        X, y, a.classifier, a.fast_mode,
        a.inner_cv_folds, a.outer_cv_folds, a.n_jobs, a.pca_components,
        multiclass=multiclass,
    )

    save_plots(res, meta, out, a.target_group, le)

    # Save predictions
    pred_df = pd.DataFrame({
        "sample_id": meta["sample_id"].values,
        "group": meta["group"].values,
        "y_true": res["y_true"],
        "y_pred": res["y_pred"],
    })
    if not multiclass:
        pred_df["y_prob"] = res["y_prob"] if res["y_prob"].ndim == 1 else res["y_prob"][:, 1]
    pred_df.to_csv(out / "preds.csv", index=False)

    # Save summary
    if multiclass:
        per_class_auc_named = {}
        if le is not None:
            for c, auc_val in res["per_class_auc"].items():
                per_class_auc_named[le.classes_[c]] = auc_val
        summary = {
            "dna_model": a.dna_model,
            "segments": a.segments,
            "target": a.target_group,
            "classifier": a.classifier,
            "fast_mode": a.fast_mode,
            "outer_cv": f"{a.outer_cv_folds}-fold",
            "macro_auc": res["macro_auc"],
            "macro_f1": res["macro_f1"],
            "balanced_acc": res["balanced_acc"],
            "per_class_auc": per_class_auc_named,
        }
    else:
        summary = {
            "dna_model": a.dna_model,
            "segments": a.segments,
            "target": a.target_group,
            "classifier": a.classifier,
            "fast_mode": a.fast_mode,
            "outer_cv": f"{a.outer_cv_folds}-fold",
            "auc": res["auc"],
            "ap": res["ap"],
            "f1": res["f1"],
            "balanced_acc": res["balanced_acc"],
            "opt_thresh": res["opt_thresh"],
        }

    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if multiclass:
        print(f"\nMacro AUC={res['macro_auc']:.4f}  Macro F1={res['macro_f1']:.4f}  "
              f"BalAcc={res['balanced_acc']:.4f}")
    else:
        print(f"\nAUC={res['auc']:.4f}  AP={res['ap']:.4f}  "
              f"F1={res['f1']:.4f}  BalAcc={res['balanced_acc']:.4f}")
    print(f"Results → {out}")


if __name__ == "__main__":
    main()
