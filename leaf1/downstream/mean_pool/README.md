# Mean pool + linear probe

Two-step downstream path for sample-level classification on top of frozen LEAF-1 embeddings.

1. **Aggregate** per-sample embeddings (each sample is one H5 from `leaf1-extract`) into a single summary H5 with one row per sample. The aggregator mean-pools across fragments and reshapes the flat 9984-d row produced by `leaf1-extract --target_layer 4` into the (13, 768) segment-major layout that the trainer expects.
2. **Train** a nested-CV SVM (or LR / RF / XGB / LGB) over the aggregated H5. Supports binary one-vs-rest (`{CellType}_vs_all`) and multi-class (`cell_type`) targets.

## Aggregate

```bash
leaf1-mean-pool-aggregate \
  --input_dir   /path/to/leaf1-extract-output \
  --output      /path/to/aggregated.h5 \
  --model_name  leaf1 \
  --workers     8 \
  --layer_idx   4
```

Output H5:

```
embeddings    (N, 13, 768) float32   # mean across fragments per sample, segment-major
sample_ids    (N,) bytes
groups        (N,) bytes             # cohort / cell-type label, passed through from extract metadata
attrs: model_name, embed_dim, num_segments, num_samples, pooling, layer_idx, timestamp
```

The aggregator reads `embeddings/bert_window_embeddings` from each per-sample H5 and skips files whose `progress/completed` attribute is false.

## Train

```bash
leaf1-mean-pool-train \
  --h5_path        /path/to/aggregated.h5 \
  --metadata_path  /path/to/metadata.csv \
  --dna_model      leaf1 \
  --segments       'cut,mod,fragment1'   \
  --target_group   cell_type             \
  --classifier     svm                   \
  --outer_cv_folds 5                     \
  --inner_cv_folds 3                     \
  --output_dir     ./results
```

`--segments` (paper-canonical):

| Modality | Segments | Why |
|---|---|---|
| cfDNA | `cut,mod,fragment1` | cleavage boundaries dominate; paper Figure 4d |
| scATAC | `mod,fragment1` | fragment body dominates Tn5 signal; paper §S2 |

Use `_mean` suffix (e.g. `cut,mod,fragment1_mean`) to average within selected segments instead of concatenating them.

`--target_group`:
- `cell_type` runs multi-class one-vs-rest over all groups in the metadata.
- `{Group}_vs_all` runs binary one-vs-rest for a single group.

`--classifier` options: `lr`, `rf`, `svm`. With `xgboost` / `lightgbm` installed, also `xgb`, `lgb`.

`--fast_mode` shrinks the inner GridSearchCV grid for quick smoke runs.

Output dir contents:

```
results/<run_name>/
  preds.csv      sample_id, group, y_true, y_pred[, y_prob]
  summary.json   metrics (binary: auc / ap / f1 / balanced_acc / opt_thresh
                          multi: macro_auc / macro_f1 / balanced_acc / per_class_auc)
  roc.png        binary only
  roc_per_class.png  multi-class only
  cm.png         confusion matrix
```

## Metadata CSV format

```
sample_id,group
SRR12345,B_Cell
SRR12346,B_Cell
SRR12347,T_Cell
...
```

Extra columns (e.g. `h5_index`) are accepted; the loader matches on `sample_id` first.
