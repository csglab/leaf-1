# Gated-attention MIL

Sample-level classifier for cfDNA (and TCGA bulk ATAC) downstream tasks. Each sample is a bag of frozen LEAF-1 fragment embeddings; the gated-attention head produces a sample-level probability.

This is the optimized variant of the paper-canonical trainer (`mil/mil_nested/train_gated_final_optimized.py`). It keeps all three data-pipeline modes: GPU-resident, CPU-batch, and the NVMe-cached optimized pipeline. Inline model classes: `FrozenStandardize1d`, `FeatureProjector`, `GatedAttentionPool`.

## Run (single fold)

```bash
leaf1-mil \
  --config     leaf1/downstream/mil/configs/cfdna.yaml \
  --split      /path/to/splits/cancer_vs_healthy_k5.json \
  --fold       0 \
  --seed       42 \
  --inner_k    5 \
  --inner_folds 0,1,2,3,4 \
  --feat_proj  linear --feat_proj_dim 512 --feat_proj_ln \
  --att_fix    hhi \
  --max_epochs 200 \
  --output_dir /path/to/results
```

The trainer reads `--config` for defaults and `--split` for the CV layout. CLI flags override config values for the run.

## Data pipeline mode

Pick one (default = GPU-resident, suitable for small bags):

| Flag | When to use |
|---|---|
| (default) | Bags fit in GPU memory (~10K fragments per bag, < ~50 samples). Fastest. |
| `--cpu_data` | Bags fit in CPU memory; per-batch H2D transfer. Use for 50K-100K fragments per bag. |
| `--gpu_fp16` | Halve GPU memory by keeping bags as float16 on device. Compatible with default mode. |
| `--optimized_pipeline --nvme_path <fast-local-dir>` | NVMe pre-collation + multi-reader + GPU prefetch. Use for 100K fragments per bag and large cohorts. |
| `--amp` | Automatic mixed precision during training (forward in bf16/fp16). |

## Config (YAML)

Two shipped presets:

| File | Use for | `data.segments` | `data.frags_per_bag` (default) |
|---|---|---|---|
| `configs/cfdna.yaml` | cfDNA tasks (Cristiano cohort, internal cohorts) | `"cut,mod,fragment1"` | 10,000 |
| `configs/tcga.yaml` | TCGA bulk ATAC tasks, scATAC supplementary | `"mod,fragment1"` | 10,000 |

> **Important: the YAML's `data.frags_per_bag: 10000` is a fast smoke / sweep-search default, NOT the paper-canonical bag size.** The paper headline cfDNA result (pan-cancer AUC 0.949) was produced with **100,000 fragments per bag**. To reproduce that, pass `--frags_per_bag 100000` on the `leaf1-mil` CLI. The CLI override always wins over the YAML.

Each YAML's `sweep.grid` block enumerates the hyperparameter search space using the trainer's CLI flag names (`head_C`, `feat_proj`, `feat_proj_dim`, `att_fix`, `base_lr`, `reg_scheme`, `scheme_value`, …). The grid expansion script `scripts/run_mil_sweep.sh` validates every grid key against `leaf1-mil --help` before submitting any jobs, so a typo fails loudly up front.

## Output (per fold)

```
<output_dir>/
  summary.json          # metrics + best inner config + training history
  preds.csv             # sample_id, fold, y_true, y_prob, attention stats
  checkpoint_best.pt    # best model weights by val AUC
  config.json           # full resolved config for the run
```

## Inference: score new samples from a fitted checkpoint

Training (`leaf1-mil`) *fits and saves* an ensemble; `leaf1-mil-predict` *loads
and scores*. A fitted model is an ensemble of the `inner_fold_*` checkpoints; the
full nested-CV model for a task is all five outer folds averaged together.

```bash
leaf1-mil-predict \
  --mil_checkpoint mil_checkpoint/cfdna/cancer_vs_healthy \
  --hdf5_pattern '/path/to/extract_output/h5_files/CFDNA_*.h5' \
  --all_folds \
  --frags_per_bag 100000 \
  --output_dir results/scored
```

- `--mil_checkpoint` points at a task dir (`.../cancer_vs_healthy`) with `F0..F4/`,
  or a single outer-fold dir with `inner_fold_*/` directly.
- `--all_folds` ensembles every inner fold across every outer fold (25 models) —
  the right choice for scoring an **external cohort**. Without it, only `F0` is used.
- Input H5s are the flat `leaf1-extract` layout (`embeddings/bert_window_embeddings`,
  shape `(n, 9984)`). Labels are read from each H5's `metadata/group` attr when present.
- Segments are read from the checkpoint's `config.json` (`cut,mod,fragment1` for
  cfDNA) so inference matches training; `--segments` can override.

Output `results/scored/`:

```text
sample_summary.csv      # sample_id, group, n_fragments, pred_prob, pred_logit, n_models
inference_config.json   # resolved checkpoint / segments / feat_dim / n_models
```

`pred_prob` is the ensemble-averaged P(cancer). See the pretrained ensembles in
[`mil_checkpoint/`](../../../mil_checkpoint) (Git LFS).

## Split JSON

The trainer expects a CV split file alongside `<split_root>/metadata_complete.csv`:

```
{
  "metadata": {
    "comparison_type": "cancer_vs_healthy" | "one_vs_all" | "stage" | "pairwise_same_tissue",
    "target": "<group label>",
    ...
  },
  "splits": [
    { "train_idx": [...], "val_idx": [...] },
    ...
  ]
}
```

`load_splits` reads `metadata_complete.csv` from the parent of `<split_dir>` (so the layout is `<root>/cv_splits/foo_k5.json` + `<root>/metadata_complete.csv`).

## Sweep

The trainer accepts a single config per call. To run the full sweep grid:

```bash
bash scripts/run_mil_sweep.sh \
  --config leaf1/downstream/mil/configs/cfdna.yaml \
  --split  /path/to/splits/cancer_vs_healthy_k5.json \
  --fold   0 \
  --output_root /path/to/sweep_output
```

The launcher iterates over the YAML's `sweep.grid` and submits one `leaf1-mil` call per point. Leaderboard reconciliation (best-config-per-fold selection) is done downstream by the user's own scripts; the trainer just emits a `summary.json` per run.
