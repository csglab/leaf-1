# LEAF-1 MIL checkpoints

Fitted gated-attention **MIL classifier** ensembles for the cfDNA downstream
tasks — the sample-level heads that sit on top of the frozen LEAF-1 encoder.
Where [`checkpoint/`](../checkpoint) holds the frozen BERT encoder, this holds
the trained classifiers that turn a bag of that encoder's fragment embeddings
into a sample-level cancer probability.

Load them with:

```bash
leaf1-mil-predict --mil_checkpoint mil_checkpoint/cfdna/cancer_vs_healthy \
                  --hdf5_pattern '/path/to/CFDNA_*.h5' --all_folds \
                  --output_dir results/
```

These classifiers were fit on embeddings from the released encoder (`checkpoint/`,
`leaf1-extract --model leaf1`) and are not valid for the other encoders in
[`encoder_checkpoints/`](../encoder_checkpoints). `leaf1-mil-predict` checks each H5's `leaf1_model`
metadata and stops if it names another encoder.

Payload files (`*.pt`) are tracked with Git LFS. After cloning:

```bash
git lfs install
git lfs pull --include="mil_checkpoint/**"
```

## Layout

```text
mil_checkpoint/cfdna/
  <task>/                      # 15 cfDNA tasks (see below)
    F0/ ... F4/                # 5 outer folds (nested CV)
      config.json              # resolved run config (incl. data_config.segments)
      summary.json             # fold metrics + best inner config
      inner_fold_0/ ... _4/    # 5 inner-fold models
        checkpoint.pt          # {proj,pool,head}_state_dict + config
        norm_stats.pt          # {mean, std} feature standardization
```

**A "fitted model" is an ensemble.** One outer fold `F<k>/` is an ensemble of its
5 `inner_fold_*` models (their predictions are averaged). The full nested-CV
model for a task is all 5 outer folds — 25 inner-fold checkpoints — averaged
together (`--all_folds`). Totals: 15 tasks × 5 × 5 = **375 inner-fold
checkpoints**.

## Tasks

`cancer_vs_healthy` is the **pan-cancer** model (the paper headline; also used for
the independent kidney-cohort validation). The remaining 14 are per-cancer:
`<Type>_cancer_vs_healthy` and `<Type>_cancer_vs_other_cancers` for Bile_duct,
Breast, Colorectal, Gastric, Lung, Ovarian, and Pancreatic.

## Provenance

Trained by `leaf1-mil` (gated-attention MIL, HHI variance rescaling) on Cristiano
cfDNA cohort embeddings, 100,000 fragments per bag, `cut,mod,fragment1` segments,
best sweep config `dropout, head_C=10, l1_ratio=0` (`att_fix=hhi`, `feat_proj=linear`
512-d with LayerNorm). Source: `final_models_cfdna_100k/` sweep outputs; assembled
with [`scripts/build_mil_checkpoint.sh`](../scripts/build_mil_checkpoint.sh). Each
`checkpoint.pt` reconstructs a `FeatureProjector` + `GatedAttentionPool` + linear
head; `config.json` records the exact training configuration for audit.
