<p align="center">
  <img src="assets/logo_LEAF-1.png" alt="LEAF-1 logo" width="150">
</p>

# LEAF-1

Implementation of **LEAF-1**, from *Semantic fragment representations for
coordinate-free analysis of genomics data* (Heydari, Zhao, et al., bioRxiv, 2026).

LEAF-1 is a transformer encoder for fragment-level genomic representations.
It is pretrained on DNA fragments rather than aligned coordinate bins. The frozen
encoder emits a 9,984-dimensional embedding per fragment from sequence context,
assay modality, fragment body, flanks, and cleavage-boundary tokens. Downstream
classifiers operate on these embeddings.

Manuscript results reported for this release include cfDNA pan-cancer detection
AUC 0.949 on the Cristiano cohort with LEAF-1 + gated-attention MIL, scATAC
cell-type classification mean AUC 0.93 across 108 cell types from ~1,000
fragments per cell, and frozen-model kidney-cancer validation AUC 0.833. The
matched benchmark comparisons include coordinate-binning, DELFI, k-mer
(k = 3..6), DNABERT-2, DNABERT-S, GENA-LM, NT-v2, and Evo 2 baselines.

---

## Model summary

| Property | Value |
|---|---|
| Architecture | DistilBERT-style encoder, 12 layers × 12 heads × 768 dim, FFN 4× (3072), learned position embeddings |
| Parameters | 90.4M (full checkpoint; transformer-block weights alone are ~67M) |
| Vocabulary | 4,099 tokens (BPE; ships in `leaf1/data/tokenizer.json`) |
| Sequence length | 2,048 tokens per fragment |
| Pretraining objective | MLM with **joint [CUT]-token masking** (base tokens 15%, [CUT] tokens 30%; both boundaries of a fragment masked together) |
| Pretraining corpus | 57.1 billion sequenced fragments across 2,525 samples spanning cell-free DNA, bulk ATAC-seq, and single-cell ATAC-seq, before modality-balancing replication |
| Reference genome | GRCh38 |

Each fragment is tokenized as

```
[CLS] [MOD] left_flank [CUT] body [CUT] right_flank
```

where `[MOD]` is a learned per-modality token (`CFDNA`, `TCGA`, `SCATAC`, …) so a
single encoder can serve assays with different fragmentation chemistries.

The frozen encoder produces a per-fragment embedding by pooling each transformer
layer over 13 positional slots: three windows over the left flank, three over the
body, three over the right flank, plus `[CUT1]`, `[CUT2]`, `[CLS]`, and `[MOD]`.
This gives a 13 × 768 = 9,984-dimensional representation. Layer 4 is the
default downstream layer.

---

## Repository layout

```
leaf-1/
├── leaf1/                          # the Python package (pip-installable)
│   ├── data/tokenizer.json         # ships inside the wheel
│   ├── common/                     # fragment tokenisation + datasets + collator + pooling
│   ├── train/pretrain.py           # MLM pretraining entry point
│   ├── infer/extract_embeddings.py # frozen-encoder embedding extraction → HDF5
│   ├── downstream/
│   │   ├── mean_pool/              # nested-CV linear probe (SVM / LR / RF) with segment selection
│   │   └── mil/                    # gated-attention MIL classifier
│   └── util/                       # preprocessing + layer reshape + extract validation
├── scripts/                        # shell launchers (torchrun, GNU parallel)
├── tests/                          # 147 tests; pytest
├── environment-gpu.yml             # paper-era CUDA 12.1 / PyTorch 2.1 conda env
├── pyproject.toml                  # PEP 517 / 518 install + CLI entry points
├── checkpoint/                     # frozen LEAF-1 encoder, Git LFS-backed
│   ├── README.md
│   └── model.safetensors           # LEAF-1 weights, Git LFS
├── mil_checkpoint/                 # fitted MIL classifier ensembles, Git LFS-backed
│   └── cfdna/<task>/F0..F4/inner_fold_0..4/{checkpoint.pt, norm_stats.pt}
└── assets/                         # reference assets
    ├── README.md
    ├── refseq.genome.fa            # GRCh38, 3.1 GB, Git LFS
    ├── refseq.genome.fa.fai        # samtools index
    └── hg38-blacklist.v2.bed.gz    # BED-level blacklist
```

The package exposes nine command-line tools:

| Stage | CLI |
|---|---|
| Preprocess raw `.frag.gz` → filtered BED + sparse `.bed.index` | `leaf1-bed-preprocess` |
| Pretrain the encoder (MLM) | `leaf1-pretrain` |
| Extract per-fragment embeddings → HDF5 | `leaf1-extract` |
| Reshape extract output to a per-sample (N, 13, 768) layout | `leaf1-reshape-layer` |
| Diff two extract outputs (with fp16 ULP tolerance) | `leaf1-validate-extract` |
| Aggregate per-sample H5s into one summary H5 | `leaf1-mean-pool-aggregate` |
| Nested-CV linear probe over aggregated embeddings | `leaf1-mean-pool-train` |
| Gated-attention MIL classifier | `leaf1-mil` |
| Score samples with a fitted MIL ensemble | `leaf1-mil-predict` |

---

## Install

The repository pins the paper-era runtime (PyTorch 2.1.2 + CUDA 12.1 + Transformers 4.37) for reproduction on hardware with a CUDA 12.x driver. A CPU environment can be used for unit tests and development.

### GPU (recommended for pretraining, embedding extraction, MIL)

```bash
conda env create -f environment-gpu.yml
conda activate leaf1-gpu
pip install -e . --no-deps   # install the package without disturbing the pinned runtime

# Smoke check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
leaf1-extract --help
pytest tests/ -v
```

The `--no-deps` flag is intentional: it keeps the pinned conda runtime stable. A
plain `pip install -e ".[test]"` would let pip pull a newer torch wheel which may
require a newer NVIDIA driver than is installed.

### CPU (development and testing)

```bash
conda create -n leaf1 python=3.10 pip bedtools samtools -y
conda activate leaf1
pip install -e ".[test]"
pytest tests/ -v   # 147 tests
```

### System dependencies

The package shells out to two well-known bioinformatics utilities:

| Tool | Used by | Install |
|---|---|---|
| `bedtools` ≥ 2.30 | `leaf1-bed-preprocess`, `leaf1-extract` | `conda install -c bioconda bedtools` |
| `samtools` | building `.fai` indices when adding new reference FASTAs | `conda install -c bioconda samtools` |
| GNU `parallel` | the cross-sample inference launcher `scripts/extract_embeddings.sh` | `conda install -c conda-forge parallel` |

---

## Reference assets

The GRCh38 reference FASTA lives at `assets/refseq.genome.fa` with its samtools
`.fai` index. The hg38 blacklist used by `leaf1-bed-preprocess` lives at
`assets/hg38-blacklist.v2.bed.gz`. The FASTA is 3.1 GB and is tracked through Git
LFS. After cloning, fetch the LFS payload if Git did not download it
automatically:

```bash
git lfs install
git lfs pull --include="assets/**,checkpoint/**,mil_checkpoint/**"
```

The reference must match the chromosomes used to preprocess your BEDs; the
manuscript experiments used the FinaleDB-distributed GRCh38 reference flavor. See
[`assets/README.md`](assets/README.md) for details.

The tokenizer (`leaf1/data/tokenizer.json`, 165 KB) ships inside the Python
package; no manual download is required.

## Model checkpoint

The pretrained LEAF-1 checkpoint is stored in `checkpoint/` through Git LFS. It
is a HuggingFace checkpoint directory, not a single file. After cloning, fetch the
LFS payload if Git did not download it automatically:

```bash
git lfs install
git lfs pull --include="checkpoint/**"
```

Use it with inference commands as `--bert_checkpoint checkpoint`. The checkpoint
corresponds to the legacy internal-model LEAF-1 encoder used for the cfDNA
layer-4 manuscript runs. See [`checkpoint/README.md`](checkpoint/README.md) for
the expected files and LFS notes.

## MIL classifier checkpoints

The fitted gated-attention MIL classifiers — the sample-level heads for the cfDNA
tasks — are stored in `mil_checkpoint/` through Git LFS (also fetched by the
`git lfs pull` above). Each task (`cancer_vs_healthy` = pan-cancer, plus 14
per-cancer tasks) is a nested-CV ensemble of 5 outer × 5 inner fold models. Score
new samples with `leaf1-mil-predict` (see below). See
[`mil_checkpoint/README.md`](mil_checkpoint/README.md) for the layout and provenance.

## Required Inputs and Generated Outputs

This git repository contains the release code, tokenizer, tests, small fixtures,
and Git LFS pointers for the reference FASTA and release checkpoint. Controlled
cohort inputs and paper-scale generated outputs are not tracked:

| Item | Status |
|---|---|
| GRCh38 reference FASTA + `.fai` | Tracked under `assets/` with Git LFS for the FASTA |
| hg38 blacklist BED | Tracked at `assets/hg38-blacklist.v2.bed.gz` |
| Pretrained LEAF-1 encoder checkpoint | Tracked under `checkpoint/` with Git LFS |
| Fitted MIL classifier ensembles | Tracked under `mil_checkpoint/` with Git LFS (15 cfDNA tasks × 5 outer × 5 inner folds) |
| Raw cohort fragment files | Access through the source cohorts or controlled-access process; not redistributed here |
| Paper-scale per-fragment embedding H5s | Generated locally from authorized cohort access using this repo and the released checkpoint; not redistributed here |

The extraction, reshape, validation, mean-pool, and MIL commands below are the
interfaces used to regenerate downstream outputs from authorized input fragment
manifests.

---

## Quick start: extract embeddings from your own BEDs

A minimal run from a sorted, MAPQ-filtered fragment BED to a per-fragment
embedding H5 uses three commands. The example assumes a BED file at
`/data/sample_x.bed` with ~10⁶ fragments.

```bash
# 1. One-time per BED: build the sparse byte-offset index
python -c "from leaf1.util.bed_preprocess import build_sparse_index; \
           build_sparse_index('/data/sample_x.bed', '/data/sample_x.bed.index')"

# 2. Write a one-row manifest CSV
cat > /tmp/manifest.csv <<'CSV'
sample_id,modality,bed_file,n_fragment,group
sample_x,CFDNA,/data/sample_x.bed,1000000,my_cohort
CSV

# 3. Extract layer-4 window-pooled embeddings for 100,000 random fragments
leaf1-extract \
  --bert_checkpoint  checkpoint \
  --n_fragment_file  /tmp/manifest.csv \
  --index            0 \
  --output_dir       /tmp/leaf1_out \
  --target_layer     4 \
  --n_fragment_per_file 100000 \
  --seed             42
```

Result: `/tmp/leaf1_out/h5_files/CFDNA_sample_x.h5` with `embeddings/bert_window_embeddings`
of shape `(100000, 9984)` (float16), plus a `fragment_info/` group with provenance
columns for every fragment.

To convert this output into the published `(N, 13, 768)` per-sample format used by
the downstream classifiers:

```bash
leaf1-reshape-layer \
  --input  /tmp/leaf1_out/h5_files/CFDNA_sample_x.h5 \
  --output /tmp/leaf1_out/reshaped/CFDNA_sample_x.h5 \
  --layer  4
```

Schema details for both H5 layouts: [`leaf1/infer/README.md`](leaf1/infer/README.md).

---

## Manifest format (shared by `leaf1-pretrain` and `leaf1-extract`)

```
sample_id,modality,bed_file,n_fragment,group
```

| Column | Description |
|---|---|
| `sample_id` | Unique sample identifier. Used in output filenames. |
| `modality` | Modality token: `CFDNA`, `TCGA`, or `SCATAC`. Common lowercase aliases (`cfdna`, `scatac`) are normalized. |
| `bed_file` | Absolute path to a sorted fragment BED. Each line must be `chr <tab> start <tab> end <tab> mapq <tab> strand`. A sibling `<bed_file>.index` byte-offset sidecar must exist; produce one with `leaf1-bed-preprocess` or `leaf1.util.bed_preprocess.build_sparse_index`. |
| `n_fragment` | Line count of the BED file. |
| `group` | Arbitrary cohort tag, passed through to HDF5 metadata. |

If you start from a raw `.frag.gz` file (e.g. from FinaleDB), one command
produces the filtered BED + index + summary statistics:

```bash
leaf1-bed-preprocess \
  --input      /data/raw/EE85723.hg38.frag.gz \
  --output_dir /data/filtered
```

This applies the paper's filter chain: MAPQ >= 30, fragment length >= 20 bp,
canonical chromosomes only, blacklist regions removed via `bedtools intersect`.
By default it uses `assets/hg38-blacklist.v2.bed.gz` and
`assets/refseq.genome.fa.fai`; pass `--blacklist` or `--ref_fai` to override.

---

## Pretraining

The paper-canonical configuration is the default: 12 layers × 12 heads × 768 dim,
sequence length 2,048, MLM 15% / [CUT] 30%, 40,000 optimizer steps, bf16,
and effective batch size 4,096 when run as 32 sequences/GPU × 32 gradient
accumulation steps × 4 GPUs.

```bash
export TRAIN_CSV=/path/to/train_manifest.csv
export VAL_CSV=/path/to/val_manifest.csv
export OUTPUT_ROOT=/path/to/checkpoints
export LOG_ROOT=/path/to/hf_logs

bash scripts/pretrain.sh
```

The launcher auto-detects available GPUs via `CUDA_VISIBLE_DEVICES` or
`nvidia-smi` and dispatches to `torchrun`. Resume is automatic: re-running
the script picks up the latest HuggingFace checkpoint under
`$OUTPUT_ROOT/<run-id>/`. A `COMPLETED` marker is written on clean exit so
subsequent runs are idempotent.

Full flag reference, evaluation-metric definitions, and scaling guidance:
[`leaf1/train/README.md`](leaf1/train/README.md).

---

## Embedding extraction (across many samples)

For more than a handful of samples, use the GNU-parallel launcher to fan out
across multiple GPUs:

```bash
bash scripts/extract_embeddings.sh \
  /path/to/manifest.csv \
  checkpoint \
  /path/to/output_dir \
  0,1,2,3 \      # comma-separated GPU IDs
  4 \             # concurrent jobs per GPU
  100000 \        # fragments per sample
  --target_layer 4
```

Each sample produces one file at `<output_dir>/h5_files/<modality>_<sample_id>.h5`.
Completed files are skipped on re-run; partial files resume from the last
batch. See [`leaf1/infer/README.md`](leaf1/infer/README.md) for the H5 schema
and a Python loading example.

---

## Downstream classifiers

Two paper-canonical downstream paths ship inside `leaf1/downstream/`:

### Mean-pool linear probe for scATAC cell-type classification and cfDNA representation benchmarks

```bash
# 1. Aggregate per-sample H5s into a single summary H5 (one row per sample)
leaf1-mean-pool-aggregate \
  --input_dir  /path/to/extract_output \
  --output     /path/to/aggregated.h5 \
  --model_name leaf1

# 2. Train a 5-fold nested-CV linear probe with segment selection
leaf1-mean-pool-train \
  --h5_path        /path/to/aggregated.h5 \
  --metadata_path  /path/to/manifest.csv \
  --dna_model      leaf1 \
  --segments       'cut,mod,fragment1' \
  --target_group   'Healthy_vs_all' \
  --classifier     svm \
  --output_dir     /path/to/results
```

Details: [`leaf1/downstream/mean_pool/README.md`](leaf1/downstream/mean_pool/README.md).

### Gated-attention MIL for cfDNA pan-cancer and per-cancer detection

```bash
leaf1-mil \
  --config    leaf1/downstream/mil/configs/cfdna.yaml \
  --split     /path/to/cv_splits/cancer_vs_healthy_k5.json \
  --fold      0 \
  --feat_proj linear --feat_proj_dim 512 --feat_proj_ln \
  --att_fix   hhi \
  --head_C    10.0 \
  --reg_scheme dropout --scheme_value 0.0 \
  --base_lr   5e-4 \
  --frags_per_bag 100000 \
  --cpu_data \
  --output_dir /path/to/mil_results
```

Three data-pipeline modes are available depending on bag size and hardware:
GPU-resident (small bags fit on the GPU), CPU-batch (per-batch H2D transfer),
and an NVMe-cached optimized pipeline for the largest sweeps. The trainer
implements the paper's HHI variance-rescaling for the attention head.

To **score new samples with a fitted classifier** (the released `mil_checkpoint/`
ensembles, or your own `leaf1-mil` output), use `leaf1-mil-predict`:

```bash
leaf1-mil-predict \
  --mil_checkpoint mil_checkpoint/cfdna/cancer_vs_healthy \
  --hdf5_pattern '/path/to/extract_output/h5_files/CFDNA_*.h5' \
  --all_folds \
  --frags_per_bag 100000 \
  --output_dir /path/to/scored
```

This averages the nested-CV ensemble and writes `sample_summary.csv`
(`sample_id, group, pred_prob, pred_logit`). `--all_folds` uses all 25 fold
models, appropriate for an external cohort.

Configuration reference, sweep launcher, and tuning notes:
[`leaf1/downstream/mil/README.md`](leaf1/downstream/mil/README.md).

---

## Reproducibility

This repository provides the code paths used to regenerate LEAF-1 outputs from
authorized fragment data and a released checkpoint:

- `leaf1-bed-preprocess` for BED-level filtering and sparse index generation.
- `leaf1-extract` and `leaf1-reshape-layer` for layer-4 fragment embeddings.
- `leaf1-mean-pool-aggregate` and `leaf1-mean-pool-train` for mean-pool probes.
- `leaf1-mil` for gated-attention multiple-instance learning.

The encoder is deterministic given the same checkpoint, tokenizer, reference
FASTA, seed, and `--target_layer`. The supplied environment file
(`environment-gpu.yml`) pins the runtime used during paper preparation. Later
Torch or Transformers releases may work, but they are not the reference runtime.

The MIL trainer reads numeric hyperparameters from YAML and fold definitions from
JSON. To reproduce a run, keep the config, split file, seed, checkpoint, fragment
selection, and torch build fixed.

---

## Tests

```bash
pytest tests/ -v
```

147 tests cover fragment tokenisation, the MLM collator, the multi-window pool,
H5 round-trip integrity, the BED preprocessing and reshape utilities, the
mean-pool linear probe, and the MIL data loader and model components. Most
tests run in seconds; a handful require `bedtools` on the PATH (these skip
automatically if not found).

---

## License

This work is released under the PolyForm Noncommercial License 1.0.0. See [`LICENSE`](LICENSE).

---

## Citing LEAF-1

If you use this code or the pretrained model, please cite the LEAF-1 manuscript:

```bibtex
@article{heydari2026leaf1,
  title     = {Semantic fragment representations for coordinate-free analysis of genomics data},
  author    = {Heydari, Hamed and Zhao, Jing and Arseneault, Madeleine and
               Younesian, Leila and Tanguay, Simon and Riazalhosseini, Yasser and
               Goodarzi, Hani and Najafabadi, Hamed},
  journal   = {bioRxiv},
  year      = {2026},
  publisher = {Cold Spring Harbor Laboratory}
}
```

The DOI will be added once the preprint is live on bioRxiv.
[`CITATION.cff`](CITATION.cff) is included for GitHub citation metadata.
