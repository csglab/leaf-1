# Embedding extraction

Frozen encoder → per-fragment window-pooled embeddings, written to HDF5. One file per sample.

## Run

```bash
bash scripts/extract_embeddings.sh \
  /path/to/manifest.csv \
  checkpoint \
  /path/to/output_dir \
  0,1,2,3 \
  4 \
  100000 \
  --target_layer 4
```

Positional args: `<manifest.csv> <checkpoint_dir> <output_dir> <gpu_list> <jobs_per_gpu> [n_fragment_per_file] [extra args ...]`. Pass `USE_SHM=1` in the env to stage the tokenizer and FASTA into `/dev/shm` first.

Paper run: 100,000 fragments per sample, layer 4, batch 64. Each H5 is ~1.9 GB.

Single-sample form:

```bash
leaf1-extract \
  --bert_checkpoint checkpoint \
  --n_fragment_file /path/to/manifest.csv \
  --index 0 \
  --output_dir /path/to/output_dir \
  --target_layer 4 \
  --n_fragment_per_file 100000
```

To extract all 13 layers instead of one, drop `--target_layer`. File size scales by ~13x.

For a checkpoint pretrained with `--ablation`, pass the same `--ablation` and run at its training length (`--max_token_length` = the checkpoint's `max_position_embeddings`). Other options:

- `--store_body_mean`: also write `embeddings/body_mean`, the mean over body tokens at `--target_layer` (n, D). Defined the same way for every ablation.
- `--presample N`: draw `--n_fragment_per_file` fragments from the first N drawn with `--seed`. `--presample 100000 --n_fragment_per_file 10000` gives the Fig 4 fragment set.
- `--per_fragment_rng`: seed strand and flank trimming per fragment, so the same fragment is processed identically by every model.

For `nocut` the two [CUT] chunks are zeros; for `noflank` the left and right flank chunks are zeros.

## H5 schema

Output path: `<output_dir>/h5_files/<modality>_<sample_id>.h5`.

```
embeddings/
  bert_window_embeddings        # (n, 9984) float16          when --target_layer is set
                                # (n, 13, 9984) float16      when omitted (embedding + 12 layers)
  attrs:
    num_windows = 3
    embed_dim = D               # 768 for the paper model; read from checkpoint config
    window_embed_dim = 13*D     # 9984 for the paper model
    num_layers = 1 or 13
    target_layer = K            # single-layer files only
    layer_names = JSON list of layer labels
    segment_offsets = JSON dict (see below)

fragment_info/
  chromosome                    (n,) vlen str
  fragment_start, fragment_end  (n,) int64   # original BED interval (0-based, half-open)
  adjusted_start, adjusted_end  (n,) int64   # widened region passed to bedtools
  reverse_complemented          (n,) bool
  n_token_left, n_token_frag,   (n,) int32   # token counts per region
    n_token_right
  n_freq                        (n,) float32 # PAD/N share in the tokenized sequence
  line_number                   (n,) int64   # BED row this fragment came from

metadata/
  attrs: bed_file, sample_id, modality, group,
         bert_checkpoint, seed, batch_size,
         n_fragments_total, n_fragments_sampled,
         total_batches, timestamp

progress/
  attrs: current_batch_idx, total_batches,
         batch_size, seed,
         processed_count, completed
```

### Column layout within one window-embedding row

`D = embed_dim`. Row width = `13 * D` = 9984 for the paper model.

| Columns | Source |
|---|---|
| `0 .. 3D-1` | left flank, 3 windows pooled |
| `3D .. 6D-1` | body, 3 windows pooled |
| `6D .. 9D-1` | right flank, 3 windows pooled |
| `9D .. 10D-1` | [CUT1] token (first boundary) |
| `10D .. 11D-1` | [CUT2] token (second boundary) |
| `11D .. 12D-1` | [CLS] token |
| `12D .. 13D-1` | [MOD] token |

The `segment_offsets` attribute on `embeddings/` stores these as a JSON dict: `{"left": 0, "fragment": 3D, "right": 6D, "cut1": 9D, "cut2": 10D, "cls": 11D, "mod": 12D}`.

## Read in Python

```python
import h5py
import numpy as np

with h5py.File("CFDNA_sample_x.h5", "r") as f:
    emb     = f["embeddings/bert_window_embeddings"][:]   # (n, 9984) float16
    chrs    = f["fragment_info/chromosome"][:]            # (n,) bytes
    starts  = f["fragment_info/fragment_start"][:]        # (n,) int64
    modality = f["metadata"].attrs["modality"]            # e.g. "CFDNA"

# scATAC-style sample-level vector: mean across the fragment bag.
cell_vec = emb.mean(axis=0).astype(np.float32)            # (9984,)

# Slice out only the [CUT] portion (the strongest cfDNA signal in the paper):
D = 768
cut_only = emb[:, 9 * D : 11 * D]                          # (n, 1536) float16
```

## Resume

H5s with `progress/completed = True` are skipped on re-run by `scripts/extract_embeddings.sh`. Partial H5s resume from `current_batch_idx + 1`; the seed and batch_size of the resumed call must match what was written, otherwise `evaluate_bed_file` raises.
