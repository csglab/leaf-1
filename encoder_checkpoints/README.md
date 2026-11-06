# Additional LEAF-1 encoder checkpoints

Encoders from the manuscript revision, next to the released model in
[`../checkpoint/`](../checkpoint). Select one with `leaf1-extract --model <name>`:

| `--model` | Directory | Input | Pretraining | `model.safetensors` md5 |
|---|---|---|---|---|
| `leaf1` (default) | `checkpoint/` | `[CLS] [MOD] left [CUT] body [CUT] right`, 2048 tokens | Released model, 100,000 updates | `1b28ca84edc43ee66365d99ccc597c65` |
| `leaf1-noflank-s42` | `encoder_checkpoints/leaf1-noflank-s42/` | `[CLS] [MOD] [CUT] body [CUT]`, 512 tokens | No flanking sequence; 30,000 updates at batch 768; Cristiano-free corpus; seed 42 | `ce70e67c59011fd1a8a4d02394be014f` |
| `leaf1-noflank-s43` | `encoder_checkpoints/leaf1-noflank-s43/` | `[CLS] [MOD] [CUT] body [CUT]`, 512 tokens | Same as above, seed 43 | `9b3ed5b1e4ac4fb2ec9ea335b97b5898` |
| `leaf1-nocristiano` | `encoder_checkpoints/leaf1-nocristiano/` | Same as `leaf1` | Released recipe without the Cristiano et al. cfDNA samples; 104,500 updates (23,500 at batch 4,088 + 81,000 at batch 768), stopped once validation loss reached the released model's | `6a8122f4eb3bd64c903cb930483574bb` |

The manuscript results use `leaf1`. Leaving out `--model` gives `leaf1`, so existing
commands are unchanged.

## Fetching the weights

These weights are tracked with Git LFS and are **not** downloaded by a plain clone
(see `.lfsconfig`). Fetch only the ones you need:

```bash
git lfs install
git lfs pull --include="encoder_checkpoints/leaf1-noflank-s42/**" --exclude=""
```

`--exclude=""` overrides the default exclude in `.lfsconfig`. Each model is ~360 MB.

## Usage

```bash
leaf1-extract --model leaf1-noflank-s42 \
  --n_fragment_file /path/to/manifest.csv --index 0 \
  --output_dir /path/to/out --target_layer 4 --n_fragment_per_file 100000
```

or with the multi-GPU wrapper, pass the directory:

```bash
bash scripts/extract_embeddings.sh manifest.csv encoder_checkpoints/leaf1-noflank-s42 out 0,1,2,3 4 100000 --target_layer 4
```

## Input settings travel with the checkpoint

Each model only gives correct embeddings for the input format it was pretrained on.
A flank-free model given flanked input, or a flanked model given body-only input,
produces wrong embeddings without any error. To prevent that, each `config.json`
here records its input settings:

```json
"leaf1_model": "leaf1-noflank-s42",
"leaf1_ablation": "noflank",
"leaf1_max_token_length": 512,
"leaf1_min_flank_token_length": 0
```

`leaf1-extract` reads these whether the checkpoint is selected with `--model` or with
`--bert_checkpoint <dir>`, and:

- uses them when `--ablation`, `--max_token_length` and `--min_flank_token_length` are not given;
- stops with an error if any of those flags is given with a different value;
- checks the md5 of `model.safetensors`, and stops if the weights are an un-pulled LFS pointer;
- writes `leaf1_model` to the H5 `metadata`, and refuses to resume an H5 written by a different model.

The released `checkpoint/` is left exactly as released (no `leaf1_*` keys). Its weights
are recognised by md5 instead, and it is held to its own input format (`none`, 2048,
100) the same way: e.g. `--bert_checkpoint checkpoint --ablation noflank` is an error.
Its H5 files get no `leaf1_model` attribute, so they are unchanged.

## MIL classifiers

The fitted classifiers in [`../mil_checkpoint/`](../mil_checkpoint) were trained on
`leaf1` embeddings and are not valid for any other encoder, including
`leaf1-nocristiano`. `leaf1-mil-predict` checks the `leaf1_model` attribute of every
H5 and stops if it isn't `leaf1`. If you fit your own classifier on another model's
embeddings, pass `--encoder_model <name>` when scoring.

## Files

Weights only: `config.json`, `model.safetensors`, the tokenizer files and
`trainer_state.json` (training log). No optimizer, scheduler or RNG state. The
weights are byte-identical to the checkpoints used in the revision. The only change
to `config.json` is the four `leaf1_*` keys above.
