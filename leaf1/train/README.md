# Pretraining

MLM pretraining for the LEAF-1 encoder. Writes a HuggingFace checkpoint directory under `$OUTPUT_ROOT/<run-id>/`.

## Run

```bash
export TRAIN_CSV=/path/to/train_manifest.csv
export VAL_CSV=/path/to/val_manifest.csv
export OUTPUT_ROOT=/path/to/checkpoints
export LOG_ROOT=/path/to/hf_logs
export USE_SHM=1

bash scripts/pretrain.sh
```

Default release recipe: 12L / 12H / 768d (~90M parameters), seq_len 2048, batch 32 x grad_accum 32, BF16, 40k optimizer steps.

For finer control, call the Python module directly:

```bash
torchrun --nproc-per-node=2 -m leaf1.train.pretrain \
  --train_csv $TRAIN_CSV --val_csv $VAL_CSV \
  --output_root $OUTPUT_ROOT --log_root $LOG_ROOT \
  --bf16
```

## Flags

Full list: `leaf1-pretrain --help`. Paper-canonical values:

| Flag | Default | Meaning |
|---|---|---|
| `--n_layers` | 12 | encoder depth |
| `--n_heads` | 12 | attention heads; `dim = n_heads * 64` |
| `--ffn_mult` | 4 | `hidden_dim = ffn_mult * dim` |
| `--seq_len` | 2048 | token budget per fragment |
| `--vocab_size` | 4099 | matches `leaf1/data/tokenizer.json` |
| `--min_flank` | 100 | minimum tokens per flank when the budget allows |
| `--batch_size` | 32 | per device |
| `--grad_accum` | 32 | per device |
| `--max_steps` | 40000 | optimizer updates; overrides epochs |
| `--learning_rate` | 1e-4 | linear warmup over 1000 steps, then linear decay |
| `--weight_decay` | 1e-5 | AdamW |
| `--mlm_probability` | 0.15 | base token mask rate |
| `--cut_mlm_probability` | 0.30 | joint [CUT] mask rate |
| `--ablation` | none | pretraining ablation, see below |
| `--bf16` | set by `pretrain.sh` | mixed precision; pretrain falls back to fp32 on CPU |
| `--seed` | 42 | |

### Ablations

`--ablation` changes one design element and leaves everything else as in the released model:

| Value | Change |
|---|---|
| `none` | released input: `[CLS] [MOD] left [CUT] body [CUT] right` |
| `nocut` | no [CUT] tokens; left + body + right tokenized as one string |
| `eqmask` | [CUT] masked at `--mlm_probability` instead of `--cut_mlm_probability` |
| `noflank` | no flanks: `[CLS] [MOD] [CUT] body [CUT]` (use `--seq_len 512 --min_flank 0`) |
| `nomod` | modality token replaced by the same placeholder (`[SEP]`) for every sample |

Non-default runs get `_ABL<value>` appended to the run name.

## Eval metrics

Logged every `--eval_steps` (default 250) by the HF Trainer. Defined in `compute_metrics()` in `pretrain.py`.

| Metric | Meaning |
|---|---|
| `overall_accuracy` | MLM accuracy across all masked positions |
| `noncut_accuracy` | MLM accuracy on masked non-[CUT] tokens; comparable across ablations |
| `cut_precision`, `cut_recall`, `cut_mcc` | precision, recall, Matthews correlation for predicting [CUT] at masked positions |
| `accuracy_{1,5,10,20}_around_cut` | MLM accuracy on non-[CUT] tokens within N positions of a [CUT] |
| `token_len_{1..6}_precision`, `_recall` | macro precision/recall per byte-length class K of the target token; K=6 covers ≥6 |
| `avg_num_predicted_cut`, `avg_num_masked_cut` | mean count of predicted vs ground-truth [CUT] tokens per row; sanity check |

Paper Figure 2f reports `cut_recall ≈ 0.92`, `cut_mcc ≈ 0.58` at the end of pretraining.

## Resume

Checkpoints land at `$OUTPUT_ROOT/<run-id>/checkpoint-<step>`. Re-running the same command picks up the latest checkpoint via `transformers.trainer_utils.get_last_checkpoint`. A `COMPLETED` marker is written at the run-id level on clean exit; subsequent re-runs no-op.

The run-id template encodes every shape and recipe knob (`<run_tag>_L<L>_H<H>_D<dim>_FF<ffn>_BS<bs>_GA<ga>_SEQ<seq>_VOC<vocab>_LR<lr>_WD<wd>_SCHED_<sched>`). Changing any of these creates a new directory rather than overwriting the old run.
