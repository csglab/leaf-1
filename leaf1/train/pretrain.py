#!/usr/bin/env python3
"""LEAF-1 pretraining entry point.

Paper-canonical run is the 12-layer, 12-head, 768-dim DistilBERT-style encoder
(~90M params) at sequence length 2048 with 15% base / 30% [CUT] joint-boundary
MLM masking. This script parameterizes all of those for scaling/ablation use as
well.

Fail-safe behavior:
  - If --output_dir contains a COMPLETED marker, exits immediately.
  - If --output_dir has HF checkpoints, auto-resumes from the latest one.
  - Writes COMPLETED on successful finish (main process only).
"""

import argparse
import inspect
import logging
import os
import random

import evaluate
import numpy as np
import torch
from sklearn.metrics import matthews_corrcoef, precision_recall_fscore_support
from transformers import (
    DistilBertConfig,
    DistilBertForMaskedLM,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import logging as transformers_logging


# `Trainer.__init__` renamed `tokenizer` to `processing_class` in transformers
# 4.41. Pick the right kwarg name at import time so both versions work.
_TRAINER_INIT_PARAMS = set(inspect.signature(Trainer.__init__).parameters)
_TRAINER_TOKENIZER_KWARG = (
    "processing_class" if "processing_class" in _TRAINER_INIT_PARAMS else "tokenizer"
)
_TRAINING_ARGS_PARAMS = set(inspect.signature(TrainingArguments.__init__).parameters)
_EVAL_STRATEGY_KWARG = "eval_strategy" if "eval_strategy" in _TRAINING_ARGS_PARAMS else "evaluation_strategy"

from leaf1.common.collator import DataCollatorForDNAFragmentModelingNoDecay
from leaf1.common.dna_fragments_dataset import DNAFragmentDataset
from leaf1.common.pytorch_dataset import TrainingDNAFragmentDataset


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Tokenizer ships inside the package so non-editable installs find it.
DEFAULT_TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "data", "tokenizer.json")
# Reference FASTA is too large (3.1 GB) to bundle; user supplies via --ref_fa
# or the repo's assets/ for editable installs.
DEFAULT_REF_FA = os.path.join(REPO_ROOT, "assets", "refseq.genome.fa")


def parse_args():
    p = argparse.ArgumentParser(description="LEAF-1 fragment-MLM pretraining")
    # Model shape (paper-canonical: 12 layers, 12 heads, FFN 4x).
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_heads", type=int, default=12,
                   help="Number of attention heads; dim = n_heads * 64")
    p.add_argument("--ffn_mult", type=int, default=4,
                   help="FFN size multiplier (hidden_dim = ffn_mult * dim)")
    # Data / sequence.
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--vocab_size", type=int, default=4099)
    p.add_argument("--min_flank", type=int, default=100)
    # Train recipe.
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--grad_accum", type=int, default=32,
                   help="Gradient accumulation steps (per process)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=40000,
                   help="Optimizer steps; overrides epochs if > 0")
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--lr_scheduler", type=str, default="linear")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--cut_mlm_probability", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_worker", type=int, default=12)
    # IO. Train/val manifests must be supplied by the user.
    p.add_argument("--train_csv", type=str, required=True,
                   help="CSV manifest: sample_id,modality,bed_file,n_fragment,group")
    p.add_argument("--val_csv", type=str, required=True,
                   help="CSV manifest in the same format as --train_csv")
    p.add_argument("--ref_fa", type=str, default=DEFAULT_REF_FA,
                   help="Reference FASTA (with .fai sibling). Default: assets/refseq.genome.fa")
    p.add_argument("--tokenizer_file", type=str, default=DEFAULT_TOKENIZER,
                   help="Tokenizer JSON. Default: packaged leaf1/data/tokenizer.json")
    p.add_argument("--output_root", type=str, required=True,
                   help="Checkpoints + COMPLETED marker write here under <run-id>/")
    p.add_argument("--log_root", type=str, required=True,
                   help="HF Trainer logging dir parent")
    p.add_argument("--run_tag", type=str, default="LEAF1",
                   help="Tag appended to run name")
    # Logging / eval.
    p.add_argument("--report_to", type=str, default="none",
                   help='Set to "wandb" if W&B is configured')
    p.add_argument("--eval_steps", type=int, default=250)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--save_total_limit", type=int, default=5)
    p.add_argument("--logging_steps", type=int, default=1)
    # Mixed precision.
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true", help="Recommended on H100/A100")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    return p.parse_args()


def build_compute_metrics(tokenizer):
    """Return MLM accuracy + [CUT]-token precision/recall/MCC + length-class metrics."""
    metric = evaluate.load("accuracy")
    cut_token_id = tokenizer.convert_tokens_to_ids('[CUT]')
    windows_sizes = [1, 5, 10, 20]

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        num_samples = preds.shape[0]
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)

        general_mask = labels != -100
        overall_acc = metric.compute(
            predictions=preds[general_mask], references=labels[general_mask]
        )['accuracy']

        cut_mask = (labels == cut_token_id)
        cut_pred = preds[general_mask] == cut_token_id
        cut_true = labels[general_mask] == cut_token_id

        if cut_mask.sum() > 0:
            cut_precision, cut_recall, _, _ = precision_recall_fscore_support(
                cut_true, cut_pred, average='binary'
            )
            cut_mcc = matthews_corrcoef(cut_true, cut_pred)
        else:
            cut_precision = cut_recall = cut_mcc = None

        avg_num_predicted_cut = cut_pred.sum() / num_samples
        avg_num_masked_cut = cut_true.sum() / num_samples

        window_accuracies = {}
        for window_size in windows_sizes:
            extended_mask = np.zeros_like(cut_mask)
            for idx in np.where(cut_mask)[0]:
                start = max(0, idx - window_size)
                end = min(len(cut_mask), idx + window_size + 1)
                extended_mask[start:idx] = True
                extended_mask[idx + 1:end] = True

            extended_mask = extended_mask & general_mask
            if extended_mask.sum() > 0:
                window_acc = metric.compute(
                    predictions=preds[extended_mask], references=labels[extended_mask]
                )['accuracy']
            else:
                window_acc = None
            window_accuracies[f"accuracy_{window_size}_around_cut"] = window_acc

        length_class_metrics = {}
        tokenizer_vocab = tokenizer.get_vocab()
        token_lengths = {
            token_id: len(token)
            for token, token_id in tokenizer_vocab.items() if token_id > cut_token_id
        }

        for token_length in range(1, 7):
            if token_length == 6:
                length_mask = np.isin(
                    labels, [tid for tid, tlen in token_lengths.items() if tlen > 5]
                )
            else:
                length_mask = np.isin(
                    labels, [tid for tid, tlen in token_lengths.items() if tlen == token_length]
                )

            length_mask = length_mask & general_mask
            if length_mask.sum() > 0:
                class_pred = preds[length_mask]
                class_true = labels[length_mask]
                precision, recall, _, _ = precision_recall_fscore_support(
                    class_true, class_pred, average='macro', zero_division=0
                )
            else:
                precision = recall = None

            length_class_metrics[f'token_len_{token_length}_precision'] = precision
            length_class_metrics[f'token_len_{token_length}_recall'] = recall

        return {
            "overall_accuracy": overall_acc,
            "cut_precision": cut_precision,
            "cut_recall": cut_recall,
            "cut_mcc": cut_mcc,
            "avg_num_predicted_cut": avg_num_predicted_cut,
            "avg_num_masked_cut": avg_num_masked_cut,
            **length_class_metrics,
            **window_accuracies,
        }

    return compute_metrics, preprocess_logits_for_metrics


def _tf32_supported() -> bool:
    """TF32 requires CUDA + an Ampere (compute capability 8.0) or newer GPU."""
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 8


def main():
    args = parse_args()

    HEAD_DIM = 64
    dim = args.n_heads * HEAD_DIM
    hidden_dim = args.ffn_mult * dim

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    n_fragment_per_training_file_read = args.batch_size * 4
    n_fragment_per_validation_file = 100

    train_data_args = {
        "n_fragment_file": args.train_csv,
        "ref_genome_file": args.ref_fa,
        "tokenizer_file": args.tokenizer_file,
        "max_token_length": args.seq_len,
        "min_flank_token_length": args.min_flank,
        "seed": args.seed,
        "n_fragments_per_iteration": n_fragment_per_training_file_read,
        "return_token_type_ids": False,
        "return_special_tokens_mask": True,
        "n_shards": args.n_worker,
    }

    validation_data_args = {
        "n_fragment_file": args.val_csv,
        "ref_genome_file": args.ref_fa,
        "tokenizer_file": args.tokenizer_file,
        "max_token_length": args.seq_len,
        "min_flank_token_length": args.min_flank,
        "seed": args.seed,
        "n_fragment_per_file": n_fragment_per_validation_file,
        "reshuffle": True,
        "return_token_type_ids": False,
        "return_special_tokens_mask": True,
    }

    model_id = (
        f"{args.run_tag}_L{args.n_layers}_H{args.n_heads}_D{dim}_FF{args.ffn_mult}"
        f"_BS{args.batch_size}_GA{args.grad_accum}_SEQ{args.seq_len}_VOC{args.vocab_size}"
        f"_LR{args.learning_rate}_WD{args.weight_decay}_SCHED_{args.lr_scheduler}"
    )
    out_dir = os.path.join(args.output_root, model_id)
    log_dir = os.path.join(args.log_root, model_id)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    completed_marker = os.path.join(out_dir, "COMPLETED")
    if os.path.exists(completed_marker):
        print(f"[FAIL-SAFE] COMPLETED marker found at {completed_marker}. Exiting.")
        return

    last_ckpt = get_last_checkpoint(out_dir)
    if last_ckpt is not None:
        print(f"[FAIL-SAFE] Found existing checkpoint: {last_ckpt}. Will resume from it.")

    train_dataset = TrainingDNAFragmentDataset(**train_data_args)
    validation_dataset = DNAFragmentDataset(**validation_data_args)

    data_collator = DataCollatorForDNAFragmentModelingNoDecay(
        tokenizer=train_dataset.tokenizer,
        mlm_probability=args.mlm_probability,
        cut_token_probability=args.cut_mlm_probability,
    )

    transformers_logging.set_verbosity_info()
    logging.basicConfig(level=logging.INFO)

    compute_metrics, preprocess_logits_for_metrics = build_compute_metrics(train_dataset.tokenizer)

    model_config = DistilBertConfig(
        model_max_length=args.seq_len,
        max_position_embeddings=args.seq_len,
        vocab_size=args.vocab_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dim=dim,
        hidden_dim=hidden_dim,
        bos_token_id=train_dataset.tokenizer.bos_token_id,
        eos_token_id=train_dataset.tokenizer.eos_token_id,
        pad_token_id=train_dataset.tokenizer.pad_token_id,
        sep_token_id=train_dataset.tokenizer.sep_token_id,
    )
    model = DistilBertForMaskedLM(model_config).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_sequences_per_update = args.batch_size * world_size * args.grad_accum
    tokens_per_update = effective_sequences_per_update * args.seq_len
    total_updates = args.max_steps
    total_sequences = effective_sequences_per_update * total_updates
    total_tokens = tokens_per_update * total_updates

    print("========== RUN CONFIG ==========")
    print(f"Model params: {n_params:,}")
    print(f"Layers: {args.n_layers} | Heads: {args.n_heads} | dim: {dim} | FFN: {hidden_dim}")
    print(f"World size (GPUs): {world_size}")
    print(f"Per-device batch: {args.batch_size} | Grad accumulation: {args.grad_accum}")
    print(f"Effective sequences/update: {effective_sequences_per_update:,}")
    print(f"Tokens/update: {tokens_per_update:,}")
    print(f"Max optimizer updates: {total_updates:,}")
    print(f"Total sequences (approx): {total_sequences:,}")
    print(f"Total tokens (approx): {total_tokens:,}")
    print(f"Mixed precision: {'bf16' if args.bf16 else 'fp16' if args.fp16 else 'fp32'}")
    print(f"Output dir: {out_dir}")
    if last_ckpt is not None:
        print(f"Resuming from checkpoint: {last_ckpt}")
    print("================================")

    training_args = TrainingArguments(
        output_dir=out_dir,
        report_to=args.report_to,
        run_name=model_id,
        max_steps=args.max_steps,
        num_train_epochs=args.epochs,
        save_safetensors=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_dir=log_dir,
        logging_strategy='steps',
        logging_first_step=True,
        logging_steps=args.logging_steps,
        log_level="debug",
        **{_EVAL_STRATEGY_KWARG: 'steps'},
        eval_steps=args.eval_steps,
        eval_delay=0,
        save_strategy='steps',
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=args.warmup_steps,
        disable_tqdm=False,
        weight_decay=args.weight_decay,
        dataloader_pin_memory=True,
        dataloader_num_workers=args.n_worker,
        dataloader_persistent_workers=False,
        # don't replay the dataloader on resume: it's bedtools-bound, so skipping N batches takes
        # hours, and the corpus is randomly sampled anyway
        ignore_data_skip=True,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        ddp_find_unused_parameters=False,
        max_grad_norm=args.max_grad_norm,
        tf32=_tf32_supported(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        **{_TRAINER_TOKENIZER_KWARG: train_dataset.tokenizer},
    )

    trainer.train(resume_from_checkpoint=last_ckpt)

    trainer.save_model()
    if trainer.is_world_process_zero:
        with open(completed_marker, "w") as f:
            f.write("ok\n")
        print(f"[FAIL-SAFE] Wrote COMPLETED marker: {completed_marker}")


if __name__ == "__main__":
    main()
