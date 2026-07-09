"""Smoke test: run leaf1.train.pretrain.main() for 2 steps on the tiny manifest.

This exercises the full argparse → dataset build → Trainer.train() → COMPLETED marker
path. Uses a tiny DistilBert (2 layers / dim 128) and a batch of 1 to keep wall time low.
"""

import os
import sys
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def disable_wandb(monkeypatch):
    """Make sure wandb is off even if WANDB_API_KEY happens to be set."""
    monkeypatch.setenv("WANDB_DISABLED", "true")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")


def test_pretrain_main_runs_two_steps_and_writes_completed_marker(
    needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest, tmp_path, monkeypatch
):
    """End-to-end pretrain pipeline for 2 optimizer steps. Runs on CPU thanks to the
    runtime tf32 gate in pretrain._tf32_supported()."""
    output_root = tmp_path / "ckpt_root"
    log_root = tmp_path / "log_root"

    argv = [
        "pretrain",
        "--n_layers", "2",
        "--n_heads", "2",
        "--ffn_mult", "2",
        "--seq_len", "128",
        "--vocab_size", str(4099),
        "--min_flank", "8",
        "--batch_size", "1",
        "--grad_accum", "1",
        "--max_steps", "2",
        "--learning_rate", "1e-4",
        "--weight_decay", "0.0",
        "--lr_scheduler", "constant",
        "--warmup_steps", "0",
        "--seed", "0",
        "--n_worker", "1",
        "--train_csv", str(tiny_manifest),
        "--val_csv", str(tiny_manifest),
        "--ref_fa", str(tiny_fa),
        "--tokenizer_file", str(tokenizer_path),
        "--output_root", str(output_root),
        "--log_root", str(log_root),
        "--run_tag", "SMOKE",
        "--report_to", "none",
        "--eval_steps", "1000",
        "--save_steps", "10",
        "--save_total_limit", "1",
        "--logging_steps", "1",
        "--max_grad_norm", "1.0",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from leaf1.train import pretrain

    try:
        pretrain.main()
    except Exception:
        # Surface the failure with the captured stdout/stderr in the pytest report.
        raise

    # COMPLETED marker was written.
    run_dirs = list(output_root.iterdir())
    assert len(run_dirs) == 1, f"Expected one run dir, got {run_dirs}"
    run_dir = run_dirs[0]
    assert (run_dir / "COMPLETED").exists(), "COMPLETED marker not written"
    # save_model() wrote model files.
    model_files = list(run_dir.glob("model.safetensors")) + list(run_dir.glob("pytorch_model.bin"))
    assert model_files, f"No model weights saved in {run_dir}"
    # Log dir was populated.
    assert log_root.exists()


def test_pretrain_main_exits_early_if_completed_marker_present(
    needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest, tmp_path, monkeypatch, capsys
):
    """If <output_root>/<run-id>/COMPLETED exists, main() should bail without training."""
    output_root = tmp_path / "ckpt_root_2"
    log_root = tmp_path / "log_root_2"

    argv = [
        "pretrain",
        "--n_layers", "2",
        "--n_heads", "2",
        "--ffn_mult", "2",
        "--seq_len", "128",
        "--vocab_size", str(4099),
        "--min_flank", "8",
        "--batch_size", "1",
        "--grad_accum", "1",
        "--max_steps", "2",
        "--learning_rate", "1e-4",
        "--weight_decay", "0.0",
        "--lr_scheduler", "constant",
        "--warmup_steps", "0",
        "--seed", "0",
        "--n_worker", "1",
        "--train_csv", str(tiny_manifest),
        "--val_csv", str(tiny_manifest),
        "--ref_fa", str(tiny_fa),
        "--tokenizer_file", str(tokenizer_path),
        "--output_root", str(output_root),
        "--log_root", str(log_root),
        "--run_tag", "SMOKE2",
        "--report_to", "none",
        "--eval_steps", "1000",
        "--save_steps", "10",
        "--save_total_limit", "1",
        "--logging_steps", "1",
        "--max_grad_norm", "1.0",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    from leaf1.train import pretrain

    # Pre-create the run dir + COMPLETED marker so main() should fast-exit.
    # The run-id template (mirrored from pretrain.py):
    HEAD_DIM = 64
    dim = 2 * HEAD_DIM      # n_heads * 64
    run_id = (
        f"SMOKE2_L2_H2_D{dim}_FF2"
        f"_BS1_GA1_SEQ128_VOC4099"
        f"_LR0.0001_WD0.0_SCHED_constant"
    )
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "COMPLETED").write_text("ok\n")

    pretrain.main()

    out = capsys.readouterr().out
    assert "COMPLETED marker found" in out
    # No checkpoint subdirs should have been created.
    ckpts = list(run_dir.glob("checkpoint-*"))
    assert ckpts == []
