"""End-to-end smoke test: build a tiny DistilBert, run evaluate_bed_file, inspect H5."""

import h5py
import numpy as np
import pytest
import torch

from leaf1.infer import extract_embeddings as ee


@pytest.fixture(scope="module")
def tiny_bert_checkpoint(tokenizer, tmp_path_factory):
    """Persist a 2-layer DistilBertForMaskedLM with EMBED_DIM=768 to a temp dir."""
    from transformers import DistilBertConfig, DistilBertForMaskedLM

    seq_len = 128
    cfg = DistilBertConfig(
        model_max_length=seq_len,
        max_position_embeddings=seq_len,
        vocab_size=tokenizer.vocab_size,
        n_layers=2,
        n_heads=12,
        dim=ee.EMBED_DIM,    # 768; required because process_batch hardcodes WINDOW_EMBED_DIM
        hidden_dim=512,      # small FFN for speed
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
    )
    model = DistilBertForMaskedLM(cfg)
    out_dir = tmp_path_factory.mktemp("tiny_ckpt")
    model.save_pretrained(out_dir)
    return str(out_dir)


def test_evaluate_bed_file_single_layer_writes_complete_h5(
    needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest,
    tiny_bert_checkpoint, tmp_path,
):
    """End-to-end: extract layer-4 embeddings for all 6 fragments in the tiny manifest."""
    output_dir = tmp_path / "embed_out"

    ee.evaluate_bed_file(
        bert_checkpoint=tiny_bert_checkpoint,
        n_fragment_file=str(tiny_manifest),
        n_fragment_per_file=4,        # smaller than the 6 records to exercise sampling
        index=0,
        output_dir=str(output_dir),
        tokenizer_file=str(tokenizer_path),
        ref_genome_file=str(tiny_fa),
        batch_size=2,
        seed=42,
        target_layer=2,               # tiny model only has 2 layers
        max_token_length=128,
        min_flank_token_length=8,
    )

    h5_path = output_dir / "h5_files" / "CFDNA_sample_x.h5"
    assert h5_path.exists()

    with h5py.File(h5_path, "r") as f:
        # Progress: completion + counts.
        assert bool(f["progress"].attrs["completed"]) is True
        assert f["progress"].attrs["processed_count"] == 4
        # Embeddings: correct shape and dtype.
        emb = f["embeddings/bert_window_embeddings"][:]
        assert emb.shape == (4, ee.WINDOW_EMBED_DIM)
        assert emb.dtype == np.float16
        assert np.isfinite(emb).all()
        # Each fragment should produce a non-zero embedding row.
        assert not np.allclose(emb, 0)
        # Fragment info: chromosomes are valid.
        chrs = [c.decode() for c in f["fragment_info/chromosome"][:]]
        assert set(chrs).issubset({"chr1", "chr2"})
        # Metadata: bert_checkpoint and seed round-tripped.
        assert f["metadata"].attrs["seed"] == 42
        assert f["metadata"].attrs["batch_size"] == 2


def test_evaluate_bed_file_resume_detects_seed_mismatch(
    needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest,
    tiny_bert_checkpoint, tmp_path,
):
    """Re-running on a partial H5 with a different seed should refuse."""
    output_dir = tmp_path / "embed_seed_resume"

    # First run finishes.
    ee.evaluate_bed_file(
        bert_checkpoint=tiny_bert_checkpoint,
        n_fragment_file=str(tiny_manifest),
        n_fragment_per_file=4,
        index=0,
        output_dir=str(output_dir),
        tokenizer_file=str(tokenizer_path),
        ref_genome_file=str(tiny_fa),
        batch_size=2,
        seed=42,
        target_layer=2,
        max_token_length=128,
        min_flank_token_length=8,
    )

    h5_path = output_dir / "h5_files" / "CFDNA_sample_x.h5"
    # A fully-completed H5 raises FileExistsError on re-run with same params.
    with pytest.raises(FileExistsError):
        ee.evaluate_bed_file(
            bert_checkpoint=tiny_bert_checkpoint,
            n_fragment_file=str(tiny_manifest),
            n_fragment_per_file=4,
            index=0,
            output_dir=str(output_dir),
            tokenizer_file=str(tokenizer_path),
            ref_genome_file=str(tiny_fa),
            batch_size=2,
            seed=42,
            target_layer=2,
            max_token_length=128,
            min_flank_token_length=8,
        )

    # Mark progress.completed=False so the resume path is taken, then run with a
    # different seed and check it refuses.
    import h5py as _h5py
    with _h5py.File(h5_path, "r+") as f:
        f["progress"].attrs["completed"] = False
        f["progress"].attrs["current_batch_idx"] = 0  # pretend only 1 batch was done

    with pytest.raises(ValueError, match="Seed mismatch"):
        ee.evaluate_bed_file(
            bert_checkpoint=tiny_bert_checkpoint,
            n_fragment_file=str(tiny_manifest),
            n_fragment_per_file=4,
            index=0,
            output_dir=str(output_dir),
            tokenizer_file=str(tokenizer_path),
            ref_genome_file=str(tiny_fa),
            batch_size=2,
            seed=99,           # different seed
            target_layer=2,
            max_token_length=128,
            min_flank_token_length=8,
        )


def test_evaluate_bed_file_invalid_index_raises(
    needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest,
    tiny_bert_checkpoint, tmp_path,
):
    output_dir = tmp_path / "embed_bad_idx"
    with pytest.raises(ValueError, match="out of bounds"):
        ee.evaluate_bed_file(
            bert_checkpoint=tiny_bert_checkpoint,
            n_fragment_file=str(tiny_manifest),
            n_fragment_per_file=4,
            index=999,
            output_dir=str(output_dir),
            tokenizer_file=str(tokenizer_path),
            ref_genome_file=str(tiny_fa),
            batch_size=2,
            seed=42,
            target_layer=2,
            max_token_length=128,
            min_flank_token_length=8,
        )
