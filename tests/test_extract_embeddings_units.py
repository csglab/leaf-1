"""Unit tests for leaf1.infer.extract_embeddings (no end-to-end model run)."""

import json

import h5py
import numpy as np
import pytest
import torch

from leaf1.infer import extract_embeddings as ee


def test_window_embed_dim_matches_paper():
    """13 * 768 = 9984 (3 segments * 3 windows + 4 special tokens)."""
    assert ee.NUM_WINDOWS == 3
    assert ee.EMBED_DIM == 768
    assert ee.NUM_BERT_LAYERS == 12
    assert ee.WINDOW_EMBED_DIM == 9984


def _make_dataset_info(target_layer=None, total_batches=2, batch_size=4, seed=42):
    return {
        "bed_file": "/tmp/example.bed",
        "sample_id": "sample_x",
        "modality": "CFDNA",
        "group": "healthy",
        "n_fragments_total": 1000,
        "n_fragments_sampled": 8,
        "bert_checkpoint": "/tmp/ckpt",
        "total_batches": total_batches,
        "seed": seed,
        "batch_size": batch_size,
        "target_layer": target_layer,
    }


def test_create_h5_structure_single_layer(tmp_path):
    h5_path = tmp_path / "out_single.h5"
    info = _make_dataset_info(target_layer=4)
    with h5py.File(h5_path, "w") as f:
        ee.create_h5_structure(f, n_samples=8, dataset_info=info)

    with h5py.File(h5_path, "r") as f:
        emb = f["embeddings/bert_window_embeddings"]
        assert emb.shape == (8, ee.WINDOW_EMBED_DIM)
        assert emb.dtype == np.float16
        assert f["embeddings"].attrs["num_layers"] == 1
        assert f["embeddings"].attrs["target_layer"] == 4
        assert f["embeddings"].attrs["num_windows"] == ee.NUM_WINDOWS
        assert f["embeddings"].attrs["embed_dim"] == ee.EMBED_DIM
        assert f["embeddings"].attrs["window_embed_dim"] == ee.WINDOW_EMBED_DIM


def test_create_h5_structure_all_layers(tmp_path):
    h5_path = tmp_path / "out_all.h5"
    info = _make_dataset_info(target_layer=None)
    with h5py.File(h5_path, "w") as f:
        ee.create_h5_structure(f, n_samples=4, dataset_info=info)

    with h5py.File(h5_path, "r") as f:
        emb = f["embeddings/bert_window_embeddings"]
        assert emb.shape == (4, ee.NUM_BERT_LAYERS + 1, ee.WINDOW_EMBED_DIM)
        assert f["embeddings"].attrs["num_layers"] == ee.NUM_BERT_LAYERS + 1


def test_create_h5_structure_segment_offsets_match_layout(tmp_path):
    """segment_offsets attr documents the exact column layout the pool head writes."""
    h5_path = tmp_path / "offsets.h5"
    with h5py.File(h5_path, "w") as f:
        ee.create_h5_structure(f, n_samples=1, dataset_info=_make_dataset_info())

    with h5py.File(h5_path, "r") as f:
        offsets = json.loads(f["embeddings"].attrs["segment_offsets"])

    D = ee.EMBED_DIM
    expected = {
        "left": 0, "fragment": 3 * D, "right": 6 * D,
        "cut1": 9 * D, "cut2": 10 * D, "cls": 11 * D, "mod": 12 * D,
    }
    assert offsets == expected


def test_create_h5_structure_fragment_info_groups(tmp_path):
    """fragment_info group has the expected columns and dtypes."""
    h5_path = tmp_path / "frag_info.h5"
    with h5py.File(h5_path, "w") as f:
        ee.create_h5_structure(f, n_samples=3, dataset_info=_make_dataset_info())

    with h5py.File(h5_path, "r") as f:
        fi = f["fragment_info"]
        for col in ("chromosome", "fragment_start", "fragment_end",
                    "adjusted_start", "adjusted_end", "line_number",
                    "reverse_complemented", "n_token_left", "n_token_frag",
                    "n_token_right", "n_freq"):
            assert col in fi
        assert fi["fragment_start"].dtype == np.int64
        assert fi["reverse_complemented"].dtype == np.bool_
        assert fi["n_token_left"].dtype == np.int32
        assert fi["n_freq"].dtype == np.float32


def test_create_h5_structure_progress_attrs(tmp_path):
    h5_path = tmp_path / "progress.h5"
    info = _make_dataset_info(total_batches=5, batch_size=16, seed=99)
    with h5py.File(h5_path, "w") as f:
        ee.create_h5_structure(f, n_samples=4, dataset_info=info)

    with h5py.File(h5_path, "r") as f:
        p = f["progress"]
        assert p.attrs["current_batch_idx"] == -1
        assert p.attrs["total_batches"] == 5
        assert p.attrs["batch_size"] == 16
        assert p.attrs["seed"] == 99
        assert p.attrs["processed_count"] == 0
        assert bool(p.attrs["completed"]) is False


def test_store_batch_data_persists_fragment_info(tmp_path):
    """store_batch_data writes embeddings and fragment_info into the right rows."""
    h5_path = tmp_path / "store.h5"
    n_samples = 4
    info = _make_dataset_info(target_layer=4)
    with h5py.File(h5_path, "w") as f:
        ee.create_h5_structure(f, n_samples=n_samples, dataset_info=info)

    embeddings = np.random.RandomState(0).randn(2, ee.WINDOW_EMBED_DIM).astype(np.float16)
    meta = {
        "chromosome": ["chr1", "chr2"],
        "fragment_start": torch.tensor([100, 500]),
        "fragment_end": torch.tensor([250, 700]),
        "adjusted_start": torch.tensor([50, 400]),
        "adjusted_end": torch.tensor([300, 800]),
        "line_number": torch.tensor([0, 1]),
        "reverse_complemented": torch.tensor([False, True]),
        "n_token_left": torch.tensor([20, 30]),
        "n_token_frag": torch.tensor([15, 25]),
        "n_token_right": torch.tensor([20, 30]),
        "n_freq": torch.tensor([0.1, 0.2]),
    }
    with h5py.File(h5_path, "r+") as f:
        ee.store_batch_data(f, embeddings, meta, start_idx=0)

    with h5py.File(h5_path, "r") as f:
        np.testing.assert_array_equal(
            f["embeddings/bert_window_embeddings"][:2], embeddings
        )
        # Rows 2-3 were not written; check they remain at the dataset's default (zero).
        np.testing.assert_array_equal(
            f["embeddings/bert_window_embeddings"][2:],
            np.zeros((2, ee.WINDOW_EMBED_DIM), dtype=np.float16),
        )
        assert f["fragment_info/chromosome"][0].decode() == "chr1"
        assert f["fragment_info/chromosome"][1].decode() == "chr2"
        assert f["fragment_info/fragment_start"][1] == 500
        assert bool(f["fragment_info/reverse_complemented"][1]) is True
        assert f["fragment_info/n_token_frag"][0] == 15
        assert abs(f["fragment_info/n_freq"][1] - 0.2) < 1e-6


# ---------- process_batch with a tiny DistilBert -------------------------

def _build_tiny_distilbert(tokenizer, n_layers=2, n_heads=4, dim=64, ffn=128, seq_len=64):
    from transformers import DistilBertConfig, DistilBertForMaskedLM

    cfg = DistilBertConfig(
        model_max_length=seq_len,
        max_position_embeddings=seq_len,
        vocab_size=tokenizer.vocab_size,
        n_layers=n_layers,
        n_heads=n_heads,
        dim=dim,
        hidden_dim=ffn,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        sep_token_id=tokenizer.sep_token_id,
    )
    return DistilBertForMaskedLM(cfg).eval()


def _make_inference_batch(tokenizer, batch_size=2, seq_len=64,
                          cut_left_pos=10, frag_token_len=20):
    """Build an input batch matching the inference contract: [CLS][MOD]...[CUT]body[CUT]..."""
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    cut_id = tokenizer.convert_tokens_to_ids("[CUT]")
    mod_id = tokenizer.convert_tokens_to_ids("[CFDNA]")
    generic = max(cls_id, sep_id, pad_id, cut_id, mod_id) + 5

    # In extract_embeddings.process_batch:
    #   start_pos = n_token_left + 2  (first [CUT] index, accounting for [CLS] + [MOD])
    #   end_pos   = start_pos + n_token_frag + 1  (second [CUT] index)
    # So if n_token_left = cut_left_pos - 2 and n_token_frag = frag_token_len, the [CUT]s
    # land at cut_left_pos and cut_left_pos + frag_token_len + 1.
    n_token_left = cut_left_pos - 2
    n_token_frag = frag_token_len
    n_token_right = seq_len - (cut_left_pos + frag_token_len + 1) - 2  # leave room for SEP

    input_ids = torch.full((batch_size, seq_len), generic, dtype=torch.long)
    input_ids[:, 0] = cls_id
    input_ids[:, 1] = mod_id
    input_ids[:, cut_left_pos] = cut_id
    input_ids[:, cut_left_pos + frag_token_len + 1] = cut_id
    input_ids[:, -1] = sep_id

    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "n_token_left": torch.full((batch_size,), n_token_left, dtype=torch.long),
        "n_token_frag": torch.full((batch_size,), n_token_frag, dtype=torch.long),
    }


def test_process_batch_single_layer_output_shape(tokenizer):
    seq_len = 64
    model = _build_tiny_distilbert(tokenizer, dim=ee.EMBED_DIM, seq_len=seq_len)
    batch = _make_inference_batch(tokenizer, batch_size=3, seq_len=seq_len)
    out = ee.process_batch(model, batch, device=torch.device("cpu"), target_layer=2)
    assert out.shape == (3, ee.WINDOW_EMBED_DIM)
    assert out.dtype == np.float16
    assert np.isfinite(out).all()


def test_process_batch_all_layers_output_shape(tokenizer):
    """Output shape now reflects the actual model size, not the paper default."""
    seq_len = 64
    n_layers = 2
    model = _build_tiny_distilbert(tokenizer, n_layers=n_layers, dim=ee.EMBED_DIM, seq_len=seq_len)
    batch = _make_inference_batch(tokenizer, batch_size=2, seq_len=seq_len)
    out = ee.process_batch(model, batch, device=torch.device("cpu"), target_layer=None)
    # Layout: (batch, n_layers + 1, WINDOW_EMBED_DIM) - one slot per (embedding + each transformer layer).
    assert out.shape == (2, n_layers + 1, ee.WINDOW_EMBED_DIM)
    assert np.isfinite(out).all()


def test_process_batch_uses_cut_positions(tokenizer):
    """The slice at [cut1] column comes from the embedding at start_pos = n_token_left + 2."""
    seq_len = 64
    model = _build_tiny_distilbert(tokenizer, dim=ee.EMBED_DIM, seq_len=seq_len)
    batch = _make_inference_batch(tokenizer, batch_size=1, seq_len=seq_len,
                                  cut_left_pos=10, frag_token_len=20)
    out_full = ee.process_batch(model, batch, device=torch.device("cpu"), target_layer=0)
    # Layout: [left(3D), frag(3D), right(3D), CUT1(D), CUT2(D), CLS(D), MOD(D)].
    D = ee.EMBED_DIM
    cut1_slice = out_full[0, 9 * D : 10 * D]
    cut2_slice = out_full[0, 10 * D : 11 * D]
    cls_slice = out_full[0, 11 * D : 12 * D]
    mod_slice = out_full[0, 12 * D : 13 * D]
    # Sanity: each slice is finite and non-trivial.
    for sl, name in [(cut1_slice, "CUT1"), (cut2_slice, "CUT2"),
                     (cls_slice, "CLS"), (mod_slice, "MOD")]:
        assert np.isfinite(sl).all(), f"{name} slice has non-finite values"
        assert not np.allclose(sl, 0), f"{name} slice is all zeros"


def test_window_embed_dim_constant_matches_formula():
    """(3 segments * NUM_WINDOWS + 4 special tokens) * EMBED_DIM."""
    expected = (3 * ee.NUM_WINDOWS + 4) * ee.EMBED_DIM
    assert ee.WINDOW_EMBED_DIM == expected


def test_process_batch_rejects_out_of_range_target_layer(tokenizer):
    """target_layer > num_layers must raise rather than silently NameError."""
    seq_len = 64
    model = _build_tiny_distilbert(tokenizer, n_layers=2, dim=ee.EMBED_DIM, seq_len=seq_len)
    batch = _make_inference_batch(tokenizer, batch_size=1, seq_len=seq_len)
    with pytest.raises(ValueError, match="target_layer"):
        ee.process_batch(model, batch, device=torch.device("cpu"), target_layer=99)
    with pytest.raises(ValueError, match="target_layer"):
        ee.process_batch(model, batch, device=torch.device("cpu"), target_layer=-1)


def test_window_embed_dim_for_helper():
    """Generic helper should match the paper constant when called with 768."""
    assert ee.window_embed_dim_for(768) == 9984
    assert ee.window_embed_dim_for(256) == (3 * 3 + 4) * 256  # 3328
    assert ee.window_embed_dim_for(128) == (3 * 3 + 4) * 128  # 1664


def test_create_h5_structure_supports_non_default_dim(tmp_path):
    """create_h5_structure with config-derived dim should produce matching shapes."""
    embed_dim = 256
    num_bert_layers = 6
    expected_window = ee.window_embed_dim_for(embed_dim)

    h5_path = tmp_path / "out_scaling.h5"
    info = _make_dataset_info(target_layer=None)
    with h5py.File(h5_path, "w") as f:
        ee.create_h5_structure(
            f, n_samples=4, dataset_info=info,
            embed_dim=embed_dim, num_bert_layers=num_bert_layers,
        )

    with h5py.File(h5_path, "r") as f:
        emb = f["embeddings/bert_window_embeddings"]
        assert emb.shape == (4, num_bert_layers + 1, expected_window)
        assert f["embeddings"].attrs["embed_dim"] == embed_dim
        assert f["embeddings"].attrs["num_layers"] == num_bert_layers + 1
        offsets = json.loads(f["embeddings"].attrs["segment_offsets"])
        # Each special token offset shifts by the new embed_dim.
        assert offsets["cut1"] == 9 * embed_dim
        assert offsets["mod"] == 12 * embed_dim


def test_process_batch_uses_model_config_dim(tokenizer):
    """Non-default dim should produce window_embed_dim_for(dim)-wide output."""
    seq_len = 64
    embed_dim = 384  # off-paper to make sure nothing's hardcoded
    model = _build_tiny_distilbert(tokenizer, n_layers=2, n_heads=4, dim=embed_dim,
                                    ffn=embed_dim * 2, seq_len=seq_len)
    batch = _make_inference_batch(tokenizer, batch_size=2, seq_len=seq_len)
    out = ee.process_batch(model, batch, device=torch.device("cpu"), target_layer=1)
    assert out.shape == (2, ee.window_embed_dim_for(embed_dim))
