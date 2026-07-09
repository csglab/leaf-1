"""Tests for leaf1.downstream.mean_pool.aggregate."""

import h5py
import numpy as np
import pytest

from leaf1.downstream.mean_pool import aggregate as agg


# ---------- _reshape_to_segments ------------------------------------------

def test_reshape_2d_flat():
    """(n, 9984) should reshape to (n, 13, 768)."""
    arr = np.arange(2 * 9984, dtype=np.float32).reshape(2, 9984)
    out = agg._reshape_to_segments(arr)
    assert out.shape == (2, 13, 768)
    # Column 0 of segment 0 must be column 0 of the flat input.
    np.testing.assert_array_equal(out[:, 0, :], arr[:, :768])


def test_reshape_3d_segment_major_passthrough():
    """(n, 13, 768) should come back unchanged."""
    arr = np.random.default_rng(0).standard_normal((4, 13, 768), dtype=np.float32)
    out = agg._reshape_to_segments(arr)
    assert out is arr or np.array_equal(out, arr)
    assert out.shape == (4, 13, 768)


def test_reshape_3d_multi_layer_flat_picks_layer():
    """(n, L, 9984) should pick layer_idx and reshape to (n, 13, 768)."""
    arr = np.zeros((1, 5, 9984), dtype=np.float32)
    arr[0, 3, :768] = 42.0   # layer 3, segment 0, first chunk
    out = agg._reshape_to_segments(arr, layer_idx=3)
    assert out.shape == (1, 13, 768)
    np.testing.assert_array_equal(out[0, 0, :], np.full(768, 42.0, dtype=np.float32))


def test_reshape_4d_multi_layer_segment_major_picks_layer():
    """(n, L, 13, 768) should pick layer_idx along axis 1."""
    arr = np.zeros((2, 5, 13, 768), dtype=np.float32)
    arr[:, 2, :, :] = 7.0
    out = agg._reshape_to_segments(arr, layer_idx=2)
    assert out.shape == (2, 13, 768)
    np.testing.assert_array_equal(out, np.full((2, 13, 768), 7.0, dtype=np.float32))


def test_reshape_clamps_layer_idx_to_max():
    """layer_idx beyond the last layer should clamp to the last."""
    arr = np.arange(1 * 3 * 9984, dtype=np.float32).reshape(1, 3, 9984)
    out = agg._reshape_to_segments(arr, layer_idx=999)
    expected_last_layer = arr[:, -1, :].reshape(1, 13, 768)
    np.testing.assert_array_equal(out, expected_last_layer)


def test_reshape_raises_on_unrecognised_shape():
    with pytest.raises(ValueError, match="Unrecognised embedding shape"):
        agg._reshape_to_segments(np.zeros((4, 100), dtype=np.float32))


# ---------- aggregate end-to-end ------------------------------------------

def test_aggregate_writes_segment_major_output(downstream_h5_dir, tmp_path):
    """End-to-end: 8 per-sample H5s -> (8, 13, 768) aggregated H5."""
    out_path = tmp_path / "agg.h5"
    n = agg.aggregate(
        input_dir=str(downstream_h5_dir.parent),  # parent contains h5_files/
        output=str(out_path),
        model_name="leaf1_test",
        workers=1,
        layer_idx=4,
    )
    assert n == 8
    with h5py.File(out_path, "r") as f:
        emb = f["embeddings"][:]
        assert emb.shape == (8, 13, 768)
        assert emb.dtype == np.float32
        assert np.isfinite(emb).all()

        sample_ids = [s.decode() for s in f["sample_ids"][:]]
        assert sample_ids == [f"sample_{i}" for i in range(8)]

        groups = [g.decode() for g in f["groups"][:]]
        assert groups == ["healthy"] * 4 + ["cancer"] * 4

        assert f.attrs["embed_dim"] == 768
        assert f.attrs["num_segments"] == 13
        assert f.attrs["num_samples"] == 8
        assert f.attrs["pooling"] == "mean_across_fragments"
        assert f.attrs["model_name"] == "leaf1_test"


def test_aggregate_skips_incomplete_h5(downstream_h5_dir, tmp_path):
    """Per-sample H5s with progress.completed=False should be skipped silently."""
    # Build a small dir with one complete + one incomplete copy.
    work_dir = tmp_path / "mixed" / "h5_files"
    work_dir.mkdir(parents=True)

    import shutil
    src_files = sorted(downstream_h5_dir.glob("*.h5"))
    shutil.copy(src_files[0], work_dir / src_files[0].name)
    incomplete = work_dir / "cfdna_incomplete.h5"
    shutil.copy(src_files[1], incomplete)
    with h5py.File(incomplete, "r+") as f:
        f["progress"].attrs["completed"] = False

    out = tmp_path / "mixed_agg.h5"
    n = agg.aggregate(
        input_dir=str(work_dir.parent),
        output=str(out),
        model_name="mixed",
        workers=1,
        layer_idx=4,
    )
    assert n == 1
    with h5py.File(out, "r") as f:
        assert f["embeddings"].shape[0] == 1


def test_aggregate_returns_zero_when_input_empty(tmp_path):
    """Empty input dir should return 0 and not crash."""
    empty = tmp_path / "empty" / "h5_files"
    empty.mkdir(parents=True)
    n = agg.aggregate(
        input_dir=str(empty.parent),
        output=str(tmp_path / "empty_agg.h5"),
        model_name="m",
        workers=1,
    )
    assert n == 0


def test_aggregate_finds_h5_files_in_root_too(downstream_h5_dir, tmp_path):
    """The aggregator should accept a dir that contains .h5 files directly
    (not under an h5_files/ subdir)."""
    flat_dir = tmp_path / "flat"
    flat_dir.mkdir()
    import shutil
    for h5 in sorted(downstream_h5_dir.glob("*.h5"))[:3]:
        shutil.copy(h5, flat_dir / h5.name)

    out = tmp_path / "flat_agg.h5"
    n = agg.aggregate(
        input_dir=str(flat_dir),
        output=str(out),
        model_name="flat",
        workers=1,
    )
    assert n == 3


# ---------- schema tolerance: published `layer4_reshaped/` format ---------

def _write_published_schema_h5(path, sample_id, group, n_frags=8, modality="CFDNA"):
    """Build a per-sample H5 in the lab's published `layer4_reshaped/CFDNA_EE*.h5`
    schema: top-level `embeddings` shape (n, 13, 768), top-level `completed=True`
    attr, metadata.attrs (no `progress` group, no nested `embeddings/`-group)."""
    rng = np.random.default_rng(hash(sample_id) & 0xFFFFFFFF)
    emb = rng.standard_normal((n_frags, 13, 768), dtype=np.float32).astype(np.float16)
    with h5py.File(path, "w") as f:
        f.create_dataset("embeddings", data=emb, dtype=np.float16)
        mg = f.create_group("metadata")
        mg.attrs["sample_id"] = sample_id
        mg.attrs["group"] = group
        mg.attrs["modality"] = modality
        f.attrs["completed"] = True
        f.attrs["layer_extracted"] = 4


def test_aggregate_reads_published_schema(tmp_path):
    """The schema-tolerant aggregator should also consume the lab's published
    per-sample H5 schema (top-level `embeddings` (n, 13, 768) + `completed`
    attr), not just leaf1-extract's nested `embeddings/bert_window_embeddings`."""
    pub_dir = tmp_path / "pub" / "h5_files"
    pub_dir.mkdir(parents=True)
    for i, group in enumerate(["healthy"] * 3 + ["cancer"] * 3):
        _write_published_schema_h5(
            pub_dir / f"CFDNA_sample_{i}.h5",
            sample_id=f"sample_{i}",
            group=group,
            n_frags=12,
        )

    out_path = tmp_path / "pub_agg.h5"
    n = agg.aggregate(
        input_dir=str(pub_dir.parent),
        output=str(out_path),
        model_name="leaf1_pub",
        workers=1,
        layer_idx=4,
    )
    assert n == 6
    with h5py.File(out_path, "r") as f:
        assert f["embeddings"].shape == (6, 13, 768)
        assert [s.decode() for s in f["sample_ids"][:]] == [f"sample_{i}" for i in range(6)]
        assert [g.decode() for g in f["groups"][:]] == ["healthy"] * 3 + ["cancer"] * 3


def test_aggregate_skips_published_schema_when_completed_false(tmp_path):
    """Published-schema files with `completed=False` (top-level attr) should be skipped."""
    pub_dir = tmp_path / "pub_incomp" / "h5_files"
    pub_dir.mkdir(parents=True)
    _write_published_schema_h5(pub_dir / "CFDNA_a.h5", "a", "healthy")
    # Write a second with completed=False.
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((4, 13, 768), dtype=np.float32).astype(np.float16)
    with h5py.File(pub_dir / "CFDNA_b.h5", "w") as f:
        f.create_dataset("embeddings", data=emb, dtype=np.float16)
        mg = f.create_group("metadata")
        mg.attrs["sample_id"] = "b"
        mg.attrs["group"] = "cancer"
        mg.attrs["modality"] = "CFDNA"
        f.attrs["completed"] = False

    n = agg.aggregate(
        input_dir=str(pub_dir.parent),
        output=str(tmp_path / "pub_incomp_agg.h5"),
        model_name="m",
        workers=1,
        layer_idx=4,
    )
    assert n == 1


def test_process_one_sample_handles_mixed_schemas(tmp_path, downstream_h5_dir):
    """One worker should produce identically-shaped results for both schemas."""
    # Pull one leaf1-extract-format sample (has progress/completed + nested embeddings).
    extract_h5 = sorted(downstream_h5_dir.glob("*.h5"))[0]
    # Build one published-schema sample with the same n_frags so we can compare shapes.
    pub_h5 = tmp_path / "CFDNA_pub.h5"
    _write_published_schema_h5(pub_h5, "pub_x", "healthy", n_frags=50)

    r1 = agg.process_one_sample((str(extract_h5), 4))
    r2 = agg.process_one_sample((str(pub_h5), 4))
    assert r1 is not None and r2 is not None
    assert r1["embedding"].shape == r2["embedding"].shape == (13, 768)
    assert r1["embedding"].dtype == r2["embedding"].dtype == np.float32
