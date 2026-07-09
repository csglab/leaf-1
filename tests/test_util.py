"""Tests for leaf1/util/: bed_preprocess, reshape_layer, validate_extract."""

import gzip
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from leaf1.util import bed_preprocess as bp
from leaf1.util import reshape_layer as rl
from leaf1.util import validate_extract as ve


# ---------- bed_preprocess ------------------------------------------------

def _write_fragment_gz(path: Path, lines: list[str]):
    """Write a gzipped FinaleDB-style fragment file (5 tab-separated columns)."""
    with gzip.open(path, "wt") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def _write_blacklist_gz(path: Path, regions: list[tuple]):
    """Write a gzipped BED with the given (chr, start, end) regions."""
    with gzip.open(path, "wt") as f:
        for chrom, start, end in regions:
            f.write(f"{chrom}\t{start}\t{end}\n")


def _write_fai(path: Path, chrom_lengths: dict):
    with open(path, "w") as f:
        for chrom, length in chrom_lengths.items():
            # Format: name, length, offset, linebases, linewidth
            f.write(f"{chrom}\t{length}\t0\t{length}\t{length+1}\n")


def test_load_chromosome_allowlist(tmp_path):
    fai = tmp_path / "ref.fai"
    _write_fai(fai, {"chr1": 5000, "chr2": 3000})
    allowed = bp.load_chromosome_allowlist(str(fai))
    assert allowed == {"chr1", "chr2"}


def test_load_chromosome_allowlist_empty_raises(tmp_path):
    fai = tmp_path / "empty.fai"
    fai.write_text("")
    with pytest.raises(ValueError, match="No chromosomes"):
        bp.load_chromosome_allowlist(str(fai))


def test_bed_preprocess_full_filter_pipeline(tmp_path):
    """End-to-end: MAPQ + length + chromosome allowlist + blacklist subtraction."""
    raw = tmp_path / "test.hg38.frag.gz"
    _write_fragment_gz(raw, [
        # chr, start, end, mapq, strand  (FinaleDB convention)
        "chr1\t100\t250\t40\t+",   # kept: pass all filters
        "chr1\t500\t510\t40\t+",   # dropped: length=10 < 20
        "chr1\t1000\t1100\t20\t+", # dropped: mapq=20 < 30
        "chrM\t10\t100\t40\t+",    # dropped: chrM not in allowlist
        "chr1\t2000\t2100\t40\t+", # dropped: hits blacklist
        "chr2\t300\t450\t40\t-",   # kept
    ])
    fai = tmp_path / "ref.fai"
    _write_fai(fai, {"chr1": 5000, "chr2": 3000})
    bl = tmp_path / "bl.bed.gz"
    _write_blacklist_gz(bl, [("chr1", 1900, 2200)])  # covers chr1:2000-2100

    out_dir = tmp_path / "out"
    paths = bp.preprocess(str(raw), str(out_dir), str(bl), str(fai))

    bed_lines = Path(paths["bed"]).read_text().strip().split("\n")
    assert len(bed_lines) == 2
    assert bed_lines[0].split("\t") == ["chr1", "100", "250", "40", "+"]
    assert bed_lines[1].split("\t") == ["chr2", "300", "450", "40", "-"]

    # Summary line: <path>\t<mean>\t<median>\t<min>\t<max>\t<count>
    summary = Path(paths["summary"]).read_text().strip().split("\t")
    assert summary[-1] == "2"
    assert int(summary[-2]) == 150  # max length
    assert int(summary[-3]) == 150  # min length
    # mean = (150 + 150) / 2 = 150
    assert abs(float(summary[1]) - 150.0) < 0.01


def test_bed_preprocess_empty_after_filter_writes_zero_summary(tmp_path):
    raw = tmp_path / "empty.frag.gz"
    # Every row gets filtered out (length < 20).
    _write_fragment_gz(raw, ["chr1\t100\t110\t40\t+"])
    fai = tmp_path / "ref.fai"
    _write_fai(fai, {"chr1": 5000})
    bl = tmp_path / "bl.bed.gz"
    _write_blacklist_gz(bl, [])

    paths = bp.preprocess(str(raw), str(tmp_path / "out"), str(bl), str(fai))
    assert Path(paths["bed"]).stat().st_size == 0
    summary = Path(paths["summary"]).read_text().strip()
    assert summary.endswith("NA\tNA\tNA\tNA\t0")


def test_build_sparse_index_matches_awk_spec(tmp_path):
    """Index format: `<line_number> <byte_offset>` every `interval` lines."""
    bed = tmp_path / "tiny.bed"
    # 5 lines, each 8 bytes incl '\n'
    bed.write_text("a\tb\tc\nd\te\tf\ng\th\ti\nj\tk\tl\nm\tn\to\n")
    idx = tmp_path / "tiny.bed.index"
    bp.build_sparse_index(str(bed), str(idx), interval=2)
    rows = idx.read_text().strip().split("\n")
    # interval=2 -> entries at line 2 (offset after 2 lines = 12) and line 4 (offset 24).
    assert rows[0] == "2 12"
    assert rows[1] == "4 24"


def test_write_summary_min_median_max(tmp_path):
    bed = tmp_path / "lens.bed"
    bed.write_text("chr1\t0\t20\n" "chr1\t100\t150\n" "chr1\t200\t300\n")
    summary = tmp_path / "lens.summary.txt"
    bp.write_summary(str(bed), str(summary))
    parts = summary.read_text().strip().split("\t")
    # lengths: 20, 50, 100  -> mean 56.66, median 50, min 20, max 100, count 3
    assert int(parts[-1]) == 3
    assert int(parts[-2]) == 100
    assert int(parts[-3]) == 20
    assert int(parts[-4]) == 50      # median (odd count, middle value)


# ---------- reshape_layer -------------------------------------------------

def _write_extract_h5(path: Path, n: int = 4, shape_3d_layers: int | None = None):
    """Build a synthetic leaf1-extract-format H5.

    If shape_3d_layers is None, single-layer (n, 9984).
    Otherwise multi-layer (n, layers, 9984).
    """
    with h5py.File(path, "w") as f:
        eg = f.create_group("embeddings")
        if shape_3d_layers is None:
            data = np.random.default_rng(0).standard_normal((n, 9984), dtype=np.float32).astype(np.float16)
        else:
            data = np.random.default_rng(0).standard_normal(
                (n, shape_3d_layers, 9984), dtype=np.float32
            ).astype(np.float16)
        eg.create_dataset("bert_window_embeddings", data=data, dtype=np.float16)
        eg.attrs["num_windows"] = 3
        eg.attrs["embed_dim"] = 768
        eg.attrs["window_embed_dim"] = 9984
        eg.attrs["target_layer"] = 4 if shape_3d_layers is None else -1

        fg = f.create_group("fragment_info")
        fg.create_dataset("chromosome", data=np.array(["chr1"] * n, dtype="S10"))
        for k in ("fragment_start", "fragment_end", "adjusted_start",
                   "adjusted_end", "line_number"):
            fg.create_dataset(k, data=np.arange(n, dtype=np.int64))
        for k in ("n_token_left", "n_token_frag", "n_token_right"):
            fg.create_dataset(k, data=np.full(n, 10, dtype=np.int32))
        fg.create_dataset("n_freq", data=np.zeros(n, dtype=np.float32))
        fg.create_dataset("reverse_complemented", data=np.zeros(n, dtype=bool))

        mg = f.create_group("metadata")
        mg.attrs["sample_id"] = "synthetic"
        mg.attrs["group"] = "test"

        pg = f.create_group("progress")
        pg.attrs["completed"] = True


def test_reshape_layer_single_layer_input(tmp_path):
    src = tmp_path / "src.h5"
    _write_extract_h5(src, n=5, shape_3d_layers=None)
    dst = tmp_path / "dst.h5"
    rl.reshape(str(src), str(dst), layer=4)

    with h5py.File(dst, "r") as f:
        assert f["embeddings"].shape == (5, 13, 768)
        assert f["embeddings"].dtype == np.float16
        assert f["segment_names"].shape == (13,)
        assert f["segment_slices"].shape == (13, 2)
        assert f["segment_slices"][0].tolist() == [0, 768]
        assert f["segment_slices"][-1].tolist() == [12 * 768, 13 * 768]
        assert f.attrs["completed"] is np.True_ or bool(f.attrs["completed"]) is True
        assert f.attrs["layer_extracted"] == 4
        # fragment_info copied verbatim.
        assert "fragment_info" in f
        assert f["fragment_info/fragment_start"][:].tolist() == list(range(5))


def test_reshape_layer_multi_layer_input_picks_layer(tmp_path):
    src = tmp_path / "src_multi.h5"
    _write_extract_h5(src, n=3, shape_3d_layers=13)
    dst = tmp_path / "dst_multi.h5"
    rl.reshape(str(src), str(dst), layer=4)

    with h5py.File(src, "r") as fin, h5py.File(dst, "r") as fout:
        expected = fin["embeddings/bert_window_embeddings"][:, 4, :].reshape(3, 13, 768)
        np.testing.assert_array_equal(fout["embeddings"][:], expected)
        assert fout.attrs["layer_extracted"] == 4


def test_reshape_layer_out_of_range_layer_raises(tmp_path):
    src = tmp_path / "src.h5"
    _write_extract_h5(src, n=2, shape_3d_layers=5)
    dst = tmp_path / "dst.h5"
    with pytest.raises(ValueError, match="out of range"):
        rl.reshape(str(src), str(dst), layer=10)


# ---------- validate_extract ----------------------------------------------

def _write_pub_h5(path: Path, embeddings: np.ndarray, fragment_info: dict, sample_id="X"):
    """Write an H5 in the published `layer4_reshaped/` schema."""
    with h5py.File(path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, dtype=np.float16)
        fg = f.create_group("fragment_info")
        for k, v in fragment_info.items():
            fg.create_dataset(k, data=v)
        mg = f.create_group("metadata")
        mg.attrs["sample_id"] = sample_id
        f.attrs["completed"] = True


def _toy_fragment_info(n: int) -> dict:
    return {
        "chromosome": np.array(["chr1"] * n, dtype="S10"),
        "fragment_start": np.arange(n, dtype=np.int64),
        "fragment_end": np.arange(n, dtype=np.int64) + 100,
        "adjusted_start": np.zeros(n, dtype=np.int64),
        "adjusted_end": np.zeros(n, dtype=np.int64),
        "line_number": np.arange(n, dtype=np.int64),
        "n_token_left": np.full(n, 20, dtype=np.int32),
        "n_token_frag": np.full(n, 15, dtype=np.int32),
        "n_token_right": np.full(n, 20, dtype=np.int32),
        "n_freq": np.zeros(n, dtype=np.float32),
        "reverse_complemented": np.zeros(n, dtype=bool),
    }


def test_validate_extract_matched_pair(tmp_path):
    """Two identical H5s in published schema should match bitwise."""
    emb = np.random.default_rng(0).standard_normal((6, 13, 768), dtype=np.float32).astype(np.float16)
    fi = _toy_fragment_info(6)
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _write_pub_h5(a, emb, fi)
    _write_pub_h5(b, emb, fi)
    report = ve.diff(str(a), str(b))
    assert report["embedding_bitwise_equal"] is True
    assert report["embedding_max_abs_diff"] == 0.0
    assert report["errors"] == []


def test_validate_extract_diverged_pair_reports_max_abs_diff(tmp_path):
    """An intentional perturbation surfaces a non-zero max-abs-diff and error."""
    emb_a = np.zeros((4, 13, 768), dtype=np.float16)
    emb_b = emb_a.copy()
    emb_b[0, 0, 0] = np.float16(1.5)
    fi = _toy_fragment_info(4)
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    _write_pub_h5(a, emb_a, fi)
    _write_pub_h5(b, emb_b, fi)
    report = ve.diff(str(a), str(b), embedding_atol=0.0)
    assert report["embedding_bitwise_equal"] is False
    assert report["embedding_max_abs_diff"] == pytest.approx(1.5, rel=1e-3)
    assert any("Embedding diverged" in e for e in report["errors"])


def test_validate_extract_compares_leaf1_extract_against_pub_schema(tmp_path):
    """A leaf1-extract H5 (flat 9984) should compare cleanly against the same data
    written in the published (N, 13, 768) layout, after the reshape pass."""
    n = 5
    rng = np.random.default_rng(0)
    flat = rng.standard_normal((n, 9984), dtype=np.float32).astype(np.float16)
    seg = flat.reshape(n, 13, 768)
    fi = _toy_fragment_info(n)

    # Side A: leaf1-extract style (nested embeddings/bert_window_embeddings flat).
    leaf1_h5 = tmp_path / "leaf1.h5"
    with h5py.File(leaf1_h5, "w") as f:
        eg = f.create_group("embeddings")
        eg.create_dataset("bert_window_embeddings", data=flat, dtype=np.float16)
        fg = f.create_group("fragment_info")
        for k, v in fi.items():
            fg.create_dataset(k, data=v)
        f.create_group("metadata").attrs["sample_id"] = "S"
        f.create_group("progress").attrs["completed"] = True

    # Side B: published schema with the same data.
    pub_h5 = tmp_path / "pub.h5"
    _write_pub_h5(pub_h5, seg, fi)

    report = ve.diff(str(leaf1_h5), str(pub_h5))
    assert report["embedding_bitwise_equal"] is True
    assert report["errors"] == []


def test_validate_extract_realigns_by_line_number(tmp_path):
    """If row order differs but line_number sets match, the diff realigns."""
    n = 6
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, 13, 768), dtype=np.float32).astype(np.float16)
    fi = _toy_fragment_info(n)

    a = tmp_path / "a.h5"
    _write_pub_h5(a, emb, fi)

    # B: same data but rows shuffled.
    perm = np.array([3, 1, 5, 0, 4, 2])
    fi_perm = {k: (v[perm] if k != "chromosome" else v) for k, v in fi.items()}
    fi_perm["chromosome"] = fi["chromosome"][perm]
    b = tmp_path / "b.h5"
    _write_pub_h5(b, emb[perm], fi_perm)

    report = ve.diff(str(a), str(b))
    assert report.get("realigned_by_line_number") is True
    assert report["embedding_bitwise_equal"] is True
