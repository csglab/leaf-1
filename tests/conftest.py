"""Shared pytest fixtures for the LEAF-1 test suite.

Generates deterministic synthetic FASTA + BED + index fixtures on first run and
reuses them across tests. Real `samtools` is used to build the .fai so
`bedtools getfasta` (which the get_fragment pipeline shells out to) sees a valid
index.
"""

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES_DIR = Path(__file__).parent / "fixtures"
TOKENIZER_PATH = REPO_ROOT / "leaf1" / "data" / "tokenizer.json"

# Synthetic genome: two chromosomes, single-line FASTA so .fai math stays simple.
CHR_LENGTHS = {"chr1": 5000, "chr2": 3000}

# Fragments picked to exercise: middle-of-chrom, near-start, near-end, a long
# fragment, and a tiny fragment. Sorted as BED expects.
TINY_BED_RECORDS = [
    # chr, start, end
    ("chr1", 100, 250),     # near chr1 start
    ("chr1", 1000, 1150),   # middle of chr1
    ("chr1", 2500, 2680),   # larger fragment in chr1
    ("chr1", 4800, 4950),   # near chr1 end
    ("chr2", 500, 600),     # small fragment on chr2
    ("chr2", 1500, 1700),   # 200 bp on chr2
]


def _random_dna(length: int, seed: int) -> str:
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=length))


def _write_fasta(path: Path):
    lines = []
    for i, (chr_name, chr_len) in enumerate(CHR_LENGTHS.items()):
        seq = _random_dna(chr_len, seed=42 + i)
        lines.append(f">{chr_name}\n")
        lines.append(seq + "\n")
    path.write_text("".join(lines))


def _write_bed(path: Path):
    with open(path, "w") as f:
        for chr_name, start, end in TINY_BED_RECORDS:
            f.write(f"{chr_name}\t{start}\t{end}\n")


def _write_bed_index(bed_path: Path, index_path: Path):
    # The .bed.index holds (line, byte_offset) entries; smallest valid form is a
    # single "0 0" row that read_index_file treats as "start from the top".
    index_path.write_text("0 0\n")


def _build_fai(fa_path: Path):
    subprocess.run(["samtools", "faidx", str(fa_path)], check=True)


def _write_manifest(manifest_path: Path, bed_path: Path):
    n_fragments = len(TINY_BED_RECORDS)
    manifest_path.write_text(
        "sample_id,modality,bed_file,n_fragment,group\n"
        f"sample_x,CFDNA,{bed_path},{n_fragments},healthy\n"
    )


def _ensure_fixtures():
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    fa = FIXTURES_DIR / "tiny.fa"
    fai = FIXTURES_DIR / "tiny.fa.fai"
    bed = FIXTURES_DIR / "tiny.bed"
    bed_index = FIXTURES_DIR / "tiny.bed.index"
    manifest = FIXTURES_DIR / "manifest.csv"

    if not fa.exists():
        _write_fasta(fa)
    if not fai.exists():
        _build_fai(fa)
    if not bed.exists():
        _write_bed(bed)
    if not bed_index.exists():
        _write_bed_index(bed, bed_index)
    # Keep path-bearing CSVs local to the current checkout. These files are
    # ignored by git because they contain absolute fixture paths.
    _write_manifest(manifest, bed)


_ensure_fixtures()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def tiny_fa() -> Path:
    return FIXTURES_DIR / "tiny.fa"


@pytest.fixture(scope="session")
def tiny_fai() -> Path:
    return FIXTURES_DIR / "tiny.fa.fai"


@pytest.fixture(scope="session")
def tiny_bed() -> Path:
    return FIXTURES_DIR / "tiny.bed"


@pytest.fixture(scope="session")
def tiny_bed_index() -> Path:
    return FIXTURES_DIR / "tiny.bed.index"


@pytest.fixture(scope="session")
def tiny_manifest() -> Path:
    return FIXTURES_DIR / "manifest.csv"


@pytest.fixture(scope="session")
def chr_lengths() -> dict:
    return dict(CHR_LENGTHS)


@pytest.fixture(scope="session")
def bed_records() -> list:
    return list(TINY_BED_RECORDS)


@pytest.fixture(scope="session")
def tokenizer_path() -> Path:
    if not TOKENIZER_PATH.exists():
        pytest.skip(f"Tokenizer asset missing: {TOKENIZER_PATH}")
    return TOKENIZER_PATH


@pytest.fixture(scope="session")
def tokenizer(tokenizer_path):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        model_max_length=128,
        padding_side="right",
        truncation_side="right",
        cls_token="[CLS]",
        bos_token="[CLS]",
        sep_token="[SEP]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        mask_token="[MASK]",
        pad_token="[PAD]",
    )


@pytest.fixture(scope="session")
def bedtools_available():
    return shutil.which("bedtools") is not None


@pytest.fixture
def needs_bedtools(bedtools_available):
    if not bedtools_available:
        pytest.skip("bedtools not on PATH")


# =============================================================================
# Downstream fixtures (mean_pool + mil)
# =============================================================================
# Synthetic per-sample H5s mimic the leaf1-extract single-layer output:
#   embeddings/bert_window_embeddings    (n, 9984) float16
#   fragment_info/*                       per-fragment metadata
#   metadata/                             sample_id, group, modality (attrs)
#   progress/                             completed = True
#
# Eight samples, 4 healthy + 4 cancer, ~50 fragments each. Embeddings are
# deterministic-random with a sample-dependent shift so a downstream classifier
# can separate the two groups (otherwise SVM CV is degenerate).

DOWNSTREAM_N_SAMPLES = 8
DOWNSTREAM_FRAGS_PER_SAMPLE = 50
DOWNSTREAM_FLAT_DIM = 13 * 768  # 9984, matches leaf1-extract single-layer

_DOWNSTREAM_GROUPS = (
    ["healthy"] * (DOWNSTREAM_N_SAMPLES // 2)
    + ["cancer"]  * (DOWNSTREAM_N_SAMPLES // 2)
)


def _synth_sample_embedding(sample_idx: int, group: str, rng: np.random.Generator):
    """One per-sample embedding block: (n_frags, 9984) float16.

    Cancer rows get a shift inside the cut1 segment (segment index 9,
    columns 9*768 .. 9*768+200) so a downstream classifier trained on the
    paper's `cut,mod,fragment1` segment selection picks it up.
    """
    base = rng.standard_normal(
        (DOWNSTREAM_FRAGS_PER_SAMPLE, DOWNSTREAM_FLAT_DIM),
        dtype=np.float32,
    )
    if group == "cancer":
        cut1_start = 9 * 768
        base[:, cut1_start: cut1_start + 200] += 1.5
    return base.astype(np.float16)


def _build_per_sample_h5(out_path, sample_id, group, modality, rng):
    import h5py
    emb = _synth_sample_embedding(int(sample_id.split("_")[1]), group, rng)
    n = emb.shape[0]
    with h5py.File(out_path, "w") as f:
        eg = f.create_group("embeddings")
        ds = eg.create_dataset("bert_window_embeddings", data=emb, dtype=np.float16)
        eg.attrs["num_windows"] = 3
        eg.attrs["embed_dim"] = 768
        eg.attrs["window_embed_dim"] = DOWNSTREAM_FLAT_DIM
        eg.attrs["num_layers"] = 1
        eg.attrs["target_layer"] = 4

        fg = f.create_group("fragment_info")
        str_dtype = h5py.special_dtype(vlen=str)
        fg.create_dataset("chromosome", data=np.array(["chr1"] * n, dtype=object), dtype=str_dtype)
        fg.create_dataset("fragment_start", data=np.arange(100, 100 + n * 10, 10, dtype=np.int64))
        fg.create_dataset("fragment_end",   data=np.arange(250, 250 + n * 10, 10, dtype=np.int64))
        fg.create_dataset("adjusted_start", data=np.zeros(n, dtype=np.int64))
        fg.create_dataset("adjusted_end",   data=np.zeros(n, dtype=np.int64))
        fg.create_dataset("line_number",    data=np.arange(n, dtype=np.int64))
        fg.create_dataset("reverse_complemented", data=np.zeros(n, dtype=bool))
        fg.create_dataset("n_token_left",   data=np.full(n, 20, dtype=np.int32))
        fg.create_dataset("n_token_frag",   data=np.full(n, 15, dtype=np.int32))
        fg.create_dataset("n_token_right",  data=np.full(n, 20, dtype=np.int32))
        fg.create_dataset("n_freq",         data=np.zeros(n, dtype=np.float32))

        mg = f.create_group("metadata")
        mg.attrs["sample_id"] = sample_id
        mg.attrs["group"] = group
        mg.attrs["modality"] = modality
        mg.attrs["bed_file"] = "/tmp/synthetic.bed"
        mg.attrs["bert_checkpoint"] = "/tmp/synthetic_ckpt"

        pg = f.create_group("progress")
        pg.attrs["completed"] = True
        pg.attrs["current_batch_idx"] = -1
        pg.attrs["total_batches"] = 1
        pg.attrs["batch_size"] = 32
        pg.attrs["seed"] = 42
        pg.attrs["processed_count"] = n


def _ensure_downstream_fixtures():
    """Build the per-sample H5 dir + metadata CSV + split JSON if missing."""
    import h5py
    dn_dir = FIXTURES_DIR / "downstream"
    h5_dir = dn_dir / "h5_files"
    h5_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2024)

    sample_records = []
    for i in range(DOWNSTREAM_N_SAMPLES):
        sample_id = f"sample_{i}"
        group = _DOWNSTREAM_GROUPS[i]
        h5_path = h5_dir / f"cfdna_{sample_id}.h5"
        if not h5_path.exists():
            _build_per_sample_h5(h5_path, sample_id, group, "cfdna", rng)
        sample_records.append({"sample_id": sample_id, "group": group, "h5_path": str(h5_path)})

    # Metadata CSV (mean_pool.train consumer). Path-bearing CSVs are refreshed
    # for the current checkout and ignored by git.
    import pandas as pd
    meta_csv = dn_dir / "metadata.csv"
    pd.DataFrame(sample_records).to_csv(meta_csv, index=False)

    # MIL splits: metadata_complete.csv + cv_splits/healthy_vs_cancer_k5.json
    split_root = dn_dir / "mil_data"
    cv_splits = split_root / "cv_splits"
    cv_splits.mkdir(parents=True, exist_ok=True)
    meta_complete = split_root / "metadata_complete.csv"
    pd.DataFrame(sample_records).to_csv(meta_complete, index=False)

    split_json = cv_splits / "healthy_vs_cancer_k5.json"
    if not split_json.exists():
        import json
        # 2 folds, simple alternating split (toy)
        splits = [
            {"train_idx": [0, 1, 4, 5], "eval_idx": [2, 3, 6, 7]},
            {"train_idx": [2, 3, 6, 7], "eval_idx": [0, 1, 4, 5]},
        ]
        split_json.write_text(json.dumps({
            "metadata": {
                "comparison_type": "cancer_vs_healthy",
                "target": "cancer",
            },
            "splits": splits,
        }))

    # Aggregated H5 used by mean_pool tests (built lazily inside the test fixture).
    return dn_dir, h5_dir, meta_csv, split_root, split_json


@pytest.fixture(scope="session")
def downstream_dir() -> Path:
    """Root of the downstream fixture tree."""
    dn_dir, *_ = _ensure_downstream_fixtures()
    return dn_dir


@pytest.fixture(scope="session")
def downstream_h5_dir() -> Path:
    """Directory containing 8 per-sample H5s (cfdna_sample_0.h5 .. cfdna_sample_7.h5)."""
    _, h5_dir, *_ = _ensure_downstream_fixtures()
    return h5_dir


@pytest.fixture(scope="session")
def downstream_metadata_csv() -> Path:
    """metadata.csv with sample_id, group, h5_path columns."""
    _, _, meta_csv, *_ = _ensure_downstream_fixtures()
    return meta_csv


@pytest.fixture(scope="session")
def mil_split_root() -> Path:
    """Root that contains metadata_complete.csv + cv_splits/ for load_splits."""
    *_, split_root, _ = _ensure_downstream_fixtures()
    return split_root


@pytest.fixture(scope="session")
def mil_split_json() -> Path:
    """cv_splits/healthy_vs_cancer_k5.json (2-fold toy split)."""
    *_, split_json = _ensure_downstream_fixtures()
    return split_json


@pytest.fixture(scope="session")
def downstream_aggregated_h5(downstream_h5_dir, tmp_path_factory) -> Path:
    """Build an aggregated (N, 13, 768) H5 once by calling the real aggregator."""
    from leaf1.downstream.mean_pool.aggregate import aggregate

    out = tmp_path_factory.mktemp("agg") / "aggregated.h5"
    aggregate(
        input_dir=str(downstream_h5_dir.parent),  # parent contains h5_files/
        output=str(out),
        model_name="leaf1_synthetic",
        workers=1,
        layer_idx=4,
    )
    return out
