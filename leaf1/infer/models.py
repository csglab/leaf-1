"""Named LEAF-1 encoder checkpoints shipped with the repo (``leaf1-extract --model``).

Each model was pretrained on one input format, and gives wrong embeddings without
any error if it is fed another one. The format therefore travels with the checkpoint:
every checkpoint under ``encoder_checkpoints/`` has ``leaf1_*`` keys in its ``config.json``
(see ``INPUT_KEYS``), and extraction takes its input settings from them. Passing a
setting that disagrees with them is an error. The released ``checkpoint/`` is left
as released (no such keys); its weights are recognised by md5 and held to the
released input format the same way.

``leaf1_model`` is also written to the H5 metadata, so ``leaf1-mil-predict`` can
refuse embeddings from a different encoder than the one its classifier was fit on.
"""

import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = "leaf1"

# Input settings of the released model (and of any checkpoint without leaf1_* keys).
RELEASE_INPUT = {"ablation": "none", "max_token_length": 2048, "min_flank_token_length": 100}

# config.json key -> input setting
INPUT_KEYS = {
    "leaf1_ablation": "ablation",
    "leaf1_max_token_length": "max_token_length",
    "leaf1_min_flank_token_length": "min_flank_token_length",
}

# path is relative to the repo root; md5 is of model.safetensors.
MODELS = {
    "leaf1": {
        "path": "checkpoint",
        "md5": "1b28ca84edc43ee66365d99ccc597c65",
        "input": RELEASE_INPUT,
        "description": "Released LEAF-1 encoder (100,000 updates). Used for all manuscript results.",
    },
    "leaf1-noflank-s42": {
        "path": "encoder_checkpoints/leaf1-noflank-s42",
        "md5": "ce70e67c59011fd1a8a4d02394be014f",
        "input": {"ablation": "noflank", "max_token_length": 512, "min_flank_token_length": 0},
        "description": "Flank-free encoder ([CLS] [MOD] [CUT] body [CUT]), 30,000 updates, seed 42.",
    },
    "leaf1-noflank-s43": {
        "path": "encoder_checkpoints/leaf1-noflank-s43",
        "md5": "9b3ed5b1e4ac4fb2ec9ea335b97b5898",
        "input": {"ablation": "noflank", "max_token_length": 512, "min_flank_token_length": 0},
        "description": "Flank-free encoder ([CLS] [MOD] [CUT] body [CUT]), 30,000 updates, seed 43.",
    },
    "leaf1-nocristiano": {
        "path": "encoder_checkpoints/leaf1-nocristiano",
        "md5": "6a8122f4eb3bd64c903cb930483574bb",
        "input": RELEASE_INPUT,
        "description": "LEAF-1 pretrained without the Cristiano et al. cfDNA samples (104,500 updates).",
    },
}

_LFS_POINTER_PREFIX = b"version https://git-lfs"


def model_names():
    return list(MODELS)


def get_model(name):
    if name not in MODELS:
        raise ValueError(f"Unknown --model {name!r}; expected one of {model_names()}")
    return MODELS[name]


def model_path(name):
    return REPO_ROOT / get_model(name)["path"]


def check_weights_present(checkpoint_dir):
    """Fail clearly if the weights are a Git LFS pointer that was never pulled."""
    weights = Path(checkpoint_dir) / "model.safetensors"
    if weights.is_file() and weights.stat().st_size < 1024:
        with open(weights, "rb") as f:
            if f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                rel = Path(checkpoint_dir).resolve()
                try:
                    rel = rel.relative_to(REPO_ROOT)
                except ValueError:
                    pass
                # --exclude="" overrides the default exclude of encoder_checkpoints/ in .lfsconfig;
                # without it git lfs pull fetches nothing for those paths.
                raise RuntimeError(
                    f"{weights} is a Git LFS pointer, not the weights. Fetch them with:\n"
                    f'  git lfs pull --include="{rel}/**" --exclude=""')


_MD5_CACHE = {}


def check_ref_genome_present(fasta):
    """Fail clearly if the reference FASTA is a Git LFS pointer (bedtools would only report EOF)."""
    fasta = Path(fasta)
    if fasta.is_file() and fasta.stat().st_size < 1024:
        with open(fasta, "rb") as f:
            if f.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                raise RuntimeError(
                    f"{fasta} is a Git LFS pointer, not the genome. Fetch it with:\n"
                    '  git lfs pull --include="assets/refseq.genome.fa"\n'
                    "or pass your own hg38 FASTA (with .fai) as --ref_genome_file.")


def _md5(path, chunk=1 << 24):
    """md5 of a file, cached per process on (path, size, mtime) so a run hashes it once."""
    st = Path(path).stat()
    key = (str(Path(path).resolve()), st.st_size, st.st_mtime_ns)
    if key not in _MD5_CACHE:
        _MD5_CACHE[key] = _md5_uncached(path, chunk)
    return _MD5_CACHE[key]


def _md5_uncached(path, chunk):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_md5(name, checkpoint_dir):
    want = get_model(name)["md5"]
    got = _md5(Path(checkpoint_dir) / "model.safetensors")
    if got != want:
        raise RuntimeError(
            f"{checkpoint_dir}/model.safetensors has md5 {got}, expected {want} for --model {name}. "
            "The file is incomplete, modified, or not this model. Delete it and fetch it again "
            "(git lfs pull does not overwrite an existing file):\n"
            f'  rm {Path(checkpoint_dir) / "model.safetensors"}\n'
            f'  git lfs pull --include="{get_model(name)["path"]}/**" --exclude=""')


def is_registered_path(name, checkpoint_dir):
    """True if checkpoint_dir is the repo's own directory for model `name`."""
    try:
        return Path(checkpoint_dir).resolve() == model_path(name).resolve()
    except OSError:
        return False


def stamped_input(model_config):
    """(leaf1_model, input settings) recorded in a checkpoint config, or (None, None)."""
    name = getattr(model_config, "leaf1_model", None)
    present = {k: getattr(model_config, k) for k in INPUT_KEYS if hasattr(model_config, k)}
    if name is None and not present:
        return None, None
    if name is None or len(present) != len(INPUT_KEYS):
        raise ValueError(
            "Checkpoint config has an incomplete LEAF-1 input stamp: expected leaf1_model and "
            f"{sorted(INPUT_KEYS)}, found leaf1_model={name!r} and {sorted(present)}")
    stamp = {INPUT_KEYS[k]: v for k, v in present.items()}
    if name in MODELS and stamp != MODELS[name]["input"]:
        raise ValueError(f"Checkpoint config for {name} has input {stamp}, "
                         f"but the registry says {MODELS[name]['input']}")
    return name, stamp


def is_released_weights(checkpoint_dir):
    """True if checkpoint_dir holds the released LEAF-1 weights (by md5 of model.safetensors).

    The released checkpoint/ has no leaf1_* config keys (its files are left as released),
    so it is recognised by its weights instead.
    """
    weights = Path(checkpoint_dir) / "model.safetensors"
    if not weights.is_file() or weights.stat().st_size < 1024:
        return False
    return _md5(weights) == MODELS[DEFAULT_MODEL]["md5"]


def resolve_input(model_config, checkpoint_dir=None, *, ablation=None, max_token_length=None,
                  min_flank_token_length=None):
    """Input settings for a checkpoint. None means "not given".

    Returns (leaf1_model or None, ablation, max_token_length, min_flank_token_length).
    A stamped checkpoint, or the released weights (recognised by md5 when checkpoint_dir
    is given), fixes all three settings: a given setting that differs is an error. For
    the released model leaf1_model is returned as None, so its H5 metadata is unchanged.
    Any other checkpoint takes the given settings, with unset ones at the released defaults.
    """
    given = {"ablation": ablation, "max_token_length": max_token_length,
             "min_flank_token_length": min_flank_token_length}
    name, stamp = stamped_input(model_config)
    shown = name
    if stamp is None and checkpoint_dir is not None and is_released_weights(checkpoint_dir):
        shown, stamp = f"{DEFAULT_MODEL} (the released LEAF-1 encoder)", RELEASE_INPUT
    if stamp is None:
        out = {k: (RELEASE_INPUT[k] if v is None else v) for k, v in given.items()}
        return None, out["ablation"], out["max_token_length"], out["min_flank_token_length"]
    bad = {k: v for k, v in given.items() if v is not None and v != stamp[k]}
    if bad:
        wrong = ", ".join(f"--{k} {v!r} (checkpoint needs {stamp[k]!r})" for k, v in bad.items())
        raise ValueError(
            f"This checkpoint is {shown}, which was pretrained with {stamp}. You passed {wrong}. "
            "A model given a different input format produces wrong embeddings. Drop these flags; "
            "the settings are fixed by the checkpoint.")
    return name, stamp["ablation"], stamp["max_token_length"], stamp["min_flank_token_length"]
