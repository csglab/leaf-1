"""leaf1-extract --model presets and the checkpoint input-settings guard (leaf1/infer/models.py)."""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from leaf1.infer import extract_embeddings as ee
from leaf1.infer import models as M

STAMP_KEYS = ("leaf1_model", "leaf1_ablation", "leaf1_max_token_length", "leaf1_min_flank_token_length")


class Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def stamp(name, abl, ml, mf):
    return dict(leaf1_model=name, leaf1_ablation=abl, leaf1_max_token_length=ml, leaf1_min_flank_token_length=mf)


# ---------- registry and shipped checkpoint dirs ---------------------------------------------

def test_registry_paths_and_inputs():
    assert M.DEFAULT_MODEL == "leaf1"
    assert M.MODELS["leaf1"]["path"] == "checkpoint"
    assert M.MODELS["leaf1"]["input"] == {"ablation": "none", "max_token_length": 2048, "min_flank_token_length": 100}
    for name, m in M.MODELS.items():
        assert set(m) == {"path", "md5", "input", "description"}
        assert len(m["md5"]) == 32 and set(m["input"]) == set(M.RELEASE_INPUT)
        assert (M.REPO_ROOT / m["path"] / "config.json").is_file(), name


@pytest.mark.parametrize("name", [n for n in M.MODELS if n != "leaf1"])
def test_shipped_checkpoint_configs_carry_their_input(name):
    from transformers import DistilBertConfig
    cfg = DistilBertConfig.from_pretrained(str(M.model_path(name)))
    got_name, got = M.stamped_input(cfg)
    assert got_name == name and got == M.MODELS[name]["input"]
    assert cfg.max_position_embeddings == got["max_token_length"]


def test_released_checkpoint_config_is_unstamped():
    raw = json.loads((M.REPO_ROOT / "checkpoint" / "config.json").read_text())
    assert not [k for k in raw if k.startswith("leaf1_")]


def _weights_pulled(name):
    w = M.model_path(name) / "model.safetensors"
    return w.is_file() and w.stat().st_size > 1024


@pytest.mark.parametrize("name", list(M.MODELS))
def test_shipped_weights_md5(name):
    if not _weights_pulled(name):
        pytest.skip(f"{name} weights not pulled from Git LFS")
    M.check_md5(name, M.model_path(name))


# ---------- resolve_input ----------------------------------------------------------------------

def test_unstamped_defaults_are_release():
    assert M.resolve_input(Cfg()) == (None, "none", 2048, 100)


def test_unstamped_passes_explicit_values_through():
    assert M.resolve_input(Cfg(), ablation="nocut", max_token_length=128, min_flank_token_length=8) == \
        (None, "nocut", 128, 8)


def test_stamped_values_are_used():
    cfg = Cfg(**stamp("leaf1-noflank-s42", "noflank", 512, 0))
    assert M.resolve_input(cfg) == ("leaf1-noflank-s42", "noflank", 512, 0)
    # the same values given explicitly are fine
    assert M.resolve_input(cfg, ablation="noflank", max_token_length=512, min_flank_token_length=0) == \
        ("leaf1-noflank-s42", "noflank", 512, 0)


@pytest.mark.parametrize("kw", [dict(ablation="none"), dict(max_token_length=2048),
                                dict(max_token_length=128), dict(min_flank_token_length=100)])
def test_stamped_conflict_raises(kw):
    cfg = Cfg(**stamp("leaf1-noflank-s42", "noflank", 512, 0))
    with pytest.raises(ValueError, match="leaf1-noflank-s42"):
        M.resolve_input(cfg, **kw)


def test_nocristiano_conflict_raises():
    cfg = Cfg(**stamp("leaf1-nocristiano", "none", 2048, 100))
    assert M.resolve_input(cfg) == ("leaf1-nocristiano", "none", 2048, 100)
    with pytest.raises(ValueError, match="leaf1-nocristiano"):
        M.resolve_input(cfg, ablation="noflank")


@pytest.mark.parametrize("drop", STAMP_KEYS)
def test_incomplete_stamp_raises(drop):
    s = stamp("leaf1-noflank-s42", "noflank", 512, 0)
    del s[drop]
    with pytest.raises(ValueError, match="incomplete"):
        M.resolve_input(Cfg(**s))


def test_stamp_disagreeing_with_registry_raises():
    with pytest.raises(ValueError, match="registry"):
        M.resolve_input(Cfg(**stamp("leaf1-noflank-s42", "none", 2048, 100)))


def test_unregistered_stamp_is_used():
    assert M.resolve_input(Cfg(**stamp("my-model", "nomod", 1024, 50))) == ("my-model", "nomod", 1024, 50)


# ---------- weight checks ----------------------------------------------------------------------

def test_lfs_pointer_detected(tmp_path):
    (tmp_path / "model.safetensors").write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 361516828\n")
    with pytest.raises(RuntimeError, match='git lfs pull --include=".*" --exclude=""'):
        M.check_weights_present(tmp_path)


def test_real_weights_pass_pointer_check(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 4096)
    M.check_weights_present(tmp_path)


def test_md5_mismatch_raises(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not the model")
    with pytest.raises(RuntimeError, match="md5"):
        M.check_md5("leaf1-noflank-s42", tmp_path)


# ---------- extraction with a stamped tiny checkpoint ------------------------------------------

def _tiny(tokenizer, out, max_pos, **extra):
    from transformers import DistilBertConfig, DistilBertForMaskedLM
    cfg = DistilBertConfig(max_position_embeddings=max_pos, vocab_size=tokenizer.vocab_size, n_layers=2,
                           n_heads=12, dim=ee.EMBED_DIM, hidden_dim=512, pad_token_id=tokenizer.pad_token_id,
                           **extra)
    torch.manual_seed(0)
    DistilBertForMaskedLM(cfg).save_pretrained(out)
    return str(out)


@pytest.fixture(scope="module")
def tiny_noflank(tokenizer, tmp_path_factory):
    return _tiny(tokenizer, tmp_path_factory.mktemp("tiny_noflank"), 128, **stamp("tiny-noflank", "noflank", 128, 0))


@pytest.fixture(scope="module")
def tiny_plain(tokenizer, tmp_path_factory):
    return _tiny(tokenizer, tmp_path_factory.mktemp("tiny_plain"), 128)


def _kw(ckpt, manifest, tok, fa, out):
    return dict(bert_checkpoint=ckpt, n_fragment_file=str(manifest), n_fragment_per_file=4, index=0,
                output_dir=str(out), tokenizer_file=str(tok), ref_genome_file=str(fa),
                batch_size=2, seed=42, target_layer=2)


def _read(h5):
    with h5py.File(h5) as f:
        return (f["embeddings/bert_window_embeddings"][:], f["fragment_info/n_token_left"][:],
                f["fragment_info/line_number"][:], dict(f["metadata"].attrs))


def test_stamped_checkpoint_sets_input_without_flags(needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest,
                                                     tiny_noflank, tmp_path):
    ee.evaluate_bed_file(**_kw(tiny_noflank, tiny_manifest, tokenizer_path, tiny_fa, tmp_path / "auto"))
    ee.evaluate_bed_file(**_kw(tiny_noflank, tiny_manifest, tokenizer_path, tiny_fa, tmp_path / "explicit"),
                         ablation="noflank", max_token_length=128, min_flank_token_length=0)
    a, b = _read(tmp_path / "auto/h5_files/CFDNA_sample_x.h5"), _read(tmp_path / "explicit/h5_files/CFDNA_sample_x.h5")
    D = ee.EMBED_DIM
    assert (a[0][:, :3 * D] == 0).all() and (a[0][:, 6 * D:9 * D] == 0).all()   # body only
    assert (a[1] == 0).all()
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[2], b[2])
    assert a[3]["leaf1_model"] == "tiny-noflank" and a[3]["ablation"] == "noflank"
    assert a[3]["max_token_length"] == 128


@pytest.mark.parametrize("kw", [dict(ablation="none"), dict(max_token_length=64), dict(min_flank_token_length=8)])
def test_stamped_checkpoint_refuses_other_input(needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest,
                                                tiny_noflank, tmp_path, kw):
    with pytest.raises(ValueError, match="tiny-noflank"):
        ee.evaluate_bed_file(**_kw(tiny_noflank, tiny_manifest, tokenizer_path, tiny_fa, tmp_path), **kw)
    assert not (tmp_path / "h5_files" / "CFDNA_sample_x.h5").exists()


def test_unstamped_checkpoint_keeps_release_behavior(needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest,
                                                     tiny_plain, tmp_path):
    # unstamped + no flags = release defaults (2048), which a 128-position model can't run
    with pytest.raises(ValueError, match="exceeds"):
        ee.evaluate_bed_file(**_kw(tiny_plain, tiny_manifest, tokenizer_path, tiny_fa, tmp_path / "a"))
    ee.evaluate_bed_file(**_kw(tiny_plain, tiny_manifest, tokenizer_path, tiny_fa, tmp_path / "b"),
                         max_token_length=128, min_flank_token_length=8)
    emb, left, _, meta = _read(tmp_path / "b/h5_files/CFDNA_sample_x.h5")
    assert set(meta) == {"bed_file", "sample_id", "modality", "group", "n_fragments_total",
                         "n_fragments_sampled", "bert_checkpoint", "seed", "batch_size", "total_batches",
                         "timestamp", "max_token_length", "min_flank_token_length"}
    # the released schema plus only the two non-default input settings, so a resume can check them
    assert (meta["max_token_length"], meta["min_flank_token_length"]) == (128, 8)
    assert (left > 0).any()


def test_resume_refuses_other_window(needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest, tiny_plain,
                                     tmp_path):
    ee.evaluate_bed_file(**_kw(tiny_plain, tiny_manifest, tokenizer_path, tiny_fa, tmp_path),
                         max_token_length=128, min_flank_token_length=8)
    with h5py.File(tmp_path / "h5_files" / "CFDNA_sample_x.h5", "r+") as f:
        f["progress"].attrs["completed"] = False
    with pytest.raises(ValueError, match="min_flank_token_length"):
        ee.evaluate_bed_file(**_kw(tiny_plain, tiny_manifest, tokenizer_path, tiny_fa, tmp_path),
                             max_token_length=128, min_flank_token_length=16)


def test_md5_is_computed_once_per_file(tmp_path, monkeypatch):
    w = tmp_path / "model.safetensors"
    w.write_bytes(b"\1" * 4096)
    calls = []
    real = M._md5_uncached
    monkeypatch.setattr(M, "_md5_uncached", lambda p, c: calls.append(p) or real(p, c))
    assert M._md5(w) == M._md5(w)
    assert len(calls) == 1
    w.write_bytes(b"\2" * 8192)          # a changed file is hashed again
    M._md5(w)
    assert len(calls) == 2


def test_resume_refuses_other_model(needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest, tiny_noflank,
                                    tiny_plain, tmp_path):
    ee.evaluate_bed_file(**_kw(tiny_noflank, tiny_manifest, tokenizer_path, tiny_fa, tmp_path))
    h5 = tmp_path / "h5_files" / "CFDNA_sample_x.h5"
    with h5py.File(h5, "r+") as f:
        f["progress"].attrs["completed"] = False
        f["metadata"].attrs["ablation"] = "none"       # get past the older ablation check
    with pytest.raises(ValueError, match="Model mismatch"):
        ee.evaluate_bed_file(**_kw(tiny_plain, tiny_manifest, tokenizer_path, tiny_fa, tmp_path),
                             max_token_length=128, min_flank_token_length=8, ablation="none")


# ---------- CLI ----------------------------------------------------------------------------------

def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["leaf1-extract", "--n_fragment_file", "x.csv", "--index", "0",
                                      "--output_dir", "out", *argv])
    called = {}
    monkeypatch.setattr(ee, "evaluate_bed_file", lambda **kw: called.update(kw))
    ee.main()
    return called


def test_cli_model_and_checkpoint_are_exclusive(monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--model", "leaf1", "--bert_checkpoint", "checkpoint")


def test_cli_unknown_model(monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--model", "leaf1-noflank")


def test_cli_bert_checkpoint_passes_through_unchanged(monkeypatch):
    kw = _main(monkeypatch, "--bert_checkpoint", "/some/ckpt")
    assert kw["bert_checkpoint"] == "/some/ckpt"
    assert kw["ablation"] is None and kw["max_token_length"] is None and kw["min_flank_token_length"] is None


def _fake_registry(monkeypatch, tmp_path, name, cfg_extra, weights=b"\0" * 4096):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "distilbert", **cfg_extra}))
    (d / "model.safetensors").write_bytes(weights)
    import hashlib
    models = dict(M.MODELS)
    models[name] = dict(M.MODELS["leaf1-noflank-s42"], path=name, md5=hashlib.md5(weights).hexdigest())
    monkeypatch.setattr(M, "MODELS", models)
    monkeypatch.setattr(M, "REPO_ROOT", tmp_path)
    return d


def test_cli_model_resolves_path(monkeypatch, tmp_path):
    d = _fake_registry(monkeypatch, tmp_path, "leaf1-noflank-s42", stamp("leaf1-noflank-s42", "noflank", 512, 0))
    kw = _main(monkeypatch, "--model", "leaf1-noflank-s42")
    assert kw["bert_checkpoint"] == str(d)


def test_cli_model_dir_with_wrong_stamp(monkeypatch, tmp_path):
    _fake_registry(monkeypatch, tmp_path, "leaf1-noflank-s42", stamp("leaf1-noflank-s43", "noflank", 512, 0))
    with pytest.raises(SystemExit, match="holds leaf1-noflank-s43"):
        _main(monkeypatch, "--model", "leaf1-noflank-s42")


def test_cli_model_lfs_pointer(monkeypatch, tmp_path):
    _fake_registry(monkeypatch, tmp_path, "leaf1-noflank-s42", stamp("leaf1-noflank-s42", "noflank", 512, 0),
                   weights=b"version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(RuntimeError, match="git lfs pull"):
        _main(monkeypatch, "--model", "leaf1-noflank-s42")


def test_cli_default_model_is_released_and_md5_checked(monkeypatch, tmp_path):
    d = tmp_path / "checkpoint"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "distilbert"}))
    (d / "model.safetensors").write_bytes(b"\0" * 4096)
    monkeypatch.setattr(M, "REPO_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="md5"):       # zeros are not the released weights
        _main(monkeypatch)
    import hashlib
    models = dict(M.MODELS)
    models["leaf1"] = dict(M.MODELS["leaf1"], md5=hashlib.md5(b"\0" * 4096).hexdigest())
    monkeypatch.setattr(M, "MODELS", models)
    assert _main(monkeypatch)["bert_checkpoint"] == str(d)
    # a stamped checkpoint sitting at the default path is refused
    (d / "config.json").write_text(json.dumps({"model_type": "distilbert",
                                               **stamp("leaf1-nocristiano", "none", 2048, 100)}))
    with pytest.raises(SystemExit, match="holds leaf1-nocristiano"):
        _main(monkeypatch)


def test_cli_missing_checkpoint_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(M, "REPO_ROOT", tmp_path)
    with pytest.raises(SystemExit, match="no checkpoint"):
        _main(monkeypatch, "--model", "leaf1-nocristiano")


# ---------- MIL inference encoder guard ----------------------------------------------------------

def _h5(path, **meta):
    with h5py.File(path, "w") as f:
        g = f.create_group("metadata")
        for k, v in meta.items():
            g.attrs[k] = v
    return path


def test_mil_encoder_guard(tmp_path):
    from leaf1.downstream.mil import inference as inf
    rel = _h5(tmp_path / "rel.h5", sample_id="a")
    nf = _h5(tmp_path / "nf.h5", sample_id="b", leaf1_model="leaf1-noflank-s42", ablation="noflank")
    nc = _h5(tmp_path / "nc.h5", sample_id="c", leaf1_model="leaf1-nocristiano")
    unreg = _h5(tmp_path / "u.h5", sample_id="d", ablation="nocut")
    inf.check_encoders([rel], "leaf1")
    for bad in (nf, nc, unreg):
        with pytest.raises(ValueError, match="not valid for other encoders"):
            inf.check_encoders([rel, bad], "leaf1")
    inf.check_encoders([nf], "leaf1-noflank-s42")
    with pytest.raises(ValueError):
        inf.check_encoders([rel], "leaf1-noflank-s42")
    # an unstamped ablation H5: the error names the exact --encoder_model value, and it works
    with pytest.raises(ValueError, match="--encoder_model unstamped-nocut"):
        inf.check_encoders([unreg], "leaf1")
    inf.check_encoders([unreg], "unstamped-nocut")


def test_cli_unknown_ablation_rejected(monkeypatch):
    with pytest.raises(SystemExit):
        _main(monkeypatch, "--ablation", "noflnak")


def test_mil_predict_cli_default_encoder():
    from leaf1.downstream.mil import inference as inf
    a = inf.parse_args(["--mil_checkpoint", "m", "--hdf5_file", "x.h5", "--output_dir", "o"])
    assert a.encoder_model == "leaf1"


# ---------- the released checkpoint is held to its input format (recognised by md5) -------------

def _as_released(monkeypatch, d):
    import hashlib
    models = dict(M.MODELS)
    models["leaf1"] = dict(M.MODELS["leaf1"], md5=hashlib.md5((Path(d) / "model.safetensors").read_bytes()).hexdigest())
    monkeypatch.setattr(M, "MODELS", models)


@pytest.mark.parametrize("kw", [dict(ablation="noflank"), dict(max_token_length=512),
                                dict(min_flank_token_length=0), dict(ablation="nocut", max_token_length=2048)])
def test_released_weights_refuse_other_input(monkeypatch, tiny_plain, kw):
    from transformers import DistilBertConfig
    _as_released(monkeypatch, tiny_plain)
    with pytest.raises(ValueError, match="released LEAF-1 encoder"):
        M.resolve_input(DistilBertConfig.from_pretrained(tiny_plain), tiny_plain, **kw)


def test_released_weights_accept_release_input(monkeypatch, tiny_plain):
    from transformers import DistilBertConfig
    _as_released(monkeypatch, tiny_plain)
    cfg = DistilBertConfig.from_pretrained(tiny_plain)
    assert M.resolve_input(cfg, tiny_plain) == (None, "none", 2048, 100)        # no leaf1_model for the release
    assert M.resolve_input(cfg, tiny_plain, ablation="none", max_token_length=2048,
                           min_flank_token_length=100) == (None, "none", 2048, 100)


def test_other_unstamped_weights_take_given_input(tiny_plain):
    from transformers import DistilBertConfig
    cfg = DistilBertConfig.from_pretrained(tiny_plain)
    assert M.resolve_input(cfg, tiny_plain, ablation="noflank", max_token_length=128, min_flank_token_length=0) == \
        (None, "noflank", 128, 0)


def test_real_released_checkpoint_refuses_noflank():
    if not _weights_pulled("leaf1"):
        pytest.skip("released weights not pulled from Git LFS")
    from transformers import DistilBertConfig
    d = M.model_path("leaf1")
    cfg = DistilBertConfig.from_pretrained(str(d))
    assert M.resolve_input(cfg, d) == (None, "none", 2048, 100)
    with pytest.raises(ValueError, match="released LEAF-1 encoder"):
        M.resolve_input(cfg, d, ablation="noflank")
