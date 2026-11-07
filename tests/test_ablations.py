"""Pretraining ablations: each arm must change exactly one thing, and nothing else.

Run on the tiny fixtures.
"""

import numpy as np
import pytest
import torch

from leaf1.common import get_fragment as gf
from leaf1.common.collator import DataCollatorForDNAFragmentModelingNoDecay
from leaf1.common.pool import compute_window_embeddings, compute_window_embeddings_from_counts

SEQ_LEN = 64            # small enough that the 5 kb fixture windows are trimmed
MIN_FLANK = 4
SEEDS = range(6)


def _bases(tokens):
    return "".join("N" if t == "[PAD]" else t for t in tokens)


def _draws(tokenizer, tiny_bed, tiny_fa, ablation, seq_len=SEQ_LEN, min_flank=MIN_FLANK):
    out = []
    n_lines = sum(1 for _ in open(tiny_bed))
    for seed in SEEDS:
        get = gf.initialize_iterable_get_fragment(tokenizer, str(tiny_bed), str(tiny_fa), seq_len,
                                                  min_flank, seed=seed, ablation=ablation)
        for ln in range(n_lines):
            try:
                out.append(get(ln))
            except ValueError:
                pass            # fixture lines that cannot fit any layout raise identically in every arm
    assert out, "no fixture fragment could be drawn"
    return out


def _model_tokens(tokenizer, final_sequence):
    """Tokens exactly as the datasets feed them: [CLS] [MOD] + re-tokenized final_sequence."""
    ids = tokenizer("[CFDNA]" + final_sequence, truncation=True, padding="max_length",
                    max_length=SEQ_LEN)["input_ids"]
    return tokenizer.convert_ids_to_tokens(ids)


def test_unknown_ablation_rejected(tokenizer, tiny_bed, tiny_fa):
    with pytest.raises(ValueError):
        gf.initialize_iterable_get_fragment(tokenizer, str(tiny_bed), str(tiny_fa), SEQ_LEN, MIN_FLANK,
                                            seed=0, ablation="nocutt")


@pytest.mark.parametrize("ablation", ["eqmask", "nomod"])
def test_input_unchanged_for_collator_and_modality_arms(tokenizer, tiny_bed, tiny_fa, ablation):
    """eqmask and nomod do not touch the sequence: identical to the release path."""
    assert _draws(tokenizer, tiny_bed, tiny_fa, "none") == _draws(tokenizer, tiny_bed, tiny_fa, ablation)


def test_nocut_has_no_cut_and_reconstructs_window(tokenizer, tiny_bed, tiny_fa):
    for d in _draws(tokenizer, tiny_bed, tiny_fa, "nocut"):
        toks = _model_tokens(tokenizer, d["final_sequence"])
        assert "[CUT]" not in toks
        assert toks[0] == "[CLS]" and toks[1] == "[CFDNA]"
        n = d["n_token_left"] + d["n_token_frag"] + d["n_token_right"]
        assert len(toks) == SEQ_LEN
        assert all(t == "[PAD]" for t in toks[2 + n:]), "only padding after the window"
        # the counts index the tokens the model really sees, and the body tokens cover the fragment
        body = _bases(toks[2 + d["n_token_left"]: 2 + d["n_token_left"] + d["n_token_frag"]])
        assert d["raw_fragment_dna"] in body
        left = _bases(toks[2: 2 + d["n_token_left"]])
        right = _bases(toks[2 + d["n_token_left"] + d["n_token_frag"]: 2 + n])
        # left/right tokens lie entirely outside the fragment (straddling tokens count as body)
        window = left + body + right
        lo = window.index(d["raw_fragment_dna"])
        assert lo >= len(left) and lo + len(d["raw_fragment_dna"]) <= len(left) + len(body)


def test_nocut_tokenization_is_joint_not_piecewise(tokenizer):
    """The boundary can't be read off the BPE segmentation: the window is tokenized in one go."""
    rng = np.random.default_rng(0)
    seq = "".join(rng.choice(list("ACGT"), 600))
    out, *_ = gf.process_sequence(seq, 250, 420, tokenizer, 4096, 0, np.random.default_rng(0),
                                  ablation="nocut")
    joint = tokenizer.tokenize(seq)
    assert out.replace("[PAD]", "") == "".join(joint)
    assert [t for t in tokenizer.tokenize(out) if t != "[PAD]"] == joint


def test_nocut_handles_iupac_and_n(tokenizer):
    """IUPAC codes (one [UNK] each) and N runs ([PAD]) still map back to the right bases."""
    rng = np.random.default_rng(1)
    seq = list("".join(rng.choice(list("ACGT"), 600)))
    for i, c in [(30, "R"), (260, "Y"), (300, "N"), (301, "N"), (500, "M")]:
        seq[i] = c
    seq = "".join(seq)
    out, _, nl, nf, nr = gf.process_sequence(seq, 250, 420, tokenizer, 4096, 0, np.random.default_rng(0),
                                             ablation="nocut")
    toks = [t for t in tokenizer.tokenize(out)][: nl + nf + nr]
    assert sum(gf._token_bases(t) for t in toks) == len(seq)
    assert sum(gf._token_bases(t) for t in toks[:nl]) <= 250
    assert sum(gf._token_bases(t) for t in toks[:nl + nf]) >= 420


def test_noflank_is_body_only(tokenizer, tiny_bed, tiny_fa):
    for d in _draws(tokenizer, tiny_bed, tiny_fa, "noflank", seq_len=SEQ_LEN, min_flank=0):
        assert d["n_token_left"] == 0 and d["n_token_right"] == 0
        assert d["adjusted_start"] == d["fragment_start"] and d["adjusted_end"] == d["fragment_end"]
        toks = _model_tokens(tokenizer, d["final_sequence"])
        assert toks[2] == "[CUT]" and toks[3 + d["n_token_frag"]] == "[CUT]"
        assert _bases(toks[3: 3 + d["n_token_frag"]]) == d["raw_fragment_dna"]
        assert all(t == "[PAD]" for t in toks[4 + d["n_token_frag"]:])


def test_nomod_placeholder_constant_and_never_masked(tokenizer):
    for real in ["[CFDNA]", "[TCGA]", "[SCATAC]"]:
        assert gf.modality_slot_token(real, "nomod") == gf.NOMOD_PLACEHOLDER
        assert gf.modality_slot_token(real, "none") == real
    ids = tokenizer(gf.NOMOD_PLACEHOLDER + "ACGTACGT[CUT]ACGTAC[CUT]ACGT", padding="max_length",
                    max_length=32)["input_ids"]
    assert ids[1] == tokenizer.sep_token_id
    col = DataCollatorForDNAFragmentModelingNoDecay(tokenizer, 0.99, 0.99)
    batch = [{"input_ids": torch.tensor(ids), "attention_mask": torch.ones(32, dtype=torch.long)}] * 64
    assert (col(batch)["labels"][:, 1] == -100).all()


def _cut_batch(tokenizer, n=4000, seq_len=40, with_cut=True):
    cut = tokenizer.convert_tokens_to_ids("[CUT]")
    base = tokenizer.convert_tokens_to_ids("ACGT") if "ACGT" in tokenizer.get_vocab() else 100
    rows = []
    for _ in range(n):
        ids = [tokenizer.cls_token_id, tokenizer.convert_tokens_to_ids("[CFDNA]")] + [base] * (seq_len - 2)
        if with_cut:
            ids[10], ids[25] = cut, cut
        rows.append({"input_ids": torch.tensor(ids), "attention_mask": torch.ones(seq_len, dtype=torch.long)})
    return rows


@pytest.mark.parametrize("cut_p", [0.30, 0.15])
def test_measured_masking_rates(tokenizer, cut_p):
    """Release (0.30) and eqmask (0.15): [CUT] pair masked jointly at cut_p, bases at 0.15."""
    torch.manual_seed(0)
    col = DataCollatorForDNAFragmentModelingNoDecay(tokenizer, 0.15, cut_p)
    lab = col(_cut_batch(tokenizer))["labels"]
    m10, m25 = lab[:, 10] != -100, lab[:, 25] != -100
    assert (m10 == m25).all(), "the two [CUT] tokens must be masked jointly"
    assert abs(m10.float().mean().item() - cut_p) < 0.03
    base_cols = [c for c in range(2, 40) if c not in (10, 25)]
    assert abs((lab[:, base_cols] != -100).float().mean().item() - 0.15) < 0.01


def test_collator_nocut_batch(tokenizer):
    torch.manual_seed(0)
    col = DataCollatorForDNAFragmentModelingNoDecay(tokenizer, 0.15, 0.30)
    lab = col(_cut_batch(tokenizer, n=2000, with_cut=False))["labels"]
    assert abs((lab[:, 2:] != -100).float().mean().item() - 0.15) < 0.01


def test_collator_rejects_partial_cut(tokenizer):
    rows = _cut_batch(tokenizer, n=2)
    rows[1]["input_ids"] = rows[1]["input_ids"].clone()
    rows[1]["input_ids"][25] = rows[1]["input_ids"][24]
    with pytest.raises(ValueError):
        DataCollatorForDNAFragmentModelingNoDecay(tokenizer)(rows)


def test_pool_from_counts_regions():
    """Counts-based pooling covers [2, 2+nl), [2+nl, 2+nl+nf) and the next nr tokens."""
    from leaf1.common.pool import AdaptiveMultiWindowPooler
    torch.manual_seed(0)
    B, L, D = 4, 40, 8
    e = torch.randn(B, L, D)
    nl, nf, nr = torch.tensor([0, 3, 10, 5]), torch.tensor([4, 9, 2, 20]), torch.tensor([5, 0, 12, 13])
    got = compute_window_embeddings_from_counts(e, nl, nf, nr)
    pos = torch.arange(L).expand(B, L)
    lo, hi = 2 + nl[:, None], 2 + nl[:, None] + nf[:, None]
    pool = AdaptiveMultiWindowPooler(3)
    W = 3 * D
    assert torch.allclose(got[:, :W], pool(e, (pos >= 2) & (pos < lo)))
    assert torch.allclose(got[:, W:2 * W], pool(e, (pos >= lo) & (pos < hi)))
    assert torch.allclose(got[:, 2 * W:3 * W], pool(e, (pos >= hi) & (pos < hi + nr[:, None])))
    assert (got[:, 9 * D:11 * D] == 0).all()                       # no CUT tokens -> zero chunks
    assert torch.allclose(got[:, 11 * D:12 * D], e[:, 0]) and torch.allclose(got[:, 12 * D:], e[:, 1])


def test_pretrain_eqmask_sets_cut_rate(monkeypatch):
    import sys
    from leaf1.train import pretrain
    base = ["x", "--train_csv", "a", "--val_csv", "b", "--output_root", "o", "--log_root", "l"]
    monkeypatch.setattr(sys, "argv", base + ["--ablation", "eqmask"])
    assert pretrain.parse_args().cut_mlm_probability == 0.15
    monkeypatch.setattr(sys, "argv", base + ["--ablation", "eqmask", "--cut_mlm_probability", "0.5"])
    with pytest.raises(SystemExit):
        pretrain.parse_args()
    monkeypatch.setattr(sys, "argv", base + ["--ablation", "eqmask", "--cut_mlm_probability", "0.30"])
    with pytest.raises(SystemExit):     # an explicit 0.30 conflicts too; it used to be overwritten
        pretrain.parse_args()
    monkeypatch.setattr(sys, "argv", base)
    assert pretrain.parse_args().cut_mlm_probability == 0.30
    monkeypatch.setattr(sys, "argv", base + ["--cut_mlm_probability", "0.4"])
    assert pretrain.parse_args().cut_mlm_probability == 0.4


@pytest.mark.parametrize("ablation", ["nocut", "eqmask", "noflank", "nomod"])
def test_pretrain_stamps_ablation_checkpoints(monkeypatch, tmp_path, ablation):
    """A checkpoint pretrained with --ablation carries its input format, so extraction
    without flags uses it and a conflicting flag is refused."""
    import sys
    from transformers import DistilBertConfig
    from leaf1.train import pretrain
    from leaf1.infer import models as M
    base = ["x", "--train_csv", "a", "--val_csv", "b", "--output_root", "o", "--log_root", "l",
            "--ablation", ablation, "--seq_len", "512", "--min_flank", "0"]
    monkeypatch.setattr(sys, "argv", base)
    cfg = DistilBertConfig()
    pretrain.stamp_input_format(cfg, pretrain.parse_args())
    cfg.save_pretrained(tmp_path)
    cfg = DistilBertConfig.from_pretrained(tmp_path)
    assert M.resolve_input(cfg) == (f"custom-{ablation}", ablation, 512, 0)
    with pytest.raises(ValueError, match="pretrained with"):
        M.resolve_input(cfg, ablation="none")


def test_pretrain_noflank_stamps_zero_flank(monkeypatch):
    import sys
    from transformers import DistilBertConfig
    from leaf1.train import pretrain
    monkeypatch.setattr(sys, "argv", ["x", "--train_csv", "a", "--val_csv", "b", "--output_root", "o",
                                      "--log_root", "l", "--ablation", "noflank", "--seq_len", "512"])
    cfg = DistilBertConfig()
    pretrain.stamp_input_format(cfg, pretrain.parse_args())   # default --min_flank 100
    assert cfg.leaf1_min_flank_token_length == 0


def test_pretrain_release_recipe_stays_unstamped(monkeypatch):
    import sys
    from transformers import DistilBertConfig
    from leaf1.train import pretrain
    from leaf1.infer import models as M
    monkeypatch.setattr(sys, "argv", ["x", "--train_csv", "a", "--val_csv", "b", "--output_root", "o",
                                      "--log_root", "l"])
    cfg = DistilBertConfig()
    pretrain.stamp_input_format(cfg, pretrain.parse_args())
    assert M.stamped_input(cfg) == (None, None)


# ---------- end-to-end extraction per arm (tiny random model, fixtures) ----------------------

@pytest.fixture(scope="module")
def tiny_ckpt(tokenizer, tmp_path_factory):
    from transformers import DistilBertConfig, DistilBertForMaskedLM
    from leaf1.infer import extract_embeddings as ee
    cfg = DistilBertConfig(max_position_embeddings=128, vocab_size=tokenizer.vocab_size, n_layers=2,
                           n_heads=12, dim=ee.EMBED_DIM, hidden_dim=512,
                           pad_token_id=tokenizer.pad_token_id)
    torch.manual_seed(0)
    out = tmp_path_factory.mktemp("abl_ckpt")
    DistilBertForMaskedLM(cfg).save_pretrained(out)
    return str(out)


@pytest.mark.parametrize("ablation", ["none", "nocut", "eqmask", "noflank", "nomod"])
def test_extraction_per_arm(needs_bedtools, tokenizer, tokenizer_path, tiny_fa, tiny_manifest,
                            tiny_ckpt, tmp_path, ablation):
    import h5py
    from transformers import DistilBertForMaskedLM
    from leaf1.infer import extract_embeddings as ee
    out = tmp_path / ablation
    kw = dict(bert_checkpoint=tiny_ckpt, n_fragment_file=str(tiny_manifest), n_fragment_per_file=4,
              index=0, output_dir=str(out), tokenizer_file=str(tokenizer_path), ref_genome_file=str(tiny_fa),
              batch_size=2, seed=42, target_layer=2, max_token_length=128,
              min_flank_token_length=0 if ablation == "noflank" else 8, ablation=ablation)
    ee.evaluate_bed_file(store_body_mean=True, **kw)
    with h5py.File(out / "h5_files" / "CFDNA_sample_x.h5") as f:
        assert bool(f["progress"].attrs["completed"])
        emb, bm = f["embeddings/bert_window_embeddings"][:], f["embeddings/body_mean"][:]
        assert emb.shape == (4, ee.WINDOW_EMBED_DIM) and bm.shape == (4, ee.EMBED_DIM)
        assert np.isfinite(emb).all() and np.isfinite(bm).all() and not np.allclose(bm, 0)
        assert f["metadata"].attrs.get("ablation", "none") == ablation
        D = ee.EMBED_DIM
        if ablation == "nocut":
            assert (emb[:, 9 * D:11 * D] == 0).all()                 # no CUT tokens
        if ablation == "noflank":
            assert (emb[:, :3 * D] == 0).all() and (emb[:, 6 * D:9 * D] == 0).all()
            assert (f["fragment_info/n_token_left"][:] == 0).all()
        lines = f["fragment_info/line_number"][:]

    # independent recomputation of body_mean for the first fragment
    ds = ee.SingleFileDNAFragmentDataset(str(tokenizer_path), str(tiny_fa),
                                         dict(bed_file=str(tiny_manifest.parent / "tiny.bed"), sample_id="x",
                                              modality="CFDNA", group="g", n_fragment=6),
                                         n_fragment_per_file=4, max_token_length=128,
                                         min_flank_token_length=kw["min_flank_token_length"], seed=42,
                                         ablation=ablation)
    item = ds[0]
    assert item["line_number"] == lines[0]
    model = DistilBertForMaskedLM.from_pretrained(tiny_ckpt).eval()
    with torch.no_grad():
        hs = model.distilbert(input_ids=item["input_ids"][None], attention_mask=item["attention_mask"][None],
                              output_hidden_states=True).hidden_states[2][0]
    toks = tokenizer.convert_ids_to_tokens(item["input_ids"])
    if ablation == "nocut":
        lo = 2 + item["n_token_left"]; hi = lo + item["n_token_frag"]
    else:
        c = [i for i, t in enumerate(toks) if t == "[CUT]"]; lo, hi = c[0] + 1, c[1]
    expected = hs[lo:hi].mean(0).numpy()
    assert np.allclose(bm[0].astype(np.float32), expected, atol=2e-2)
    if ablation == "nomod":
        assert toks[1] == "[SEP]"


def test_store_body_mean_off_keeps_release_schema(needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest,
                                                  tiny_ckpt, tmp_path):
    import h5py
    from leaf1.infer import extract_embeddings as ee
    ee.evaluate_bed_file(bert_checkpoint=tiny_ckpt, n_fragment_file=str(tiny_manifest), n_fragment_per_file=4,
                         index=0, output_dir=str(tmp_path), tokenizer_file=str(tokenizer_path),
                         ref_genome_file=str(tiny_fa), batch_size=2, seed=42, target_layer=2,
                         max_token_length=128, min_flank_token_length=8)
    with h5py.File(tmp_path / "h5_files" / "CFDNA_sample_x.h5") as f:
        assert list(f["embeddings"].keys()) == ["bert_window_embeddings"]
        assert "ablation" not in f["metadata"].attrs and "store_body_mean" not in f["metadata"].attrs


# ---------- extraction controls: matched fragments + paired strand -----------------------

@pytest.mark.parametrize("ablation", ["nocut", "noflank", "nomod"])
def test_per_fragment_rng_pairs_strand_across_arms(tokenizer, tiny_bed, tiny_fa, ablation):
    """With per_fragment_rng every arm gets the same strand for the same line, in any call order."""
    def strands(abl, order):
        get = gf.initialize_iterable_get_fragment(tokenizer, str(tiny_bed), str(tiny_fa), SEQ_LEN,
                                                  0 if abl == "noflank" else MIN_FLANK, seed=42,
                                                  ablation=abl, per_fragment_rng=True)
        out = {}
        for ln in order:
            try:
                out[ln] = get(ln)["reverse_complemented"]
            except ValueError:
                pass
        return out
    n = sum(1 for _ in open(tiny_bed))
    ref = strands("none", range(n))
    assert ref
    assert strands(ablation, range(n)) == ref
    assert strands("none", reversed(range(n))) == ref           # independent of order / worker layout


def test_per_fragment_rng_off_is_release_stream(tokenizer, tiny_bed, tiny_fa):
    n = sum(1 for _ in open(tiny_bed))
    a = gf.initialize_iterable_get_fragment(tokenizer, str(tiny_bed), str(tiny_fa), SEQ_LEN, MIN_FLANK, seed=3)
    b = gf.initialize_iterable_get_fragment(tokenizer, str(tiny_bed), str(tiny_fa), SEQ_LEN, MIN_FLANK, seed=3,
                                            per_fragment_rng=False)
    for ln in range(n):
        try:
            ra = a(ln)
        except ValueError:
            with pytest.raises(ValueError):
                b(ln)
            continue
        assert ra["final_sequence"] == b(ln)["final_sequence"]


def test_per_fragment_rng_requires_seed(tokenizer, tiny_bed, tiny_fa):
    with pytest.raises(ValueError):
        gf.initialize_iterable_get_fragment(tokenizer, str(tiny_bed), str(tiny_fa), SEQ_LEN, MIN_FLANK,
                                            seed=None, per_fragment_rng=True)


def test_presample_is_nested_fresh_seed_draw(tokenizer_path, tiny_fa, tiny_manifest, monkeypatch):
    from leaf1.infer import extract_embeddings as ee
    monkeypatch.setattr(ee, "initialize_iterable_get_fragment", lambda *a, **k: None)
    row = dict(bed_file=str(tiny_manifest.parent / "tiny.bed"), sample_id="x", modality="CFDNA", group="g",
               n_fragment=5_000_000)
    ds = ee.SingleFileDNAFragmentDataset(str(tokenizer_path), str(tiny_fa), row, n_fragment_per_file=1000,
                                         max_token_length=128, seed=42, presample=20000)
    pool = np.random.default_rng(42).choice(5_000_000, 20000, replace=False)
    expected = pool[np.random.default_rng(42).choice(20000, 1000, replace=False)]
    assert np.array_equal(ds.sampled_indices, expected)
    plain = ee.SingleFileDNAFragmentDataset(str(tokenizer_path), str(tiny_fa), row, n_fragment_per_file=1000,
                                            max_token_length=128, seed=42)
    assert np.array_equal(plain.sampled_indices, np.random.default_rng(42).choice(5_000_000, 1000, replace=False))


def test_ablation_requires_training_length(needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest, tiny_ckpt,
                                           tmp_path):
    from leaf1.infer import extract_embeddings as ee
    kw = dict(bert_checkpoint=tiny_ckpt, n_fragment_file=str(tiny_manifest), n_fragment_per_file=4, index=0,
              tokenizer_file=str(tokenizer_path), ref_genome_file=str(tiny_fa), batch_size=2, seed=42,
              target_layer=2, min_flank_token_length=0)
    with pytest.raises(ValueError, match="training length"):      # tiny_ckpt has max_position_embeddings=128
        ee.evaluate_bed_file(output_dir=str(tmp_path / "a"), max_token_length=64, ablation="noflank", **kw)
    with pytest.raises(ValueError, match="exceeds"):
        ee.evaluate_bed_file(output_dir=str(tmp_path / "b"), max_token_length=256, **kw)


def test_extraction_records_controls_and_refuses_mismatched_resume(
        needs_bedtools, tokenizer_path, tiny_fa, tiny_manifest, tiny_ckpt, tmp_path):
    import h5py
    from leaf1.infer import extract_embeddings as ee
    kw = dict(bert_checkpoint=tiny_ckpt, n_fragment_file=str(tiny_manifest), n_fragment_per_file=4, index=0,
              output_dir=str(tmp_path), tokenizer_file=str(tokenizer_path), ref_genome_file=str(tiny_fa),
              batch_size=2, seed=42, target_layer=2, max_token_length=128, min_flank_token_length=8,
              ablation="nomod", store_body_mean=True)
    ee.evaluate_bed_file(presample=6, per_fragment_rng=True, **kw)
    h5 = tmp_path / "h5_files" / "CFDNA_sample_x.h5"
    with h5py.File(h5, "r+") as f:
        m = f["metadata"].attrs
        assert m["presample"] == 6 and bool(m["per_fragment_rng"]) and m["max_token_length"] == 128
        f["progress"].attrs["completed"] = False                    # simulate an interrupted run
    with pytest.raises(ValueError, match="per_fragment_rng"):
        ee.evaluate_bed_file(presample=6, per_fragment_rng=False, **kw)
