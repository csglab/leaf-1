#!/usr/bin/env python3
"""LEAF-1 inference: per-fragment window-pooled embeddings to HDF5.

For a single sample (one row of the BED manifest), randomly sample up to
``n_fragment_per_file`` fragments, run them through the frozen encoder, pool each
fragment into the 9,984-dim window representation (3 windows over {left, body,
right} plus CLS, MOD, CUT1, CUT2), and write to HDF5 with full provenance.

Use ``--target_layer 4`` for the paper's frozen-layer-4 setup. Omit it to store
the embedding layer + all 12 transformer-layer outputs (much larger file).
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import DistilBertForMaskedLM, PreTrainedTokenizerFast

# Keep PyTorch from contending with the BLAS threads spun up per worker.
N_THREAD = 1
torch.set_num_threads(N_THREAD)
torch.set_num_interop_threads(N_THREAD)
for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[k] = f"{N_THREAD}"

# Ensure leaf1.* is importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from leaf1.common.get_fragment import initialize_iterable_get_fragment
from leaf1.common.modality import resolve_modality_token
from leaf1.common.pool import compute_window_embeddings


REPO_ROOT = Path(__file__).resolve().parents[2]
# Tokenizer ships inside the package so non-editable installs find it.
DEFAULT_TOKENIZER = str(Path(__file__).resolve().parents[1] / "data" / "tokenizer.json")
# Reference FASTA is too large to bundle; user supplies via --ref_genome_file
# or the repo's assets/ for editable installs.
DEFAULT_REF_FA = str(REPO_ROOT / "assets" / "refseq.genome.fa")

NUM_WINDOWS = 3
# Paper defaults - used when a caller doesn't supply config-derived values.
# Both create_h5_structure and process_batch derive their own values from the
# checkpoint config at runtime to support scaling-sweep models (e.g. dim != 768).
EMBED_DIM = 768
NUM_BERT_LAYERS = 12
# 3 segments * 3 windows + 4 special tokens (CUT1, CUT2, CLS, MOD) = 13 chunks of EMBED_DIM.
WINDOW_EMBED_DIM = (3 * NUM_WINDOWS + 4) * EMBED_DIM


def window_embed_dim_for(embed_dim: int) -> int:
    """Per-fragment representation size for a given encoder hidden dim."""
    return (3 * NUM_WINDOWS + 4) * embed_dim


class SingleFileDNAFragmentDataset(Dataset):
    def __init__(self, tokenizer_file, ref_genome_file, bed_file_row,
                 n_fragment_per_file=100000, max_token_length=2048,
                 min_flank_token_length=100, seed=42):
        self.ref_genome_file = ref_genome_file
        self.total_length = max_token_length
        self.min_flank = min_flank_token_length
        self.n_fragment_per_file = n_fragment_per_file
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        if not os.path.isfile(tokenizer_file):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_file}")

        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=tokenizer_file,
            model_max_length=self.total_length,
            padding_side="right",
            truncation_side="right",
            cls_token="[CLS]", bos_token="[CLS]", sep_token="[SEP]",
            eos_token="[SEP]", unk_token="[UNK]", mask_token="[MASK]", pad_token="[PAD]",
        )

        self.bed_file = bed_file_row["bed_file"]
        self.sample_id = bed_file_row.get("sample_id", "unknown")
        self.modality = bed_file_row.get("modality", "unknown")
        self.group = bed_file_row.get("group", "unknown")
        self.total_fragments_in_file = bed_file_row["n_fragment"]

        if not os.path.isfile(self.bed_file):
            raise FileNotFoundError(f"BED file not found: {self.bed_file}")

        n_to_sample = min(self.total_fragments_in_file, self.n_fragment_per_file)
        self.sampled_indices = self.rng.choice(
            self.total_fragments_in_file, size=n_to_sample, replace=False
        )

        self.get_fragment = initialize_iterable_get_fragment(
            self.tokenizer, self.bed_file, self.ref_genome_file,
            self.total_length, self.min_flank, seed=self.seed,
        )

        print(f"Initialized dataset for {self.bed_file}")
        print(f"Total fragments: {self.total_fragments_in_file}, Sampled: {len(self.sampled_indices)}")

    def __len__(self):
        return len(self.sampled_indices)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.sampled_indices):
            raise IndexError("Index out of bounds")

        line_number = self.sampled_indices[idx]
        fragment = self.get_fragment(line_number)

        modality_token = resolve_modality_token(self.tokenizer, self.modality)
        sequence_with_modality = modality_token + fragment["final_sequence"]

        tokenized_input = self.tokenizer(
            sequence_with_modality, return_special_tokens_mask=True,
            truncation=True, padding="max_length", max_length=self.total_length,
            return_tensors="pt",
        )

        del tokenized_input["special_tokens_mask"]
        if "token_type_ids" in tokenized_input:
            del tokenized_input["token_type_ids"]

        data = {key: value.squeeze(0) for key, value in tokenized_input.items()}
        data.update({
            "chromosome": fragment["chromosome"],
            "fragment_start": fragment["fragment_start"],
            "fragment_end": fragment["fragment_end"],
            "adjusted_start": fragment["adjusted_start"],
            "adjusted_end": fragment["adjusted_end"],
            "n_token_left": fragment["n_token_left"],
            "n_token_frag": fragment["n_token_frag"],
            "n_token_right": fragment["n_token_right"],
            "n_freq": fragment["relative_frequency_of_n"],
            "reverse_complemented": fragment["reverse_complemented"],
            "line_number": line_number,
        })
        return data


def create_h5_structure(h5_file, n_samples, dataset_info, *,
                         embed_dim: int = EMBED_DIM,
                         num_bert_layers: int = NUM_BERT_LAYERS):
    """Build the on-disk H5 schema.

    `embed_dim` and `num_bert_layers` should come from the checkpoint config so
    scaling-sweep models (e.g. dim=256, n_layers=6) get the right shapes.
    """
    window_embed_dim = window_embed_dim_for(embed_dim)

    emb_group = h5_file.create_group("embeddings")

    if dataset_info.get("target_layer") is not None:
        emb_group.create_dataset(
            "bert_window_embeddings",
            shape=(n_samples, window_embed_dim),
            dtype=np.float16,
        )
        emb_group.attrs["target_layer"] = dataset_info["target_layer"]
        emb_group.attrs["num_layers"] = 1
    else:
        emb_group.create_dataset(
            "bert_window_embeddings",
            shape=(n_samples, num_bert_layers + 1, window_embed_dim),
            dtype=np.float16,
        )
        emb_group.attrs["num_layers"] = num_bert_layers + 1

    emb_group.attrs["num_windows"] = NUM_WINDOWS
    emb_group.attrs["embed_dim"] = embed_dim
    emb_group.attrs["window_embed_dim"] = window_embed_dim
    emb_group.attrs["layer_names"] = json.dumps(
        ["embedding"] + [f"layer_{i}" for i in range(num_bert_layers)]
    )
    emb_group.attrs["segment_offsets"] = json.dumps({
        "left": 0, "fragment": 3 * embed_dim, "right": 6 * embed_dim,
        "cut1": 9 * embed_dim, "cut2": 10 * embed_dim,
        "cls": 11 * embed_dim, "mod": 12 * embed_dim,
    })

    frag_group = h5_file.create_group("fragment_info")
    str_dtype = h5py.special_dtype(vlen=str)

    frag_group.create_dataset("chromosome", shape=(n_samples,), dtype=str_dtype)
    for name in ["fragment_start", "fragment_end", "adjusted_start", "adjusted_end", "line_number"]:
        frag_group.create_dataset(name, shape=(n_samples,), dtype="int64")
    frag_group.create_dataset("reverse_complemented", shape=(n_samples,), dtype="bool")
    for name in ["n_token_left", "n_token_frag", "n_token_right"]:
        frag_group.create_dataset(name, shape=(n_samples,), dtype="int32")
    frag_group.create_dataset("n_freq", shape=(n_samples,), dtype="float32")

    meta_group = h5_file.create_group("metadata")
    for k, v in dataset_info.items():
        if k != "target_layer":
            meta_group.attrs[k] = v
    meta_group.attrs["timestamp"] = datetime.now().isoformat()

    prog_group = h5_file.create_group("progress")
    prog_group.attrs["current_batch_idx"] = -1
    prog_group.attrs["total_batches"] = dataset_info["total_batches"]
    prog_group.attrs["seed"] = dataset_info["seed"]
    prog_group.attrs["batch_size"] = dataset_info["batch_size"]
    prog_group.attrs["processed_count"] = 0
    prog_group.attrs["completed"] = False


def process_batch(bert_model, batch, device, target_layer=None):
    num_layers = len(bert_model.distilbert.transformer.layer)
    embed_dim = bert_model.config.dim
    window_embed_dim = window_embed_dim_for(embed_dim)

    if target_layer is not None:
        if target_layer < 0 or target_layer > num_layers:
            raise ValueError(
                f"target_layer={target_layer} out of range. Model has "
                f"{num_layers} transformer layers; valid range is 0 (embedding) "
                f"to {num_layers} (final layer)."
            )

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    n_token_left = batch["n_token_left"].to(device)
    n_token_frag = batch["n_token_frag"].to(device)

    # First [CUT] sits right after the left flank ([CLS] [MOD] left_flank [CUT]).
    start_pos = n_token_left + 2
    end_pos = start_pos + n_token_frag + 1
    batch_size = input_ids.size(0)

    # The paper H5 embeddings were generated through DistilBERT's legacy layer
    # API with a 2-D boolean mask. Keep that path as the first choice for
    # saved-embedding reproducibility. Newer Transformers releases route
    # DistilBERT through SDPA and require a 4-D additive mask, so fall back only
    # when the installed layer implementation rejects the legacy mask shape.
    legacy_attention_mask = attention_mask.bool()
    sdpa_attention_mask = None

    def run_layer(layer, states):
        nonlocal sdpa_attention_mask
        try:
            return layer(states, attn_mask=legacy_attention_mask, head_mask=None)[-1]
        except RuntimeError as exc:
            msg = str(exc)
            mask_shape_error = "expanded size of the tensor" in msg or "must match" in msg
            if not mask_shape_error:
                raise
            if sdpa_attention_mask is None:
                sdpa_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).float()
                sdpa_attention_mask = (1.0 - sdpa_attention_mask) * torch.finfo(torch.float32).min
            return layer(states, attn_mask=sdpa_attention_mask, head_mask=None)[-1]

    with torch.no_grad():
        hidden_states = bert_model.distilbert.embeddings(input_ids)

        if target_layer is None:
            all_window_embeddings = torch.zeros(
                batch_size, num_layers + 1, window_embed_dim,
                dtype=torch.float16, device="cpu",
            )
            all_window_embeddings[:, 0, :] = compute_window_embeddings(
                hidden_states, start_pos, end_pos, NUM_WINDOWS
            ).half().cpu()

            for layer_idx, layer in enumerate(bert_model.distilbert.transformer.layer):
                hidden_states = run_layer(layer, hidden_states)
                all_window_embeddings[:, layer_idx + 1, :] = compute_window_embeddings(
                    hidden_states, start_pos, end_pos, NUM_WINDOWS
                ).half().cpu()
        else:
            if target_layer == 0:
                layer_emb = compute_window_embeddings(
                    hidden_states, start_pos, end_pos, NUM_WINDOWS
                )
            else:
                for layer_idx, layer in enumerate(bert_model.distilbert.transformer.layer):
                    hidden_states = run_layer(layer, hidden_states)
                    if layer_idx + 1 == target_layer:
                        layer_emb = compute_window_embeddings(
                            hidden_states, start_pos, end_pos, NUM_WINDOWS
                        )
                        break
            all_window_embeddings = layer_emb.half().cpu()

    return all_window_embeddings.numpy()


def store_batch_data(h5_file, window_embeddings, batch_metadata, start_idx):
    batch_size = window_embeddings.shape[0]
    end_idx = start_idx + batch_size

    h5_file["embeddings/bert_window_embeddings"][start_idx:end_idx] = window_embeddings

    for i in range(batch_size):
        idx = start_idx + i
        h5_file["fragment_info/chromosome"][idx] = batch_metadata["chromosome"][i]
        for name in ["fragment_start", "fragment_end", "adjusted_start", "adjusted_end", "line_number"]:
            h5_file[f"fragment_info/{name}"][idx] = batch_metadata[name][i].item()
        h5_file["fragment_info/reverse_complemented"][idx] = batch_metadata["reverse_complemented"][i].item()
        for name in ["n_token_left", "n_token_frag", "n_token_right"]:
            h5_file[f"fragment_info/{name}"][idx] = batch_metadata[name][i].item()
        h5_file["fragment_info/n_freq"][idx] = batch_metadata["n_freq"][i].item()


def evaluate_bed_file(bert_checkpoint, n_fragment_file, n_fragment_per_file, index, output_dir,
                       tokenizer_file, ref_genome_file,
                       batch_size=32, seed=42, target_layer=None,
                       max_token_length=2048, min_flank_token_length=100):
    output_dir = Path(output_dir)
    h5_dir = output_dir / "h5_files"
    log_dir = output_dir / "logs"
    h5_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading fragment metadata from: {n_fragment_file}")
    fragment_df = pd.read_csv(n_fragment_file, sep=",")

    if index < 0 or index >= len(fragment_df):
        raise ValueError(f"Index {index} out of bounds for file with {len(fragment_df)} rows")

    bed_file_row = fragment_df.iloc[index]
    sample_id = bed_file_row["sample_id"]
    modality = bed_file_row["modality"]

    h5_filename = f"{modality}_{sample_id}.h5"
    h5_path = h5_dir / h5_filename
    log_path = log_dir / f"{modality}_{sample_id}.log"

    # Read the checkpoint config up front so the H5 schema matches the model's
    # hidden dim and layer count (supports scaling-sweep models).
    from transformers import DistilBertConfig
    model_config = DistilBertConfig.from_pretrained(bert_checkpoint)
    embed_dim = model_config.dim
    num_bert_layers = model_config.n_layers
    window_embed_dim = window_embed_dim_for(embed_dim)

    print(f"Output H5 file: {h5_path}")
    print(f"Output log file: {log_path}")
    print(f"\nWindow Embedding Configuration:")
    print(f"  - Number of windows per region : {NUM_WINDOWS}")
    print(f"  - BERT embedding dimension     : {embed_dim}")
    if target_layer is not None:
        print(f"  - Extracting only layer        : {target_layer} (embedding=0)")
    else:
        print(f"  - Number of BERT layers        : {num_bert_layers + 1} (embedding + {num_bert_layers} transformer)")
    print(f"  - Window embedding dimension   : {window_embed_dim}")
    print(f"  - Storage dtype                : float16")

    resume_from_batch = 0
    if h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            if f["progress"].attrs.get("completed", False):
                raise FileExistsError(f"Complete H5 file already exists: {h5_path}")
            stored_seed = f["progress"].attrs["seed"]
            stored_batch_size = f["progress"].attrs["batch_size"]
            if stored_seed != seed:
                raise ValueError(f"Seed mismatch: stored {stored_seed} vs current {seed}")
            if stored_batch_size != batch_size:
                raise ValueError(f"Batch size mismatch: stored {stored_batch_size} vs current {batch_size}")
            resume_from_batch = f["progress"].attrs["current_batch_idx"] + 1
            print(f"Resuming from batch {resume_from_batch}")

    print("Initializing dataset...")
    dataset = SingleFileDNAFragmentDataset(
        tokenizer_file=tokenizer_file,
        ref_genome_file=ref_genome_file,
        bed_file_row=bed_file_row,
        n_fragment_per_file=n_fragment_per_file,
        max_token_length=max_token_length,
        min_flank_token_length=min_flank_token_length,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    total_batches = len(dataloader)

    if resume_from_batch == 0:
        print("Creating new H5 file...")
        h5_file = h5py.File(h5_path, "w")
        dataset_info = {
            "bed_file": bed_file_row["bed_file"],
            "sample_id": sample_id,
            "modality": modality,
            "group": bed_file_row.get("group", "unknown"),
            "n_fragments_total": bed_file_row["n_fragment"],
            "n_fragments_sampled": len(dataset),
            "bert_checkpoint": bert_checkpoint,
            "total_batches": total_batches,
            "seed": seed,
            "batch_size": batch_size,
            "target_layer": target_layer,
        }
        create_h5_structure(
            h5_file, len(dataset), dataset_info,
            embed_dim=embed_dim, num_bert_layers=num_bert_layers,
        )
    else:
        print("Opening existing H5 file for resumption...")
        h5_file = h5py.File(h5_path, "r+")

    print(f"Loading BERT model from: {bert_checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bert_model = DistilBertForMaskedLM.from_pretrained(bert_checkpoint)
    bert_model.to(device)
    bert_model.eval()

    for param in bert_model.parameters():
        param.requires_grad = False

    print(f"Model loaded on {device}")

    n_samples = len(dataset)
    if target_layer is not None:
        embedding_size_gb = n_samples * window_embed_dim * 2 / (1024 ** 3)
    else:
        embedding_size_gb = n_samples * (num_bert_layers + 1) * window_embed_dim * 2 / (1024 ** 3)
    print(f"Expected H5 file size: ~{embedding_size_gb:.2f} GB")

    try:
        with tqdm(total=total_batches, initial=resume_from_batch, desc="Processing fragments") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx < resume_from_batch:
                    continue

                batch_metadata = {
                    "chromosome": batch["chromosome"],
                    "fragment_start": batch["fragment_start"],
                    "fragment_end": batch["fragment_end"],
                    "adjusted_start": batch["adjusted_start"],
                    "adjusted_end": batch["adjusted_end"],
                    "line_number": batch["line_number"],
                    "reverse_complemented": batch["reverse_complemented"],
                    "n_token_left": batch["n_token_left"],
                    "n_token_frag": batch["n_token_frag"],
                    "n_token_right": batch["n_token_right"],
                    "n_freq": batch["n_freq"],
                }

                window_embeddings = process_batch(bert_model, batch, device, target_layer)

                start_idx = batch_idx * batch_size
                store_batch_data(h5_file, window_embeddings, batch_metadata, start_idx)

                h5_file["progress"].attrs["current_batch_idx"] = batch_idx
                h5_file["progress"].attrs["processed_count"] = min(
                    (batch_idx + 1) * batch_size, len(dataset)
                )
                h5_file.flush()

                pbar.update(1)

        h5_file["progress"].attrs["completed"] = True
        h5_file["progress"].attrs["processed_count"] = len(dataset)
        print(f"Successfully processed {len(dataset)} fragments")

    except Exception as e:
        print(f"Error during processing: {e}")
        traceback.print_exc()
        raise

    finally:
        h5_file.close()

    actual_size = os.path.getsize(h5_path) / (1024 ** 3)
    print(f"Evaluation complete! Output saved to {h5_path}")
    print(f"Final H5 file size: {actual_size:.2f} GB")


def main():
    parser = argparse.ArgumentParser(
        description="Extract LEAF-1 window-pooled per-fragment embeddings from one sample"
    )
    parser.add_argument("--bert_checkpoint", type=str, required=True,
                        help="Path to a HuggingFace DistilBertForMaskedLM checkpoint dir")
    parser.add_argument("--n_fragment_file", type=str, required=True,
                        help="CSV manifest: sample_id,modality,bed_file,n_fragment,group")
    parser.add_argument("--index", type=int, required=True,
                        help="Row index in --n_fragment_file to process")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dir: H5 files go to <output_dir>/h5_files/")
    parser.add_argument("--tokenizer_file", type=str, default=DEFAULT_TOKENIZER,
                        help="Tokenizer JSON. Default: packaged leaf1/data/tokenizer.json")
    parser.add_argument("--ref_genome_file", type=str, default=DEFAULT_REF_FA,
                        help="Reference FASTA (with .fai sibling). Default: assets/refseq.genome.fa")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target_layer", type=int, default=None,
                        help="Single layer to extract (paper uses 4). Omit to store all 13 layers.")
    parser.add_argument("--n_fragment_per_file", type=int, default=100000,
                        help="Max fragments to sample per sample (paper cfDNA default: 100000)")
    parser.add_argument("--max_token_length", type=int, default=2048)
    parser.add_argument("--min_flank_token_length", type=int, default=100)

    args = parser.parse_args()

    evaluate_bed_file(
        bert_checkpoint=args.bert_checkpoint,
        n_fragment_file=args.n_fragment_file,
        n_fragment_per_file=args.n_fragment_per_file,
        index=args.index,
        output_dir=args.output_dir,
        tokenizer_file=args.tokenizer_file,
        ref_genome_file=args.ref_genome_file,
        batch_size=args.batch_size,
        seed=args.seed,
        target_layer=args.target_layer,
        max_token_length=args.max_token_length,
        min_flank_token_length=args.min_flank_token_length,
    )


if __name__ == "__main__":
    main()
