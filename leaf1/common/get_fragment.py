"""Random-access fragment retrieval and tokenization.

A BED file plus its byte-offset .index sidecar (every 1000 lines) and a reference
FASTA (with .fai) are required. `bedtools getfasta` is used to extract sequence.
Inputs are emitted as ``[CLS] left_flank [CUT] body [CUT] right_flank`` strings
ready for tokenization; the modality token is prepended by the caller.

Pretraining and inference both go through ``initialize_iterable_get_fragment``,
which returns a closure that maps a line number to the processed fragment record.
"""

import os
import subprocess
import traceback

import numpy as np
import pandas as pd
from transformers import PreTrainedTokenizerFast


def read_fai_file(fai_file):
    return pd.read_csv(fai_file, sep='\t', header=None, usecols=[0, 1], names=['chr', 'length'])


def read_index_file(index_df, line_number):
    if index_df.empty:
        return [0, 0]

    min_line = index_df['line'].min()
    max_line = index_df['line'].max()

    if line_number < min_line:
        return [0, 0]
    elif line_number > max_line:
        return index_df.iloc[-1].tolist()
    else:
        idx = index_df[index_df['line'] <= line_number].last_valid_index()
        return index_df.iloc[idx].tolist()


def get_bed_line(bed_file, offset, line_number):
    with open(bed_file, 'r') as f:
        f.seek(offset)
        for _ in range(line_number + 1):
            line = f.readline()
            if not line:
                raise ValueError(f"Line number {line_number} is out of the file's range.")
            if _ == line_number:
                return line.strip()
    raise ValueError(f"Line number {line_number} is out of the file's range.")


def process_line(line, chr_lengths, reverse_complemented, max_token_length, min_flank_token_length):
    CONST_SEQ_TO_TOKEN_RATIO = 7
    adjusted_min_flank = min_flank_token_length * CONST_SEQ_TO_TOKEN_RATIO
    adjusted_max_token_length = max_token_length * CONST_SEQ_TO_TOKEN_RATIO

    chr, start, end = line.split('\t')[:3]
    start, end = int(start), int(end)
    frag_length = end - start

    # Catch fragments that cannot possibly fit even with no flanks. Without this
    # check, the budget arithmetic below goes negative and process_sequence later
    # raises an obscure "Insufficient room" error after the bedtools call.
    if frag_length > adjusted_max_token_length:
        raise ValueError(
            f"Fragment {chr}:{start}-{end} is {frag_length} bp, which exceeds the "
            f"sequence budget of {adjusted_max_token_length} bp "
            f"(max_token_length={max_token_length} * {CONST_SEQ_TO_TOKEN_RATIO}). "
            "Filter long fragments during BED preprocessing or increase --max_token_length."
        )

    remaining_length = adjusted_max_token_length - frag_length
    left_flank_min = min(start, adjusted_min_flank)
    right_flank_min = min(chr_lengths.get(chr) - end, adjusted_min_flank)

    remaining_length -= (left_flank_min + right_flank_min)

    left_flank_extra = right_flank_extra = 0
    if remaining_length > 0:
        left_flank_extra = remaining_length // 2
        right_flank_extra = remaining_length - left_flank_extra

    start_adj = max(0, start - left_flank_min - left_flank_extra)
    end_adj = min(chr_lengths.get(chr), end + right_flank_min + right_flank_extra)

    actual_left_added = start - start_adj
    actual_right_added = end_adj - end

    # When one side hits a chromosome boundary, redirect the remaining budget to the other side.
    if start_adj == 0:
        remaining_for_right = remaining_length - actual_left_added
        end_adj = min(chr_lengths.get(chr), end + right_flank_min + right_flank_extra + remaining_for_right)
    elif end_adj == chr_lengths.get(chr):
        remaining_for_left = remaining_length - actual_right_added
        start_adj = max(0, start - left_flank_min - left_flank_extra - remaining_for_left)

    if reverse_complemented:
        adj_frag_start = end_adj - end
        adj_frag_end = end_adj - start
    else:
        adj_frag_start = start - start_adj
        adj_frag_end = end - start_adj

    return chr, start_adj, end_adj, start, end, adj_frag_start, adj_frag_end


def reverse_complement(seq):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, base) for base in reversed(seq))


def get_sequence(ref_genome, bed_line, reverse_complemented):
    process_args = ['bedtools', 'getfasta', '-fi', ref_genome, '-bed', '-']
    process = subprocess.Popen(
        process_args,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    output, errput = process.communicate(bed_line)
    if process.returncode != 0:
        raise RuntimeError(
            f"bedtools getfasta failed (returncode={process.returncode}) for "
            f"bed_line={bed_line!r}: {errput.strip()}"
        )

    lines = output.split('\n')
    if len(lines) < 2 or not lines[1]:
        raise RuntimeError(
            f"bedtools getfasta returned no sequence for bed_line={bed_line!r}. "
            f"stdout={output!r}, stderr={errput.strip()!r}"
        )
    sequence = lines[1]

    if reverse_complemented:
        sequence = reverse_complement(sequence)

    return sequence


def process_sequence(seq, adj_frag_start, adj_frag_end, tokenizer, max_token_length, min_flank_token_length, rng):
    tokenized_left = tokenizer.tokenize(seq[:adj_frag_start].replace('N', '[PAD]'), verbose=False)
    tokenized_frag = tokenizer.tokenize(("[CUT]" + seq[adj_frag_start:adj_frag_end] + "[CUT]").replace('N', '[PAD]'), verbose=False)
    tokenized_right = tokenizer.tokenize(seq[adj_frag_end:].replace('N', '[PAD]'), verbose=False)

    # +2 accounts for [CLS] and the modality token prepended later by the dataset.
    current_token_count = len(tokenized_left) + len(tokenized_frag) + len(tokenized_right) + 2

    if current_token_count == max_token_length:
        processed_seq = ''.join(tokenized_left + tokenized_frag + tokenized_right)
    elif current_token_count < max_token_length:
        pad_token_count = max_token_length - current_token_count
        pad_tokens = [tokenizer.pad_token] * pad_token_count
        processed_seq = ''.join(tokenized_left + tokenized_frag + tokenized_right + pad_tokens)
    else:
        excess_tokens = current_token_count - max_token_length

        max_left_remove = max(0, len(tokenized_left) - min_flank_token_length)
        max_right_remove = max(0, len(tokenized_right) - min_flank_token_length)
        total_removable = min(excess_tokens, max_left_remove + max_right_remove)

        if total_removable < excess_tokens:
            raise ValueError(
                "Insufficient room for token removal: "
                f"Current Total Token Length: {current_token_count}, "
                f"Max Allowed Token Length: {max_token_length}, "
                f"Excess Tokens: {excess_tokens}, "
                f"Left Flank Token Length: {len(tokenized_left)}, "
                f"Fragment Token Length: {len(tokenized_frag)}, "
                f"Right Flank Token Length: {len(tokenized_right)}, "
                f"Max Left Removable: {max_left_remove}, "
                f"Max Right Removable: {max_right_remove}, "
                f"Total Removable: {total_removable}."
            )

        if max_left_remove > 0 and max_right_remove > 0:
            # Bernoulli split between flanks; clamp at each flank's maximum.
            p = rng.uniform()
            bernoulli_vector = rng.binomial(1, p, total_removable)
            left_cumsum = np.cumsum(bernoulli_vector)
            right_cumsum = np.cumsum(1 - bernoulli_vector)

            left_remove_vector = np.where(left_cumsum <= max_left_remove, left_cumsum, max_left_remove)
            right_remove_vector = np.where(right_cumsum <= max_right_remove, right_cumsum, max_right_remove)

            total_remove_vector = left_remove_vector + right_remove_vector
            if total_remove_vector[-1] >= total_removable:
                min_index = np.where(total_remove_vector >= total_removable)[0][0]
                left_remove = left_remove_vector[min_index]
                right_remove = right_remove_vector[min_index]
            else:
                left_remove = left_remove_vector[-1]
                right_remove = right_remove_vector[-1]
                if left_remove == max_left_remove and right_remove == max_right_remove:
                    raise ValueError("Unreachable: both flanks maxed out before total_removable reached.")
                elif left_remove == max_left_remove:
                    right_remove = min(max_right_remove, total_removable - left_remove)
                elif right_remove == max_right_remove:
                    left_remove = min(max_left_remove, total_removable - right_remove)
        elif max_left_remove > 0:
            left_remove = min(max_left_remove, total_removable)
            right_remove = 0
        elif max_right_remove > 0:
            right_remove = min(max_right_remove, total_removable)
            left_remove = 0
        else:
            left_remove = right_remove = 0

        tokenized_left = tokenized_left[left_remove:]
        tokenized_right = tokenized_right[:-right_remove] if right_remove > 0 else tokenized_right

        processed_seq = ''.join(tokenized_left + tokenized_frag + tokenized_right)

    n_count = processed_seq.count('[PAD]')
    n_relative_frequency = n_count / len(processed_seq) if processed_seq else 0
    n_frag_token = len(tokenized_frag) - 2

    return processed_seq, n_relative_frequency, len(tokenized_left), n_frag_token, len(tokenized_right)


def random_access_get_fragment(tokenizer, bed_file, ref_genome, line_number,
                                max_token_length=1024, min_flank_token_length=100, seed=None):
    """One-shot version: open index + fai for every call. Use the iterable variant for throughput."""
    rng = np.random.default_rng(seed)

    index_file = f"{bed_file}.index"
    ref_genome_fai = f"{ref_genome}.fai"

    missing_files = [f for f in [bed_file, index_file, ref_genome, ref_genome_fai] if not os.path.isfile(f)]
    if missing_files:
        raise FileNotFoundError(f"The following file(s) do not exist: {', '.join(missing_files)}")

    chr_lengths = read_fai_file(ref_genome_fai).set_index('chr')['length'].to_dict()
    index_df = pd.read_csv(index_file, sep=' ', header=None, names=['line', 'offset'])

    index_line_number, offset = read_index_file(index_df, line_number)
    bed_line = get_bed_line(bed_file, offset, line_number - index_line_number)

    reverse_complemented = rng.uniform() < 0.5

    chr, start_adj, end_adj, frag_start, frag_end, adj_frag_start, adj_frag_end = process_line(
        bed_line, chr_lengths, reverse_complemented, max_token_length, min_flank_token_length
    )

    bed_line_for_bedtools = f"{chr}\t{start_adj}\t{end_adj}\n"
    sequence = get_sequence(ref_genome, bed_line_for_bedtools, reverse_complemented)

    final_sequence, n_relative_frequency, n_token_left, n_token_frag, n_token_right = process_sequence(
        sequence, adj_frag_start, adj_frag_end, tokenizer, max_token_length, min_flank_token_length, rng
    )

    return {
        'chromosome': chr,
        'adjusted_start': start_adj,
        'adjusted_end': end_adj,
        'fragment_start': frag_start,
        'fragment_end': frag_end,
        'n_token_left': n_token_left,
        'n_token_frag': n_token_frag,
        'n_token_right': n_token_right,
        'relative_frequency_of_n': n_relative_frequency,
        'reverse_complemented': reverse_complemented,
        'final_sequence': final_sequence,
    }


def initialize_iterable_get_fragment(tokenizer, bed_file, ref_genome,
                                      max_token_length=1024, min_flank_token_length=100, seed=None):
    """Return a closure that processes a single BED line by line number.

    The .fai chromosome-length table and .bed.index byte-offset table are loaded once
    here and captured by the inner function for fast repeated lookups.
    """
    rng = np.random.default_rng(seed)

    index_file = f"{bed_file}.index"
    ref_genome_fai = f"{ref_genome}.fai"

    missing_files = [f for f in [bed_file, index_file, ref_genome, ref_genome_fai] if not os.path.isfile(f)]
    if missing_files:
        raise FileNotFoundError(f"The following file(s) do not exist: {', '.join(missing_files)}")

    chr_lengths = read_fai_file(ref_genome_fai).set_index('chr')['length'].to_dict()
    index_df = pd.read_csv(index_file, sep=' ', header=None, names=['line', 'offset'])

    def process_bed_line(line_number):
        try:
            index_line_number, offset = read_index_file(index_df, line_number)
            bed_line = get_bed_line(bed_file, offset, line_number - index_line_number)

            reverse_complemented = rng.uniform() < 0.5

            chr, start_adj, end_adj, frag_start, frag_end, adj_frag_start, adj_frag_end = process_line(
                bed_line, chr_lengths, reverse_complemented, max_token_length, min_flank_token_length
            )

            bed_line_for_bedtools = f"{chr}\t{start_adj}\t{end_adj}\n"
            sequence = get_sequence(ref_genome, bed_line_for_bedtools, reverse_complemented)
            raw_fragment_dna = sequence[adj_frag_start:adj_frag_end]

            final_sequence, n_relative_frequency, n_token_left, n_token_frag, n_token_right = process_sequence(
                sequence, adj_frag_start, adj_frag_end, tokenizer, max_token_length, min_flank_token_length, rng
            )

            return {
                'chromosome': chr,
                'adjusted_start': start_adj,
                'adjusted_end': end_adj,
                'fragment_start': frag_start,
                'fragment_end': frag_end,
                'n_token_left': n_token_left,
                'n_token_frag': n_token_frag,
                'n_token_right': n_token_right,
                'relative_frequency_of_n': n_relative_frequency,
                'reverse_complemented': reverse_complemented,
                'final_sequence': final_sequence,
                'raw_fragment_dna': raw_fragment_dna,
            }
        except Exception as e:
            print("\n" + "=" * 80)
            print(f"Error processing BED line {line_number} of {bed_file}: {type(e).__name__}: {e}")
            print(traceback.format_exc())
            print("=" * 80 + "\n")
            raise

    return process_bed_line
