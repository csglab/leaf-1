"""Filter a raw FinaleDB fragment file into the BED+index format the pipeline expects.

Mirrors `finaledb/scripts/decompress_and_filter.sh` (the AWK + bedtools pipeline)
in Python so the output can be byte-compared against reference
filtered BED artifacts for validation. The order of operations and column emission must match exactly.

Pipeline (matches the lab's source-of-truth shell script):
  1. zcat the raw `.frag.gz`
  2. Per-line filter: MAPQ >= 30, length >= 20, chromosome in the .fai allowlist
  3. Emit 5 tab-separated columns: chr, start, end, mapq, strand
  4. `bedtools intersect -v -a stdin -b <(zcat blacklist)` to remove blacklist hits
  5. Build a sparse `.bed.index` sidecar: `<line_number> <byte_offset>` every 1000 lines
  6. Compute summary stats (count, mean/median/min/max fragment length)
"""

import argparse
import gzip
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BLACKLIST = REPO_ROOT / "assets" / "hg38-blacklist.v2.bed.gz"
DEFAULT_REF_FAI = REPO_ROOT / "assets" / "refseq.genome.fa.fai"


def load_chromosome_allowlist(fai_path: str) -> set:
    """Return the set of chromosome names listed in column 1 of a .fai index."""
    names = set()
    with open(fai_path) as f:
        for line in f:
            parts = line.split("\t")
            if parts and parts[0]:
                names.add(parts[0])
    if not names:
        raise ValueError(f"No chromosomes loaded from .fai: {fai_path}")
    return names


def _filter_lines(input_gz: str, allowed: set) -> Iterable[str]:
    """Yield filtered tab-separated 5-col BED lines from a gzipped FinaleDB .frag.gz."""
    with gzip.open(input_gz, "rt") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            chrom, start, end, mapq, strand = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                istart = int(start)
                iend = int(end)
                imapq = int(mapq)
            except ValueError:
                continue
            if imapq < 30:
                continue
            if iend - istart < 20:
                continue
            if chrom not in allowed:
                continue
            yield f"{chrom}\t{istart}\t{iend}\t{imapq}\t{strand}\n"


def _bedtools_subtract_blacklist(filtered_iter: Iterable[str], blacklist_gz: str,
                                  output_bed: str) -> None:
    """Stream filtered lines through `bedtools intersect -v -a stdin -b <blacklist>`.

    bedtools is required on PATH. The blacklist is gunzipped to a temp file once
    so bedtools can mmap it; the filtered lines stream through stdin.
    """
    import tempfile
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".bed", delete=False) as tmp:
        tmp_path = tmp.name
        with gzip.open(blacklist_gz, "rb") as gz:
            tmp.write(gz.read())

    try:
        with open(output_bed, "w") as out:
            bt = subprocess.Popen(
                ["bedtools", "intersect", "-v", "-a", "stdin", "-b", tmp_path],
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=subprocess.PIPE,
                text=True,
            )
            for line in filtered_iter:
                bt.stdin.write(line)
            bt.stdin.close()
            rc = bt.wait()
            if rc != 0:
                err = bt.stderr.read()
                raise RuntimeError(f"bedtools intersect failed (rc={rc}): {err}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def build_sparse_index(bed_path: str, index_path: str, interval: int = 1000) -> None:
    """Write `<line_number> <byte_offset>` rows every `interval` lines.

    Matches the AWK in the lab's `decompress_and_filter.sh`:
        BEGIN {s = 0; count = 0}
        {s += length($0) + 1; count++; if (count % interval == 0) print NR, s}

    `length($0)` is byte length of the line WITHOUT the trailing newline; the `+1`
    accounts for `\n`. AWK's `NR` is 1-based.
    """
    with open(bed_path, "rb") as f, open(index_path, "w") as idx:
        offset = 0
        count = 0
        for line_no, raw in enumerate(f, start=1):
            offset += len(raw)
            count += 1
            if count % interval == 0:
                idx.write(f"{line_no} {offset}\n")


def write_summary(bed_path: str, summary_path: str) -> None:
    """Mean/median/min/max fragment length + count. Matches the lab's AWK summary."""
    lengths = []
    with open(bed_path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                ln = int(parts[2]) - int(parts[1])
                lengths.append(ln)
            except ValueError:
                continue

    if not lengths:
        with open(summary_path, "w") as f:
            f.write(f"{bed_path}\tNA\tNA\tNA\tNA\t0\n")
        return

    lengths.sort()
    n = len(lengths)
    total = sum(lengths)
    mean = total / n
    if n % 2 == 1:
        median = lengths[n // 2]
    else:
        median = (lengths[n // 2 - 1] + lengths[n // 2]) / 2
    mn, mx = lengths[0], lengths[-1]
    with open(summary_path, "w") as f:
        f.write(f"{bed_path}\t{mean}\t{median}\t{mn}\t{mx}\t{n}\n")


def preprocess(input_gz: str, output_dir: str,
               blacklist_gz: str = str(DEFAULT_BLACKLIST),
               ref_fai: str = str(DEFAULT_REF_FAI),
               index_interval: int = 1000) -> dict:
    """Run the full pipeline end-to-end. Returns paths of all generated files.

    Output filename matches the lab convention: the .frag suffix from the input
    becomes `<stem>_filtered.bed`.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    in_name = Path(input_gz).name  # e.g. "EE85723.hg38.frag.gz"
    stem = in_name[:-3] if in_name.endswith(".gz") else in_name  # "EE85723.hg38.frag"
    output_bed = out_dir / f"{stem}_filtered.bed"
    index_path = out_dir / f"{output_bed.name}.index"
    summary_path = out_dir / f"{output_bed.stem}.summary.txt"

    if not Path(input_gz).is_file():
        raise FileNotFoundError(f"Input not found: {input_gz}")
    if not Path(blacklist_gz).is_file():
        raise FileNotFoundError(f"Blacklist not found: {blacklist_gz}")
    if not Path(ref_fai).is_file():
        raise FileNotFoundError(f"Reference .fai not found: {ref_fai}")

    allowed = load_chromosome_allowlist(ref_fai)
    filtered = _filter_lines(input_gz, allowed)
    _bedtools_subtract_blacklist(filtered, blacklist_gz, str(output_bed))

    if output_bed.stat().st_size == 0:
        with open(summary_path, "w") as f:
            f.write(f"{output_bed}\tNA\tNA\tNA\tNA\t0\n")
        index_path.touch()
        return {"bed": str(output_bed), "index": str(index_path), "summary": str(summary_path)}

    build_sparse_index(str(output_bed), str(index_path), interval=index_interval)
    write_summary(str(output_bed), str(summary_path))

    return {"bed": str(output_bed), "index": str(index_path), "summary": str(summary_path)}


def main():
    p = argparse.ArgumentParser(
        description="Filter a raw FinaleDB .frag.gz into BED + .bed.index + summary."
    )
    p.add_argument("--input", required=True, help="Raw .frag.gz from FinaleDB")
    p.add_argument("--output_dir", required=True, help="Where filtered BED + index land")
    p.add_argument("--blacklist", default=str(DEFAULT_BLACKLIST),
                   help="Path to hg38-blacklist.v2.bed.gz (or equivalent). "
                        "Default: assets/hg38-blacklist.v2.bed.gz")
    p.add_argument("--ref_fai", default=str(DEFAULT_REF_FAI),
                   help="Path to refseq.genome.fa.fai (defines canonical chromosomes). "
                        "Default: assets/refseq.genome.fa.fai")
    p.add_argument("--index_interval", type=int, default=1000,
                   help="Lines between .bed.index entries (default 1000, paper default)")
    args = p.parse_args()

    out = preprocess(args.input, args.output_dir, args.blacklist, args.ref_fai,
                      index_interval=args.index_interval)
    print(f"Wrote: {out['bed']}")
    print(f"Wrote: {out['index']}")
    print(f"Wrote: {out['summary']}")


if __name__ == "__main__":
    main()
