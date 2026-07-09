#!/usr/bin/env bash
# Copy tokenizer + reference FASTA (+ .fai) into /dev/shm so per-fragment
# bedtools getfasta calls and tokenizer loads are served from RAM. This is
# what makes the get_fragment path fast under high worker counts.
#
# Override SHM_DIR if your /dev/shm is too small (the FASTA is ~3.1 GB).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHM_DIR="${SHM_DIR:-/dev/shm}"

TOKENIZER_SRC="${REPO_ROOT}/leaf1/data/tokenizer.json"
REF_FA_SRC="${REPO_ROOT}/assets/refseq.genome.fa"
REF_FAI_SRC="${REPO_ROOT}/assets/refseq.genome.fa.fai"

for f in "${TOKENIZER_SRC}" "${REF_FA_SRC}" "${REF_FAI_SRC}"; do
  if [[ ! -f "${f}" ]]; then
    echo "Missing asset: ${f}" >&2
    exit 1
  fi
done

mkdir -p "${SHM_DIR}"

echo "Copying assets to ${SHM_DIR} ..."
cp -v "${TOKENIZER_SRC}" "${SHM_DIR}/tokenizer.json"
cp -v "${REF_FA_SRC}"    "${SHM_DIR}/refseq.genome.fa"
cp -v "${REF_FAI_SRC}"   "${SHM_DIR}/refseq.genome.fa.fai"

ls -lh "${SHM_DIR}/tokenizer.json" "${SHM_DIR}/refseq.genome.fa" "${SHM_DIR}/refseq.genome.fa.fai"
echo "Done."
