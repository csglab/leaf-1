#!/usr/bin/env bash
# GNU-parallel wrapper for per-sample LEAF-1 embedding extraction.
#
# Usage:
#   ./extract_embeddings.sh <N_FRAGMENT_FILE> <BERT_CHECKPOINT> <OUTPUT_DIR> \
#                          <GPU_LIST> <JOBS_PER_GPU> [N_FRAGMENT_PER_FILE] [EXTRA_ARGS...]
#
# Example (paper-canonical: layer 4, 100k fragments per sample):
#   ./extract_embeddings.sh manifest.csv /path/to/leaf1-ckpt out/ 0,1 4 100000 --target_layer 4
#
# Optional:
#   USE_SHM=1    Copy assets to /dev/shm first (recommended)
#   TOKENIZER    Tokenizer JSON (default: <repo>/leaf1/data/tokenizer.json
#                 or /dev/shm/tokenizer.json when USE_SHM=1)
#   REF_FA       Reference FASTA (analogous default)
#   BATCH_SIZE   (default: 64)
#   SEED         (default: 42)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <N_FRAGMENT_FILE> <BERT_CHECKPOINT> <OUTPUT_DIR> <GPU_LIST> <JOBS_PER_GPU> [N_FRAGMENT_PER_FILE] [EXTRA_ARGS...]"
  exit 1
fi

N_FRAGMENT_FILE="$1"
BERT_CHECKPOINT="$2"
OUTPUT_DIR="$3"
GPU_LIST="$4"
JOBS_PER_GPU="$5"
N_FRAGMENT_PER_FILE="${6:-100000}"
shift 6 || true
EXTRA_ARGS="${*:-}"

BATCH_SIZE="${BATCH_SIZE:-64}"
SEED="${SEED:-42}"
USE_SHM="${USE_SHM:-0}"
PYTHON_BIN="${PYTHON:-python}"

if [[ "${USE_SHM}" == "1" ]]; then
  bash "${SCRIPT_DIR}/prepare_shm.sh"
  TOKENIZER="${TOKENIZER:-/dev/shm/tokenizer.json}"
  REF_FA="${REF_FA:-/dev/shm/refseq.genome.fa}"
else
  TOKENIZER="${TOKENIZER:-${REPO_ROOT}/leaf1/data/tokenizer.json}"
  REF_FA="${REF_FA:-${REPO_ROOT}/assets/refseq.genome.fa}"
fi

[[ -f "${N_FRAGMENT_FILE}" ]] || { echo "Fragment CSV not found: ${N_FRAGMENT_FILE}"; exit 1; }
[[ -d "${BERT_CHECKPOINT}" ]] || { echo "Checkpoint dir not found: ${BERT_CHECKPOINT}"; exit 1; }
command -v parallel >/dev/null 2>&1 || { echo "GNU parallel not found on PATH"; exit 1; }
"${PYTHON_BIN}" - <<'PY' >/dev/null
import h5py, torch, transformers
PY

mkdir -p "${OUTPUT_DIR}/h5_files" "${OUTPUT_DIR}/logs"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
JOB_DIR="${OUTPUT_DIR}/jobs_${RUN_TS}"
PAR_LOG_DIR="${OUTPUT_DIR}/parallel_logs_${RUN_TS}"
mkdir -p "${JOB_DIR}" "${PAR_LOG_DIR}"

IFS=',' read -ra GPU_ARR <<< "${GPU_LIST}"
NUM_GPUS=${#GPU_ARR[@]}

echo "============================================================"
echo "LEAF-1 embedding extraction"
echo "Manifest    : ${N_FRAGMENT_FILE}"
echo "Checkpoint  : ${BERT_CHECKPOINT}"
echo "Output      : ${OUTPUT_DIR}"
echo "GPUs        : ${GPU_LIST}  (n=${NUM_GPUS}, ${JOBS_PER_GPU} jobs/GPU)"
echo "Tokenizer   : ${TOKENIZER}"
echo "Ref FASTA   : ${REF_FA}"
echo "Batch size  : ${BATCH_SIZE}"
echo "Seed        : ${SEED}"
echo "Frag/file   : ${N_FRAGMENT_PER_FILE}"
echo "Extra args  : ${EXTRA_ARGS}"
echo "Run tag     : ${RUN_TS}"
echo "============================================================"

TOTAL_ROWS=$(( $(wc -l < "${N_FRAGMENT_FILE}") - 1 ))
echo "Rows / samples: ${TOTAL_ROWS}"

for gid in "${GPU_ARR[@]}"; do
  : > "${JOB_DIR}/gpu${gid}_jobs.txt"
done

row_idx=0
skipped=0
assigned=0

tail -n +2 "${N_FRAGMENT_FILE}" | while IFS=',' read -r sample_id modality bed_file n_fragment group _
do
  h5_path="${OUTPUT_DIR}/h5_files/${modality}_${sample_id}.h5"

  if [[ -f "${h5_path}" ]]; then
    complete=$("${PYTHON_BIN}" - <<PY
import h5py
try:
    with h5py.File("${h5_path}", "r") as f:
        print("1" if f["progress"].attrs.get("completed", False) else "0")
except Exception:
    print("0")
PY
)
    if [[ "${complete}" == "1" ]]; then
      skipped=$((skipped + 1))
      row_idx=$((row_idx + 1))
      continue
    fi
  fi

  gpu_sel="${GPU_ARR[$((assigned % NUM_GPUS))]}"
  job_file="${JOB_DIR}/gpu${gpu_sel}_jobs.txt"
  log_file="${OUTPUT_DIR}/logs/${modality}_${sample_id}.log"

  {
    echo -n "CUDA_VISIBLE_DEVICES=${gpu_sel} PYTHONPATH=\"${REPO_ROOT}:\${PYTHONPATH:-}\""
    echo -n " ${PYTHON_BIN} -u -m leaf1.infer.extract_embeddings"
    echo -n " --bert_checkpoint \"${BERT_CHECKPOINT}\""
    echo -n " --n_fragment_file \"${N_FRAGMENT_FILE}\""
    echo -n " --index ${row_idx}"
    echo -n " --output_dir \"${OUTPUT_DIR}\""
    echo -n " --tokenizer_file \"${TOKENIZER}\""
    echo -n " --ref_genome_file \"${REF_FA}\""
    echo -n " --batch_size ${BATCH_SIZE}"
    echo -n " --seed ${SEED}"
    echo -n " --n_fragment_per_file ${N_FRAGMENT_PER_FILE}"
    [[ -n "${EXTRA_ARGS}" ]] && echo -n " ${EXTRA_ARGS}"
    echo " > \"${log_file}\" 2>&1"
  } >> "${job_file}"

  assigned=$((assigned + 1))
  row_idx=$((row_idx + 1))
done

echo
echo "Job distribution per GPU:"
total_jobs=0
for gid in "${GPU_ARR[@]}"; do
  jf="${JOB_DIR}/gpu${gid}_jobs.txt"
  cnt=$( [[ -f "${jf}" ]] && wc -l < "${jf}" || echo 0 )
  echo "  GPU ${gid}: ${cnt} jobs"
  total_jobs=$((total_jobs + cnt))
done

echo "Skipped (already complete): ${skipped}"
[[ ${total_jobs} -eq 0 ]] && { echo "Nothing to do."; exit 0; }

pids=()
for gid in "${GPU_ARR[@]}"; do
  jf="${JOB_DIR}/gpu${gid}_jobs.txt"
  [[ -s "${jf}" ]] || continue
  par_log="${PAR_LOG_DIR}/gpu${gid}_parallel.log"
  echo "Starting GPU ${gid}  (max ${JOBS_PER_GPU} concurrent) ..."
  ( cat "${jf}" | parallel --progress --eta --joblog "${par_log}" -j "${JOBS_PER_GPU}" ) \
    > "${PAR_LOG_DIR}/gpu${gid}_progress.log" 2>&1 &
  pids+=( $! )
done

alive () { for pid in "${pids[@]}"; do kill -0 "${pid}" 2>/dev/null && return 0; done; return 1; }

echo
echo "Monitoring ..."
while alive; do
  sleep 30
  done_cnt=$(find "${OUTPUT_DIR}/h5_files" -name '*.h5' -type f | wc -l)
  echo "[$(date '+%H:%M:%S')] H5 files present: ${done_cnt}"
done

echo "All workers exited."

succ=0; fail=0
for gid in "${GPU_ARR[@]}"; do
  pl="${PAR_LOG_DIR}/gpu${gid}_parallel.log"
  [[ -f "${pl}" ]] || continue
  s=$(awk 'NR>1 && $7==0 {c++} END{print c+0}' "${pl}")
  f=$(awk 'NR>1 && $7!=0 {c++} END{print c+0}' "${pl}")
  echo "GPU ${gid}: ${s} ok, ${f} failed"
  succ=$((succ + s))
  fail=$((fail + f))
done
echo "Overall: ${succ} ok, ${fail} failed"

{
  echo "Run               : ${RUN_TS}"
  echo "Manifest          : ${N_FRAGMENT_FILE}"
  echo "Checkpoint        : ${BERT_CHECKPOINT}"
  echo "GPUs              : ${GPU_LIST} (jobs/GPU=${JOBS_PER_GPU})"
  echo "Total rows        : ${TOTAL_ROWS}"
  echo "Skipped completed : ${skipped}"
  echo "Jobs launched     : ${total_jobs}"
  echo "Succeeded         : ${succ}"
  echo "Failed            : ${fail}"
  echo "N_fragment_per_file: ${N_FRAGMENT_PER_FILE}"
  echo "Extra args        : ${EXTRA_ARGS}"
} | tee "${OUTPUT_DIR}/evaluation_summary_${RUN_TS}.txt"

echo "Done. H5 -> ${OUTPUT_DIR}/h5_files"
