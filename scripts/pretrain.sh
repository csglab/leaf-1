#!/usr/bin/env bash
# Launch LEAF-1 pretraining (paper-canonical 12L/12H/768d, seq_len=2048).
#
# Required env vars:
#   TRAIN_CSV    CSV manifest: sample_id,modality,bed_file,n_fragment,group
#   VAL_CSV      Same format as TRAIN_CSV
#   OUTPUT_ROOT  Checkpoint root
#   LOG_ROOT     HF Trainer logging root
#
# Optional:
#   N_GPUS         (default: auto-detect from CUDA_VISIBLE_DEVICES, fallback 1)
#   REF_FA         (default: <repo>/assets/refseq.genome.fa)
#   TOKENIZER      (default: <repo>/leaf1/data/tokenizer.json)
#   USE_SHM=1      Copy assets to /dev/shm first (recommended on busy fs)
#   RUN_TAG        Tag appended to run name (default: LEAF1)
#   MASTER_PORT    DDP master port (default: 29501)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${TRAIN_CSV:?Set TRAIN_CSV to the training manifest CSV}"
: "${VAL_CSV:?Set VAL_CSV to the validation manifest CSV}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the checkpoint root dir}"
: "${LOG_ROOT:?Set LOG_ROOT to the HF Trainer logging dir}"

REF_FA="${REF_FA:-${REPO_ROOT}/assets/refseq.genome.fa}"
TOKENIZER="${TOKENIZER:-${REPO_ROOT}/leaf1/data/tokenizer.json}"
RUN_TAG="${RUN_TAG:-LEAF1}"
MASTER_PORT="${MASTER_PORT:-29501}"
USE_SHM="${USE_SHM:-0}"

if [[ "${USE_SHM}" == "1" ]]; then
  bash "${SCRIPT_DIR}/prepare_shm.sh"
  REF_FA="/dev/shm/refseq.genome.fa"
  TOKENIZER="/dev/shm/tokenizer.json"
fi

# Auto-detect GPU count if not provided.
if [[ -z "${N_GPUS:-}" ]]; then
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    N_GPUS=$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")
  elif command -v nvidia-smi >/dev/null 2>&1; then
    N_GPUS=$(nvidia-smi -L | wc -l)
  else
    N_GPUS=1
  fi
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

echo "============================================================"
echo "LEAF-1 pretraining"
echo "Repo root  : ${REPO_ROOT}"
echo "Train CSV  : ${TRAIN_CSV}"
echo "Val CSV    : ${VAL_CSV}"
echo "Ref FASTA  : ${REF_FA}"
echo "Tokenizer  : ${TOKENIZER}"
echo "Output     : ${OUTPUT_ROOT}"
echo "Log        : ${LOG_ROOT}"
echo "GPUs       : ${N_GPUS}"
echo "Run tag    : ${RUN_TAG}"
echo "============================================================"

cd "${REPO_ROOT}"

torchrun --nproc-per-node="${N_GPUS}" --master_port="${MASTER_PORT}" \
  -m leaf1.train.pretrain \
    --n_layers 12 --n_heads 12 --ffn_mult 4 \
    --seq_len 2048 --vocab_size 4099 --min_flank 100 \
    --batch_size 32 --grad_accum 32 \
    --max_steps 40000 \
    --learning_rate 1e-4 --weight_decay 1e-5 \
    --lr_scheduler linear --warmup_steps 1000 \
    --mlm_probability 0.15 --cut_mlm_probability 0.30 \
    --seed 42 --n_worker 12 \
    --train_csv "${TRAIN_CSV}" \
    --val_csv "${VAL_CSV}" \
    --ref_fa "${REF_FA}" \
    --tokenizer_file "${TOKENIZER}" \
    --output_root "${OUTPUT_ROOT}" \
    --log_root "${LOG_ROOT}" \
    --run_tag "${RUN_TAG}" \
    --report_to "${REPORT_TO:-none}" \
    --eval_steps 250 --save_steps 1000 --save_total_limit 5 \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --bf16
