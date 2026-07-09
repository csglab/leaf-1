#!/usr/bin/env bash
# Iterate over a YAML sweep.grid block and launch one leaf1-mil call per point.
#
# Required env vars:
#   CONFIG      Path to a sweep YAML (cfdna.yaml or tcga.yaml)
#   SPLIT       Path to the CV split JSON
#   FOLD        Outer fold (int)
#   OUTPUT_ROOT Sweep output root; each grid point lands in a subdir
#
# Optional env vars:
#   GPU_DEVICE  CUDA device index (default: 0)
#   N_JOBS      Concurrent grid points; clamped to grid size (default: 1)
#   SEED        Trainer seed (default: 42)
#
# Each call writes summary.json to OUTPUT_ROOT/<grid_point_id>/; leaderboard
# reconciliation (best-config selection) is up to the caller.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${CONFIG:?Set CONFIG to the sweep YAML path}"
: "${SPLIT:?Set SPLIT to the CV split JSON path}"
: "${FOLD:?Set FOLD to the outer fold index}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the sweep output root}"

GPU_DEVICE="${GPU_DEVICE:-0}"
N_JOBS="${N_JOBS:-1}"
SEED="${SEED:-42}"

mkdir -p "${OUTPUT_ROOT}"

# Expand the YAML grid into a list of CLI override strings (one per line).
# Validates first that every sweep.grid key matches a real `leaf1-mil` flag,
# so a typo in the YAML fails loudly here instead of inside an inner argparse
# call after dozens of jobs have already been scheduled.
GRID_FILE="${OUTPUT_ROOT}/_grid.txt"
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 - "${CONFIG}" > "${GRID_FILE}" <<'PY'
import sys, itertools, yaml, argparse

# Build the set of valid flags by introspecting the trainer's parser.
sys.path.insert(0, ".")
from leaf1.downstream.mil.train import parse_args  # noqa: E402

# parse_args() calls argparse with [] which would error on required args;
# fish the parser out by mocking sys.argv minimally.
import io
buf = io.StringIO()
orig_stderr, orig_argv = sys.stderr, sys.argv
try:
    sys.stderr = buf
    sys.argv = ["leaf1-mil", "--config", "/dev/null", "--split", "/dev/null", "--output_dir", "/dev/null"]
    try:
        parse_args()  # exits because /dev/null isn't valid YAML; we just want to know the flag set
    except SystemExit:
        pass
finally:
    sys.stderr, sys.argv = orig_stderr, orig_argv
# Fallback: scrape the help text.
import subprocess
help_text = subprocess.run(["leaf1-mil", "--help"], capture_output=True, text=True).stdout
import re
known_flags = set(re.findall(r"--([a-zA-Z0-9_]+)\b", help_text))

cfg = yaml.safe_load(open(sys.argv[1]))
grid = cfg.get("sweep", {}).get("grid", {})
if not grid:
    sys.stderr.write("No sweep.grid block in config.\n")
    sys.exit(1)

unknown = [k for k in grid if k not in known_flags]
if unknown:
    sys.stderr.write(
        "Sweep grid contains keys that are not `leaf1-mil` CLI flags: "
        + ", ".join(unknown)
        + "\nValid flags: " + ", ".join(sorted(known_flags))
        + "\n"
    )
    sys.exit(2)

keys = list(grid.keys())
vals = [grid[k] for k in keys]
for combo in itertools.product(*vals):
    parts = []
    for k, v in zip(keys, combo):
        if isinstance(v, bool):
            if v:
                parts.append(f"--{k}")
        else:
            parts.append(f"--{k} {v}")
    print(" ".join(parts))
PY

N_POINTS=$(wc -l < "${GRID_FILE}")
echo "Sweep grid expanded to ${N_POINTS} points."

i=0
while IFS= read -r OVERRIDES; do
  i=$((i + 1))
  RUN_DIR="${OUTPUT_ROOT}/point_${i}_F${FOLD}"
  if [[ -f "${RUN_DIR}/summary.json" ]]; then
    echo "[${i}/${N_POINTS}] skip (already complete) -> ${RUN_DIR}"
    continue
  fi
  mkdir -p "${RUN_DIR}"
  echo "[${i}/${N_POINTS}] -> ${RUN_DIR}"
  CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" \
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" \
  python3 -u -m leaf1.downstream.mil.train \
    --config "${CONFIG}" \
    --split  "${SPLIT}" \
    --fold   "${FOLD}" \
    --seed   "${SEED}" \
    --output_dir "${RUN_DIR}" \
    ${OVERRIDES} \
    > "${RUN_DIR}/stdout.log" 2>&1 || echo "    failed (see ${RUN_DIR}/stdout.log)"
done < "${GRID_FILE}"

echo "Sweep complete. Outputs under ${OUTPUT_ROOT}."
