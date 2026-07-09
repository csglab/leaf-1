#!/usr/bin/env bash
# Assemble the release MIL checkpoint folder from the sweep's final models.
#
# Copies only the essentials (checkpoint.pt, norm_stats.pt, config.json,
# summary.json) for every cfDNA task x outer fold x inner fold, collapsing the
# constant best-config directory level so the release layout is:
#
#   mil_checkpoint/cfdna/<task>/F<k>/inner_fold_<i>/{checkpoint.pt, norm_stats.pt}
#   mil_checkpoint/cfdna/<task>/F<k>/{config.json, summary.json}
#
# Source is volatile /scratch; run this on the host that has it.
set -Eeuo pipefail

SRC="${SRC:-/scratch/goodarzilab/hamedh/mil_sweep/final_models_cfdna_100k}"
DST="${DST:-/home/hamedh/projects/fragmentomics/leaf-1/mil_checkpoint/cfdna}"

[[ -d "$SRC" ]] || { echo "ERROR: source not found: $SRC" >&2; exit 1; }
mkdir -p "$DST"

n_tasks=0; n_folds=0; n_inner=0
for task_dir in "$SRC"/*/; do
    task="$(basename "$task_dir")"
    # A task has outer folds F0..Fk; skip anything else (logs, rerun dirs, ...)
    compgen -G "${task_dir}F*/" >/dev/null 2>&1 || continue
    ((n_tasks++)) || true
    for F in "${task_dir}"F*/; do
        [[ -d "$F" ]] || continue
        fk="$(basename "$F")"                 # F0..F4
        # The single best-config dir under this fold (e.g. dropout_headC10_l1r0)
        cfg="$(find "$F" -maxdepth 1 -type d -name '*_*' | head -1)"
        [[ -n "$cfg" && -d "$cfg" ]] || { echo "  skip $task/$fk (no config dir)"; continue; }
        outF="${DST}/${task}/${fk}"
        mkdir -p "$outF"
        [[ -f "$cfg/config.json"  ]] && cp -f "$cfg/config.json"  "$outF/config.json"
        [[ -f "$cfg/summary.json" ]] && cp -f "$cfg/summary.json" "$outF/summary.json"
        ((n_folds++)) || true
        for inner in "$cfg"/inner_fold_*/; do
            [[ -d "$inner" ]] || continue
            ii="$(basename "$inner")"
            mkdir -p "$outF/$ii"
            cp -f "$inner/checkpoint.pt" "$outF/$ii/checkpoint.pt"
            cp -f "$inner/norm_stats.pt" "$outF/$ii/norm_stats.pt"
            ((n_inner++)) || true
        done
    done
    echo "  ${task}: done"
done

echo "-------------------------------------------------"
echo "tasks=${n_tasks}  outer_folds=${n_folds}  inner_folds=${n_inner}"
du -sh "$DST" 2>/dev/null || true
