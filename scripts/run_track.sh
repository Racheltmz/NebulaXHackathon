#!/usr/bin/env bash
# Run (or resume) one OpenEvolve track.
#   usage: scripts/run_track.sh <door|acv|rail|shm> <classical|deep> [extra codex_evolution args]
#   env:   OPENEVOLVE_SCRIPT  path to openevolve-scientist's codex_evolution.py (required)
#          PS3_DATA_DIR       dir holding <task>_{train,test}_{x,y}.npy (default: <repo>/data/ps3_arrays)
# Runs from <task>/ so the evaluator's relative imports resolve; output/checkpoints go to runs/<task>_<kind>/.
set -euo pipefail
TASK=${1:?task}; KIND=${2:?classical|deep}; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SCRIPT=${OPENEVOLVE_SCRIPT:?set OPENEVOLVE_SCRIPT to .../openevolve-scientist/skill/openevolve-scientist/scripts/codex_evolution.py}
OUT="$ROOT/runs/${TASK}_${KIND}/openevolve_output"
CKPT=$(ls -d "$OUT"/checkpoints/checkpoint_* 2>/dev/null | sort -t_ -k3 -n | tail -1 || true)
mkdir -p "$OUT"
cd "$ROOT/$TASK"
args=(--config "config_${KIND}.yaml" --output "$OUT" --iterations 2000 --target-score 0.99)
[ -n "${CKPT:-}" ] && args+=(--checkpoint "$CKPT")
exec python "$SCRIPT" "baseline_${KIND}.py" "eval_${KIND}.py" "${args[@]}" "$@"
