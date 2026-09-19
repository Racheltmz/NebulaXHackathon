#!/bin/bash
# NOTE: this is the exact auto-resume supervisor used for the runs summarised in README.md.
# It expects the original flat run layout (<runs>/<task> classical and <runs>/<task>_gpu deep),
# not this repo layout -- use scripts/run_track.sh here. Paths at the top are the author's.
# Auto-resume supervisor: keeps each of the 4 openevolve tracks running,
# relaunching from its latest checkpoint whenever it stops (early-stop or
# iteration exhaustion) without having reached the 0.99 target yet. Prints a
# consolidated status line every 15 minutes (900s).
set -uo pipefail
cd ~/projects/ps3-evolve
source ~/projects/openevolve-venv/bin/activate
SCRIPT=~/projects/openevolve-scientist/skill/openevolve-scientist/scripts/codex_evolution.py
TASKS="${TASKS:-door rail acv shm door_gpu rail_gpu acv_gpu shm_gpu}"
TARGET=0.99
TICK=900   # 15 minutes

latest_checkpoint() {
  ls -d "$1"/openevolve_output/checkpoints/checkpoint_* 2>/dev/null | sort -t_ -k2 -n | tail -1
}

running_count_for_task() {
  local t=$1
  local target_dir
  target_dir=$(realpath "$HOME/projects/ps3-evolve/$t")
  local count=0
  for pid in $(pgrep -f "codex_evolution.py" 2>/dev/null); do
    local cwd
    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
    if [ "$cwd" = "$target_dir" ]; then
      count=$((count + 1))
    fi
  done
  echo "$count"
}

best_traintest() {
  # last reported traintest_metric for this task's log
  grep -oE "traintest_metric=[0-9.]+" "logs/$1.log" 2>/dev/null | tail -1 | cut -d= -f2
}

launch() {
  local t=$1
  local ckpt
  ckpt=$(latest_checkpoint "$t")
  cd "$t"
  local args=(--config config.yaml --output openevolve_output --iterations 2000 --target-score $TARGET)
  if [ -n "$ckpt" ]; then
    args+=(--checkpoint "$ckpt")
  fi
  PS3_SANDBOX_TIMEOUT=180 nohup python "$SCRIPT" initial_program.py evaluator.py "${args[@]}" \
    > "../logs/$t.log" 2>&1 &
  echo "$(date +%T) [$t] (re)launched pid $! resuming from ${ckpt:-scratch}"
  cd ..
}

echo "$(date +%T) supervisor starting, target=$TARGET, tick=${TICK}s"
while true; do
  for t in $TASKS; do
    cd ~/projects/ps3-evolve
    running=$(running_count_for_task "$t")
    if [ "$running" -eq 0 ]; then
      bt=$(best_traintest "$t")
      bt=${bt:-0}
      reached=$(python3 -c "print(1 if $bt >= $TARGET else 0)")
      if [ "$reached" = "1" ]; then
        echo "$(date +%T) [$t] TARGET REACHED (traintest=$bt) -- not relaunching"
      else
        echo "$(date +%T) [$t] stopped (last traintest=$bt, target=$TARGET) -- relaunching"
        launch "$t"
      fi
    fi
  done
  sleep $TICK
  echo "----- status @ $(date +%T) -----"
  for t in $TASKS; do
    cd ~/projects/ps3-evolve
    m=$(grep "Metrics:" "logs/$t.log" 2>/dev/null | tail -1)
    echo "[$t] $m"
  done
done
