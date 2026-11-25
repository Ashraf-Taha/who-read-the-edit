#!/usr/bin/env bash
# Runs experiment commands one after another (never in parallel: CPU and GPU share 24 GB).
# Each line of the queue file is: <script.py> <args...>. Lines starting with # are skipped.
# Every script resumes from its saved results, so rerunning the queue after an interruption
# only computes what is missing.
#   bash experiments/scripts/run_queue.sh experiments/scripts/reproduce.txt
# Set PYTHON to the interpreter of the environment from env/setup.sh (default: ./.venv/bin/python).
set -u
QUEUE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-$HERE/../../.venv/bin/python}"
LOGS="$HERE/../logs"
mkdir -p "$LOGS"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$HERE/../.."
cd "$HERE"
while read -r script args; do
  [[ -z "${script:-}" || "$script" == \#* ]] && continue
  log="$LOGS/$(basename "$script" .py)_$(echo "$args" | tr ' ,' '__').log"
  echo "[$(date '+%F %T')] start $script $args -> $(basename "$log")" | tee -a "$LOGS/queue.log"
  # shellcheck disable=SC2086
  "$PY" "$script" $args > "$log" 2>&1
  status=$?
  echo "[$(date '+%F %T')] exit $status $script $args" | tee -a "$LOGS/queue.log"
done < "$QUEUE"
