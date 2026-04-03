#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/wangshengping/04_Nero/code/Nematic"
PYTHON_BIN="/home/wangshengping/myconda/envs/sp_mamba/bin/python"
LOG_DIR="$ROOT_DIR/Outputs/tmux_logs"
mkdir -p "$LOG_DIR"

CHECKPOINT="${1:?checkpoint dir or file required}"
DATASET="${2:-NEURO}"
GPU="${3:-0}"
shift $(( $# >= 3 ? 3 : $# )) || true
EXTRA_ARGS=("$@")

SESSION="nematic-test-$(date +%H%M%S)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${STAMP}_${SESSION}.log"
EXTRA_CMD=""
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  printf -v EXTRA_CMD ' %q' "${EXTRA_ARGS[@]}"
fi
CMD="cd $ROOT_DIR && $PYTHON_BIN test.py --checkpoint $CHECKPOINT --dataset $DATASET --gpu $GPU$EXTRA_CMD 2>&1 | tee $LOG_FILE"

tmux new-session -d -s "$SESSION" "$CMD"
echo "session=$SESSION"
echo "log=$LOG_FILE"
