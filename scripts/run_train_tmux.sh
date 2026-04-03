#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/wangshengping/04_Nero/code/Nematic"
PYTHON_BIN="/home/wangshengping/myconda/envs/sp_mamba/bin/python"
LOG_DIR="$ROOT_DIR/Outputs/tmux_logs"
mkdir -p "$LOG_DIR"

DATASET="${1:-NEURO}"
GPU="${2:-0}"
EXPERIMENT="${3:-baseline}"
SSM_VARIANT="${4:-baseline}"
SSM_DT_SCALE="${5:-1.0}"
SSM_A_SCALE="${6:-1.0}"
SSM_TRAP_SCALE="${7:-1.0}"
SSM_ANGLE_SCALE="${8:-1.0}"
shift $(( $# >= 8 ? 8 : $# )) || true
EXTRA_ARGS=("$@")

SESSION="nematic-train-${DATASET,,}-${EXPERIMENT}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${STAMP}_${SESSION}.log"
EXTRA_CMD=""
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
  printf -v EXTRA_CMD ' %q' "${EXTRA_ARGS[@]}"
fi
CMD="cd $ROOT_DIR && $PYTHON_BIN train.py --dataset $DATASET --gpu $GPU --experiment $EXPERIMENT --ssm_variant $SSM_VARIANT --ssm_dt_scale $SSM_DT_SCALE --ssm_a_scale $SSM_A_SCALE --ssm_trap_scale $SSM_TRAP_SCALE --ssm_angle_scale $SSM_ANGLE_SCALE$EXTRA_CMD 2>&1 | tee $LOG_FILE"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" "$CMD"
echo "session=$SESSION"
echo "log=$LOG_FILE"
