#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/wangshengping/04_Nero/code/Nematic"
PYTHON_BIN="/home/wangshengping/myconda/envs/sp_mamba/bin/python"
LOG_DIR="$ROOT_DIR/Outputs/tmux_logs"
mkdir -p "$LOG_DIR"

GPU="${1:-0}"
BASE_EPOCHS="${2:-200}"
EXP_EPOCHS="${3:-200}"
EXP_NAME="${4:-state-exp}"
SSM_VARIANT="${5:-scaled_transition}"
SSM_DT_SCALE="${6:-0.90}"
SSM_A_SCALE="${7:-1.10}"
SSM_TRAP_SCALE="${8:-1.00}"
SSM_ANGLE_SCALE="${9:-1.00}"

SESSION="nematic-neuro-compare"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/${STAMP}_${SESSION}.log"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session already exists: $SESSION"
  exit 1
fi

CMD=$(cat <<EOF
cd "$ROOT_DIR"
set -euo pipefail

echo '[1/4] baseline train'
"$PYTHON_BIN" train.py --dataset NEURO --gpu "$GPU" --epochs "$BASE_EPOCHS" --experiment baseline --ssm_variant baseline --ssm_dt_scale 1.0 --ssm_a_scale 1.0 --ssm_trap_scale 1.0 --ssm_angle_scale 1.0
BASE_DIR=$(find "$ROOT_DIR/Outputs/Train" -maxdepth 1 -type d -name '*_Nematic_neuro_baseline_baseline' | sort | tail -n 1)
echo "BASE_DIR=$BASE_DIR"

echo '[2/4] baseline test'
"$PYTHON_BIN" test.py --dataset NEURO --gpu "$GPU" --checkpoint "$BASE_DIR"

echo '[3/4] experiment train'
"$PYTHON_BIN" train.py --dataset NEURO --gpu "$GPU" --epochs "$EXP_EPOCHS" --experiment "$EXP_NAME" --ssm_variant "$SSM_VARIANT" --ssm_dt_scale "$SSM_DT_SCALE" --ssm_a_scale "$SSM_A_SCALE" --ssm_trap_scale "$SSM_TRAP_SCALE" --ssm_angle_scale "$SSM_ANGLE_SCALE"
EXP_DIR=$(find "$ROOT_DIR/Outputs/Train" -maxdepth 1 -type d -name "*_Nematic_neuro_${EXP_NAME}_${SSM_VARIANT}" | sort | tail -n 1)
echo "EXP_DIR=$EXP_DIR"

echo '[4/4] experiment test'
"$PYTHON_BIN" test.py --dataset NEURO --gpu "$GPU" --checkpoint "$EXP_DIR"
EOF
)

tmux new-session -d -s "$SESSION" "bash -lc '$CMD' 2>&1 | tee $LOG_FILE"
echo "session=$SESSION"
echo "log=$LOG_FILE"
