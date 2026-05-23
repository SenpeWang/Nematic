#!/usr/bin/env bash
# SGNet3 训练脚本（单卡）

CONFIG=$1
export CUDA_VISIBLE_DEVICES=${2:-0}

PYTHON=/home/wangshengping/myconda/envs/sp_demo/bin/python
export PYTHONPATH=/home/public/tjj-workspace/code/mmsegmentation:$(dirname "$0"):$PYTHONPATH

$PYTHON $(dirname "$0")/segmentation/tools/train.py \
    $CONFIG \
    --work-dir $(dirname "$0")/Outputs/train_results \
    ${@:3}
