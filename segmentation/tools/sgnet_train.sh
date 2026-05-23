#!/usr/bin/env bash
# SGNet3 训练脚本（单卡）
# 位于 segmentation/tools/，项目根目录为其父目录的父目录

ROOT_DIR=$(dirname "$0")/../..

CONFIG=$1
export CUDA_VISIBLE_DEVICES=${2:-0}

PYTHON=/home/wangshengping/myconda/envs/sp_demo/bin/python
export PYTHONPATH=/home/public/tjj-workspace/code/mmsegmentation:$ROOT_DIR:$PYTHONPATH

$PYTHON $ROOT_DIR/segmentation/tools/train.py \
    $CONFIG \
    --work-dir $ROOT_DIR/Outputs/train_results \
    ${@:3}
