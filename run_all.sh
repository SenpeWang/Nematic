#!/bin/bash
set -e
cd /home/wangshengping/04_Nero/code/NFAM-Net

echo "========================================"
echo "[1/4] NEURITE"
echo "========================================"
python train.py --dataset NEURITE --gpu 0
echo "[1/4] NEURITE done"

echo "========================================"
echo "[2/4] LS"
echo "========================================"
python train.py --dataset LS --gpu 0
echo "[2/4] LS done"

echo "========================================"
echo "[3/4] NEURO"
echo "========================================"
python train.py --dataset NEURO --gpu 0
echo "[3/4] NEURO done"

echo "========================================"
echo "[4/4] SY5Y"
echo "========================================"
python train.py --dataset SY5Y --gpu 0
echo "[4/4] SY5Y done"

echo "ALL DONE"
