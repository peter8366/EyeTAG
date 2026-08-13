#!/usr/bin/env bash
# Main Gaze360 model (paper Table 2: All 9.29 / Semi-Front 9.12 / Front 8.08 / Mean 8.83).
# Usage: bash scripts/train_gaze360.sh <DATA_ROOT> [GPU] [WORK_DIR]
set -euo pipefail

DATA_ROOT=${1:?"usage: $0 <DATA_ROOT> [GPU] [WORK_DIR]"}
GPU=${2:-0}
WORK_DIR=${3:-./work_dir/eyetag_t48_delta}

python gaze360/train.py \
    --data-root "$DATA_ROOT" \
    --work-dir  "$WORK_DIR" \
    --face-backbone vggface2 --eye-backbone resnet18 \
    --pretrained-vggface checkpoints/resnet50_ft_weight.pkl \
    --face-size 128 --eye-size 128 \
    --prev-mode mlp --prev-input delta --prev-repr pitchyaw \
    --fusion-type cross_attn --temporal-type causal --gaze-space vector \
    --num-frames 48 --frame-stride 1 --seq-stride 4 \
    --d-model 256 --nhead 4 --num-layers 2 \
    --epochs 15 --batch-size 32 --lr 3e-4 --warmup-epochs 3 \
    --weight-decay 0.02 --grad-clip 1.0 --backbone-lr-scale 0.1 \
    --prev-gt-start 1.0 --prev-gt-end 0.0 --prev-gt-end-epoch 5 \
    --cache-mode ar --autoreg-val --lr-restart-at-prev-zero \
    --seed 42 --gpus "$GPU"
