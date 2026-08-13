#!/usr/bin/env bash
# Main EVE model (paper Table 2: 2.56 deg on the official validation split).
# Usage: bash scripts/train_eve.sh <DATA_ROOT> [GPU] [WORK_DIR]
set -euo pipefail

DATA_ROOT=${1:?"usage: $0 <DATA_ROOT> [GPU] [WORK_DIR]"}
GPU=${2:-0}
WORK_DIR=${3:-./work_dir/eyetag_eve_t48_delta}

python eve/train.py \
    --data-root "$DATA_ROOT" \
    --work-dir  "$WORK_DIR" \
    --cameras webcam_c --target-hz 30 \
    --face-backbone vggface2 --eye-backbone resnet18 \
    --pretrained-vggface checkpoints/resnet50_ft_weight.pkl \
    --face-size 128 --eye-size 128 \
    --prev-mode mlp --prev-input delta \
    --fusion-type cross_attn --temporal-type causal --gaze-space vector \
    --no-pog --num-frames 48 --seq-stride 4 \
    --d-model 256 --nhead 4 --num-layers 2 \
    --epochs 15 --batch-size 32 --lr 3e-4 --warmup-epochs 3 \
    --weight-decay 0.02 --grad-clip 1.0 --backbone-lr-scale 0.1 \
    --prev-gt-start 1.0 --prev-gt-end 0.0 --prev-gt-end-epoch 5 \
    --cache-mode ar --autoreg-val --lr-restart-at-prev-zero \
    --seed 42 --gpus "$GPU"
