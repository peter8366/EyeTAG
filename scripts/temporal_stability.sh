#!/usr/bin/env bash
# Reproduce the temporal-stability analysis (paper Table 3).
# Usage: bash scripts/temporal_stability.sh <DATA_ROOT> <CHECKPOINT> [GPU]
set -euo pipefail

DATA_ROOT=${1:?"usage: $0 <DATA_ROOT> <CHECKPOINT> [GPU]"}
CKPT=${2:?"usage: $0 <DATA_ROOT> <CHECKPOINT> [GPU]"}
GPU=${3:-0}

mkdir -p analysis/data
python analysis/dump_predictions.py \
    --data-root "$DATA_ROOT" --ckpt "$CKPT" \
    --out analysis/data/eyetag.npz --gpu "$GPU"

python analysis/temporal_stability.py \
    --npz-dir analysis/data --out analysis/stability
