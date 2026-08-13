#!/usr/bin/env bash
# Autoregressive evaluation on the three yaw subsets reported in the paper.
# Usage: bash scripts/eval_gaze360.sh <DATA_ROOT> <CHECKPOINT> [GPU]
set -euo pipefail

DATA_ROOT=${1:?"usage: $0 <DATA_ROOT> <CHECKPOINT> [GPU]"}
CKPT=${2:?"usage: $0 <DATA_ROOT> <CHECKPOINT> [GPU]"}
GPU=${3:-0}
OUT_DIR=$(dirname "$CKPT")

for SUBSET in full semi-front front; do
    echo "=== subset: $SUBSET ==="
    python gaze360/test.py \
        --data-root "$DATA_ROOT" \
        --checkpoint "$CKPT" \
        --split test --autoregressive \
        --gaze-subset "$SUBSET" \
        --out "$OUT_DIR/eval_test_autoreg_${SUBSET//-/_}.json" \
        --gpu "$GPU"
done
