#!/usr/bin/env bash
# Usage: bash scripts/eval_eve.sh <DATA_ROOT> <CHECKPOINT> [GPU]
set -euo pipefail

DATA_ROOT=${1:?"usage: $0 <DATA_ROOT> <CHECKPOINT> [GPU]"}
CKPT=${2:?"usage: $0 <DATA_ROOT> <CHECKPOINT> [GPU]"}
GPU=${3:-0}

python eve/test.py \
    --data-root "$DATA_ROOT" \
    --checkpoint "$CKPT" \
    --split test --cameras webcam_c --target-hz 30 \
    --out "$(dirname "$CKPT")/eval_val.json" \
    --gpu "$GPU"
