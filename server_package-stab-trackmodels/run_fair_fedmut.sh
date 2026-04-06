#!/bin/bash
# Usage: bash run_fair_fedmut.sh <alpha> <seed>
# Example: bash run_fair_fedmut.sh 0.1 44

ALPHA=${1:?"Usage: bash run_fair_fedmut.sh <alpha> <seed>"}
SEED=${2:?"Usage: bash run_fair_fedmut.sh <alpha> <seed>"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ALPHA_STR=$(echo "$ALPHA" | tr '.' 'p')
OUT_DIR="./results/fair"
mkdir -p "$OUT_DIR"
OUT_CSV="$OUT_DIR/fedmut_cifar10_alpha${ALPHA_STR}_seed${SEED}.csv"

if [ -f "$OUT_CSV" ]; then
  echo "[SKIP] $OUT_CSV already exists"
  exit 0
fi

echo "[RUN] FedMut | alpha=$ALPHA | seed=$SEED"

python3 main.py \
  --algo fedmut \
  --dataset cifar10 \
  --alpha "$ALPHA" \
  --rounds 1000 \
  --num_clients 100 \
  --clients_per_round 10 \
  --epochs 5 \
  --batch_size 50 \
  --lr 0.01 \
  --lr_decay 0.998 \
  --momentum 0.9 \
  --weight_decay 5e-4 \
  --fedmut_radius 4.0 \
  --fedmut_mut_acc_rate 0.5 \
  --fedmut_mut_bound 50 \
  --seed_data 42 \
  --seed_sample 43 \
  --seed_train "$SEED" \
  --bn_recalibrate_batches 0 \
  --out_csv "$OUT_CSV"

echo "[DONE] $OUT_CSV"
