#!/bin/bash
# Usage: bash run_fair_fedevo.sh <alpha> <seed>
# Example: bash run_fair_fedevo.sh 0.1 44

ALPHA=${1:?"Usage: bash run_fair_fedevo.sh <alpha> <seed>"}
SEED=${2:?"Usage: bash run_fair_fedevo.sh <alpha> <seed>"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ALPHA_STR=$(echo "$ALPHA" | tr '.' 'p')
OUT_DIR="./results"
mkdir -p "$OUT_DIR"
OUT_CSV="$OUT_DIR/fedevo_stab_cifar10_alpha${ALPHA_STR}_seed${SEED}_interpolation1.csv"

if [ -f "$OUT_CSV" ]; then
  echo "[SKIP] $OUT_CSV already exists"
  exit 0
fi

echo "[RUN] FedEvo(stab) | alpha=$ALPHA | seed=$SEED"

python3 main.py \
  --algo fedevo \
  --deploy_model stab \
  --bn_recalibrate_batches 100 \
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
  --m 10 \
  --rho 0.3 \
  --gamma 1.5 \
  --tau_factor 0.8 \
  --sigma_mut 0.01 \
  --num_interp 1 \
  --num_orth 1 \
  --state_mode params \
  --seed_data 42 \
  --seed_sample 43 \
  --seed_train "$SEED" \
  --seed_evo 2025 \
  --out_csv "$OUT_CSV"

echo "[DONE] $OUT_CSV"
