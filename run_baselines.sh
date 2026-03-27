#!/bin/bash
# ============================================================
# Fair Baseline Comparison for Advanced FedEvo Paper
# Baselines: FedAvg, FedMut, FedProx, FedDyn
# Ground Rules: seed_data=42, seed_sample=43, seed_train={44..48}
# Dataset: cifar10, alpha: 0.1 / 1 / 10
# 100 clients, 10/round, 1000 rounds, 5 epochs, batch=50
# LR=0.01, lr_decay=0.998, momentum=0.9, weight_decay=5e-4
# FedEvo/FedAvg/FedProx/FedDyn: state_mode=params, bn_recalibrate_batches=100
# FedMut: full state_dict aggregation (bn_recalibrate_batches=0, by design)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET="cifar10"
ROUNDS=1000
CLIENTS=100
CPR=10
EPOCHS=5
BATCH=50
LR=0.01
LR_DECAY=0.998
MOMENTUM=0.9
WD=5e-4
SEED_DATA=42
SEED_SAMPLE=43
SEEDS=(44 45 46 47 48)
ALPHAS=("0.1" "1" "10")
OUT_DIR="./results/baselines"

mkdir -p "$OUT_DIR"

# ---- FedAvg ----
for ALPHA in "${ALPHAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    ALPHA_STR=$(echo "$ALPHA" | tr '.' 'p')
    OUT_CSV="$OUT_DIR/fedavg_cifar10_alpha${ALPHA_STR}_seed${SEED}.csv"
    if [ -f "$OUT_CSV" ]; then echo "[SKIP] $OUT_CSV"; continue; fi
    echo "=== FedAvg alpha=${ALPHA} seed=${SEED} ==="
    python3 main.py \
      --algo fedavg \
      --dataset $DATASET --alpha "$ALPHA" \
      --num_clients $CLIENTS --clients_per_round $CPR \
      --rounds $ROUNDS --epochs $EPOCHS \
      --batch_size $BATCH --lr $LR --lr_decay $LR_DECAY \
      --momentum $MOMENTUM --weight_decay $WD \
      --state_mode params --bn_recalibrate_batches 100 \
      --seed_data $SEED_DATA --seed_sample $SEED_SAMPLE --seed_train $SEED \
      --out_csv "$OUT_CSV" --data_dir ./data
  done
done

# ---- FedMut (full state aggregation; bn_recalibrate=0 is correct for FedMut) ----
for ALPHA in "${ALPHAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    ALPHA_STR=$(echo "$ALPHA" | tr '.' 'p')
    OUT_CSV="$OUT_DIR/fedmut_cifar10_alpha${ALPHA_STR}_seed${SEED}.csv"
    if [ -f "$OUT_CSV" ]; then echo "[SKIP] $OUT_CSV"; continue; fi
    echo "=== FedMut alpha=${ALPHA} seed=${SEED} ==="
    python3 main.py \
      --algo fedmut \
      --dataset $DATASET --alpha "$ALPHA" \
      --num_clients $CLIENTS --clients_per_round $CPR \
      --rounds $ROUNDS --epochs $EPOCHS \
      --batch_size $BATCH --lr $LR --lr_decay $LR_DECAY \
      --momentum $MOMENTUM --weight_decay $WD \
      --fedmut_radius 4.0 --fedmut_mut_acc_rate 0.5 --fedmut_mut_bound 50 \
      --bn_recalibrate_batches 0 \
      --seed_data $SEED_DATA --seed_sample $SEED_SAMPLE --seed_train $SEED \
      --out_csv "$OUT_CSV" --data_dir ./data
  done
done

# ---- FedProx (mu=0.01; state_mode=params; needs BN recalibration) ----
for ALPHA in "${ALPHAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    ALPHA_STR=$(echo "$ALPHA" | tr '.' 'p')
    OUT_CSV="$OUT_DIR/fedprox_mu0p01_cifar10_alpha${ALPHA_STR}_seed${SEED}.csv"
    if [ -f "$OUT_CSV" ]; then echo "[SKIP] $OUT_CSV"; continue; fi
    echo "=== FedProx alpha=${ALPHA} seed=${SEED} ==="
    python3 main.py \
      --algo fedprox \
      --dataset $DATASET --alpha "$ALPHA" \
      --num_clients $CLIENTS --clients_per_round $CPR \
      --rounds $ROUNDS --epochs $EPOCHS \
      --batch_size $BATCH --lr $LR --lr_decay $LR_DECAY \
      --momentum $MOMENTUM --weight_decay $WD \
      --fedprox_mu 0.01 \
      --state_mode params --bn_recalibrate_batches 100 \
      --seed_data $SEED_DATA --seed_sample $SEED_SAMPLE --seed_train $SEED \
      --out_csv "$OUT_CSV" --data_dir ./data
  done
done

# ---- FedDyn (alpha=0.01; state_mode=params; needs BN recalibration)
# NOTE: FedDyn uses SGD without momentum (enforced internally); lr=0.01 (same as others).
# The lr mismatch vs FedDyn paper (lr=0.1) is acknowledged in paper's experimental setup section. ----
for ALPHA in "${ALPHAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    ALPHA_STR=$(echo "$ALPHA" | tr '.' 'p')
    OUT_CSV="$OUT_DIR/feddyn_alpha0p01_cifar10_alpha${ALPHA_STR}_seed${SEED}.csv"
    if [ -f "$OUT_CSV" ]; then echo "[SKIP] $OUT_CSV"; continue; fi
    echo "=== FedDyn alpha=${ALPHA} seed=${SEED} ==="
    python3 main.py \
      --algo feddyn \
      --dataset $DATASET --alpha "$ALPHA" \
      --num_clients $CLIENTS --clients_per_round $CPR \
      --rounds $ROUNDS --epochs $EPOCHS \
      --batch_size $BATCH --lr $LR --lr_decay $LR_DECAY \
      --momentum $MOMENTUM --weight_decay $WD \
      --feddyn_alpha 0.01 \
      --state_mode params --bn_recalibrate_batches 100 \
      --seed_data $SEED_DATA --seed_sample $SEED_SAMPLE --seed_train $SEED \
      --out_csv "$OUT_CSV" --data_dir ./data
  done
done

echo "All baseline runs complete. Results in: $OUT_DIR"
