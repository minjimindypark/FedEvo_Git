#!/bin/bash
# ============================================================
# Advanced FedEvo 실험 스크립트
# (theta_fedavg anchor + interp range [0.2, 0.8])
#
# 사용법:
#   1. 이 파일을 프로젝트 루트(main.py가 있는 곳)에 복사
#   2. algorithms/fedevo.py 를 이 폴더의 fedevo.py로 교체
#   3. bash run_advanced_fedevo.sh
# ============================================================

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
OUT_DIR="./results/advanced_fedevo"
DATASET="cifar10"

mkdir -p "$OUT_DIR"

# ---- FedEvo-topk (advanced) ----
for ALPHA in "${ALPHAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    ALPHA_STR=$(echo "$ALPHA" | tr '.' 'p')
    OUT_CSV="$OUT_DIR/fedevo_advanced_topk_${DATASET}_alpha${ALPHA_STR}_seed${SEED}.csv"
    if [ -f "$OUT_CSV" ]; then
      echo "[SKIP] $OUT_CSV already exists"
      continue
    fi
    echo "=== FedEvo-advanced topk | alpha=${ALPHA} | seed=${SEED} ==="
    python3 main.py \
      --algo fedevo \
      --deploy_model topk \
      --dataset $DATASET \
      --alpha "$ALPHA" \
      --rounds $ROUNDS \
      --num_clients $CLIENTS \
      --clients_per_round $CPR \
      --epochs $EPOCHS \
      --batch_size $BATCH \
      --lr $LR \
      --lr_decay $LR_DECAY \
      --momentum $MOMENTUM \
      --weight_decay $WD \
      --m 10 \
      --rho 0.3 \
      --gamma 1.5 \
      --tau_factor 0.8 \
      --sigma_mut 0.01 \
      --num_interp 4 \
      --num_orth 1 \
      --state_mode params \
      --bn_recalibrate_batches 100 \
      --seed_data $SEED_DATA \
      --seed_sample $SEED_SAMPLE \
      --seed_train "$SEED" \
      --seed_evo 2025 \
      --out_csv "$OUT_CSV" \
      --data_dir ./data
  done
done

# ---- FedEvo-stab (advanced) ----
for ALPHA in "${ALPHAS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    ALPHA_STR=$(echo "$ALPHA" | tr '.' 'p')
    OUT_CSV="$OUT_DIR/fedevo_advanced_stab_${DATASET}_alpha${ALPHA_STR}_seed${SEED}.csv"
    if [ -f "$OUT_CSV" ]; then
      echo "[SKIP] $OUT_CSV already exists"
      continue
    fi
    echo "=== FedEvo-advanced stab | alpha=${ALPHA} | seed=${SEED} ==="
    python3 main.py \
      --algo fedevo \
      --deploy_model stab \
      --dataset $DATASET \
      --alpha "$ALPHA" \
      --rounds $ROUNDS \
      --num_clients $CLIENTS \
      --clients_per_round $CPR \
      --epochs $EPOCHS \
      --batch_size $BATCH \
      --lr $LR \
      --lr_decay $LR_DECAY \
      --momentum $MOMENTUM \
      --weight_decay $WD \
      --m 10 \
      --rho 0.3 \
      --gamma 1.5 \
      --tau_factor 0.8 \
      --sigma_mut 0.01 \
      --num_interp 4 \
      --num_orth 1 \
      --state_mode params \
      --bn_recalibrate_batches 100 \
      --seed_data $SEED_DATA \
      --seed_sample $SEED_SAMPLE \
      --seed_train "$SEED" \
      --seed_evo 2025 \
      --out_csv "$OUT_CSV" \
      --data_dir ./data
  done
done

echo "Done. Results in: $OUT_DIR"
