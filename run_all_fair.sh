#!/bin/bash
# Master script: 전체 fair comparison 실험
# FedAvg + FedMut + FedEvo(stab) x alpha{0.1,1,10} x seeds{44..48}
# 총 45 runs

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

chmod +x run_fair_fedavg.sh run_fair_fedmut.sh run_fair_fedevo.sh

echo "================================================"
echo " Fair Comparison Experiments"
echo " FedAvg / FedMut / FedEvo(stab)"
echo " alpha: 0.1, 1, 10  |  seeds: 44~48"
echo " total: 45 runs"
echo "================================================"
echo ""

ALPHAS=(0.1 1 10)
SEEDS=(44 45 46 47 48)

echo ">>> [1/3] FedAvg"
for alpha in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    bash run_fair_fedavg.sh "$alpha" "$seed"
  done
done

echo ""
echo ">>> [2/3] FedMut"
for alpha in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    bash run_fair_fedmut.sh "$alpha" "$seed"
  done
done

echo ""
echo ">>> [3/3] FedEvo (stab)"
for alpha in "${ALPHAS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    bash run_fair_fedevo.sh "$alpha" "$seed"
  done
done

echo ""
echo "================================================"
echo " ALL DONE — results in ./results/fair/"
echo "================================================"
