"""
Fair comparison 결과 분석
results/fair/ 폴더의 CSV를 읽어 seed별 평균/std 계산
Usage: python analyze_fair.py
"""

import os
import glob
import pandas as pd
import numpy as np

RESULTS_DIR = "./results"
LAST_N = 50  # last 50 rounds mean (GROUND_RULES 주 지표)

ALGOS  = ["fedavg", "fedmut", "fedevo_stab", "feddyn"]
ALPHAS = ["0p1", "1", "10"]
SEEDS  = [44, 45, 46, 47, 48]

def alpha_label(a): return a.replace("p", ".")

rows = []
for algo in ALGOS:
    for alpha_str in ALPHAS:
        accs = []
        for seed in SEEDS:
            pattern = f"{RESULTS_DIR}/{algo}_cifar10_alpha{alpha_str}_seed{seed}.csv"
            files = glob.glob(pattern)
            if not files:
                continue
            df = pd.read_csv(files[0])
            last50 = df["test_accuracy"].values[-LAST_N:]
            accs.append(float(np.mean(last50)))

        if not accs:
            continue

        rows.append({
            "algo": algo,
            "alpha": alpha_label(alpha_str),
            "n_seeds": len(accs),
            "mean_acc": np.mean(accs),
            "std_acc":  np.std(accs),
            "per_seed": accs,
        })

if not rows:
    print("결과 없음 — results/ 폴더를 확인하세요.")
else:
    print(f"\n{'='*65}")
    print(f" Fair Comparison — Last {LAST_N} rounds mean accuracy")
    print(f"{'='*65}")
    print(f"{'Algo':<15} {'Alpha':>6} {'Seeds':>6}  {'Mean':>7}  {'Std':>6}")
    print(f"{'-'*65}")
    for r in rows:
        seeds_str = f"({r['n_seeds']}/5)"
        print(f"{r['algo']:<15} {r['alpha']:>6} {seeds_str:>6}  {r['mean_acc']*100:>6.2f}%  {r['std_acc']*100:>5.2f}%")
    print(f"{'='*65}\n")

    # 완료된 실험 목록
    print("완료된 실험:")
    for algo in ALGOS:
        for alpha_str in ALPHAS:
            done = []
            missing = []
            for seed in SEEDS:
                pattern = f"{RESULTS_DIR}/{algo}_cifar10_alpha{alpha_str}_seed{seed}.csv"
                if glob.glob(pattern):
                    done.append(seed)
                else:
                    missing.append(seed)
            status = f"done={done}" + (f"  MISSING={missing}" if missing else "")
            print(f"  {algo:<15} alpha={alpha_label(alpha_str):>4}  {status}")
