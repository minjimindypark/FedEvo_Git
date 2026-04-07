import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── 설정 ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR  = RESULTS_DIR

ALG_DISPLAY = {
    "fedavg":                               "FedAvg",
    "fedmut":                               "FedMut",
    "fedevo_stab":                          "FedEvo-Stab",
    "fedevo_stab_interpolation1":           "FedEvo-Stab(interp=1)",
    "fedevo_stab-lamdachangeto08":          "FedEvo-Stab(λ=0.8)",
    "fedevo_topk":                          "FedEvo-TopK",
    "feddyn":                               "FedDyn",
    "fedevo_stab_track_m5":                 "FedEvo-Stab m=5",
}

COLORS = {
    "FedAvg":                              "#4C72B0",
    "FedMut":                              "#DD8452",
    "FedEvo-Stab":                         "#55A868",
    "FedEvo-Advanced(Stab)":               "#C44E52",
    "FedEvo-Advanced(Stab, interp=1)":     "#E08000",
    "FedEvo-Stab(λ=0.8)":                  "#8172B2",
    "FedEvo-TopK":                         "#E377C2",
    "FedDyn":                              "#937860",
    "FedEvo-Stab m=5":                     "#17BECF",
}

# ── CSV 파싱 ──────────────────────────────────────────────────────────────────
PATTERN = re.compile(
    r"^(?P<alg>.+?)_cifar10_alpha(?P<alpha>[0-9p]+)(?:_seed(?P<seed>\d+))?(?:_(?P<suffix>interpolation\d+|m\d+))?\.csv$"
)

EXCLUDE_PATTERNS = ["fedevo_advanced"]

records = []
for fname in os.listdir(RESULTS_DIR):
    # fedevo_advanced 제외
    if any(ex in fname for ex in EXCLUDE_PATTERNS):
        continue
    m = PATTERN.match(fname)
    if not m:
        continue
    alg    = m.group("alg")
    alpha  = m.group("alpha").replace("p", ".")
    seed   = m.group("seed") or "unknown"
    suffix = m.group("suffix")

    # feddyn은 파일명이 feddyn_alpha0p01_cifar10_alpha... 형태
    if alg.startswith("feddyn_alpha"):
        alg = "feddyn"

    # interpolation 등 suffix가 있으면 alg 키에 포함
    if suffix:
        alg = f"{alg}_{suffix}"

    # seed44 제외 (논문 기준: α=0.1에서 population collapse 이상치)
    if seed == "44":
        continue

    df = pd.read_csv(os.path.join(RESULTS_DIR, fname))
    df["alg"]   = alg
    df["alpha"] = alpha
    df["seed"]  = seed
    records.append(df)

if not records:
    raise RuntimeError("CSV 파일을 찾지 못했습니다.")

data = pd.concat(records, ignore_index=True)
data["alg_label"] = data["alg"].map(ALG_DISPLAY).fillna(data["alg"])
alphas = sorted(data["alpha"].unique(), key=float)

# ── 1. 최종 accuracy 테이블 (논문 기준: last-50 rounds mean, ddof=1) ──────────
last = (data.groupby(["alg", "alg_label", "alpha", "seed"])
            .apply(lambda g: g.nlargest(50, "round")["test_accuracy"].mean())
            .reset_index(name="test_accuracy"))
summary = (last.groupby(["alg_label", "alpha"])["test_accuracy"]
               .agg(mean="mean", std=lambda x: x.std(ddof=1), n="count")
               .reset_index())
summary["mean±std"] = summary.apply(
    lambda r: f"{r['mean']*100:.2f} ± {r['std']*100:.2f}", axis=1)

pivot = summary.pivot(index="alg_label", columns="alpha", values="mean±std")
pivot.index.name = "Algorithm"
pivot.columns = [f"α={a}" for a in pivot.columns]

print("\n=== 최종 Test Accuracy (mean ± std %) ===")
print(pivot.to_string())
pivot.to_csv(os.path.join(OUTPUT_DIR, "summary_table.csv"))
print("\n→ summary_table.csv 저장됨")

# ── 2. Learning curve 플롯 (α별 서브플롯) ────────────────────────────────────
fig, axes = plt.subplots(1, len(alphas), figsize=(6 * len(alphas), 5), sharey=False)
if len(alphas) == 1:
    axes = [axes]

for ax, alpha in zip(axes, alphas):
    subset = data[data["alpha"] == alpha]
    for alg_label, grp in subset.groupby("alg_label"):
        curve = grp.groupby("round")["test_accuracy"].agg(["mean", "std"]).reset_index()
        color = COLORS.get(alg_label, None)
        ax.plot(curve["round"], curve["mean"] * 100,
                label=alg_label, color=color, linewidth=1.8)
        ax.fill_between(curve["round"],
                        (curve["mean"] - curve["std"]) * 100,
                        (curve["mean"] + curve["std"]) * 100,
                        alpha=0.15, color=color)

    ax.set_title(f"CIFAR-10  α={alpha}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(True, linestyle="--", alpha=0.4)

fig.suptitle("Federated Learning: Algorithm Comparison", fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
curve_path = os.path.join(OUTPUT_DIR, "learning_curves.png")
fig.savefig(curve_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"→ learning_curves.png 저장됨")

# ── 3. 최종 accuracy 바 차트 ─────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 5), sharey=False)
if len(alphas) == 1:
    axes2 = [axes2]

for ax, alpha in zip(axes2, alphas):
    sub = summary[summary["alpha"] == alpha].sort_values("mean", ascending=False)
    colors = [COLORS.get(lbl, "#999999") for lbl in sub["alg_label"]]
    bars = ax.bar(sub["alg_label"], sub["mean"] * 100,
                  yerr=sub["std"] * 100, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.6)
    ax.set_title(f"α={alpha}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=11)
    ax.set_ylim(0, min(100, sub["mean"].max() * 100 * 1.15))
    ax.set_xticklabels(sub["alg_label"], rotation=25, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f%%"))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

fig2.suptitle("Final Test Accuracy Comparison", fontsize=15, fontweight="bold", y=1.01)
fig2.tight_layout()
bar_path = os.path.join(OUTPUT_DIR, "final_accuracy_bar.png")
fig2.savefig(bar_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"→ final_accuracy_bar.png 저장됨")

print("\n분석 완료.")
