# autotune_fedevo_paper.py
"""
FedEvo auto-tuning (paper-oriented)

What this does
- Repeatedly runs `main.py --algo fedevo ...` over a small search space.
- Parses FedEvo server logs:
    [FedEvo R{r}] H=... τ=... |S_j|=[...] mut=Y/N attr_r=... attr_c=... margin(mean/p50/p90)=...
- Scores trials with a focus on pushing margin into [1e-3, 1e-2] while keeping attribution healthy.
- "Paper-ish" scoring: ignore warmup rounds (default: round 1), score only rounds >= 2.

Quality-of-life extras
- bias_only 자동 d 상한:
  If main.py fails with "Low-sensitivity pool too small: POOL < need=NEED",
  we infer m ≈ NEED / d and retry once with d' = floor(POOL / m).
- 결과 요약 CSV: autotune_summary.csv (one row per trial)
- 결과 JSON: autotune_summary.json (full details)

Assumed main.py CLI (from your repo)
- Required: --algo fedevo --dataset {cifar10,cifar100} --rounds R
- FedEvo knobs: --nu_scale, --d, --local_steps, --low_sens_mode, --seed_train
- Data knobs: --alpha, --data_dir
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------
# Regex: FedEvo metrics line
# -----------------------------
_FEDEVO_LINE_RE = re.compile(
    r"\[FedEvo R(?P<round>\d+)\]\s+"
    r"H=(?P<H>[0-9.eE+-]+)\s+τ=(?P<tau>[0-9.eE+-]+)\s+"
    r"\|S_j\|=\[(?P<Sj>[0-9,\s]+)\]\s+"
    r"mut=(?P<mut>[YN])\s+"
    r"attr_r=(?P<attr_r>[0-9.eE+-]+)\s+attr_c=(?P<attr_c>[0-9.eE+-]+)\s+"
    r"margin\(mean/p50/p90\)=(?P<m_mean>[0-9.eE+-]+)/(?P<m_p50>[0-9.eE+-]+)/(?P<m_p90>[0-9.eE+-]+)"
)

# Parse this FedEvo init check (pool too small)
_POOL_TOO_SMALL_RE = re.compile(
    r"Low-sensitivity pool too small:\s*(?P<pool>\d+)\s*<\s*need=(?P<need>\d+)",
    re.IGNORECASE,
)


# -----------------------------
# Data classes
# -----------------------------
@dataclass(frozen=True)
class TrialConfig:
    dataset: str
    alpha: float
    data_dir: str
    rounds: int
    seed_train: int

    # FedEvo knobs
    nu_scale: float
    d: int
    local_steps: int
    low_sens_mode: str  # "bias_only" | "bias_norm"


@dataclass
class TrialResult:
    cfg: TrialConfig
    ok: bool
    reason: str

    # parsed (averaged over scored rounds)
    attr_r: float = 0.0
    attr_c: float = 0.0
    margin_mean: float = 0.0
    margin_p50: float = 0.0
    margin_p90: float = 0.0
    H: float = 0.0
    tau: float = 0.0

    # scoring
    score: float = -1e9

    # bookkeeping
    adjusted_d: Optional[int] = None
    log_path: Optional[str] = None


# -----------------------------
# Helpers
# -----------------------------
def _safe_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _parse_fedevo_rows(stdout: str) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for line in stdout.splitlines():
        m = _FEDEVO_LINE_RE.search(line.strip())
        if not m:
            continue
        rows.append(
            {
                "round": float(_safe_float(m.group("round"))),
                "H": _safe_float(m.group("H")),
                "tau": _safe_float(m.group("tau")),
                "attr_r": _safe_float(m.group("attr_r")),
                "attr_c": _safe_float(m.group("attr_c")),
                "margin_mean": _safe_float(m.group("m_mean")),
                "margin_p50": _safe_float(m.group("m_p50")),
                "margin_p90": _safe_float(m.group("m_p90")),
            }
        )
    return rows


def _avg(rows: Sequence[Dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return float(sum(r.get(key, 0.0) for r in rows) / len(rows))


def _score_metrics(
    attr_r: float,
    attr_c: float,
    margin_mean: float,
    H: float,
    margin_target_low: float = 1e-3,
    margin_target_high: float = 1e-2,
) -> float:
    """
    Objective:
    - margin_mean in [1e-3, 1e-2]
    - increase attribution
    - keep entropy non-collapsing (soft)
    """
    eps = 1e-12
    logm = math.log10(max(margin_mean, eps))
    log_lo = math.log10(margin_target_low)
    log_hi = math.log10(margin_target_high)

    if log_lo <= logm <= log_hi:
        margin_reward = 1.0
    else:
        dist = min(abs(logm - log_lo), abs(logm - log_hi))
        margin_reward = max(0.0, 1.0 - dist / 2.0)  # 2 decades away -> ~0

    # m=5 => H in [0, ln(5)~1.609]. Prefer around ~1.0 (not too collapsed)
    H_pref = max(0.0, 1.0 - abs(H - 1.0) / 1.5)

    attr_reward = 0.5 * (attr_r + attr_c)  # 0..1

    return 3.0 * margin_reward + 2.0 * attr_reward + 0.5 * H_pref


def _build_cmd(
    python_exe: str,
    main_py: str,
    out_dir: str,
    cfg: TrialConfig,
) -> List[str]:
    # Note: pass ONLY flags that main.py actually defines (yours does).
    return [
        python_exe,
        main_py,
        "--algo",
        "fedevo",
        "--dataset",
        cfg.dataset,
        "--alpha",
        str(cfg.alpha),
        "--rounds",
        str(cfg.rounds),
        "--data_dir",
        cfg.data_dir,
        "--seed_train",
        str(cfg.seed_train),
        "--nu_scale",
        str(cfg.nu_scale),
        "--d",
        str(cfg.d),
        "--local_steps",
        str(cfg.local_steps),
        "--low_sens_mode",
        cfg.low_sens_mode,
        "--out_dir",
        out_dir,
    ]


def _maybe_adjust_d_for_bias_only(stdout: str, cfg: TrialConfig) -> Optional[int]:
    """
    If failure shows pool too small, propose a smaller d that fits:
      need = m * d
      d_max = floor(pool / m)
    We infer m from need/d (rounded).
    """
    if cfg.low_sens_mode != "bias_only":
        return None

    m = _POOL_TOO_SMALL_RE.search(stdout)
    if not m:
        return None

    pool = int(m.group("pool"))
    need = int(m.group("need"))
    if cfg.d <= 0:
        return None

    # infer m ≈ need / d
    m_est = max(1, int(round(need / float(cfg.d))))
    d_max = pool // m_est
    if d_max <= 0:
        return None

    # only adjust if it actually reduces d
    if d_max < cfg.d:
        return d_max
    return None


# -----------------------------
# Trial runner
# -----------------------------
def run_trial(
    python_exe: str,
    main_py: str,
    out_dir: Path,
    cfg: TrialConfig,
    warmup_rounds: int,
    timeout_sec: int,
) -> TrialResult:
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = f"ds={cfg.dataset}_a={cfg.alpha}_nu={cfg.nu_scale}_d={cfg.d}_s={cfg.local_steps}_m={cfg.low_sens_mode}_seed={cfg.seed_train}_{stamp}"
    log_path = out_dir / f"trial_{tag}.log"

    # 1) run
    cmd = _build_cmd(python_exe, main_py, str(out_dir), cfg)
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        stdout = proc.stdout or ""
        log_path.write_text(stdout, encoding="utf-8")

        # 2) parse rows
        rows = _parse_fedevo_rows(stdout)
        if not rows:
            # 2.1) bias_only d 자동 제한 후 1회 재시도
            new_d = _maybe_adjust_d_for_bias_only(stdout, cfg)
            if new_d is not None:
                cfg2 = TrialConfig(**{**asdict(cfg), "d": int(new_d)})
                cmd2 = _build_cmd(python_exe, main_py, str(out_dir), cfg2)
                proc2 = subprocess.run(
                    cmd2,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=timeout_sec,
                    check=False,
                )
                stdout2 = proc2.stdout or ""
                # append to same log (so you can see both attempts)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write("\n\n===== RETRY (auto d cap) =====\n")
                    f.write("CMD: " + " ".join(cmd2) + "\n\n")
                    f.write(stdout2)

                rows2 = _parse_fedevo_rows(stdout2)
                if not rows2:
                    return TrialResult(cfg=cfg, ok=False, reason="No FedEvo metrics parsed (even after d-cap retry)", adjusted_d=int(new_d), log_path=str(log_path))
                rows = rows2
                cfg = cfg2  # treat adjusted cfg as effective
                adjusted_d = int(new_d)
            else:
                return TrialResult(cfg=cfg, ok=False, reason="No FedEvo metrics parsed", log_path=str(log_path))
        else:
            adjusted_d = None

        # 3) warmup drop
        if warmup_rounds > 0 and len(rows) > warmup_rounds:
            scored_rows = rows[warmup_rounds:]
        else:
            scored_rows = rows

        # 4) aggregate
        attr_r = _avg(scored_rows, "attr_r")
        attr_c = _avg(scored_rows, "attr_c")
        margin_mean = _avg(scored_rows, "margin_mean")
        margin_p50 = _avg(scored_rows, "margin_p50")
        margin_p90 = _avg(scored_rows, "margin_p90")
        H = _avg(scored_rows, "H")
        tau = _avg(scored_rows, "tau")

        score = _score_metrics(attr_r=attr_r, attr_c=attr_c, margin_mean=margin_mean, H=H)

        return TrialResult(
            cfg=cfg,
            ok=True,
            reason="OK",
            attr_r=attr_r,
            attr_c=attr_c,
            margin_mean=margin_mean,
            margin_p50=margin_p50,
            margin_p90=margin_p90,
            H=H,
            tau=tau,
            score=score,
            adjusted_d=adjusted_d,
            log_path=str(log_path),
        )

    except subprocess.TimeoutExpired:
        return TrialResult(cfg=cfg, ok=False, reason=f"Timeout({timeout_sec}s)", log_path=str(log_path))
    except FileNotFoundError as e:
        return TrialResult(cfg=cfg, ok=False, reason=f"FileNotFound: {e}", log_path=None)
    except Exception as e:
        return TrialResult(cfg=cfg, ok=False, reason=f"Exception: {e}", log_path=str(log_path))


# -----------------------------
# Search space
# -----------------------------
def generate_space(dataset: str, alpha: float, data_dir: str, rounds: int, seed_train: int) -> List[TrialConfig]:
    """
    Practical heuristic:
    - local_steps 1..2 first (sentinel survives; separation first)
    - bias_only then bias_norm
    - d: moderate -> larger (but bias_only will auto-cap if too large)
    - nu_scale: small -> larger
    """
    local_steps_list = [1, 2]
    modes = ["bias_only", "bias_norm"]
    d_list = [512, 1024, 1536, 2048]
    nu_list = [0.005, 0.01, 0.02, 0.05, 0.08, 0.12]

    out: List[TrialConfig] = []
    for steps in local_steps_list:
        for mode in modes:
            for d in d_list:
                for nu in nu_list:
                    out.append(
                        TrialConfig(
                            dataset=dataset,
                            alpha=float(alpha),
                            data_dir=str(data_dir),
                            rounds=int(rounds),
                            seed_train=int(seed_train),
                            nu_scale=float(nu),
                            d=int(d),
                            local_steps=int(steps),
                            low_sens_mode=str(mode),
                        )
                    )
    return out


def print_best_cmd(best: TrialResult, python_exe: str, main_py: str) -> None:
    c = best.cfg
    cmd = " ".join(
        _build_cmd(
            python_exe=python_exe,
            main_py=main_py,
            out_dir="<OUT_DIR>",
            cfg=c,
        )
    )
    print("\n=== Recommended command (fill OUT_DIR) ===")
    print(cmd)
    print("=========================================\n")


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable, help="Path to python executable (venv python recommended)")
    ap.add_argument("--main_py", default="main.py", help="Path to main.py")
    ap.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    ap.add_argument("--alpha", type=float, default=0.1, choices=[0.1, 0.5])
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--rounds", type=int, default=2, help="Rounds per trial (keep small for tuning)")
    ap.add_argument("--seed_train", type=int, default=44)
    ap.add_argument("--trials", type=int, default=30, help="Budget (top-N configs in heuristic order)")
    ap.add_argument("--out_dir", default="results_autotune", help="Directory to store logs/results")
    ap.add_argument("--warmup_rounds", type=int, default=1, help="Ignore first N rounds when scoring")
    ap.add_argument("--timeout_sec", type=int, default=1800)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    space = generate_space(args.dataset, args.alpha, args.data_dir, args.rounds, args.seed_train)
    space = space[: max(1, int(args.trials))]

    print(f"[AutoTune] trials={len(space)} out_dir={out_dir.resolve()}")
    print(f"[AutoTune] scoring: warmup_rounds={int(args.warmup_rounds)} (ignore first N rounds)")

    results: List[TrialResult] = []
    best: Optional[TrialResult] = None

    # CSV summary (streaming)
    csv_path = out_dir / "autotune_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(
            fcsv,
            fieldnames=[
                "ok",
                "reason",
                "score",
                "dataset",
                "alpha",
                "rounds",
                "seed_train",
                "nu_scale",
                "d",
                "adjusted_d",
                "local_steps",
                "low_sens_mode",
                "attr_r",
                "attr_c",
                "H",
                "tau",
                "margin_mean",
                "margin_p50",
                "margin_p90",
                "log_path",
            ],
        )
        w.writeheader()

        for i, cfg in enumerate(space, 1):
            print(f"\n[AutoTune] Trial {i}/{len(space)}: nu={cfg.nu_scale} d={cfg.d} steps={cfg.local_steps} mode={cfg.low_sens_mode}")
            tr = run_trial(
                python_exe=args.python,
                main_py=args.main_py,
                out_dir=out_dir,
                cfg=cfg,
                warmup_rounds=int(args.warmup_rounds),
                timeout_sec=int(args.timeout_sec),
            )
            results.append(tr)

            if tr.ok:
                print(
                    f"  -> OK: score={tr.score:.3f} attr_r={tr.attr_r:.3f} attr_c={tr.attr_c:.3f} "
                    f"margin_mean={tr.margin_mean:.4g} (p50={tr.margin_p50:.4g}, p90={tr.margin_p90:.4g}) "
                    f"H={tr.H:.3f}"
                    + (f" [d capped -> {tr.cfg.d}]" if tr.adjusted_d is not None else "")
                )
                if best is None or tr.score > best.score:
                    best = tr
                    print("  -> NEW BEST ✓")
            else:
                print(f"  -> FAIL: {tr.reason}")

            w.writerow(
                {
                    "ok": tr.ok,
                    "reason": tr.reason,
                    "score": tr.score,
                    "dataset": tr.cfg.dataset,
                    "alpha": tr.cfg.alpha,
                    "rounds": tr.cfg.rounds,
                    "seed_train": tr.cfg.seed_train,
                    "nu_scale": tr.cfg.nu_scale,
                    "d": tr.cfg.d,
                    "adjusted_d": tr.adjusted_d,
                    "local_steps": tr.cfg.local_steps,
                    "low_sens_mode": tr.cfg.low_sens_mode,
                    "attr_r": tr.attr_r,
                    "attr_c": tr.attr_c,
                    "H": tr.H,
                    "tau": tr.tau,
                    "margin_mean": tr.margin_mean,
                    "margin_p50": tr.margin_p50,
                    "margin_p90": tr.margin_p90,
                    "log_path": tr.log_path,
                }
            )
            fcsv.flush()

    # JSON dump
    json_path = out_dir / "autotune_summary.json"
    payload = {"best": asdict(best) if best else None, "results": [asdict(r) for r in results]}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[AutoTune] Saved: {csv_path.resolve()}")
    print(f"[AutoTune] Saved: {json_path.resolve()}")

    if best:
        print_best_cmd(best, args.python, args.main_py)
        print(f"[AutoTune] Best log: {best.log_path}")
        return 0

    print("[AutoTune] No successful trials.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
