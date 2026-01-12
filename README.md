# FedEvo: Federated Learning Algorithms (FedMut / FedImpro / FedEvo)

This repository provides a small, reproducible **CIFAR-10/100** federated-learning runner with three algorithms:

- **FedMut**: FedAvg-style aggregation with mutation after round 0
- **FedImpro**: FedAvg-style aggregation with class-wise feature gap statistics
- **FedEvo**: Evolutionary FL with a *population* of candidates and implicit client feedback (sentinel-based attribution)

> Note: The current `main.py` CLI exposes **fedmut / fedimpro / fedevo** (not FedAvg as a standalone option).

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start (your default run)

You said you'll run experiments like this each time:

```powershell
python main.py --algo fedevo --dataset cifar10 --rounds 2 --out_dir "C:\Users\박민지\OneDrive - 한국에너지공과대학교\바탕 화면\FedEvo\FedEvo_Git\results" --data_dir ./data
```

### What happens
- Results are saved as **CSV** inside `--out_dir`.
- The filename is auto-generated:

`{algo}_{dataset}_{YYYYMMDD-HHMMSS}.csv`

Example:
`fedevo_cifar10_20260112-104233.csv`

If you want to override the full output path (including the filename), use:

```powershell
python main.py --algo fedevo --dataset cifar10 --rounds 2 --out_csv "C:\path\to\my_custom_name.csv" --data_dir ./data
```

---

## CLI Arguments

- `--algo` : `fedmut | fedimpro | fedevo`
- `--dataset` : `cifar10 | cifar100`
- `--alpha` : Dirichlet partition parameter for label-skew (default `0.1`, options `0.1` or `0.5`)
- `--rounds` : number of FL rounds (default `1000`)
- `--out_dir` : directory for auto-named CSV output (default `./results`)
- `--out_csv` : optional explicit CSV path (overrides `--out_dir`)
- `--data_dir` : dataset directory (default `./data`)

---

## Output CSV Format

`main.py` writes one row per round:

- `round`
- `test_accuracy`
- `train_loss`
- `uplink_bytes`

---

## Important Reproducibility Note (Determinism)

`algorithms/base.py:set_global_seed()` enables **hard determinism** (cuDNN deterministic + `torch.use_deterministic_algorithms(True)`).
This improves reproducibility, but some PyTorch ops can error if they don't have deterministic implementations.

If you see errors like:
`RuntimeError: deterministic algorithms are not available for this operation`

the fix is to make determinism optional (e.g., `--deterministic 0/1`) and default it to OFF for beginners.

---

## Repository Structure

```
.
├── algorithms/
│   ├── base.py       # utilities + aggregation + evaluate + deterministic seeding
│   ├── fedmut.py
│   ├── fedimpro.py
│   └── fedevo.py
├── data_utils.py      # CIFAR loading + Dirichlet partition + per-client splits
├── models.py          # ResNet18_CIFAR (+ split model builder for FedImpro)
├── main.py            # end-to-end runner (CLI)
├── requirements.txt
└── README.md
```
