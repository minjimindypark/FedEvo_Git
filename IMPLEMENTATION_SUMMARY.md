# IJCNN Implementation Contract - Summary (Updated to current `main.py`)

This document summarizes the implementation of the IJCNN contract requirements and the **current**
end-to-end runner behavior in `main.py`.

> This file was updated to match the current CLI (`--algo`, `--dataset`, `--out_dir`, `--out_csv`) and
> the current FedEvo evaluation behavior (`runner.get_best_model()` loads `theta_base`).

---

## Requirements Implemented

### 1. FedMut: Skip mutation at r=0 ✓
**File:** `algorithms/fedmut.py`

**Intent:** round 0 performs FedAvg-only aggregation; mutation starts from round 1.

---

### 2. FedImpro: Optimize per-class feature stats collection ✓
**File:** `algorithms/fedimpro.py`

**Intent:** collect feature stats only for classes present in the batch (avoid full `for c in range(C)` each time).

---

### 3. FedImpro: Deterministic RNG for y_syn sampling ✓
**File:** `algorithms/fedimpro.py`

**Intent:** synthetic label sampling is deterministic given the training seed.

---

### 4. BatchNorm buffers policy (parameters-only aggregation) ✓
**Files:** `algorithms/base.py`, `main.py`

**Policy:** only **learnable parameters** are transmitted/aggregated (`model.named_parameters()`).
BN running stats (buffers) are **not** aggregated.

---

### 5. FedEvo evaluation policy (as currently implemented) ✓
**Files:** `algorithms/fedevo.py`, `main.py`

**Current behavior:**
- `main.py` evaluates FedEvo by calling `runner.get_best_model()`.
- `get_best_model()` loads `runner.theta_base` into `runner.model` and returns it.
- Therefore, the evaluated model is the **current stabilizer/base model** (`theta_base`), not “the most-used candidate” from the population.

If you want “evaluate most-used candidate” instead, it must be explicitly implemented and documented as a design choice.

---

## End-to-end Runner Defaults (Locked Setup)

`main.py` uses fixed defaults unless you change the code:
- `N=100` total clients
- `K=10` clients per round
- `E=5` local epochs
- `batch_size=50`
- `lr=0.01`, `momentum=0.9`, `weight_decay=1e-4`
- `gamma=0.998` (applied per-round to client LR)
- seeds: `seed_data=42`, `seed_sample=43`, `seed_train=44`

---

## CSV Output Format ✓

`main.py` writes:
- `round, test_accuracy, train_loss, uplink_bytes`

---

## Testing (Updated CLI)

Minimal smoke runs (2 rounds):

```bash
python main.py --algo fedmut  --dataset cifar10 --rounds 2 --out_dir ./results --data_dir ./data
python main.py --algo fedimpro --dataset cifar10 --rounds 2 --out_dir ./results --data_dir ./data
python main.py --algo fedevo  --dataset cifar10 --rounds 2 --out_dir ./results --data_dir ./data
```

(Windows example with a full absolute output directory is also supported via `--out_dir "C:\...\results"`.)

---

## Known Implementation Notes (FedEvo)

The FedEvo module documents these invariants in-code:
- `pop_raw` is sentinel-free and is the only persisted population state across rounds.
- `pop_sent` is recreated each round from `pop_raw` and used for attribution.
- Sentinel magnitude uses `nu` clamped to `[nu_min, nu_max]`.

See `algorithms/fedevo.py` module docstring for the complete list.
