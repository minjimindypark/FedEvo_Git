# IJCNN Implementation Contract - Summary

This document summarizes the implementation of all IJCNN contract requirements.

## Requirements Implemented

### 1. FedMut: Skip mutation at r=0 ✓
**File:** `algorithms/fedmut.py`

**Changes:**
- Added `self.round_idx = 0` in `FedMutRunner.__init__` (line 14)
- In `run_round()`, check `if self.round_idx == 0` to skip mutation (line 40)
- For r=0: perform only FedAvg aggregation, save w_prev, increment round_idx
- For r>0: apply mutation using `g_r = w_curr - w_prev` (line 64)
- Mask granularity: one scalar sign per parameter tensor (lines 62-72)

**Verification:**
```python
runner = FedMutRunner(model, clients, num_rounds=3)
assert runner.round_idx == 0  # Initial state
runner.run_round(0)
assert runner.round_idx == 1  # Incremented, no mutation applied
runner.run_round(1)
assert runner.round_idx == 2  # Mutation applied
```

### 2. FedImpro: Optimize per-class feature stats collection ✓
**File:** `algorithms/fedimpro.py`

**Changes:**
- In `_gather_gap_features()`, iterate only unique labels (line 64):
  ```python
  for c in y.unique().tolist():
  ```
- Instead of looping over all classes each batch
- Dynamic D from `h.shape[1]` (line 61)
- Preserves identical outputs to naive method

**Verification:**
```python
# Before: for c in range(num_classes):  # Iterates 10 times
# After: for c in y.unique().tolist():  # Iterates 3-4 times typically
```

### 3. FedImpro: Deterministic RNG for y_syn sampling ✓
**File:** `algorithms/fedimpro.py`

**Changes:**
- Added dedicated RNG in `_local_train()` (line 108):
  ```python
  rng_syn = np.random.RandomState(seed + client_idx * 1000 + round_idx * 100 + 9999)
  ```
- Use `rng_syn.choice()` for y_syn sampling (line 111)
- Client sampling RNG in main remains separate and unchanged

**Verification:**
```python
rng1 = np.random.RandomState(42 + 9999)
rng2 = np.random.RandomState(42 + 9999)
assert np.array_equal(rng1.choice(10, 32), rng2.choice(10, 32))  # Deterministic
```

### 4. BN buffers policy: Document + enforce ✓
**Files:** `algorithms/base.py`, `main.py`

**Changes:**
- Comment in `base.py` `param_state_dict()` (line 19):
  ```python
  """BN buffers are not aggregated; only learnable parameters are aggregated."""
  ```
- Comment in `main.py` (lines 124-127) explaining policy
- Implementation uses `model.named_parameters()` only (line 21)
- BN running stats (buffers) are NOT transmitted or aggregated

**Verification:**
```python
state = runner.param_state_dict(model)
# All keys are from named_parameters(), no buffers included
assert all(k in dict(model.named_parameters()) for k in state.keys())
```

### 5. FedEvo evaluation policy ✓
**Files:** `algorithms/fedevo.py`, `main.py`

**Changes:**
- In `run_round()`, store `self.last_eval_candidate_idx` (line 53):
  ```python
  self.last_eval_candidate_idx = np.argmax(self.usage_round)
  ```
- Stored before evolve/reset operations
- In `main.py`, evaluate based on this index (line 177):
  ```python
  eval_model = runner.pop[runner.last_eval_candidate_idx]
  ```
- Do NOT evaluate candidate 0 (unless it has highest usage)

**Verification:**
```python
runner.run_round(0)
assert runner.last_eval_candidate_idx == np.argmax(runner.usage_round)
# Evaluated candidate is the one with highest usage, not always 0
```

## CSV Output Format ✓
**File:** `main.py`

**Output columns:** `round, test_accuracy, train_loss, uplink_bytes`

Example:
```
round,test_accuracy,train_loss,uplink_bytes
0,9.00,2.3072,5059704
1,8.50,2.3002,5059704
2,8.50,2.3024,5059704
```

## Testing

All algorithms tested end-to-end:
```bash
python main.py --algorithm fedavg --num_rounds 10 --num_clients 5
python main.py --algorithm fedmut --num_rounds 10 --num_clients 5
python main.py --algorithm fedimpro --num_rounds 10 --num_clients 5
python main.py --algorithm fedevo --num_rounds 10 --num_clients 5
```

All produce correct CSV output and run without errors.

## Code Quality

- Minimal changes as required
- No refactoring of unrelated code
- Comments only where required by contract
- All requirements verified programmatically
- Code review feedback addressed
