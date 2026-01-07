# FedEvo: Federated Learning Algorithms

This repository implements several federated learning algorithms according to the IJCNN implementation contract.

## Algorithms

- **FedAvg**: Standard Federated Averaging
- **FedMut**: FedAvg with gradient-based mutation
- **FedImpro**: FedAvg with class-wise feature gap analysis
- **FedEvo**: Evolutionary federated learning with population of models

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run any algorithm with the main script:

```bash
python main.py --algorithm fedavg --num_rounds 10 --num_clients 5
python main.py --algorithm fedmut --num_rounds 10 --num_clients 5
python main.py --algorithm fedimpro --num_rounds 10 --num_clients 5
python main.py --algorithm fedevo --num_rounds 10 --num_clients 5
```

### Arguments

- `--algorithm`: Algorithm to use (fedavg, fedmut, fedimpro, fedevo)
- `--num_rounds`: Number of federated learning rounds
- `--num_clients`: Number of clients
- `--seed`: Random seed for reproducibility

## Output

The script outputs CSV format with columns:
- `round`: Round index
- `test_accuracy`: Test accuracy percentage
- `train_loss`: Training loss
- `uplink_bytes`: Total uplink bytes from all clients

## Implementation Details

### FedMut (algorithms/fedmut.py)
- Skips mutation at round 0 (only performs FedAvg)
- For rounds > 0, applies gradient-based mutation: `g_r = w_curr - w_prev`
- Mask granularity: one scalar sign per parameter tensor

### FedImpro (algorithms/fedimpro.py)
- Optimized feature collection: iterates only over unique labels in each batch
- Uses deterministic RNG for synthetic label sampling: `seed + 9999`
- Dynamic feature dimension D from `h.shape[1]`

### FedEvo (algorithms/fedevo.py)
- Maintains population of candidate models
- Evaluates candidate with highest usage in current round
- Does not evaluate candidate 0

### BatchNorm Buffers Policy
- Only learnable parameters are aggregated (via `named_parameters()`)
- BN running stats (buffers) are NOT aggregated
- BN buffers remain local to each client
- See comments in `algorithms/base.py` and `main.py`

## File Structure

```
.
├── algorithms/
│   ├── __init__.py
│   ├── base.py       # Base classes and FedAvg
│   ├── fedmut.py     # FedMut implementation
│   ├── fedimpro.py   # FedImpro implementation
│   └── fedevo.py     # FedEvo implementation
├── main.py           # Main training script
├── requirements.txt  # Python dependencies
└── README.md         # This file
```
