# FedEvo: Genetic Candidate Evolution in Federated Learning

This repository contains the official implementation of **FedEvo** (Federated Genetic Optimization) and **FedAvg** (Federated Averaging) on CIFAR-10/100 datasets using PyTorch.

## 📋 Prerequisites

Ensure your system meets the following requirements:
* **OS:** Linux (Recommended) or Windows
* **Python:** 3.12+
* **CUDA:** 12.1 (for GPU acceleration)

---

## 🛠️ Installation & Setup

Follow these steps to set up the environment using `venv`.

### 1. Create a Virtual Environment
It is highly recommended to run this project in an isolated virtual environment to avoid dependency conflicts.

```bash
# 1. Create venv (named 'venv')
python3.12 -m venv venv

# 2. Activate venv
# For Windows (PowerShell/CMD):
venv\Scripts\activate

# For Linux/Mac:
source venv/bin/activate
2. Install Dependencies (CUDA 12.1)Install PyTorch with CUDA 12.1 support and other required libraries.Bash# Upgrade pip first
pip install --upgrade pip

# Install PyTorch (CUDA 12.1 specific)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

# Install other dependencies
pip install numpy pandas matplotlib scipy
Note: If you do not have a GPU, you can remove --index-url ... to install the CPU version, but training will be slow.📂 Project StructureEnsure your directory looks like this:Plaintext.
├── algorithms/             # Algorithm implementations
│   ├── __init__.py
│   ├── base.py             # Base classes & utilities
│   ├── fedavg.py           # FedAvg implementation (if separated)
│   └── fedevo.py           # FedEvo runner & genetic operators
├── data_utils.py           # Data loading & Dirichlet/IID partitioning
├── models.py               # ResNet-18 model architecture
├── main.py                 # Main entry point for training
└── README.md               # This file

🚀 How to RunThe main.py script handles both algorithms. Use --algo to switch between them.1. Run FedEvo (Proposed Method)Example: CIFAR-10, Non-IID (Dirichlet alpha=0.1), Population size=10.Bashpython main.py \
    --algo fedevo \
    --dataset cifar10 \
    --alpha 0.1 \
    --rounds 1000 \
    --m 10 \
    --rho 0.3 \
    --gamma 1.5 \
    --state_mode params \
    --gpu 0
2. Run FedAvg (Baseline)Example: CIFAR-10, Non-IID (Dirichlet alpha=0.1).Bashpython main.py \
    --algo fedavg \
    --dataset cifar10 \
    --alpha 0.1 \
    --rounds 1000 \
    --clients_per_round 10 \
    --epochs 5 \
    --state_mode params
3. Run on CIFAR-100Simply change the dataset flag and adjust alpha if needed.Bashpython main.py --algo fedevo --dataset cifar100 --alpha 0.5 --rounds 1000

⚙️ Key Arguments Configuration

Argument,Description,Recommended Default
--algo,"Algorithm to choose (fedevo, fedavg)",fedevo
--dataset,"Target dataset (cifar10, cifar100)",-
--alpha,"Dirichlet concentration (0.1 = high hetero, 100 = IID)",0.1 or 0.5
--rounds,Total communication rounds,1000
--m,Population size (Only for FedEvo),10
--rho,Top-k selection ratio (Only for FedEvo),0.3
--gamma,Selection weight amplification (Only for FedEvo),1.5
--no_orth,Disable orthogonality injection (Ablation),False
--no_mut,Disable mutation (Ablation),False
--state_mode,params (weights only) or float (weights + buffers),params

📊 Outputs
Console: Real-time logs of test accuracy and loss.

CSV Logs: Saved in ./results/ with timestamps.

Checkpoints: Saved in ./results/ (or --ckpt_dir) as .pt files.

To resume a run from a checkpoint:

Bash
python main.py --resume ./results/last.pt