"""
main_iid_ga.py

Entry point for IID experiments focusing on Genetic Candidate Evolution (FedEvo-FL).
- Removes non-IID Dirichlet partition.
- Adds --algo {fedavg, fedevo}. (fedavg is FedEvo with m=1 and mutation off)
- Keeps evaluation pipeline consistent with existing codebase.

This file is designed to live alongside your existing main.py without breaking it.
"""

import os
from datetime import datetime

import argparse
import csv
from typing import Dict, List

import numpy as np
import torch

from algorithms.base import evaluate, make_loader, set_global_seed
from algorithms.fedevo import FedEvoRunner, GAConfig
from data_utils import client_train_val_split, load_cifar, make_subset
from models import ResNet18_CIFAR


def build_out_csv_path(out_dir: str, algo: str, dataset: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(out_dir, f"{algo}_{dataset}_IID_{ts}.csv")


def iid_partition(num_samples: int, num_clients: int, seed: int) -> Dict[int, List[int]]:
    """
    IID partition: randomly shuffle all indices, split as evenly as possible across clients.
    Returns dict {client_id: [sample_indices]}
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(idx)

    splits: Dict[int, List[int]] = {c: [] for c in range(num_clients)}
    # near-equal split
    per = num_samples // num_clients
    rem = num_samples % num_clients
    start = 0
    for c in range(num_clients):
        size = per + (1 if c < rem else 0)
        splits[c] = idx[start:start + size].tolist()
        start += size
    return splits


def build_client_schedule(num_clients: int, clients_per_round: int, rounds: int, seed: int) -> List[List[int]]:
    rng = np.random.RandomState(seed)
    return [rng.choice(num_clients, size=clients_per_round, replace=False).tolist() for _ in range(rounds)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, required=True, choices=["fedavg", "fedevo"])
    p.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"])
    p.add_argument("--rounds", type=int, default=200)
    p.add_argument("--out_dir", type=str, default="./results")
    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--seed_train", type=int, default=44)

    # FedEvo knobs
    p.add_argument("--m", type=int, default=5, help="Population size (candidates).")
    p.add_argument("--enable_mutation", action="store_true", help="Enable mutation (FedEvo only).")
    p.add_argument("--rho", type=float, default=0.4, help="Top-k ratio for exploitation.")
    p.add_argument("--gamma", type=float, default=1.5, help="Selection weight exponent.")
    p.add_argument("--sigma_mut", type=float, default=0.01, help="Mutation scale (fraction of layer std).")

    p.add_argument("--data_dir", type=str, default="./data")
    args = p.parse_args()

    if args.out_csv and args.out_csv.strip():
        out_csv_path = args.out_csv
    else:
        out_csv_path = build_out_csv_path(args.out_dir, args.algo, args.dataset)

    # Locked experimental setup (match your existing main.py style)
    N = 100
    K = 10
    E = 5
    batch_size = 50
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    gamma_lr = 0.998

    seed_data = 42
    seed_sample = 43
    seed_train = args.seed_train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed_train)

    bundle = load_cifar(args.dataset, data_dir=args.data_dir)

    # --- IID partition here ---
    num_train = len(bundle.train_dataset)
    client_indices = iid_partition(num_samples=num_train, num_clients=N, seed=seed_data)

    # keep same local train/val split helper
    train_idx, val_idx = client_train_val_split(client_indices, val_ratio=0.1, seed_data=seed_data)

    schedule = build_client_schedule(N, K, args.rounds, seed=seed_sample)

    client_train_loaders: Dict[int, torch.utils.data.DataLoader] = {}
    client_val_loaders: Dict[int, torch.utils.data.DataLoader] = {}
    for cid in range(N):
        ds_train = make_subset(bundle.train_dataset, train_idx[cid])
        ds_val = make_subset(bundle.train_dataset, val_idx[cid])

        client_train_loaders[cid] = make_loader(ds_train, batch_size=batch_size, shuffle=True, seed_train=seed_train + cid, device=device)
        client_val_loaders[cid] = make_loader(ds_val, batch_size=batch_size, shuffle=False, seed_train=seed_train + cid, device=device)

    test_loader = make_loader(bundle.test_dataset, batch_size=200, shuffle=False, seed_train=seed_train, device=device)

    # Build runner
    if args.algo == "fedavg":
        ga_cfg = GAConfig(m=1, enable_mutation=False)
    else:
        ga_cfg = GAConfig(
            m=int(args.m),
            rho=float(args.rho),
            gamma=float(args.gamma),
            sigma_mut=float(args.sigma_mut),
            enable_mutation=bool(args.enable_mutation),
        )

    runner = FedEvoRunner(
        model_ctor=ResNet18_CIFAR,
        num_classes=bundle.num_classes,
        device=device,
        ga=ga_cfg,
        seed=2025,
        val_batches=None,
        weight_by_samples=True,
        deterministic=False,
    )

    lr_current = lr

    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "test_accuracy", "train_loss", "uplink_bytes", "usage_counts"])
        writer.writeheader()

        for r in range(args.rounds):
            client_ids = schedule[r]

            train_loss, uplink = runner.run_round(
                client_ids=client_ids,
                client_train_loaders=client_train_loaders,
                client_val_loaders=client_val_loaders,
                epochs=E,
                sgd_cfg=(lr_current, momentum, weight_decay),
                seed_train=seed_train,
            )

            model_for_eval = runner.get_best_model()
            _, acc = evaluate(model_for_eval, test_loader, device=device)

            writer.writerow(
                {
                    "round": r,
                    "test_accuracy": acc,
                    "train_loss": train_loss,
                    "uplink_bytes": uplink,
                    "usage_counts": str(runner.last_usage_counts),
                }
            )
            f.flush()

            lr_current *= gamma_lr


if __name__ == "__main__":
    main()
