"""
models.py

Model definitions used in the FedEvo experiments.
- ResNet18_CIFAR: ResNet-18 variant adapted for CIFAR-10/100
- build_split_resnet18_cifar: helper to construct split models for FedImpro

Design note:
Models are intentionally kept minimal and deterministic-friendly
to isolate the effect of the learning algorithm (FedEvo) rather than architecture choices.
"""
"""
main.py

Entry point for federated learning experiments, including FedMut, FedImpro, and FedEvo.

This script reproduces all experimental results reported in the paper.

CLI design:
- --algo, --dataset: select algorithm and dataset
- --alpha: Dirichlet concentration parameter controlling Non-IID severity
- --rounds: number of federated communication rounds

FedEvo-specific knobs (Section X in the paper):
- --d: sentinel dimension per candidate (must satisfy low-sensitivity pool size)
- --nu_scale: sentinel magnitude scaling, controlling attribution separability
- --local_steps: number of local SGD steps; excessive values may weaken sentinel signals

Recommended quick smoke test:
python main.py --algo fedevo --dataset cifar10 --rounds 2 --d 512 --nu_scale 0.02 --local_steps 1
"""


import os
from datetime import datetime

import argparse
import csv
import copy
from typing import Dict, List, Sequence

import numpy as np
import torch

from algorithms.base import evaluate, make_loader, set_global_seed
from algorithms.fedimpro import FedImproRunner
from algorithms.fedmut import FedMutRunner
from algorithms.fedevo import FedEvoRunner
from data_utils import client_train_val_split, dirichlet_partition, load_cifar, make_subset
from models import ResNet18_CIFAR, build_split_resnet18_cifar

def build_out_csv_path(out_dir: str, algo: str, dataset: str) -> str:
    """
    결과 CSV 파일명을 자동 생성한다.
    예: ./results/fedevo_cifar10_20260112-104233.csv
    """
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{algo}_{dataset}_{ts}.csv"
    return os.path.join(out_dir, filename)


def build_client_schedule(num_clients: int, clients_per_round: int, rounds: int, seed_sample: int) -> List[List[int]]:
    rng = np.random.RandomState(seed_sample)
    schedule: List[List[int]] = []
    for _ in range(rounds):
        ids = rng.choice(num_clients, size=clients_per_round, replace=False).tolist()
        schedule.append(ids)
    return schedule


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", type=str, required=True, choices=["fedmut", "fedimpro", "fedevo"])
    p.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"])
    p.add_argument("--alpha", type=float, default=0.1, choices=[0.1, 0.5])
    p.add_argument("--rounds", type=int, default=1000)
    p.add_argument("--out_dir", type=str, default="./results", help="Directory to save results CSV")
    p.add_argument("--out_csv", type=str, default="", help="(Optional) Override output CSV path")

    # ---- FedEvo auto-tune knobs (only used when --algo=fedevo) ----
    p.add_argument("--d", type=int, default=1536, help="FedEvo sentinel dimension per candidate")
    p.add_argument("--nu_scale", type=float, default=0.01, help="FedEvo nu scale")
    p.add_argument("--local_steps", type=int, default=5, help="Local training steps/epochs (replaces E when fedEvo tuning)")
    p.add_argument("--low_sens_mode", type=str, default="bias_norm", choices=["bias_only", "bias_norm"])
    p.add_argument("--seed_train", type=int, default=44)
    # --------------------------------------------------------------

    p.add_argument("--data_dir", type=str, default="./data")
    args = p.parse_args()

    if args.out_csv and args.out_csv.strip():
        out_csv_path = args.out_csv
    else:
        out_csv_path = build_out_csv_path(args.out_dir, args.algo, args.dataset)

    os.makedirs("logs", exist_ok=True)

    # Locked setup
    N = 100
    K = 10
    E = 5
    batch_size = 50
    lr = 0.01
    momentum = 0.9
    weight_decay = 1e-4
    gamma = 0.998

    seed_data = 42
    seed_sample = 43
    seed_train = args.seed_train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed_train)

    bundle = load_cifar(args.dataset, data_dir=args.data_dir)

    targets = bundle.train_dataset.targets  # list[int] for CIFAR datasets
    client_indices = dirichlet_partition(targets, num_clients=N, alpha=float(args.alpha), seed_data=seed_data)
    train_idx, val_idx = client_train_val_split(client_indices, val_ratio=0.1, seed_data=seed_data)

    schedule = build_client_schedule(N, K, args.rounds, seed_sample=seed_sample)

    # Build per-client loaders (train + val).
    client_train_loaders: Dict[int, torch.utils.data.DataLoader] = {}
    client_val_loaders: Dict[int, torch.utils.data.DataLoader] = {}
    for cid in range(N):
        ds_train = make_subset(bundle.train_dataset, train_idx[cid])
        ds_val = make_subset(bundle.train_dataset, val_idx[cid])

        client_train_loaders[cid] = make_loader(ds_train, batch_size=batch_size, shuffle=True, seed_train=seed_train + cid, device=device)
        client_val_loaders[cid] = make_loader(ds_val, batch_size=batch_size, shuffle=False, seed_train=seed_train + cid, device=device)

    test_loader = make_loader(bundle.test_dataset, batch_size=200, shuffle=False, seed_train=seed_train, device=device)

    # Build algorithm runner
    if args.algo == "fedmut":
        model = ResNet18_CIFAR(num_classes=bundle.num_classes)
        runner = FedMutRunner(model=model, device=device, alpha_mut=4.0, seed_mut=9991)
        server_model_for_eval = runner.model
    elif args.algo == "fedimpro":
        base = ResNet18_CIFAR(num_classes=bundle.num_classes)
        split = build_split_resnet18_cifar(base)
        runner = FedImproRunner(split_model=split, num_classes=bundle.num_classes, device=device)
        server_model_for_eval = runner.model  # wrapper has forward
    else:
        runner = FedEvoRunner(
        model_ctor=ResNet18_CIFAR,
        num_classes=bundle.num_classes,
        device=device,
        d=args.d,
        m=5,
        seed_evo=2025,
        nu_scale=args.nu_scale,
        nu_min=1e-6,
        nu_max=1e-1,
        feedback_log_path="logs/implicit_feedback.csv",
        low_sens_mode=args.low_sens_mode,   # ← (fedevo.py가 이 인자 받으면 사용)
    )

        server_model_for_eval = runner.model


    # Server-side LR scheduler stepped per round.
    # For FedMut/FedImpro we can attach scheduler to a dummy optimizer on server model params.
    # We'll implement: apply scheduler scale to client lr by updating lr each round.
    lr_current = lr

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "test_accuracy", "train_loss", "uplink_bytes"])
        writer.writeheader()

        for r in range(args.rounds):
            client_ids = schedule[r]

            # Run one round
            if args.algo == "fedmut":
                train_loss, uplink = runner.run_round(
                    client_ids=client_ids,
                    client_train_loaders=client_train_loaders,
                    epochs=E,
                    sgd_cfg=(lr_current, momentum, weight_decay),
                    seed_train=seed_train,
                )
            elif args.algo == "fedimpro":
                train_loss, uplink = runner.run_round(
                    client_ids=client_ids,
                    client_train_loaders=client_train_loaders,
                    epochs=E,
                    batch_size=batch_size,
                    sgd_cfg=(lr_current, momentum, weight_decay),
                    seed_train=seed_train,
                )
            else:
                train_loss, uplink = runner.run_round(
                    client_ids=client_ids,
                    client_train_loaders=client_train_loaders,
                    client_val_loaders=client_val_loaders,
                    epochs=args.local_steps,
                    sgd_cfg=(lr_current, momentum, weight_decay),
                    seed_train=seed_train,
                )

            # Evaluate global model
            if args.algo == "fedevo":
                model_for_eval = runner.get_best_model()  # loads runner.theta_base into runner.model
                _, acc = evaluate(model_for_eval, test_loader, device=device)
            else:
                _, acc = evaluate(server_model_for_eval, test_loader, device=device)


            writer.writerow(
                {
                    "round": r,
                    "test_accuracy": acc,
                    "train_loss": train_loss,
                    "uplink_bytes": uplink,
                }
            )
            f.flush()

            # Step scheduler once per round (ExponentialLR with gamma=0.998)
            lr_current = lr_current * gamma


if __name__ == "__main__":
    main()
