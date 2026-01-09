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
    p.add_argument("--out_csv", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="./data")
    args = p.parse_args()

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
    seed_train = 44

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
            d=512,
            m=5,
            seed_evo=2025,
            nu_scale=0.01,     # ← 추가/수정
            nu_min=1e-6,       # ← 추가
            nu_max=5e-3,       # ← 추가
            feedback_log_path="logs/implicit_feedback.csv",
        )
        server_model_for_eval = runner.model


    # Server-side LR scheduler stepped per round.
    # For FedMut/FedImpro we can attach scheduler to a dummy optimizer on server model params.
    # We'll implement: apply scheduler scale to client lr by updating lr each round.
    lr_current = lr

    with open(args.out_csv, "w", newline="") as f:
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
                    epochs=E,
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
