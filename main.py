import os
import csv
import argparse
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

from algorithms.base import evaluate, make_loader, set_global_seed, bn_recalibrate
from algorithms.fedevo import FedEvoRunner, GAConfig, FedEvoClient
from data_utils import client_train_val_split, dirichlet_partition, iid_partition, load_cifar, make_subset
from models import ResNet18_CIFAR


def build_client_schedule(num_clients: int, clients_per_round: int, rounds: int, seed: int) -> List[List[int]]:
    rng = np.random.RandomState(int(seed))
    schedule: List[List[int]] = []
    for _ in range(int(rounds)):
        sampled = rng.choice(int(num_clients), size=int(clients_per_round), replace=False).tolist()
        schedule.append(sampled)
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description="FedEvo: Federated Genetic Optimization", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, required=True, choices=["cifar10", "cifar100"], help="Dataset to use")
    parser.add_argument("--alpha", type=str, default="0.1", help="Data partition: 'iid' or Dirichlet 慣 (0.1, 0.5)")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data directory")

    parser.add_argument("--num_clients", type=int, default=100, help="Total number of clients")
    parser.add_argument("--clients_per_round", type=int, default=10, help="Number of clients sampled per round")
    parser.add_argument("--rounds", type=int, default=200, help="Number of communication rounds")
    parser.add_argument("--epochs", type=int, default=5, help="Local training epochs per round")

    parser.add_argument("--batch_size", type=int, default=50, help="Client batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr_decay", type=float, default=0.998, help="Learning rate decay per round")

    parser.add_argument("--rho", type=float, default=0.3, help="Top-k selection ratio")
    parser.add_argument("--gamma", type=float, default=1.5, help="Top-k weight exponent")
    parser.add_argument("--tau_factor", type=float, default=0.8, help="Entropy threshold factor")
    parser.add_argument("--sigma_mut", type=float, default=0.01, help="Mutation magnitude")
    parser.add_argument("--num_interp", type=int, default=4, help="Number of interpolation candidates")
    parser.add_argument("--num_orth", type=int, default=1, help="Number of orthogonality injections")

    parser.add_argument("--no_mutation", action="store_true", help="Disable mutation")
    parser.add_argument("--no_orth", action="store_true", help="Disable orthogonality injection")

    parser.add_argument("--val_batches", type=int, default=None, help="Limit validation batches")
    parser.add_argument("--no_weight_by_samples", action="store_true", help="Disable sample-weighted aggregation")

    parser.add_argument("--seed_data", type=int, default=42, help="Seed for data partitioning")
    parser.add_argument("--seed_sample", type=int, default=43, help="Seed for client sampling")
    parser.add_argument("--seed_train", type=int, default=44, help="Seed for training")
    parser.add_argument("--seed_evo", type=int, default=2025, help="Seed for genetic algorithm")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic mode")

    parser.add_argument("--out_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--out_csv", type=str, default="", help="Output CSV path")

    parser.add_argument("--m", type=int, default=10, help="Population size")

    parser.add_argument("--state_mode", type=str, default="params", choices=["params", "float"],
                        help="State representation: params=parameters only (recommended), float=parameters+floating buffers (BN stats).")
    parser.add_argument("--deploy_model", type=str, default="topk", choices=["topk", "usage", "stab"],
                        help="Which model to evaluate/deploy each round.")
    parser.add_argument("--bn_recalibrate_batches", type=int, default=0,
                        help="If >0, run BN recalibration for this many batches on train_plain before evaluation (helps when state_mode=params).")

    # aliases (optional)
    parser.add_argument("--local_steps", type=int, default=None, help="Alias of --epochs")
    parser.add_argument("--topk_rho", type=float, default=None, help="Alias of --rho")
    parser.add_argument("--topk_gamma", type=float, default=None, help="Alias of --gamma")
    parser.add_argument("--orth_warmup_rounds", type=int, default=0, help="(Not used) accepted for compatibility")
    parser.add_argument("--mut_warmup_rounds", type=int, default=0, help="(Not used) accepted for compatibility")
    parser.add_argument("--seed_group_size", type=int, default=0, help="(Not used) accepted for compatibility")
    parser.add_argument("--algo", type=str, default="fedevo", help="(Not used) accepted for compatibility")

    args = parser.parse_args()

    if args.local_steps is not None:
        args.epochs = args.local_steps
    if args.topk_rho is not None:
        args.rho = args.topk_rho
    if args.topk_gamma is not None:
        args.gamma = args.topk_gamma

    print(f"[FedEvo] Population size m={args.m}, rho={args.rho}, gamma={args.gamma}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed_train, deterministic=bool(args.deterministic))

    if args.out_csv and args.out_csv.strip():
        out_csv_path = args.out_csv
    else:
        os.makedirs(args.out_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alpha_str = args.alpha.replace(".", "p")
        out_csv_path = os.path.join(args.out_dir, f"fedevo_{args.dataset}_alpha{alpha_str}_r{args.rounds}_{timestamp}.csv")

    print(f"[Setup] Device: {device}")
    print(f"[Setup] Output: {out_csv_path}")

    bundle = load_cifar(args.dataset, data_dir=args.data_dir)
    targets = bundle.train_dataset.targets

    if args.alpha.lower() == "iid":
        print(f"[Data] Partitioning: IID across {args.num_clients} clients")
        client_indices = iid_partition(targets, num_clients=args.num_clients, seed_data=args.seed_data)
    else:
        alpha_val = float(args.alpha)
        print(f"[Data] Partitioning: Dirichlet(alpha={alpha_val}) across {args.num_clients} clients")
        client_indices = dirichlet_partition(targets, num_clients=args.num_clients, alpha=alpha_val, seed_data=args.seed_data)

    train_idx, val_idx = client_train_val_split(client_indices, val_ratio=0.1, seed_data=args.seed_data)

    client_train_loaders: Dict[int, torch.utils.data.DataLoader] = {}
    client_val_loaders: Dict[int, torch.utils.data.DataLoader] = {}

    for cid in range(args.num_clients):
        ds_train = make_subset(bundle.train_dataset, train_idx[cid])
        ds_val = make_subset(bundle.train_plain_dataset, val_idx[cid])

        client_train_loaders[cid] = make_loader(ds_train, batch_size=args.batch_size, shuffle=True, seed_train=args.seed_train + cid, device=device)
        client_val_loaders[cid] = make_loader(ds_val, batch_size=args.batch_size, shuffle=False, seed_train=args.seed_train + cid, device=device)

    test_loader = make_loader(bundle.test_dataset, batch_size=200, shuffle=False, seed_train=args.seed_train, device=device)

    print(f"[Data] Train: {len(bundle.train_dataset)}, Test: {len(bundle.test_dataset)}")

    ga_config = GAConfig(
        state_mode=str(args.state_mode),
        m=args.m,
        rho=args.rho,
        gamma=args.gamma,
        tau_factor=args.tau_factor,
        sigma_mut=args.sigma_mut,
        num_interp=args.num_interp,
        num_orth=args.num_orth,
        enable_mutation=not args.no_mutation,
        enable_orth_injection=not args.no_orth,
    )

    runner = FedEvoRunner(
        model_ctor=ResNet18_CIFAR,
        num_classes=bundle.num_classes,
        device=device,
        ga=ga_config,
        seed=args.seed_evo,
        val_batches=args.val_batches,
        weight_by_samples=not args.no_weight_by_samples,
        deterministic=bool(args.deterministic),
    )

    print(f"[FedEvo] Population size m={args.m}, rho={args.rho}, gamma={args.gamma}")
    print(f"[FedEvo] Mutation: {ga_config.enable_mutation}, Orthogonality: {ga_config.enable_orth_injection}")

    schedule = build_client_schedule(args.num_clients, args.clients_per_round, args.rounds, args.seed_sample)

    lr_current = float(args.lr)

    with open(out_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "test_accuracy", "train_loss", "uplink_bytes", "learning_rate"])
        writer.writeheader()

        for r in range(int(args.rounds)):
            client_ids = schedule[r]

            clients = [
                FedEvoClient(
                    cid=int(cid),
                    train_loader=client_train_loaders[int(cid)],
                    val_loader=client_val_loaders[int(cid)],
                )
                for cid in client_ids
            ]

            train_loss, uplink_bytes = runner.run_round(
                clients=clients,
                epochs=args.epochs,
                sgd_cfg=(lr_current, args.momentum, args.weight_decay),
                seed_train=args.seed_train,
            )

            deploy_model = runner.get_deploy_model(policy=str(args.deploy_model))

            if int(args.bn_recalibrate_batches) > 0:
                # Recalibrate BN running stats using a small number of batches from train_plain.
                bn_loader = make_loader(
                    bundle.train_plain_dataset,
                    batch_size=200,
                    shuffle=True,
                    seed_train=args.seed_train + 999,
                    device=device,
                )
                bn_recalibrate(deploy_model, bn_loader, num_batches=int(args.bn_recalibrate_batches), device=device)

            test_loss, test_acc = evaluate(deploy_model, test_loader, device=device)

            writer.writerow(
                {
                    "round": r,
                    "test_accuracy": f"{test_acc:.4f}",
                    "train_loss": f"{train_loss:.4f}",
                    "uplink_bytes": int(uplink_bytes),
                    "learning_rate": f"{lr_current:.6f}",
                }
            )
            f.flush()

            lr_current = lr_current * float(args.lr_decay)

            if (r + 1) % 10 == 0 or r == 0:
                print(f"Round {r+1:3d}/{args.rounds}: Test Acc={test_acc:.4f}, Train Loss={train_loss:.4f}, LR={lr_current:.6f}")


if __name__ == "__main__":
    main()
