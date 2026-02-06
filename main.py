import os
import csv
import argparse
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

from algorithms.base import evaluate, make_loader, set_global_seed, bn_recalibrate, FedAvgRunner
from algorithms.fedevo import FedEvoRunner, GAConfig, FedEvoClient
from data_utils import client_train_val_split, dirichlet_partition, iid_partition, load_cifar, make_subset
from models import ResNet18_CIFAR

import random

def _save_checkpoint(
    path: str,
    *,
    algo: str,
    args_dict: dict,
    round_idx: int,
    lr_current: float,
    schedule: list,
    runner,
) -> None:
    ckpt = {
        "algo": str(algo),
        "args": dict(args_dict),
        "meta": {
        "dataset": args_dict.get("dataset"),
        "model": "ResNet18_CIFAR",
        "state_mode": str(args_dict.get("state_mode")),
        "m": int(args_dict.get("m")) if args_dict.get("m") is not None else None,
        "num_clients": int(args_dict.get("num_clients")) if args_dict.get("num_clients") is not None else None,
        "clients_per_round": int(args_dict.get("clients_per_round")) if args_dict.get("clients_per_round") is not None else None,
        "num_classes": int(args_dict.get("num_classes")) if args_dict.get("num_classes") is not None else None,
        },
        "round_idx": int(round_idx),       # next round to run (0-based)
        "lr_current": float(lr_current),
        "schedule": schedule,
        # Global RNG states (for exact reproducibility)
        "py_random_state": random.getstate(),
        "np_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }

    if str(algo) == "fedevo":
        ckpt["fedevo"] = {
            "ga": runner.ga.__dict__,
            "state_mode": runner.state_mode,
            "num_classes": runner.num_classes,
            "round_idx_runner": runner.round_idx,
            "theta_base": runner.theta_base,
            "population": runner.population,
            "rng_state": runner.rng.get_state(),
            "runner._delta_cache_by_cid" : {},
            "runner._delta_cache_fifo" : [],
            "_delta_cache_max": runner._delta_cache_max,
        }
    else:
        # FedAvg
        ckpt["fedavg"] = {
            "state_mode": runner.state_mode,
            "model_state": runner.model.state_dict(),
        }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(ckpt, path)


def _load_checkpoint(path: str, *, device: torch.device):
    ckpt = torch.load(path, map_location=device)

    # Restore RNG
    random.setstate(ckpt["py_random_state"])
    np.random.set_state(ckpt["np_random_state"])
    torch.set_rng_state(ckpt["torch_rng_state"])
    if torch.cuda.is_available() and ckpt.get("torch_cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state_all"])

    return ckpt


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
    parser.add_argument("--no_mut", action="store_true", help="Alias of --no_mutation")

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

    parser.add_argument("--resume", type=str, default="", help="Path to a checkpoint .pt to resume from")
    parser.add_argument("--save_every", type=int, default=10, help="Save checkpoint every N rounds (0 disables)")
    parser.add_argument("--ckpt_dir", type=str, default="", help="Directory to save checkpoints (default: out_dir)")

    # aliases (optional)
    parser.add_argument("--local_steps", type=int, default=None, help="Alias of --epochs")
    parser.add_argument("--topk_rho", type=float, default=None, help="Alias of --rho")
    parser.add_argument("--topk_gamma", type=float, default=None, help="Alias of --gamma")
    parser.add_argument("--orth_warmup_rounds", type=int, default=0, help="accepted for compatibility")
    parser.add_argument("--mut_warmup_rounds", type=int, default=0, help="accepted for compatibility")
    parser.add_argument("--seed_group_size", type=int, default=0, help="accepted for compatibility")
    parser.add_argument("--algo", type=str, default="fedevo", choices=["fedevo", "fedavg"],
                        help="Which algorithm to run")


    args = parser.parse_args()
    args.no_mutation = bool(args.no_mutation) or bool(args.no_mut)
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
        out_csv_path = os.path.join(
            args.out_dir,
            f"{args.algo}_{args.dataset}_alpha{alpha_str}_r{args.rounds}_{timestamp}.csv"
        )

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

    # --- runner selection (FedEvo vs FedAvg) ---
    if args.algo == "fedevo":
        ga_config = GAConfig(
            state_mode=str(args.state_mode),
            m=args.m,
            seed_group_size=args.seed_group_size,
            rho=args.rho,
            gamma=args.gamma,
            tau_factor=args.tau_factor,
            sigma_mut=args.sigma_mut,
            num_interp=args.num_interp,
            num_orth=args.num_orth,
            enable_mutation=not args.no_mutation,
            enable_orth_injection=not args.no_orth,
            warmup_no_orth_rounds=args.orth_warmup_rounds,
            warmup_no_mut_rounds=args.mut_warmup_rounds,

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
        bn_recalib_loader = make_loader(
            bundle.train_plain_dataset,
            batch_size=200,
            shuffle=True,
            seed_train=args.seed_train + 777,
            device=device,
        )
        runner.set_bn_recalib_loader(bn_recalib_loader)
    # --- nice, unambiguous run header ---
    if args.algo == "fedevo":
        # ga_config는 FedEvo일 때만 존재하므로 여기에서만 사용
        print(
            "[Run] algo=fedevo "
            f"dataset={args.dataset} alpha={args.alpha} rounds={args.rounds} "
            f"m={args.m} rho={args.rho} gamma={args.gamma} tau_factor={args.tau_factor} "
            f"mutation={ga_config.enable_mutation} orth={ga_config.enable_orth_injection} "
            f"state_mode={args.state_mode} bn_recalib={args.bn_recalibrate_batches}"
        )
    else:
        model = ResNet18_CIFAR(int(bundle.num_classes)).to(device)
        runner = FedAvgRunner(model=model, device=device, state_mode=str(args.state_mode))
        print(
            "[Run] algo=fedavg "
            f"dataset={args.dataset} alpha={args.alpha} rounds={args.rounds} "
            f"clients_per_round={args.clients_per_round} epochs={args.epochs} "
            f"state_mode={args.state_mode} bn_recalib={args.bn_recalibrate_batches}"
        )

    # ===== Resume / schedule / CSV mode =====
    r_start = 0
    lr_current = float(args.lr)
    schedule = None

    ckpt = None
    if args.resume and args.resume.strip():
        ckpt = _load_checkpoint(args.resume, device=device)

    meta = (ckpt.get("meta", {}) if ckpt is not None else {}) or {}
    mismatches = []

    def _mm(key: str, got, exp):
        if got is None or exp is None:
            return
        if got != exp:
            mismatches.append(f"{key}: ckpt={got} vs args={exp}")

    _mm("dataset", meta.get("dataset"), args.dataset)
    _mm("state_mode", meta.get("state_mode"), str(args.state_mode))
    _mm("m", meta.get("m"), int(args.m))
    _mm("num_clients", meta.get("num_clients"), int(args.num_clients))
    _mm("clients_per_round", meta.get("clients_per_round"), int(args.clients_per_round))

    if mismatches:
        raise ValueError(
            "[Resume Error] Incompatible checkpoint for current run:\n  - "
            + "\n  - ".join(mismatches)
            + "\nFix: resume with matching arguments or start a new run (remove --resume)."
        )

    # ---- resume apply (only when ckpt exists) ----
    if ckpt is not None:
        r_start = int(ckpt["round_idx"])
        lr_current = float(ckpt["lr_current"])
        schedule = ckpt["schedule"]
        print(f"[Resume] Loaded checkpoint={args.resume} -> start_round={r_start}, lr={lr_current:.6f}")

    # Build schedule if not resumed
    if schedule is None:
        schedule = build_client_schedule(args.num_clients, args.clients_per_round, args.rounds, args.seed_sample)

    # Checkpoint path
    ckpt_dir = args.ckpt_dir.strip() or args.out_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    last_ckpt_path = os.path.join(ckpt_dir, "last.pt")

    # CSV open mode: new run -> write, resume -> append
    csv_mode = "a" if (args.resume and args.resume.strip()) else "w"
    need_header = (csv_mode == "w") or (not os.path.exists(out_csv_path)) or (os.path.getsize(out_csv_path) == 0)

    with open(out_csv_path, csv_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "test_accuracy", "train_loss", "uplink_bytes", "learning_rate"])
        if need_header:
            writer.writeheader()

        for r in range(int(r_start), int(args.rounds)):
            client_ids = schedule[r]

            if args.algo == "fedevo":
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
            else:
                train_loss, uplink_bytes = runner.run_round(
                    client_ids=client_ids,
                    client_train_loaders=client_train_loaders,
                    epochs=args.epochs,
                    sgd_cfg=(lr_current, args.momentum, args.weight_decay),
                    seed_train=args.seed_train,
                    weight_by_samples=not args.no_weight_by_samples,
                )
                deploy_model = runner.model

            if int(args.bn_recalibrate_batches) > 0:
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

            # Save checkpoint
            if int(args.save_every) > 0 and (((r + 1) % int(args.save_every) == 0) or ((r + 1) == int(args.rounds))):
                _save_checkpoint(
                    last_ckpt_path,
                    algo=args.algo,
                    args_dict=vars(args),
                    round_idx=r + 1,      # next round
                    lr_current=lr_current,
                    schedule=schedule,
                    runner=runner,
                )
                print(f"[CKPT] saved: {last_ckpt_path} (next_round={r+1})")

            lr_current = lr_current * float(args.lr_decay)

            if (r + 1) % 10 == 0 or r == 0:
                print(f"Round {r+1:3d}/{args.rounds}: Test Acc={test_acc:.4f}, Train Loss={train_loss:.4f}, LR={lr_current:.6f}")


if __name__ == "__main__":
    main()
