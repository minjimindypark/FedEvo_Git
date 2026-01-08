import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class CIFARDataBundle:
    train_dataset: Dataset
    test_dataset: Dataset
    num_classes: int


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_cifar(dataset: str, data_dir: str = "./data") -> CIFARDataBundle:
    """
    Loads CIFAR-10/100 with standard normalization and standard splits.
    """
    dataset = dataset.lower()
    if dataset not in {"cifar10", "cifar100"}:
        raise ValueError("dataset must be one of: cifar10, cifar100")

    if dataset == "cifar10":
        num_classes = 10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        ds_cls = datasets.CIFAR10
    else:
        num_classes = 100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        ds_cls = datasets.CIFAR100

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = ds_cls(root=data_dir, train=True, download=True, transform=train_tf)
    test_dataset = ds_cls(root=data_dir, train=False, download=True, transform=test_tf)

    return CIFARDataBundle(train_dataset=train_dataset, test_dataset=test_dataset, num_classes=num_classes)


def dirichlet_partition(
    targets: List[int],
    num_clients: int,
    alpha: float,
    seed_data: int,
) -> Dict[int, List[int]]:
    """
    Deterministic Dirichlet label-skew partition.

    Returns:
        dict client_id -> list of sample indices
    Requirements:
        - no empty clients
        - deterministic given seed_data
        - partitions reusable across algorithms
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    _set_seed(seed_data)

    targets = np.asarray(targets, dtype=np.int64)
    num_classes = int(targets.max() + 1)

    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        np.random.shuffle(class_indices[c])

    # We retry until no empty clients.
    for _attempt in range(1000):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idx_c = class_indices[c]
            if len(idx_c) == 0:
                continue

            proportions = np.random.dirichlet(alpha=np.full(num_clients, alpha))
            # Convert proportions to counts.
            counts = (proportions * len(idx_c)).astype(int)

            # Fix rounding so total matches.
            diff = len(idx_c) - counts.sum()
            if diff > 0:
                # add remaining samples to largest proportions
                for k in np.argsort(-proportions)[:diff]:
                    counts[k] += 1
            elif diff < 0:
                # remove extras from largest counts
                for k in np.argsort(-counts)[: (-diff)]:
                    if counts[k] > 0:
                        counts[k] -= 1

            assert counts.sum() == len(idx_c)

            start = 0
            for client_id in range(num_clients):
                cnt = int(counts[client_id])
                if cnt > 0:
                    client_indices[client_id].extend(idx_c[start : start + cnt].tolist())
                start += cnt

        # Shuffle each client's index list deterministically (same RNG state).
        for cid in range(num_clients):
            np.random.shuffle(client_indices[cid])

        if all(len(v) > 0 for v in client_indices):
            return {cid: client_indices[cid] for cid in range(num_clients)}

    raise RuntimeError("Failed to generate a Dirichlet partition with no empty clients after many attempts.")


def client_train_val_split(
    client_indices: Dict[int, List[int]],
    val_ratio: float,
    seed_data: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Deterministic per-client train/val split (for FedEvo selection and FedImpro stats).

    Returns:
        train_indices, val_indices : dict client_id -> list of indices
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0,1)")
    _set_seed(seed_data + 12345)

    train_split: Dict[int, List[int]] = {}
    val_split: Dict[int, List[int]] = {}

    for cid, idxs in client_indices.items():
        idxs = list(idxs)
        rng = np.random.RandomState(seed_data + 1000 + cid)
        rng.shuffle(idxs)

        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_split[cid] = idxs[:n_val]
        train_split[cid] = idxs[n_val:]
        if len(train_split[cid]) == 0:
            # Guarantee non-empty train too.
            train_split[cid] = val_split[cid][:1]
            val_split[cid] = val_split[cid][1:]
            if len(val_split[cid]) == 0:
                # Degenerate tiny client: just put one sample in val, rest in train
                # (should not happen much with CIFAR + N=100)
                val_split[cid] = train_split[cid][:1]
                train_split[cid] = train_split[cid][1:]

    return train_split, val_split


def make_subset(dataset: Dataset, indices: List[int]) -> Subset:
    return Subset(dataset, indices)