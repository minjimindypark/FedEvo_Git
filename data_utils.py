from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class CIFARDataBundle:
    train_dataset: Dataset
    test_dataset: Dataset
    num_classes: int


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

    - 전역 np.random.seed()를 건드리지 않음
    - seed_data로 만든 로컬 RNG(rng)만 사용
    """
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    rng = np.random.default_rng(seed_data)  # ✅ 로컬 RNG

    targets = np.asarray(targets, dtype=np.int64)
    num_classes = int(targets.max() + 1)

    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])  # ✅ 전역 shuffle -> rng.shuffle

    # We retry until no empty clients.
    for _attempt in range(1000):
        client_indices: List[List[int]] = [[] for _ in range(num_clients)]

        for c in range(num_classes):
            idx_c = class_indices[c]
            if len(idx_c) == 0:
                continue

            proportions = rng.dirichlet(alpha=np.full(num_clients, alpha))  # ✅ 전역 -> rng
            counts = (proportions * len(idx_c)).astype(int)

            diff = len(idx_c) - counts.sum()
            if diff > 0:
                for k in np.argsort(-proportions)[:diff]:
                    counts[k] += 1
            elif diff < 0:
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

        for cid in range(num_clients):
            rng.shuffle(client_indices[cid])  # ✅ 전역 shuffle -> rng.shuffle

        if all(len(v) > 0 for v in client_indices):
            return {cid: client_indices[cid] for cid in range(num_clients)}

    raise RuntimeError(
        "Failed to generate a Dirichlet partition with no empty clients after many attempts."
    )


def client_train_val_split(
    client_indices: Dict[int, List[int]],
    val_ratio: float,
    seed_data: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Split each client's indices into train/val deterministically,
    without touching global RNG state.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0,1)")

    train_split: Dict[int, List[int]] = {}
    val_split: Dict[int, List[int]] = {}

    for cid, idxs in client_indices.items():
        idxs = list(idxs)

        rng = np.random.default_rng(seed_data + 1000 + cid)  # ✅ 클라이언트별 로컬 RNG
        rng.shuffle(idxs)

        n_val = max(1, int(round(len(idxs) * val_ratio)))
        val_split[cid] = idxs[:n_val]
        train_split[cid] = idxs[n_val:]

        # 안전장치: train/val 비는 것 방지(원래 의도 유지)
        if len(train_split[cid]) == 0:
            train_split[cid] = val_split[cid][:1]
            val_split[cid] = val_split[cid][1:]
            if len(val_split[cid]) == 0:
                val_split[cid] = train_split[cid][:1]
                train_split[cid] = train_split[cid][1:]

    return train_split, val_split



def make_subset(dataset: Dataset, indices: List[int]) -> Subset:
    return Subset(dataset, indices)