from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class CIFARDataBundle:
    train_dataset: Dataset
    train_plain_dataset: Dataset
    test_dataset: Dataset
    num_classes: int


def load_cifar(dataset: str, data_dir: str = "./data") -> CIFARDataBundle:
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
    plain_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_dataset = ds_cls(root=data_dir, train=True, download=True, transform=train_tf)
    train_plain_dataset = ds_cls(root=data_dir, train=True, download=False, transform=plain_tf)
    test_dataset = ds_cls(root=data_dir, train=False, download=True, transform=plain_tf)

    return CIFARDataBundle(
        train_dataset=train_dataset,
        train_plain_dataset=train_plain_dataset,
        test_dataset=test_dataset,
        num_classes=num_classes,
    )


def dirichlet_partition(
    targets: List[int],
    num_clients: int,
    alpha: float,
    seed_data: int,
) -> Dict[int, List[int]]:
    if alpha <= 0:
        raise ValueError("alpha must be > 0")

    rng = np.random.default_rng(int(seed_data))

    targets = np.asarray(targets, dtype=np.int64)
    num_classes = int(targets.max() + 1)

    class_indices = [np.where(targets == c)[0] for c in range(num_classes)]
    for c in range(num_classes):
        rng.shuffle(class_indices[c])

    for _ in range(1000):
        client_indices: List[List[int]] = [[] for _ in range(int(num_clients))]

        for c in range(num_classes):
            idx_c = class_indices[c]
            if len(idx_c) == 0:
                continue

            proportions = rng.dirichlet(alpha=np.full(int(num_clients), float(alpha)))
            counts = (proportions * len(idx_c)).astype(int)

            diff = len(idx_c) - int(counts.sum())
            if diff > 0:
                for k in np.argsort(-proportions)[:diff]:
                    counts[k] += 1
            elif diff < 0:
                for k in np.argsort(-counts)[: (-diff)]:
                    if counts[k] > 0:
                        counts[k] -= 1

            if int(counts.sum()) != int(len(idx_c)):
                raise RuntimeError("Dirichlet partition internal error: counts mismatch")

            start = 0
            for client_id in range(int(num_clients)):
                cnt = int(counts[client_id])
                if cnt > 0:
                    client_indices[client_id].extend(idx_c[start : start + cnt].tolist())
                start += cnt

        for cid in range(int(num_clients)):
            rng.shuffle(client_indices[cid])

        if all(len(v) > 0 for v in client_indices):
            return {cid: client_indices[cid] for cid in range(int(num_clients))}

    raise RuntimeError("Failed to generate a Dirichlet partition with no empty clients after many attempts.")


def iid_partition(
    targets: List[int],
    num_clients: int,
    seed_data: int,
) -> Dict[int, List[int]]:
    rng = np.random.default_rng(int(seed_data))

    num_samples = len(targets)
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    chunk_size = num_samples // int(num_clients)
    client_indices: Dict[int, List[int]] = {}

    for cid in range(int(num_clients)):
        start = cid * chunk_size
        end = num_samples if cid == int(num_clients) - 1 else start + chunk_size
        client_indices[cid] = indices[start:end].tolist()

    if any(len(v) == 0 for v in client_indices.values()):
        raise RuntimeError("IID partition resulted in empty clients")

    return client_indices


def client_train_val_split(
    client_indices: Dict[int, List[int]],
    val_ratio: float,
    seed_data: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    if not (0.0 < float(val_ratio) < 1.0):
        raise ValueError("val_ratio must be in (0,1)")

    train_split: Dict[int, List[int]] = {}
    val_split: Dict[int, List[int]] = {}

    for cid, idxs in client_indices.items():
        idxs = list(idxs)
        rng = np.random.default_rng(int(seed_data) + 1000 + int(cid))
        rng.shuffle(idxs)

        n_val = max(1, int(round(len(idxs) * float(val_ratio))))
        val_split[int(cid)] = idxs[:n_val]
        train_split[int(cid)] = idxs[n_val:]

        if len(train_split[int(cid)]) == 0:
            train_split[int(cid)] = val_split[int(cid)][:1]
            val_split[int(cid)] = val_split[int(cid)][1:]
            if len(val_split[int(cid)]) == 0:
                val_split[int(cid)] = train_split[int(cid)][:1]
                train_split[int(cid)] = train_split[int(cid)][1:]

    return train_split, val_split


def make_subset(dataset: Dataset, indices: List[int]) -> Subset:
    return Subset(dataset, indices)
