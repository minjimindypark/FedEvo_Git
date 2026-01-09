from __future__ import annotations

import copy
import math
import os
import random
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Hard determinism (may error if nondeterministic ops are used).
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def param_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Parameters only (no buffers). This is the core state we transmit + aggregate.
    """
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def load_param_state_dict_(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """
    In-place load of parameter-only state.
    """
    with torch.no_grad():
        for k, p in model.named_parameters():
            if k not in state:
                raise KeyError(f"Missing key in param state: {k}")
            p.copy_(state[k])


def zero_like(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in state.items()}


def sub_state(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: (a[k] - b[k]) for k in a.keys()}


def add_state_(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
    with torch.no_grad():
        for k in a.keys():
            a[k].add_(b[k], alpha=scale)


def scale_state_(a: Dict[str, torch.Tensor], scale: float) -> None:
    with torch.no_grad():
        for k in a.keys():
            a[k].mul_(scale)


def fedavg_aggregate(server_params: Dict[str, torch.Tensor], deltas: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Uniform FedAvg over parameter deltas (client_params - server_params).
    """
    if len(deltas) == 0:
        return server_params

    avg_delta = zero_like(server_params)
    for d in deltas:
        add_state_(avg_delta, d, scale=1.0)
    scale_state_(avg_delta, 1.0 / len(deltas))

    new_params = {k: server_params[k] + avg_delta[k] for k in server_params.keys()}
    return new_params


def uplink_bytes_for_delta(delta: Dict[str, torch.Tensor]) -> int:
    """
    Bytes transmitted client->server if sending full-precision delta tensors.
    """
    total = 0
    for t in delta.values():
        total += t.numel() * t.element_size()
    return int(total)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    """
    Returns (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed_train: int,
    device: torch.device,
) -> DataLoader:
    """
    Deterministic DataLoader:
    - single worker (num_workers=0) to avoid worker seeding complexity
    - uses a per-loader torch.Generator for deterministic shuffles
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed_train)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=g if shuffle else None,
    )


def local_train_sgd(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    seed_train: int,
) -> float:
    """
    Minimal local SGD training. Returns average train loss over all samples.
    """
    # Per-client determinism: re-seed torch before local training loop.
    torch.manual_seed(seed_train)

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    total_loss = 0.0
    total = 0

    for _ in range(epochs):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * int(y.numel())
            total += int(y.numel())

    return total_loss / max(1, total)

# ============================================================
# Runner base classes (public API for algorithms package)
# ============================================================

class BaseRunner:
    """
    Public base runner interface exported by algorithms package.
    All algorithm runners should implement:
      run_round(...) -> (train_loss: float, uplink_bytes: int)
    """
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def run_round(self, *args, **kwargs):
        raise NotImplementedError


class FedAvgRunner(BaseRunner):
    """
    Simple FedAvg baseline runner.
    Uses:
      - param_state_dict / load_param_state_dict_
      - local_train_sgd (defined in this file)
      - uplink_bytes_for_delta
      - fedavg_aggregate
    """

    def __init__(self, model: nn.Module, device: torch.device) -> None:
        super().__init__(device=device)
        self.model = model.to(device)

    def run_round(
        self,
        client_ids: Sequence[int],
        client_train_loaders: Dict[int, DataLoader],
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
    ) -> Tuple[float, int]:
        lr, momentum, weight_decay = sgd_cfg

        server_params = param_state_dict(self.model)
        deltas: List[Dict[str, torch.Tensor]] = []
        uplink = 0
        losses: List[float] = []

        for cid in client_ids:
            cid_int = int(cid)

            local = copy.deepcopy(self.model)
            load_param_state_dict_(local, server_params)

            loss = local_train_sgd(
                model=local,
                loader=client_train_loaders[cid_int],
                device=self.device,
                epochs=int(epochs),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                seed_train=int(seed_train) + cid_int,
            )
            losses.append(loss)

            client_params = param_state_dict(local)
            delta = {k: (client_params[k] - server_params[k]) for k in server_params.keys()}
            deltas.append(delta)
            uplink += uplink_bytes_for_delta(delta)

        if deltas:
            new_params = fedavg_aggregate(server_params, deltas)
            load_param_state_dict_(self.model, new_params)

        return float(np.mean(losses)) if losses else 0.0, int(uplink)
