from __future__ import annotations

import copy
import os
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def _is_float_tensor(t: torch.Tensor) -> bool:
    return isinstance(t, torch.Tensor) and t.is_floating_point()


def param_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Floating-point state = parameters + floating buffers.
    Includes BatchNorm running_mean/var; excludes integer buffers (e.g., num_batches_tracked).
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        if _is_float_tensor(v):
            out[k] = v.detach().clone()
    return out



def params_only_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Trainable-parameter-only state dict (excludes all buffers).

    This matches the common theoretical notation θ ∈ R^P, where P counts only trainable parameters.
    Keys follow model.named_parameters(), which align with state_dict() keys for parameters.
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.named_parameters():
        out[k] = v.detach().clone()
    return out


def load_params_only_state_dict_(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """In-place load for parameter-only state dict."""
    with torch.no_grad():
        missing = [k for k, _ in model.named_parameters() if k not in state]
        if missing:
            raise KeyError(f"Missing param keys in state: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        for k, v in model.named_parameters():
            v.copy_(state[k])


def get_state_dict(model: nn.Module, mode: str = "float") -> Dict[str, torch.Tensor]:
    """Return a state dict under the requested mode.

    mode:
      - "float": parameters + floating buffers (BatchNorm running stats included)
      - "params": trainable parameters only (buffers excluded)
    """
    mode = str(mode).lower().strip()
    if mode == "float":
        return param_state_dict(model)
    if mode in ("params", "param", "parameters"):
        return params_only_state_dict(model)
    raise ValueError(f"Unknown state mode: {mode}")


def load_state_dict_(model: nn.Module, state: Dict[str, torch.Tensor], mode: str = "float") -> None:
    """Load a state dict produced by get_state_dict()."""
    mode = str(mode).lower().strip()
    if mode == "float":
        load_param_state_dict_(model, state)
        return
    if mode in ("params", "param", "parameters"):
        load_params_only_state_dict_(model, state)
        return
    raise ValueError(f"Unknown state mode: {mode}")


def load_param_state_dict_(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    """In-place load of floating-point state (parameters + floating buffers)."""
    with torch.no_grad():
        model_sd = model.state_dict()
        float_keys = [k for k, v in model_sd.items() if _is_float_tensor(v)]
        missing = [k for k in float_keys if k not in state]
        if missing:
            raise KeyError(f"Missing float keys in state: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        for k in float_keys:
            model_sd[k].copy_(state[k])


def zero_like(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in state.items()}


def add_state_(a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
    with torch.no_grad():
        for k in a.keys():
            a[k].add_(b[k], alpha=float(scale))


def scale_state_(a: Dict[str, torch.Tensor], scale: float) -> None:
    with torch.no_grad():
        for k in a.keys():
            a[k].mul_(float(scale))


def fedavg_aggregate(server_state: Dict[str, torch.Tensor], deltas: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if len(deltas) == 0:
        return server_state

    avg_delta = zero_like(server_state)
    for d in deltas:
        add_state_(avg_delta, d, scale=1.0)
    scale_state_(avg_delta, 1.0 / len(deltas))

    return {k: server_state[k] + avg_delta[k] for k in server_state.keys()}


def uplink_bytes_for_delta(delta: Dict[str, torch.Tensor]) -> int:
    total = 0
    for t in delta.values():
        total += t.numel() * t.element_size()
    return int(total)


@torch.no_grad()
def bn_recalibrate(model: nn.Module, loader: DataLoader, device: torch.device, num_batches: int = 20) -> None:
    """
    Recompute BatchNorm running stats by running a few batches in train() mode
    without gradient updates.
    """
    was_training = bool(model.training)
    model.train(True)

    n = 0
    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        _ = model(x)
        n += 1
        if n >= int(num_batches):
            break

    model.train(was_training)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
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
    num_workers: int = 0,
) -> DataLoader:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed_train))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
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
    torch.manual_seed(int(seed_train))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed_train))

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=float(lr), momentum=float(momentum), weight_decay=float(weight_decay))

    total_loss = 0.0
    total = 0

    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            bs = int(y.numel())
            total_loss += float(loss.item()) * bs
            total += bs

    return total_loss / max(1, total)


class BaseRunner:
    def __init__(self, device: torch.device) -> None:
        self.device = device

    def run_round(self, *args, **kwargs):
        raise NotImplementedError


class FedAvgRunner(BaseRunner):
    def __init__(self, model: nn.Module, device: torch.device, state_mode: str = "float") -> None:
        super().__init__(device=device)
        self.model = model.to(device)
        self.state_mode = str(state_mode).lower().strip()

    def run_round(
        self,
        client_ids: Sequence[int],
        client_train_loaders: Dict[int, DataLoader],
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
        weight_by_samples: bool = True,
    ) -> Tuple[float, int]:
        lr, momentum, weight_decay = sgd_cfg

        # server state under requested mode
        server_state = get_state_dict(self.model, mode=self.state_mode)

        # accumulate weighted delta
        deltas: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []
        uplink = 0
        losses: List[float] = []

        for cid in client_ids:
            cid_int = int(cid)

            # local model = copy of server
            local = copy.deepcopy(self.model)
            load_state_dict_(local, server_state, mode=self.state_mode)

            # train
            loss = local_train_sgd(
                model=local,
                loader=client_train_loaders[cid_int],
                epochs=int(epochs),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                seed_train=int(seed_train) + cid_int,
                device=self.device,
            )
            losses.append(float(loss))

            # delta
            client_state = get_state_dict(local, mode=self.state_mode)
            delta = {k: (client_state[k] - server_state[k]) for k in server_state.keys()}
            deltas.append(delta)

            # weights (optional)
            n_samples = len(client_train_loaders[cid_int].dataset)
            weights.append(float(n_samples) if weight_by_samples else 1.0)

            # uplink bytes
            uplink += uplink_bytes_for_delta(delta)

        # aggregate
        if deltas:
            # weighted average delta
            wsum = float(sum(weights))
            avg_delta = zero_like(server_state)
            for d, w in zip(deltas, weights):
                add_state_(avg_delta, d, scale=(w / max(wsum, 1e-12)))
            new_state = {k: server_state[k] + avg_delta[k] for k in server_state.keys()}
            load_state_dict_(self.model, new_state, mode=self.state_mode)

        return float(np.mean(losses)) if losses else 0.0, int(uplink)

