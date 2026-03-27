# algorithms/fedprox.py
"""FedProx: Federated Optimization in Heterogeneous Networks (Li et al., ICLR 2020).
Adds a proximal term mu/2 * ||w - w_global||^2 to the local objective.
Aggregation is identical to FedAvg.
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    BaseRunner,
    get_state_dict,
    load_state_dict_,
    uplink_bytes_for_delta,
    zero_like,
    add_state_,
)


def local_train_fedprox(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    seed_train: int,
    global_params: Dict[str, torch.Tensor],  # named_parameters snapshot (CPU)
    mu: float,
) -> float:
    """Local SGD with proximal term. Returns average CE loss (proximal term excluded)."""
    torch.manual_seed(int(seed_train))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed_train))

    model.train()
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )

    # Move global params to device once
    g_params = {k: v.to(device) for k, v in global_params.items()}

    total_loss = 0.0
    total = 0

    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            ce_loss = F.cross_entropy(logits, y)

            # Proximal term: (mu/2) * ||w - w_global||^2
            prox = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                if name in g_params:
                    diff = param - g_params[name]
                    prox = prox + diff.pow(2).sum()
            prox = (float(mu) / 2.0) * prox

            (ce_loss + prox).backward()
            opt.step()

            bs = int(y.numel())
            total_loss += float(ce_loss.item()) * bs  # log CE only
            total += bs

    return total_loss / max(1, total)


class FedProxRunner(BaseRunner):
    """FedProx runner. Drop-in replacement for FedAvgRunner with proximal regularization."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        state_mode: str = "params",
        mu: float = 0.01,
    ) -> None:
        super().__init__(device=device)
        self.model = model.to(device)
        self.state_mode = str(state_mode).lower().strip()
        self.mu = float(mu)

    def run_round(
        self,
        client_ids: Sequence[int],
        client_train_loaders: Dict,
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
        weight_by_samples: bool = True,
    ) -> Tuple[float, int]:
        lr, momentum, weight_decay = sgd_cfg

        server_state = get_state_dict(self.model, mode=self.state_mode)

        # Global params for proximal term (named_parameters only, CPU)
        global_params = {
            name: param.detach().clone().cpu()
            for name, param in self.model.named_parameters()
        }

        deltas: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []
        uplink = 0
        losses: List[float] = []

        for cid in client_ids:
            cid_int = int(cid)

            local = copy.deepcopy(self.model)
            load_state_dict_(local, server_state, mode=self.state_mode)

            loss = local_train_fedprox(
                model=local,
                loader=client_train_loaders[cid_int],
                epochs=int(epochs),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                seed_train=int(seed_train) + cid_int,
                global_params=global_params,
                mu=self.mu,
                device=self.device,
            )
            losses.append(float(loss))

            client_state = get_state_dict(local, mode=self.state_mode)
            delta = {k: (client_state[k] - server_state[k]) for k in server_state}
            deltas.append(delta)

            n_samples = len(client_train_loaders[cid_int].dataset)
            weights.append(float(n_samples) if weight_by_samples else 1.0)
            uplink += uplink_bytes_for_delta(delta)

        if deltas:
            wsum = float(sum(weights))
            avg_delta = zero_like(server_state)
            for d, w in zip(deltas, weights):
                add_state_(avg_delta, d, scale=(w / max(wsum, 1e-12)))
            new_state = {k: server_state[k] + avg_delta[k] for k in server_state}
            load_state_dict_(self.model, new_state, mode=self.state_mode)

        return float(np.mean(losses)) if losses else 0.0, int(uplink)
