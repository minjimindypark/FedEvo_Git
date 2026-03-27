# algorithms/scaffold.py
"""SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
(Karimireddy et al., ICML 2020).

Key idea: each client maintains a control variate c_i to correct
for client drift. The server maintains a global control variate c.

Local update:  w_{t+1} = w_t - lr * (grad_f_i(w_t) + c - c_i)
Control update (Option I):
    c_i_new = c_i - c + (w_start - w_end) / (K * lr)
Server control: c_new = c + (1/N) * sum(c_i_new - c_i)

NOTE: SCAFFOLD theory assumes SGD without momentum. We use momentum=0
in local training to be faithful to the original paper.
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


def _param_zero_like(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Zero tensors matching named_parameters, stored on CPU."""
    return {name: torch.zeros_like(p.detach().cpu()) for name, p in model.named_parameters()}


def _param_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Snapshot of named_parameters, CPU, detached."""
    return {name: p.detach().clone().cpu() for name, p in model.named_parameters()}


def local_train_scaffold(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed_train: int,
    server_control: Dict[str, torch.Tensor],  # c  (CPU)
    client_control: Dict[str, torch.Tensor],  # c_i (CPU)
) -> Tuple[float, Dict[str, torch.Tensor]]:
    """
    Local SCAFFOLD training (no momentum, as per original paper).
    Returns (avg_loss, new_client_control).
    new_client_control is on CPU.
    """
    torch.manual_seed(int(seed_train))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed_train))

    model.train()
    # SCAFFOLD: no momentum to avoid interaction with control variate correction
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=0.0,
        weight_decay=float(weight_decay),
    )

    # Snapshot start weights (CPU)
    w_start = _param_state(model)

    # Move control variates to device
    c_server = {k: v.to(device) for k, v in server_control.items()}
    c_client = {k: v.to(device) for k, v in client_control.items()}

    total_loss = 0.0
    total = 0
    n_steps = 0

    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()

            # Apply control variate correction to gradient: grad += (c - c_i)
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        c = c_server.get(name)
                        ci = c_client.get(name)
                        if c is not None and ci is not None:
                            param.grad.add_(c - ci)

            opt.step()

            bs = int(y.numel())
            total_loss += float(loss.item()) * bs
            total += bs
            n_steps += 1

    # Update client control variate (Option I)
    # c_i_new = c_i - c + (w_start - w_end) / (K * lr)
    w_end = _param_state(model)
    K = max(1, n_steps)
    new_client_control: Dict[str, torch.Tensor] = {}
    for name in w_start:
        ci = client_control.get(name, torch.zeros_like(w_start[name]))
        c = server_control.get(name, torch.zeros_like(w_start[name]))
        delta_w = w_start[name] - w_end[name]
        new_client_control[name] = (ci - c + delta_w / (K * float(lr))).detach()

    return total_loss / max(1, total), new_client_control


class SCAFFOLDRunner(BaseRunner):
    """SCAFFOLD runner. Server stores per-client and global control variates."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_clients: int,
        state_mode: str = "params",
    ) -> None:
        super().__init__(device=device)
        self.model = model.to(device)
        self.state_mode = str(state_mode).lower().strip()
        self.num_clients = int(num_clients)

        # Initialize control variates as zeros (CPU)
        template = _param_zero_like(model)
        self.server_control: Dict[str, torch.Tensor] = template
        self.client_controls: Dict[int, Dict[str, torch.Tensor]] = {
            i: {k: v.clone() for k, v in template.items()}
            for i in range(self.num_clients)
        }

    def run_round(
        self,
        client_ids: Sequence[int],
        client_train_loaders: Dict,
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
        weight_by_samples: bool = True,
    ) -> Tuple[float, int]:
        lr, _momentum, weight_decay = sgd_cfg  # momentum unused (SCAFFOLD uses 0)

        server_state = get_state_dict(self.model, mode=self.state_mode)

        deltas: List[Dict[str, torch.Tensor]] = []
        control_deltas: List[Dict[str, torch.Tensor]] = []  # c_i_new - c_i
        weights: List[float] = []
        uplink = 0
        losses: List[float] = []

        for cid in client_ids:
            cid_int = int(cid)

            local = copy.deepcopy(self.model)
            load_state_dict_(local, server_state, mode=self.state_mode)

            loss, new_ci = local_train_scaffold(
                model=local,
                loader=client_train_loaders[cid_int],
                epochs=int(epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                seed_train=int(seed_train) + cid_int,
                server_control=self.server_control,
                client_control=self.client_controls[cid_int],
            )
            losses.append(float(loss))

            # Δc_i = c_i_new - c_i_old
            old_ci = self.client_controls[cid_int]
            dc = {k: new_ci[k] - old_ci[k] for k in new_ci}
            control_deltas.append(dc)

            # Update stored client control
            self.client_controls[cid_int] = new_ci

            # Model delta
            client_state = get_state_dict(local, mode=self.state_mode)
            delta = {k: (client_state[k] - server_state[k]) for k in server_state}
            deltas.append(delta)

            n_samples = len(client_train_loaders[cid_int].dataset)
            weights.append(float(n_samples) if weight_by_samples else 1.0)
            uplink += uplink_bytes_for_delta(delta)

        # Aggregate model (FedAvg-style weighted average)
        if deltas:
            wsum = float(sum(weights))
            avg_delta = zero_like(server_state)
            for d, w in zip(deltas, weights):
                add_state_(avg_delta, d, scale=(w / max(wsum, 1e-12)))
            new_state = {k: server_state[k] + avg_delta[k] for k in server_state}
            load_state_dict_(self.model, new_state, mode=self.state_mode)

        # Update server control: c += (1/N) * sum(Δc_i)
        # N = total number of clients (not just participants)
        if control_deltas:
            for name in self.server_control:
                delta_c_sum = sum(dc[name] for dc in control_deltas if name in dc)
                self.server_control[name] = (
                    self.server_control[name] + delta_c_sum / float(self.num_clients)
                ).detach()

        return float(np.mean(losses)) if losses else 0.0, int(uplink)
