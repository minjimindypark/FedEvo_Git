# algorithms/feddyn.py
"""FedDyn: Federated Learning Based on Dynamic Regularization
(Acar et al., ICLR 2021).

Local objective per client i:
    F_i(w) = f_i(w) + alpha * <w, h_i - w_cld>
    + (alpha/2)*||w||^2  (absorbed into optimizer weight_decay = alpha + wd)

After local training:
    h_i_new = h_i_old + (w_i_trained - w_cld)

Server update (partial participation with N total clients):
    avg_state  = mean(w_i for i in selected)
    h_global  += (1/N) * sum(h_i_new - h_i_old for i in selected)
    cld_state  = avg_state + h_global   ← sent to clients & used for evaluation

NOTE: FedDyn uses no momentum in local training (same as original GitHub implementation).
Reference: https://github.com/alpemreacar/FedDyn
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


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _param_zeros(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Zero tensors for each named_parameter (CPU)."""
    return {name: torch.zeros_like(p.detach().cpu()) for name, p in model.named_parameters()}


def _param_snapshot(model: nn.Module) -> Dict[str, torch.Tensor]:
    """CPU snapshot of named_parameters."""
    return {name: p.detach().clone().cpu() for name, p in model.named_parameters()}


# ─────────────────────────────────────────────
# Local training
# ─────────────────────────────────────────────

def local_train_feddyn(
    *,
    model: nn.Module,
    loader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed_train: int,
    cld_params: Dict[str, torch.Tensor],   # w_cld sent from server (CPU)
    h_i: Dict[str, torch.Tensor],          # client correction (CPU)
    alpha: float,
) -> float:
    """
    Local FedDyn training (no momentum, per original paper).

    Effective local objective at each step:
        loss_ce(w) + alpha * <w, h_i - w_cld>
    Optimizer weight_decay = alpha + weight_decay  →  absorbs (alpha/2)||w||^2.

    Returns average CE loss (regularization excluded).
    """
    torch.manual_seed(int(seed_train))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed_train))

    model.train()

    # weight_decay = alpha + original_wd  (proximal term absorbed)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=0.0,
        weight_decay=float(alpha) + float(weight_decay),
    )

    # Precompute correction vector: (h_i - w_cld) on device — fixed for entire round
    correction = {
        name: (h_i[name] - cld_params[name]).to(device)
        for name in h_i
    }

    total_loss = 0.0
    total = 0

    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss_ce = F.cross_entropy(logits, y)

            # Dynamic regularization: alpha * <w, h_i - w_cld>
            loss_dyn = torch.tensor(0.0, device=device)
            for name, param in model.named_parameters():
                if name in correction:
                    loss_dyn = loss_dyn + (param * correction[name]).sum()
            loss_dyn = float(alpha) * loss_dyn

            (loss_ce + loss_dyn).backward()
            opt.step()

            bs = int(y.numel())
            total_loss += float(loss_ce.item()) * bs   # log CE only
            total += bs

    return total_loss / max(1, total)


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────

class FedDynRunner(BaseRunner):
    """
    FedDyn runner.

    self.model holds the cloud model (w_cld = avg + h_global),
    which is what clients use as starting point and what we evaluate.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        num_clients: int,
        state_mode: str = "params",
        alpha: float = 0.01,
    ) -> None:
        super().__init__(device=device)
        self.model = model.to(device)
        self.state_mode = str(state_mode).lower().strip()
        self.num_clients = int(num_clients)
        self.alpha = float(alpha)

        # Initialize h_i = 0 for all clients (CPU)
        template = _param_zeros(model)
        self.client_h: Dict[int, Dict[str, torch.Tensor]] = {
            i: {k: v.clone() for k, v in template.items()}
            for i in range(self.num_clients)
        }
        # h_global = (1/N) * sum_all(h_i), updated incrementally (CPU)
        self.h_global: Dict[str, torch.Tensor] = {k: v.clone() for k, v in template.items()}

    def run_round(
        self,
        client_ids: Sequence[int],
        client_train_loaders: Dict,
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
        weight_by_samples: bool = True,
    ) -> Tuple[float, int]:
        lr, _momentum, weight_decay = sgd_cfg  # _momentum intentionally ignored: FedDyn requires momentum=0.0
        # per Acar et al. ICLR 2021 and original repo (alpemreacar/FedDyn).
        # Passing non-zero momentum via sgd_cfg is silently overridden to 0.0 inside local_train_feddyn.

        # cld_params: what the server sends to clients this round
        # = current self.model params (which stores w_cld)
        cld_params = _param_snapshot(self.model)

        deltas: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []
        h_deltas: List[Dict[str, torch.Tensor]] = []  # h_i_new - h_i_old per participant
        uplink = 0
        losses: List[float] = []

        for cid in client_ids:
            cid_int = int(cid)

            # Client starts from cld_model (same as server sends)
            local = copy.deepcopy(self.model)
            # state_mode may differ from params_only; load cld_state properly
            cld_state = get_state_dict(self.model, mode=self.state_mode)
            load_state_dict_(local, cld_state, mode=self.state_mode)

            loss = local_train_feddyn(
                model=local,
                loader=client_train_loaders[cid_int],
                epochs=int(epochs),
                lr=float(lr),
                weight_decay=float(weight_decay),
                seed_train=int(seed_train) + cid_int,
                cld_params=cld_params,
                h_i=self.client_h[cid_int],
                alpha=self.alpha,
                device=self.device,
            )
            losses.append(float(loss))

            # w_i trained params (CPU)
            w_i = _param_snapshot(local)

            # h_i update: h_i_new = h_i_old + (w_i - w_cld)
            old_h = self.client_h[cid_int]
            new_h = {
                name: old_h[name] + (w_i[name] - cld_params[name])
                for name in old_h
            }
            h_delta = {name: new_h[name] - old_h[name] for name in old_h}
            h_deltas.append(h_delta)
            self.client_h[cid_int] = new_h

            # Model delta for aggregation (using state_mode state)
            client_state = get_state_dict(local, mode=self.state_mode)
            delta = {k: (client_state[k] - cld_state[k]) for k in cld_state}
            deltas.append(delta)

            n_samples = len(client_train_loaders[cid_int].dataset)
            weights.append(float(n_samples) if weight_by_samples else 1.0)
            uplink += uplink_bytes_for_delta(delta)

        if not deltas:
            return 0.0, 0

        # ── Server update ──────────────────────────────────────────────────

        # 1. avg_state = weighted average of trained client models
        wsum = float(sum(weights))
        avg_delta = zero_like(cld_state)
        for d, w in zip(deltas, weights):
            add_state_(avg_delta, d, scale=(w / max(wsum, 1e-12)))
        avg_state = {k: cld_state[k] + avg_delta[k] for k in cld_state}

        # 2. h_global += (1/N) * sum(h_delta_i for selected clients)
        for name in self.h_global:
            dh_sum = sum(hd[name] for hd in h_deltas if name in hd)
            self.h_global[name] = (
                self.h_global[name] + dh_sum / float(self.num_clients)
            ).detach()

        # 3. cld_state = avg_state + h_global  (for params keys only)
        #    h_global lives in named_parameter space; avg_state may include BN buffers
        #    → apply h_global only to parameter keys
        param_names = set(cld_params.keys())
        cld_new_state = {}
        for k in avg_state:
            if k in param_names and k in self.h_global:
                cld_new_state[k] = avg_state[k] + self.h_global[k].to(avg_state[k].device)
            else:
                cld_new_state[k] = avg_state[k]

        load_state_dict_(self.model, cld_new_state, mode=self.state_mode)

        return float(np.mean(losses)) if losses else 0.0, int(uplink)
