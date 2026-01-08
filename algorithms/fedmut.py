from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import (
    add_state_,
    fedavg_aggregate,
    load_param_state_dict_,
    local_train_sgd,
    param_state_dict,
    sub_state,
    uplink_bytes_for_delta,
)


class FedMutRunner:
    """
    FedMut baseline:
    - FedAvg aggregation over deltas
    - server-side mutation applied to NEXT broadcast model using global delta g_r = w_glb[r] - w_glb[r-1]
    - mask is scalar sign per parameter tensor (per model.named_parameters() entry)
    - dedicated RNG stream for mutation mask
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        alpha_mut: float = 4.0,
        seed_mut: int = 123,  # independent stream
    ) -> None:
        self.device = device
        self.model = model.to(device)
        self.alpha_mut = float(alpha_mut)
        self.rng_mut = np.random.RandomState(seed_mut)

        self.w_prev: Dict[str, torch.Tensor] | None = None  # parameter-only
        self.w_curr: Dict[str, torch.Tensor] | None = None

    def run_round(
        self,
        client_ids: Sequence[int],
        client_train_loaders: Dict[int, torch.utils.data.DataLoader],
        epochs: int,
        sgd_cfg: Tuple[float, float, float],  # (lr, momentum, weight_decay)
        seed_train: int,
    ) -> Tuple[float, int]:
        """
        Returns:
          (avg_train_loss_over_clients, uplink_bytes_this_round)
        """
        lr, momentum, weight_decay = sgd_cfg

        # Broadcast current server model params.
        server_params = param_state_dict(self.model)

        client_losses: List[float] = []
        deltas: List[Dict[str, torch.Tensor]] = []
        uplink = 0

        for cid in client_ids:
            local_model = copy.deepcopy(self.model)
            load_param_state_dict_(local_model, server_params)

            loss = local_train_sgd(
                local_model,
                client_train_loaders[cid],
                self.device,
                epochs=epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                seed_train=seed_train + cid,
            )
            client_losses.append(loss)

            client_params = param_state_dict(local_model)
            delta = sub_state(client_params, server_params)
            deltas.append(delta)
            uplink += uplink_bytes_for_delta(delta)

        # FedAvg update -> w_glb[r]
        if self.w_curr is None:
            self.w_curr = server_params
        self.w_prev = self.w_curr
        self.w_curr = fedavg_aggregate(server_params, deltas)

        # Apply aggregated weights to model.
        load_param_state_dict_(self.model, self.w_curr)

        # Mutation affects the NEXT broadcast model: we mutate self.model in-place now.
        if self.w_prev is not None:
            self._mutate_in_place(self.w_prev, self.w_curr)

        avg_loss = float(np.mean(client_losses)) if len(client_losses) else 0.0
        return avg_loss, uplink

    def _mutate_in_place(self, w_prev: Dict[str, torch.Tensor], w_curr: Dict[str, torch.Tensor]) -> None:
        # If this is the very first round (w_prev==w_curr initial), g may be ~0; OK.
        g = sub_state(w_curr, w_prev)

        mutated = {k: v.detach().clone() for k, v in w_curr.items()}

        # For each parameter tensor (named_parameters key), sample scalar sign.
        for k in mutated.keys():
            v = self.rng_mut.choice([-1.0, 1.0])
            mutated[k].add_(g[k], alpha=self.alpha_mut * float(v))

        load_param_state_dict_(self.model, mutated)