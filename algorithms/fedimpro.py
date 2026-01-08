from __future__ import annotations

import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    fedavg_aggregate,
    load_param_state_dict_,
    param_state_dict,
    sub_state,
    uplink_bytes_for_delta,
)


def _gather_gap_features(
    low: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[int, int], torch.Tensor, torch.Tensor, int]:
    """
    Computes per-class mean/var of GAP features on local TRAIN data.

    Returns:
      mean_c: dict class -> (D,)
      var_c: dict class -> (D,)
      n_c: dict class -> int
      mean_all: (D,)
      var_all: (D,)
      n_all: int
    """
    low.eval()
    sums: Dict[int, torch.Tensor] = {}
    sums2: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {}

    sum_all = None
    sum2_all = None
    n_all = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            feat_map = low(x)  # (B,128,H,W) for split after layer2
            h = feat_map.mean(dim=(2, 3))  # (B,128)
            D = h.shape[1]

            if sum_all is None:
                sum_all = torch.zeros(D, device=device)
                sum2_all = torch.zeros(D, device=device)

            sum_all += h.sum(dim=0)
            sum2_all += (h * h).sum(dim=0)
            n_all += int(h.shape[0])

            for c in range(num_classes):
                mask = (y == c)
                if not mask.any():
                    continue
                hc = h[mask]
                if c not in sums:
                    sums[c] = torch.zeros(D, device=device)
                    sums2[c] = torch.zeros(D, device=device)
                    counts[c] = 0
                sums[c] += hc.sum(dim=0)
                sums2[c] += (hc * hc).sum(dim=0)
                counts[c] += int(hc.shape[0])

    assert sum_all is not None and sum2_all is not None
    mean_all = sum_all / max(1, n_all)
    var_all = (sum2_all / max(1, n_all)) - mean_all * mean_all

    mean_c: Dict[int, torch.Tensor] = {}
    var_c: Dict[int, torch.Tensor] = {}
    for c, n in counts.items():
        mu = sums[c] / max(1, n)
        vv = (sums2[c] / max(1, n)) - mu * mu
        mean_c[c] = mu
        var_c[c] = vv

    return mean_c, var_c, counts, mean_all, var_all, n_all


class FedImproRunner:
    """
    FedImpro baseline (minimal):
    - Split model after layer2 (D=128 GAP features).
    - Server maintains per-class diagonal Gaussian stats + global fallback.
    - Local objective per batch:
        loss = CE(high(low(x)), y) + CE(high(syn_map), y_syn), lambda=1.0
      where syn_map is expanded from sampled h_syn.
    - Synthetic branch updates HIGH only (low is not involved in syn forward).
    """

    def __init__(
        self,
        split_model,  # SplitResNet18CIFAR (low, high)
        num_classes: int,
        device: torch.device,
        stats_noise_sigma: float = 0.1,
        min_var: float = 1e-6,
    ) -> None:
        self.device = device
        self.model = split_model
        self.model.low.to(device)
        self.model.high.to(device)

        self.num_classes = int(num_classes)
        self.stats_noise_sigma = float(stats_noise_sigma)
        self.min_var = float(min_var)

        # Global stats: dict class -> (mean,var) in R^128, kept on device.
        self.stats_class: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.stats_all: Tuple[torch.Tensor, torch.Tensor] | None = None

    def run_round(
        self,
        client_ids: Sequence[int],
        client_train_loaders: Dict[int, torch.utils.data.DataLoader],
        epochs: int,
        batch_size: int,
        sgd_cfg: Tuple[float, float, float],  # (lr, momentum, weight_decay)
        seed_train: int,
    ) -> Tuple[float, int]:
        lr, momentum, weight_decay = sgd_cfg

        server_params = param_state_dict(self._full_model())

        deltas: List[Dict[str, torch.Tensor]] = []
        uplink = 0
        losses: List[float] = []

        # Collect client stats for server aggregation
        stats_payload = []  # list of (mean_c, var_c, n_c, mean_all, var_all, n_all)
        for cid in client_ids:
            local = copy.deepcopy(self.model)
            
                # ---- FIX: server_params uses "low.*" / "high.*" keys, but local.low/high expect no prefix.
            low_state = {
                  k.split("low.", 1)[1]: v
                  for k, v in server_params.items()
                  if k.startswith("low.")
            }
            high_state = {
                  k.split("high.", 1)[1]: v
                   for k, v in server_params.items()
                   if k.startswith("high.")
             }
            
            load_param_state_dict_(local.low, low_state)
            load_param_state_dict_(local.high, high_state)
            # ---- FIX END


            # stats computed on local TRAIN data
            mean_c, var_c, n_c, mean_all, var_all, n_all = _gather_gap_features(
                local.low, client_train_loaders[cid], self.device, self.num_classes
            )
            stats_payload.append((mean_c, var_c, n_c, mean_all, var_all, n_all))

            # local training
            loss = self._local_train(local, client_train_loaders[cid], epochs, lr, momentum, weight_decay, seed_train + cid)
            losses.append(loss)

            client_params = param_state_dict(self._full_model(local))
            delta = sub_state(client_params, server_params)
            deltas.append(delta)
            uplink += uplink_bytes_for_delta(delta)

        # Aggregate model
        new_params = fedavg_aggregate(server_params, deltas)
        load_param_state_dict_(self._full_model(), new_params)

        # Aggregate feature stats on server
        self._aggregate_stats(stats_payload)

        return float(np.mean(losses)) if losses else 0.0, uplink

    def _full_model(self, model_override=None) -> nn.Module:
        """
        Expose a combined module view for param_state_dict/fedavg.
        Minimal: just an nn.Module with low/high as submodules.
        """
        if model_override is None:
            return self.model
        return model_override

    def _aggregate_stats(self, stats_payload) -> None:
        # Aggregate per-class using sample-count weights. Also aggregate global stats_all.
        # Add Gaussian noise sigma=0.1 to mean/var for privacy.
        device = self.device

        # Global-all
        sum_all = None
        sum2_all = None
        n_all_total = 0

        # Per-class
        sum_c: Dict[int, torch.Tensor] = {}
        sum2_c: Dict[int, torch.Tensor] = {}
        n_c_total: Dict[int, int] = {}

        for mean_c, var_c, n_c, mean_all, var_all, n_all in stats_payload:
            # for all-stats: use E[x] and E[x^2]
            if sum_all is None:
                D = mean_all.numel()
                sum_all = torch.zeros(D, device=device)
                sum2_all = torch.zeros(D, device=device)

            sum_all += mean_all * n_all
            # recover sum of squares: E[x^2] = var + mean^2
            ex2_all = (var_all + mean_all * mean_all)
            sum2_all += ex2_all * n_all
            n_all_total += int(n_all)

            for c, n in n_c.items():
                mu = mean_c[c]
                vv = var_c[c]
                if c not in sum_c:
                    sum_c[c] = torch.zeros_like(mu)
                    sum2_c[c] = torch.zeros_like(mu)
                    n_c_total[c] = 0
                sum_c[c] += mu * n
                sum2_c[c] += (vv + mu * mu) * n
                n_c_total[c] += int(n)

        if sum_all is not None and n_all_total > 0:
            mean_all = sum_all / n_all_total
            ex2_all = sum2_all / n_all_total
            var_all = ex2_all - mean_all * mean_all
            mean_all = mean_all + torch.randn_like(mean_all) * self.stats_noise_sigma
            var_all = var_all + torch.randn_like(var_all) * self.stats_noise_sigma
            var_all = torch.clamp(var_all, min=self.min_var)
            self.stats_all = (mean_all.detach(), var_all.detach())

        for c, n in n_c_total.items():
            if n <= 0:
                continue
            mean = sum_c[c] / n
            ex2 = sum2_c[c] / n
            var = ex2 - mean * mean
            mean = mean + torch.randn_like(mean) * self.stats_noise_sigma
            var = var + torch.randn_like(var) * self.stats_noise_sigma
            var = torch.clamp(var, min=self.min_var)
            self.stats_class[c] = (mean.detach(), var.detach())

    def _local_train(
        self,
        split_model,
        loader,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        seed: int,
    ) -> float:
        torch.manual_seed(seed)

        split_model.low.train()
        split_model.high.train()

        opt = torch.optim.SGD(
            list(split_model.low.parameters()) + list(split_model.high.parameters()),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        total_loss = 0.0
        total = 0

        # Precompute label distribution from loader.dataset targets via one pass of y's
        # (deterministic, and cheap). We do it once per client per round.
        y_all = []
        for _, y in loader:
            y_all.append(y)
        y_all = torch.cat(y_all, dim=0)
        counts = torch.bincount(y_all, minlength=self.num_classes).cpu().numpy().astype(np.float64)
        probs = counts / max(1.0, counts.sum())
        # fallback if client has pathological empty (shouldn't happen)
        if probs.sum() <= 0:
            probs = np.full(self.num_classes, 1.0 / self.num_classes)

        for _ in range(epochs):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                B = x.shape[0]

                opt.zero_grad(set_to_none=True)

                # Real branch (updates low + high)
                feat_map = split_model.low(x)  # (B,128,H,W)
                logits_real = split_model.high(feat_map)
                loss_real = F.cross_entropy(logits_real, y)

                # Synthetic branch (updates high only)
                H, W = feat_map.shape[2], feat_map.shape[3]
                y_syn_np = np.random.choice(self.num_classes, size=B, replace=True, p=probs)
                y_syn = torch.from_numpy(y_syn_np).to(self.device, non_blocking=True, dtype=y.dtype)

                h_syn = self._sample_h_syn(y_syn)  # (B,128)
                syn_map = h_syn[:, :, None, None].expand(B, 128, H, W)
                logits_syn = split_model.high(syn_map)
                loss_syn = F.cross_entropy(logits_syn, y_syn)

                loss = loss_real + loss_syn
                loss.backward()
                opt.step()

                total_loss += float(loss.item()) * int(B)
                total += int(B)

        return total_loss / max(1, total)

    def _sample_h_syn(self, y_syn: torch.Tensor) -> torch.Tensor:
        """
        Sample (B,128) from per-class stats; fallback to stats_all if missing.
        """
        B = int(y_syn.shape[0])
        D = 128
        if self.stats_all is None:
            # Cold start: standard normal (should be short-lived).
            return torch.randn(B, D, device=self.device)

        mean_all, var_all = self.stats_all
        out = torch.empty(B, D, device=self.device)
        for i in range(B):
            c = int(y_syn[i].item())
            if c in self.stats_class:
                mu, var = self.stats_class[c]
            else:
                mu, var = mean_all, var_all
            eps = torch.randn_like(mu)
            out[i] = mu + eps * torch.sqrt(var)
        return out
