"""
algorithms/fedimpro.py

FedImpro (ICLR 2024) - clean, integrated version for THIS codebase.

This implementation is designed to match your project's main.py interface and
your models.py SplitResNet18CIFAR split structure:

- split_model.low(x)  -> feat_map [B, C, H, W]
- split_model.high(feat_map) -> logits [B, num_classes]

Key idea implemented:
- Maintain global label-conditional Gaussian in feature space H|y:
    feat_vec = GAP(feat_map) -> [B, C]
    For each class y: mu_g[y], var_g[y] in R^C
- Each client updates its local stats mu_m, var_m via EMA (beta_m) from minibatches.
- Synthetic features are sampled from global stats:
    h_syn_vec ~ N(mu_g[y], (sqrt(var_g[y]) * noise_std)^2)
  Then broadcast to map:
    h_syn_map = h_syn_vec[:, :, None, None] expanded to [B, C, H, W]
  and fed into high().

Server-side:
- Aggregate model deltas with FedAvg, but ONLY for float/complex tensors.
  (Integer buffers like BN num_batches_tracked are kept from server_sd.)
- Update global stats with momentum beta_g using count-weighted aggregation.

Compatibility:
- main.py should import `FedImproRunner` from algorithms.fedimpro
- runner.model is the server model for evaluate()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import copy
import numpy as np
import torch
import torch.nn as nn


# ---------------------------
# Helpers
# ---------------------------

def _parse_batch(batch):
    """Loader yields (x, y) or (x, y, *extras)."""
    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise ValueError("Client loader must yield (x, y) tuples/lists.")


def _gap(feat_map: torch.Tensor) -> torch.Tensor:
    """Global average pool: [B,C,H,W] -> [B,C]."""
    if feat_map.dim() != 4:
        raise ValueError(f"Expected 4D feat_map [B,C,H,W], got {tuple(feat_map.shape)}")
    return feat_map.mean(dim=(2, 3))


def _vec_to_map(vec: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """[B,C] -> [B,C,H,W] via broadcast."""
    if vec.dim() != 2:
        raise ValueError(f"Expected 2D vec [B,C], got {tuple(vec.shape)}")
    B, C = vec.shape
    return vec.view(B, C, 1, 1).expand(B, C, H, W)


def _is_avg_dtype(t: torch.Tensor) -> bool:
    """Only average float/complex tensors in FedAvg."""
    return torch.is_tensor(t) and (torch.is_floating_point(t) or torch.is_complex(t))


def _state_dict_bytes(sd: Dict[str, torch.Tensor]) -> int:
    total = 0
    for v in sd.values():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return int(total)


def _clamp_var(var: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.clamp(var, min=eps)


# ---------------------------
# Client stats container
# ---------------------------

@dataclass
class _ClientStats:
    count: torch.Tensor  # [K] long on CPU
    mu: torch.Tensor     # [K, D] float32 on CPU
    var: torch.Tensor    # [K, D] float32 on CPU


# ---------------------------
# Runner
# ---------------------------

class FedImproRunner:
    """
    Runner compatible with your main.py interface.
    """

    def __init__(
        self,
        split_model: nn.Module,
        num_classes: int,
        device: torch.device,
        *,
        # FedImpro knobs
        beta_m: float = 0.9,            # client EMA for stats
        beta_g: float = 0.9,            # server momentum for stats
        noise_std: float = 0.0,         # for paper Table-1 style reproduction prefer 0.0
        syn_ratio: float = 1.0,         # synthetic batch size ratio (N_hat = N => 1.0)
        syn_loss_weight: float = 1.0,   # weight of synthetic CE
        var_eps: float = 1e-6,
        seed_syn: int = 1234,
    ) -> None:
        self.device = device
        self.model = split_model.to(device)

        self.num_classes = int(num_classes)
        self.beta_m = float(beta_m)
        self.beta_g = float(beta_g)
        self.noise_std = float(noise_std)
        self.syn_ratio = float(syn_ratio)
        self.syn_loss_weight = float(syn_loss_weight)
        self.var_eps = float(var_eps)

        self.ce = nn.CrossEntropyLoss()

        # Lazy init once we see first batch
        self._D: Optional[int] = None
        self._mu_g: Optional[torch.Tensor] = None   # [K,D] on device
        self._var_g: Optional[torch.Tensor] = None  # [K,D] on device

        self._rng = np.random.RandomState(int(seed_syn))

    # -----------------------
    # Public: one FL round
    # -----------------------

    def run_round(
        self,
        *,
        client_ids: List[int],
        client_train_loaders: Dict[int, Iterable],
        epochs: int,
        batch_size: int,
        sgd_cfg: Tuple[float, float, float],  # (lr, momentum, weight_decay)
        seed_train: int,
    ) -> Tuple[float, int]:
        lr, momentum, weight_decay = sgd_cfg
        epochs = int(epochs)

        server_sd = copy.deepcopy(self.model.state_dict())

        deltas: List[Dict[str, torch.Tensor]] = []
        stats_list: List[_ClientStats] = []

        total_loss = 0.0
        total_steps = 0
        uplink_bytes = 0

        for cid in client_ids:
            (
                delta_sd,
                cstats,
                loss_sum,
                steps,
                uplink_one,
            ) = self._client_update(
                cid=int(cid),
                loader=client_train_loaders[int(cid)],
                epochs=epochs,
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                seed=int(seed_train) + int(cid),
                server_sd=server_sd,
            )

            deltas.append(delta_sd)
            stats_list.append(cstats)

            total_loss += loss_sum
            total_steps += steps
            uplink_bytes += uplink_one

        # Server: FedAvg (float/complex only)
        self._server_apply_fedavg(server_sd=server_sd, deltas=deltas)

        # Server: update global stats
        self._server_update_global_stats(stats_list)

        mean_loss = float(total_loss / max(1, total_steps))
        return mean_loss, int(uplink_bytes)

    # -----------------------
    # Client update
    # -----------------------

    def _ensure_global_stats(self, D: int) -> None:
        if self._D is None:
            self._D = int(D)
        if self._mu_g is None or self._var_g is None:
            K = self.num_classes
            self._mu_g = torch.zeros((K, self._D), device=self.device, dtype=torch.float32)
            self._var_g = torch.ones((K, self._D), device=self.device, dtype=torch.float32)

    def _sample_from_global(self, y_syn: torch.Tensor) -> torch.Tensor:
        """
        Sample synthetic feature vectors from global stats:
            h ~ N(mu_g[y], (sqrt(var_g[y]) * noise_std)^2)
        If noise_std == 0, this becomes deterministic mu_g[y].
        """
        assert self._mu_g is not None and self._var_g is not None
        mu = self._mu_g[y_syn]  # [B,D]
        if self.noise_std <= 0:
            return mu
        var = _clamp_var(self._var_g[y_syn], self.var_eps)
        std = torch.sqrt(var) * self.noise_std
        eps = torch.randn_like(std)
        return mu + eps * std

    def _client_update(
        self,
        *,
        cid: int,
        loader: Iterable,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        seed: int,
        server_sd: Dict[str, torch.Tensor],
    ) -> Tuple[Dict[str, torch.Tensor], _ClientStats, float, int, int]:
        torch.manual_seed(seed)
        np.random.seed(seed)

        local_model = copy.deepcopy(self.model).to(self.device)
        local_model.load_state_dict(server_sd, strict=True)
        local_model.train()

        opt = torch.optim.SGD(
            local_model.parameters(),
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )

        # Lazy init local stats after first batch tells us D
        mu_m: Optional[torch.Tensor] = None   # [K,D] on device
        var_m: Optional[torch.Tensor] = None  # [K,D] on device
        cnt_m: Optional[torch.Tensor] = None  # [K] long on device

        loss_sum = 0.0
        steps = 0

        for _ in range(epochs):
            for batch in loader:
                x, y = _parse_batch(batch)
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True).long()

                # Real forward
                feat_map = local_model.low(x)          # [B,C,H,W]
                B, C, H, W = feat_map.shape
                feat_vec = _gap(feat_map)              # [B,C]  stats space

                # Init global stats (D == C here for this split)
                if self._D is None:
                    self._ensure_global_stats(D=int(feat_vec.shape[1]))

                # Init local stats
                if mu_m is None:
                    assert self._mu_g is not None and self._var_g is not None
                    mu_m = self._mu_g.detach().clone()
                    var_m = self._var_g.detach().clone()
                    cnt_m = torch.zeros((self.num_classes,), device=self.device, dtype=torch.long)

                assert mu_m is not None and var_m is not None and cnt_m is not None

                # Update local EMA stats per class from current batch
                with torch.no_grad():
                    for cls in torch.unique(y).tolist():
                        cls = int(cls)
                        mask = (y == cls)
                        if not torch.any(mask):
                            continue
                        fv = feat_vec[mask]  # [n,D]
                        b_mu = fv.mean(dim=0)
                        b_var = fv.var(dim=0, unbiased=False)

                        mu_m[cls] = self.beta_m * mu_m[cls] + (1.0 - self.beta_m) * b_mu
                        var_m[cls] = self.beta_m * var_m[cls] + (1.0 - self.beta_m) * b_var
                        cnt_m[cls] += int(mask.sum().item())

                # Loss: real
                logits_real = local_model.high(feat_map)  # high expects 4D map
                loss_real = self.ce(logits_real, y)

                # Loss: synthetic (high only, but we backprop through high params naturally)
                syn_B = int(max(1, round(B * self.syn_ratio)))
                # Keep it simple and safe: syn_B cannot exceed B in this implementation
                syn_B = min(syn_B, B)

                # sample labels (uniform over classes, same as many repo baselines)
                y_syn = torch.from_numpy(self._rng.randint(0, self.num_classes, size=(syn_B,))).to(self.device).long()
                syn_vec = self._sample_from_global(y_syn)      # [syn_B, D]
                syn_map = _vec_to_map(syn_vec, H=H, W=W)       # [syn_B, D, H, W]
                logits_syn = local_model.high(syn_map)
                loss_syn = self.ce(logits_syn, y_syn)

                loss = loss_real + self.syn_loss_weight * loss_syn

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                loss_sum += float(loss.detach().item())
                steps += 1

        # Build delta state_dict (send CPU tensors)
        local_sd = local_model.state_dict()
        delta_sd: Dict[str, torch.Tensor] = {}
        for k, w_server in server_sd.items():
            w_local = local_sd[k]
            if torch.is_tensor(w_local) and torch.is_tensor(w_server):
                delta_sd[k] = (w_local.detach().cpu() - w_server.detach().cpu())
            else:
                # Extremely rare; keep safe.
                delta_sd[k] = copy.deepcopy(w_local)

        # Client stats to CPU
        assert mu_m is not None and var_m is not None and cnt_m is not None
        cstats = _ClientStats(
            count=cnt_m.detach().cpu(),
            mu=mu_m.detach().cpu().float(),
            var=_clamp_var(var_m.detach(), self.var_eps).cpu().float(),
        )

        # Uplink bytes: model delta + stats
        uplink = _state_dict_bytes(delta_sd)
        uplink += int(cstats.count.numel() * cstats.count.element_size())
        uplink += int(cstats.mu.numel() * cstats.mu.element_size())
        uplink += int(cstats.var.numel() * cstats.var.element_size())

        return delta_sd, cstats, float(loss_sum), int(steps), int(uplink)

    # -----------------------
    # Server aggregation
    # -----------------------

    def _server_apply_fedavg(self, *, server_sd: Dict[str, torch.Tensor], deltas: List[Dict[str, torch.Tensor]]) -> None:
        """
        FedAvg update:
        - Average ONLY float/complex tensors.
        - Keep server buffers for non-avg dtypes (Long/Bool), e.g. BN num_batches_tracked.
        """
        if len(deltas) == 0:
            return

        new_sd: Dict[str, torch.Tensor] = {}

        for k, w_server in server_sd.items():
            if not torch.is_tensor(w_server):
                new_sd[k] = w_server
                continue

            if not _is_avg_dtype(w_server):
                # Keep server value (do NOT average integer buffers)
                new_sd[k] = w_server
                continue

            # stack deltas on CPU as float32 for mean stability
            stack = torch.stack([d[k].to(dtype=torch.float32) for d in deltas], dim=0)  # [M,...]
            avg_delta = stack.mean(dim=0).to(dtype=w_server.dtype)

            new_sd[k] = (w_server.detach().cpu() + avg_delta).to(w_server.device)

        self.model.load_state_dict(new_sd, strict=True)

    def _server_update_global_stats(self, stats_list: List[_ClientStats]) -> None:
        """
        Update global mu_g/var_g with count-weighted aggregation + momentum beta_g.
        """
        if len(stats_list) == 0:
            return
        if self._D is None:
            return
        self._ensure_global_stats(self._D)
        assert self._mu_g is not None and self._var_g is not None

        K, D = self._mu_g.shape

        # Aggregate count-weighted across clients
        sum_cnt = torch.zeros((K,), device=self.device, dtype=torch.float32)
        sum_mu = torch.zeros((K, D), device=self.device, dtype=torch.float32)
        sum_var = torch.zeros((K, D), device=self.device, dtype=torch.float32)

        for st in stats_list:
            cnt = st.count.to(self.device).to(torch.float32).clamp(min=0.0)  # [K]
            mu = st.mu.to(self.device).to(torch.float32)                      # [K,D]
            var = st.var.to(self.device).to(torch.float32)                    # [K,D]
            var = _clamp_var(var, self.var_eps)

            sum_cnt += cnt
            sum_mu += mu * cnt.unsqueeze(1)
            sum_var += var * cnt.unsqueeze(1)

        # Where count==0, keep previous
        mask = (sum_cnt > 0).view(K, 1)  # [K,1] bool
        mu_hat = self._mu_g.clone()
        var_hat = self._var_g.clone()

        if mask.any():
            denom = sum_cnt.clamp(min=1.0).unsqueeze(1)  # [K,1]
            mu_new = sum_mu / denom
            var_new = sum_var / denom
            var_new = _clamp_var(var_new, self.var_eps)

            mu_hat = torch.where(mask, mu_new, mu_hat)
            var_hat = torch.where(mask, var_new, var_hat)

        # Momentum update (Eq.7 spirit)
        self._mu_g = self.beta_g * self._mu_g + (1.0 - self.beta_g) * mu_hat
        self._var_g = self.beta_g * self._var_g + (1.0 - self.beta_g) * var_hat
        self._var_g = _clamp_var(self._var_g, self.var_eps)
