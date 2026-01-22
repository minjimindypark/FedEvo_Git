"""
algorithms/fedga.py

FedGA-FL (IID): Genetic Candidate Evolution for Federated Learning under IID condition
- Sentinel attribution removed (PDF Sec. 4.3 removed).
- Non-IID partition removed (IID clients).
- Focus: genetic operators over a candidate population can improve accuracy vs single-model baselines.

Design goals
- Deterministic-friendly (seeded RNG).
- Clean separation: client selection + local training + candidate-wise aggregation + GA evolution.
- No hidden state mixing: population models are always full parameter state_dict clones.

Expected dependencies (already present in your repo):
- algorithms.base: param_state_dict, load_param_state_dict_, uplink_bytes_for_delta
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    param_state_dict,
    load_param_state_dict_,
    uplink_bytes_for_delta,
)

_EPS = 1e-12


def _clone_params(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in params.items()}


def _avg_state_dicts(
    sds: List[Dict[str, torch.Tensor]],
    keys: List[str],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    if len(sds) == 0:
        raise ValueError("Cannot average empty list")
    if len(sds) == 1:
        return _clone_params(sds[0])

    if weights is None:
        weights = [1.0 / len(sds)] * len(sds)
    else:
        if len(weights) != len(sds):
            raise ValueError("weights length must match number of state_dicts")
        tot = float(sum(weights))
        if tot <= 0:
            raise ValueError("weights must sum to a positive value")
        weights = [float(w) / tot for w in weights]

    out: Dict[str, torch.Tensor] = {}
    ref = sds[0]
    for k in keys:
        acc = torch.zeros_like(ref[k], dtype=torch.float32)
        for sd, w in zip(sds, weights):
            acc += float(w) * sd[k].float()
        out[k] = acc.to(ref[k].dtype)
    return out


def _add_sd(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
    return {k: (base[k].float() + delta[k].float()).to(base[k].dtype) for k in keys}


@dataclass
class GAConfig:
    m: int = 5              # population size
    rho: float = 0.4        # top-k ratio for exploitation
    gamma: float = 1.5      # selection-weight exponent
    elitism: int = 1        # number of elites carried over
    lam_low: float = 0.4    # crossover lambda range
    lam_high: float = 0.6
    mutate_prob_layer: float = 0.05  # per-layer mutation probability (coarse)
    sigma_mut: float = 0.01          # mutation scale as fraction of layer std
    enable_mutation: bool = True


class FedEvoRunner:
    """
    Genetic Federated Optimization (IID)

    Protocol (round t):
    1) Server holds population P(t) = {theta_j}.
    2) Each client selects best candidate by local val loss.
    3) Client trains locally from that candidate and uploads delta only.
    4) Server aggregates deltas per candidate to form theta_bar_j.
    5) Server evolves population using GA operators on {theta_bar_j}.
    6) For evaluation convenience, server exposes a "best" anchor model (by usage).

    Notes:
    - This runner intentionally keeps uplink cost identical to FedAvg: one delta per client.
    - Downlink cost grows with m (population broadcast), same as original FedEvo concept.
    """

    def __init__(
        self,
        model_ctor,
        num_classes: int,
        device: torch.device,
        ga: GAConfig = GAConfig(),
        seed: int = 42,
        val_batches: Optional[int] = None,
        weight_by_samples: bool = True,
        deterministic: bool = False,
    ) -> None:
        self.device = device
        self.model_ctor = model_ctor
        self.num_classes = int(num_classes)
        self.ga = ga
        self.val_batches = val_batches
        self.weight_by_samples = bool(weight_by_samples)

        self.rng = np.random.RandomState(int(seed))

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        base_model: nn.Module = model_ctor(self.num_classes).to(self.device)
        self.theta_base = param_state_dict(base_model)
        self.param_keys = sorted(list(self.theta_base.keys()))

        self.population: List[Dict[str, torch.Tensor]] = [
            _clone_params(self.theta_base) for _ in range(int(self.ga.m))
        ]

        # shared scratch model for cheap validation evaluation
        self._scratch_model: nn.Module = model_ctor(self.num_classes).to(self.device)

        self.round_idx = 0
        self.last_usage_counts: List[int] = [0 for _ in range(int(self.ga.m))]

    # ----------------------- public API -----------------------

    def run_round(
        self,
        client_ids: Sequence[int],
        client_train_loaders: Dict[int, torch.utils.data.DataLoader],
        client_val_loaders: Dict[int, torch.utils.data.DataLoader],
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
    ) -> Tuple[float, int]:
        """
        Returns:
            mean_train_loss: float
            uplink_bytes: int
        """
        lr, momentum, weight_decay = sgd_cfg
        self.round_idx += 1
        St = list(map(int, client_ids))

        # per-candidate buckets
        deltas_by_j: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(self.ga.m)]
        n_by_j: List[List[int]] = [[] for _ in range(self.ga.m)]
        usage_counts = np.zeros((self.ga.m,), dtype=np.int64)

        train_losses: List[float] = []
        uplink_bytes = 0

        # ---------- client loop ----------
        for cid in St:
            j_star = self._client_select_best(client_val_loaders[cid])

            local_model = self.model_ctor(self.num_classes).to(self.device)
            load_param_state_dict_(local_model, self.population[j_star])

            loss, n = self._local_train(
                model=local_model,
                loader=client_train_loaders[cid],
                epochs=int(epochs),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                seed=int(seed_train) + cid,
            )
            train_losses.append(loss)

            local_params = param_state_dict(local_model)

            # delta-only uplink (FedAvg-level)
            delta = {
                k: (local_params[k].float() - self.population[j_star][k].float()).to(local_params[k].dtype)
                for k in self.param_keys
            }
            uplink_bytes += uplink_bytes_for_delta(delta)

            deltas_by_j[j_star].append(delta)
            n_by_j[j_star].append(int(n))
            usage_counts[j_star] += 1

        self.last_usage_counts = usage_counts.tolist()

        # ---------- candidate-wise aggregation ----------
        theta_bars: List[Dict[str, torch.Tensor]] = []
        for j in range(self.ga.m):
            if len(deltas_by_j[j]) == 0:
                theta_bars.append(_clone_params(self.population[j]))
                continue

            if self.weight_by_samples:
                weights = [float(n) for n in n_by_j[j]]
                avg_delta = _avg_state_dicts(deltas_by_j[j], self.param_keys, weights=weights)
            else:
                avg_delta = _avg_state_dicts(deltas_by_j[j], self.param_keys, weights=None)

            theta_bars.append(_add_sd(self.population[j], avg_delta, self.param_keys))

        # ---------- GA evolution ----------
        self.population = self._evolve(theta_bars, usage_counts)

        # "base" anchor for eval: most-used theta_bar (ties resolved by smallest index)
        best_j = int(np.argmax(usage_counts)) if usage_counts.sum() > 0 else 0
        self.theta_base = _clone_params(theta_bars[best_j])

        mean_loss = float(np.mean(train_losses)) if train_losses else 0.0
        return mean_loss, int(uplink_bytes)

    def get_best_model(self) -> nn.Module:
        load_param_state_dict_(self._scratch_model, self.theta_base)
        return self._scratch_model

    # ----------------------- internal: GA ops -----------------------

    def _evolve(self, theta_bars: List[Dict[str, torch.Tensor]], usage_counts: np.ndarray) -> List[Dict[str, torch.Tensor]]:
        m = int(self.ga.m)
        if m != len(theta_bars):
            raise ValueError(f"Population size mismatch: ga.m={m} but got {len(theta_bars)} theta_bars")

        # Prepare ordering
        order_desc = np.argsort(-usage_counts)  # most used first
        new_pop: List[Dict[str, torch.Tensor]] = []

        # 1) Elitism: keep top elites unchanged
        elite_n = max(0, min(int(self.ga.elitism), m))
        for i in range(elite_n):
            j = int(order_desc[i])
            new_pop.append(_clone_params(theta_bars[j]))

        remaining = m - len(new_pop)

        # 2) Top-k weighted aggregate (exploitation)
        if remaining > 0:
            k_star = max(1, int(math.floor(float(self.ga.rho) * m)))
            top = order_desc[:k_star]

            w = (usage_counts[top].astype(np.float64) + 1e-8) ** float(self.ga.gamma)
            w = w / (w.sum() + _EPS)

            theta_topk: Dict[str, torch.Tensor] = {}
            for key in self.param_keys:
                acc = torch.zeros_like(theta_bars[0][key], dtype=torch.float32)
                for wi, j in zip(w, top):
                    acc += float(wi) * theta_bars[int(j)][key].float()
                theta_topk[key] = acc.to(theta_bars[0][key].dtype)

            new_pop.append(theta_topk)
            remaining -= 1

        # 3) Crossover fill
        while remaining > 0:
            p_idx, q_idx = self._sample_two_parents(usage_counts)
            lam = float(self.rng.uniform(float(self.ga.lam_low), float(self.ga.lam_high)))

            child = {
                k: (lam * theta_bars[p_idx][k].float() + (1.0 - lam) * theta_bars[q_idx][k].float()).to(theta_bars[p_idx][k].dtype)
                for k in self.param_keys
            }
            new_pop.append(child)
            remaining -= 1

        # 4) Mutation (optional): mutate non-elite slots only (index >= elite_n)
        if self.ga.enable_mutation and float(self.ga.mutate_prob_layer) > 0.0:
            for i in range(elite_n, len(new_pop)):
                self._mutate_inplace(new_pop[i])

        return new_pop[:m]

    def _sample_two_parents(self, usage_counts: np.ndarray) -> Tuple[int, int]:
        m = int(self.ga.m)
        # If no usage signal, sample uniformly
        if usage_counts.sum() <= 0:
            p, q = self.rng.choice(m, size=2, replace=False).tolist()
            return int(p), int(q)

        probs = usage_counts.astype(np.float64) + 1e-8
        probs = probs / (probs.sum() + _EPS)
        p = int(self.rng.choice(m, p=probs))
        q = int(self.rng.choice(m, p=probs))
        # ensure distinct parents (retry a few times then force)
        for _ in range(5):
            if q != p:
                break
            q = int(self.rng.choice(m, p=probs))
        if q == p:
            q = int((p + 1) % m)
        return p, q

    def _mutate_inplace(self, sd: Dict[str, torch.Tensor]) -> None:
        # Coarse mutation: with probability mutate_prob_layer per tensor (layer),
        # add Gaussian noise scaled by sigma_mut * std(layer).
        for k in self.param_keys:
            if float(self.rng.rand()) >= float(self.ga.mutate_prob_layer):
                continue
            w = sd[k]
            std = float(w.float().std().item())
            if not np.isfinite(std) or std <= 0.0:
                std = 1e-4
            sigma = float(self.ga.sigma_mut) * std
            noise = torch.randn_like(w.float()) * sigma
            sd[k] = (w.float() + noise).to(w.dtype)

    # ----------------------- internal: client work -----------------------

    @torch.inference_mode()
    def _client_select_best(self, val_loader) -> int:
        best_j, best_loss = 0, float("inf")
        for j in range(int(self.ga.m)):
            load_param_state_dict_(self._scratch_model, self.population[j])
            loss = self._eval_loss_limited(self._scratch_model, val_loader)
            if loss < best_loss:
                best_loss = loss
                best_j = j
        return int(best_j)

    @torch.inference_mode()
    def _eval_loss_limited(self, model: nn.Module, loader) -> float:
        model.eval()
        total_loss, total_samples = 0.0, 0
        batches = 0
        for x, y in loader:
            if self.val_batches is not None and batches >= int(self.val_batches):
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_samples += int(y.numel())
            batches += 1
        return total_loss / max(1, total_samples)

    def _local_train(
        self,
        model: nn.Module,
        loader,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        seed: int,
    ) -> Tuple[float, int]:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

        model.train()
        opt = torch.optim.SGD(
            model.parameters(),
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )

        total_loss, total_samples = 0.0, 0
        for _ in range(int(epochs)):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()

                bs = int(y.numel())
                total_loss += float(loss.item()) * bs
                total_samples += bs

        if total_samples == 0:
            try:
                total_samples = int(len(loader.dataset))
            except Exception:
                total_samples = 1

        return total_loss / max(1, total_samples), total_samples
