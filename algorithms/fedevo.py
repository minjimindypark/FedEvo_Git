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
    make_loader,
    param_state_dict,
    sub_state,
    uplink_bytes_for_delta,
)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten()
    b = b.flatten()
    return float(F.cosine_similarity(a, b, dim=0).item())


class FedEvoRunner:
    """
    FedEvo (ours):
    - population size m=5 candidates (parameter-only)
    - each candidate has a unique sentinel vector s_j in {+nu, -nu}^64 as model.sentinel (requires_grad=False)
    - client receives all m, evaluates on local VAL, picks best, trains on TRAIN, returns delta (including sentinel)
    - server attributes by cosine similarity with s_j using delta_sentinel (or returned sentinel)
    - evolution: selection by usage count, crossover, mutation, keep m fixed
    """

    def __init__(
        self,
        model_ctor,
        num_classes: int,
        device: torch.device,
        m: int = 5,
        seed_evo: int = 777,
    ) -> None:
        self.device = device
        self.m = int(m)
        self.rng = np.random.RandomState(seed_evo)

        # Build initial population with identical weights then different sentinels.
        base_model: nn.Module = model_ctor(num_classes).to(device)
        base_params = param_state_dict(base_model)

        # nu depends on initial fc.weight std
        fc_w = getattr(base_model, "fc").weight.detach()
        nu = 0.005 * float(fc_w.std().item())

        self.sentinels: List[torch.Tensor] = []
        self.pop: List[Dict[str, torch.Tensor]] = []

        for j in range(self.m):
            s_bits = self.rng.choice([-1.0, 1.0], size=(64,)).astype(np.float32)
            s = torch.from_numpy(s_bits).to(device) * nu
            self.sentinels.append(s.detach().clone())

            cand = {k: v.detach().clone() for k, v in base_params.items()}
            cand["sentinel"] = s.detach().clone()  # treat as parameter-only key
            self.pop.append(cand)

        self.usage_counts = np.zeros(self.m, dtype=np.int64)

        # One shared model instance used for eval/train by loading params.
        self.model: nn.Module = model_ctor(num_classes).to(device)
        # Attach sentinel param placeholder to satisfy loading.
        if not hasattr(self.model, "sentinel"):
            self.model.sentinel = nn.Parameter(torch.zeros(64, device=device), requires_grad=False)

    def run_round(
        self,
        client_ids: Sequence[int],
        client_train_loaders: Dict[int, torch.utils.data.DataLoader],
        client_val_loaders: Dict[int, torch.utils.data.DataLoader],
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
    ) -> Tuple[float, int]:
        lr, momentum, weight_decay = sgd_cfg

        deltas_by_candidate: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(self.m)]
        uplink = 0
        train_losses: List[float] = []
        usage_round = np.zeros(self.m, dtype=np.int64)

        for cid in client_ids:
            # Evaluate all candidates on val, pick best.
            best_j, _best_loss = self._client_select_best(cid, client_val_loaders[cid])

            usage_round[best_j] += 1

            # Train selected candidate on train.
            local_model = copy.deepcopy(self.model)
            self._load_candidate_into_model(local_model, self.pop[best_j])

            loss = self._local_train(local_model, client_train_loaders[cid], epochs, lr, momentum, weight_decay, seed_train + cid)
            train_losses.append(loss)

            client_params = param_state_dict(local_model)
            # include sentinel (it exists as a Parameter in model)
            client_params["sentinel"] = local_model.sentinel.detach().clone()

            server_params = self.pop[best_j]
            delta = {k: (client_params[k] - server_params[k]) for k in server_params.keys()}
            uplink += uplink_bytes_for_delta(delta)

            # Attribute by cosine on delta_sentinel
            j_attr = self._attribute(delta["sentinel"])
            deltas_by_candidate[j_attr].append(delta)

        # Aggregate per-candidate using deltas attributed to it.
        for j in range(self.m):
            if len(deltas_by_candidate[j]) == 0:
                continue
            self.pop[j] = fedavg_aggregate(self.pop[j], deltas_by_candidate[j])

        self.usage_counts += usage_round

        # Evolve population (selection by usage count)
        self._evolve()

        return float(np.mean(train_losses)) if train_losses else 0.0, uplink

    @torch.no_grad()
    def _client_select_best(self, cid: int, val_loader) -> Tuple[int, float]:
        best_j = 0
        best_loss = float("inf")

        for j in range(self.m):
            self._load_candidate_into_model(self.model, self.pop[j])
            loss = self._eval_loss(self.model, val_loader)
            if loss < best_loss:
                best_loss = loss
                best_j = j
        return best_j, best_loss

    @torch.no_grad()
    def _eval_loss(self, model: nn.Module, loader) -> float:
        model.eval()
        total = 0
        loss_sum = 0.0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            loss_sum += float(loss.item())
            total += int(y.numel())
        return loss_sum / max(1, total)

    def _attribute(self, delta_sentinel: torch.Tensor) -> int:
        # Compare with known sentinel vectors using cosine similarity.
        sims = [float(F.cosine_similarity(delta_sentinel, s, dim=0).item()) for s in self.sentinels]
        return int(np.argmax(sims))

    def _load_candidate_into_model(self, model: nn.Module, cand: Dict[str, torch.Tensor]) -> None:
        # Load named_parameters from cand; plus sentinel.
        with torch.no_grad():
            for k, p in model.named_parameters():
                if k == "sentinel":
                    p.copy_(cand["sentinel"])
                else:
                    p.copy_(cand[k])

    def _local_train(
        self,
        model: nn.Module,
        loader,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        seed: int,
    ) -> float:
        torch.manual_seed(seed)
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

        total_loss = 0.0
        total = 0
        for _ in range(epochs):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                opt.step()
                total_loss += float(loss.item()) * int(y.numel())
                total += int(y.numel())
        return total_loss / max(1, total)

    def _evolve(self) -> None:
        """
        Minimal evolution:
        - keep top by usage count
        - refill rest by crossover + mutation
        """
        m = self.m
        # Rank candidates by usage_counts desc
        order = np.argsort(-self.usage_counts)
        top_k = max(1, m // 2)
        elites = [self.pop[int(order[i])] for i in range(top_k)]
        elite_sentinels = [self.sentinels[int(order[i])] for i in range(top_k)]

        new_pop: List[Dict[str, torch.Tensor]] = []
        new_sentinels: List[torch.Tensor] = []

        # Keep elites as-is
        for i in range(top_k):
            new_pop.append({k: v.detach().clone() for k, v in elites[i].items()})
            new_sentinels.append(elite_sentinels[i].detach().clone())

        # Refill via crossover between random elites, then mutation
        while len(new_pop) < m:
            a = self.rng.randint(0, top_k)
            b = self.rng.randint(0, top_k)
            lam = float(self.rng.uniform(0.4, 0.6))

            child = {}
            for k in elites[a].keys():
                wa = elites[a][k]
                wb = elites[b][k]
                wc = lam * wa + (1.0 - lam) * wb
                # mutation noise
                std = float(wa.std().item())
                noise_std = 0.001 * std if std > 0 else 0.0
                if noise_std > 0:
                    wc = wc + torch.randn_like(wc) * noise_std
                child[k] = wc.detach()

            # Assign a fresh sentinel (unique ID) for the child.
            # Re-sample bits; keep nu scale implicit in stored sentinels.
            base_scale = float(new_sentinels[0].abs().mean().item()) if new_sentinels else float(child["sentinel"].abs().mean().item())
            s_bits = self.rng.choice([-1.0, 1.0], size=(64,)).astype(np.float32)
            s = torch.from_numpy(s_bits).to(self.device) * base_scale
            child["sentinel"] = s.detach().clone()

            new_pop.append(child)
            new_sentinels.append(s.detach().clone())

        self.pop = new_pop
        self.sentinels = new_sentinels
        self.usage_counts = np.zeros(self.m, dtype=np.int64)