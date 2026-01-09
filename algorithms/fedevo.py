"""
FedEvo: Genetic Candidate Evolution in Federated Learning
via Implicit Client Feedback

Key invariants
- A: pop_raw is always sentinel-free and is the only state persisted across rounds.
- B: Evolution uses pop_raw as base; sentinels are only for attribution.

Round flow
1) Start: pop_sent = clone(pop_raw) → refresh + embed sentinels on pop_sent.
2) Clients select/train on pop_sent; deltas are w.r.t. pop_sent.
3) Server attribution uses sentinels; θ̄_j = pop_raw[j] + avg_delta_j (sentinel-free base).
4) Evolution/mutation produce new sentinel-free pop_raw.
5) Persist pop_raw only; next round repeats with fresh sentinels.

Implementation notes (paper-unspecified choices)
- Low-sensitivity: keyword heuristic (bias/bn/norm/running). Fallback: fc.weight → all params.
- Population seeding: configurable (seed_every_round=False for continuity, True for paper-strict).
- Interp parents: top-k by usage (design choice).
- Soft elitism: bottom-ζ inserted first; only these are protected.
- Mutation: targets slots whose source_id is in bottom 50% usage; synthetic excluded.
- Sentinel: additive (θ[I_j] += w_j); ν clamped to [nu_min, nu_max].
- Stabilizer: sample-weighted average of local final params (FedAvg style, default).
- Attribution logging: round & cumulative accuracy + margin (mean, p50, p90).
- Perf options: val_batches, recompute_nu_each_round, orth_subset_ratio, reuse_local_model.
- Ablation switches: enable_sentinels, enable_mutation, enable_orth_injection.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import (
    load_param_state_dict_,
    param_state_dict,
    uplink_bytes_for_delta,
)


@dataclass
class CandidateSlot:
    params: Dict[str, torch.Tensor]
    source_id: Optional[int]
    is_protected: bool


def _l2_score(delta_slice: torch.Tensor, w_j: torch.Tensor) -> float:
    # Paper-style attribution scoring (example: ||delta + w||^2)
    return float((delta_slice + w_j).norm().pow(2).item())


def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat, b_flat = a.flatten().float(), b.flatten().float()
    na, nb = a_flat.norm().item(), b_flat.norm().item()
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0), dim=1).item())


def _vec(params: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    return torch.cat([params[k].flatten().float() for k in keys])


def _avg_params(
    params_list: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Weighted average of parameter state dicts.
    - If weights is None: uniform mean.
    - If weights provided: normalized to sum to 1.0.
    """
    if len(params_list) == 0:
        raise ValueError("Cannot average empty list")
    if len(params_list) == 1:
        return {k: v.clone() for k, v in params_list[0].items()}

    if weights is None:
        weights = [1.0 / len(params_list)] * len(params_list)
    else:
        if len(weights) != len(params_list):
            raise ValueError("weights length must match params_list length")
        total = float(sum(weights))
        if total <= 0.0:
            raise ValueError("weights must sum to a positive value")
        weights = [float(w) / total for w in weights]

    result: Dict[str, torch.Tensor] = {}
    keys = list(params_list[0].keys())
    for k in keys:
        weighted_sum = torch.zeros_like(params_list[0][k], dtype=torch.float32)
        for p, w in zip(params_list, weights):
            weighted_sum += w * p[k].float()
        result[k] = weighted_sum.to(params_list[0][k].dtype)
    return result


def _add_params(
    base: Dict[str, torch.Tensor],
    delta: Dict[str, torch.Tensor],
    scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    return {
        k: (base[k].float() + scale * delta[k].float()).to(base[k].dtype)
        for k in base.keys()
    }


def _clone_params(params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in params.items()}


def _get_low_sensitivity_keys(params: Dict[str, torch.Tensor]) -> List[str]:
    low_sens_keys: List[str] = []
    for key in params.keys():
        kl = key.lower()
        if any(pat in kl for pat in ["bias", "bn", "norm", "running"]):
            low_sens_keys.append(key)
    return low_sens_keys


class FedEvoRunner:
    def __init__(
        self,
        model_ctor,
        num_classes: int,
        device: torch.device,
        m: int = 10,
        k: int = 5,
        rho: float = 0.3,
        gamma: float = 1.5,
        d: int = 64,
        nu_scale: float = 0.005,
        tau_factor: float = 0.8,
        zeta: int = 1,
        sigma_mut: float = 0.01,
        seed_evo: int = 777,
        nu_min: float = 1e-6,
        nu_max: float = 1e-3,
        val_batches: Optional[int] = None,
        recompute_nu_each_round: bool = False,
        orth_subset_ratio: float = 1.0,
        resample_orth_keys_each_round: bool = False,
        reuse_local_model: bool = False,
        deterministic: bool = False,
        weight_by_samples: bool = True,
        seed_every_round: bool = False,
        enable_sentinels: bool = True,
        enable_mutation: bool = True,
        enable_orth_injection: bool = True,
        feedback_log_path: Optional[str] = None,
    ) -> None:
        self.device = device
        self.m = int(m)
        self.k = int(k)
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.d = int(d)
        self.nu_scale = float(nu_scale)
        self.nu_min = float(nu_min)
        self.nu_max = float(nu_max)
        self.tau = float(tau_factor) * math.log(max(2, self.m))
        self.zeta = int(zeta)
        self.sigma_mut = float(sigma_mut)

        self.rng = np.random.RandomState(int(seed_evo))
        self.model_ctor = model_ctor
        self.num_classes = num_classes
        self.val_batches = val_batches
        self.recompute_nu_each_round = recompute_nu_each_round
        self.orth_subset_ratio = float(orth_subset_ratio)
        self.resample_orth_keys_each_round = resample_orth_keys_each_round
        self.reuse_local_model = reuse_local_model
        self.deterministic = deterministic
        self.weight_by_samples = weight_by_samples
        self.seed_every_round = seed_every_round
        self.enable_sentinels = enable_sentinels
        self.enable_mutation = enable_mutation
        self.enable_orth_injection = enable_orth_injection

        if self.deterministic:
            # Determinism flags (still not perfect across all ops, but much better)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Fix seeds globally for reproducibility
            random.seed(seed_evo)
            np.random.seed(seed_evo)
            torch.manual_seed(seed_evo)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_evo)

        base_model: nn.Module = model_ctor(num_classes).to(device)
        self.base_params = param_state_dict(base_model)
        self.param_keys = sorted(self.base_params.keys())

        self._init_orth_keys()

        self.low_sens_keys = _get_low_sensitivity_keys(self.base_params)
        self._build_low_sens_index_pool()

        self.theta_base: Dict[str, torch.Tensor] = _clone_params(self.base_params)
        self._compute_nu(self.theta_base)

        self.pop_raw: List[Dict[str, torch.Tensor]] = [
            _clone_params(self.theta_base) for _ in range(self.m)
        ]
        self.pop_sent: Optional[List[Dict[str, torch.Tensor]]] = None

        self.cached_deltas: List[Dict[str, torch.Tensor]] = []
        self.sentinel_indices: List[List[Tuple[str, int]]] = []
        self.sentinel_values: List[torch.Tensor] = []

        self._refresh_sentinels()

        self.model: nn.Module = model_ctor(num_classes).to(device)
        self._local_model: Optional[nn.Module] = None
        if self.reuse_local_model:
            self._local_model = model_ctor(num_classes).to(device)

        self.round_idx = 0

        self.attribution_correct_total = 0
        self.attribution_total_total = 0

    def _init_orth_keys(self) -> None:
        num_orth_keys = max(1, int(len(self.param_keys) * self.orth_subset_ratio))
        if self.orth_subset_ratio < 1.0:
            indices = self.rng.choice(len(self.param_keys), size=num_orth_keys, replace=False)
            self.orth_keys = [self.param_keys[i] for i in sorted(indices)]
        else:
            self.orth_keys = self.param_keys.copy()

    def _resample_orth_keys(self) -> None:
        if self.orth_subset_ratio < 1.0:
            num_orth_keys = max(1, int(len(self.param_keys) * self.orth_subset_ratio))
            indices = self.rng.choice(len(self.param_keys), size=num_orth_keys, replace=False)
            self.orth_keys = [self.param_keys[i] for i in sorted(indices)]

    def _compute_nu(self, ref_params: Dict[str, torch.Tensor]) -> None:
        if len(self.low_sens_pool) == 0:
            self.nu = self.nu_min
            return

        values: List[float] = []
        for key, idx in self.low_sens_pool[: min(1000, len(self.low_sens_pool))]:
            if key in ref_params:
                w = ref_params[key].flatten()
                if idx < w.numel():
                    values.append(float(w[idx].item()))

        if len(values) > 0:
            std = float(np.std(values))
            raw_nu = self.nu_scale * std
        else:
            raw_nu = self.nu_scale * 0.01

        self.nu = float(np.clip(raw_nu, self.nu_min, self.nu_max))

    def _build_low_sens_index_pool(self) -> None:
        self.low_sens_pool: List[Tuple[str, int]] = []
        for key in self.low_sens_keys:
            if key in self.base_params:
                numel = self.base_params[key].numel()
                self.low_sens_pool.extend([(key, i) for i in range(numel)])

        need = self.m * self.d
        if len(self.low_sens_pool) >= need:
            return

        fc_key = None
        for key in self.param_keys:
            kl = key.lower()
            if "fc" in kl and "weight" in kl:
                fc_key = key
                break

        if fc_key and self.base_params[fc_key].numel() >= need:
            warnings.warn(f"Low-sens pool insufficient; using {fc_key} as fallback.")
            self.low_sens_pool = [(fc_key, i) for i in range(self.base_params[fc_key].numel())]
            return

        warnings.warn("Low-sens pool insufficient; using ALL params.")
        self.low_sens_pool = []
        for key in self.param_keys:
            self.low_sens_pool.extend([(key, i) for i in range(self.base_params[key].numel())])

    def _refresh_sentinels(self) -> None:
        if not self.enable_sentinels:
            self.sentinel_indices = [[] for _ in range(self.m)]
            self.sentinel_values = [torch.tensor([], device=self.device) for _ in range(self.m)]
            return

        shuffled = self.low_sens_pool.copy()
        self.rng.shuffle(shuffled)

        self.sentinel_indices = []
        self.sentinel_values = []
        for j in range(self.m):
            start = j * self.d
            end = (j + 1) * self.d
            I_j = shuffled[start:end] if end <= len(shuffled) else shuffled[start:]
            self.sentinel_indices.append(I_j)

            bits = self.rng.choice([-1.0, 1.0], size=(len(I_j),)).astype(np.float32)
            w_j = torch.from_numpy(bits).to(self.device) * float(self.nu)
            self.sentinel_values.append(w_j)

    def _embed_sentinel(self, j: int) -> None:
        if self.pop_sent is None or not self.enable_sentinels:
            return
        I_j = self.sentinel_indices[j]
        w_j = self.sentinel_values[j]
        with torch.no_grad():
            for idx, (key, flat_idx) in enumerate(I_j):
                if key in self.pop_sent[j] and idx < len(w_j):
                    w = self.pop_sent[j][key]
                    w_flat = w.reshape(-1)
                    if flat_idx < w_flat.numel():
                        w_flat[flat_idx] += w_j[idx].to(w_flat.dtype)

    def _embed_all_sentinels(self) -> None:
        if self.pop_sent is None or not self.enable_sentinels:
            return
        for j in range(self.m):
            self._embed_sentinel(j)

    def _extract_delta_slice(self, delta: Dict[str, torch.Tensor], j: int) -> torch.Tensor:
        I_j = self.sentinel_indices[j]
        values: List[float] = []
        for key, flat_idx in I_j:
            if key in delta:
                d = delta[key].reshape(-1)
                values.append(float(d[flat_idx].item()) if flat_idx < d.numel() else 0.0)
            else:
                values.append(0.0)
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _attribute(self, delta: Dict[str, torch.Tensor]) -> Tuple[int, float]:
        if not self.enable_sentinels:
            # Ablation: sentinel off => attribution collapses (all routed to 0)
            return 0, float("inf")

        scores: List[float] = []
        for j in range(self.m):
            delta_slice = self._extract_delta_slice(delta, j)
            w_j = self.sentinel_values[j]
            min_len = min(delta_slice.numel(), w_j.numel())
            if min_len == 0:
                scores.append(float("inf"))
            else:
                score = _l2_score(delta_slice[:min_len], w_j[:min_len])
                scores.append(score)

        best_j = int(np.argmin(scores))
        sorted_scores = sorted(scores)
        margin = float("inf") if len(sorted_scores) < 2 else float(sorted_scores[1] - sorted_scores[0])
        return best_j, margin

    def _seed_population(self) -> None:
        warm_deltas = self.cached_deltas.copy()
        if len(warm_deltas) == 0:
            self.pop_raw = [_clone_params(self.theta_base) for _ in range(self.m)]
            return

        self.rng.shuffle(warm_deltas)
        new_pop: List[Dict[str, torch.Tensor]] = []
        for j in range(self.m):
            start_idx = j * self.k
            end_idx = min((j + 1) * self.k, len(warm_deltas))
            if start_idx >= len(warm_deltas):
                new_pop.append(_clone_params(self.theta_base))
            else:
                group_deltas = warm_deltas[start_idx:end_idx]
                avg_delta = _avg_params(group_deltas)
                new_pop.append(_add_params(self.theta_base, avg_delta))
        self.pop_raw = new_pop

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
        self.round_idx += 1
        St = list(client_ids)

        # Seeding policy (paper-strict vs continuity)
        if self.seed_every_round or self.round_idx == 1:
            self._seed_population()

        if self.recompute_nu_each_round:
            self._compute_nu(self.theta_base)

        if self.resample_orth_keys_each_round:
            self._resample_orth_keys()

        # Sentinel round init
        self.pop_sent = [_clone_params(p) for p in self.pop_raw]
        self._refresh_sentinels()
        self._embed_all_sentinels()

        all_deltas: List[Dict[str, torch.Tensor]] = []
        all_local_params: List[Dict[str, torch.Tensor]] = []
        all_sample_counts: List[int] = []

        S_j: List[Set[int]] = [set() for _ in range(self.m)]
        train_losses: List[float] = []
        uplink_bytes = 0
        client_to_delta: Dict[int, Dict[str, torch.Tensor]] = {}

        round_correct = 0
        round_total = 0
        round_margins: List[float] = []

        for cid in St:
            cid_int = int(cid)
            j_star = self._client_select_best(client_val_loaders[cid_int])

            if self.reuse_local_model and self._local_model is not None:
                local_model = self._local_model
            else:
                local_model = self.model_ctor(self.num_classes).to(self.device)

            load_param_state_dict_(local_model, self.pop_sent[j_star])

            loss, num_samples = self._local_train(
                local_model,
                client_train_loaders[cid_int],
                epochs=epochs,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                seed=seed_train + cid_int,
            )
            train_losses.append(loss)
            all_sample_counts.append(num_samples)

            local_params = param_state_dict(local_model)
            all_local_params.append(local_params)

            # Delta w.r.t. SENTINEL-EMBEDDED candidate (client trained on pop_sent[j_star])
            delta = {
                k: (local_params[k].float() - self.pop_sent[j_star][k].float()).to(local_params[k].dtype)
                for k in self.param_keys
            }
            uplink_bytes += uplink_bytes_for_delta(delta)

            all_deltas.append(delta)
            client_to_delta[cid_int] = delta

            j_attr, margin = self._attribute(delta)
            S_j[j_attr].add(cid_int)

            round_total += 1
            round_margins.append(margin)
            if j_attr == j_star:
                round_correct += 1

        self.attribution_correct_total += round_correct
        self.attribution_total_total += round_total

        # θ̄_j computed on sentinel-free base pop_raw[j]
        theta_bars: List[Dict[str, torch.Tensor]] = []
        for j in range(self.m):
            if len(S_j[j]) == 0:
                theta_bars.append(_clone_params(self.pop_raw[j]))
            else:
                deltas_j = [client_to_delta[k] for k in S_j[j]]
                avg_delta = _avg_params(deltas_j)
                theta_bars.append(_add_params(self.pop_raw[j], avg_delta))

        total = max(1, sum(len(s) for s in S_j))
        p_j = np.array([len(S_j[j]) / total for j in range(self.m)], dtype=np.float64)
        H_t = float(-np.sum(p_j[p_j > 0] * np.log(p_j[p_j > 0] + 1e-12)))
        usage_counts = np.array([len(s) for s in S_j], dtype=np.int64)

        # Stabilizer: average of local final params (optionally sample-weighted)
        if len(all_local_params) > 0:
            if self.weight_by_samples:
                theta_stab = _avg_params(all_local_params, weights=[float(n) for n in all_sample_counts])
            else:
                theta_stab = _avg_params(all_local_params)
        else:
            theta_stab = _clone_params(self.theta_base)

        used_bars = [theta_bars[j] for j in range(self.m) if len(S_j[j]) > 0]
        theta_used = _avg_params(used_bars) if len(used_bars) > 0 else _clone_params(theta_stab)

        self._evolve(theta_bars, theta_stab, theta_used, p_j, H_t, usage_counts, all_deltas)

        # Persist only sentinel-free base
        self.theta_base = _clone_params(theta_stab)

        # Cache deltas for population seeding
        self.cached_deltas = all_deltas.copy()
        max_cache = self.m * self.k
        if len(self.cached_deltas) > max_cache:
            self.cached_deltas = self.cached_deltas[-max_cache:]

        self.pop_sent = None

        # Logging
        attr_acc_round = round_correct / max(1, round_total)
        attr_acc_cum = self.attribution_correct_total / max(1, self.attribution_total_total)

        if round_margins:
            margin_arr = np.array(round_margins, dtype=np.float64)
            margin_mean = float(np.mean(margin_arr))
            margin_p50 = float(np.percentile(margin_arr, 50))
            margin_p90 = float(np.percentile(margin_arr, 90))
        else:
            margin_mean = margin_p50 = margin_p90 = 0.0

        print(
            f"[FedEvo R{self.round_idx}] H={H_t:.3f} τ={self.tau:.3f} "
            f"|S_j|={[len(s) for s in S_j]} "
            f"mut={'Y' if (H_t < self.tau and self.enable_mutation) else 'N'} "
            f"attr_r={attr_acc_round:.3f} attr_c={attr_acc_cum:.3f} "
            f"margin(mean/p50/p90)={margin_mean:.4f}/{margin_p50:.4f}/{margin_p90:.4f}"
        )

        return float(np.mean(train_losses)) if train_losses else 0.0, int(uplink_bytes)

    def _evolve(
        self,
        theta_bars: List[Dict[str, torch.Tensor]],
        theta_stab: Dict[str, torch.Tensor],
        theta_used: Dict[str, torch.Tensor],
        p_j: np.ndarray,
        H_t: float,
        usage_counts: np.ndarray,
        all_deltas: List[Dict[str, torch.Tensor]],
    ) -> None:
        order_asc = np.argsort(usage_counts)

        new_slots: List[CandidateSlot] = []
        capacity = self.m

        # Soft elitism: protect bottom-ζ usage candidates
        for i in range(min(self.zeta, len(order_asc))):
            j_rare = int(order_asc[i])
            new_slots.append(
                CandidateSlot(_clone_params(theta_bars[j_rare]), source_id=j_rare, is_protected=True)
            )

        remaining = capacity - len(new_slots)

        # Inject stabilizer & used-average candidates
        if remaining > 0:
            new_slots.append(CandidateSlot(_clone_params(theta_stab), source_id=None, is_protected=False))
            remaining -= 1

        if remaining > 0:
            new_slots.append(CandidateSlot(_clone_params(theta_used), source_id=None, is_protected=False))
            remaining -= 1

        # Top-k weighted average
        if remaining > 0:
            k_star = max(1, int(self.rho * self.m))
            top_k_indices = np.argsort(-usage_counts)[:k_star]
            weights = np.power(usage_counts[top_k_indices].astype(np.float64) + 1e-8, self.gamma)
            weights = weights / (weights.sum() + 1e-12)

            theta_topk: Dict[str, torch.Tensor] = {}
            for key in self.param_keys:
                weighted_sum = torch.zeros_like(theta_bars[0][key], dtype=torch.float32)
                for wi, j in enumerate(top_k_indices):
                    weighted_sum += float(weights[wi]) * theta_bars[int(j)][key].float()
                theta_topk[key] = weighted_sum.to(theta_bars[0][key].dtype)

            new_slots.append(CandidateSlot(theta_topk, source_id=None, is_protected=False))
            remaining -= 1

        # Stochastic interpolation (parents restricted to top-by-usage set)
        num_interp = max(1, remaining // 2)
        parent_pool = np.argsort(-usage_counts)[: max(2, int(self.rho * self.m))]

        for _ in range(num_interp):
            if remaining <= 0:
                break

            probs = p_j[parent_pool]
            if probs.sum() > 0:
                probs = probs / (probs.sum() + 1e-12)
            else:
                probs = np.ones(len(parent_pool), dtype=np.float64) / len(parent_pool)

            p_idx = int(self.rng.choice(parent_pool, p=probs))
            q_idx = int(self.rng.choice(parent_pool, p=probs))
            lam = float(self.rng.uniform(0.4, 0.6))

            child: Dict[str, torch.Tensor] = {}
            for key in self.param_keys:
                wp = theta_bars[p_idx][key].float()
                wq = theta_bars[q_idx][key].float()
                child[key] = (lam * wp + (1.0 - lam) * wq).to(theta_bars[p_idx][key].dtype)

            new_slots.append(CandidateSlot(child, source_id=None, is_protected=False))
            remaining -= 1

        # Orthogonal injection (optional)
        if remaining > 0 and self.enable_orth_injection:
            stab_vec = _vec(theta_stab, self.orth_keys)
            min_cos, j_orth = float("inf"), 0

            for j in range(self.m):
                bar_vec = _vec(theta_bars[j], self.orth_keys)
                cos = _cosine_similarity(bar_vec, stab_vec)
                if cos < min_cos:
                    min_cos, j_orth = cos, j

            if len(all_deltas) > 0:
                delta_bar = _avg_params(all_deltas)
                theta_orth = _add_params(theta_bars[j_orth], delta_bar, scale=1.0)
            else:
                theta_orth = _clone_params(theta_bars[j_orth])

            new_slots.append(CandidateSlot(theta_orth, source_id=j_orth, is_protected=False))
            remaining -= 1

        # Fill remaining with random interpolations
        while remaining > 0:
            p_idx = int(self.rng.randint(0, self.m))
            q_idx = int(self.rng.randint(0, self.m))
            lam = float(self.rng.uniform(0.4, 0.6))

            child: Dict[str, torch.Tensor] = {}
            for key in self.param_keys:
                wp = theta_bars[p_idx][key].float()
                wq = theta_bars[q_idx][key].float()
                child[key] = (lam * wp + (1.0 - lam) * wq).to(theta_bars[p_idx][key].dtype)

            new_slots.append(CandidateSlot(child, source_id=None, is_protected=False))
            remaining -= 1

        new_slots = new_slots[:capacity]

        # Mutation on low-usage sourced slots only
        if H_t < self.tau and self.enable_mutation:
            self._apply_mutation(new_slots, usage_counts)

        self.pop_raw = [slot.params for slot in new_slots]

    def _apply_mutation(self, slots: List[CandidateSlot], usage_counts: np.ndarray) -> None:
        all_keys = self.param_keys.copy()
        num_layers_to_mutate = max(1, len(all_keys) // 3)

        m = int(len(usage_counts))
        num_low = max(1, int(math.ceil(m * 0.5)))
        low_freq_ids = set(np.argsort(usage_counts)[:num_low].tolist())

        for slot in slots:
            if slot.is_protected:
                continue
            if slot.source_id is None:
                continue
            if slot.source_id not in low_freq_ids:
                continue

            layers_to_mutate = self.rng.choice(
                all_keys,
                size=min(num_layers_to_mutate, len(all_keys)),
                replace=False,
            )
            for key in layers_to_mutate:
                w = slot.params[key]
                std = float(w.float().std().item())
                sigma = float(self.sigma_mut * std) if std > 0 else 1e-4
                noise = torch.randn_like(w.float()) * sigma
                slot.params[key] = (w.float() + noise).to(w.dtype)

    @torch.inference_mode()
    def _client_select_best(self, val_loader) -> int:
        best_j, best_loss = 0, float("inf")
        for j in range(self.m):
            load_param_state_dict_(self.model, self.pop_sent[j])
            loss = self._eval_loss_limited(self.model, val_loader)
            if loss < best_loss:
                best_loss = loss
                best_j = j
        return best_j

    @torch.inference_mode()
    def _eval_loss_limited(self, model: nn.Module, loader) -> float:
        model.eval()
        total_loss, total_samples = 0.0, 0
        batch_count = 0
        for x, y in loader:
            if self.val_batches is not None and batch_count >= self.val_batches:
                break
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total_loss += float(loss.item())
            total_samples += int(y.numel())
            batch_count += 1
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
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        total_loss, total_samples = 0.0, 0
        for _ in range(epochs):
            for x, y in loader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()

                bs = int(y.numel())
                total_loss += float(loss.item()) * bs
                total_samples += bs

        return total_loss / max(1, total_samples), total_samples

    def get_best_model(self) -> nn.Module:
        load_param_state_dict_(self.model, self.theta_base)
        return self.model

    def get_attribution_accuracy(self) -> float:
        return self.attribution_correct_total / max(1, self.attribution_total_total)
