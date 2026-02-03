from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import load_state_dict_, get_state_dict, uplink_bytes_for_delta

_EPS = 1e-12




def _reset_bn_running_stats(model: nn.Module) -> None:
    """Reset BatchNorm running stats to avoid cross-round contamination when reusing a cached model."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()

def _clone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in state.items()}


def _avg_state_dicts(
    sds: List[Dict[str, torch.Tensor]],
    keys: List[str],
    weights: Optional[List[float]] = None,
) -> Dict[str, torch.Tensor]:
    if len(sds) == 0:
        raise ValueError("Cannot average empty list")
    if len(sds) == 1:
        return _clone_state(sds[0])

    if weights is None:
        weights = [1.0 / len(sds)] * len(sds)
    else:
        if len(weights) != len(sds):
            raise ValueError("weights length mismatch")
        tot = float(sum(weights))
        if tot <= 0:
            raise ValueError("weights must sum to positive")
        weights = [float(w) / tot for w in weights]

    out: Dict[str, torch.Tensor] = {}
    ref = sds[0]
    for k in keys:
        acc = torch.zeros_like(ref[k], dtype=torch.float32)
        for sd, w in zip(sds, weights):
            acc += float(w) * sd[k].float()
        out[k] = acc.to(ref[k].dtype)
    return out



@torch.inference_mode()
def _eval_loss_limited(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    max_batches: Optional[int],
) -> float:
    """Evaluate average cross-entropy loss on up to max_batches batches."""
    model.eval()
    total_loss = 0.0
    total_n = 0
    for b_idx, (x, y) in enumerate(loader):
        if max_batches is not None and b_idx >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")
        total_loss += float(loss.item())
        total_n += int(y.numel())
    if total_n == 0:
        return float("inf")
    return float(total_loss / max(1, total_n))


def local_train_sgd(
    *,
    model: nn.Module,
    loader,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> Tuple[float, int]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model.train()
    opt = torch.optim.SGD(
        model.parameters(),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )

    total_loss = 0.0
    total_seen = 0
    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="mean")
            loss.backward()
            opt.step()
            bs = int(y.shape[0])
            total_loss += float(loss.item()) * bs
            total_seen += bs

    try:
        n_k = int(len(loader.dataset))
    except Exception:
        n_k = int(total_seen)

    if total_seen == 0:
        return 0.0, n_k
    return float(total_loss / total_seen), n_k


def _add_sd(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
    return {k: (base[k].float() + delta[k].float()).to(base[k].dtype) for k in keys}


def _flatten(sd: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    vecs = [sd[k].detach().float().flatten().cpu() for k in keys]
    return torch.cat(vecs, dim=0)


@dataclass
class GAConfig:
    m: int = 10
    rho: float = 0.3
    gamma: float = 1.5

    lam_low: float = 0.4
    lam_high: float = 0.6
    tau_factor: float = 0.8
    sigma_mut: float = 0.01
    mutate_frac_layers: float = 0.33
    bottom_zeta: float = 0.2
    retain_rare_unchanged: int = 1

    num_interp: int = 4
    num_orth: int = 1

    use_r_parent_avg: bool = False
    r_parent_choices: Tuple[int, ...] = (3, 4)

    enable_mutation: bool = True
    enable_orth_injection: bool = True
    debug_diag: bool = True

    warmup_no_orth_rounds: int = 20
    warmup_no_mut_rounds: int = 50

    init_noise: float = 0.001
    tie_eps: float = 1e-12
    # Population seeding (paper Eq.(seed))
    # Fill remaining slots in P(t+1) using warm cached deltas grouped by participating clients.
    seed_group_size: int = 5
    enable_seed_fill: bool = True

    # State representation
    # - "params": trainable parameters only (θ ∈ R^P)
    # - "float": parameters + floating buffers (e.g., BN running stats)
    state_mode: str = "params"


@dataclass
class FedEvoClient:
    """A lightweight client container for simulation.

    In a real FL system, the server would broadcast the population and each client
    would locally: (1) evaluate candidates on its private validation split,
    (2) select one candidate index, (3) fine-tune on its private training split,
    (4) upload only (j*, delta). In this simulator we keep loaders attached to the
    client object to avoid passing validation data through the server API.
    """
    cid: int
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader

    def select_best(
        self,
        population: Sequence[Dict[str, torch.Tensor]],
        *,
        model_ctor,
        num_classes: int,
        device: torch.device,
        val_batches: Optional[int],
        tie_eps: float,
        state_mode: str,
    ) -> int:
        """Client-side candidate selection (Eq. (select)).

        Evaluates each candidate on the client's private validation split and returns j*.
        """
        model = model_ctor(int(num_classes)).to(device)
        model.eval()

        best_loss = float("inf")
        best_js: List[int] = []
        eps = float(tie_eps)

        for j, cand_state in enumerate(population):
            load_state_dict_(model, cand_state, mode=state_mode)
            loss = _eval_loss_limited(model, self.val_loader, device=device, max_batches=val_batches)
            if loss < best_loss - eps:
                best_loss = loss
                best_js = [int(j)]
            elif abs(loss - best_loss) <= eps:
                best_js.append(int(j))

        if len(best_js) == 0:
            return 0
        # Deterministic tie-break (paper: epsilon-tolerant, deterministic)
        # Choose the smallest candidate index among ties to ensure reproducibility.
        return int(min(best_js))

    def local_update(
        self,
        selected_state: Dict[str, torch.Tensor],
        *,
        model_ctor,
        num_classes: int,
        device: torch.device,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        seed: int,
        state_mode: str,
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
        """Client-side local training and delta computation (Eq. (delta)).

        Returns (delta, n_samples, avg_train_loss).
        """
        model = model_ctor(int(num_classes)).to(device)
        load_state_dict_(model, selected_state, mode=state_mode)

        loss, n_samples = local_train_sgd(
            model=model,
            loader=self.train_loader,
            epochs=int(epochs),
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
            seed=int(seed),
            device=device,
        )

        local_state = get_state_dict(model, mode=state_mode)
        # Delta is w.r.t. selected candidate state
        delta = {}
        for k in selected_state.keys():
            delta[k] = (local_state[k].float() - selected_state[k].float()).to(local_state[k].dtype)

        return delta, int(n_samples), float(loss)

    def run_round(
        self,
        population: Sequence[Dict[str, torch.Tensor]],
        *,
        model_ctor,
        num_classes: int,
        device: torch.device,
        epochs: int,
        lr: float,
        momentum: float,
        weight_decay: float,
        seed: int,
        val_batches: Optional[int],
        tie_eps: float,
        state_mode: str,
    ) -> Tuple[int, Dict[str, torch.Tensor], int, float]:
        """Full client routine: select → train → return (j*, delta)."""
        j_star = self.select_best(
            population,
            model_ctor=model_ctor,
            num_classes=num_classes,
            device=device,
            val_batches=val_batches,
            tie_eps=tie_eps,
            state_mode=state_mode,
        )
        delta, n, train_loss = self.local_update(
            population[int(j_star)],
            model_ctor=model_ctor,
            num_classes=num_classes,
            device=device,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            seed=seed,
            state_mode=state_mode,
        )
        return int(j_star), delta, int(n), float(train_loss)

class FedEvoRunner:
    def _sample_two_parents(self, p):
        """Sample two DISTINCT parent indices according to probability vector p.

        Uses self.rng for reproducibility (seed_evo).
        """
        import numpy as np

        p = np.asarray(p, dtype=np.float64).copy()
        m = int(p.shape[0])
        if m < 2:
            raise ValueError(f"Need at least 2 candidates, got m={m}")

        p[~np.isfinite(p)] = 0.0
        p[p < 0.0] = 0.0
        s = float(p.sum())
        p = (np.ones(m, dtype=np.float64) / m) if s <= 0.0 else (p / s)

        # sample first parent
        p_idx = int(self.rng.choice(m, p=p))

        # sample second parent without replacement
        p2 = p.copy()
        p2[p_idx] = 0.0
        s2 = float(p2.sum())
        if s2 <= 0.0:
            p2 = np.ones(m, dtype=np.float64)
            p2[p_idx] = 0.0
            p2 /= float(p2.sum())
        else:
            p2 /= s2

        q_idx = int(self.rng.choice(m, p=p2))
        return p_idx, q_idx

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
        self.state_mode = str(self.ga.state_mode).lower().strip()
        self.theta_base = get_state_dict(base_model, mode=self.state_mode)

        self.state_keys = sorted(list(self.theta_base.keys()))
        self.param_only_keys = sorted([k for k, _ in base_model.named_parameters()])

        self.population: List[Dict[str, torch.Tensor]] = []
        self.population.append(_clone_state(self.theta_base))
        for _ in range(int(self.ga.m) - 1):
            cand = _clone_state(self.theta_base)
            if float(self.ga.init_noise) > 0.0:
                self._add_param_noise_inplace(cand, sigma_scale=float(self.ga.init_noise))
            self.population.append(cand)

        self._scratch_model: nn.Module = model_ctor(self.num_classes).to(self.device)

        self.round_idx = 0
        self.last_usage_counts: List[int] = [0] * int(self.ga.m)
        self.last_entropy: float = 0.0

        # Warm-start delta cache (sentinel-free, relative to theta_base of the round they were produced in).
        # Used for population seeding described in the paper (Eq.(seed)).
        self._delta_cache_by_cid: Dict[int, Tuple[Dict[str, torch.Tensor], int]] = {}
        self._delta_cache_fifo: List[Tuple[Dict[str, torch.Tensor], int]] = []
        self._delta_cache_max: int = int(self.ga.m) * int(self.ga.seed_group_size) * 5

    def _add_param_noise_inplace(self, sd: Dict[str, torch.Tensor], sigma_scale: float) -> None:
        for k in self.param_only_keys:
            w = sd[k]
            std = float(w.float().std().item())
            if not np.isfinite(std) or std <= 0.0:
                std = 1e-4
            sigma = float(sigma_scale) * std
            noise = torch.randn_like(w.float()) * sigma
            sd[k] = (w.float() + noise).to(w.dtype)


    def get_deploy_model(self, policy: str = "topk") -> nn.Module:
        """Return an nn.Module for evaluation/deployment.

        IMPORTANT: main.py expects an nn.Module because bn_recalibrate/evaluate call .train()/.eval().

        policy:
          - base: theta_base
          - stab: last stabilizer (if available)
          - used: last usage anchor (if available)
          - topk: last top-k aggregated candidate (default, if available)
          - best: best post-trained candidate from last round by usage_counts
        """
        import numpy as _np

        pol = (policy or "topk").lower()

        # Choose state_dict source (ALL are sentinel-free state_dicts)
        if pol == "base":
            sd = self.theta_base
        elif pol == "stab":
            sd = getattr(self, "last_theta_stab", None) or getattr(self, "theta_stab_last", None) or self.theta_base
        elif pol == "used":
            sd = getattr(self, "last_theta_used", None) or getattr(self, "theta_used_last", None) or self.theta_base
        elif pol == "topk":
            sd = getattr(self, "last_theta_topk", None) or getattr(self, "theta_topk_last", None) or self.theta_base
        elif pol == "best":
            bars = getattr(self, "last_theta_bars", None) or getattr(self, "theta_bars_last", None)
            usage = getattr(self, "last_usage_counts", None) or getattr(self, "usage_counts_last", None)
            if bars is not None and usage is not None and len(bars) == len(usage) and len(bars) > 0:
                j = int(_np.argmax(_np.asarray(usage, dtype=_np.int64)))
                sd = bars[j]
            else:
                sd = getattr(self, "last_theta_topk", None) or getattr(self, "theta_topk_last", None) or self.theta_base
        else:
            sd = getattr(self, "last_theta_topk", None) or getattr(self, "theta_topk_last", None) or self.theta_base

        # Reuse a cached model instance to avoid per-round construction overhead.


        # IMPORTANT: We still (a) reset BN running stats, then (b) load the latest params each call.


        if not hasattr(self, "_deploy_model_cache"):


            self._deploy_model_cache = None


            self._deploy_model_cache_num_classes = None


            self._deploy_model_cache_device = None


        dev_str = str(self.device)


        need_new = (


            self._deploy_model_cache is None


            or self._deploy_model_cache_num_classes != self.num_classes


            or self._deploy_model_cache_device != dev_str


        )


        if need_new:


            self._deploy_model_cache = self.model_ctor(self.num_classes).to(self.device)


            self._deploy_model_cache_num_classes = self.num_classes


            self._deploy_model_cache_device = dev_str


        model: nn.Module = self._deploy_model_cache


        _reset_bn_running_stats(model)


        load_state_dict_(model, sd, mode=self.state_mode)


        return model
    def run_round(
        self,
        clients: Sequence[FedEvoClient],
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
    ) -> Tuple[float, int]:
        lr, momentum, weight_decay = sgd_cfg
        self.round_idx += 1

        St = list(clients)
        m = int(self.ga.m)

        deltas_by_j: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(m)]
        samples_by_j: List[List[int]] = [[] for _ in range(m)]
        usage_counts = np.zeros((m,), dtype=np.int64)

        all_deltas_cand: List[Dict[str, torch.Tensor]] = []
        all_deltas_base: List[Dict[str, torch.Tensor]] = []
        all_samples: List[int] = []

        train_losses: List[float] = []
        uplink_bytes = 0

        # Collect per-client deltas for warm-start cache update (used by seeding).
        round_abs_by_cid: Dict[int, Tuple[Dict[str, torch.Tensor], int]] = {}

        for client in St:
            cid = int(client.cid)

            # Client routine (paper Alg. 1): local selection + local training; server receives only (j*, delta).
            j_star, delta_cand, n_samples, tr_loss = client.run_round(
                self.population,
                model_ctor=self.model_ctor,
                num_classes=self.num_classes,
                device=self.device,
                epochs=int(epochs),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                seed=int(seed_train) + cid,
                val_batches=self.val_batches,
                tie_eps=float(self.ga.tie_eps),
                state_mode=self.state_mode,
            )
            train_losses.append(float(tr_loss))
            uplink_bytes += uplink_bytes_for_delta(delta_cand)

            deltas_by_j[j_star].append(delta_cand)
            samples_by_j[j_star].append(int(n_samples))
            usage_counts[j_star] += 1

            all_deltas_cand.append(delta_cand)

            delta_base = {k: (delta_cand[k].float() + (self.population[j_star][k].float() - self.theta_base[k].float())).to(delta_cand[k].dtype) for k in self.state_keys}
            all_deltas_base.append(delta_base)
            all_samples.append(int(n_samples))
            theta_abs = {k: (self.population[j_star][k].float() + delta_cand[k].float()).to(self.population[j_star][k].dtype) for k in self.state_keys}
            round_abs_by_cid[cid] = (theta_abs, int(n_samples))

        # Update warm-start cache with latest (theta_abs, n_samples) per participating client.
        # NOTE: We cache absolute client post-local states to avoid coordinate-frame drift when the base changes across rounds.
        for cid, (theta_abs, n_samp) in round_abs_by_cid.items():
            self._delta_cache_by_cid[int(cid)] = (_clone_state(theta_abs), int(n_samp))
            self._delta_cache_fifo.append((_clone_state(theta_abs), int(n_samp)))
        # Keep cache bounded (FIFO).
        if self._delta_cache_max <= 0:
            # Guard: if seed_group_size=0 (or any config yields 0), disable FIFO cache to avoid unbounded growth.
            self._delta_cache_fifo.clear()
        elif len(self._delta_cache_fifo) > self._delta_cache_max:
            self._delta_cache_fifo = self._delta_cache_fifo[-self._delta_cache_max:]


        self.last_usage_counts = usage_counts.tolist()

        theta_bars: List[Dict[str, torch.Tensor]] = []
        for j in range(m):
            if len(deltas_by_j[j]) == 0:
                theta_bars.append(_clone_state(self.population[j]))
            else:
                if self.weight_by_samples:
                    weights = [float(n) for n in samples_by_j[j]]
                    avg_delta = _avg_state_dicts(deltas_by_j[j], self.state_keys, weights=weights)
                else:
                    avg_delta = _avg_state_dicts(deltas_by_j[j], self.state_keys, weights=None)
                theta_bars.append(_add_sd(self.population[j], avg_delta, self.state_keys))

        p = usage_counts.astype(np.float64)
        s = float(p.sum())
        if s <= 0:
            p = np.ones((m,), dtype=np.float64) / float(m)
        else:
            p = p / s
        p_clip = np.clip(p, 1e-12, 1.0)
        H = float(-np.sum(p_clip * np.log(p_clip)))
        self.last_entropy = float(H)

        tau = float(self.ga.tau_factor) * math.log(m + _EPS)

        if len(all_deltas_base) == 0:
            avg_delta_base = {k: torch.zeros_like(self.theta_base[k]) for k in self.state_keys}
        else:
            if self.weight_by_samples:
                avg_delta_base = _avg_state_dicts(all_deltas_base, self.state_keys, weights=[float(n) for n in all_samples])
            else:
                avg_delta_base = _avg_state_dicts(all_deltas_base, self.state_keys, weights=None)

        theta_stab = _add_sd(self.theta_base, avg_delta_base, self.state_keys)

        if len(all_deltas_cand) == 0:
            avg_delta_cand = {k: torch.zeros_like(self.theta_base[k]) for k in self.state_keys}
        else:
            if self.weight_by_samples:
                avg_delta_cand = _avg_state_dicts(all_deltas_cand, self.state_keys, weights=[float(n) for n in all_samples])
            else:
                avg_delta_cand = _avg_state_dicts(all_deltas_cand, self.state_keys, weights=None)

        used_indices = [j for j in range(m) if usage_counts[j] > 0]
        theta_used = _clone_state(theta_stab) if len(used_indices) == 0 else _avg_state_dicts([theta_bars[j] for j in used_indices], self.state_keys, weights=None)

        k_star = max(1, int(math.floor(float(self.ga.rho) * m)))
        order_desc = np.argsort(-usage_counts)
        top_k_indices = [int(j) for j in order_desc[:k_star]]

        w = np.power(usage_counts[top_k_indices].astype(np.float64) + 1e-8, float(self.ga.gamma))
        w = w / (float(w.sum()) + _EPS)

        theta_topk: Dict[str, torch.Tensor] = {}
        for key in self.state_keys:
            acc = torch.zeros_like(theta_bars[0][key], dtype=torch.float32)
            for wi, j in zip(w, top_k_indices):
                acc += float(wi) * theta_bars[j][key].float()
            theta_topk[key] = acc.to(theta_bars[0][key].dtype)

        interps: List[Dict[str, torch.Tensor]] = []
        for _ in range(int(self.ga.num_interp)):
            if self.ga.use_r_parent_avg:
                r = int(self.rng.choice(list(self.ga.r_parent_choices)))
                parents = self._sample_parents(p, r=r)
                interps.append(_avg_state_dicts([theta_bars[j] for j in parents], self.state_keys, weights=[1.0 / r] * r))
            else:
                p_idx, q_idx = self._sample_two_parents(p)
                lam = float(self.rng.uniform(self.ga.lam_low, self.ga.lam_high))
                interps.append({k: (lam * theta_bars[p_idx][k].float() + (1.0 - lam) * theta_bars[q_idx][k].float()).to(theta_bars[p_idx][k].dtype) for k in self.state_keys})

        enable_orth_now = bool(self.ga.enable_orth_injection) and (int(self.round_idx) > int(self.ga.warmup_no_orth_rounds))

        orths: List[Dict[str, torch.Tensor]] = []
        if enable_orth_now and int(self.ga.num_orth) > 0:
            stab_vec = _flatten(theta_stab, self.param_only_keys)
            stab_norm = float(torch.norm(stab_vec).item()) + 1e-12

            for _ in range(int(self.ga.num_orth)):
                j_perp = 0
                min_cos = 1.0
                for j in range(m):
                    vec_j = _flatten(theta_bars[j], self.param_only_keys)
                    cos = float(torch.dot(vec_j, stab_vec).item()) / (float(torch.norm(vec_j).item()) + 1e-12) / stab_norm
                    if cos < min_cos:
                        min_cos = cos
                        j_perp = j

                theta_orth = _add_sd(theta_bars[j_perp], avg_delta_cand, self.state_keys)
                orths.append(theta_orth)

        next_pop: List[Dict[str, torch.Tensor]] = []
        next_pop.append(_clone_state(theta_stab))
        next_pop.append(_clone_state(theta_used))
        next_pop.append(_clone_state(theta_topk))
        next_pop.extend([_clone_state(x) for x in interps])
        next_pop.extend([_clone_state(x) for x in orths])

        # Population seeding (paper Eq.(seed)): fill remaining slots using warm cached deltas grouped by clients.
        if bool(self.ga.enable_seed_fill):
            # Use the stabilizer as the reference base for seeding P(t+1).
            while len(next_pop) < m and len(self._delta_cache_fifo) > 0:
                cand = self._seed_one_candidate_from_warm_cache(base_sd=theta_stab, participants=St)
                if cand is None:
                    break
                next_pop.append(cand)

        while len(next_pop) < m:
            p_idx, q_idx = self._sample_two_parents(p)
            lam = float(self.rng.uniform(self.ga.lam_low, self.ga.lam_high))
            next_pop.append({k: (lam * theta_bars[p_idx][k].float() + (1.0 - lam) * theta_bars[q_idx][k].float()).to(theta_bars[p_idx][k].dtype) for k in self.state_keys})

        next_pop = next_pop[:m]

        enable_mut_now = bool(self.ga.enable_mutation) and (int(self.round_idx) > int(self.ga.warmup_no_mut_rounds))
        mut_src: List[int] = []
        if enable_mut_now and (H < tau):
            mutants, mut_src = self._make_mutants_from_rare(theta_bars, usage_counts)

            # Replace tail slots (excluding anchors 0..2) so mutants are never silently dropped.
            num_mut = min(len(mutants), max(0, m - 3))
            slots = list(range(m - 1, m - 1 - num_mut, -1))
            for s, mu in zip(slots, mutants[:num_mut]):
                next_pop[s] = _clone_state(mu)

            if num_mut > 0:
                print(f"[FedEvo R{self.round_idx}] mutation_from_rare={mut_src[:num_mut]} -> slots={slots}")

        self.population = next_pop
        old_base = _clone_state(self.theta_base)  # debug: keep previous base for logging

        self.theta_base = _clone_state(theta_stab)
        # Cache round products for deployment/evaluation
        self.last_usage_counts = [int(x) for x in usage_counts.tolist()]
        self.last_entropy = float(H)
        self.last_theta_stab = _clone_state(theta_stab)
        self.last_theta_used = _clone_state(theta_used)
        self.last_theta_topk = _clone_state(theta_topk)
        self.last_theta_bars = [_clone_state(sd) for sd in theta_bars]


        if getattr(self.ga, "debug_diag", False):
            stab_update_norm = float(torch.norm(_flatten(avg_delta_base, self.param_only_keys)).item())
            cand_update_norm = float(torch.norm(_flatten(avg_delta_cand, self.param_only_keys)).item())
            base_norm = float(torch.norm(_flatten(old_base, self.param_only_keys)).item())
            stab_norm = float(torch.norm(_flatten(theta_stab, self.param_only_keys)).item())
            top1 = int(np.argmax(usage_counts)) if usage_counts.size > 0 else -1

            print(
                f"[DIAG R{self.round_idx}] ||base||={base_norm:.3e} ||stab||={stab_norm:.3e} "
                f"||avgΔ_base||={stab_update_norm:.3e} ||avgΔ_cand||={cand_update_norm:.3e} top1={top1}"
            )

        print(
            f"[FedEvo R{self.round_idx}] H={H:.3f} (τ={tau:.3f}) usage={usage_counts.tolist()} "
            f"orth={'Y' if enable_orth_now else 'N'} mut={'Y' if (enable_mut_now and H < tau) else 'N'}"
        )

        mean_loss = float(np.mean(train_losses)) if train_losses else 0.0
        return mean_loss, int(uplink_bytes)

    def _seed_one_candidate_from_warm_cache(
        self,
        *,
        base_sd: Dict[str, torch.Tensor],
        participants: Sequence[FedEvoClient],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Seed one candidate from warm cache.

        NOTE: The cache stores *absolute* states (theta_abs) of past client-local models.
        To avoid coordinate drift across rounds (changing bases), we rebase each cached
        theta_abs into the current base: delta_curr = theta_abs - base_sd, then average
        deltas and add back to base_sd.
        """
        k = int(self.ga.seed_group_size)
        if k <= 0:
            return None

        # Pick a group of participants (prefer current round participants when possible).
        if len(participants) > 0:
            take = min(k, len(participants))
            pick = self.rng.choice(len(participants), size=take, replace=False)
            group = [participants[int(i)] for i in pick]
        else:
            group = []

        deltas: List[Dict[str, torch.Tensor]] = []
        weights: List[float] = []

        # 1) Prefer per-client cache (cid-indexed) using current participants
        for c in group:
            cid = int(c.cid)
            if cid in self._delta_cache_by_cid:
                theta_abs, n = self._delta_cache_by_cid[cid]
                # Rebase to current base
                delta_curr = {
                    key: (theta_abs[key].float() - base_sd[key].float()).to(base_sd[key].dtype)
                    for key in self.state_keys
                }
                deltas.append(delta_curr)
                weights.append(float(n))

        # 2) Fallback: sample from FIFO cache if no usable per-cid entries were found
        if len(deltas) == 0 and len(self._delta_cache_fifo) > 0:
            take = min(k, len(self._delta_cache_fifo))
            pick = self.rng.choice(len(self._delta_cache_fifo), size=take, replace=False)
            for pi in pick:
                theta_abs, n = self._delta_cache_fifo[int(pi)]
                delta_curr = {
                    key: (theta_abs[key].float() - base_sd[key].float()).to(base_sd[key].dtype)
                    for key in self.state_keys
                }
                deltas.append(delta_curr)
                weights.append(float(n))

        if len(deltas) == 0:
            return None

        avg_d = _avg_state_dicts(
            deltas,
            self.state_keys,
            weights=weights if bool(self.weight_by_samples) else None,
        )
        cand = _add_sd(_clone_state(base_sd), avg_d, self.state_keys)
        return cand


    def _make_mutants_from_rare(
        self,
        theta_bars: List[Dict[str, torch.Tensor]],
        usage_counts,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
        """Bottom-diversify:
        - Choose rare candidates (least-used) from usage_counts
        - Mutate COPIES of those rare candidates (post-trained theta_bars[j])
        - Return mutants and their source indices for logging
        """
        if len(theta_bars) == 0:
            return [], []
        if len(usage_counts) != len(theta_bars):
            usage_counts = list(usage_counts)[: len(theta_bars)] + [0] * max(0, len(theta_bars) - len(usage_counts))

        m = len(theta_bars)
        usage = np.asarray(usage_counts, dtype=np.int64)

        # ---- pick rare set (bottom_zeta fraction) ----
        zeta = float(self.ga.bottom_zeta)
        num_rare = int(max(1, math.floor(zeta * m)))
        order_asc = np.argsort(usage)  # least-used first
        rare_indices = [int(j) for j in order_asc[:num_rare]]

        # keep the rarest 'retain_rare_unchanged' unchanged (optional)
        keep_k = int(max(0, self.ga.retain_rare_unchanged))
        keep_set = set(rare_indices[:keep_k])

        # how many mutants to propose (upper bounded later by m-3 in run_round)
        # I recommend: one mutant per rare candidate (excluding kept ones)
        sigma = float(self.ga.sigma_mut)

        # layer grouping (same as your top version)
        layer_names: List[str] = []
        for k in self.param_only_keys:
            ln = k.split(".")[0] if "." in k else k
            if ln not in layer_names:
                layer_names.append(ln)

        frac = float(self.ga.mutate_frac_layers)
        n_layers = len(layer_names)
        n_pick = int(max(1, math.ceil(frac * n_layers))) if n_layers > 0 else 0

        mutants: List[Dict[str, torch.Tensor]] = []
        mut_src: List[int] = []

        for j in rare_indices:
            if j in keep_set:
                continue

            src = theta_bars[j]          # ✅ rare candidate itself (post-trained)
            mstate = _clone_state(src)   # mutate COPY

            if n_layers > 0:
                perm = torch.randperm(n_layers)
                picked = {layer_names[int(i)] for i in perm[:n_pick].tolist()}
            else:
                picked = set()

            for k in self.param_only_keys:
                ln = k.split(".")[0] if "." in k else k
                if ln in picked:
                    noise = torch.randn_like(mstate[k].float()) * sigma
                    mstate[k] = (mstate[k].float() + noise).to(mstate[k].dtype)

            mutants.append(mstate)
            mut_src.append(int(j))

        # Defensive: if everything got "kept unchanged", force at least 1 mutant from the rarest
        if len(mutants) == 0 and len(rare_indices) > 0:
            j = int(rare_indices[0])
            src = theta_bars[j]
            mstate = _clone_state(src)
            # mutate all param keys lightly (fallback)
            for k in self.param_only_keys:
                noise = torch.randn_like(mstate[k].float()) * sigma
                mstate[k] = (mstate[k].float() + noise).to(mstate[k].dtype)
            mutants.append(mstate)
            mut_src.append(j)

        return mutants, mut_src
