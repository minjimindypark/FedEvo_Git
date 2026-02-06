
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# absolute imports (works with `python3 main.py`)
from .base import load_state_dict_, get_state_dict, uplink_bytes_for_delta, _reset_bn_running_stats

_EPS = 1e-12


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


def _add_sd(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor], keys: List[str]) -> Dict[str, torch.Tensor]:
    return {k: (base[k].float() + delta[k].float()).to(base[k].dtype) for k in keys}


def _flatten(sd: Dict[str, torch.Tensor], keys: List[str]) -> torch.Tensor:
    vecs = [sd[k].detach().float().flatten().cpu() for k in keys]
    return torch.cat(vecs, dim=0)


@torch.inference_mode()
def _eval_loss_limited(
    model: nn.Module,
    loader,
    *,
    device: torch.device,
    max_batches: Optional[int],
) -> float:
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
    clip_grad_norm: Optional[float] = None,
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
            if clip_grad_norm is not None and float(clip_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(clip_grad_norm))
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


@dataclass
class GAConfig:
    # population size
    m: int = 10

    # warmups (critical)
    warmup_pop_rounds: int = 10        # collapse population to fedavg
    warmup_no_xor_rounds: int = 50     # disable mut/orth entirely
    warmup_no_seedfill_rounds: int = 50
    warmup_entropy_delay: int = 50

    # Selection / anchor smoothing
    usage_ema_beta: float = 0.8
    anti_pop_alpha: float = 1.0
    sel_temp: float = 2.0
    sel_weight_cap: float = 0.25
    sel_weight_floor: float = 0.01

    # FedAvg baseline / stabilizer momentum
    server_momentum: float = 0.9
    client_grad_clip_norm: Optional[float] = None

    # XOR exploration schedule
    xor_policy: str = "alternate"
    entropy_low: float = 0.35
    entropy_high: float = 0.60

    # Rare/bottom selection
    bottom_zeta: float = 0.4
    rare_pool_min: int = 3

    # Mutation operator (rare-only)
    sigma_mut: float = 0.01
    mutate_frac_layers: float = 0.33

    # Orth injection operator (rare-only)
    sigma_orth: float = 0.50
    orth_eps: float = 1e-12

    # Diversity guard
    cos_sim_thresh: float = 0.995
    l2_eps: float = 1e-3
    max_resample: int = 6

    # Selection anti-bias
    select_penalty_fedavg: float = 0.00

    # State representation
    state_mode: str = "float"
    eval_state_mode: str = ""  # defaults to state_mode if empty

    # BN recalibration
    bn_recalib_enabled: bool = True
    bn_recalib_batches: int = 50

    # Fields used by main.py CLI interface
    seed_group_size: int = 0
    rho: float = 0.3
    gamma: float = 1.5
    tau_factor: float = 0.8
    num_interp: int = 4
    num_orth: int = 1
    enable_mutation: bool = True
    enable_orth_injection: bool = True
    warmup_no_orth_rounds: int = 0
    warmup_no_mut_rounds: int = 0

    debug_diag: bool = True


@dataclass
class FedEvoClient:
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
        penalty_by_shown_index: Optional[Dict[int, float]] = None,
    ) -> int:
        model = model_ctor(int(num_classes)).to(device)
        model.eval()

        best_loss = float("inf")
        best_js: List[int] = []
        eps = float(tie_eps)

        for j, cand_state in enumerate(population):
            load_state_dict_(model, cand_state, mode=state_mode)
            loss = _eval_loss_limited(model, self.val_loader, device=device, max_batches=val_batches)
            if penalty_by_shown_index is not None:
                loss = float(loss) + float(penalty_by_shown_index.get(int(j), 0.0))
            if loss < best_loss - eps:
                best_loss = loss
                best_js = [int(j)]
            elif abs(loss - best_loss) <= eps:
                best_js.append(int(j))

        if len(best_js) == 0:
            return 0
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
        clip_grad_norm: Optional[float] = None,
    ) -> Tuple[Dict[str, torch.Tensor], int, float]:
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
            clip_grad_norm=clip_grad_norm,
        )

        local_state = get_state_dict(model, mode=state_mode)
        delta = {k: (local_state[k].float() - selected_state[k].float()).to(local_state[k].dtype) for k in selected_state.keys()}
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
        clip_grad_norm: Optional[float] = None,
        penalty_by_shown_index: Optional[Dict[int, float]] = None,
    ) -> Tuple[int, Dict[str, torch.Tensor], int, float]:
        j_star = self.select_best(
            population,
            model_ctor=model_ctor,
            num_classes=num_classes,
            device=device,
            val_batches=val_batches,
            tie_eps=tie_eps,
            state_mode=state_mode,
            penalty_by_shown_index=penalty_by_shown_index,
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
            clip_grad_norm=clip_grad_norm,
        )
        return int(j_star), delta, int(n), float(train_loss)


class FedEvoRunner:
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
        self.eval_state_mode = str(getattr(self.ga, "eval_state_mode", "")).lower().strip() or self.state_mode

        self._bn_recalib_loader = None
        self._bn_recalib_max_batches = int(getattr(self.ga, "bn_recalib_batches", 50))

        self._server_select_loader = None
        self._server_select_max_batches = 10

        self.theta_base = get_state_dict(base_model, mode=self.state_mode)

        self.state_keys = sorted(list(self.theta_base.keys()))
        self.param_only_keys = sorted([k for k, _ in base_model.named_parameters()])

        self.population: List[Dict[str, torch.Tensor]] = [_clone_state(self.theta_base) for _ in range(int(self.ga.m))]

        self.round_idx = 0
        self.last_usage_counts: List[int] = [0] * int(self.ga.m)
        self.last_entropy: float = 0.0

        self._usage_ema = np.zeros((int(self.ga.m),), dtype=np.float64)
        self._sel_ema = np.zeros((int(self.ga.m),), dtype=np.float64)
        self._mom_stab: Optional[Dict[str, torch.Tensor]] = None

        self.last_theta_fedavg: Optional[Dict[str, torch.Tensor]] = None
        self.last_theta_stab: Optional[Dict[str, torch.Tensor]] = None
        self.last_theta_used: Optional[Dict[str, torch.Tensor]] = None
        self.last_theta_topk: Optional[Dict[str, torch.Tensor]] = None
        self.last_theta_prev: Optional[Dict[str, torch.Tensor]] = None

        self._deploy_model_cache: Optional[nn.Module] = None
        self._deploy_model_cache_device: Optional[str] = None
        self._deploy_model_cache_num_classes: Optional[int] = None

    # -------- hooks --------
    def set_bn_recalib_loader(self, loader, *, max_batches: Optional[int] = None) -> None:
        self._bn_recalib_loader = loader
        if max_batches is not None:
            self._bn_recalib_max_batches = int(max_batches)

    def set_server_select_loader(self, loader, *, max_batches: Optional[int] = None) -> None:
        self._server_select_loader = loader
        if max_batches is not None:
            self._server_select_max_batches = int(max_batches)

    @torch.inference_mode()
    def _bn_recalibrate(self, model: nn.Module, *, max_batches: int) -> None:
        if self._bn_recalib_loader is None:
            return
        _reset_bn_running_stats(model)
        model.train()
        for b_idx, (x, _) in enumerate(self._bn_recalib_loader):
            if max_batches is not None and b_idx >= int(max_batches):
                break
            x = x.to(self.device, non_blocking=True)
            _ = model(x)
        model.eval()

    # -------- safe deploy --------
    def _get_safe_deploy_sd(self) -> Dict[str, torch.Tensor]:
        # candidates (no test leakage): pick by server_val loss
        cands: List[Tuple[str, Dict[str, torch.Tensor]]] = []
        cands.append(("fedavg", self.last_theta_fedavg or self.theta_base))
        cands.append(("stab", self.last_theta_stab or self.theta_base))
        cands.append(("usage", self.last_theta_used or self.theta_base))
        cands.append(("topk", self.last_theta_topk or self.theta_base))
        cands.append(("prev", self.last_theta_prev or self.last_theta_topk or self.theta_base))

        if self._server_select_loader is None:
            # fallback: topk
            return self.last_theta_topk or self.theta_base

        model = self.model_ctor(self.num_classes).to(self.device)
        best_name = "topk"
        best_loss = float("inf")
        best_sd = self.last_theta_topk or self.theta_base

        for name, sd in cands:
            load_state_dict_(model, sd, mode=self.eval_state_mode)
            # BN recalib if evaluating params without BN stats
            if bool(getattr(self.ga, "bn_recalib_enabled", True)) and self.eval_state_mode == "params":
                self._bn_recalibrate(model, max_batches=self._bn_recalib_max_batches)
            loss = _eval_loss_limited(
                model,
                self._server_select_loader,
                device=self.device,
                max_batches=self._server_select_max_batches,
            )
            if loss < best_loss:
                best_loss = loss
                best_name = name
                best_sd = sd

        if getattr(self.ga, "debug_diag", False):
            print(f"[SAFE DEPLOY] picked={best_name} loss={best_loss:.4f}")

        return best_sd

    def get_deploy_model(self, policy: str = "topk") -> nn.Module:
        pol = (policy or "topk").lower().strip()

        if pol in ("safe", "safeguard"):
            sd = self._get_safe_deploy_sd()
        elif pol in ("fedavg", "fedavg_base"):
            sd = self.last_theta_fedavg or self.theta_base
        elif pol in ("stab", "anchor_stab"):
            sd = self.last_theta_stab or self.theta_base
        elif pol in ("usage", "used", "anchor_usage"):
            sd = self.last_theta_used or self.theta_base
        else:  # topk default
            sd = self.last_theta_topk or self.theta_base

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

        model = self._deploy_model_cache
        load_state_dict_(model, sd, mode=self.eval_state_mode)
        if bool(getattr(self.ga, "bn_recalib_enabled", True)) and self.eval_state_mode == "params":
            self._bn_recalibrate(model, max_batches=self._bn_recalib_max_batches)
        return model

    # -------- evo core --------
    def _choose_xor_mode(self, *, round_idx: int, Hn: float) -> str:
        # hard warmup: no xor
        if int(round_idx) < int(self.ga.warmup_no_xor_rounds):
            return "NONE"
        pol = str(self.ga.xor_policy).lower().strip()
        if pol == "alternate":
            return "MUT" if (int(round_idx) % 2 == 1) else "ORTH"
        if float(Hn) < float(self.ga.entropy_low):
            return "ORTH"
        if float(Hn) > float(self.ga.entropy_high):
            return "MUT"
        return "ORTH" if (int(round_idx) % 2 == 1) else "MUT"

    def _get_rare_pool(self, usage_counts: np.ndarray) -> List[int]:
        m = int(self.ga.m)
        zeta = float(self.ga.bottom_zeta)
        k = int(max(self.ga.rare_pool_min, math.ceil(zeta * m)))
        order = np.argsort(usage_counts.astype(np.int64))
        return [int(j) for j in order[:k]]

    def _mutate_rare(self, sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        layer_names: List[str] = []
        for k in self.param_only_keys:
            ln = k.split(".")[0] if "." in k else k
            if ln not in layer_names:
                layer_names.append(ln)
        n_layers = len(layer_names)
        if n_layers == 0:
            return sd

        frac = float(self.ga.mutate_frac_layers)
        n_pick = int(max(1, math.ceil(frac * n_layers)))

        picked = set()
        perm = torch.randperm(n_layers)
        for i in perm[:n_pick].tolist():
            picked.add(layer_names[int(i)])

        sigma = float(self.ga.sigma_mut)
        for k in self.param_only_keys:
            ln = k.split(".")[0] if "." in k else k
            if ln in picked:
                noise = torch.randn_like(sd[k].float()) * sigma
                sd[k] = (sd[k].float() + noise).to(sd[k].dtype)
        return sd

    def _orth_inject_rare(
        self,
        sd: Dict[str, torch.Tensor],
        theta_stab: Dict[str, torch.Tensor],
        theta_bars: List[Dict[str, torch.Tensor]],
        rare_pool: List[int],
    ) -> Dict[str, torch.Tensor]:
        if len(rare_pool) < 2:
            return sd

        a, b = self.rng.choice(rare_pool, size=2, replace=False).tolist()
        va = _flatten(theta_bars[int(a)], self.param_only_keys)
        vb = _flatten(theta_bars[int(b)], self.param_only_keys)
        vstab = _flatten(theta_stab, self.param_only_keys)

        d = (va - vb)
        d_norm = float(torch.norm(d).item()) + float(self.ga.orth_eps)
        s_norm = float(torch.norm(vstab).item()) + float(self.ga.orth_eps)

        proj = (torch.dot(d, vstab) / (s_norm * s_norm)) * vstab
        orth = d - proj
        o_norm = float(torch.norm(orth).item()) + float(self.ga.orth_eps)

        scale = float(self.ga.sigma_orth) * (d_norm / o_norm)

        ptr = 0
        for k in self.param_only_keys:
            w = sd[k].detach().float().flatten().cpu()
            n = int(w.numel())
            chunk = orth[ptr: ptr + n]
            ptr += n
            w2 = (w + scale * chunk).view_as(sd[k].float())
            sd[k] = w2.to(sd[k].dtype).to(sd[k].device)
        return sd

    def _diversity_guard(
        self,
        cand: Dict[str, torch.Tensor],
        *,
        already: List[Dict[str, torch.Tensor]],
        resample_fn: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, torch.Tensor]:
        cos_thr = float(self.ga.cos_sim_thresh)
        l2_eps = float(self.ga.l2_eps)
        max_try = int(self.ga.max_resample)

        def too_similar(x: Dict[str, torch.Tensor], y: Dict[str, torch.Tensor]) -> bool:
            vx = _flatten(x, self.param_only_keys)
            vy = _flatten(y, self.param_only_keys)
            nx = float(torch.norm(vx).item()) + 1e-12
            ny = float(torch.norm(vy).item()) + 1e-12
            cos = float(torch.dot(vx, vy).item()) / (nx * ny)
            l2 = float(torch.norm(vx - vy).item())
            return (cos > cos_thr) or (l2 < l2_eps)

        out = cand
        for _ in range(max_try + 1):
            if all(not too_similar(out, ref) for ref in already):
                return out
            out = resample_fn(_clone_state(out)) if resample_fn is not None else self._mutate_rare(_clone_state(out))
        return out

    def run_round(
        self,
        clients: Sequence[FedEvoClient],
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
    ) -> Tuple[float, int]:
        lr, momentum, weight_decay = sgd_cfg
        m = int(self.ga.m)
        self.round_idx += 1

        # warmup: collapse population to base fedavg anchor
        if int(self.round_idx) <= int(self.ga.warmup_pop_rounds):
            self.population = [_clone_state(self.theta_base) for _ in range(m)]

        # permutation (hidden index)
        perm = self.rng.permutation(m).tolist()
        inv_perm = [0] * m
        for internal_slot, shown_pos in enumerate(perm):
            inv_perm[int(shown_pos)] = int(internal_slot)

        shown_population = [None] * m
        for internal_slot in range(m):
            shown_pos = perm[internal_slot]
            shown_population[shown_pos] = self.population[internal_slot]

        # selection penalty (optional)
        penalty_by_shown_index: Dict[int, float] = {}
        if float(self.ga.select_penalty_fedavg) > 0:
            fedavg_shown_pos = perm[0]
            penalty_by_shown_index[int(fedavg_shown_pos)] = float(self.ga.select_penalty_fedavg)

        deltas_by_slot: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(m)]
        samples_by_slot: List[List[int]] = [[] for _ in range(m)]
        usage_counts = np.zeros((m,), dtype=np.int64)

        all_deltas_base: List[Dict[str, torch.Tensor]] = []
        all_samples: List[int] = []
        train_losses: List[float] = []
        uplink_bytes = 0

        # warmup: freeze selection to fedavg slot (internal 0)
        freeze_selection = (int(self.round_idx) <= int(self.ga.warmup_no_xor_rounds))

        for client in list(clients):
            cid = int(client.cid)

            if freeze_selection:
                shown_j = perm[0]  # shown position of internal slot 0
                theta_selected = self.population[0]
                # local update from theta_selected
                delta_shown, n_samples, tr_loss = client.local_update(
                    theta_selected,
                    model_ctor=self.model_ctor,
                    num_classes=self.num_classes,
                    device=self.device,
                    epochs=int(epochs),
                    lr=float(lr),
                    momentum=float(momentum),
                    weight_decay=float(weight_decay),
                    seed=int(seed_train) + cid,
                    state_mode=self.state_mode,
                    clip_grad_norm=self.ga.client_grad_clip_norm,
                )
            else:
                shown_j, delta_shown, n_samples, tr_loss = client.run_round(
                    shown_population,
                    model_ctor=self.model_ctor,
                    num_classes=self.num_classes,
                    device=self.device,
                    epochs=int(epochs),
                    lr=float(lr),
                    momentum=float(momentum),
                    weight_decay=float(weight_decay),
                    seed=int(seed_train) + cid,
                    val_batches=self.val_batches,
                    tie_eps=1e-12,
                    state_mode=self.state_mode,
                    clip_grad_norm=self.ga.client_grad_clip_norm,
                    penalty_by_shown_index=penalty_by_shown_index if len(penalty_by_shown_index) > 0 else None,
                )

            train_losses.append(float(tr_loss))
            uplink_bytes += uplink_bytes_for_delta(delta_shown)

            internal_slot = inv_perm[int(shown_j)]
            deltas_by_slot[internal_slot].append(delta_shown)
            samples_by_slot[internal_slot].append(int(n_samples))
            usage_counts[internal_slot] += 1

            # convert to base delta
            theta_sel = self.population[internal_slot]
            delta_base = {k: (delta_shown[k].float() + (theta_sel[k].float() - self.theta_base[k].float())).to(delta_shown[k].dtype) for k in self.state_keys}
            all_deltas_base.append(delta_base)
            all_samples.append(int(n_samples))

        self.last_usage_counts = [int(x) for x in usage_counts.tolist()]

        # theta bars per slot
        theta_bars: List[Dict[str, torch.Tensor]] = []
        for s in range(m):
            if len(deltas_by_slot[s]) == 0:
                theta_bars.append(_clone_state(self.population[s]))
            else:
                weights = [float(n) for n in samples_by_slot[s]] if self.weight_by_samples else None
                avg_delta = _avg_state_dicts(deltas_by_slot[s], self.state_keys, weights=weights)
                theta_bars.append(_add_sd(self.population[s], avg_delta, self.state_keys))

        # entropy
        p = usage_counts.astype(np.float64)
        ssum = float(p.sum())
        p = (np.ones((m,), dtype=np.float64) / float(m)) if ssum <= 0 else (p / ssum)
        p_clip = np.clip(p, 1e-12, 1.0)
        H = float(-np.sum(p_clip * np.log(p_clip)))
        Hmax = float(math.log(m + _EPS))
        Hn = float(H / max(_EPS, Hmax))
        self.last_entropy = float(H)

        # base FedAvg update
        if len(all_deltas_base) == 0:
            avg_delta_base = {k: torch.zeros_like(self.theta_base[k]) for k in self.state_keys}
        else:
            weights = [float(n) for n in all_samples] if self.weight_by_samples else None
            avg_delta_base = _avg_state_dicts(all_deltas_base, self.state_keys, weights=weights)
        theta_fedavg = _add_sd(self.theta_base, avg_delta_base, self.state_keys)

        # stabilizer momentum (disabled in early warmup)
        if int(self.round_idx) <= int(self.ga.warmup_no_xor_rounds):
            theta_stab = _clone_state(theta_fedavg)
        else:
            beta = float(self.ga.server_momentum)
            if self._mom_stab is None:
                self._mom_stab = _clone_state(theta_fedavg)
            else:
                mom = {}
                for k in self.state_keys:
                    mom[k] = (beta * self._mom_stab[k].float() + (1.0 - beta) * theta_fedavg[k].float()).to(theta_fedavg[k].dtype)
                self._mom_stab = mom
            theta_stab = _clone_state(self._mom_stab)

        # base update
        self.theta_base = _clone_state(theta_fedavg if int(self.round_idx) <= int(self.ga.warmup_no_xor_rounds) else theta_stab)
        warm_end = int(self.ga.warmup_no_xor_rounds)

        # warmup 끝난 직후 1회 리셋 (round_idx == warm_end + 1)
        if int(self.round_idx) == warm_end + 1:
            self._usage_ema = np.zeros((m,), dtype=np.float64)
            self._sel_ema   = np.ones((m,), dtype=np.float64) / float(m)
            self.last_usage_counts = [0] * m
            self.last_entropy = 0.0

        # usage/selection EMA
        if int(self.round_idx) > warm_end:
            b = float(self.ga.usage_ema_beta)
            self._usage_ema = b * self._usage_ema + (1.0 - b) * usage_counts.astype(np.float64)
            self._sel_ema   = b * self._sel_ema   + (1.0 - b) * p.astype(np.float64)
        else:
            # warmup 동안은 균등 prior 유지(선택사항)
            self._sel_ema = np.ones((m,), dtype=np.float64) / float(m)


        # anchors: usage weighted & score weighted
        anti = 1.0 / np.power(self._usage_ema + 1.0, float(self.ga.anti_pop_alpha))
        w1 = anti / (float(anti.sum()) + _EPS)
        theta_used = _avg_state_dicts([theta_bars[i] for i in range(m)], self.state_keys, weights=w1.tolist())

        temp = float(self.ga.sel_temp)
        logits = np.log(self._sel_ema + 1e-12) * temp
        logits = logits - float(np.max(logits))
        w2 = np.exp(logits)
        w2 = w2 / (float(w2.sum()) + _EPS)

        cap = float(self.ga.sel_weight_cap)
        floor = float(self.ga.sel_weight_floor)
        if cap > 0:
            w2 = np.minimum(w2, cap)
        if floor > 0:
            w2 = np.maximum(w2, floor)
        w2 = w2 / (float(w2.sum()) + _EPS)

        theta_topk = _avg_state_dicts([theta_bars[i] for i in range(m)], self.state_keys, weights=w2.tolist())
        winner = int(np.argmax(usage_counts)) if usage_counts.size > 0 else 0
        theta_prev = _clone_state(theta_bars[winner])

        self.last_theta_fedavg = _clone_state(theta_fedavg)
        self.last_theta_stab = _clone_state(theta_stab)
        self.last_theta_used = _clone_state(theta_used)
        self.last_theta_topk = _clone_state(theta_topk)
        self.last_theta_prev = _clone_state(theta_prev)

        # hard warmup: keep population anchored (no explore / no xor)
        if int(self.round_idx) <= int(self.ga.warmup_no_xor_rounds):
            self.population = [_clone_state(theta_fedavg) for _ in range(m)]
        else:
            # =================================================================
            # FULL EVOLUTIONARY STEP (Final Production Version + NaN-safe probs)
            # =================================================================

            # 1. Elitism (Anchors)
            next_pop: List[Dict[str, torch.Tensor]] = []
            next_pop.append(_clone_state(theta_fedavg))
            if len(next_pop) < m: next_pop.append(_clone_state(theta_stab))
            if len(next_pop) < m: next_pop.append(_clone_state(theta_used))
            if len(next_pop) < m: next_pop.append(_clone_state(theta_topk))
            if len(next_pop) < m: next_pop.append(_clone_state(theta_prev))

            # -------------------------------------------------------------
            # 2. Crossover (Interpolation)
            # -------------------------------------------------------------
            num_fill = m - len(next_pop)
            mutate_start_idx = len(next_pop)

            child_parent_map: List[int] = []

            if num_fill > 0:
                probs = w2.astype(np.float64)
                probs_sum = float(probs.sum())

                # NaN/Inf/degenerate guard
                if (not np.isfinite(probs_sum)) or probs_sum <= 0.0:
                    probs = np.ones(m, dtype=np.float64) / float(m)
                else:
                    probs = probs / probs_sum

                p_indices = self.rng.choice(m, size=num_fill, p=probs)
                q_indices = self.rng.choice(m, size=num_fill, p=probs)

                for i in range(num_fill):
                    p_idx = int(p_indices[i])
                    q_idx = int(q_indices[i])

                    if p_idx == q_idx:
                        q_idx = int(self.rng.choice(m, p=probs))
                        if p_idx == q_idx:
                            q_idx = int(self.rng.choice(m))

                    child_parent_map.append(p_idx)

                    # base = primary parent clone (keeps BN buffers)
                    child = _clone_state(theta_bars[p_idx])
                    sd_q  = theta_bars[q_idx]


                    lam = float(self.rng.uniform(0.4, 0.6))

                    for k in self.param_only_keys:
                        # KeyError guard
                        if k not in child or k not in sd_q:
                            continue
                        child[k] = (lam * child[k].float() + (1.0 - lam) * sd_q[k].float()).to(child[k].dtype)

                    next_pop.append(child)

            # -------------------------------------------------------------
            # 3. Apply XOR Strategy (Mutation vs Orth)
            # -------------------------------------------------------------
            xor_mode = self._choose_xor_mode(round_idx=self.round_idx, Hn=Hn)

            rare_pool_indices = self._get_rare_pool(usage_counts)
            rare_set = set(int(x) for x in rare_pool_indices)

            target_len = m - mutate_start_idx
            if target_len < 0:
                target_len = 0

            if len(child_parent_map) != target_len:
                child_parent_map = [int(self.rng.choice(m)) for _ in range(target_len)]

            for j in range(mutate_start_idx, m):
                if xor_mode == "NONE":
                    continue

                rel = j - mutate_start_idx
                if rel < 0 or rel >= len(child_parent_map):
                    parent_idx = int(self.rng.choice(m))
                else:
                    parent_idx = int(child_parent_map[rel])

                is_parent_rare = (parent_idx in rare_set)

                apply_prob = 0.60 if is_parent_rare else 0.20
                if self.rng.rand() < apply_prob:
                    if xor_mode == "MUT":
                        self._mutate_rare(next_pop[j])
                    elif xor_mode == "ORTH":
                        self._orth_inject_rare(
                            sd=next_pop[j],
                            theta_stab=theta_stab,
                            theta_bars=theta_bars,
                            rare_pool=rare_pool_indices,
                        )

            # -------------------------------------------------------------
            # 4. Diversity Guard
            # -------------------------------------------------------------
            final_pop: List[Dict[str, torch.Tensor]] = []
            for j in range(m):
                cand = self._diversity_guard(next_pop[j], already=final_pop)
                final_pop.append(cand)

            self.population = final_pop
