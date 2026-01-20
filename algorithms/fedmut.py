# algorithms/fedmut.py
from __future__ import annotations

import copy
import random
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn


StateDict = Dict[str, torch.Tensor]


def _clone_sd(sd: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: v.detach().clone() for k, v in sd.items()}


def _sd_add(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: a[k] + b[k] for k in a.keys()}


def _sd_sub(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor]) -> StateDict:
    return {k: a[k] - b[k] for k in a.keys()}


def _sd_avg(sds: Sequence[Mapping[str, torch.Tensor]]) -> StateDict:
    """
    Aggregation(w_locals, None) equivalent:
    - Average ONLY floating tensors.
    - For non-floating tensors (e.g., BN num_batches_tracked: int64), keep the first value.
    """
    if len(sds) == 0:
        raise ValueError("sds is empty")

    keys = list(sds[0].keys())
    m = len(sds)

    out: StateDict = {}
    for k in keys:
        t0 = sds[0][k]
        if torch.is_floating_point(t0):
            acc = torch.zeros_like(t0)
            for sd in sds:
                acc += sd[k] / m
            out[k] = acc
        else:
            # Keep as-is (int/bool buffers should not be averaged)
            out[k] = t0.detach().clone()
    return out


def _uplink_bytes_of_delta(delta: Mapping[str, torch.Tensor]) -> int:
    total = 0
    for t in delta.values():
        total += int(t.numel() * t.element_size())
    return total


class FedMutRunner:
    """
    FedMut runner wired to your main.py.

    This version intentionally mirrors the HMHelloWorld/FedMut repo logic:

    - Maintain a pool of m local models (w_locals), size m = clients_per_round (K in your main.py)
    - Each round:
        1) For each selected client i:
             load w_locals[i] into model and train locally -> update w_locals[i]
        2) Aggregate: w_glob = average(w_locals)
        3) Compute delta: w_delta = w_glob - w_old
        4) Mutation spread: create next round's w_locals by mutating w_glob with w_delta
           using ctrl_cmd_list built per parameter key.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        device: torch.device,
        alpha_mut: float,
        seed_mut: int,
        mut_acc_rate: float = 0.5,   # repo args.mut_acc_rate
        mut_bound: int = 200,        # repo args.mut_bound (early accel window)
        debug: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.device = device

        # repo naming: args.radius
        self.radius = float(alpha_mut)

        # repo naming: args.mut_acc_rate, args.mut_bound
        self.mut_acc_rate = float(mut_acc_rate)
        self.mut_bound = int(mut_bound)

        self.seed_mut = int(seed_mut)
        self.debug = bool(debug)

        # maintained across rounds
        self._round = 0
        self._w_locals: Optional[List[StateDict]] = None  # size m (=K)
        self._rng = random.Random(self.seed_mut)

    def _train_one_client(
        self,
        *,
        start_sd: StateDict,
        loader: torch.utils.data.DataLoader,
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
        cid: int,
    ) -> Tuple[StateDict, float, int]:
        """
        Train a local client starting from start_sd.
        Returns:
          (final_state_dict, mean_loss, uplink_bytes(delta))
        """
        lr, momentum, weight_decay = sgd_cfg

        # fresh model instance
        net = copy.deepcopy(self.model).to(self.device)
        net.train()
        net.load_state_dict(start_sd, strict=True)

        opt = torch.optim.SGD(
            net.parameters(),
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(weight_decay),
        )
        loss_fn = nn.CrossEntropyLoss()

        # deterministic-ish
        torch.manual_seed(int(seed_train) + int(cid) + 10007 * self._round)

        loss_sum = 0.0
        n_batches = 0

        for _ in range(epochs):
            for xb, yb in loader:
                xb = xb.to(self.device, non_blocking=True)
                yb = yb.to(self.device, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                logits = net(xb)  # ResNet18_CIFAR in your project returns logits
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

                loss_sum += float(loss.detach().item())
                n_batches += 1

        mean_loss = loss_sum / max(1, n_batches)
        final_sd = net.state_dict()

        # uplink bytes based on delta = final - start
        delta = {k: final_sd[k] - start_sd[k] for k in final_sd.keys()}
        uplink = _uplink_bytes_of_delta(delta)

        return _clone_sd(final_sd), mean_loss, uplink

    def _mutation_spread_repo(
        self,
        *,
        iter0: int,
        w_glob: StateDict,
        w_delta: StateDict,
        m: int,
        alpha: float,
        rng: random.Random,
    ) -> List[StateDict]:
        """
        Direct translation of your pasted HMHelloWorld/FedMut `mutation_spread`:

            ctrl_rate = mut_acc_rate * (1 - min(iter/mut_bound,1))
            for each key:
                ctrl_list length m:
                    for i in 0..m/2-1:
                        append 1.0 and (-1.0 + ctrl_rate) in random order
                    shuffle ctrl_list
            for j in 0..m-1:
                w_sub = w_glob
                if not (j==m-1 and m odd): # last one unmutated when m is odd
                    for each key index ind:
                        w_sub[k] = w_sub[k] + w_delta[k]*ctrl_cmd_list[ind][j]*alpha
                w_locals_new.append(w_sub)
        """
        # ctrl_rate schedule (repo)
        frac = min(iter0 * 1.0 / max(1, self.mut_bound), 1.0)
        ctrl_rate = self.mut_acc_rate * (1.0 - frac)

        # ctrl_cmd_list: list over params; each is length m
        ctrl_cmd_list: List[List[float]] = []
        keys = list(w_glob.keys())

        for _k in keys:
            ctrl_list: List[float] = []
            for _ in range(0, int(m / 2)):
                ctrl = rng.random()
                if ctrl > 0.5:
                    ctrl_list.append(1.0)
                    ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
                else:
                    ctrl_list.append(1.0 * (-1.0 + ctrl_rate))
                    ctrl_list.append(1.0)
            rng.shuffle(ctrl_list)
            # NOTE: if m is odd, ctrl_list length is m-1 (because int(m/2)*2)
            # repo then uses last model as "unmutated", so this is OK.
            ctrl_cmd_list.append(ctrl_list)

        w_locals_new: List[StateDict] = []
        for j in range(m):
            w_sub = _clone_sd(w_glob)

            # last one unmutated when odd m
            if not (j == m - 1 and (m % 2 == 1)):
                for ind, k in enumerate(keys):
                    base = w_sub[k]
                    if torch.is_floating_point(base):
                        # mutate only float parameters (weights/bias/bn running stats)
                        w_sub[k] = base + (w_delta[k] * ctrl_cmd_list[ind][j] * alpha)
                    else:
                        # keep int/bool buffers as-is (e.g., num_batches_tracked)
                        w_sub[k] = base

            w_locals_new.append(w_sub)

        return w_locals_new


    def run_round(
        self,
        *,
        client_ids: Sequence[int],
        client_train_loaders: Dict[int, torch.utils.data.DataLoader],
        epochs: int,
        sgd_cfg: Tuple[float, float, float],
        seed_train: int,
    ) -> Tuple[float, int]:
        """
        main.py expected interface.
        """
        self._round += 1
        iter0 = self._round - 1  # repo uses iter starting at 0
        m = len(client_ids)
        if m <= 0:
            raise ValueError("client_ids is empty")

        # initialize w_locals at first call (pool size = m)
        if self._w_locals is None:
            base = _clone_sd(self.model.state_dict())
            self._w_locals = [copy.deepcopy(base) for _ in range(m)]
        else:
            # safety: if schedule K changes (shouldn't in your main.py), re-init
            if len(self._w_locals) != m:
                base = _clone_sd(self.model.state_dict())
                self._w_locals = [copy.deepcopy(base) for _ in range(m)]

        assert self._w_locals is not None

        # keep w_old (pre-round global)
        w_old = _clone_sd(self.model.state_dict())

        # Train selected clients: each slot i uses its own local model state
        client_losses: List[float] = []
        uplink_total = 0

        for i, cid in enumerate(client_ids):
            if cid not in client_train_loaders:
                raise KeyError(f"client_train_loaders missing cid={cid}")

            start_sd = self._w_locals[i]
            loader = client_train_loaders[cid]

            final_sd, loss_i, uplink_i = self._train_one_client(
                start_sd=start_sd,
                loader=loader,
                epochs=epochs,
                sgd_cfg=sgd_cfg,
                seed_train=seed_train,
                cid=cid,
            )

            self._w_locals[i] = final_sd
            client_losses.append(loss_i)
            uplink_total += uplink_i

        # Aggregate (repo Aggregation)
        w_glob = _sd_avg(self._w_locals)
        self.model.load_state_dict(w_glob, strict=True)

        # Compute delta (repo FedSub with weight=1.0)
        w_delta = _sd_sub(w_glob, w_old)

        # Mutation spread to create next round pool
        # Use deterministic round RNG
        round_rng = random.Random(self.seed_mut + 1000003 * self._round)

        self._w_locals = self._mutation_spread_repo(
            iter0=iter0,
            w_glob=w_glob,
            w_delta=w_delta,
            m=m,
            alpha=self.radius,
            rng=round_rng,
        )

        if self.debug:
            # Helpful debug: show ctrl_rate and delta norm
            with torch.no_grad():
                flat = torch.cat([w_delta[k].reshape(-1) for k in w_delta.keys()])
                dnorm = float(torch.norm(flat).item())
                frac = min(iter0 * 1.0 / max(1, self.mut_bound), 1.0)
                ctrl_rate = self.mut_acc_rate * (1.0 - frac)
                print(f"[FEDMUT_REPO_DEBUG] round={self._round} ctrl_rate={ctrl_rate:.6f} delta_norm={dnorm:.6e}")

        train_loss = float(sum(client_losses) / len(client_losses))
        return train_loss, int(uplink_total)
