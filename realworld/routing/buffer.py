"""Offline buffer for routing datasets with lazy state reconstruction.

The FJSP pipeline replays every action list through the environment and stores
the full state tensors of every transition.  For routing that would cost
N x features floats per transition (hundreds of GB for Amazon), but a routing
state is fully determined by (instance, visited prefix, clock), so this buffer
stores only

* the instances (once, as tensors -- see :class:`RoutingTensors`),
* per trajectory: the action sequence, the rank of every node in it and the
  decision clock of every step (computed once by :func:`replay_trajectories`),
* per transition: (trajectory, step), reward, done and the discount factor,

and rebuilds state / next-state tensors for each sampled batch.  Batches have
exactly the ``cdqac.buffer.buffer.Buffer`` formats::

    sample / epoch_generator          -> (state, next_state, actions, rewards, dones, mc_returns)
    sample_td3 / epoch_generator_td3  -> (state, next_state, actions, next_actions, rewards, dones, mc_returns)

and every sampler additionally returns the per-transition discount ``[B, 1]``
(``gamma ** (tau / sojourn_unit)`` for SMDP datasets, ``gamma`` otherwise),
which the training loop hands to the trainer (see ``train_realworld.py``).
"""
from __future__ import annotations

from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm

from realworld.episodes import RoutingEpisodeSet
from realworld.routing.core import RewardConfig, RoutingTensors, replay_trajectories

NEVER = 32767


class RoutingBuffer:
    def __init__(self, eps: RoutingEpisodeSet, reward: RewardConfig,
                 sources: Optional[Sequence[str]] = None, time_source: str = "logged",
                 gamma: float = 1.0, smdp_discount: bool = False, sojourn_unit: float = 1.0,
                 device: str = "cpu", replay_chunk: int = 4096, max_trajectories: Optional[int] = None,
                 seed: int = 0):
        """
        Args:
            eps: episode set (instances + trajectories).
            reward: reward weights.
            sources: trajectory sources to include (None = all); prefix match with "*".
            time_source: "logged" (realized completion times where the log has
                them) or "model" (travel + service everywhere).
            gamma: discount factor; per decision, or per ``sojourn_unit`` minutes
                when ``smdp_discount``.
            smdp_discount: discount by ``gamma ** (tau / sojourn_unit)`` (SMDP).
            device: where instance tensors and transition indices live.
        """
        self.eps = eps
        self.reward_cfg = reward
        self.device = torch.device(device)
        self.rt = RoutingTensors(eps, device=device)
        self.gamma = gamma
        self.smdp_discount = smdp_discount
        self.sojourn_unit = sojourn_unit
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

        idx = eps.trajectory_indices(sources)
        if max_trajectories is not None and len(idx) > max_trajectories:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_trajectories, replace=False))
        if len(idx) == 0:
            raise ValueError(f"no trajectories for sources={sources} (have {eps.source_names})")
        self.traj_idx = idx
        K = len(idx)
        T = int(eps.traj_len[idx].max())
        dev = self.device
        self.K, self.T = K, T
        self.inst = torch.as_tensor(eps.traj_inst[idx], dtype=torch.long, device=dev)
        self.lengths = torch.as_tensor(eps.traj_len[idx], dtype=torch.long, device=dev)
        acts = torch.as_tensor(eps.traj_actions[idx, :T], dtype=torch.long, device=dev)
        self.actions = acts
        times = torch.as_tensor(eps.traj_times[idx, :T], dtype=torch.float32, device=dev)

        # rank[k, node] = step at which the trajectory serves node (-1 depot, NEVER padding)
        N = eps.max_nodes
        rank = torch.full((K, N), NEVER, dtype=torch.int16, device=dev)
        rank[:, 0] = -1
        steps = torch.arange(T, device=dev).expand(K, T)
        ok = steps < self.lengths[:, None]
        rows = torch.arange(K, device=dev)[:, None].expand(K, T)
        rank[rows[ok], acts[ok]] = steps[ok].to(torch.int16)
        self.rank = rank

        # replay (chunked) -> decision clocks and per-step reward components
        self.clock = torch.zeros(K, T + 1, device=dev)
        reward_kt = torch.zeros(K, T, device=dev)
        sojourn_kt = torch.zeros(K, T, device=dev)
        self.components: Dict[str, torch.Tensor] = {}
        need_zone = reward.zone_leave != 0
        for s in tqdm.tqdm(range(0, K, replay_chunk), desc="Replaying trajectories", leave=False):
            e = min(K, s + replay_chunk)
            out = replay_trajectories(self.rt, self.inst[s:e], acts[s:e], self.lengths[s:e],
                                      times[s:e], time_source=time_source, need_zone_leave=need_zone)
            self.clock[s:e] = out["clock"]
            sojourn_kt[s:e] = out["sojourn"]
            reward_kt[s:e] = reward.combine(out)
            for key in ("travel", "lateness", "zone_leave"):
                self.components.setdefault(key, torch.zeros(K, T, device=dev))[s:e] = out[key]
        self.reward_kt = reward_kt * ok
        self.sojourn_kt = sojourn_kt * ok

        # discounts and discounted returns-to-go
        if smdp_discount:
            disc_kt = torch.pow(torch.tensor(gamma, device=dev), sojourn_kt / sojourn_unit)
        else:
            disc_kt = torch.full((K, T), float(gamma), device=dev)
        self.disc_kt = disc_kt
        rtg = torch.zeros(K, T + 1, device=dev)
        for t in range(T - 1, -1, -1):
            nxt = torch.where(t + 1 < self.lengths, rtg[:, t + 1], torch.zeros(K, device=dev))
            rtg[:, t] = torch.where(ok[:, t], self.reward_kt[:, t] + disc_kt[:, t] * nxt, torch.zeros(K, device=dev))
        self.rtg_kt = rtg[:, :T]

        # flat transition index
        self.tr_k = rows[ok].contiguous()
        self.tr_t = steps[ok].contiguous()
        self.size = int(self.tr_k.shape[0])
        self.curr_size = self.size
        self.ptr = self.size

    def __len__(self) -> int:
        return self.size

    # ------------------------------------------------------------------ states
    def states_at(self, k: torch.Tensor, t: torch.Tensor, normalize: bool = True):
        """State 8-tuple of trajectories ``k`` before step ``t`` (t may equal the length)."""
        inst = self.inst[k]
        Nb = int(self.rt.n_nodes[inst].max())
        served = self.rank[k, :Nb].long() < t[:, None]
        prev = self.actions[k, (t - 1).clamp(min=0)]
        current = torch.where(t > 0, prev, torch.zeros_like(prev))
        clock = self.clock[k, t]
        return self.rt.build(inst, current, served, clock, Nb=Nb, normalize=normalize)

    def _batch(self, j: torch.Tensor, device, td3: bool):
        j = j.to(self.device)
        k, t = self.tr_k[j], self.tr_t[j]
        state = self.states_at(k, t)
        next_state = self.states_at(k, t + 1)
        actions = self.actions[k, t][:, None]
        rewards = self.reward_kt[k, t][:, None]
        dones = (t + 1 >= self.lengths[k]).float()[:, None]
        mc = self.rtg_kt[k, t][:, None]
        disc = self.disc_kt[k, t][:, None]
        mv = (lambda x: tuple(a.to(device, non_blocking=True) for a in x))
        state, next_state = mv(state), mv(next_state)
        actions, rewards, dones, mc, disc = mv((actions, rewards, dones, mc, disc))
        if td3:
            nxt_t = (t + 1).clamp(max=self.T - 1)
            next_actions = self.actions[k, nxt_t].clamp(min=0)[:, None].to(device)
            return (state, next_state, actions, next_actions, rewards, dones, mc), disc
        return (state, next_state, actions, rewards, dones, mc), disc

    # ---------------------------------------------------------------- samplers
    def sample(self, batch_size: int, device="cpu", td3: bool = False):
        j = torch.randint(0, self.size, (batch_size,), generator=self.rng)
        return self._batch(j, device, td3)

    def sample_td3(self, batch_size: int, device="cpu"):
        return self.sample(batch_size, device, td3=True)

    def epoch_generator(self, batch_size: int, device="cpu", td3: bool = False) -> Iterator[Tuple]:
        for j in torch.randperm(self.size, generator=self.rng).split(batch_size):
            yield self._batch(j, device, td3)

    def epoch_generator_td3(self, batch_size: int, device="cpu"):
        return self.epoch_generator(batch_size, device, td3=True)

    # ----------------------------------------------------------- normalization
    def normalize_state(self, num_samples: int = 20000, chunk: int = 1024) -> Dict[str, torch.Tensor]:
        """Estimate feature statistics from a sample of states and enable them."""
        self.rt.set_stats(None)
        n = min(num_samples, self.size)
        j = torch.randperm(self.size, generator=self.rng)[:n].to(self.device)
        states = [self.states_at(self.tr_k[c], self.tr_t[c], normalize=False) for c in j.split(chunk)]
        stats = RoutingTensors.stats_from_states(states)
        self.rt.set_stats(stats)
        return stats

    # ------------------------------------------------------------ diagnostics
    def describe(self) -> Dict:
        """Returns / sojourn statistics per behaviour source (datasheet numbers)."""
        src = torch.as_tensor(self.eps.traj_source[self.traj_idx].astype(np.int64), device=self.device)
        ret = self.reward_kt.sum(1)
        dur = self.sojourn_kt.sum(1)
        out = dict(transitions=self.size, trajectories=self.K, smdp_discount=self.smdp_discount,
                   mean_sojourn_min=float(self.sojourn_kt.sum() / max(1, self.size)))
        per = {}
        for i, name in enumerate(self.eps.source_names):
            m = src == i
            if m.any():
                per[name] = dict(trajectories=int(m.sum()), return_mean=float(ret[m].mean()),
                                 duration_min_mean=float(dur[m].mean()))
        out["per_source"] = per
        return out
