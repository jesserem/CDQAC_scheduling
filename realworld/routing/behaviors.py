"""Synthetic behaviour policies for coverage augmentation.

The CDQAC paper showed that state-action *coverage* matters more than
trajectory quality for offline RL in dense-reward scheduling.  Real-world
routing logs are the opposite regime: one expert-like trajectory per instance.
These policies add the same kind of behaviour sources the FJSP datasets have
(random, dispatching-rule-like heuristics, eps-perturbed heuristics) on top of
the *real* instances, so one can study real vs. synthetic behaviour data and
their mixtures (D4RL-style "expert / medium / random / mixed" splits).

Every policy has the env-policy signature ``policy(env, state) -> actions`` and
only reads ``env.open_mask()`` / ``env.travel_out()`` / zone ids, so rollouts
can skip feature construction (``needs_state=False``).

Source names understood by :func:`make_behavior`:

========================  ===================================================
``random``                uniform over open stops
``nn``                    nearest open stop (travel time)
``eps_nn_<eps>``          nearest with prob. 1-eps, uniform otherwise
``softmin_<temp>``        P(stop) ~ exp(-travel / temp)  (temp in minutes)
``zone_nn``               nearest open stop in the current zone, else nearest
``eps_zone_nn_<eps>``     eps-perturbed ``zone_nn``
========================  ===================================================
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
import tqdm

from realworld.episodes import RoutingEpisodeSet
from realworld.routing.core import RewardConfig, RoutingEnv, RoutingTensors


def _sample_uniform(mask: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    w = mask.float() + 1e-12 * (~mask.any(1, keepdim=True)).float()
    return torch.multinomial(w.cpu(), 1, generator=gen).squeeze(1).to(mask.device)


def _nearest(env: RoutingEnv, mask: torch.Tensor) -> torch.Tensor:
    tt = env.travel_out().masked_fill(~mask, float("inf"))
    return tt.argmin(1)


def _zone_mask(env: RoutingEnv, open_: torch.Tensor) -> torch.Tensor:
    zone = env.rt.zone[env.inst, :env.Nb]
    z_cur = zone.gather(1, env.cur[:, None])
    same = open_ & (zone == z_cur) & (z_cur >= 0)
    return torch.where(same.any(1, keepdim=True), same, open_)


def make_behavior(name: str, seed: int = 0) -> Callable:
    gen = torch.Generator().manual_seed(seed)

    def with_eps(base: Callable, eps: float) -> Callable:
        def pol(env, state):
            open_ = env.open_mask()
            greedy = base(env, open_)
            rnd = _sample_uniform(open_, gen)
            coin = torch.rand(open_.shape[0], generator=gen).to(open_.device) < eps
            return torch.where(coin, rnd, greedy)
        return pol

    if name == "random":
        return lambda env, state: _sample_uniform(env.open_mask(), gen)
    if name == "nn":
        return lambda env, state: _nearest(env, env.open_mask())
    if name == "zone_nn":
        return lambda env, state: _nearest(env, _zone_mask(env, env.open_mask()))
    if name.startswith("eps_zone_nn_"):
        eps = float(name[len("eps_zone_nn_"):])
        return with_eps(lambda env, m: _nearest(env, _zone_mask(env, m)), eps)
    if name.startswith("eps_nn_"):
        eps = float(name[len("eps_nn_"):])
        return with_eps(_nearest, eps)
    if name.startswith("softmin_"):
        temp = float(name[len("softmin_"):])

        def pol(env, state):
            open_ = env.open_mask()
            logits = (-env.travel_out() / temp).masked_fill(~open_, float("-inf"))
            probs = torch.softmax(logits, 1).nan_to_num(0.0)
            return torch.multinomial(probs.cpu() + 1e-12, 1, generator=gen).squeeze(1).to(open_.device)
        return pol
    raise ValueError(f"unknown behaviour {name!r}")


def generate_behavior(eps: RoutingEpisodeSet, source: str, per_instance: int = 1,
                      batch_size: int = 512, device: str = "cpu", seed: int = 0,
                      rt: Optional[RoutingTensors] = None, instances: Optional[np.ndarray] = None) -> int:
    """Roll out behaviour ``source`` ``per_instance`` times on every instance and
    append the trajectories to ``eps`` (completion times are left unlogged, so
    they are recomputed from the travel model when replayed).

    Returns the number of trajectories added.
    """
    rt = rt or RoutingTensors(eps, device=device)
    env = RoutingEnv(rt, RewardConfig())
    inst_all = np.arange(eps.num_instances) if instances is None else np.asarray(instances)
    added = 0
    for rep in range(per_instance):
        policy = make_behavior(source, seed=seed + 7919 * rep)
        for s in tqdm.tqdm(range(0, len(inst_all), batch_size), desc=f"{source} #{rep}", leave=False):
            inst = torch.as_tensor(inst_all[s:s + batch_size], dtype=torch.long, device=rt.device)
            res = env.rollout(inst, policy, needs_state=False)
            seqs = res["sequence"].cpu().numpy()
            n = eps.n_nodes[inst.cpu().numpy()]
            actions = [seqs[b, :n[b] - 1] for b in range(len(n))]
            eps.add_trajectories(inst.cpu().numpy(), actions, source)
            added += len(actions)
    return added
