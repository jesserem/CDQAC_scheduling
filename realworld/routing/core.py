"""Batched routing core: state construction, trajectory replay and rollout env.

Everything that turns a (partial) route into network inputs lives here, and it
is shared by the offline buffer (which rebuilds states lazily from logged
action sequences) and the evaluation env (which rolls policies out), so the
two can never disagree on what a state is.

Decision process (both datasets)
--------------------------------
A decision is made at clock ``d_t`` at the current node; the action is the next
stop.  Serving stop ``a`` finishes at ``f_t = d_t + travel(cur, a) + service(a)``
(or at the *logged* completion time when the log has one).  The next decision
epoch is ``d_{t+1} = f_t`` unless no stop is open, in which case the clock jumps
to the next release (a new request arriving).  For replayed logs the rule is
``d_{t+1} = max(f_t, release[a_{t+1}])``: the latest decision time that is
consistent with the logged next action (see realworld/README.md).  The SMDP
sojourn of step ``t`` is ``tau_t = d_{t+1} - d_t`` (``f_t - d_t`` on the last
step), so sum_t tau_t is the episode duration.

State layout (8-tuple, positions chosen to mirror the FJSP state so that every
method in ``cdqac.methods`` works unchanged: the feasibility mask is slot 6 and
``get_embedding`` receives slots 0-5 and 7)::

    0 node_feat   [B, N, Fn] float  static + dynamic node features
    1 node_valid  [B, N]     bool   real (non-padding) nodes
    2 current     [B]        long   current node (depot = 0)
    3 global_feat [B, Fg]    float  context features
    4 node_open   [B, N]     bool   candidate stops (released and unserved)
    5 zone        [B, N]     long   instance-local zone id (-1 = none)
    6 action_mask [B, N]     bool   True = infeasible action
    7 edge_feat   [B, N, Fe] float  current -> node relational features
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch

from realworld.episodes import RoutingEpisodeSet

STATE_FIELDS = ("node_feat", "node_valid", "current", "global_feat", "node_open", "zone",
                "action_mask", "edge_feat")

GENERIC_STATIC_NAMES = ["x_rel", "y_rel", "log_dist_depot", "sin_angle", "cos_angle", "is_depot",
                        "has_tw", "tw_start_h", "tw_end_h", "log_service", "release_h",
                        "log_tt_from_depot", "log_tt_to_depot", "zone_size_frac", "zone2_size_frac"]
DYNAMIC_NODE_NAMES = ["zone_open_frac", "zone2_open_frac", "slack_h", "age_h"]
EDGE_NAMES = ["log_tt_out", "log_tt_in", "rank_tt_out", "same_zone", "same_zone2", "rel_tt_out",
              "log_dist_km"]
GLOBAL_NAMES = ["clock_h", "sin_tod", "cos_tod", "frac_served", "frac_open", "frac_unreleased",
                "log_n", "log_mean_tt_open", "cur_zone_has_open", "cur_is_depot"]

NO_TW_SLACK_H = 8.0
EPS_TIME = 1e-4


@dataclass
class RewardConfig:
    """Weights of the per-step cost; reward = -(cost) * scale + bias.

    All weights are "minutes-equivalent", so e.g. ``zone_leave = 10`` means that
    leaving a zone that still has open stops costs as much as 10 minutes.
    """
    time: float = 1.0          # sojourn tau_t: travel + service + idle time
    travel: float = 0.0        # pure travel time of the leg (model)
    lateness: float = 0.0      # minutes past the stop's time-window end
    zone_leave: float = 0.0    # leaving a zone that still has open stops (backtracking proxy)
    latency: float = 0.0       # tau_t x remaining stops -> sum of completion times
    return_to_depot: bool = True
    scale: float = 1.0 / 60.0  # minutes -> hours
    bias: float = 0.0

    def combine(self, c: Dict[str, torch.Tensor]) -> torch.Tensor:
        cost = (self.time * c["sojourn"] + self.travel * c["travel"] + self.lateness * c["lateness"]
                + self.latency * c["sojourn"] * c["remaining"])
        if self.zone_leave:
            cost = cost + self.zone_leave * c["zone_leave"]
        if self.return_to_depot:
            cost = cost + (self.time + self.travel) * c["return_travel"]
        return -cost * self.scale + self.bias


class RoutingTensors:
    """Instance arrays of an episode set as tensors, plus feature construction."""

    def __init__(self, eps: RoutingEpisodeSet, device: str = "cpu",
                 travel_dtype: torch.dtype = torch.float32):
        self.device = torch.device(device)
        self.name = eps.name

        def t(a, dt=torch.float32):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=dt, device=self.device)

        self.E, self.N = eps.num_instances, eps.max_nodes
        self.n_nodes = t(eps.n_nodes, torch.long)
        self.coords = t(eps.coords)
        self.zone = t(eps.zone, torch.long)
        self.zone2 = t(eps.zone2, torch.long)
        self.release = t(eps.release)
        self.tw_start = t(eps.tw_start)
        self.tw_end = t(eps.tw_end)
        self.service = t(eps.service)
        self.speed = t(eps.speed).clamp(min=1e-3)
        self.start_tod = t(eps.start_tod)
        self.inst_global = t(eps.inst_global)
        self.detour = float(eps.detour)
        self.matrix = eps.travel_mode == "matrix"
        if self.matrix:
            self.travel_flat = t(eps.travel_flat, travel_dtype)
            self.travel_off = t(eps.travel_offsets, torch.long)
        self.arange_n = torch.arange(self.N, device=self.device)
        self.valid = self.arange_n[None, :] < self.n_nodes[:, None]
        self.dynamic = bool((eps.release > 0).any())
        self.node_names = GENERIC_STATIC_NAMES + list(eps.node_feat_names) + DYNAMIC_NODE_NAMES
        self.edge_names = list(EDGE_NAMES)
        self.global_names = GLOBAL_NAMES + list(eps.glob_feat_names)
        self.static = self._static_features(t(eps.node_feats))
        self.stats: Optional[Dict[str, torch.Tensor]] = None

    # ------------------------------------------------------------------ dims
    @property
    def node_dim(self) -> int:
        return len(self.node_names)

    @property
    def edge_dim(self) -> int:
        return len(self.edge_names)

    @property
    def global_dim(self) -> int:
        return len(self.global_names)

    # ---------------------------------------------------------------- travel
    def travel_from(self, inst: torch.Tensor, src: torch.Tensor, Nb: int) -> torch.Tensor:
        """Travel minutes from node ``src[b]`` to every node, shape [B, Nb] (0 on padding)."""
        if self.matrix:
            n = self.n_nodes[inst]
            j = self.arange_n[:Nb]
            ok = j[None, :] < n[:, None]
            idx = self.travel_off[inst][:, None] + src[:, None] * n[:, None] + j[None, :]
            out = self.travel_flat[torch.where(ok, idx, torch.zeros_like(idx))].float()
            return torch.where(ok, out, torch.zeros_like(out))
        c = self.coords[inst, :Nb]
        s = c.gather(1, src.view(-1, 1, 1).expand(-1, 1, 2))
        return self.detour * (c - s).norm(dim=-1) / self.speed[inst][:, None]

    def travel_to(self, inst: torch.Tensor, dst: torch.Tensor, Nb: int) -> torch.Tensor:
        """Travel minutes from every node to ``dst[b]``, shape [B, Nb]."""
        if self.matrix:
            n = self.n_nodes[inst]
            j = self.arange_n[:Nb]
            ok = j[None, :] < n[:, None]
            idx = self.travel_off[inst][:, None] + j[None, :] * n[:, None] + dst[:, None]
            out = self.travel_flat[torch.where(ok, idx, torch.zeros_like(idx))].float()
            return torch.where(ok, out, torch.zeros_like(out))
        return self.travel_from(inst, dst, Nb)

    def travel_pair(self, inst: torch.Tensor, i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
        """Travel minutes from ``i[b]`` to ``j[b]``, shape [B]."""
        if self.matrix:
            n = self.n_nodes[inst]
            return self.travel_flat[self.travel_off[inst] + i * n + j].float()
        ci = self.coords[inst, i]
        cj = self.coords[inst, j]
        return self.detour * (ci - cj).norm(dim=-1) / self.speed[inst]

    # ------------------------------------------------------------- features
    def _static_features(self, extra: torch.Tensor) -> torch.Tensor:
        E, N = self.E, self.N
        valid = self.valid
        stop = valid.clone()
        stop[:, 0] = False
        n_stops = stop.sum(1).clamp(min=1).float()
        xy = self.coords
        dist = xy.norm(dim=-1)
        scale = torch.sqrt((dist ** 2 * stop).sum(1) / n_stops).clamp(min=0.1)
        theta = torch.atan2(xy[..., 1], xy[..., 0])
        is_depot = torch.zeros(E, N, device=self.device)
        is_depot[:, 0] = 1.0
        has_tw = torch.isfinite(self.tw_end) & stop
        tw_s = torch.where(has_tw, self.tw_start / 60.0, torch.zeros_like(self.tw_start)).clamp(-2, 24)
        tw_e = torch.where(has_tw, self.tw_end / 60.0, torch.full_like(self.tw_end, 24.0)).clamp(-2, 24)
        inst = torch.arange(E, device=self.device)
        zeros = torch.zeros(E, dtype=torch.long, device=self.device)
        tt_from = self.travel_from(inst, zeros, N)
        tt_to = self.travel_to(inst, zeros, N)
        feats = [xy[..., 0] / scale[:, None], xy[..., 1] / scale[:, None], torch.log1p(dist),
                 torch.sin(theta), torch.cos(theta), is_depot, has_tw.float(), tw_s, tw_e,
                 torch.log1p(self.service.clamp(min=0)), (self.release / 60.0).clamp(0, 24),
                 torch.log1p(tt_from.clamp(min=0)), torch.log1p(tt_to.clamp(min=0)),
                 self._zone_size_frac(self.zone, stop, n_stops),
                 self._zone_size_frac(self.zone2, stop, n_stops)]
        out = torch.cat([torch.stack(feats, -1), extra], -1)
        return out * valid[..., None]

    @staticmethod
    def _zone_counts(zone: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-row count of masked nodes in each zone; column 0 collects zone -1."""
        B, Nb = zone.shape
        counts = torch.zeros(B, Nb + 1, device=zone.device)
        return counts.scatter_add_(1, (zone + 1).clamp(min=0, max=Nb), mask.float())

    def _zone_size_frac(self, zone, stop, n_stops):
        counts = self._zone_counts(zone, stop)
        same = counts.gather(1, (zone + 1).clamp(min=0, max=zone.shape[1]))
        return torch.where(zone >= 0, same / n_stops[:, None], torch.zeros_like(same))

    def build(self, inst: torch.Tensor, current: torch.Tensor, served: torch.Tensor,
              clock: torch.Tensor, Nb: Optional[int] = None, normalize: bool = True):
        """State 8-tuple for B partial routes (see module docstring).

        Args:
            inst: [B] instance indices.
            current: [B] current node.
            served: [B, >=Nb] bool, served nodes (depot included).
            clock: [B] decision clock in minutes.
            Nb: pad width (default: largest instance in the batch).
        """
        if Nb is None:
            Nb = int(self.n_nodes[inst].max())
        B = inst.shape[0]
        dev = self.device
        valid = self.valid[inst, :Nb]
        served = served[:, :Nb]
        n = self.n_nodes[inst]
        release = self.release[inst, :Nb]
        released = release <= clock[:, None] + EPS_TIME
        open_ = valid & ~served & released
        n_open = open_.sum(1)
        n_open_f = n_open.clamp(min=1).float()
        n_stops = (n - 1).clamp(min=1).float()

        tt_out = self.travel_from(inst, current, Nb)
        tt_in = self.travel_to(inst, current, Nb)
        coords = self.coords[inst, :Nb]
        cur_xy = coords.gather(1, current.view(-1, 1, 1).expand(-1, 1, 2))
        dist = (coords - cur_xy).norm(dim=-1)

        zone = self.zone[inst, :Nb]
        zone2 = self.zone2[inst, :Nb]
        zc = self._zone_counts(zone, open_)
        zc2 = self._zone_counts(zone2, open_)
        zone_open = torch.where(zone >= 0, zc.gather(1, (zone + 1).clamp(min=0)), torch.zeros_like(tt_out))
        zone2_open = torch.where(zone2 >= 0, zc2.gather(1, (zone2 + 1).clamp(min=0)), torch.zeros_like(tt_out))
        z_cur = zone.gather(1, current[:, None]).squeeze(1)
        z2_cur = zone2.gather(1, current[:, None]).squeeze(1)
        same_zone = (zone >= 0) & (zone == z_cur[:, None])
        same_zone2 = (zone2 >= 0) & (zone2 == z2_cur[:, None])
        cur_zone_open = torch.where(z_cur >= 0, zc.gather(1, (z_cur + 1).clamp(min=0)[:, None]).squeeze(1),
                                    torch.zeros(B, device=dev)) > 0

        big = torch.full_like(tt_out, float("inf"))
        tt_open = torch.where(open_, tt_out, big)
        rank = torch.argsort(torch.argsort(tt_open, dim=1), dim=1).float() / n_open_f[:, None]
        mean_tt = (tt_out * open_).sum(1) / n_open_f
        rel_tt = (tt_out / (mean_tt[:, None] + 1e-3)).clamp(0, 10)

        tw_end = self.tw_end[inst, :Nb]
        service = self.service[inst, :Nb]
        slack = (tw_end - (clock[:, None] + tt_out + service)) / 60.0
        slack = torch.where(torch.isfinite(tw_end), slack.clamp(-4, NO_TW_SLACK_H),
                            torch.full_like(slack, NO_TW_SLACK_H))
        age = torch.where(released, ((clock[:, None] - release) / 60.0).clamp(0, 8), torch.zeros_like(slack))

        node_dyn = torch.stack([zone_open / n_open_f[:, None], zone2_open / n_open_f[:, None], slack, age], -1)
        node_feat = torch.cat([self.static[inst, :Nb], node_dyn], -1)
        edge_feat = torch.stack([torch.log1p(tt_out.clamp(min=0)), torch.log1p(tt_in.clamp(min=0)), rank,
                                 same_zone.float(), same_zone2.float(), rel_tt,
                                 torch.log1p(dist)], -1)
        tod = (self.start_tod[inst] + clock) / 1440.0 * 2 * math.pi
        n_served = (served & valid).sum(1).float() - 1.0  # minus depot
        n_unrel = (valid & ~served & ~released).sum(1).float()
        glob = torch.stack([(clock / 60.0).clamp(0, 24), torch.sin(tod), torch.cos(tod),
                            n_served / n_stops, n_open.float() / n_stops, n_unrel / n_stops,
                            torch.log(n.float()), torch.log1p(mean_tt), cur_zone_open.float(),
                            (current == 0).float()], -1)
        glob = torch.cat([glob, self.inst_global[inst]], -1)

        action_mask = ~open_
        terminal = n_open == 0
        if terminal.any():
            # Absorbing state: expose one dummy action (the depot) so softmax /
            # dueling means stay finite.  Its value is gated by done = 1.
            action_mask[terminal, 0] = False

        if normalize and self.stats is not None:
            s = self.stats
            node_feat = (node_feat - s["mean_node"]) / s["std_node"]
            edge_feat = (edge_feat - s["mean_edge"]) / s["std_edge"]
            glob = (glob - s["mean_glob"]) / s["std_glob"]
        node_feat = node_feat * valid[..., None]
        edge_feat = edge_feat * valid[..., None]
        return (node_feat, valid, current, glob, open_, zone, action_mask, edge_feat)

    # ----------------------------------------------------------- normalization
    def set_stats(self, stats: Optional[Dict[str, torch.Tensor]]) -> None:
        self.stats = None if stats is None else {k: v.to(self.device).float() for k, v in stats.items()}

    @staticmethod
    def stats_from_states(states: Sequence[tuple]) -> Dict[str, torch.Tensor]:
        """Feature mean/std over un-normalized states (nodes: valid, edges: open)."""
        node = torch.cat([s[0][s[1]] for s in states])
        edge = torch.cat([s[7][s[4]] for s in states])
        glob = torch.cat([s[3] for s in states])

        def ms(x):
            mean = x.mean(0)
            std = x.std(0)
            std = torch.where(std < 1e-4, torch.ones_like(std), std)
            return mean.cpu(), std.cpu()

        mn, sn = ms(node)
        me, se = ms(edge)
        mg, sg = ms(glob)
        return dict(mean_node=mn, std_node=sn, mean_edge=me, std_edge=se, mean_glob=mg, std_glob=sg)


# ============================================================================
# Replay of logged / synthetic action sequences
# ============================================================================
def replay_trajectories(rt: RoutingTensors, inst: torch.Tensor, actions: torch.Tensor,
                        lengths: torch.Tensor, times: Optional[torch.Tensor] = None,
                        time_source: str = "logged", need_zone_leave: bool = True) -> Dict[str, torch.Tensor]:
    """Per-step quantities of K trajectories, vectorized over K.

    Args:
        inst: [K] instance per trajectory.
        actions: [K, T] node sequence (-1 padded).
        lengths: [K] trajectory lengths.
        times: [K, T] logged completion times (NaN = not logged) or None.
        time_source: "logged" uses logged completion times where available,
            "model" always uses travel + service.
        need_zone_leave: compute the zone-leave indicator (costs a [K, N] op per step).

    Returns:
        dict with ``start`` [K] (episode start: 0, or the first release if nothing
        is open at 0), ``clock`` [K, T+1] (decision epochs; entry T_k is the final
        completion time) and per-step [K, T] tensors ``sojourn``, ``travel``,
        ``service``, ``lateness``, ``zone_leave``, ``remaining``, plus
        ``return_travel`` [K, T] (travel back to the depot on the last step).
    """
    K, T = actions.shape
    dev = rt.device
    N = int(rt.n_nodes[inst].max()) if K else 1
    acts = actions.clamp(min=0)
    cur = torch.zeros(K, dtype=torch.long, device=dev)
    served = torch.zeros(K, N, dtype=torch.bool, device=dev)
    served[:, 0] = True
    valid = rt.valid[inst, :N]
    release = rt.release[inst, :N]
    zone = rt.zone[inst, :N]
    n_stops = rt.n_nodes[inst] - 1

    # episode start = the env's first decision epoch: 0 if some stop is open, else the first release
    stop_rel = torch.where(valid & (torch.arange(N, device=dev)[None, :] > 0), release,
                           torch.full_like(release, float("inf")))
    start = torch.where((stop_rel <= EPS_TIME).any(1), torch.zeros(K, device=dev), stop_rel.min(1).values)
    start = torch.where(torch.isfinite(start), start, torch.zeros_like(start))

    clock = torch.zeros(K, T + 1, device=dev)
    out = {k: torch.zeros(K, T, device=dev) for k in
           ("sojourn", "travel", "service", "lateness", "zone_leave", "remaining", "return_travel")}
    prev_finish = start.clone()
    finish_all = torch.zeros(K, T, device=dev)
    use_logged = times is not None and time_source == "logged"
    for t in range(T):
        active = t < lengths
        if not bool(active.any()):
            break
        a = acts[:, t]
        d = torch.maximum(prev_finish, release.gather(1, a[:, None]).squeeze(1))
        clock[:, t] = torch.where(active, d, clock[:, t])
        tt = rt.travel_pair(inst, cur, a)
        svc = rt.service[inst, a]
        f = d + tt + svc
        if use_logged:
            logged = times[:, t]
            f = torch.where(torch.isfinite(logged), torch.maximum(logged, d), f)
        late = torch.relu(f - rt.tw_end[inst, a])
        late = torch.where(torch.isfinite(late), late, torch.zeros_like(late))
        if need_zone_leave:
            released = release <= d[:, None] + EPS_TIME
            open_ = valid & ~served & released
            z_cur = zone.gather(1, cur[:, None]).squeeze(1)
            z_a = zone.gather(1, a[:, None]).squeeze(1)
            other = open_.clone()
            other.scatter_(1, a[:, None], False)
            has_open = ((zone == z_cur[:, None]) & other).any(1) & (z_cur >= 0)
            leave = (has_open & (z_a != z_cur)).float()
        else:
            leave = torch.zeros(K, device=dev)
        for key, val in (("travel", tt), ("service", svc), ("lateness", late), ("zone_leave", leave),
                         ("remaining", (n_stops - t).float())):
            out[key][:, t] = torch.where(active, val, out[key][:, t])
        finish_all[:, t] = torch.where(active, f, finish_all[:, t])
        served = served | (torch.nn.functional.one_hot(a, N).bool() & active[:, None])
        cur = torch.where(active, a, cur)
        prev_finish = torch.where(active, f, prev_finish)

    # next decision epoch d_{t+1} = max(f_t, release[a_{t+1}]); last one = f_{T-1}
    ar = torch.arange(T, device=dev)
    nxt = torch.cat([acts[:, 1:], torch.zeros(K, 1, dtype=torch.long, device=dev)], 1)
    nxt_rel = release.gather(1, nxt)
    is_last = ar[None, :] == (lengths[:, None] - 1)
    d_next = torch.where(is_last, finish_all, torch.maximum(finish_all, nxt_rel))
    in_traj = ar[None, :] < lengths[:, None]
    clock[:, 1:] = torch.where(in_traj, d_next, clock[:, 1:])
    sojourn = d_next - clock[:, :-1]
    # idle time before the first decision (the log waited for a later task) counts in step 0,
    # so that sum(sojourn) = final completion - episode start, as in the env
    sojourn[:, 0] = sojourn[:, 0] + (clock[:, 0] - start)
    out["sojourn"] = torch.where(in_traj, sojourn.clamp(min=0), torch.zeros_like(d_next))
    last = acts.gather(1, (lengths - 1).clamp(min=0)[:, None]).squeeze(1)
    back = rt.travel_pair(inst, last, torch.zeros_like(last))
    out["return_travel"] = torch.where(is_last, back[:, None].expand(-1, T), torch.zeros_like(d_next))
    out["clock"] = clock
    out["finish"] = finish_all
    out["start"] = start
    return out


# ============================================================================
# Rollout environment
# ============================================================================
class RoutingEnv:
    """Batched rollout env over instances of a :class:`RoutingTensors`.

    ``step`` uses the travel model (matrix or distance / speed); rows that are
    done ignore their actions.  The env never lets a policy idle voluntarily:
    it only waits (jumping the clock) when no stop is open.
    """

    def __init__(self, rt: RoutingTensors, reward: RewardConfig):
        self.rt = rt
        self.reward = reward

    def reset(self, inst: torch.Tensor, build_state: bool = True):
        rt = self.rt
        self.inst = inst.to(rt.device)
        B = self.inst.shape[0]
        self.B = B
        self.Nb = int(rt.n_nodes[self.inst].max())
        self.cur = torch.zeros(B, dtype=torch.long, device=rt.device)
        self.served = torch.zeros(B, self.Nb, dtype=torch.bool, device=rt.device)
        self.served[:, 0] = True
        self.valid = rt.valid[self.inst, :self.Nb]
        self.clock = torch.zeros(B, device=rt.device)
        self.t = 0
        self.done = rt.n_nodes[self.inst] <= 1
        self.sequence = torch.full((B, self.Nb - 1), -1, dtype=torch.long, device=rt.device)
        self.finish = torch.full((B, self.Nb - 1), float("nan"), device=rt.device)
        self.totals = {k: torch.zeros(B, device=rt.device) for k in
                       ("return", "time", "travel", "lateness", "late_stops", "zone_leave", "latency")}
        self._advance_idle()
        return self.state() if build_state else None

    # -------------------------------------------------------------- helpers
    def open_mask(self) -> torch.Tensor:
        released = self.rt.release[self.inst, :self.Nb] <= self.clock[:, None] + EPS_TIME
        return self.valid & ~self.served & released

    def travel_out(self) -> torch.Tensor:
        return self.rt.travel_from(self.inst, self.cur, self.Nb)

    def _advance_idle(self):
        """If nothing is open but stops remain, jump the clock to the next release."""
        remaining = self.valid & ~self.served
        idle = (~self.open_mask().any(1)) & remaining.any(1) & ~self.done
        if idle.any():
            rel = torch.where(remaining, self.rt.release[self.inst, :self.Nb],
                              torch.full_like(self.clock[:, None].expand(-1, self.Nb), float("inf")))
            nxt = rel.min(1).values
            self.clock = torch.where(idle, torch.maximum(self.clock, nxt), self.clock)

    def state(self, normalize: bool = True):
        return self.rt.build(self.inst, self.cur, self.served, self.clock, Nb=self.Nb, normalize=normalize)

    # ----------------------------------------------------------------- step
    def step(self, action: torch.Tensor, build_state: bool = True):
        rt = self.rt
        action = action.to(rt.device).long()
        live = ~self.done
        open_ = self.open_mask()
        ok = open_.gather(1, action.clamp(min=0, max=self.Nb - 1)[:, None]).squeeze(1)
        if bool((live & ~ok).any()):
            raise ValueError("infeasible action for a live episode")
        a = torch.where(live, action, torch.zeros_like(action))
        d = self.clock
        tt = rt.travel_pair(self.inst, self.cur, a)
        svc = rt.service[self.inst, a]
        f = d + tt + svc
        late = torch.relu(f - rt.tw_end[self.inst, a])
        late = torch.where(torch.isfinite(late), late, torch.zeros_like(late))
        zone = rt.zone[self.inst, :self.Nb]
        z_cur = zone.gather(1, self.cur[:, None]).squeeze(1)
        z_a = zone.gather(1, a[:, None]).squeeze(1)
        other = open_.clone()
        other.scatter_(1, a[:, None], False)
        leave = (((zone == z_cur[:, None]) & other).any(1) & (z_cur >= 0) & (z_a != z_cur)).float()
        remaining = (self.valid & ~self.served).sum(1).float()

        self.served = self.served | (torch.nn.functional.one_hot(a, self.Nb).bool() & live[:, None])
        self.cur = torch.where(live, a, self.cur)
        self.clock = torch.where(live, f, self.clock)
        if self.t < self.sequence.shape[1]:
            self.sequence[:, self.t] = torch.where(live, a, self.sequence[:, self.t])
            self.finish[:, self.t] = torch.where(live, f, self.finish[:, self.t])
        self.t += 1
        now_done = live & ~(self.valid & ~self.served).any(1)
        self._advance_idle()
        sojourn = torch.where(live, self.clock - d, torch.zeros_like(d))
        back = torch.where(now_done, rt.travel_pair(self.inst, self.cur, torch.zeros_like(self.cur)),
                           torch.zeros_like(d))
        comps = dict(sojourn=sojourn, travel=tt * live, lateness=late * live, zone_leave=leave * live,
                     remaining=remaining, return_travel=back)
        r = self.reward.combine(comps) * live
        tot = self.totals
        tot["return"] += r
        tot["time"] += sojourn + (back if self.reward.return_to_depot else 0.0)
        tot["travel"] += tt * live + back
        tot["lateness"] += late * live
        tot["late_stops"] += (late > 1e-6).float() * live
        tot["zone_leave"] += leave * live
        tot["latency"] += sojourn * remaining
        self.done = self.done | now_done
        return (self.state() if build_state else None), r, self.done.clone()

    # ------------------------------------------------------------------ run
    def rollout(self, inst: torch.Tensor, policy: Callable, needs_state: bool = True,
                max_steps: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Run ``policy(env, state) -> actions`` until all episodes finish.

        Heuristic policies that only use ``env.open_mask()`` / ``env.travel_out()``
        can pass ``needs_state=False`` (state is then None) to skip feature building.
        """
        state = self.reset(inst, build_state=needs_state)
        steps = 0
        limit = max_steps or self.Nb
        while not bool(self.done.all()) and steps < limit:
            with torch.no_grad():
                act = policy(self, state)
            state, _, _ = self.step(act, build_state=needs_state)
            steps += 1
        res = {k: v.clone() for k, v in self.totals.items()}
        res["sequence"] = self.sequence.clone()
        res["finish"] = self.finish.clone()
        res["makespan"] = self.clock.clone()
        return res


def model_policy(model, deterministic: bool = True) -> Callable:
    """Wrap a network with ``get_action(*state, deterministic)`` as an env policy."""
    def act(env: RoutingEnv, state):
        return model.get_action(*state, deterministic=deterministic)
    return act
