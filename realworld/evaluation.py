"""Evaluation of policies (learned or heuristic) and of the logged behaviour."""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from realworld.episodes import RoutingEpisodeSet
from realworld.registry import DatasetSpec
from realworld.routing.core import RewardConfig, RoutingEnv, RoutingTensors, replay_trajectories
from realworld.routing.metrics import (amazon_route_score, route_prediction_metrics, summarize,
                                       zone_statistics)


def load_split(data_root: str, dataset: str, split: str, instance_filter: str = "",
               num_instances: Optional[int] = None) -> RoutingEpisodeSet:
    path = os.path.join(data_root, dataset, f"{split}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run `python prepare_realworld.py convert --dataset {dataset}`")
    eps = RoutingEpisodeSet.load(path)
    if instance_filter:
        eps = filter_instances(eps, instance_filter)
    if num_instances is not None and num_instances < eps.num_instances:
        eps = eps.select_instances(np.arange(num_instances))
    return eps


def filter_instances(eps: RoutingEpisodeSet, spec: str) -> RoutingEpisodeSet:
    """``"key=v1,v2;key2=v3"`` keeps instances whose metadata matches every key."""
    keep = np.ones(eps.num_instances, bool)
    for clause in spec.split(";"):
        if not clause.strip():
            continue
        key, vals = clause.split("=", 1)
        allowed = {v.strip() for v in vals.split(",")}
        keep &= np.array([str(m.get(key.strip())) in allowed for m in eps.inst_meta])
    if not keep.any():
        raise ValueError(f"instance filter {spec!r} keeps no instances")
    return eps.select_instances(np.nonzero(keep)[0])


def reference_index(eps: RoutingEpisodeSet, source: str) -> Dict[int, int]:
    """instance -> first trajectory of ``source``."""
    out = {}
    for k in eps.trajectory_indices([source]):
        out.setdefault(int(eps.traj_inst[k]), int(k))
    return out


def replay_totals(eps: RoutingEpisodeSet, rt: RoutingTensors, traj: Sequence[int], reward: RewardConfig,
                  time_source: str) -> Dict[str, np.ndarray]:
    """Episode totals of stored trajectories (e.g. the logged routes)."""
    traj = np.asarray(traj)
    dev = rt.device
    T = int(eps.traj_len[traj].max())
    out = replay_trajectories(
        rt, torch.as_tensor(eps.traj_inst[traj], dtype=torch.long, device=dev),
        torch.as_tensor(eps.traj_actions[traj, :T], dtype=torch.long, device=dev),
        torch.as_tensor(eps.traj_len[traj], dtype=torch.long, device=dev),
        torch.as_tensor(eps.traj_times[traj, :T], dtype=torch.float32, device=dev),
        time_source=time_source, need_zone_leave=True)
    ret = reward.combine(out).sum(1)
    time = out["sojourn"].sum(1) + (out["return_travel"].sum(1) if reward.return_to_depot else 0.0)
    return {k: v.cpu().numpy() for k, v in dict(
        **{"return": ret}, time=time, travel=out["travel"].sum(1) + out["return_travel"].sum(1),
        lateness=out["lateness"].sum(1), late_stops=(out["lateness"] > 1e-6).float().sum(1),
        zone_leave=out["zone_leave"].sum(1),
        latency=(out["sojourn"] * out["remaining"]).sum(1)).items()}


def evaluate(eps: RoutingEpisodeSet, rt: RoutingTensors, policy: Optional[Callable], reward: RewardConfig,
             spec: DatasetSpec, instances: Optional[Sequence[int]] = None, batch_size: int = 128,
             needs_state: bool = True, reference_metrics: bool = True, num_samples: int = 1,
             label: str = "policy") -> Tuple[List[Dict], Dict[str, float]]:
    """Roll out ``policy`` (or, if None, replay the logged routes) and score it.

    With ``num_samples > 1`` the policy is rolled out that many times (it should
    sample) and the best-return route per instance is kept, as in the paper's
    100-sample evaluation.
    """
    inst_all = np.arange(eps.num_instances) if instances is None else np.asarray(instances)
    ref = reference_index(eps, spec.logged_source)
    ref_model = ref_logged = None
    ref_traj = [ref[e] for e in inst_all if e in ref]
    if ref_traj:
        ref_pos = {e: i for i, e in enumerate(e for e in inst_all if e in ref)}
        ref_model = replay_totals(eps, rt, ref_traj, reward, "model")
        if np.isfinite(eps.traj_times[ref_traj]).any():
            ref_logged = replay_totals(eps, rt, ref_traj, reward, "logged")
    rows: List[Dict] = []
    if policy is None:
        if not ref_traj:
            raise ValueError("no logged trajectories to replay")
        for e in inst_all:
            if e not in ref:
                continue
            i = ref_pos[e]
            row = dict(inst=int(e), **{k: float(v[i]) for k, v in ref_model.items()})
            seq = [0] + eps.traj_actions[ref[e], :eps.traj_len[ref[e]]].tolist()
            row.update(zone_statistics(seq, eps.zone[e]))
            if ref_logged is not None:
                row["logged_time"] = float(ref_logged["time"][i])
            if reference_metrics:
                row.update(dict(hr3=1.0, krc=1.0, lsd=0.0, ed=0.0, time_ratio=1.0))
                if spec.amazon_score:
                    row["amazon_score"] = 0.0
            rows.append(row)
        return rows, summarize(rows)

    env = RoutingEnv(rt, reward)
    for s in range(0, len(inst_all), batch_size):
        batch = torch.as_tensor(inst_all[s:s + batch_size], dtype=torch.long, device=rt.device)
        best = None
        for _ in range(num_samples):
            res = env.rollout(batch, policy, needs_state=needs_state)
            res = {k: v.cpu() for k, v in res.items()}
            if best is None:
                best = res
            else:
                better = res["return"] > best["return"]
                for k in best:
                    m = better.view(-1, *([1] * (best[k].dim() - 1)))
                    best[k] = torch.where(m, res[k], best[k])
        for b, e in enumerate(inst_all[s:s + batch_size]):
            n = int(eps.n_nodes[e])
            seq = [0] + best["sequence"][b, :n - 1].tolist()
            row = dict(inst=int(e), **{k: float(best[k][b]) for k in
                                       ("return", "time", "travel", "lateness", "late_stops", "zone_leave",
                                        "latency", "makespan")})
            row.update(zone_statistics(seq, eps.zone[e]))
            if e in ref and ref_model is not None:
                i = ref_pos[e]
                row["ref_time"] = float(ref_model["time"][i])
                row["time_ratio"] = row["time"] / max(row["ref_time"], 1e-6)
                row["ref_return"] = float(ref_model["return"][i])
                if reference_metrics:
                    ref_seq = [0] + eps.traj_actions[ref[e], :eps.traj_len[ref[e]]].tolist()
                    row.update(route_prediction_metrics(seq[1:], ref_seq[1:]))
                    if spec.amazon_score:
                        row["amazon_score"] = amazon_route_score(ref_seq, seq, eps.travel_matrix(e))
            rows.append(row)
    for r in rows:
        r["label"] = label
    return rows, summarize(rows)
