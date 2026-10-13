"""Add epsilon-PDR trajectories (App. F.3) to a dataset as its "eps_pdr_<eps>" source.

    python -m data_generation.add_eps_pdr_rules --dataset fjsp/sd1_10x5 --eps_random 0.1 \
        --num_runs 100 --num_instances 500

Trajectory k of an instance follows dispatching rule k mod 4 of {MWR, LWR, LOPNR, MOPNR} x SPT
and, at every step, takes a uniformly random feasible action instead with probability eps.
The released datasets hold 100 trajectories for the first 500 instances, for eps 0.1 and 0.2.
"""
import argparse
import itertools

import numpy as np
import torch
import tqdm

from cdqac.data.dataset import OfflineDataset
from cdqac.data.format import eps_pdr_source
from cdqac.data.writer import add_source, dataset_dir
from env.dispatching import FJSPDispatching
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums

JOB_RULES = ["MWR", "LWR", "LOPNR", "MOPNR"]
MACHINE_RULES = ["SPT"]


def source_info(eps) -> dict:
    return {"description": f"PDR (MWR/LWR/LOPNR/MOPNR x SPT, cycled per trajectory) that takes "
                           f"a uniformly random feasible action with probability {eps}",
            "order": "generation", "eps": float(eps)}


def run_trajectory(job_length, op_pt, num_runs, comb_dispatching, eps_random):
    """``num_runs`` epsilon-PDR trajectories on one instance: (action lists, makespans)."""
    job_length = np.tile(np.asarray(job_length)[None], (num_runs, 1))  # [num_runs, J]
    op_pt = np.tile(np.asarray(op_pt)[None], (num_runs, 1, 1))       # [num_runs, N, M]
    n_m = op_pt.shape[-1]
    n_j = job_length.shape[-1]

    env = FJSPEnvForVariousOpNums(n_j, n_m, device="cpu", mask_actions=True)
    env.set_initial_data(job_length, op_pt)

    dispatchers = []
    for i in range(num_runs):
        curr_rule = comb_dispatching[i % len(comb_dispatching)]
        d = FJSPDispatching()
        d.set_rules(job_rule=curr_rule[0], machine_rule=curr_rule[1])
        dispatchers.append(d)

    action_lists = [[] for _ in range(num_runs)]
    done = env.done_flag.copy()  # [num_runs] bool array

    while not done.all():
        incomplete = np.where(done == 0)[0]
        actions = np.empty(len(incomplete), dtype=int)
        for k, env_idx in enumerate(incomplete):
            action = dispatchers[env_idx].select_action(
                env.state, env.state.dynamic_pair_mask_tensor, env,
                eps_random=eps_random, env_idx=int(env_idx)
            )
            actions[k] = action
            action_lists[env_idx].append(int(action))
        _, _, done = env.step(actions)

    return action_lists, env.current_makespan.astype(int)


def eps_source(split, eps_random, num_runs, seed, num_instances=None):
    """Epsilon-PDR trajectories for the first ``num_instances`` instances of ``split``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    comb = list(itertools.product(JOB_RULES, MACHINE_RULES))
    n = len(split) if num_instances is None else min(num_instances, len(split))
    actions, makespans = [], []
    op_pts = split.op_pts()
    for i in tqdm.trange(n, desc=f"eps-PDR {eps_random}"):
        a, m = run_trajectory(split.job_length[i], op_pts[i], num_runs, comb, eps_random)
        actions.append(a)
        makespans.append(m)
    return actions, makespans


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset id, e.g. fjsp/sd1_10x5")
    parser.add_argument("--data_root", default="./train_datasets")
    parser.add_argument("--eps_random", type=str, default="0.1",
                        help="random-action probability; also names the source (eps_pdr_<eps>)")
    parser.add_argument("--num_runs", type=int, default=100, help="trajectories per instance")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_instances", type=int, default=None,
                        help="only the first N train instances (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing source")
    args = parser.parse_args()

    split = OfflineDataset.open(args.data_root, args.dataset).split("train")
    actions, makespans = eps_source(split, float(args.eps_random), args.num_runs, args.seed,
                                    args.num_instances)
    name = eps_pdr_source(args.eps_random)
    entry = add_source(dataset_dir(args.data_root, args.dataset), name, actions, makespans,
                       info=dict(source_info(args.eps_random), seed=args.seed),
                       overwrite=args.overwrite)
    print(f"Added {name}: {entry['num_instances']} instances x {entry['per_instance']}")


if __name__ == "__main__":
    main()
