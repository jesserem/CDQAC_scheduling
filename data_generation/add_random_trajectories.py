"""Add uniformly random trajectories to a dataset as its "random" source.

    python -m data_generation.add_random_trajectories --dataset fjsp/sd1_10x5 --num_random 300

Every step takes a uniformly random feasible action. The instances are processed in order
from one seeded random stream; --seed 100 (the default) is the seed the released datasets
were generated with.
"""
import argparse

import numpy as np
import torch
import tqdm

from cdqac.data.dataset import OfflineDataset
from cdqac.data.writer import add_source, dataset_dir
from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums

SOURCE_INFO = {"description": "uniformly random feasible action at every step",
               "order": "generation"}


def run_trajectory(job_length, op_pt, num_random):
    """``num_random`` random trajectories on one instance: (actions [num_random, n_ops],
    makespans [num_random])."""
    job_length = np.asarray(job_length)[None].repeat(num_random, axis=0)
    op_pt = np.asarray(op_pt)[None].repeat(num_random, axis=0)
    n_j, n_o, n_m = job_length.shape[-1], op_pt.shape[1], op_pt.shape[-1]
    action_array = np.zeros((n_o, num_random), dtype=int)

    env = FJSPEnvForSameOpNums(n_j, n_m, device="cpu", mask_actions=True)
    state = env.set_initial_data(job_length, op_pt)
    done = env.done()
    i = 0
    while not done.all():
        mask = (~state.dynamic_pair_mask_tensor.flatten(1))
        actions = torch.multinomial(mask.float(), 1).squeeze(-1).cpu().numpy()
        state, reward, done = env.step(actions)
        action_array[i] = actions
        i += 1
    return np.transpose(action_array), env.current_makespan.astype(int)


def random_source(split, num_random, seed, num_instances=None):
    """Random trajectories for the first ``num_instances`` instances of ``split``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(split) if num_instances is None else min(num_instances, len(split))
    actions, makespans = [], []
    op_pts = split.op_pts()
    for i in tqdm.trange(n, desc="random trajectories"):
        a, m = run_trajectory(split.job_length[i], op_pts[i], num_random)
        actions.append(a)
        makespans.append(m)
    return actions, makespans


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset id, e.g. fjsp/sd1_10x5")
    parser.add_argument("--data_root", default="./train_datasets")
    parser.add_argument("--num_random", type=int, default=300, help="trajectories per instance")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num_instances", type=int, default=None,
                        help="only the first N train instances (default: all)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing source")
    args = parser.parse_args()

    split = OfflineDataset.open(args.data_root, args.dataset).split("train")
    actions, makespans = random_source(split, args.num_random, args.seed, args.num_instances)
    entry = add_source(dataset_dir(args.data_root, args.dataset), "random", actions, makespans,
                       info=dict(SOURCE_INFO, seed=args.seed), overwrite=args.overwrite)
    print(f"Added random: {entry['num_instances']} instances x {entry['per_instance']}")


if __name__ == "__main__":
    main()
