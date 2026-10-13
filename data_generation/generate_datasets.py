"""Generate an offline dataset from scratch, in the dataset-directory format.

    # instances + PDR + random + eps-PDR, like the released fjsp/sd1_10x5 (GA: see below)
    python -m data_generation.generate_datasets --dataset fjsp/sd1_10x5 --generator sd1 \
        --n_j 10 --n_m 5 --num_train 1000 --seed 101 \
        --sources pdr random:300 eps_pdr:0.1:100:500 eps_pdr:0.2:100:500

    # JSP, like the released jsp/taillard_10x5
    python -m data_generation.generate_datasets --dataset jsp/taillard_10x5 \
        --generator taillard --n_j 10 --n_m 5 --num_train 500 --seed 101 --sources pdr random:300

Instance generators (--generator):
    sd1       FJSP, processing times 1-20, 0.8-1.2 n_m operations per job (CaseGenerator);
              all train instances share one job-length vector, val instances draw their own
    sd2       FJSP, n_m operations per job, processing times 1-99, 1-5 machines per operation
    taillard  JSP, one machine per operation, processing times 1-98 (create_jsp_instance)

Train instance i is drawn after reseeding with seed + i, val instance i with val_seed + i
(default val_seed = seed + 1000). --seed 101 reproduces the training instances of the
released fjsp/sd1_* and jsp/taillard_* datasets, and the JSP validation instances. (The
released FJSP validation instances come from songwenas12/fjsp-drl.)

Trajectory sources (--sources), each from its own --source_seed stream:
    pdr                  dispatching rules of env/dispatching.py (16 for FJSP; for JSP, where
                         every operation has one machine, the 4 job rules with SPT)
    random:K             K uniformly random trajectories per instance
    eps_pdr:EPS:K[:N]    K epsilon-PDR trajectories per instance, for the first N instances
More sources can be added later with add_random_trajectories / add_eps_pdr_rules. GA
trajectories come from an external GA: run data_generation/run_GA_custom.py on the dataset
directory, then add them with data_generation/add_ga_data.py.
"""
import argparse
import random

import tqdm

from cdqac.data.dataset import OfflineDataset
from cdqac.data.writer import add_source, create_dataset
from data_generation.add_eps_pdr_rules import eps_source, source_info as eps_info
from data_generation.add_random_trajectories import SOURCE_INFO as RANDOM_INFO, random_source
from env.common_utils import setup_seed
from env.data_utils import CaseGenerator, SD2_instance_generator_no_config, create_jsp_instance
from env.dispatching import PDR_JOB_RULES, PDR_MACHINE_RULES, pdr_rule_name, run_dispatching


def sd1_instances(n_j, n_m, num, seed, same_job_length):
    lo, hi = int(0.8 * n_m), int(1.2 * n_m)
    setup_seed(seed)
    prepare_job_length = [random.randint(lo, hi) for _ in range(n_j)]
    job_lengths, op_pts = [], []
    for i in range(num):
        setup_seed(seed + i)
        case = CaseGenerator(n_j, n_m, lo, hi, nums_ope=prepare_job_length, path="./test",
                             flag_same_opes=same_job_length, flag_doc=False)
        job_length, op_pt, _ = case.get_case(i)
        job_lengths.append(job_length)
        op_pts.append(op_pt)
    return job_lengths, op_pts


def sd2_instances(n_j, n_m, num, seed):
    job_lengths, op_pts = [], []
    for i in range(num):
        setup_seed(seed + i)
        job_length, op_pt, _ = SD2_instance_generator_no_config(
            n_j=n_j, n_m=n_m, op_per_job=0, low=1, high=99, op_per_mch_min=1,
            op_per_mch_max=min(5, n_m))
        job_lengths.append(job_length)
        op_pts.append(op_pt)
    return job_lengths, op_pts


def taillard_instances(n_j, n_m, num, seed):
    job_lengths, op_pts = [], []
    for i in range(num):
        setup_seed(seed + i)
        job_length, op_pt = create_jsp_instance(n_j, n_m, 1, 99)
        job_lengths.append(job_length)
        op_pts.append(op_pt)
    return job_lengths, op_pts


GENERATOR_INFO = {
    "sd1": {"name": "SD1", "code": "env.data_utils.CaseGenerator",
            "params": {"processing_times": [1, 20], "ops_per_job": "0.8-1.2 n_m"},
            "notes": "all train instances share one job_length vector; val instances draw "
                     "their own"},
    "sd2": {"name": "SD2", "code": "env.data_utils.SD2_instance_generator_no_config",
            "params": {"processing_times": [1, 99], "ops_per_job": "n_m",
                       "machines_per_op": "1-min(5, n_m)"}},
    "taillard": {"name": "taillard", "code": "env.data_utils.create_jsp_instance",
                 "params": {"processing_times": [1, 98]},
                 "notes": "every operation has exactly one machine"},
}


def make_instances(generator, n_j, n_m, num, seed, split):
    if generator == "sd1":
        return sd1_instances(n_j, n_m, num, seed, same_job_length=(split == "train"))
    if generator == "sd2":
        return sd2_instances(n_j, n_m, num, seed)
    if generator == "taillard":
        return taillard_instances(n_j, n_m, num, seed)
    raise ValueError(f"unknown generator {generator}")


def pdr_source(split, machine_rules):
    rules = [(j, m) for j in PDR_JOB_RULES for m in PDR_MACHINE_RULES if m in machine_rules]
    actions, makespans = [], []
    op_pts = split.op_pts()
    for i in tqdm.trange(len(split), desc="dispatching rules"):
        runs = [run_dispatching(split.job_length[i], op_pts[i], j, m) for j, m in rules]
        actions.append([a for a, _ in runs])
        makespans.append([m for _, m in runs])
    return actions, makespans, [pdr_rule_name(j, m) for j, m in rules]


def parse_source(spec: str):
    """'pdr' | 'random:K' | 'eps_pdr:EPS:K[:N]' -> (kind, params)."""
    parts = spec.split(":")
    if parts[0] == "pdr" and len(parts) == 1:
        return "pdr", {}
    if parts[0] == "random" and len(parts) == 2:
        return "random", {"num_random": int(parts[1])}
    if parts[0] == "eps_pdr" and len(parts) in (3, 4):
        return "eps_pdr", {"eps": parts[1], "num_runs": int(parts[2]),
                           "num_instances": int(parts[3]) if len(parts) == 4 else None}
    raise ValueError(f"cannot parse source '{spec}' (expected pdr, random:K or eps_pdr:EPS:K[:N])")


def generate(dataset_id, generator, n_j, n_m, num_train, num_val, seed, val_seed, sources,
             source_seed=100, data_root="./train_datasets", overwrite=False):
    specs = [parse_source(s) for s in sources]
    train = make_instances(generator, n_j, n_m, num_train, seed, "train")
    val = make_instances(generator, n_j, n_m, num_val, val_seed, "val") if num_val else None
    info = dict(GENERATOR_INFO[generator], seed=seed, val_seed=val_seed if num_val else None)
    path = create_dataset(data_root, dataset_id, train, val=val, generator=info,
                          description=f"{generator} {n_j}x{n_m} offline dataset "
                                      f"(data_generation/generate_datasets.py)",
                          overwrite=overwrite)
    split = OfflineDataset.from_dir(path).split("train")
    for kind, params in specs:
        if kind == "pdr":
            machine_rules = ["SPT"] if generator == "taillard" else list(PDR_MACHINE_RULES)
            actions, makespans, names = pdr_source(split, machine_rules)
            add_source(path, "pdr", actions, makespans, names=names,
                       info={"description": "priority dispatching rules, job rule x machine "
                                            "rule, action mask on; slot k is rule names[k]",
                             "order": "rule"})
        elif kind == "random":
            actions, makespans = random_source(split, params["num_random"], source_seed)
            add_source(path, "random", actions, makespans,
                       info=dict(RANDOM_INFO, seed=source_seed))
        else:
            actions, makespans = eps_source(split, float(params["eps"]), params["num_runs"],
                                            source_seed, params["num_instances"])
            add_source(path, f"eps_pdr_{params['eps']}", actions, makespans,
                       info=dict(eps_info(params["eps"]), seed=source_seed))
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="id of the new dataset, e.g. fjsp/sd1_10x5")
    p.add_argument("--generator", required=True, choices=sorted(GENERATOR_INFO))
    p.add_argument("--n_j", type=int, required=True)
    p.add_argument("--n_m", type=int, required=True)
    p.add_argument("--num_train", type=int, default=1000)
    p.add_argument("--num_val", type=int, default=100)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--val_seed", type=int, default=None, help="default: seed + 1000")
    p.add_argument("--sources", nargs="*", default=["pdr", "random:300"])
    p.add_argument("--source_seed", type=int, default=100)
    p.add_argument("--data_root", default="./train_datasets")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    val_seed = args.seed + 1000 if args.val_seed is None else args.val_seed
    path = generate(args.dataset, args.generator, args.n_j, args.n_m, args.num_train,
                    args.num_val, args.seed, val_seed, args.sources, args.source_seed,
                    args.data_root, args.overwrite)
    print(f"Wrote {path}; check it with: python -m cdqac.data validate {args.dataset}")


if __name__ == "__main__":
    main()
