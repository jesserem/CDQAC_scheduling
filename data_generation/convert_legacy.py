"""Convert legacy single-file datasets (train_10_5.npy, ...) to the dataset-directory format.

    # the five released datasets, from train_datasets/{fjsp,jsp}/ into train_datasets/
    python -m data_generation.convert_legacy --all

    # one dataset
    python -m data_generation.convert_legacy --train train_datasets/fjsp/train_10_5.npy \
        --val train_datasets/fjsp/val_10_5.npy --dataset fjsp/sd1_10x5

Only the PDR rules in ``cdqac.data.format.PDR_RULES`` are kept. The legacy files store no
makespans for the epsilon-PDR trajectories; they are computed here by replaying them.

After writing, the dataset is read back and every array is compared with the legacy file;
``--replay_check N`` also replays every trajectory of the first N instances and compares
the makespans with the stored ones.
"""
import argparse
import hashlib
import os
import time

import numpy as np
import tqdm

from cdqac.data import format as fmt
from cdqac.data.dataset import OfflineDataset, validate
from cdqac.data.legacy import (legacy_coverage, legacy_pdr_rules, legacy_rule_name, legacy_source,
                               legacy_split, read_legacy)
from cdqac.data.replay import replay_makespans

# The released datasets: legacy files (relative to --legacy_root) and the paper-era names
# of the same files (md5-identical).
RELEASED = [
    {"dataset": "fjsp/sd1_10x5", "train": "fjsp/train_10_5.npy", "val": "fjsp/val_10_5.npy",
     "paper_era": ["dataset/SD1_train_10_5_1000.npy", "dataset/SD1_10_5_val.npy"]},
    {"dataset": "fjsp/sd1_15x10", "train": "fjsp/train_15_10.npy", "val": "fjsp/val_15_10.npy",
     "paper_era": ["dataset/SD1_train_15_10_1000.npy", "dataset/SD1_15_10_val.npy"]},
    {"dataset": "fjsp/sd1_20x10", "train": "fjsp/train_20_10.npy", "val": "fjsp/val_20_10.npy",
     "paper_era": ["dataset/SD1_train_20_10_500.npy", "dataset/SD1_20_10_val.npy"]},
    {"dataset": "jsp/taillard_10x5", "train": "jsp/train_10_5.npy", "val": "jsp/val_10_5.npy",
     "paper_era": ["jsp_dataset/train_10_5_500.npy", "jsp_dataset/eval_10_5.npy"]},
    {"dataset": "jsp/taillard_15x10", "train": "jsp/train_15_10.npy", "val": "jsp/val_15_10.npy",
     "paper_era": ["jsp_dataset/train_15_10_500.npy", "jsp_dataset/eval_15_10.npy"]},
]

GENERATORS = {
    "fjsp": {
        "name": "SD1",
        "code": "env.data_utils.CaseGenerator (data_generation/generate_datasets.py --generator sd1)",
        "notes": "all train instances share one job_length vector; validation instances "
                 "draw it per instance",
    },
    "jsp": {
        "name": "taillard",
        "code": "env.data_utils.create_jsp_instance "
                "(data_generation/generate_datasets.py --generator taillard)",
        "notes": "every operation has exactly one machine",
    },
}

# What is known about how the released data was made (verified by regenerating it, see
# tests/test_data_generation.py): merged into the manifests of the released datasets.
RELEASED_GENERATORS = {
    "fjsp": {"seed": 101, "val_seed": None,
             "val_origin": "SD1 validation instances from songwenas12/fjsp-drl"},
    "jsp": {"seed": 101, "val_seed": 1101},
}
GA_NOTE = "external GA (data_generation/run_GA_custom.py); cannot be regenerated bit for bit"
RELEASED_SOURCES = {
    "pdr": {"reproduce": "deterministic: env/dispatching.py via data_generation/generate_datasets.py"},
    "random": {"seed": 100, "reproduce": "data_generation/add_random_trajectories.py --seed 100"},
    "ga_hof": {"reproduce": GA_NOTE},
    "ga_pop": {"reproduce": GA_NOTE},
    "eps_pdr": {"seed": None,
                "reproduce": "data_generation/add_eps_pdr_rules.py, but --seed 100 does not "
                             "reproduce the released trajectories"},
}


def add_released_metadata(manifest: dict) -> dict:
    """Add the known seeds and regeneration notes to the manifest of a released dataset."""
    manifest["generator"].update(GENERATORS[manifest["problem"]])
    manifest["generator"].update(RELEASED_GENERATORS[manifest["problem"]])
    for name, entry in manifest["sources"].items():
        entry.update(RELEASED_SOURCES["eps_pdr" if name.startswith("eps_pdr_") else name])
    return manifest


SOURCE_INFO = {
    "pdr": {"description": "priority dispatching rules, job rule x machine rule, action mask on; "
                           "slot k is rule names[k]",
            "order": "rule"},
    "random": {"description": "uniformly random feasible action at every step",
               "order": "generation"},
    "ga_hof": {"description": "genetic algorithm hall of fame", "order": "makespan_asc"},
    "ga_pop": {"description": "final genetic algorithm population (holds many duplicates)",
               "order": "generation"},
}


def eps_info(eps: str) -> dict:
    return {"description": f"PDR (MWR/LWR/LOPNR/MOPNR x SPT, cycled per trajectory) that takes a "
                           f"uniformly random feasible action with probability {eps}; makespans "
                           f"computed by replay during conversion",
            "order": "generation", "eps": float(eps)}


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def instance_stats(split) -> dict:
    pts = split.op_pts()
    nonzero = np.concatenate([p[p > 0] for p in pts])
    machines = np.concatenate([(p > 0).sum(1) for p in pts])
    return {"processing_time_range": [int(nonzero.min()), int(nonzero.max())],
            "ops_per_job_range": [int(split.job_length.min()), int(split.job_length.max())],
            "machines_per_op_range": [int(machines.min()), int(machines.max())]}


def convert(train_path: str, val_path: str, dataset_id: str, out_root: str,
            paper_era=None, replay_check: int = 0, released: bool = False) -> str:
    t0 = time.time()
    out_dir = os.path.join(out_root, *dataset_id.split("/"))
    problem = dataset_id.split("/")[0]
    print(f"\n=== {dataset_id}: {train_path} + {val_path} -> {out_dir}")

    instances = read_legacy(train_path)
    val_instances = read_legacy(val_path)
    train_split = legacy_split(instances)
    val_split = legacy_split(val_instances)
    n_j, n_m = train_split.job_length.shape[1], train_split.op_pt.shape[2]

    all_rules = list(instances[0]["rules"])
    kept_rules = [old for old, _ in legacy_pdr_rules(instances)]
    generator = dict(GENERATORS.get(problem, {}))
    generator["observed_train"] = instance_stats(train_split)
    generator["observed_val"] = instance_stats(val_split)
    provenance = {
        "converted_from": {
            "train": {"file": os.path.basename(train_path), "md5": md5(train_path)},
            "val": {"file": os.path.basename(val_path), "md5": md5(val_path)},
        },
        "paper_era_names": paper_era or [],
        "dropped_pdr_rules": [legacy_rule_name(r) or r for r in all_rules if r not in kept_rules],
        "renamed_pdr_rules": "legacy names end in _masked (e.g. MWR_EST_masked -> MWR_EST)",
        "notes": "generator seeds and parameters were not stored with the legacy files",
    }
    manifest = fmt.new_manifest(
        dataset_id, n_j, n_m,
        description=f"{problem.upper()} {n_j}x{n_m} offline dataset converted from "
                    f"{os.path.basename(train_path)}",
        generator=generator, provenance=provenance)

    manifest["splits"]["train"] = dict(file="train.npz", description="training instances",
                                       **fmt.write_split(os.path.join(out_dir, "train.npz"),
                                                         train_split.job_lengths(),
                                                         train_split.op_pts()))
    manifest["splits"]["val"] = dict(file="val.npz", description="validation instances",
                                     **fmt.write_split(os.path.join(out_dir, "val.npz"),
                                                       val_split.job_lengths(),
                                                       val_split.op_pts()))

    coverage = legacy_coverage(instances)
    op_pts = train_split.op_pts()
    legacy_sources = {}
    for name, n_cov in coverage.items():
        src = legacy_source(instances, name, n_cov)
        legacy_sources[name] = src
        makespan = src.makespan
        if name.startswith("eps_pdr_"):
            makespan = np.stack([
                replay_makespans(train_split.job_length[i], op_pts[i], src.trajectories(i))
                for i in tqdm.trange(n_cov, desc=f"replaying {name}")])
        info = SOURCE_INFO.get(name) or eps_info(name[len("eps_pdr_"):])
        file = f"sources/{name}.npz"
        entry = fmt.write_source(os.path.join(out_dir, file),
                                 [src.trajectories(i) for i in range(n_cov)],
                                 [makespan[i, :src.counts[i]] for i in range(n_cov)],
                                 names=src.names)
        manifest["sources"][name] = dict(file=file, **info, **entry)
        print(f"  {name:14s} {n_cov} instances x {entry['per_instance']} trajectories")
    if released:
        add_released_metadata(manifest)
    fmt.write_manifest(out_dir, manifest)

    verify(out_dir, train_split, val_split, legacy_sources)
    if replay_check:
        replay_verify(out_dir, replay_check)
    print(f"  done in {time.time() - t0:.0f}s")
    return out_dir


def verify(out_dir: str, train_split, val_split, legacy_sources) -> None:
    """Read the dataset back and require it to equal the legacy data exactly."""
    ds = OfflineDataset.from_dir(out_dir)
    problems = validate(ds)
    if problems:
        raise AssertionError(f"{out_dir} is not valid:\n  " + "\n  ".join(problems))
    for name, legacy in [("train", train_split), ("val", val_split)]:
        split = ds.split(name)
        assert np.array_equal(split.job_length, legacy.job_length), name
        assert np.array_equal(split.n_ops, legacy.n_ops), name
        assert all(np.array_equal(a, b) for a, b in zip(split.op_pts(), legacy.op_pts())), name
    for name, legacy in legacy_sources.items():
        src = ds.source(name)
        assert src.num_instances == legacy.num_instances, name
        assert np.array_equal(src.counts, legacy.counts), name
        assert np.array_equal(src.actions.astype(np.int64), legacy.actions.astype(np.int64)), name
        assert src.names == legacy.names, name
        if name.startswith("eps_pdr_"):
            assert np.isfinite(src.makespan).all() and (src.makespan > 0).all(), name
        else:
            assert np.array_equal(src.makespan, legacy.makespan), name
    print("  verified: arrays identical to the legacy file, dataset valid")


def replay_verify(out_dir: str, n_instances: int) -> None:
    """Replay every trajectory of the first instances; stored makespans must match."""
    ds = OfflineDataset.from_dir(out_dir)
    train = ds.split("train")
    op_pts = train.op_pts()
    worst = 0.0
    for name in ds.source_names:
        src = ds.source(name)
        for i in range(min(n_instances, src.num_instances)):
            spans = replay_makespans(train.job_length[i], op_pts[i], src.trajectories(i))
            worst = max(worst, float(np.abs(spans - src.makespans(i)).max()))
    print(f"  replay check ({n_instances} instances, all sources): "
          f"max |replayed - stored makespan| = {worst:g}")
    if worst > 1e-6:
        raise AssertionError(f"{out_dir}: replayed makespans differ from the stored ones")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true", help="convert the five released datasets")
    p.add_argument("--legacy_root", default="./train_datasets",
                   help="folder holding fjsp/ and jsp/ with the legacy files (for --all)")
    p.add_argument("--train", help="legacy training file")
    p.add_argument("--val", help="legacy validation file")
    p.add_argument("--dataset", help="id of the new dataset, e.g. fjsp/sd1_10x5")
    p.add_argument("--out_root", default="./train_datasets", help="where datasets are written")
    p.add_argument("--only", nargs="+", help="with --all: convert only these dataset ids")
    p.add_argument("--replay_check", type=int, default=0,
                   help="replay all trajectories of the first N instances as a check")
    p.add_argument("--update_metadata", action="store_true",
                   help="with --all: only refresh the descriptive manifest fields (seeds, notes) "
                        "of already converted datasets; arrays are not rewritten")
    args = p.parse_args()

    if args.all and args.update_metadata:
        for spec in RELEASED:
            if args.only and spec["dataset"] not in args.only:
                continue
            out_dir = os.path.join(args.out_root, *spec["dataset"].split("/"))
            manifest = add_released_metadata(fmt.read_manifest(out_dir))
            fmt.write_manifest(out_dir, manifest, update_checksums=False)
            print(f"updated the manifest of {spec['dataset']}")
    elif args.all:
        for spec in RELEASED:
            if args.only and spec["dataset"] not in args.only:
                continue
            convert(os.path.join(args.legacy_root, spec["train"]),
                    os.path.join(args.legacy_root, spec["val"]), spec["dataset"],
                    args.out_root, paper_era=spec["paper_era"], replay_check=args.replay_check,
                    released=True)
    elif args.train and args.val and args.dataset:
        convert(args.train, args.val, args.dataset, args.out_root,
                replay_check=args.replay_check)
    else:
        p.error("pass --all, or --train, --val and --dataset")


if __name__ == "__main__":
    main()
