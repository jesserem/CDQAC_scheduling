"""Inspect and check offline datasets.

    python -m cdqac.data list                              # datasets under ./train_datasets
    python -m cdqac.data info fjsp/sd1_10x5                # what a dataset holds
    python -m cdqac.data validate fjsp/sd1_10x5 --replay 5 # format check + makespan replay

A dataset can be given by id (resolved under --data_root) or by its directory path.
"""
import argparse
import json
import os
import sys

import numpy as np

from cdqac.data import format as fmt
from cdqac.data.dataset import OfflineDataset, list_datasets, validate


def open_dataset(ref: str, data_root: str) -> OfflineDataset:
    if os.path.isfile(os.path.join(ref, fmt.MANIFEST_FILE)):
        return OfflineDataset.from_dir(ref)
    return OfflineDataset.open(data_root, ref)


def cmd_list(args) -> int:
    ids = list_datasets(args.data_root)
    if not ids:
        print(f"no datasets under {args.data_root}")
        return 1
    for dataset_id in ids:
        ds = OfflineDataset.open(args.data_root, dataset_id)
        m = ds.manifest
        splits = ", ".join(f"{k} {v['num_instances']}" for k, v in m["splits"].items())
        sources = [n if ds.source_downloaded(n) else f"({n})" for n in m["sources"]]
        print(f"{dataset_id:22s} {m['n_j']}x{m['n_m']:<3d} {splits:22s} "
              f"sources: {', '.join(sources)}")
    return 0


def cmd_info(args) -> int:
    ds = open_dataset(args.dataset, args.data_root)
    m = ds.manifest
    print(f"{m['id']}  ({ds.path})")
    if m.get("description"):
        print(f"  {m['description']}")
    print(f"  problem {m['problem']}, {m['n_j']} jobs x {m['n_m']} machines, "
          f"action = {m['action_encoding']}")
    if m.get("generator"):
        print(f"  generator: {json.dumps(m['generator'])}")
    print("  splits:")
    for name, e in m["splits"].items():
        print(f"    {name:6s} {e['num_instances']:5d} instances, {e['n_ops'][0]}-{e['n_ops'][1]} ops"
              f"   ({e['file']})")
    print("  sources:            instances  per inst  order          makespan mean [min, max]")
    for name, e in m["sources"].items():
        if not ds.source_downloaded(name):
            print(f"    {name:16s} {e['num_instances']:9d}  {str(e['per_instance']):>8s}  "
                  f"{e.get('order', ''):13s}  (not downloaded)")
            continue
        src = ds.source(name)
        spans = np.concatenate([src.makespans(i) for i in range(src.num_instances)])
        print(f"    {name:16s} {e['num_instances']:9d}  {str(e['per_instance']):>8s}  "
              f"{e.get('order', ''):13s}  {spans.mean():8.1f} [{spans.min():g}, {spans.max():g}]")
        if src.names is not None and args.verbose:
            print(f"      names: {', '.join(src.names)}")
    if m.get("provenance") and args.verbose:
        print(f"  provenance: {json.dumps(m['provenance'], indent=2)}")
    return 0


def cmd_validate(args) -> int:
    ds = open_dataset(args.dataset, args.data_root)
    problems = validate(ds, check_checksums=not args.no_checksums)
    if args.replay and not problems:
        from cdqac.data.replay import replay_makespans
        train = ds.split("train")
        op_pts = train.op_pts()
        for name in ds.source_names:
            if not ds.source_downloaded(name):
                continue
            src = ds.source(name)
            worst = 0.0
            try:
                for i in range(min(args.replay, src.num_instances)):
                    spans = replay_makespans(train.job_length[i], op_pts[i], src.trajectories(i))
                    worst = max(worst, float(np.abs(spans - src.makespans(i)).max()))
            except ValueError as e:  # an action the action mask forbids
                problems.append(f"source {name}, instance {i}: {e}")
                continue
            print(f"  replay {name:16s} first {min(args.replay, src.num_instances)} instances: "
                  f"max |replayed - stored makespan| = {worst:g}")
            if worst > 1e-6:
                problems.append(f"source {name}: stored makespans differ from the replay")
    if problems:
        print(f"{ds.id}: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    missing = [n for n in ds.source_names if not ds.source_downloaded(n)]
    print(f"{ds.id}: valid" + (f" (not downloaded: {', '.join(missing)})" if missing else ""))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m cdqac.data", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default="./train_datasets")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list the datasets under --data_root")
    info = sub.add_parser("info", help="describe one dataset")
    info.add_argument("dataset", help="dataset id or directory")
    info.add_argument("-v", "--verbose", action="store_true", help="also print rule names and provenance")
    val = sub.add_parser("validate", help="check one dataset against the format")
    val.add_argument("dataset", help="dataset id or directory")
    val.add_argument("--replay", type=int, default=0,
                     help="also replay every trajectory of the first N instances")
    val.add_argument("--no_checksums", action="store_true", help="skip the sha256 check")
    args = p.parse_args(argv)
    return {"list": cmd_list, "info": cmd_info, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
