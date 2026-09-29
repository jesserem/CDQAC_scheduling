"""Download, convert, augment and describe the real-world offline-RL datasets.

    python prepare_realworld.py list
    python prepare_realworld.py download --dataset amazon_lmrrc --raw_dir ./raw/amazon
    python prepare_realworld.py convert  --dataset amazon_lmrrc --raw_dir ./raw/amazon
    python prepare_realworld.py convert  --dataset lade_d --raw_dir ./raw/lade --opt cities=jl,yt max_stops=100
    python prepare_realworld.py augment  --dataset amazon_lmrrc --split train --behaviors random:5,eps_nn_0.1:5,nn:1
    python prepare_realworld.py describe --dataset amazon_lmrrc --split train

Prepared splits are written to ``<data_root>/<dataset>/<split>.npz``
(see realworld/README.md for the format).  ``augment`` adds synthetic
behaviour trajectories (realworld/routing/behaviors.py) to a split in place,
so the offline datasets can mix real and synthetic behaviour like the FJSP
datasets of the paper.
"""
import argparse
import json
import os
import sys

import torch

from realworld.registry import get_dataset, list_datasets


def _parse_opts(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        else:
            if v in ("True", "true", "False", "false"):
                v = v.lower() == "true"
        out[k] = v
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["list", "download", "convert", "augment", "describe"])
    p.add_argument("--dataset", default="amazon_lmrrc")
    p.add_argument("--raw_dir", default=None, help="raw files (default: ./raw/<dataset>)")
    p.add_argument("--data_root", default="./realworld_data")
    p.add_argument("--out_dir", default=None, help="default: <data_root>/<dataset>")
    p.add_argument("--split", default="train")
    p.add_argument("--behaviors", default="random:1,nn:1",
                   help="augment: comma-separated name:count (see realworld/routing/behaviors.py)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--opt", nargs="*", help="dataset-specific converter options key=value")
    args = p.parse_args(argv)

    if args.command == "list":
        for name in list_datasets():
            s = get_dataset(name)
            print(f"{name:14s} {s.description}\n{'':14s} MDP: {s.mdp}\n{'':14s} license: {s.license}; {s.url}")
        return

    spec = get_dataset(args.dataset)
    raw_dir = args.raw_dir or os.path.join("raw", args.dataset)
    out_dir = args.out_dir or os.path.join(args.data_root, args.dataset)

    if args.command == "download":
        if spec.download is None:
            sys.exit(f"{args.dataset} has no automatic download; see {spec.url}")
        os.makedirs(raw_dir, exist_ok=True)
        spec.download(raw_dir)
    elif args.command == "convert":
        paths = spec.prepare(raw_dir, out_dir, **_parse_opts(args.opt))
        print(json.dumps(paths, indent=2))
    elif args.command == "augment":
        from realworld.episodes import RoutingEpisodeSet
        from realworld.routing.behaviors import generate_behavior
        path = os.path.join(out_dir, f"{args.split}.npz")
        eps = RoutingEpisodeSet.load(path)
        for i, item in enumerate(args.behaviors.split(",")):
            name, _, count = item.partition(":")
            name = name.strip()
            if name in eps.source_names and name != spec.logged_source:
                print(f"replacing existing source {name}")
                eps.remove_sources([name])
            n = generate_behavior(eps, name, per_instance=int(count or 1), device=args.device,
                                  seed=args.seed + 1000 * i)
            print(f"added {n} {name} trajectories")
        eps.save(path)
        print(json.dumps(eps.summary(), indent=2))
    elif args.command == "describe":
        from realworld.episodes import RoutingEpisodeSet
        from realworld.routing.buffer import RoutingBuffer
        from realworld.routing.core import RewardConfig
        eps = RoutingEpisodeSet.load(os.path.join(out_dir, f"{args.split}.npz"))
        print(json.dumps(eps.summary(), indent=2))
        print(json.dumps({k: v for k, v in eps.meta.items() if k not in ("options",)}, indent=2, default=str)[:4000])
        reward = RewardConfig(return_to_depot=spec.return_to_depot, **spec.reward)
        buf = RoutingBuffer(eps, reward, sources=None, time_source=spec.time_source, device=args.device)
        print(json.dumps(buf.describe(), indent=2))


if __name__ == "__main__":
    main()
