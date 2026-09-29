"""Evaluate checkpoints, heuristics and the logged behaviour on a real-world split.

    python eval_realworld.py checkpoints/amazon_lmrrc/<run>            # run folder (config.yml + best_det_offline.pt)
    python eval_realworld.py checkpoints/amazon_lmrrc/<run>/latest_check.pt --num_samples 16
    python eval_realworld.py logged nn zone_nn random --dataset amazon_lmrrc --split test

Positional arguments are checkpoints (run folder or .pt next to a config.yml)
or baseline names: ``logged`` (the real driver / courier routes), or any
behaviour of realworld/routing/behaviors.py (``nn``, ``zone_nn``, ``random``,
``eps_nn_0.1``, ...).  Writes per-episode CSVs and a summary JSON to
``--save_path`` and prints a summary table.
"""
import argparse
import json
import os

import pandas as pd
import torch
import yaml

from realworld.config import make_config_class, reward_config
from realworld.evaluation import evaluate, load_split
from realworld.nets import build_eval_net
from realworld.registry import get_dataset
from realworld.routing.behaviors import make_behavior
from realworld.routing.core import RoutingTensors, model_policy
from train import METHODS

SUMMARY_KEYS = ["return", "time", "time_ratio", "lateness", "late_stops", "zone_leave", "zone_revisits",
                "amazon_score", "hr3", "krc", "lsd", "ed", "logged_time"]


def load_checkpoint(path: str, device: str):
    folder, ckpt = (path, os.path.join(path, "best_det_offline.pt")) if os.path.isdir(path) else \
        (os.path.dirname(path), path)
    with open(os.path.join(folder, "config.yml")) as f:
        raw = yaml.safe_load(f)
    method = raw.pop("method", "cdqac")
    cfg_cls = make_config_class(METHODS[method]["config"])
    fields = set(cfg_cls.__dataclass_fields__)
    config = cfg_cls(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items() if k in fields})
    config.device = device
    net = build_eval_net(config, method)
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    net.load_state_dict(sd["actor_net"])
    net.eval()
    return config, net, sd["feature_stats"], method


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("policies", nargs="+")
    p.add_argument("--dataset", default=None, help="default: from the checkpoint config")
    p.add_argument("--data_root", default=None)
    p.add_argument("--split", default="test")
    p.add_argument("--num_instances", type=int, default=None)
    p.add_argument("--num_samples", type=int, default=1, help=">1: best of N sampled rollouts")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--no_reference_metrics", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save_path", default="./results/realworld")
    args = p.parse_args(argv)

    loaded = {}
    base_cfg = None
    for pol in args.policies:
        if os.path.exists(pol):
            loaded[pol] = load_checkpoint(pol, args.device)
            base_cfg = base_cfg or loaded[pol][0]
    dataset = args.dataset or (base_cfg.dataset if base_cfg else "amazon_lmrrc")
    data_root = args.data_root or (base_cfg.data_root if base_cfg else "./realworld_data")
    ds = get_dataset(dataset)
    if base_cfg is None:
        from realworld.config import resolve
        base_cfg = make_config_class(METHODS["cdqac"]["config"])()
        base_cfg.dataset = dataset
        resolve(base_cfg, ds)
    reward = reward_config(base_cfg)
    eps = load_split(data_root, dataset, args.split, num_instances=args.num_instances)
    os.makedirs(args.save_path, exist_ok=True)
    table = {}
    for pol in args.policies:
        rt = RoutingTensors(eps, device=args.device)
        if pol in loaded:
            cfg, net, stats, method = loaded[pol]
            rt.set_stats(stats)
            label = f"{method}:{os.path.basename(os.path.normpath(pol))}"
            policy = model_policy(net, deterministic=args.num_samples == 1)
            rows, summ = evaluate(eps, rt, policy, reward, ds, batch_size=args.batch_size,
                                  reference_metrics=not args.no_reference_metrics,
                                  num_samples=args.num_samples, label=label)
        elif pol in ("logged", ds.logged_source):
            label = f"logged:{ds.logged_source}"
            rows, summ = evaluate(eps, rt, None, reward, ds, reference_metrics=not args.no_reference_metrics)
        else:
            label = pol
            rows, summ = evaluate(eps, rt, make_behavior(pol, seed=0), reward, ds, batch_size=args.batch_size,
                                  needs_state=False, reference_metrics=not args.no_reference_metrics,
                                  num_samples=args.num_samples, label=label)
        table[label] = {k: summ[k] for k in SUMMARY_KEYS if k in summ}
        safe = label.replace(":", "_").replace("/", "_")
        pd.DataFrame(rows).to_csv(os.path.join(args.save_path, f"{dataset}_{args.split}_{safe}.csv"), index=False)
    df = pd.DataFrame(table).T
    print(df.to_string(float_format=lambda v: f"{v:.4g}"))
    with open(os.path.join(args.save_path, f"{dataset}_{args.split}_summary.json"), "w") as f:
        json.dump(table, f, indent=1)


if __name__ == "__main__":
    main()
