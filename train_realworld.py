"""Train offline-RL methods on the real-world routing datasets.

Same methods, hyperparameters and CLI as ``train.py``; the dataset, reward,
SMDP and routing-network options come from ``realworld/config.py``:

    python train_realworld.py --dataset amazon_lmrrc                      # CDQAC on the driver routes
    python train_realworld.py --method iql --dataset lade_d --sources courier,random
    python train_realworld.py --dataset amazon_lmrrc --sources driver,eps_nn* --instance_filter route_score=High
    python train_realworld.py --dataset lade_p --gamma 0.999 --smdp_discount true --sojourn_unit 1

SMDP datasets (LaDe) use per-transition discounts gamma ** (tau / sojourn_unit)
when ``smdp_discount`` is on: the buffer returns them with every batch and they
replace the trainer's scalar discount for that update.  Every method in
``cdqac.methods`` only uses ``self.discount`` inside ``r + (1 - done) * discount
* next_value``, so a [B, 1] tensor gives the exact SMDP target without touching
the method code.
"""
import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import pyrallis
import torch
import tqdm

from cdqac.utils import write_yml_file
from realworld.config import make_config_class, resolve, reward_config, source_list
from realworld.evaluation import evaluate, load_split
from realworld.nets import RoutingNets
from realworld.registry import HIGHER_IS_BETTER, get_dataset
from realworld.routing.buffer import RoutingBuffer
from realworld.routing.core import RoutingTensors, model_policy
from train import METHODS, get_state_dict, set_seed, wandb_init

try:
    import wandb
except ImportError:
    wandb = None


def train(method: str, config) -> dict:
    spec = METHODS[method]
    ds = get_dataset(config.dataset)
    resolve(config, ds)
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        config.device = "cpu"
    if getattr(config, "use_disk_buffer", False):
        raise ValueError("use_disk_buffer is not needed for the routing buffer (states are rebuilt lazily)")
    if getattr(config, "n_step", 1) > 1:
        raise ValueError("n_step > 1 is not supported by the routing buffer")
    set_seed(config.seed, device=config.device)
    td3_batches = spec.get("td3_batches", False)
    buffer_device = config.buffer_device or config.device

    train_eps = load_split(config.data_root, config.dataset, config.train_split, config.instance_filter,
                           config.num_instances)
    eval_eps = load_split(config.data_root, config.dataset, config.eval_split)
    eval_inst = np.arange(eval_eps.num_instances)
    if config.num_eval_instances is not None and config.num_eval_instances < len(eval_inst):
        eval_inst = np.sort(np.random.default_rng(0).choice(len(eval_inst), config.num_eval_instances,
                                                            replace=False))
    reward = reward_config(config)
    print(f"[train_realworld] {config.dataset}: {train_eps.num_instances} train instances, "
          f"sources={config.sources}, reward={reward}")
    buffer = RoutingBuffer(train_eps, reward, sources=source_list(config.sources), time_source=config.time_source,
                           gamma=config.gamma, smdp_discount=config.smdp_discount,
                           sojourn_unit=config.sojourn_unit, device=buffer_device,
                           max_trajectories=config.max_trajectories, seed=config.seed)
    print(json.dumps(buffer.describe(), indent=1))
    stats = buffer.normalize_state(config.stats_samples)
    config.node_dim, config.edge_dim, config.global_dim = buffer.rt.node_dim, buffer.rt.edge_dim, buffer.rt.global_dim

    save_folder = None
    if config.save_folder is not None:
        save_folder = os.path.join(config.save_folder, config.dataset, config.name + "_seed_" + str(config.seed))
        os.makedirs(save_folder, exist_ok=True)
        cfg = asdict(config)
        cfg["method"] = method
        write_yml_file(os.path.join(save_folder, "config.yml"), cfg)

    trainer, eval_net = spec["build"](config, nets=RoutingNets)
    eval_rt = RoutingTensors(eval_eps, device=config.device)
    eval_rt.set_stats(stats)
    feature_stats = {"feature_stats": stats, "method": method}
    sign = 1.0 if config.select_metric in HIGHER_IS_BETTER else -1.0
    best = -np.inf
    history = []

    if config.use_wandb:
        wandb_init(asdict(config))

    def run_eval(step):
        nonlocal best
        eval_net.eval()
        _, summ = evaluate(eval_eps, eval_rt, model_policy(eval_net), reward, ds, instances=eval_inst,
                           batch_size=config.eval_batch_size, reference_metrics=config.eval_reference_metrics)
        eval_net.train()
        summ = {k: v for k, v in summ.items() if k != "inst"}
        history.append(dict(step=step, **summ))
        if config.use_wandb:
            wandb.log({f"eval/{k}": v for k, v in summ.items()}, step=step)
        score = sign * summ[config.select_metric]
        if score > best:
            best = score
            save("best_det_offline.pt")
        save("latest_check.pt")
        print(f"step {step}: " + ", ".join(f"{k}={summ[k]:.4g}" for k in
                                          ("return", "time", "time_ratio", "amazon_score", "krc") if k in summ))
        return summ

    def save(filename):
        if save_folder is None:
            return
        sd = get_state_dict(trainer, feature_stats, spec.get("save_with_critic", False))
        torch.save(sd, os.path.join(save_folder, filename))

    def update(batch, disc):
        trainer.discount = disc  # [B, 1]: gamma ** (tau / unit) for SMDP data, gamma otherwise
        return trainer.train(batch)

    if config.do_epoch:
        step = 0
        for epoch in range(1, config.train_epochs + 1):
            gen = buffer.epoch_generator(config.batch_size, device=config.device, td3=td3_batches)
            for batch, disc in tqdm.tqdm(gen, total=(len(buffer) + config.batch_size - 1) // config.batch_size,
                                         desc=f"Epoch {epoch}/{config.train_epochs}"):
                info = update(batch, disc)
                if config.use_wandb:
                    wandb.log(info, step=step)
                step += 1
            if epoch % config.eval_every_epoch == 0:
                run_eval(step)
    else:
        for step in tqdm.tqdm(range(config.num_train_step_offline), desc="Training"):
            batch, disc = buffer.sample(config.batch_size, device=config.device, td3=td3_batches)
            info = update(batch, disc)
            if config.use_wandb:
                wandb.log(info, step=step)
            if (step + 1) % config.eval_freq == 0:
                run_eval(step + 1)
    save("model_final_offline.pt")
    if save_folder is not None:
        with open(os.path.join(save_folder, "eval_history.json"), "w") as f:
            json.dump(history, f, indent=1)
    return dict(history=history, save_folder=save_folder)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", choices=sorted(METHODS), default="cdqac")
    args, remaining = parser.parse_known_args(argv)
    cfg_cls = make_config_class(METHODS[args.method]["config"])
    config = pyrallis.parse(config_class=cfg_cls, args=remaining)
    train(args.method, config)


if __name__ == "__main__":
    main()
