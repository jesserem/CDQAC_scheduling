"""Unified evaluation script for all trained models in cdqac/methods.

Works with any checkpoint produced by the training scripts (CDQAC, BC, D-MSAC,
IQL, mQRDQN, TD3+AWR). The network type is inferred from the run's config.yml,
and the instance format (FJSP .fjs / JSP .jsp) is inferred from the files in
the evaluation folder.

Usage:
    python evaluation/eval_model.py <checkpoint> [--eval_folder ./data/SD1/10x5]
        [--save_path ./results] [--num_samples 1] [--device cuda]

<checkpoint> is either a run folder (containing config.yml and
best_det_offline.pt) or a .pt file. If no config.yml is found next to the
checkpoint, the network architecture is inferred from the state dict itself.

With --num_samples 1 the policy acts greedily (deterministic). With
--num_samples N > 1, N solutions are sampled in a single batched rollout and
the best (lowest) makespan is kept.

Results are saved as a CSV with three columns: name, makespan, runtime.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from cdqac.network.main_model import ActorNet, mQRDQNNet
from cdqac.utils import do_run, load_data_from_files, load_data_from_files_jsp, load_yml
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums


def resolve_checkpoint(checkpoint_path):
    """Return (weights_file, run_folder) from a run folder or a .pt file path."""
    if os.path.isdir(checkpoint_path):
        run_folder = checkpoint_path
        weights_file = os.path.join(run_folder, "best_det_offline.pt")
        if not os.path.exists(weights_file):
            raise FileNotFoundError(
                f"No best_det_offline.pt in {run_folder}; pass the .pt file directly instead.")
    else:
        weights_file = checkpoint_path
        run_folder = os.path.dirname(os.path.abspath(checkpoint_path))
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Checkpoint not found: {weights_file}")
    return weights_file, run_folder


def infer_config_from_state_dict(state_dict):
    """Reconstruct the architecture hyperparameters from a saved actor_net state dict.

    Used when no config.yml is available next to the checkpoint. Only the keys
    that build_network() reads are produced; dropout is irrelevant in eval mode.
    """
    import re

    def attention_shape(prefix):
        # blocks are named {prefix}.{block}.attention_{head}.W
        blocks = {}
        for key in state_dict:
            m = re.match(rf"{re.escape(prefix)}\.(\d+)\.attention_(\d+)\.W$", key)
            if m:
                block, head = int(m.group(1)), int(m.group(2))
                blocks.setdefault(block, set()).add(head)
        num_heads = tuple(len(blocks[b]) for b in sorted(blocks))
        out_dims = tuple(state_dict[f"{prefix}.{b}.attention_0.W"].shape[1] for b in sorted(blocks))
        return num_heads, out_dims

    num_heads_OAB, layer_fea_output_dim = attention_shape("feature_exact.op_attention_blocks")
    num_heads_MAB, _ = attention_shape("feature_exact.mch_attention_blocks")

    config = {
        "fea_j_input_dim": state_dict["feature_exact.op_attention_blocks.0.attention_0.W"].shape[0],
        "fea_m_input_dim": state_dict["feature_exact.mch_attention_blocks.0.attention_0.W"].shape[0],
        "layer_fea_output_dim": list(layer_fea_output_dim),
        "num_heads_OAB": list(num_heads_OAB),
        "num_heads_MAB": list(num_heads_MAB),
        "dropout_prob_actor": 0,
        "use_mask": True,
    }

    if any(key.startswith("actor.") for key in state_dict):
        # ActorNet: actor.net.{0,2,4,...}.weight are the MLP's linear layers
        mlp_layers = [k for k in state_dict if re.match(r"actor\.net\.\d+\.weight$", k)]
        config["update_freq_policy"] = None  # marks this as an actor-based model
        config["num_mlp_layers_actor"] = len(mlp_layers)
        config["hidden_dim_actor"] = state_dict["actor.net.0.weight"].shape[0]
    elif any(key.startswith("Q.") for key in state_dict):
        mlp_layers = sorted(
            (k for k in state_dict if re.match(r"Q\.q\.net\.\d+\.weight$", k)),
            key=lambda k: int(k.split(".")[3]))
        config["num_mlp_layers_actor"] = len(mlp_layers)
        config["hidden_dim_actor"] = state_dict["Q.q.net.0.weight"].shape[0]
        config["num_quantiles"] = state_dict[mlp_layers[-1]].shape[0]
        config["use_adv_net"] = any(key.startswith("Q.v.") for key in state_dict)
    else:
        raise ValueError("Unrecognized checkpoint: no actor.* or Q.* keys in the state dict")
    return config


def build_network(model_config, device):
    # Actor-based methods (cdqac, bc, d_msac, iql, td3_awr) all have
    # update_freq_policy or beta in their config (bc configs are recognized by
    # their group instead); mqrdqn has none of these.
    if ("update_freq_policy" in model_config or "beta" in model_config
            or model_config.get("group") == "bc"):
        net = ActorNet(
            fea_j_input_dim=model_config["fea_j_input_dim"],
            fea_m_input_dim=model_config["fea_m_input_dim"],
            layer_fea_output_dim=model_config["layer_fea_output_dim"],
            num_heads_OAB=model_config["num_heads_OAB"],
            num_heads_MAB=model_config["num_heads_MAB"],
            num_mlp_layers_actor=model_config["num_mlp_layers_actor"],
            hidden_dim_actor=model_config["hidden_dim_actor"],
            dropout_prob=model_config.get("dropout_prob_actor", 0),
        )
    else:
        net = mQRDQNNet(
            fea_j_input_dim=model_config["fea_j_input_dim"],
            fea_m_input_dim=model_config["fea_m_input_dim"],
            layer_fea_output_dim=model_config["layer_fea_output_dim"],
            num_heads_OAB=model_config["num_heads_OAB"],
            num_heads_MAB=model_config["num_heads_MAB"],
            num_mlp_layers_critic=model_config["num_mlp_layers_actor"],
            use_adv_net=model_config["use_adv_net"],
            hidden_dim_critic=model_config["hidden_dim_actor"],
            dropout_prob=model_config.get("dropout_prob_actor", 0),
            num_quantiles=model_config["num_quantiles"],
        )
    return net.to(device)


def load_normalization(weights, device):
    if "mean_fea_j" in weights:
        return {k: weights[k].to(device) for k in
                ("mean_fea_j", "std_fea_j", "mean_fea_m", "std_fea_m",
                 "mean_fea_pairs", "std_fea_pairs")}
    zeros = torch.tensor(0).to(device)
    ones = torch.tensor(1).to(device)
    return {"mean_fea_j": zeros, "std_fea_j": ones,
            "mean_fea_m": zeros, "std_fea_m": ones,
            "mean_fea_pairs": zeros, "std_fea_pairs": ones}


def load_instances(eval_folder):
    extensions = {os.path.splitext(f)[1] for _, _, files in os.walk(eval_folder) for f in files}
    if ".fjs" in extensions:
        return load_data_from_files(eval_folder)
    if ".jsp" in extensions:
        return load_data_from_files_jsp(eval_folder)
    raise ValueError(f"No .fjs or .jsp instance files found in {eval_folder}")


def resolve_save_path(save_path, weights_file, run_folder, eval_folder, num_samples):
    if save_path.lower().endswith(".csv"):
        out_dir = os.path.dirname(save_path) or "."
        out_file = save_path
    else:
        out_dir = save_path
        ckpt_stem = os.path.splitext(os.path.basename(weights_file))[0]
        if ckpt_stem == "best_det_offline":
            run_name = os.path.basename(os.path.normpath(run_folder))
        else:
            run_name = ckpt_stem
        parts = os.path.normpath(eval_folder).split(os.sep)
        dataset_name = "_".join(p for p in parts[-2:] if p not in (".", ".."))
        out_file = os.path.join(out_dir, f"{run_name}_{dataset_name}_samples{num_samples}.csv")
    os.makedirs(out_dir, exist_ok=True)
    return out_file


def evaluate(net, names, job_lengths, op_pts, device, use_mask, num_samples, norm):
    net.eval()
    results = {"name": [], "makespan": [], "runtime": []}
    deterministic = num_samples == 1
    for instance_name, job_length, op_pt in zip(names, job_lengths, op_pts):
        n_j = job_length.shape[0]
        n_m = op_pt.shape[1]
        job_length_batch = np.tile(np.expand_dims(job_length, axis=0), (num_samples, 1))
        op_pt_batch = np.tile(np.expand_dims(op_pt, axis=0), (num_samples, 1, 1))

        env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m, device=device, mask_actions=use_mask)
        state = env.set_initial_data(job_length_batch, op_pt_batch)
        makespan, runtime = do_run(state, env, net, num_samples, deterministic=deterministic, **norm)

        print(f"Instance: {instance_name}  Makespan: {makespan}  Runtime: {runtime:.3f}s")
        results["name"].append(instance_name)
        results["makespan"].append(makespan)
        results["runtime"].append(runtime)
    return pd.DataFrame.from_dict(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint",
                        help="Run folder (with config.yml and best_det_offline.pt) or a .pt file")
    parser.add_argument("--eval_folder", default="./data/SD1/10x5",
                        help="Folder with evaluation instances (.fjs or .jsp)")
    parser.add_argument("--save_path", default="./results",
                        help="Output folder (auto-named CSV) or an explicit .csv file path")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="1 = deterministic (best actions); >1 = sample this many "
                             "solutions and keep the best")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (default: cuda if available)")
    args = parser.parse_args()

    if args.num_samples < 1:
        parser.error("--num_samples must be >= 1")

    weights_file, run_folder = resolve_checkpoint(args.checkpoint)
    weights = torch.load(weights_file, map_location=args.device, weights_only=True)

    config_file = os.path.join(run_folder, "config.yml")
    if os.path.exists(config_file):
        model_config = load_yml(config_file)
    else:
        print(f"No config.yml next to {weights_file}; inferring architecture from the state dict")
        model_config = infer_config_from_state_dict(weights["actor_net"])

    net = build_network(model_config, args.device)
    net.load_state_dict(weights["actor_net"])
    norm = load_normalization(weights, args.device)

    names, job_lengths, op_pts = load_instances(args.eval_folder)
    print(f"Evaluating {len(names)} instances from {args.eval_folder} "
          f"with num_samples={args.num_samples} on {args.device}")

    res_df = evaluate(net, names, job_lengths, op_pts, args.device,
                      model_config.get("use_mask", True), args.num_samples, norm)

    out_file = resolve_save_path(args.save_path, weights_file, run_folder,
                                 args.eval_folder, args.num_samples)
    res_df.to_csv(out_file, index=False)
    print(f"\nMean makespan: {res_df['makespan'].mean():.2f}  "
          f"Mean runtime: {res_df['runtime'].mean():.3f}s")
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
