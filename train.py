"""Unified training entry point for all offline-RL methods in this repo.

Select the method with ``--method`` (default: cdqac); every other command-line
argument is forwarded to pyrallis and overrides fields of that method's
TrainConfig, e.g.:

    python train.py --train_instance SD1_train_15_10_500.npy --seed 2
    python train.py --method td3_awr --use_disk_buffer false
    python train.py --method iql --q_lr 1e-4

Available methods: cdqac (default), td3_awr, iql, dmsac, mqrdqn, bc.

Every method supports ``--use_disk_buffer true`` (default for td3_awr only):
transitions are stored in memmaps on disk and batches are prepared by
DataLoader workers, so dataset size is bounded by disk instead of RAM. See
cdqac/buffer/disk_buffer.py for the tuning knobs (buffer_dir, num_workers,
sampler_chunk_len, ...), which exist on every method's TrainConfig.
"""
import argparse
import os.path
import pickle
import sys

import torch

from cdqac.buffer.buffer import Buffer
from cdqac.buffer.disk_buffer import (DiskBuffer, TD3TransitionDataset,
                                           RandomBatchSampler, EpochBatchSampler,
                                           make_td3_dataloader, batch_to_device)
from cdqac.network.main_model import ActorNet, QRDQNNet, mQRDQNNet, IQL_value
from cdqac.methods.cdqac import CDQAC
from cdqac.methods.td3_awr import TD3AWR
from cdqac.methods.iql import IQL
from cdqac.methods.d_msac import discrete_mSAC
from cdqac.methods.mqrdqn import mQRDQN
from cdqac.methods.behavioral_cloning import Imitation_Learning
from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from cdqac.configs.cdqac_config import TrainConfig as CDQACConfig
from cdqac.configs.td3_awr_config import TrainConfig as TD3AWRConfig
from cdqac.configs.iql_config import TrainConfig as IQLConfig
from cdqac.configs.d_msac_config import TrainConfig as DMSACConfig
from cdqac.configs.mqrdqn_config import TrainConfig as MQRDQNConfig
from cdqac.configs.bc_config import TrainConfig as BCConfig
from cdqac.utils import (write_yml_file, collect_data_new, remove_duplicate_actions,
                              eval_model_all)
import numpy as np
from typing import List
import tqdm

import uuid
import pyrallis
from dataclasses import asdict
try:
    import wandb
except ImportError:
    wandb = None
import random


def weighted_choice(items, scores, amount):
    # Convert scores to probabilities (lower score -> higher probability)
    weights = [1 / score for score in scores]
    return random.choices(items, weights=weights, k=amount)


def wandb_init(config: dict) -> None:
    wandb.init(
        config=config,
        project=config["project"],
        group=config["group"],
        name=config["name"],
        id=str(uuid.uuid4()),
    )


def load_instance(path: str) -> List:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def set_seed(seed: int, device: str = "cpu") -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Network builders shared across methods
# ---------------------------------------------------------------------------

def make_actor(config) -> ActorNet:
    return ActorNet(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_actor=config.num_mlp_layers_actor,
        hidden_dim_actor=config.hidden_dim_actor,
        dropout_prob=config.dropout_prob_actor,
    ).to(config.device)


def make_qrdqn(config, num_quantiles: int, use_adv_net: bool) -> QRDQNNet:
    return QRDQNNet(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_critic=config.num_mlp_layers_critic,
        num_quantiles=num_quantiles,
        hidden_dim_critic=config.hidden_dim_critic,
        use_adv_net=use_adv_net,
        dropout_prob_q=config.dropout_prob_q,
        n_critics=config.n_critics,
        layer_norm=config.layer_norm,
    ).to(config.device)


def make_value_net(config) -> IQL_value:
    return IQL_value(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_critic=config.num_mlp_layers_value,
        hidden_dim_critic=config.hidden_dim_value,
    ).to(config.device)


# ---------------------------------------------------------------------------
# Per-method trainer builders. Each returns (trainer, eval_net).
# ---------------------------------------------------------------------------

def build_cdqac(config):
    actor_net = make_actor(config)
    target_actor_net = make_actor(config)
    target_actor_net.load_state_dict(actor_net.state_dict())

    if not config.use_qrdqn:
        config.num_quantiles = 1

    q_net = make_qrdqn(config, config.num_quantiles, config.use_adv_net)
    target_net = make_qrdqn(config, config.num_quantiles, config.use_adv_net)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=config.p_lr)
    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=config.q_lr)

    trainer = CDQAC(
        actor_net=actor_net,
        target_actor_net=target_actor_net,
        q_net=q_net,
        target_net=target_net,
        use_calql=config.use_calql,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        target_update_freq=config.target_update_freq,
        update_freq_policy=config.update_freq_policy,
        cql_alpha=config.cql_alpha_offline,
        alpha_multiplier=config.alpha_multiplier,
        tau=config.tau,
        discount=config.gamma,
        max_steps=config.num_train_step_offline // 2,
        device=config.device,
        N=config.num_quantiles,
        use_cql=config.use_cql,
        target_entropy=config.target_entropy,
        anneal_entropy=config.anneal_entropy,
        anneal_lr=config.anneal_lr,
        kappa=config.kappa,
        normalize_q=config.normalize_q,
        max_steps_lr=config.num_train_step_offline,
        backup_entropy=config.backup_entropy,
        use_qrdqn=config.use_qrdqn,
        max_grad_norm=config.max_grad_norm,
        q_pretrain_steps=config.q_pretrain_steps,
        use_codac=config.use_codac,
        entropy_coef=config.entropy_coef,
    )
    return trainer, actor_net


def build_td3_awr(config):
    actor_net = make_actor(config)
    target_actor_net = make_actor(config)
    target_actor_net.load_state_dict(actor_net.state_dict())
    target_actor_net.eval()

    q_net = make_qrdqn(config, num_quantiles=1, use_adv_net=False)
    target_net = make_qrdqn(config, num_quantiles=1, use_adv_net=False)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    value_net = make_value_net(config)

    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=config.p_lr)
    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=config.q_lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=config.v_lr)

    trainer = TD3AWR(
        actor_net=actor_net,
        target_actor_net=target_actor_net,
        q_net=q_net,
        target_net=target_net,
        value_net=value_net,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        value_optimizer=value_optimizer,
        target_update_freq=config.target_update_freq,
        update_freq_policy=config.update_freq_policy,
        discount=config.gamma,
        tau=config.tau,
        awr_temperature=config.awr_temperature,
        awr_adv_clip=config.awr_adv_clip,
        value_expectile=config.value_expectile,
        beta_q=config.beta_q,
        beta_bc=config.beta_bc,
        normalize_q_loss=config.normalize_q_loss,
        critic_bc_coef=config.critic_bc_coef,
        critic_bc_soft=config.critic_bc_soft,
        policy_noise=config.policy_noise,
        noise_clip=config.noise_clip,
        bc_distance=config.bc_distance,
        pre_activ_reg=config.pre_activ_reg,
        gumbel_tau=config.gumbel_tau,
        max_grad_norm=config.max_grad_norm,
        device=config.device,
    )
    return trainer, actor_net


def build_iql(config):
    actor_net = make_actor(config)

    q_net = make_qrdqn(config, num_quantiles=1, use_adv_net=False)
    target_net = make_qrdqn(config, num_quantiles=1, use_adv_net=False)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    value_net = make_value_net(config)

    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=config.p_lr)
    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=config.q_lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=config.v_lr)

    trainer = IQL(
        actor_net=actor_net,
        value_net=value_net,
        value_optimizer=value_optimizer,
        q_net=q_net,
        target_net=target_net,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        target_update_freq=config.target_update_freq,
        update_freq_policy=config.update_freq_policy,
        tau=config.tau,
        discount=config.gamma,
        device=config.device,
        iql_tau=config.iql_tau,
        beta=config.beta,
    )
    return trainer, actor_net


def build_dmsac(config):
    actor_net = make_actor(config)
    target_actor_net = make_actor(config)
    target_actor_net.load_state_dict(actor_net.state_dict())

    if not config.use_qrdqn:
        config.num_quantiles = 1

    q_net = make_qrdqn(config, config.num_quantiles, config.use_adv_net)
    target_net = make_qrdqn(config, config.num_quantiles, config.use_adv_net)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=config.p_lr)
    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=config.q_lr)

    trainer = discrete_mSAC(
        actor_net=actor_net,
        target_actor_net=target_actor_net,
        q_net=q_net,
        target_net=target_net,
        use_calql=config.use_calql,
        actor_optimizer=actor_optimizer,
        q_optimizer=q_optimizer,
        target_update_freq=config.target_update_freq,
        update_freq_policy=config.update_freq_policy,
        cql_alpha=config.cql_alpha_offline,
        alpha_multiplier=config.alpha_multiplier,
        tau=config.tau,
        discount=config.gamma,
        max_steps=config.num_train_step_offline // 2,
        device=config.device,
        N=config.num_quantiles,
        use_cql=config.use_cql,
        target_entropy=config.target_entropy,
        anneal_entropy=config.anneal_entropy,
        anneal_lr=config.anneal_lr,
        kappa=config.kappa,
        normalize_q=config.normalize_q,
        max_steps_lr=config.num_train_step_offline,
        backup_entropy=config.backup_entropy,
        use_qrdqn=config.use_qrdqn,
        max_grad_norm=config.max_grad_norm,
        q_pretrain_steps=config.q_pretrain_steps,
    )
    return trainer, actor_net


def build_mqrdqn(config):
    if not config.use_qrdqn:
        config.num_quantiles = 1

    q_net = mQRDQNNet(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_critic=config.num_mlp_layers_actor,
        num_quantiles=config.num_quantiles,
        hidden_dim_critic=config.hidden_dim_actor,
        use_adv_net=config.use_adv_net,
        dropout_prob_q=config.dropout_prob_q,
        layer_norm=config.layer_norm,
    ).to(config.device)

    target_net = mQRDQNNet(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_critic=config.num_mlp_layers_actor,
        num_quantiles=config.num_quantiles,
        hidden_dim_critic=config.hidden_dim_actor,
        use_adv_net=config.use_adv_net,
        dropout_prob_q=config.dropout_prob_q,
        layer_norm=config.layer_norm,
    ).to(config.device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    q_optimizer = torch.optim.Adam(q_net.parameters(), lr=config.q_lr)

    trainer = mQRDQN(
        q_net=q_net,
        target_net=target_net,
        use_calql=config.use_calql,
        q_optimizer=q_optimizer,
        target_update_freq=config.target_update_freq,
        cql_alpha=config.cql_alpha_offline,
        tau=config.tau,
        discount=config.gamma,
        max_steps=config.num_train_step_offline,
        device=config.device,
        N=config.num_quantiles,
        use_cql=config.use_cql,
        anneal_lr=config.anneal_lr,
        kappa=config.kappa,
        max_steps_lr=config.num_train_step_offline,
        use_qrdqn=config.use_qrdqn,
    )
    # mQRDQN has no separate actor; the greedy Q-network is the policy.
    return trainer, q_net


def build_bc(config):
    actor_net = make_actor(config)
    actor_optimizer = torch.optim.Adam(actor_net.parameters(), lr=config.p_lr)
    trainer = Imitation_Learning(
        actor_net=actor_net,
        actor_optimizer=actor_optimizer,
        device=config.device,
    )
    return trainer, actor_net


METHODS = {
    "cdqac": {"config": CDQACConfig, "build": build_cdqac, "save_with_critic": True},
    "td3_awr": {"config": TD3AWRConfig, "build": build_td3_awr, "td3_batches": True},
    "iql": {"config": IQLConfig, "build": build_iql},
    "dmsac": {"config": DMSACConfig, "build": build_dmsac},
    "mqrdqn": {"config": MQRDQNConfig, "build": build_mqrdqn},
    "bc": {"config": BCConfig, "build": build_bc},
}


def get_state_dict(trainer, feature_stats: dict, with_critic: bool) -> dict:
    state_dict = trainer.get_dict_with_critic() if with_critic else trainer.get_dict()
    state_dict.update(feature_stats)
    return state_dict


def train(method: str, config):
    spec = METHODS[method]
    save_with_critic = spec.get("save_with_critic", False)
    td3_batches = spec.get("td3_batches", False)
    use_disk = config.use_disk_buffer
    if use_disk and getattr(config, "n_step", 1) > 1:
        raise ValueError("n_step > 1 is not supported with use_disk_buffer "
                         "(DiskBuffer has no n_step_buffer)")

    set_seed(config.seed, device=config.device)
    train_instance_path = os.path.join(config.data_path, config.train_instance)
    train_instances = np.load(train_instance_path, allow_pickle=True)
    eval_instance_path = os.path.join(config.data_path, config.eval_instance)
    eval_instances = np.load(eval_instance_path, allow_pickle=True)
    if config.num_instances is not None:
        train_instances = train_instances[:config.num_instances]
    data_env_func = FJSPEnvForSameOpNums
    eval_env_func = FJSPEnvForVariousOpNums

    if config.save_folder is not None:
        save_folder = os.path.join(config.save_folder, config.name + "_seed_" + str(config.seed))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        write_yml_file(os.path.join(save_folder, "config.yml"), asdict(config))
    else:
        save_folder = None

    use_eps_pdr = getattr(config, "use_eps_pdr", False)
    if not (config.use_dispatching or config.use_ga_hof or config.use_ga_pop
            or config.use_random or use_eps_pdr):
        raise ValueError("At least one of use_dispatching, use_ga_hof, use_ga_pop, "
                         "use_random, use_eps_pdr must be True")

    job_lenght_list = []
    opt_list = []
    action_list_list = []
    total_runs = 0
    eps_random_ver = f"eps_random_{getattr(config, 'eps_pdr_version', '0.1')}"
    for instance in train_instances:
        curr_action_list = []
        if config.use_dispatching:
            for rule in instance["rules"]:
                is_masked = instance["rules_info"][rule]["masked"]

                if config.use_mask and not is_masked:
                    continue

                curr_action_list.append(instance["rules"][rule])
        if config.use_random and "random" in instance:
            if config.n_random is None:
                curr_action_list += instance["random"]
            else:
                curr_action_list += instance["random"][:config.n_random]
        job_lenght_list.append(instance["JobLength"])
        opt_list.append(instance["OpPT"])
        if "ga" in instance and config.use_ga_hof:
            if config.n_ga_hof is None:
                curr_action_list += instance["ga"]
            else:
                if config.n_ga_hof > len(instance["ga"]):
                    curr_action_list += instance["ga"]
                else:
                    if config.ga_hof_random:
                        curr_action_list += np.random.choice(instance["ga"], size=config.n_ga_hof, replace=False)
                    else:
                        curr_action_list += instance["ga"][:config.n_ga_hof]
        if "ga_pop" in instance and config.use_ga_pop:
            if config.n_ga_pop is None:
                curr_action_list += instance["ga_pop"]
            else:
                if config.n_ga_pop > len(instance["ga_pop"]):
                    curr_action_list += instance["ga_pop"]
                else:
                    if config.ga_pop_random:
                        makespans = instance['ga_pop_makespan']
                        selected_actions = weighted_choice(instance["ga_pop"], makespans, config.n_ga_pop)
                        curr_action_list += selected_actions
                    else:
                        curr_action_list += instance["ga_pop"][:config.n_ga_pop]
        if eps_random_ver in instance and use_eps_pdr:
            if config.n_eps_pdr is None:
                curr_action_list += instance[eps_random_ver]
            else:
                if config.n_eps_pdr > len(instance[eps_random_ver]):
                    curr_action_list += instance[eps_random_ver]
                else:
                    curr_action_list += instance[eps_random_ver][:config.n_eps_pdr]

        if config.remove_duplicate:
            orig_len = len(curr_action_list)
            curr_action_list, n_removed = remove_duplicate_actions(curr_action_list)
            print(f"Removed {n_removed} duplicate actions from {orig_len} actions")

        action_list_list.append(curr_action_list)
        total_runs += len(curr_action_list)
    jobLength = np.array(train_instances[0]["JobLength"])
    opPT = np.array(train_instances[0]["OpPT"])
    n_j = jobLength.shape[0]
    n_op, n_m = opPT.shape
    size = total_runs * n_op

    if use_disk:
        buffer_dir = config.buffer_dir
        if buffer_dir is None:
            # On SLURM prefer node-local scratch: random reads on the shared
            # filesystem collapse once the buffer exceeds the page cache.
            slurm_tmp = os.environ.get("SLURM_TMPDIR")
            if slurm_tmp is None and os.environ.get("SLURM_JOB_ID") is not None:
                slurm_tmp = os.environ.get("TMPDIR")
            if slurm_tmp is not None and os.path.isdir(slurm_tmp):
                buffer_dir = os.path.join(slurm_tmp, "buffer_cache_" + config.name)
                print(f"Using node-local scratch for the disk buffer: {buffer_dir}")
            else:
                buffer_dir = os.path.join(save_folder if save_folder is not None else ".",
                                          "buffer_cache")
        buffer = DiskBuffer(size=size, n_j=n_j, n_m=n_m, n_op=n_op, device="cpu",
                            dir_path=buffer_dir)
    else:
        buffer = Buffer(size=size, n_j=n_j, n_m=n_m, n_op=n_op, device="cpu")

    trainer, eval_net = spec["build"](config)

    n_gradient_updates = 0
    print("Start collecting data")
    collect_kwargs = {}
    if hasattr(config, "cutt_off_episode_prob"):
        collect_kwargs["cutt_off_episode_prob"] = config.cutt_off_episode_prob
    if hasattr(config, "use_proxy_reward"):
        collect_kwargs["use_proxy_reward"] = config.use_proxy_reward
    if hasattr(config, "use_sparse_reward"):
        collect_kwargs["use_sparse_reward"] = config.use_sparse_reward
    collect_data_new(buffer, job_lenght_list, opt_list, action_list_list,
                     data_env_func, reward_scale=config.reward_scale,
                     reward_bias=config.reward_bias, debug=False,
                     reward_scaling=config.reward_scaling,
                     use_mask=config.use_mask, **collect_kwargs)
    mean_fea_j, mean_fea_m, std_fea_j, std_fea_m, mean_fea_pairs, std_fea_pairs = buffer.normalize_state()
    print("Data collected")
    feature_stats = {
        "mean_fea_j": mean_fea_j,
        "std_fea_j": std_fea_j,
        "mean_fea_m": mean_fea_m,
        "std_fea_m": std_fea_m,
        "mean_fea_pairs": mean_fea_pairs,
        "std_fea_pairs": std_fea_pairs,
    }

    if not use_disk and getattr(config, "n_step", 1) > 1:
        buffer.n_step_buffer(config.n_step, config.gamma)

    dataloader = None
    if use_disk:
        dataset = TD3TransitionDataset(buffer,
                                       madvise_random=(config.sampler_chunk_len == 1),
                                       td3_format=td3_batches)
        if config.do_epoch:
            sampler = EpochBatchSampler(len(dataset), config.batch_size, seed=config.seed,
                                        chunk_len=config.sampler_chunk_len)
        else:
            sampler = RandomBatchSampler(len(dataset), config.batch_size,
                                         num_batches=config.num_train_step_offline,
                                         seed=config.seed,
                                         chunk_len=config.sampler_chunk_len)
        dataloader = make_td3_dataloader(dataset, sampler,
                                         num_workers=config.num_workers,
                                         prefetch_factor=config.prefetch_factor,
                                         pin_memory=config.pin_memory,
                                         persistent_workers=config.persistent_workers)

    if config.use_wandb:
        wandb_init(asdict(config))
    best_det_offline = np.inf

    def evaluate():
        return eval_model_all(eval_net, eval_instances, eval_env_func,
                              device=config.device, num_runs=1, deterministic=True,
                              use_mask=config.use_mask, mean_fea_j=mean_fea_j,
                              std_fea_j=std_fea_j, mean_fea_m=mean_fea_m,
                              std_fea_m=std_fea_m, mean_fea_pairs=mean_fea_pairs,
                              std_fea_pairs=std_fea_pairs)

    def save_checkpoint(filename):
        if save_folder is None:
            return
        state_dict = get_state_dict(trainer, feature_stats, save_with_critic)
        torch.save(state_dict, os.path.join(save_folder, filename))

    def eval_and_checkpoint(step):
        nonlocal best_det_offline
        eval_reward_det, runtimes_det = evaluate()
        dict_info = {
            "eval_deterministic/makespan": eval_reward_det,
            "eval_deterministic/runtime_mean": np.mean(runtimes_det),
            "eval_deterministic/runtime_std": np.std(runtimes_det),
        }
        if config.use_wandb: wandb.log(dict_info, step=step)
        if eval_reward_det < best_det_offline:
            best_det_offline = eval_reward_det
            if save_folder is not None:
                print("Saving best model")
                save_checkpoint("best_det_offline.pt")
        save_checkpoint("latest_check.pt")
        return eval_reward_det

    if config.do_epoch:
        step = 0
        for epoch in range(1, config.train_epochs + 1):

            t_bar = tqdm.tqdm(range(len(buffer)), desc=f"Epoch {epoch}/{config.train_epochs}")
            if use_disk:
                epoch_batches = (batch_to_device(b, config.device) for b in dataloader)
            elif td3_batches:
                epoch_batches = buffer.epoch_generator_td3(config.batch_size, device=config.device)
            else:
                epoch_batches = buffer.epoch_generator(config.batch_size, device=config.device)
            for batch in epoch_batches:
                batch_size = batch[2].shape[0]
                n_gradient_updates += 1
                info = trainer.train(batch)
                if config.use_wandb: wandb.log(info, step=step)
                step += 1
                t_bar.update(batch_size)
            if epoch % config.eval_every_epoch == 0:
                print("Evaluating Model at epoch: ", epoch)
                eval_reward_det = eval_and_checkpoint(step)
                print(
                    f"\nEpoch {epoch}: Eval reward {eval_reward_det}, Best {best_det_offline}")
    else:
        batch_iter = iter(dataloader) if use_disk else None
        for step in range(config.num_train_step_offline):
            n_gradient_updates += 1
            if use_disk:
                batch = batch_to_device(next(batch_iter), config.device)
            elif td3_batches:
                batch = buffer.sample_td3(config.batch_size, device=config.device)
            else:
                batch = buffer.sample(config.batch_size, device=config.device)
            info = trainer.train(batch)

            if config.use_wandb: wandb.log(info, step=step)

            if (step + 1) % 10000 == 0:
                save_checkpoint(f"model_step_{step + 1}.pt")

            if (step + 1) % config.eval_freq == 0:
                print("Evaluating Model at Step: ", step)
                eval_reward_det = eval_and_checkpoint(step)
                print(
                    f"Step: {step} Eval reward {eval_reward_det}, Best {best_det_offline}")

    save_checkpoint("model_final_offline.pt")

    if use_disk:
        # Shut down loader workers before touching the memmap files (Windows
        # keeps them locked while any process has a handle open).
        batch_iter = None
        del dataloader
        import gc
        gc.collect()
        buffer.close(delete=config.cleanup_buffer_files)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--method", choices=sorted(METHODS), default="cdqac",
                        help="Which training method to run (default: cdqac). "
                             "All other arguments are forwarded to that method's TrainConfig.")
    args, remaining = parser.parse_known_args()
    config = pyrallis.parse(config_class=METHODS[args.method]["config"], args=remaining)
    train(args.method, config)


if __name__ == "__main__":
    # The guard is required: DataLoader workers on Windows spawn fresh
    # interpreters that re-import this module.
    main()
