import os.path
from collections.abc import Callable
import pickle
import torch
import time

from cdqac.buffer.buffer import Buffer
from cdqac.buffer.disk_buffer import (DiskBuffer, TD3TransitionDataset,
                                           RandomBatchSampler, EpochBatchSampler,
                                           make_td3_dataloader, batch_to_device)
from cdqac.network.main_model import ActorNet, QRDQNNet, IQL_value
from cdqac.methods.td3_awr import TD3AWR
from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from cdqac.configs.td3_awr_config import TrainConfig
from cdqac.utils import (handle_reward, eval_model, write_yml_file, collect_data_new, remove_duplicate_actions,
                              comb_batch, eval_model_all)
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import tqdm
import yaml


import copy
import uuid
import pyrallis
from collections import defaultdict
from dataclasses import asdict, dataclass
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


@pyrallis.wrap()
def train(config: TrainConfig):
    set_seed(config.seed, device=config.device)
    train_instance_path = os.path.join(config.data_path, config.train_instance)
    train_instances = np.load(train_instance_path, allow_pickle=True)
    eval_instance_path = os.path.join(config.data_path, config.eval_instance)
    eval_instances = np.load(eval_instance_path, allow_pickle=True)
    if config.num_instances is not None:
        train_instances = train_instances[:config.num_instances]
    if config.train_instance.startswith("SD2"):
        data_env_func = FJSPEnvForSameOpNums
    elif config.train_instance.startswith("SD1"):
        data_env_func = FJSPEnvForSameOpNums
    else:
        raise ValueError("Invalid data source")

    eval_env_func = FJSPEnvForVariousOpNums

    if config.save_folder is not None:
        save_folder = os.path.join(config.save_folder, config.name + "_seed_" + str(config.seed))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        write_yml_file(os.path.join(save_folder, "config.yml"), asdict(config))


    else:
        save_folder = None
    if not config.use_dispatching and not config.use_ga_hof and not config.use_ga_pop and not config.use_random:
        raise ValueError("At least one of use_dispatching, use_ga_hof, use_ga_pop must be True")
    job_lenght_list = []
    opt_list = []
    action_list_list = []
    total_runs = 0
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

    if config.use_disk_buffer:
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


    actor_net = ActorNet(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_actor=config.num_mlp_layers_actor,
        hidden_dim_actor=config.hidden_dim_actor,
        dropout_prob=config.dropout_prob_actor,
    ).to(config.device)

    target_actor_net = ActorNet(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_actor=config.num_mlp_layers_actor,
        hidden_dim_actor=config.hidden_dim_actor,
        dropout_prob=config.dropout_prob_actor,
    ).to(config.device)
    target_actor_net.load_state_dict(actor_net.state_dict())
    target_actor_net.eval()

    q_net = QRDQNNet(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_critic=config.num_mlp_layers_critic,
        num_quantiles=1,
        hidden_dim_critic=config.hidden_dim_critic,
        use_adv_net=False,
        dropout_prob_q=config.dropout_prob_q,
        n_critics=config.n_critics,
        layer_norm=config.layer_norm,
    ).to(config.device)

    target_net = QRDQNNet(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_critic=config.num_mlp_layers_critic,
        num_quantiles=1,
        hidden_dim_critic=config.hidden_dim_critic,
        use_adv_net=False,
        dropout_prob_q=config.dropout_prob_q,
        n_critics=config.n_critics,
        layer_norm=config.layer_norm,
    ).to(config.device)

    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    value_net = IQL_value(
        fea_j_input_dim=config.fea_j_input_dim,
        fea_m_input_dim=config.fea_m_input_dim,
        layer_fea_output_dim=config.layer_fea_output_dim,
        num_heads_OAB=config.num_heads_OAB,
        num_heads_MAB=config.num_heads_MAB,
        num_mlp_layers_critic=config.num_mlp_layers_value,
        hidden_dim_critic=config.hidden_dim_value,
    ).to(config.device)


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


    n_gradient_updates = 0
    print("Start collecting data")
    collect_data_new(buffer, job_lenght_list, opt_list, action_list_list,
                                                                    data_env_func, reward_scale=config.reward_scale,
                                                                    reward_bias=config.reward_bias, debug=False,
                                                                    reward_scaling=config.reward_scaling,
                                                                    use_mask=config.use_mask)
    mean_fea_j, mean_fea_m, std_fea_j, std_fea_m, mean_fea_pairs, std_fea_pairs = buffer.normalize_state()
    print("Data collected")

    dataloader = None
    if config.use_disk_buffer:
        dataset = TD3TransitionDataset(buffer,
                                       madvise_random=(config.sampler_chunk_len == 1))
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


    if config.do_epoch:
        step = 0
        for epoch in range(1, config.train_epochs + 1):

            t_bar = tqdm.tqdm(range(len(buffer)), desc=f"Epoch {epoch}/{config.train_epochs}")
            if config.use_disk_buffer:
                epoch_batches = (batch_to_device(b, config.device) for b in dataloader)
            else:
                epoch_batches = buffer.epoch_generator_td3(config.batch_size, device=config.device)
            for batch in epoch_batches:
                batch_size = batch[2].shape[0]
                n_gradient_updates += 1
                info = trainer.train(batch)
                if config.use_wandb: wandb.log(info, step=step)
                step += 1
                t_bar.update(batch_size)
            if epoch % config.eval_every_epoch == 0:
                print("Evaluating Model at epoch: ", epoch)

                eval_reward_det, runtimes_det = eval_model_all(actor_net, eval_instances, eval_env_func,
                                                               device=config.device, num_runs=1, deterministic=True,
                                                               use_mask=config.use_mask, mean_fea_j=mean_fea_j,
                                                               std_fea_j=std_fea_j, mean_fea_m=mean_fea_m,
                                                               std_fea_m=std_fea_m, mean_fea_pairs=mean_fea_pairs,
                                                               std_fea_pairs=std_fea_pairs,
                                                               )

                mean_runtimes_det = np.mean(runtimes_det)
                std_runtimes_det = np.std(runtimes_det)
                dict_info = {
                    "eval_deterministic/makespan": eval_reward_det,
                    "eval_deterministic/runtime_mean": mean_runtimes_det,
                    "eval_deterministic/runtime_std": std_runtimes_det
                }
                if config.use_wandb: wandb.log(dict_info, step=step)
                if eval_reward_det < best_det_offline:
                    best_det_offline = eval_reward_det
                    if save_folder is not None:
                        state_dict = trainer.get_dict()
                        state_dict["mean_fea_j"] = mean_fea_j
                        state_dict["std_fea_j"] = std_fea_j
                        state_dict["mean_fea_m"] = mean_fea_m
                        state_dict["std_fea_m"] = std_fea_m
                        state_dict["mean_fea_pairs"] = mean_fea_pairs
                        state_dict["std_fea_pairs"] = std_fea_pairs

                        print("Saving best model")
                        torch.save(state_dict, os.path.join(save_folder, f"best_det_offline.pt"))
                if save_folder is not None:
                    state_dict = trainer.get_dict()
                    state_dict["mean_fea_j"] = mean_fea_j
                    state_dict["std_fea_j"] = std_fea_j
                    state_dict["mean_fea_m"] = mean_fea_m
                    state_dict["std_fea_m"] = std_fea_m
                    state_dict["mean_fea_pairs"] = mean_fea_pairs
                    state_dict["std_fea_pairs"] = std_fea_pairs
                    torch.save(state_dict, os.path.join(save_folder, f"latest_check.pt"))

                print(
                    f"\nEpoch {epoch}: Eval reward {eval_reward_det}, Best {best_det_offline}")
        cont_step = step
    else:
        batch_iter = iter(dataloader) if config.use_disk_buffer else None
        for step in range(config.num_train_step_offline):
            n_gradient_updates += 1
            if config.use_disk_buffer:
                batch = batch_to_device(next(batch_iter), config.device)
            else:
                batch = buffer.sample_td3(config.batch_size, device=config.device)
            info = trainer.train(batch)

            if config.use_wandb: wandb.log(info, step=step)

            if (step + 1) % 10000 == 0:
                if save_folder is not None:
                    state_dict = trainer.get_dict()
                    state_dict["mean_fea_j"] = mean_fea_j
                    state_dict["std_fea_j"] = std_fea_j
                    state_dict["mean_fea_m"] = mean_fea_m
                    state_dict["std_fea_m"] = std_fea_m
                    state_dict["mean_fea_pairs"] = mean_fea_pairs
                    state_dict["std_fea_pairs"] = std_fea_pairs
                    torch.save(state_dict, os.path.join(save_folder, f"model_step_{step + 1}.pt"))

            if (step+1) % config.eval_freq == 0:
                print("Evaluating Model at Step: ", step)

                eval_reward_det, runtimes_det = eval_model_all(actor_net, eval_instances, eval_env_func,
                                                               device=config.device, num_runs=1, deterministic=True,
                                                               use_mask=config.use_mask, mean_fea_j=mean_fea_j,
                                                               std_fea_j=std_fea_j, mean_fea_m=mean_fea_m,
                                                                std_fea_m=std_fea_m, mean_fea_pairs=mean_fea_pairs,
                                                                std_fea_pairs=std_fea_pairs,
                                                               )

                mean_runtimes_det = np.mean(runtimes_det)
                std_runtimes_det = np.std(runtimes_det)
                dict_info = {
                    "eval_deterministic/makespan": eval_reward_det,
                    "eval_deterministic/runtime_mean": mean_runtimes_det,
                    "eval_deterministic/runtime_std": std_runtimes_det
                }
                if config.use_wandb: wandb.log(dict_info, step=step)
                if eval_reward_det < best_det_offline:
                    best_det_offline = eval_reward_det
                    if save_folder is not None:
                        state_dict = trainer.get_dict()
                        state_dict["mean_fea_j"] = mean_fea_j
                        state_dict["std_fea_j"] = std_fea_j
                        state_dict["mean_fea_m"] = mean_fea_m
                        state_dict["std_fea_m"] = std_fea_m
                        state_dict["mean_fea_pairs"] = mean_fea_pairs
                        state_dict["std_fea_pairs"] = std_fea_pairs

                        print("Saving best model")
                        torch.save(state_dict, os.path.join(save_folder, f"best_det_offline.pt"))
                if save_folder is not None:
                    state_dict = trainer.get_dict()
                    state_dict["mean_fea_j"] = mean_fea_j
                    state_dict["std_fea_j"] = std_fea_j
                    state_dict["mean_fea_m"] = mean_fea_m
                    state_dict["std_fea_m"] = std_fea_m
                    state_dict["mean_fea_pairs"] = mean_fea_pairs
                    state_dict["std_fea_pairs"] = std_fea_pairs

                    torch.save(state_dict, os.path.join(save_folder, f"latest_check.pt"))

                print(
                    f"Step: {step} Eval reward {eval_reward_det}, Best {best_det_offline}")
        cont_step = config.num_train_step_offline
    if save_folder is not None:
        state_dict = trainer.get_dict()
        state_dict["mean_fea_j"] = mean_fea_j
        state_dict["std_fea_j"] = std_fea_j
        state_dict["mean_fea_m"] = mean_fea_m
        state_dict["std_fea_m"] = std_fea_m
        state_dict["mean_fea_pairs"] = mean_fea_pairs
        state_dict["std_fea_pairs"] = std_fea_pairs
        torch.save(state_dict, os.path.join(save_folder, f"model_final_offline.pt"))

    if config.use_disk_buffer:
        # Shut down loader workers before touching the memmap files (Windows
        # keeps them locked while any process has a handle open).
        batch_iter = None
        del dataloader
        import gc
        gc.collect()
        buffer.close(delete=config.cleanup_buffer_files)


if __name__ == "__main__":
    # The guard is required: DataLoader workers on Windows spawn fresh
    # interpreters that re-import this module.
    train()
