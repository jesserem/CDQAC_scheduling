import os.path
from collections.abc import Callable
import pickle
import torch
import time

from cdqac.buffer.buffer import Buffer
from cdqac.network.main_model import mQRDQNNet
from cdqac.methods.mqrdqn import mQRDQN

from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from cdqac.configs.mqrdqn_config import TrainConfig
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
    wandb.run.save()

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
    # config.device = torch.device(config.device)
    train_instance_path = os.path.join(config.data_path, config.train_instance)
    train_instances = np.load(train_instance_path, allow_pickle=True)
    eval_instance_path = os.path.join(config.data_path, config.eval_instance)
    eval_instances = np.load(eval_instance_path, allow_pickle=True)
    if config.num_instances is not None:
        train_instances = train_instances[:config.num_instances]
    if config.train_instance.startswith("SD2"):
        data_env_func = FJSPEnvForSameOpNums
    elif config.train_instance.startswith("SD1"):
        # data_env_func = FJSPEnvForVariousOpNums
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
    # print(train_instances)
    # exit()
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
                # curr_action_list += np.transpose(instance["random"]).tolist()
                curr_action_list += instance["random"]
                # print(len(np.transpose(instance["random"]).tolist()))
            else:
                curr_action_list += instance["random"][:config.n_random]
        # curr_action_list = copy.deepcopy([instance["rules"][rule] for rule in instance["rules"]])
        job_lenght_list.append(instance["JobLength"])
        opt_list.append(instance["OpPT"])
        if "ga" in instance and config.use_ga_hof:
            if config.n_ga_hof is None:
                # print(len(instance["ga"]))
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
                        # curr_action_list += np.random.choice(instance["ga_pop"], size=config.n_ga_pop, replace=False)
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

    buffer = Buffer(size=size, n_j=n_j, n_m=n_m, n_op=n_op, device="cpu")


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
    # exit()

    # if config.use_qrdqn:
    #     q_net = QRDQNNet(
    #         fea_j_input_dim=config.fea_j_input_dim,
    #         fea_m_input_dim=config.fea_m_input_dim,
    #         layer_fea_output_dim=config.layer_fea_output_dim,
    #         num_heads_OAB=config.num_heads_OAB,
    #         num_heads_MAB=config.num_heads_MAB,
    #         num_mlp_layers_critic=config.num_mlp_layers_critic,
    #         num_quantiles=config.num_quantiles,
    #         hidden_dim_critic=config.hidden_dim_critic,
    #         use_adv_net=config.use_adv_net,
    #         dropout_prob_q=config.dropout_prob_q,
    #         n_critics=config.n_critics,
    #         layer_norm=config.layer_norm,
    #     ).to(config.device)
    #
    #     target_net = QRDQNNet(
    #         fea_j_input_dim=config.fea_j_input_dim,
    #         fea_m_input_dim=config.fea_m_input_dim,
    #         layer_fea_output_dim=config.layer_fea_output_dim,
    #         num_heads_OAB=config.num_heads_OAB,
    #         num_heads_MAB=config.num_heads_MAB,
    #         num_mlp_layers_critic=config.num_mlp_layers_critic,
    #         num_quantiles=config.num_quantiles,
    #         hidden_dim_critic=config.hidden_dim_critic,
    #         use_adv_net=config.use_adv_net,
    #         dropout_prob_q=config.dropout_prob_q,
    #         n_critics=config.n_critics,
    #         layer_norm=config.layer_norm,
    #     ).to(config.device)
    # else:
    #
    #     q_net = DQNNet(
    #         fea_j_input_dim=config.fea_j_input_dim,
    #         fea_m_input_dim=config.fea_m_input_dim,
    #         layer_fea_output_dim=config.layer_fea_output_dim,
    #         num_heads_OAB=config.num_heads_OAB,
    #         num_heads_MAB=config.num_heads_MAB,
    #         num_mlp_layers_critic=config.num_mlp_layers_critic,
    #         num_quantiles=config.num_quantiles,
    #         hidden_dim_critic=config.hidden_dim_critic,
    #         use_adv_net=config.use_adv_net,
    #         dropout_prob_q=config.dropout_prob_q
    #     ).to(config.device)
    #
    #     target_net = DQNNet(
    #         fea_j_input_dim=config.fea_j_input_dim,
    #         fea_m_input_dim=config.fea_m_input_dim,
    #         layer_fea_output_dim=config.layer_fea_output_dim,
    #         num_heads_OAB=config.num_heads_OAB,
    #         num_heads_MAB=config.num_heads_MAB,
    #         num_mlp_layers_critic=config.num_mlp_layers_critic,
    #         num_quantiles=config.num_quantiles,
    #         hidden_dim_critic=config.hidden_dim_critic,
    #         use_adv_net=config.use_adv_net,
    #         dropout_prob_q=config.dropout_prob_q
    #     ).to(config.device)


    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # q_optimizer = torch.optim.Adam(q_net.parameters(), lr=config.q_lr, eps=1e-2/config.batch_size)
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
    # if config.use_qrdqn:
    #     trainer = CDQAC(
    #         actor_net=actor_net,
    #         target_actor_net=target_actor_net,
    #         q_net=q_net,
    #         target_net=target_net,
    #         use_calql=config.use_calql,
    #         actor_optimizer=actor_optimizer,
    #         q_optimizer=q_optimizer,
    #         target_update_freq=config.target_update_freq,
    #         update_freq_policy=config.update_freq_policy,
    #         cql_alpha=config.cql_alpha_offline,
    #         alpha_multiplier=config.alpha_multiplier,
    #         tau=config.tau,
    #         discount=config.gamma,
    #         max_steps=config.num_train_step_offline,
    #         device=config.device,
    #         N=config.num_quantiles,
    #         use_cql=config.use_cql,
    #         target_entropy=config.target_entropy,
    #         anneal_entropy=config.anneal_entropy,
    #         anneal_lr=config.anneal_lr,
    #         kappa=config.kappa,
    #         normalize_q=config.normalize_q,
    #         max_steps_lr=config.num_train_step_offline,
    #         backup_entropy=config.backup_entropy
    #     )
    # else:
    #     trainer = TD3_DQN(
    #         actor_net=actor_net,
    #         target_actor_net=target_actor_net,
    #         q_net=q_net,
    #         target_net=target_net,
    #         actor_optimizer=actor_optimizer,
    #         q_optimizer=q_optimizer,
    #         target_update_freq=config.target_update_freq,
    #         update_freq_policy=config.update_freq_policy,
    #         cql_alpha=config.cql_alpha_offline,
    #         alpha_multiplier=config.alpha_multiplier,
    #         tau=config.tau,
    #         discount=config.gamma,
    #         max_steps=config.num_train_step_offline,
    #         device=config.device,
    #         N=config.num_quantiles,
    #         use_cql=config.use_cql,
    #         target_entropy=config.target_entropy,
    #         anneal_entropy=config.anneal_entropy,
    #         anneal_lr=config.anneal_lr,
    #         kappa=config.kappa,
    #         max_steps_lr=config.num_train_step_offline,
    #         backup_entropy=config.backup_entropy
    #     )

    total_loss = 0
    total_steps = 0
    n_gradient_updates = 0
    print("Start collecting data")
    collect_data_new(buffer, job_lenght_list, opt_list, action_list_list,
                                                                    data_env_func, reward_scale=config.reward_scale,
                                                                    reward_bias=config.reward_bias, debug=False,
                                                                    reward_scaling=config.reward_scaling,
                                                                    use_mask=config.use_mask)
    mean_fea_j, mean_fea_m, std_fea_j, std_fea_m, mean_fea_pairs, std_fea_pairs = buffer.normalize_state()
    print("Data collected")


    if config.use_wandb:
        wandb_init(asdict(config))
    best_det = np.inf
    best_samp = np.inf
    best_det_offline = np.inf
    best_samp_offline = np.inf


    if config.do_epoch:
        step = 0
        for epoch in range(1, config.train_epochs + 1):

            t_bar = tqdm.tqdm(range(len(buffer)), desc=f"Epoch {epoch}/{config.train_epochs}")
            for batch in buffer.epoch_generator(config.batch_size):
                batch_size = batch[2].shape[0]
                n_gradient_updates += 1
                info = trainer.train(batch)
                total_loss += 0
                if config.use_wandb: wandb.log(info, step=step)
                step += 1
                # t_bar.set_postfix({"td_loss_1" : info["td_q1_loss"], "td_loss_2" : info["td_q2_loss"], "q_loss" : info["q_loss"]})
                t_bar.update(batch_size)
            if epoch % config.eval_every_epoch == 0:
                print("Evaluating Model at epoch: ", epoch)

                # eval_reward, runtimes = eval_model(actor_net, eval_instances, eval_env_func, device=config.device,
                #                                    num_runs=100, deterministic=False, use_mask=config.use_mask)
                eval_reward_det, runtimes_det = eval_model(actor_net, eval_instances, eval_env_func,
                                                           device=config.device,
                                                           num_runs=1, deterministic=True, use_mask=config.use_mask)

                # mean_runtimes = np.mean(runtimes)
                # std_runtimes = np.std(runtimes)
                mean_runtimes_det = np.mean(runtimes_det)
                std_runtimes_det = np.std(runtimes_det)
                dict_info = {
                    # "eval_stochastic/makespan": eval_reward,
                    # "eval_stochastic/runtime_mean": mean_runtimes,
                    # "eval_stochastic/runtime_std": std_runtimes,
                    "eval_deterministic/makespan": eval_reward_det,
                    "eval_deterministic/runtime_mean": mean_runtimes_det,
                    "eval_deterministic/runtime_std": std_runtimes_det
                }
                if config.use_wandb: wandb.log(dict_info, step=step)
                if eval_reward_det < best_det_offline:
                    best_det_offline = eval_reward_det
                    if save_folder is not None:
                        print("Saving best model")
                        torch.save(state_dict, os.path.join(save_folder, f"best_det_offline.pt"))


                print(
                    f"\nEpoch: {epoch}, Eval reward Stochastic: {eval_reward}, Eval reward Deterministic: {eval_reward_det}\n")
        cont_step = step
    else:
        for step in range(config.num_train_step_offline):
            n_gradient_updates += 1
            batch = buffer.sample(config.batch_size, device=config.device)
            # start_time = time.time()
            info = trainer.train(batch)
            # print(info)
            # runtimes_train.append(time.time() - start_time)
            total_loss += 0

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

                # eval_reward, runtimes = eval_model(actor_net, eval_instances, eval_env_func, device=config.device,
                #                                    num_runs=100, deterministic=False, use_mask=config.use_mask)
                eval_reward_det, runtimes_det = eval_model_all(q_net, eval_instances, eval_env_func,
                                                               device=config.device, num_runs=1, deterministic=True,
                                                               use_mask=config.use_mask, mean_fea_j=mean_fea_j,
                                                               std_fea_j=std_fea_j, mean_fea_m=mean_fea_m,
                                                                std_fea_m=std_fea_m, mean_fea_pairs=mean_fea_pairs,
                                                                std_fea_pairs=std_fea_pairs,
                                                               )

                # mean_runtimes = np.mean(runtimes)
                # std_runtimes = np.std(runtimes)
                mean_runtimes_det = np.mean(runtimes_det)
                std_runtimes_det = np.std(runtimes_det)
                dict_info = {
                    # "eval_stochastic/makespan": eval_reward,
                    # "eval_stochastic/runtime_mean": mean_runtimes,
                    # "eval_stochastic/runtime_std": std_runtimes,
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



train()