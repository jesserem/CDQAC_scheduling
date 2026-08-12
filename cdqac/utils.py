import numpy as np
import copy
from collections import defaultdict
import tqdm
from pathlib import Path, PureWindowsPath, PurePosixPath
import sys
import pandas as pd
import time
from typing import List
import torch
import yaml
import re
import os
from dataclasses import asdict, dataclass


def read_basic(f_path: str, sep: str = ' ', device: str = 'cpu'):
    """
    Load the basic information about a JSP instance.

    Structure of the file:
        1. num_jobs num_machines
        2. instance matrix (each row is a job)
        3. makespan
    """
    with open(f_path) as f:
        # Load the shape
        shape = next(f).split(sep)
        n = int(shape[0])
        m = int(shape[1])
        if sys.platform == 'win32':
            name = f_path.rsplit('\\', 1)[1].rsplit('.', 1)[0]
        else:
            name = f_path.rsplit('/', 1)[1].rsplit('.', 1)[0]

        # Load the instance
        instance = torch.empty((n, 2 * m), dtype=torch.float32, device=device)
        for j in range(n):
            instance[j] = torch.tensor(
                [float(x) for x in next(f).split(sep) if x and not x.isspace()],
                device=device
            )

        # Load the makespan of a reference solution, if any
        ms = 1.
        try:
            _ms = next(f)
            if _ms != '':
                ms = float(_ms)
        except StopIteration:
            pass
    return name, n, m, instance, ms

def jsp_instance_to_format(instance, n_j, n_m):

    n_op = int(n_j * n_m)
    job_length = np.full(shape=(n_j,), fill_value=n_m, dtype=int)
    op_pt = np.zeros((n_op, n_m), dtype=int)
    instance_flat = instance.flatten().to(torch.long)
    n_shape = instance_flat.shape[0]
    op_i = 0
    for i in range(0, n_shape, 2):
        machine_index = instance_flat[i].item()
        p_time = instance_flat[i + 1].item()
        op_pt[op_i, machine_index] = p_time
        op_i += 1
    return job_length, op_pt


def text_to_matrix(text):
    """
            Convert text form of the data into matrix form
    :param text: the standard text form of the instance
    :return:  the matrix form of the instance
            job_length: the number of operations in each job (shape [J])
            op_pt: the processing time matrix with shape [N, M],
                where op_pt[i,j] is the processing time of the ith operation
                on the jth machine or 0 if $O_i$ can not process on $M_j$
    """
    n_j = int(re.findall(r'\d+\.?\d*', text[0])[0])
    n_m = int(re.findall(r'\d+\.?\d*', text[0])[1])

    job_length = np.zeros(n_j, dtype='int32')
    op_pt = []

    for i in range(n_j):
        content = np.array([int(s) for s in re.findall(r'\d+\.?\d*', text[i + 1])])
        job_length[i] = content[0]

        idx = 1
        for j in range(content[0]):
            op_pt_row = np.zeros(n_m, dtype='int32')
            mch_num = content[idx]
            next_idx = idx + 2 * mch_num + 1
            for k in range(mch_num):
                mch_idx = content[idx + 2 * k + 1]
                pt = content[idx + 2 * k + 2]
                op_pt_row[mch_idx - 1] = pt

            idx = next_idx
            op_pt.append(op_pt_row)

    op_pt = np.array(op_pt)

    return job_length, op_pt


def load_data_from_files_jsp(directory):
    """
        load all files within the specified directory
    :param directory: the directory of files
    :return: a list of data (matrix form) in the directory
    """
    if not os.path.exists(directory):
        raise FileNotFoundError("The directory does not exist")

    data_name = []
    dataset_job_length = []
    dataset_op_pt = []
    for root, dirs, files in os.walk(directory):
        # sort files by index
        files.sort(key=lambda s: int(re.findall(r"\d+", s)[0]))
        files.sort(key=lambda s: int(re.findall(r"\d+", s)[-1]))
        for f in files:
            filepath = os.path.join(root, f)
            name, n, m, instance, ms = read_basic(str(filepath))

            job_length, op_pt = jsp_instance_to_format(instance, n, m)

            # print(f)
            # g = open(os.path.join(root, f), 'r').readlines()
            # job_length, op_pt = text_to_matrix(g)
            data_name.append(name)

            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)
    return data_name, dataset_job_length, dataset_op_pt


def load_data_from_files(directory):
    """
        load all files within the specified directory
    :param directory: the directory of files
    :return: a list of data (matrix form) in the directory
    """
    if not os.path.exists(directory):
        raise FileNotFoundError("The directory does not exist")

    data_name = []
    dataset_job_length = []
    dataset_op_pt = []
    for root, dirs, files in os.walk(directory):
        # sort files by index
        files.sort(key=lambda s: int(re.findall(r"\d+", s)[0]))
        files.sort(key=lambda s: int(re.findall(r"\d+", s)[-1]))
        for f in files:

            # print(f)
            g = open(os.path.join(root, f), 'r').readlines()
            job_length, op_pt = text_to_matrix(g)
            data_name.append(f.replace('.fjs',''))

            dataset_job_length.append(job_length)
            dataset_op_pt.append(op_pt)
    return data_name, dataset_job_length, dataset_op_pt

def tuple_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data)

def write_yml_file(file_path, data):
    yaml.add_representer(tuple, tuple_representer)
    with open(file_path, "w") as f:
        yaml.dump(data, f)

def load_yml(file_path):
    with open(file_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data

def calculate_discounted_return(rewards, gamma=1):
    """
        Calculate the discounted return
    :param rewards: the rewards collected in the episode
    :param gamma: the discount factor
    :return: the discounted return
    """
    mc_return = np.zeros_like(rewards)
    prev_return = 0
    for i in range(len(rewards) - 1, -1, -1):
        prev_return = rewards[i] + gamma * prev_return
        mc_return[i] = prev_return
    # print(mc_return)
    return mc_return


def handle_reward(reward, v1, v2, process="max_min_reward"):
    if process is None:
        return reward
    elif process == "max_min_return":
        return (reward - v1) / (v2 - v1)
    elif process == "max_min_return_step":
        raise ValueError("Max min return does not work with finetuning")
    elif process == "max_min_reward":
        return (reward - v1) / (v2 - v1)
    elif process == "min_reward":
        return reward / v1
    elif process == "standardization" or process == "iqr":
        return (reward - v1) / (v2 + 1e-8)




def preprocess_reward(buffer, reward_arr, start_idx, end_idx, process="max_min_reward"):
    if process is None:
        return reward_arr, 0, 0
    if process == "max_min_return":
        returns = reward_arr.sum(axis=0)
        min_return = np.min(returns)
        max_return = np.max(returns)
        # print(buffer.reward[start_idx:end_idx])
        buffer.reward[start_idx:end_idx] = ((buffer.reward[start_idx:end_idx]) / (max_return - min_return))
        reward_arr = (reward_arr / (max_return - min_return))
        return reward_arr, min_return, max_return
    elif process == "max_min_return_step":
        returns = reward_arr.sum(axis=0)
        min_return = np.min(returns)
        max_return = np.max(returns)

        # print(buffer.reward[start_idx:end_idx])
        buffer.reward[start_idx:end_idx] = ((buffer.reward[start_idx:end_idx]) / (max_return - min_return)) * reward_arr.shape[0]
        reward_arr = (reward_arr / (max_return - min_return)) * reward_arr.shape[0]
        return reward_arr, min_return, max_return
    elif process == "max_min_reward":
        min_reward = np.min(reward_arr)
        max_reward = np.max(reward_arr)
        buffer.reward[start_idx:end_idx] = ((buffer.reward[start_idx:end_idx] - min_reward) / (max_reward - min_reward))
        reward_arr = (reward_arr - min_reward) / (max_reward - min_reward)
        return reward_arr, min_reward, max_reward
    elif process == "standardization":
        mean_reward = np.mean(reward_arr)
        std_reward = np.std(reward_arr)
        buffer.reward[start_idx:end_idx] = ((buffer.reward[start_idx:end_idx] - mean_reward) / (std_reward + 1e-8))
        reward_arr = (reward_arr - mean_reward) / (std_reward + 1e-8)
        return reward_arr, mean_reward, std_reward
    elif process == "iqr":
        q3, q1 = np.percentile(reward_arr, [75, 25])
        iqr = q3 - q1

        median = np.median(reward_arr)

        buffer.reward[start_idx:end_idx] = ((buffer.reward[start_idx:end_idx] - median) / (iqr + 1e-8))
        reward_arr = (reward_arr - median) / (iqr + 1e-8)
        return reward_arr, median, iqr
    elif process == "min_reward":
        min_reward = np.abs(np.min(reward_arr))
        buffer.reward[start_idx:end_idx] = (buffer.reward[start_idx:end_idx] / min_reward)
        reward_arr = reward_arr / min_reward
        return reward_arr, min_reward, 0
    elif process == "std":
        std_reward = np.std(reward_arr)
        buffer.reward[start_idx:end_idx] = (buffer.reward[start_idx:end_idx] / std_reward)
        reward_arr = reward_arr / std_reward
        return reward_arr, 0, std_reward
    else:
        raise ValueError("Invalid process type")



def open_path(path):
    if sys.platform == "win32":
        return PureWindowsPath(path)
    else:
        return PurePosixPath(path)



def collect_data(buffer, train_instances, FJSP_env, debug=False, use_mean=False):
    reward_mean_list = []
    for instance_dict in train_instances:
        rules = instance_dict["rules"]
        num_runs = len(rules)
        if "ga" in instance_dict:
            num_runs += len(instance_dict["ga"])

        job_lenght = np.array(instance_dict["JobLength"])
        opt = np.array(instance_dict["OpPT"])
        # print(instance_dict.keys())
        # exit()
        n_j = job_lenght.shape[0]
        n_op, n_m = opt.shape
        JobLength_dataset = np.tile(np.expand_dims(job_lenght, axis=0), (num_runs, 1))
        OpPT_dataset = np.tile(np.expand_dims(opt, axis=0), (num_runs, 1, 1))
        env = FJSP_env(n_j=n_j, n_m=n_m)
        state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
        state = copy.deepcopy(state)
        action_list = [rules[rule] for rule in rules]
        curr_rewards = np.zeros((len(action_list[0]), num_runs))

        if "ga" in instance_dict:
            action_list += [instance_dict["ga"][i] for i in range(len(instance_dict["ga"]))]
        action_arr = np.array(action_list)

        if use_mean:
            mean_reward = instance_dict["mean_reward"]
            std_reward = instance_dict["std_reward"]
        else:
            mean_reward = 0
            std_reward = 1
        i = 0
        start_idx = buffer.ptr
        num_steps = 0
        for action in action_arr.T:
            # print(action.shape)
            # exit()
            next_state, reward, done = env.step(action)
            # print(action, state.dynamic_pair_mask_tensor)
            reward = (reward - mean_reward) / std_reward
            # reward *= 100

            buffer.push(state, next_state, action, reward, done)

            curr_rewards[i] = reward
            i += 1
            state = copy.deepcopy(next_state)
            num_steps += 1

        end_idx = buffer.ptr


        reward_mean_list.append(np.sum(curr_rewards, axis=0))
        preprocess_reward(buffer, curr_rewards, start_idx, end_idx)
        # print(env.current_makespan)
        # exit()
        if debug:
            break

def collect_data_new(buffer, job_lenght_list, opt_list, action_list_list, FJSP_env, reward_scale, reward_bias, debug=False, use_mean=False, reward_scaling=None, use_mask=True, cutt_off_episode_prob=0,
                                                                    use_proxy_reward=False, use_sparse_reward=False):
    reward_mean_list = []
    return_list = []
    makespan_list = []
    v1_list = []
    v2_list = []
    comb_list = zip(job_lenght_list, opt_list, action_list_list)
    for i in tqdm.tqdm(range(len(action_list_list))):
        job_lenght, opt, action_list = next(comb_list)

        num_runs = len(action_list)


        job_lenght = np.array(job_lenght)
        opt = np.array(opt)
        # print(instance_dict.keys())
        # exit()
        n_j = job_lenght.shape[0]
        n_op, n_m = opt.shape
        JobLength_dataset = np.tile(np.expand_dims(job_lenght, axis=0), (num_runs, 1))
        OpPT_dataset = np.tile(np.expand_dims(opt, axis=0), (num_runs, 1, 1))
        env = FJSP_env(n_j=n_j, n_m=n_m, mask_actions=use_mask)
        state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
        state = copy.deepcopy(state)
        # action_list = [rules[rule] for rule in rules]
        curr_rewards = np.zeros((len(action_list[0]), num_runs))

        action_arr = np.array(action_list)


        i = 0
        start_idx = buffer.ptr
        curr_sum_rewards = np.zeros((num_runs,))

        for action in action_arr.T:
            # print(action.shape)
            # exit()
            cutt_off = np.random.rand() < cutt_off_episode_prob
            next_state, reward, done = env.step(action)
            reward = (reward * reward_scale) + reward_bias
            # reward *= 100
            if cutt_off:
                done[:] = True
                print("Episode cutt off")
            if use_proxy_reward:
                reward[reward < 0] = -1
            elif use_sparse_reward:
                print("Using sparse reward")
                if not done.all():
                    curr_sum_rewards += reward
                    reward[:] = 0
                else:
                    reward = curr_sum_rewards + reward

            buffer.push(state, next_state, action, reward, done)
                

            curr_rewards[i] = reward
            i += 1
            # del state
            state = copy.deepcopy(next_state)
            if cutt_off:
                break
        makespan_list.append(env.current_makespan.tolist())

        end_idx = buffer.ptr

        reward_mean_list.append(np.sum(curr_rewards, axis=0))
        curr_rewards, v1, v2 = preprocess_reward(buffer, curr_rewards, start_idx, end_idx, process=reward_scaling)

        # mc_returns = calculate_discounted_return(curr_rewards, gamma=1)
        # mc_returns = mc_returns.clip(max=0)
        curr_returns = np.sum(curr_rewards, axis=0)
        # min_mc_return_row = np.min(mc_returns, axis=1)
        # max_mc_return_row = np.max(mc_returns, axis=1)
        # norm_returns = (mc_returns - min_mc_return_row[:, np.newaxis]) / (max_mc_return_row - min_mc_return_row)[:, np.newaxis]
        #
        # # print(mc_returns / np.max(np.abs(mc_returns), axis=1, keepdims=True))
        # # print(np.max(np.abs(mc_returns), axis=1).shape)
        # # exit()
        #
        # # norm_returns = 1 - (np.abs(curr_returns) / np.max(np.abs(curr_returns)))
        # # norm_returns = np.repeat(norm_returns[np.newaxis, :], curr_rewards.shape[0], axis=0)
        # # print(norm_returns.shape)
        # # print(mc_returns.shape)
        # # exit()
        #TODO: Fix this
        # buffer.set_mc_return(mc_returns.ravel(order="C"), start_idx, end_idx)

        return_list.append(curr_returns.tolist())


        v1_list.append(v1)
        v2_list.append(v2)

        if debug:
            break
    return return_list, makespan_list, v1_list, v2_list

def remove_duplicate_actions(action_lists):
    unique_action_list = []
    for action_list in action_lists:
        not_found = True
        for current_action_list in unique_action_list:
            if action_list == current_action_list:
                # print("hij komt hier")
                not_found = False
                break
        if not_found:
            unique_action_list.append(action_list)
    removed_len = len(action_lists) - len(unique_action_list)
    return unique_action_list, removed_len


def eval_model_all(model, eval_instances: List, FJSP_env, device="cpu", deterministic=False, num_runs=100,
                   use_mask=True, mean_fea_j=torch.tensor(0), std_fea_j=torch.tensor(1), mean_fea_m=torch.tensor(0),
                   std_fea_m=torch.tensor(1), mean_fea_pairs=torch.tensor(0), std_fea_pairs=torch.tensor(1)):
    model.eval()
    mean_fea_j = mean_fea_j.to(device)
    std_fea_j = std_fea_j.to(device)
    mean_fea_m = mean_fea_m.to(device)
    std_fea_m = std_fea_m.to(device)
    mean_fea_pairs = mean_fea_pairs.to(device)
    std_fea_pairs = std_fea_pairs.to(device)

    job_lenght = [np.array(instance_dict["JobLength"]) for instance_dict in eval_instances]

    opt = [np.array(instance_dict["OpPT"]) for instance_dict in eval_instances]
    n_j = job_lenght[0].shape[0]
    n_op, n_m = opt[0].shape
    env = FJSP_env(n_j=n_j, n_m=n_m, device=device, mask_actions=use_mask)
    state = env.set_initial_data(job_lenght, opt)
    done = torch.zeros(len(job_lenght), dtype=torch.bool)

    start_time = time.time()
    while not done.all():
        batch_idx = ~torch.from_numpy(env.done_flag)
        state.fea_j_tensor = (state.fea_j_tensor - mean_fea_j) / (std_fea_j + 1e-8)
        state.fea_m_tensor = (state.fea_m_tensor - mean_fea_m) / (std_fea_m + 1e-8)
        state.fea_pairs_tensor = (state.fea_pairs_tensor - mean_fea_pairs) / (std_fea_pairs + 1e-8)
        with torch.no_grad():
            action = model.get_action(
                fea_j=state.fea_j_tensor[batch_idx],
                op_mask=state.op_mask_tensor[batch_idx],
                candidate=state.candidate_tensor[batch_idx],
                fea_m=state.fea_m_tensor[batch_idx],
                mch_mask=state.mch_mask_tensor[batch_idx],
                comp_idx=state.comp_idx_tensor[batch_idx],
                dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],
                fea_pairs=state.fea_pairs_tensor[batch_idx],
                deterministic=deterministic
            )
            action = action.cpu().numpy()
        next_state, reward, done = env.step(action)
        state = next_state
    run_time = time.time() - start_time
    makespans = env.current_makespan
    model.train()

    return np.mean(makespans), run_time


def eval_model(model, eval_instances: List, FJSP_env, device="cpu", deterministic=False, num_runs=100, use_mask=True, mean_fj=0, std_fj=1, mean_fm=0, std_fm=1):
    model.eval()
    makespans = []
    runtimes = []
    for instance_dict in eval_instances:
        job_lenght = np.array(instance_dict["JobLength"])
        opt = np.array(instance_dict["OpPT"])
        n_j = job_lenght.shape[0]
        n_op, n_m = opt.shape
        JobLength_dataset = np.tile(np.expand_dims(job_lenght, axis=0), (num_runs, 1))
        OpPT_dataset = np.tile(np.expand_dims(opt, axis=0), (num_runs, 1, 1))
        # rules = instance_dict["rules_results"]
        env = FJSP_env(n_j=n_j, n_m=n_m, device=device, mask_actions=use_mask)
        state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
        # print(state)
        # exit()
        done = torch.zeros(num_runs, dtype=torch.bool)
        start_time = time.time()
        while not done.all():
            with torch.no_grad():
                action = model.get_action(
                    fea_j=state.fea_j_tensor,
                    op_mask=state.op_mask_tensor,
                    candidate=state.candidate_tensor,
                    fea_m=state.fea_m_tensor,
                    mch_mask=state.mch_mask_tensor,
                    comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                    fea_pairs=state.fea_pairs_tensor,
                    deterministic=deterministic
                )
                # action = action.cpu().numpy()
            next_state, reward, done = env.step(action)
            state = next_state
        run_time = time.time() - start_time
        runtimes.append(run_time)
        curr_makespan = np.min(env.current_makespan)
        makespans.append(curr_makespan)

    model.train()
    return np.mean(makespans), runtimes

def do_run(state, env, model, num_runs, deterministic=False, mean_fea_j=torch.tensor(0), std_fea_j=torch.tensor(1),
                       mean_fea_m=torch.tensor(0), std_fea_m=torch.tensor(1), mean_fea_pairs=torch.tensor(0),
                       std_fea_pairs=torch.tensor(1)):
    done = torch.zeros(num_runs, dtype=torch.bool)
    start_time = time.time()
    while not done.all():
        state.fea_j_tensor = (state.fea_j_tensor - mean_fea_j) / (std_fea_j + 1e-8)
        state.fea_m_tensor = (state.fea_m_tensor - mean_fea_m) / (std_fea_m + 1e-8)
        state.fea_pairs_tensor = (state.fea_pairs_tensor - mean_fea_pairs) / (std_fea_pairs + 1e-8)

        with torch.no_grad():

            action = model.get_action(
                fea_j=state.fea_j_tensor,
                op_mask=state.op_mask_tensor,
                candidate=state.candidate_tensor,
                fea_m=state.fea_m_tensor,
                mch_mask=state.mch_mask_tensor,
                comp_idx=state.comp_idx_tensor,
                dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                fea_pairs=state.fea_pairs_tensor,
                deterministic=deterministic
            )
            action = action.cpu().numpy()
        next_state, reward, done = env.step(action)
        state = next_state
        # state.fea_j_tensor = (state.fea_j_tensor - mean_fj) / (std_fj + 1e-8)
        # state.fea_m_tensor = (state.fea_m_tensor - mean_fm) / (std_fm + 1e-8)
    run_time = time.time() - start_time

    # runtimes.append(run_time)
    curr_makespan = np.min(env.current_makespan)
    return curr_makespan, run_time


def eval_model_complex(model, name_instance_list, jobLength_list, opList, FJSP_env, benchmark_df, device="cpu", num_runs=100,
                       deterministic=False, use_mask=True, mean_fea_j=torch.tensor(0), std_fea_j=torch.tensor(1),
                       mean_fea_m=torch.tensor(0), std_fea_m=torch.tensor(1), mean_fea_pairs=torch.tensor(0),
                       std_fea_pairs=torch.tensor(1), num_stoch_runs=5):
    model.eval()
    res_dict = {
        "name": [],
        "makespan": [],
        "runtime": [],
        "gap": []
    }
    for instance_name, job_lenght, opt in zip(name_instance_list, jobLength_list, opList):
        # job_lenght = np.array(instance_dict["JobLength"])
        # opt = np.array(instance_dict["OpPT"])

        ub = benchmark_df[benchmark_df["filename"] == instance_name]["ub"].tolist()[0]

        n_j = job_lenght.shape[0]
        n_op, n_m = opt.shape
        JobLength_dataset = np.tile(np.expand_dims(job_lenght, axis=0), (num_runs, 1))
        OpPT_dataset = np.tile(np.expand_dims(opt, axis=0), (num_runs, 1, 1))
        # rules = instance_dict["rules_results"]



        env = FJSP_env(n_j=n_j, n_m=n_m, device=device, mask_actions=use_mask)
        if deterministic:
            state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
            curr_makespan, run_time = do_run(state, env, model, num_runs, deterministic, mean_fea_j=mean_fea_j,
                                             std_fea_j=std_fea_j, mean_fea_m=mean_fea_m, std_fea_m=std_fea_m,
                                             mean_fea_pairs=mean_fea_pairs, std_fea_pairs=std_fea_pairs)
        else:
            makespan_list = []
            runtimes = []
            for i in range(num_stoch_runs):
                state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
                curr_makespan_s, curr_run_time = do_run(state, env, model, num_runs, deterministic, mean_fea_j=mean_fea_j,
                                             std_fea_j=std_fea_j, mean_fea_m=mean_fea_m, std_fea_m=std_fea_m,
                                             mean_fea_pairs=mean_fea_pairs, std_fea_pairs=std_fea_pairs)
                makespan_list.append(curr_makespan_s)
                runtimes.append(curr_run_time)
            curr_makespan = np.mean(makespan_list)
            run_time = np.mean(runtimes)

        # state.fea_j_tensor = (state.fea_j_tensor - mean_fj) / (std_fj + 1e-8)
        # state.fea_m_tensor = (state.fea_m_tensor - mean_fm) / (std_fm + 1e-8)
        # print(state)
        # exit()
        # done = torch.zeros(num_runs, dtype=torch.bool)
        # start_time = time.time()
        # while not done.all():
        #     with torch.no_grad():
        #         action = model.get_action(
        #             fea_j=state.fea_j_tensor,
        #             op_mask=state.op_mask_tensor,
        #             candidate=state.candidate_tensor,
        #             fea_m=state.fea_m_tensor,
        #             mch_mask=state.mch_mask_tensor,
        #             comp_idx=state.comp_idx_tensor,
        #             dynamic_pair_mask=state.dynamic_pair_mask_tensor,
        #             fea_pairs=state.fea_pairs_tensor,
        #             deterministic=deterministic
        #         )
        #         action = action.cpu().numpy()
        #     next_state, reward, done = env.step(action)
        #     state = next_state
        #     # state.fea_j_tensor = (state.fea_j_tensor - mean_fj) / (std_fj + 1e-8)
        #     # state.fea_m_tensor = (state.fea_m_tensor - mean_fm) / (std_fm + 1e-8)
        # run_time = time.time() - start_time

        # runtimes.append(run_time)
        # curr_makespan = np.min(env.current_makespan)
        gap = ((curr_makespan - ub) / ub) * 100
        print("Instance: ", instance_name, "Makespan: ", curr_makespan, "gap", gap, "Runtime: ", run_time)
        # exit()
        res_dict["name"].append(instance_name)
        res_dict["makespan"].append(curr_makespan)
        res_dict["runtime"].append(run_time)

        res_dict["gap"].append(gap)
        # makespans.append(curr_makespan)

    # model.train()
    res_df = pd.DataFrame.from_dict(res_dict)
    # print(res_df)
    return res_df


def eval_model_complex_no_gap(model, name_instance_list, jobLength_list, opList, FJSP_env, benchmark_df, device="cpu", num_runs=100,
                       deterministic=False, mean_fj=0, std_fj=1, mean_fm=0, std_fm=1, use_mask=True):
    model.eval()
    res_dict = {
        "name": [],
        "makespan": [],
        "runtime": [],
        # "gap": []
    }
    for instance_name, job_lenght, opt in zip(name_instance_list, jobLength_list, opList):
        # job_lenght = np.array(instance_dict["JobLength"])
        # opt = np.array(instance_dict["OpPT"])

        n_j = job_lenght.shape[0]
        n_op, n_m = opt.shape
        JobLength_dataset = np.tile(np.expand_dims(job_lenght, axis=0), (num_runs, 1))
        OpPT_dataset = np.tile(np.expand_dims(opt, axis=0), (num_runs, 1, 1))
        # rules = instance_dict["rules_results"]



        env = FJSP_env(n_j=n_j, n_m=n_m, device=device, mask_actions=use_mask)
        state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
        # state.fea_j_tensor = (state.fea_j_tensor - mean_fj) / (std_fj + 1e-8)
        # state.fea_m_tensor = (state.fea_m_tensor - mean_fm) / (std_fm + 1e-8)
        # print(state)
        # exit()
        done = torch.zeros(num_runs, dtype=torch.bool)
        start_time = time.time()
        while not done.all():
            with torch.no_grad():
                action = model.get_action(
                    fea_j=state.fea_j_tensor,
                    op_mask=state.op_mask_tensor,
                    candidate=state.candidate_tensor,
                    fea_m=state.fea_m_tensor,
                    mch_mask=state.mch_mask_tensor,
                    comp_idx=state.comp_idx_tensor,
                    dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                    fea_pairs=state.fea_pairs_tensor,
                    deterministic=deterministic
                )
                action = action.cpu().numpy()
            next_state, reward, done = env.step(action)
            state = next_state
            # state.fea_j_tensor = (state.fea_j_tensor - mean_fj) / (std_fj + 1e-8)
            # state.fea_m_tensor = (state.fea_m_tensor - mean_fm) / (std_fm + 1e-8)
        run_time = time.time() - start_time

        # runtimes.append(run_time)
        curr_makespan = np.min(env.current_makespan)
        print("Instance: ", instance_name, "Makespan: ", curr_makespan, "Runtime: ", run_time)
        # exit()
        res_dict["name"].append(instance_name)
        res_dict["makespan"].append(curr_makespan)
        res_dict["runtime"].append(run_time)

        # makespans.append(curr_makespan)

    # model.train()
    res_df = pd.DataFrame.from_dict(res_dict)

    return res_df


class WrapperEnv:
    def __init__(self, env, mean_fj, std_fj, mean_fm, std_fm):
        self.env = env
        self.mean_fj = mean_fj
        self.std_fj = std_fj
        self.mean_fm = mean_fm
        self.std_fm = std_fm

    def step(self, action):
        next_state, reward, done = self.env.step(action)
        next_state.fea_j_tensor = (next_state.fea_j_tensor - self.mean_fj) / (self.std_fj + 1e-8)
        next_state.fea_m_tensor = (next_state.fea_m_tensor - self.mean_fm) / (self.std_fm + 1e-8)
        return next_state, reward, done

    def set_initial_data(self, job_length, op_pt):
        state = self.env.set_initial_data(job_length, op_pt)
        state.fea_j_tensor = (state.fea_j_tensor - self.mean_fj) / (self.std_fj + 1e-8)
        state.fea_m_tensor = (state.fea_m_tensor - self.mean_fm) / (self.std_fm + 1e-8)
        return state

    def reset(self):
        state = self.env.reset()
        state.fea_j_tensor = (state.fea_j_tensor - self.mean_fj) / (self.std_fj + 1e-8)
        state.fea_m_tensor = (state.fea_m_tensor - self.mean_fm) / (self.std_fm + 1e-8)
        return state

    @property
    def current_makespan(self):
        return self.env.current_makespan

def comb_batch(online_batch, offline_batch):
    online_state, online_next_state, online_actions, online_rewards, online_dones, online_mc_returns = online_batch
    offline_state, offline_next_state, offline_actions, offline_rewards, offline_dones, offline_mc_returns = offline_batch
    state = []
    for i in range(len(online_state)):
        state.append(torch.cat([online_state[i], offline_state[i]], dim=0))
    next_state = []
    for i in range(len(online_next_state)):
        next_state.append(torch.cat([online_next_state[i], offline_next_state[i]], dim=0))
    actions = torch.cat([online_actions, offline_actions], dim=0)
    rewards = torch.cat([online_rewards, offline_rewards], dim=0)
    dones = torch.cat([online_dones, offline_dones], dim=0)
    mc_returns = torch.cat([online_mc_returns, offline_mc_returns], dim=0)
    return state, next_state, actions, rewards, dones, mc_returns


