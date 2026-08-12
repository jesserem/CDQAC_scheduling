import json
import random

from torch.distributions.categorical import Categorical
import sys
import numpy as np
import torch
import copy

"""
    agent utils
"""


def sample_action(p):
    """
        sample an action by the distribution p
    :param p: this distribution with the probability of choosing each action
    :return: an action sampled by p
    """
    dist = Categorical(p)
    s = dist.sample()  # index
    return s, dist.log_prob(s)


def eval_actions(p, actions):
    """
    :param p: the policy
    :param actions: action sequences
    :return: the log probability of actions and the entropy of p
    """
    softmax_dist = Categorical(p.squeeze())
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p):
    _, index = torch.max(p, dim=1)
    return index


def min_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the minimum element of the array
    """
    min_element = np.min(array)
    candidate = np.where(array == min_element)
    return candidate


def max_element_index(array):
    """
    :param array: an array with numbers
    :return: Index set corresponding to the maximum element of the array
    """
    max_element = np.max(array)
    candidate = np.where(array == max_element)
    return candidate


def available_mch_list_for_job(chosen_job, env):
    """
    :param chosen_job: the selected job
    :param env: the production environment
    :return: the machines which can immediately process the chosen job
    """
    # print(env)
    mch_state = ~env.candidate_process_relation[0, chosen_job]
    available_mch_list = np.where(mch_state == True)[0]
    mch_free_time = env.mch_free_time[0][available_mch_list]
    job_free_time = env.candidate_free_time[0][chosen_job]
    # case1 eg:
    # JF: 50
    # MchF: 55 60 65 70
    if (job_free_time < mch_free_time).all():
        chosen_mch_list = available_mch_list[min_element_index(mch_free_time)]
    # case2 eg:
    # JF: 50
    # MchF: 35 40 55 60
    else:
        chosen_mch_list = available_mch_list[np.where(mch_free_time <= job_free_time)]

    return chosen_mch_list

def heuristic_select_action(method, env, deterministic=True):
    """
    :param method: the name of heuristic method
    :param env: the environment
    :return: the action selected by the heuristic method

    here are heuristic methods selected for comparison:

    FIFO: First in first out
    MOR(or MOPNR): Most operations remaining
    SPT: Shortest processing time
    MWKR: Most work remaining
    """
    chosen_job = -1
    chosen_mch = -1

    job_state = (env.mask[0] == 0)


    process_job_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
    job_state = process_job_state & job_state
    # print(job_state)

    # available_jobs = np.where(job_state == True)[0]
    # print(available_jobs)
    # print(~env.state.dynamic_pair_mask_tensor[0].sum(1))
    num_dim = env.state.dynamic_pair_mask_tensor[0].shape[1]

    available_jobs = np.where(env.state.dynamic_pair_mask_tensor[0].sum(1) != num_dim)[0]
    # print(env.state.dynamic_pair_mask_tensor[0])
    # print(available_jobs)
    # exit()

    available_ops = env.candidate[0][available_jobs]

    if method == 'FIFO':
        # print(available_ops)
        # exit()
        # selecting the earliest ready candidate operation
        candidate_free_time = env.candidate_free_time[0][available_jobs]
        # print(env.candidate_free_time[0][available_jobs])


        chosen_job_list = available_jobs[min_element_index(candidate_free_time)]
        # print(chosen_job_list)

        if deterministic:
            chosen_job = chosen_job_list[0]
        else:
            chosen_job = np.random.choice(chosen_job_list)

        # select the earliest ready machine
        mch_state = ~env.candidate_process_relation[0, chosen_job]
        # print(mch_state)
        available_mchs = np.where(mch_state == True)[0]
        # print(mch_state)
        # print(available_mchs)
        # print(env.candidate_process_relation[0, chosen_job])
        # exit()
        mch_free_time = env.mch_free_time[0][available_mchs]

        chosen_mch_list = available_mchs[min_element_index(mch_free_time)[0]]
        # print(chosen_mch_list)
        # print(available_jobs, available_mchs)
        # print(chosen_mch_list)

        if deterministic:
            chosen_mch = chosen_mch_list[0]
        else:
            chosen_mch = np.random.choice(chosen_mch_list)

        # print("chosen_job", chosen_job, "chosen_machien", chosen_mch)

    elif method == 'MOR':

        remain_ops = env.op_match_job_left_op_nums[0][available_ops]
        chosen_job_list = available_jobs[max_element_index(remain_ops)]
        # chosen_job = np.random.choice(chosen_job_list)
        if deterministic:
            chosen_job = chosen_job_list[0]
        else:
            chosen_job = np.random.choice(chosen_job_list)

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)

        if deterministic:
            chosen_mch = chosen_mch_list[0]
        else:
            chosen_mch = np.random.choice(chosen_mch_list)
    elif method == 'LOR':

        remain_ops = env.op_match_job_left_op_nums[0][available_ops]
        chosen_job_list = available_jobs[min_element_index(remain_ops)]
        # chosen_job = np.random.choice(chosen_job_list)
        if deterministic:
            chosen_job = chosen_job_list[0]
        else:
            chosen_job = np.random.choice(chosen_job_list)

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)

        if deterministic:
            chosen_mch = chosen_mch_list[0]
        else:
            chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'SPT':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")
        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.min(pt_list))[0]
        if deterministic:
            action = action_list[0]
        else:
            action = np.random.choice(action_list)
        return action

    elif method == 'SPT_DIV_TWKR':

        temp_pt = copy.deepcopy(env.candidate_pt[0])

        # print(env.ops[0])
        # exit()
        job_remain_work_list = env.op_match_job_remain_work[0]
        job_remain_candidate = job_remain_work_list[env.candidate[0]]
        job_remain_candidate = np.abs(np.tile(job_remain_candidate[:, np.newaxis], (1, temp_pt.shape[1])))
        temp_pt = temp_pt / job_remain_candidate
        # print(temp_pt)
        # print(job_remain_candidate)
        # # print(job_remain_candidate.shape)
        # # print(temp_pt.shape)
        # # print(job_remain_work_list.shape, temp_pt.shape)
        # # print(job_remain_work_list)
        # exit()

        temp_pt[env.dynamic_pair_mask[0]] = float("inf")

        pt_list = temp_pt.reshape(-1)
        # print(pt_list)


        action_list = np.where(pt_list == np.min(pt_list))[0]
        if deterministic:
            action = action_list[0]
        else:
            action = np.random.choice(action_list)
        return action

    elif method == 'SPT_DIV_TWK':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        # all_pt = copy.deepcopy(env.unmasked_op_pt)
        # print(temp_pt, all_pt[0].shape)
        # exit()
        # print(temp_pt.shape)
        job_total_work_list = env.op_match_job_total_work[0]
        job_remain_candidate = job_total_work_list[env.candidate[0]]
        job_remain_candidate = np.abs(np.tile(job_remain_candidate[:, np.newaxis], (1, temp_pt.shape[1])))
        temp_pt = temp_pt / job_remain_candidate
        temp_pt[env.dynamic_pair_mask[0]] = float("inf")


        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.min(pt_list))[0]
        if deterministic:
            action = action_list[0]
        else:
            action = np.random.choice(action_list)
        return action

    elif method == 'LPT':

        temp_pt = copy.deepcopy(env.candidate_pt[0])
        temp_pt[env.dynamic_pair_mask[0]] = float("-inf")
        pt_list = temp_pt.reshape(-1)

        action_list = np.where(pt_list == np.max(pt_list))[0]
        if deterministic:
            action = action_list[0]
        else:
            action = np.random.choice(action_list)
        return action

    elif method == 'MWKR':
        job_remain_work_list = env.op_match_job_remain_work[0][available_ops]

        if deterministic:
            ind = max_element_index(job_remain_work_list)[0][0]
        else:
            ind = np.random.choice(max_element_index(job_remain_work_list)[0])

        chosen_job = available_jobs[ind]

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        if deterministic:
            chosen_mch = chosen_mch_list[0]
        else:
            chosen_mch = np.random.choice(chosen_mch_list)

    elif method == 'LWKR':
        job_remain_work_list = env.op_match_job_remain_work[0][available_ops]

        if deterministic:
            ind = min_element_index(job_remain_work_list)[0][0]
        else:
            ind = np.random.choice(min_element_index(job_remain_work_list)[0])

        chosen_job = available_jobs[ind]

        # select a machine which can immediately process the chosen job
        chosen_mch_list = available_mch_list_for_job(chosen_job, env)
        if deterministic:
            chosen_mch = chosen_mch_list[0]
        else:
            chosen_mch = np.random.choice(chosen_mch_list)

    else:
        print(f'Error From rule select: undefined method {method}')
        sys.exit()

    if chosen_job == -1 or chosen_mch == -1:
        print(f'Error From choosing action: choose job {chosen_job}, mch {chosen_mch}')
        sys.exit()

    action = chosen_job * env.number_of_machines + chosen_mch
    return action

# def heuristic_select_action(method, env, deterministic=True):
#     """
#     :param method: the name of heuristic method
#     :param env: the environment
#     :return: the action selected by the heuristic method
#
#     here are heuristic methods selected for comparison:
#
#     FIFO: First in first out
#     MOR(or MOPNR): Most operations remaining
#     SPT: Shortest processing time
#     MWKR: Most work remaining
#     """
#     chosen_job = -1
#     chosen_mch = -1
#
#     job_state = (env.mask[0] == 0)
#
#     process_job_state = (env.candidate_free_time[0] <= env.next_schedule_time[0])
#     job_state = process_job_state & job_state
#
#     available_jobs = np.where(job_state == True)[0]
#     available_ops = env.candidate[0][available_jobs]
#
#     if method == 'FIFO':
#         # selecting the earliest ready candidate operation
#         candidate_free_time = env.candidate_free_time[0][available_jobs]
#         chosen_job_list = available_jobs[min_element_index(candidate_free_time)]
#         if deterministic:
#             chosen_job = chosen_job_list[0]
#         else:
#             chosen_job = np.random.choice(chosen_job_list)
#
#         # select the earliest ready machine
#         mch_state = ~env.candidate_process_relation[0, chosen_job]
#         available_mchs = np.where(mch_state == True)[0]
#         mch_free_time = env.mch_free_time[0][available_mchs]
#         chosen_mch_list = available_mchs[min_element_index(mch_free_time)]
#
#     elif method == 'MOR':
#         remain_ops = env.op_match_job_left_op_nums[0][available_ops]
#         chosen_job_list = available_jobs[max_element_index(remain_ops)]
#         chosen_job = np.random.choice(chosen_job_list)
#
#         # select a machine which can immediately process the chosen job
#         chosen_mch_list = available_mch_list_for_job(chosen_job, env)
#
#     elif method == 'SPT':
#
#         temp_pt = copy.deepcopy(env.candidate_pt[0])
#         temp_pt[env.dynamic_pair_mask[0]] = float("inf")
#         pt_list = temp_pt.reshape(-1)
#
#         action_list = np.where(pt_list == np.min(pt_list))[0]
#         if deterministic:
#             action = action_list[0]
#         else:
#             action = np.random.choice(action_list)
#         return action
#
#     elif method == 'MWKR':
#         job_remain_work_list = env.op_match_job_remain_work[0][available_ops]
#
#         chosen_job = available_jobs[np.random.choice(max_element_index(job_remain_work_list)[0])]
#
#         # select a machine which can immediately process the chosen job
#         chosen_mch_list = available_mch_list_for_job(chosen_job, env)
#
#     else:
#         print(f'Error From rule select: undefined method {method}')
#         sys.exit()
#     if deterministic:
#         chosen_mch = chosen_mch_list[0]
#     else:
#         chosen_mch = np.random.choice(chosen_mch_list)
#     if chosen_job == -1 or chosen_mch == -1:
#         print(f'Error From choosing action: choose job {chosen_job}, mch {chosen_mch}')
#         sys.exit()
#
#     action = chosen_job * env.number_of_machines + chosen_mch
#     return action


"""
    common utils
"""


def save_default_params(config):
    """
        save parameters in the config
    :param config: a package of parameters
    :return:
    """
    with open('./config_default.json', 'wt') as f:
        json.dump(vars(config), f, indent=4)
    print("successfully save default params")


def nonzero_averaging(x):
    """
        remove zero vectors and then compute the mean of x
        (The deleted nodes are represented by zero vectors)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)


def strToSuffix(str):
    if str == '':
        return str
    else:
        return '+' + str


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    print('123')
