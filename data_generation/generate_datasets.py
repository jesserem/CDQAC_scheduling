import json

from env.common_utils import *
from env.data_utils import SD2_instance_generator_no_config, CaseGenerator
from env.params import configs
from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from tqdm import tqdm
from env.data_utils import pack_data_from_config
import time
import numpy as np
import torch
from collections import defaultdict
import sys
import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id


# def collect_heuristic_method(data_set, heuristic, seed):
#     """
#         test one heuristic method on the given data
#     :param data_set:  test data
#     :param heuristic: the name of heuristic method
#     :param seed: seed for testing
#     :return: the test results including the makespan and time
#     """
#     setup_seed(seed)
#     result = []
#     action_list = []
#
#     for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
#         n_j = data_set[0][i].shape[0]
#         n_op, n_m = data_set[1][i].shape
#         env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
#
#         env.set_initial_data([data_set[0][i]], [data_set[1][i]])
#
#         t1 = time.time()
#         while True:
#             action = heuristic_select_action(heuristic, env)
#             # print(action)
#             # exit()
#
#             state, _, done = env.step(np.array([action]))
#             # print(state)
#
#             if done:
#                 break
#
#         t2 = time.time()
#         # tqdm.write(f'Instance {i + 1} , makespan:{-ep_reward} , time:{t2 - t1}')
#         result.append([env.current_makespan[0], t2 - t1])
#
#     return np.array(result)

def collect_heuristic_method(job_lenght, opt, heuristic, seed, use_sd1=False, noisy_prob=0.1):
    """
        test one heuristic method on the given data
    :param data_set:  test data
    :param heuristic: the name of heuristic method
    :param seed: seed for testing
    :return: the test results including the makespan and time
    """
    setup_seed(seed)
    result = []
    action_list = []


    n_j = job_lenght.shape[0]
    n_op, n_m = opt.shape
    if use_sd1:
        env = FJSPEnvForVariousOpNums(n_j=n_j, n_m=n_m)

    else:
        env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m)
    # print(job_lenght)
    # print(opt)
    # exit()

    state = env.set_initial_data([job_lenght], [opt])

    # print(state.fea_pair_tensor.shape)
    reward_list = []



    while True:
        if np.random.rand() < noisy_prob:
            avail_actions = (~state.dynamic_pair_mask_tensor[0].flatten()).to(torch.float32)
            action = torch.multinomial(avail_actions, 1).item()
        else:
            action = heuristic_select_action(heuristic, env, deterministic=True)
        action_list.append(int(action))
        # print(action)
        job = action // env.number_of_machines
        machine = action % env.number_of_machines
        # print(action)
        # print(state.dynamic_pair_mask_tensor[0].flatten())

        invalid_action = state.dynamic_pair_mask_tensor[0][job][machine]
        if invalid_action:
            print(f"Invalid action: job {job}, machine {machine} violates dynamic_pair_mask")
            print(state.dynamic_pair_mask_tensor[0])
            print(action)
            exit()
        # print(action)
            # exit()

        state, reward, done = env.step(np.array([action]))
        reward_list.append(reward[0])
        # print(state)
        # print()
            # print(state)

        if done:
            break

    # print(env.current_makespan)
    # print(reward_list)
    # exit()
    return action_list, env.current_makespan[0], reward_list


def create_dataset(n_j, n_m, type, num_runs=250, seed=9999):
    """
        test heuristic methods following the config and save the results:
        here are heuristic methods selected for comparison:

        FIFO: First in first out
        MOR(or MOPNR): Most operations remaining
        SPT: Shortest processing time
        MWKR: Most work remaining
    """
    if type == 'SD1':
        use_sd1 = True
    elif type == 'SD2':
        use_sd1 = False
    else:
        raise ValueError("Invalid type")

    op_per_job = 0
    low = 1
    high = 99
    op_per_job_min = int(0.8 * n_m)
    op_per_job_max = int(1.2 * n_m)
    op_per_machine_min = 1
    op_per_machine_max = 5

    noisy_prob = 0
    method = ['SPT_DIV_TWK', 'SPT_DIV_TWKR', 'FIFO', 'MOR', 'SPT', 'MWKR', "LOR", "LWKR", "LPT"]
    # method = ['SPT_DIV_TWKR', 'FIFO', 'MOR', 'SPT', 'MWKR', "LOR", "LWKR", "LPT"]
    # exit()d

    name_dataset = f'{type}_train_{n_j}_{n_m}_{num_runs}.npy'


    setup_seed(seed)
    data_list = []
    makespan_rules = defaultdict(list)
    if not os.path.exists('dataset_backup'):
        os.makedirs('dataset_backup')
    prepare_JobLength = [random.randint(op_per_job_min, op_per_job_max) for _ in range(n_j)]

    for i in range(num_runs):
        curr_instance = {}
        setup_seed(seed + i)


        if use_sd1:
            case = CaseGenerator(n_j, n_m, op_per_job_min, op_per_job_max, nums_ope=prepare_JobLength, path='./test',
                                 flag_doc=False)
            JobLength, OpPT, info = case.get_case(i)

        else:
            JobLength, OpPT, info = SD2_instance_generator_no_config(n_j=n_j, n_m=n_m, op_per_job=op_per_job, low=low,
                                                                  high=high,
                                                                  op_per_mch_min=op_per_machine_min,
                                                                  op_per_mch_max=op_per_machine_max)
        curr_instance['JobLength'] = JobLength.tolist()

        curr_instance['OpPT'] = OpPT.tolist()
        curr_instance["rules"] = {}
        # curr_instance["info"] = info.tolist()
        data_list.append(curr_instance)

    # JobLength, OpPT, _ = SD2_instance_generator_no_config(n_j=n_j, n_m=n_m, op_per_job=op_per_job, low=low, high=high,
    #                                         op_per_mch_min=op_per_machine_min, op_per_mch_max=op_per_machine_max)



    # exit()


    for i in range(num_runs):
        print(f"Generating {i + 1}th instance")
        jobLength = np.array(data_list[i]['JobLength'])
        OpPT = np.array(data_list[i]['OpPT'])
        # print(jobLength)
        # exit()
        current_reward_list = []
        for m in method:
            print("Testing method: ", m)
            result, makespan, reward_list = collect_heuristic_method(jobLength, OpPT, m, i, use_sd1=use_sd1, noisy_prob=noisy_prob)
            data_list[i]["rules"][m] = result
            makespan_rules[m].append(makespan)
            current_reward_list += reward_list
        current_reward_list = np.array(current_reward_list)
        mean_reward = np.mean(current_reward_list)
        std_reward = np.std(current_reward_list)
        data_list[i]["mean_reward"] = mean_reward
        data_list[i]["std_reward"] = std_reward
        # print(f"Mean reward: {mean_reward}, std reward: {std_reward}")
        # exit()

        # print(curr_instance)


    np.save(f'dataset_backup/{name_dataset}', data_list)
    for m in method:
        print(f"Method: {m}, makespan: {np.mean(makespan_rules[m])}")
    # print(data_list[0], data_list[1])




    # for data in test_data:
    #     for i in range(len(data[0])):
    #         job_length_list = data[0][i]
    #         op_pt_list = data[1][i]

            # exit()
        # print(data)
        # exit()
        # print("-" * 25 + "Test Heuristic Methods" + "-" * 25)
        # print('Test Methods:', test_method)
        # print(f"test data name: {configs.data_source},{data[1]}")
        # save_direc = f'./test_results/{configs.data_source}/{data[1]}'
        #
        # if not os.path.exists(save_direc):
        #     os.makedirs(save_direc)
        # for method in test_method:
        #     save_path = save_direc + f'/Result_{method}_{data[1]}.npy'
        #
        #     if (not os.path.exists(save_path)) or configs.cover_heu_flag:
        #         print(f"Heuristic method : {method}")
        #         seed = configs.seed_test
        #
        #         result_5_times = []
        #         # test 5 times, record average makespan and time.
        #         for j in range(5):
        #             result = collect_heuristic_method(data[0], method, seed + j)
        #             result_5_times.append(result)
        #
        #             print(f"the {j + 1}th makespan:", np.mean(result[:, 0]))
        #         result_5_times = np.array(result_5_times)
        #         save_result = np.mean(result_5_times, axis=0)
        #         print(f"testing results of {method}:")
        #         print(f"makespan(sampling): ", save_result[:, 0].mean())
        #         print(f"time: ", save_result[:, 1].mean())
        #         np.save(save_path, save_result)


def main():
    setup_list = [
        (10, 5, "SD2"),
        (10, 5, "SD1"),
        (15, 10, "SD1"),
        (20, 5, "SD1"),
        (20, 10, "SD1"),
        # (10, 5, "SD2"),
        (15, 10, "SD2"),
        (20, 5, "SD2"),
        (20, 10, "SD2"),
    ]
    seed=9999
    num_runs_list = [100, 250, 500, 1000]
    for num_runs in num_runs_list:
        for setup in setup_list:
            n_j, n_m, type = setup
            print(f"Generating dataset for {n_j} jobs, {n_m} machines, {type} type, {num_runs} runs")
            create_dataset(n_j, n_m, type, num_runs=num_runs, seed=seed)

if __name__ == '__main__':
    main()