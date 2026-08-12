from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums as FJSPEnv
import numpy as np
import torch
import os
import tqdm
import argparse

def run_trajectory(curr_instance, num_random):
    jobLength = np.array([curr_instance['JobLength']]).repeat(num_random, axis=0)
    OpPT = np.array([curr_instance['OpPT']]).repeat(num_random, axis=0)
    n_m = OpPT.shape[-1]
    n_o = OpPT.shape[1]
    # print(n_m, n_o)
    # exit()
    n_j = jobLength.shape[-1]
    action_array = np.zeros((n_o, num_random), dtype=int)
    # print(OpPT.shape)

    env = FJSPEnv(n_j, n_m, device="cpu", mask_actions=True)
    state = env.set_initial_data(jobLength, OpPT)
    done = env.done()
    i = 0
    while not done.all():
        mask = (~state.dynamic_pair_mask_tensor.flatten(1))
        actions = torch.multinomial(mask.float(), 1)

        test = mask[0, actions[0].item()].item()
        test2 = mask[1, actions[1].item()].item()
        if not test or not test2:
            print("ERROR")


        actions = actions.squeeze(-1).cpu().numpy()

        state, reward, done = env.step(actions)
        action_array[i] = actions
        i += 1

    # print("DONE")
    curr_makespan = env.current_makespan.astype(int)
    return action_array, curr_makespan

def main(seed, num_random, path_file):
    # SEED = 100
    # NUM_RANDON_TRAJECTORIES = 300
    # path_file = './dataset/SD1_train_10_5_1000.npy'
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(path_file, allow_pickle=True)
    num_instances = len(data)
    for i in tqdm.tqdm(range(num_instances)):
        curr_data = data[i]
        action_array, curr_makespan = run_trajectory(curr_data, num_random)
        data[i]["random"] = np.transpose(action_array).tolist()
        data[i]["random_makespan"] = curr_makespan.tolist()
    np.save(path_file, data, allow_pickle=True)
        # get


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_file", type=str, required=True)
    parser.add_argument("--num_random", type=int, default=300)
    parser.add_argument("--seed", type=int, default=100)
    args = parser.parse_args()

    main(args.seed, args.num_random, args.path_file)
