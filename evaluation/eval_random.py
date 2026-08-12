from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums as FJSPEnv
import numpy as np
import torch
import os
from cdqac.utils import open_path, load_yml, load_data_from_files, eval_model_complex, eval_model_complex_no_gap, load_data_from_files_jsp
import tqdm
import argparse
import pandas as pd

def run_trajectory(jobLengthI, OpPTI, num_random):
    jobLength = np.array([jobLengthI]).repeat(num_random, axis=0)
    OpPT = np.array([OpPTI]).repeat(num_random, axis=0)
    n_m = OpPT.shape[-1]
    n_o = OpPT.shape[1]
    # print(n_m, n_o)
    # exit()
    n_j = jobLength.shape[-1]
    # print(OpPT.shape)

    env = FJSPEnv(n_j, n_m, device="cpu", mask_actions=True)
    state = env.set_initial_data(jobLength, OpPT)
    done = env.done()
    i = 0
    while not done.all():
        mask = (~state.dynamic_pair_mask_tensor.flatten(1))
        actions = torch.multinomial(mask.float(), 1)



        actions = actions.squeeze(-1).cpu().numpy()

        state, reward, done = env.step(actions)

        i += 1

    # print("DONE")
    curr_makespan = env.current_makespan.astype(int)
    return curr_makespan

def main(eval_instance_path, nameFile, seed, num_random):
    # SEED = 100
    # NUM_RANDON_TRAJECTORIES = 300
    # path_file = './dataset/SD1_train_10_5_1000.npy'
    torch.manual_seed(seed)
    np.random.seed(seed)

    # eval_instance_path: str = "./data/BenchData/Hurink_rdata"
    # benchmark_result_path: str = "./data/BenchData/BenchDataSolution.csv"
    # nameFile: str = "rdata_random"

    # eval_instance_path: str = "./data/JSP/LA"
    benchmark_result_path: str = "./data/JSP/benchmark_results.csv"
    # nameFile: str = "sd1_20x5_random"

    gap_df = pd.read_csv(benchmark_result_path)
    dict_makespan = {}
    dict_gap = {}

    nameList, jobLengthList, OpPTList = load_data_from_files_jsp(eval_instance_path)
    num_instances = len(nameList)
    for i in tqdm.tqdm(range(num_instances)):
        name, jobLength, OpPT = nameList[i], jobLengthList[i], OpPTList[i]
        ub = gap_df[gap_df["filename"] == name]["ub"].values[0]
        # print(gap_df)
        # exit()
        curr_makespan = run_trajectory(jobLength, OpPT, num_random)

        curr_gap = ((curr_makespan - ub) / ub) * 100
        dict_makespan[name] = curr_makespan
        dict_gap[name] = curr_gap

    df_gap = pd.DataFrame(dict_gap)
    df_makespan = pd.DataFrame(dict_makespan)

    df_makespan.to_csv(os.path.join("./jsp_solutions", f"{nameFile}_makespan.csv"), index=False)
    df_gap.to_csv(os.path.join("./jsp_solutions", f"{nameFile}_gap.csv"), index=False)

main("./data/JSP/LA", "la_random", 100, 100)
main("./data/JSP/TA", "tai_random", 100, 100)
main("./data/JSP/DMU", "dmu_random", 100, 100)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--eval_instance_path', type=str, required=True)
#     parser.add_argument('--name', type=str, required=True)
#     parser.add_argument("--num_random", type=int, default=100)
#     parser.add_argument("--seed", type=int, default=100)
#     args = parser.parse_args()
#
#     main(args.eval_instance_path, args.name, args.seed, args.num_random)
