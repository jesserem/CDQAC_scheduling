import numpy as np
import os
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from env.common_utils import *
from env.data_utils import SD2_instance_generator_no_config, CaseGenerator
from cdqac.utils import open_path, load_yml, load_data_from_files, eval_model_complex, eval_model_complex_no_gap
import pandas as pd
from collections import defaultdict
import itertools
from tqdm import tqdm
import time

from env.dispatching import FJSPDispatching


def main(seed=101):
    """
        test heuristic methods following the config and save the results:
        here are heuristic methods selected for comparison:

        FIFO: First in first out
        MOR(or MOPNR): Most operations remaining
        SPT: Shortest processing time
        MWKR: Most work remaining
    """
    eval_instance_path: str = "./data/BenchData/Hurink_vdata"
    benchmark_result_path: str = "./data/BenchData/BenchDataSolution.csv"
    nameFile: str = "vdata_10x5_dispatching"

    gap_df = pd.read_csv(benchmark_result_path)





    job_selection_rules_masked = ["MWR", "LWR", "LOPNR", "MOPNR"]
    # machine_selection_rules_masked = ["FAM", "LPT", "SPT"]
    machine_selection_rules_masked = ["EST", "LST", "LPT", "SPT"]
    comb_all_masked = list(itertools.product(job_selection_rules_masked, machine_selection_rules_masked))
    name_comb_all_masked = [f"{job_rule}_{machine_rule}" for job_rule, machine_rule in comb_all_masked]
    dict_gap_results = {
        "methods": name_comb_all_masked,
    }
    dict_makespan_results = {
        "methods": name_comb_all_masked,
    }


    setup_seed(seed)
    nameList, jobLengthList, OpPTList = load_data_from_files(eval_instance_path)
    # print(data_list)
    # exit()



    dispatcher = FJSPDispatching()
    for i in tqdm(range(len(nameList)), desc="Running Dispatching Rules"):
        name, jobLength, OpPT = nameList[i], jobLengthList[i], OpPTList[i]
        n_j = jobLength.shape[-1]
        n_m = OpPT.shape[-1]

        env = FJSPEnvForVariousOpNums(n_j, n_m, device="cpu", mask_actions=True)
        makespan_list = []
        gap_list = []
        time_list = []

        ub = gap_df[gap_df["filename"] == name]["ub"].values[0]


        for job_select, machine_select in comb_all_masked:
            dispatcher.set_rules(job_rule=job_select, machine_rule=machine_select)
            env.set_initial_data(np.array([jobLength]), np.array([OpPT]))
            done = False

            start_time = time.time()
            while not done:
                action = dispatcher.select_action(env.state, env.state.dynamic_pair_mask_tensor, env)

                _, _, done = env.step(np.array([action]))
            runtime = (time.time() - start_time)
            makespan = int(env.current_makespan[0])
            gap = ((makespan - ub) / ub) * 100
            makespan_list.append(makespan)
            gap_list.append(gap)
            time_list.append(runtime)
        dict_makespan_results[name + "_makespan"] = makespan_list
        dict_makespan_results[name + "_time"] = time_list
        dict_gap_results[name+ "_gap"] = gap_list
        dict_gap_results[name + "_time"] = time_list

    df_makespan_result = pd.DataFrame(dict_makespan_results)
    print(df_makespan_result)
    # exit()
    df_gap_result = pd.DataFrame(dict_gap_results)
    print(df_gap_result)
    df_makespan_result.to_csv(os.path.join("solutions_generating", f"{nameFile}_makespan.csv"), index=False)
    df_gap_result.to_csv(os.path.join("solutions_generating", f"{nameFile}_gap.csv"), index=False)

main()




