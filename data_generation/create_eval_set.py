import numpy as np
import os
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from env.common_utils import *
from env.data_utils import SD2_instance_generator_no_config, CaseGenerator
from collections import defaultdict
import itertools
from tqdm import tqdm

def main(seed=300):
    """
        test heuristic methods following the config and save the results:
        here are heuristic methods selected for comparison:

        FIFO: First in first out
        MOR(or MOPNR): Most operations remaining
        SPT: Shortest processing time
        MWKR: Most work remaining
    """
    dataset_folder = './jsp_dataset'
    op_per_job = 0
    n_j = 15
    n_m = 10
    low = 1
    high = 99
    op_per_job_min = int(0.8 * n_m)
    op_per_job_max = int(1.2 * n_m)
    op_per_machine_min = 1
    op_per_machine_max = 1
    # checkpoints = [500]
    num_runs = 100
    use_sd1 = False
    noisy_prob = 0
    method = ['SPT_DIV_TWK', 'SPT_DIV_TWKR', 'FIFO', 'MOR', 'SPT', 'MWKR', "LOR", "LWKR", "LPT"]
    # method = ['SPT_DIV_TWKR', 'FIFO', 'MOR', 'SPT', 'MWKR', "LOR", "LWKR", "LPT"]
    # exit()d


    job_selection_rules_masked = ["MWR", "LWR", "LOPNR", "MOPNR"]
    # machine_selection_rules_masked = ["FAM", "LPT", "SPT"]
    machine_selection_rules_masked = ["LCT", "MCT", "EST", "LST",  "LPT", "SPT"]
    machine_selection_rules_masked = ["SPT"]

    comb_all_masked = list(itertools.product(job_selection_rules_masked, machine_selection_rules_masked))
    # job_selection_rules_unmasked = ["MWR", "LWR", "LOPNR", "MOPNR"]
    # machine_selection_rules_unmasked = ["LCT", "MCT", "EST", "LST", "LPT", "SPT"]
    # comb_all_unmasked = list(itertools.product(job_selection_rules_unmasked, machine_selection_rules_unmasked))
    # print(len(comb_all))
    # exit()
    makespan_masked = []
    makespan_unmasked = []


    setup_seed(seed)
    data_list = []
    makespan_rules = defaultdict(list)
    # if not os.path.exists('./dataset'):
    #     os.makedirs('./dataset')
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
        curr_instance["rules_info"] = defaultdict(dict)
        # curr_instance["info"] = info.tolist()
        data_list.append(curr_instance)
    name_dataset = f'eval_{n_j}_{n_m}.npy'
    # name_dataset = f'SD2_train_{n_j}_{n_m}_{c_point}.npy'
    np.save(os.path.join(dataset_folder, name_dataset), data_list)

if __name__ == "__main__":
    main()