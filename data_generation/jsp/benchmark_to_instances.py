import json

from env.common_utils import *
from env.data_utils import pack_data_from_config, load_data_from_files
from cdqac.utils import load_data_from_files_jsp
import os
# from data_utils import SD2_instance_generator_no_config, CaseGenerator
# from params import configs
# from fjsp_env_same_op_nums import FJSPEnvForSameOpNums
# from fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
# from tqdm import tqdm
# from data_utils import pack_data_from_config
# import time
# import numpy as np
# from collections import defaultdict
# import sys
# import os


def main():
    """
        test heuristic methods following the config and save the results:
        here are heuristic methods selected for comparison:

        FIFO: First in first out
        MOR(or MOPNR): Most operations remaining
        SPT: Shortest processing time
        MWKR: Most work remaining
    """

    # method = ['SPT_DIV_TWKR', 'FIFO', 'MOR', 'SPT', 'MWKR', "LOR", "LWKR", "LPT"]
    # exit()d

    name_dataset = 'TA.npy'
    # path_folder = './data/SD1/15x10'
    # if not os.path.exists(path_folder):
    #     raise Exception("Path not found", path_folder)
    # print(path_folder, name_dataset)
    # collect_data
    data_list = []
    name, joblength, opts = load_data_from_files_jsp('./data/JSP/TA')
    print(name)
    for i in range(len(joblength)):
        job_list = joblength[i].tolist()
        opt = opts[i].tolist()
        curr_instance = {}
        curr_instance['JobLength'] = job_list
        curr_instance['OpPT'] = opt
        data_list.append(curr_instance)
    #     print(job_list)
    # exit()
    # print(p1, p2)
    # exit()
    # all_datasets = [list(zip(p1, p2))]
    # print(all_datasets)
    # all_datasets = pack_data_from_config("data_train_vali", ["Hurink_vdata"])
    # print(all_datasets)
    # exit()


    # data_list = []
    # # if not os.path.exists('./dataset'):
    # #     os.makedirs('./dataset')
    #
    # for dataset in all_datasets:
    #     job_list = dataset[0][0]
    #     pt_list = dataset[0][1]
    #     print(job_list)
    #     print(pt_list)
    #     exit()
    #     for job, pt in zip(job_list, pt_list):
    #         curr_instance = {}
    #         curr_instance['JobLength'] = job.tolist()
    #         curr_instance['OpPT'] = pt.tolist()
    #         data_list.append(curr_instance)

    np.save(f'./jsp_benchmark_instances/{name_dataset}', data_list)
    # print(data_list)


if __name__ == '__main__':
    main()