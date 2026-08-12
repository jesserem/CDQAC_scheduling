"""Merge GA trajectories into an existing packed instance dataset.

Reads the per-instance JSON files produced by ``run_GA_custom.py`` (one file
per instance, named ``<dataset>_GA_<idx>.json``, each holding the GA
hall-of-fame and final-population action lists with their makespans) and
stores them in the ``.npy`` dataset under the keys ``ga``/``ga_makespan``
(hall of fame) and ``ga_pop``/``ga_pop_makespan`` (population).

Example:
    python -m data_generation.add_ga_data \
        --dataset ./dataset/SD1_train_10_5_500.npy \
        --ga_folder ./GA_data/10_5 \
        --num_instances 500
"""
import argparse
import json
import os

import numpy as np
import tqdm


def get_info_file(path):
    with open(path, "r") as f:
        dict_json = json.load(f)
    action_list_pop = dict_json["pop"]["action_list"]
    makespans_pop = dict_json["pop"]["makespan"]
    action_list_hof = dict_json["hof"]["action_list"]
    makespans_hof = dict_json["hof"]["makespan"]
    return action_list_pop, makespans_pop, action_list_hof, makespans_hof


def check_folder(path_folder, base_name, start_idx, end_idx):
    list_folder = os.listdir(path_folder)
    for i in range(start_idx, end_idx):
        file_name = f"{base_name}_GA_{i}.json"
        if file_name not in list_folder:
            raise FileNotFoundError(f"File {file_name} not found in {path_folder}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Path to the packed .npy instance dataset to update in place")
    parser.add_argument("--ga_folder", required=True, help="Folder containing the <dataset>_GA_<idx>.json files")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_instances", type=int, default=500)
    args = parser.parse_args()

    base_name = os.path.basename(args.dataset)[:-4]
    data = np.load(args.dataset, allow_pickle=True)
    end_idx = args.start_idx + args.num_instances
    check_folder(args.ga_folder, base_name, args.start_idx, end_idx)
    for i in tqdm.tqdm(range(args.start_idx, end_idx)):
        file_name = f"{base_name}_GA_{i}.json"
        path = os.path.join(args.ga_folder, file_name)
        action_list_pop, makespans_pop, action_list_hof, makespans_hof = get_info_file(path)
        data[i]["ga"] = action_list_hof
        data[i]["ga_makespan"] = makespans_hof
        data[i]["ga_pop"] = action_list_pop
        data[i]["ga_pop_makespan"] = makespans_pop
    np.save(args.dataset, data)


if __name__ == "__main__":
    main()
