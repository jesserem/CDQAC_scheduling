"""Add GA trajectories to a dataset as its "ga_hof" and "ga_pop" sources.

Reads the per-instance JSON files produced by ``run_GA_custom.py`` (one file per instance,
named ``<prefix>_GA_<idx>.json``, each holding the GA hall-of-fame and final-population
action lists with their makespans).

Example:
    python -m data_generation.add_ga_data --dataset fjsp/sd1_10x5 \
        --ga_folder ./GA_data/sd1_10x5 --num_instances 500

``--prefix`` defaults to the dataset name (sd1_10x5), which is what run_GA_custom.py uses for
a dataset directory.
"""
import argparse
import json
import os

import tqdm

from cdqac.data.writer import add_source, dataset_dir


def get_info_file(path):
    with open(path, "r") as f:
        dict_json = json.load(f)
    action_list_pop = dict_json["pop"]["action_list"]
    makespans_pop = dict_json["pop"]["makespan"]
    action_list_hof = dict_json["hof"]["action_list"]
    makespans_hof = dict_json["hof"]["makespan"]
    return action_list_pop, makespans_pop, action_list_hof, makespans_hof


def check_folder(path_folder, prefix, start_idx, end_idx):
    list_folder = set(os.listdir(path_folder))
    for i in range(start_idx, end_idx):
        file_name = f"{prefix}_GA_{i}.json"
        if file_name not in list_folder:
            raise FileNotFoundError(f"File {file_name} not found in {path_folder}.")


def read_ga_folder(ga_folder, prefix, num_instances, start_idx=0):
    """(hof actions, hof makespans, pop actions, pop makespans), one entry per instance."""
    end_idx = start_idx + num_instances
    check_folder(ga_folder, prefix, start_idx, end_idx)
    hof, hof_ms, pop, pop_ms = [], [], [], []
    for i in tqdm.trange(start_idx, end_idx, desc="reading GA files"):
        a_pop, m_pop, a_hof, m_hof = get_info_file(os.path.join(ga_folder, f"{prefix}_GA_{i}.json"))
        hof.append(a_hof)
        hof_ms.append(m_hof)
        pop.append(a_pop)
        pop_ms.append(m_pop)
    return hof, hof_ms, pop, pop_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="dataset id, e.g. fjsp/sd1_10x5")
    parser.add_argument("--data_root", default="./train_datasets")
    parser.add_argument("--ga_folder", required=True, help="Folder containing the <prefix>_GA_<idx>.json files")
    parser.add_argument("--prefix", default=None, help="JSON file prefix (default: the dataset name)")
    parser.add_argument("--num_instances", type=int, default=500,
                        help="instances 0..N-1 (sources cover the leading instances)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing GA sources")
    args = parser.parse_args()

    prefix = args.prefix or args.dataset.split("/")[-1]
    hof, hof_ms, pop, pop_ms = read_ga_folder(args.ga_folder, prefix, args.num_instances)
    path = dataset_dir(args.data_root, args.dataset)
    hof_sorted = all(list(m) == sorted(m) for m in hof_ms)
    add_source(path, "ga_hof", hof, hof_ms, overwrite=args.overwrite,
               info={"description": "genetic algorithm hall of fame",
                     "order": "makespan_asc" if hof_sorted else "generation"})
    add_source(path, "ga_pop", pop, pop_ms, overwrite=args.overwrite,
               info={"description": "final genetic algorithm population (may hold duplicates)",
                     "order": "generation"})
    print(f"Added ga_hof and ga_pop for {len(hof)} instances to {path}")


if __name__ == "__main__":
    main()
