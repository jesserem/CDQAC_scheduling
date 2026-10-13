"""Generate GA trajectories for the instances of a dataset.

This script must be run from inside a clone of the Job Shop Scheduling
Benchmark repository (Reijnen et al., 2023):
    https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances
Copy this file into the root of that repository, then run e.g.:

    python run_GA_custom.py --path <cdqac>/train_datasets/fjsp/sd1_10x5         --output_path <repo>/GA_data/sd1_10x5 --start_index 0 --end_index 500

``--path`` is a dataset directory (its train.npz is read; no cdqac import needed) or a
legacy .npy file. It writes one JSON file per instance ("<name>_GA_<idx>.json", <name> being
the dataset directory or file name) with the hall-of-fame and final-population action lists
and makespans; data_generation/add_ga_data.py adds them to the dataset as the ga_hof and
ga_pop sources.
"""
from data_parsers.custom_instance_parser import parse
from solution_methods.GA.src.initialization import initialize_run
from solution_methods.GA.run_GA import run_GA_get_best as run_GA
import os
import time
import numpy as np
import argparse
import json


def load_data(path):
    """Instances as {"JobLength", "OpPT"} dicts, from a dataset directory or a legacy .npy."""
    print("Loading data from", path)
    if os.path.isdir(path):
        path = os.path.join(path, "train.npz")
    if path.endswith(".npz"):
        with np.load(path) as f:
            job_length, op_pt, n_ops = f["job_length"], f["op_pt"], f["n_ops"]
        return [{"JobLength": job_length[i].astype(int), "OpPT": op_pt[i, :n_ops[i]].astype(int)}
                for i in range(len(n_ops))]
    return np.load(path, allow_pickle=True)


def dataset_name(path):
    """Prefix of the output JSON files: the dataset directory or legacy file name."""
    path = os.path.normpath(path)
    if os.path.isdir(path):
        return os.path.basename(path)
    if os.path.basename(path) == "train.npz":
        return os.path.basename(os.path.dirname(path))
    return os.path.splitext(os.path.basename(path))[0]


def get_processing_data(data):
    jobLength = np.array(data['JobLength'])
    n_j = jobLength.shape[0]
    opPT = np.array(data['OpPT'])
    n_op, n_m = opPT.shape

    p_info = {
        "instance_name": "custom_problem_instance",
        "nr_machines": n_m,
        "jobs": [],
    }
    curr_o_id = 0
    for j in range(n_j):
        curr_job = {}
        curr_job["job_id"] = j
        curr_job["operations"] = []
        precedence = None
        for o_j in range(jobLength[j]):
            machines = opPT[curr_o_id]
            valid_machines = np.where(machines != 0)[0]
            op_dict = {}
            op_dict["operation_id"] = curr_o_id
            op_dict["processing_times"] = {}
            for m in valid_machines:
                op_dict["processing_times"]["machine_" + str(m + 1)] = int(machines[m])
            op_dict["predecessors"] = precedence
            precedence = [curr_o_id]
            curr_job["operations"].append(op_dict)

            curr_o_id += 1
        p_info["jobs"].append(curr_job)
    p_info["sequence_dependent_setup_times"] = {}
    for m in range(n_m):
        p_info["sequence_dependent_setup_times"]["machine_" + str(m + 1)] = np.zeros((n_op, n_op)).tolist()
    return p_info

def run_ga(data, p_info, population_size=200, ngen=100, seed=5, cr=0.7, indpb=0.2):
    parameters = {"instance": {"problem_instance": "custom_problem_instance"},
                  "algorithm": {"population_size": population_size, "ngen": ngen, "seed": seed, "cr": cr,
                                "indpb": indpb,
                                'multiprocessing': True},
                  "output": {"logbook": False}
                  }
    jobShopEnv = parse(p_info)
    jobLength = data['JobLength']
    OpPT = data['OpPT']
    # print(p_info['jobs'])
    # for j in p_info['jobs']:
    #     # print(j)
    #     for o in j['operations']:
    #         print(o)
    #         print(OpPT[o['operation_id']])
    # # print(OpPT)
    # exit()
    population, toolbox, stats, hof = initialize_run(jobLength, OpPT, jobShopEnv, n_hof=100, **parameters)
    makespans_hof, makespans_pop, _, action_lists_hof, action_lists_pop = run_GA(jobLength, OpPT, jobShopEnv, population, toolbox, stats, hof, **parameters)
    return makespans_hof, makespans_pop, action_lists_hof, action_lists_pop
    # print(jobShopEnv)

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours == 0:
        return "{:.0f}m".format(minutes)
    return "{:.0f}h {:.0f}m".format(hours, minutes)


def write_json_file(action_list_hof, makespan_hof, action_list_pop, makespan_pop, runtime, path):
    data = {
        "hof": {
            "action_list": action_list_hof,
            "makespan": makespan_hof
        },
        "pop": {
            "action_list": action_list_pop,
            "makespan": makespan_pop
        },
        "runtime": runtime
    }
    with open(path, 'w') as f:
        json.dump(data, f)


def main(args):
    path = args.path
    data = load_data(path)
    # exit()
    base_name = dataset_name(path)
    # print("Running GA on", base_name)
    # exit()
    runtimes = []

    # makespan_list = []
    total_instances = args.end_index - args.start_index + 1
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    for i in range(args.start_index, args.end_index):
        p_info = get_processing_data(data[i])
        start_time = time.time()
        makespans_hof, makespans_pop, action_lists_hof, action_lists_pop = run_ga(data[i], p_info,
                                                                                  population_size=args.population_size,
                                                                                  ngen=args.ngen, seed=args.seed,
                                                                                  cr=args.cr, indpb=args.indpb)

        curr_runtime = time.time() - start_time
        runtimes.append(curr_runtime)
        mean_makespan_hof = np.mean(makespans_hof)
        mean_makespan_pop = np.mean(makespans_pop)
        best_makepan_hof = np.min(makespans_hof)
        best_makepan_pop = np.min(makespans_pop)
        std_makepan_hof = np.std(makespans_hof)
        std_makepan_pop = np.std(makespans_pop)
        expected_runtime = np.mean(runtimes) * (total_instances - i)
        expected_runtime_str = format_time(expected_runtime)
        print("Instance {}, Expected Remaining Runtime {}".format(i + 1, expected_runtime_str))
        print("\tHof: makespan {:.2f}±{:.2f}, Best {}".format(mean_makespan_hof, std_makepan_hof, best_makepan_hof))
        print("\tPop: makespan {:.2f}±{:.2f}, Best {}".format(mean_makespan_pop, std_makepan_pop, best_makepan_pop))
        # makespan_list.append(makespan)
        write_json_file(action_lists_hof, makespans_hof, action_lists_pop, makespans_pop, curr_runtime,
                        os.path.join(args.output_path, f"{base_name}_GA_{i}.json"))
        # data[i]['ga'] = action_lists_hof
        # data[i]['ga_makespan'] = makespans_hof
        # data[i]['ga_pop'] = action_lists_pop
        # data[i]['ga_pop_makespan'] = makespans_pop
        # print(data[0]["ga"])
        # exit()
    # print("Mean makespan:", np.mean(makespan_list))

    # np.save(path, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the GA on packed FJSP instances and export trajectories.")

    # Adding arguments
    parser.add_argument('--path', type=str, required=True, help="dataset directory or legacy .npy file")
    parser.add_argument('--output_path', type=str, required=True, help="path to output file")
    parser.add_argument('--seed', type=int, default=5, help="Seed for the GA")
    parser.add_argument('--population_size', type=int, default=200, help="Population size for the GA")
    parser.add_argument('--ngen', type=int, default=100, help="Number of generations for the GA")
    parser.add_argument('--cr', type=float, default=0.7, help="Crossover probability for the GA")
    parser.add_argument('--indpb', type=float, default=0.2, help="Mutation probability for the GA")
    parser.add_argument('--start_index', type=int, required=True, help="Start index for the GA")
    parser.add_argument('--end_index', type=int, required=True, help="End index for the GA")

    parser.add_argument('--verbose', action='store_true', help="Enable verbose mode")

    # Parse the arguments
    args = parser.parse_args()
    # exit()

    # Pass arguments to the main function
    main(args)