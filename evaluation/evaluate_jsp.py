import os.path
import os
import torch
from torch.backends.cudnn import benchmark

from cdqac.network.main_model import ActorNet, mQRDQNNet
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from cdqac.utils import open_path, load_yml, load_data_from_files, eval_model_complex, eval_model_complex_no_gap, load_data_from_files_jsp
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import pyrallis
from dataclasses import asdict, dataclass
import pandas as pd

@dataclass
class TrainConfig:
    # JSSP Environment

    device: str = "cuda"

    eval_instance_path: str = "./data/BenchData/Brandimarte"
    benchmark_result_path: str = "./data/BenchData/BenchDataSolution.csv"
    model_path: str = "./checkpoints/<run_folder>"
    name_file: str = "hk"
    datapath: Optional[str] = "./data"
    save_results_path: str = "./results"
    checkpoint_num = 24999


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)



def eval(eval_instance_path, benchmark_result_path, model_path, name_file, save_results_path, device, use_final=False,
         num_runs=100):

    benchmark_df = pd.read_csv(benchmark_result_path)
    # print("test")
    # return

    # set_seed(config.seed)
    # config.device = torch.device(config.device)
    eval_func = eval_model_complex
    model_path = open_path(model_path)
    eval_instance_path = open_path(eval_instance_path)
    benchmark_result_path = open_path(benchmark_result_path)
    save_results_path = open_path(save_results_path)
    model_config = load_yml(os.path.join(model_path, "config.yml"))

    if use_final:
        save_path = os.path.join(save_results_path, f"{name_file}_{num_runs}_final.csv")
    else:
        save_path = os.path.join(save_results_path, f"{name_file}_{num_runs}.csv")

    if os.path.exists(save_path):
        print("Already evaluated: ", save_path)
        return


    # print("Mask,")
    if "update_freq_policy" in model_config.keys() or "beta" in model_config.keys():
        net = ActorNet(
            fea_j_input_dim=model_config["fea_j_input_dim"],
            fea_m_input_dim=model_config["fea_m_input_dim"],
            layer_fea_output_dim=model_config["layer_fea_output_dim"],
            num_heads_OAB=model_config["num_heads_OAB"],
            num_heads_MAB=model_config["num_heads_MAB"],
            num_mlp_layers_actor=model_config["num_mlp_layers_actor"],
            hidden_dim_actor=model_config["hidden_dim_actor"],
            dropout_prob=model_config["dropout_prob_actor"],
        ).to(device)
    else:
        net = mQRDQNNet(
            fea_j_input_dim=model_config["fea_j_input_dim"],
            fea_m_input_dim=model_config["fea_m_input_dim"],
            layer_fea_output_dim=model_config["layer_fea_output_dim"],
            num_heads_OAB=model_config["num_heads_OAB"],
            num_heads_MAB=model_config["num_heads_MAB"],
            num_mlp_layers_critic=model_config["num_mlp_layers_actor"],
            use_adv_net=model_config["use_adv_net"],
            hidden_dim_critic=model_config["hidden_dim_actor"],
            dropout_prob=model_config["dropout_prob_actor"],
            num_quantiles=model_config["num_quantiles"],

        ).to(device)
    if use_final:
        # model_weights_offline = torch.load(os.path.join(model_path, f"model_final_offline.pt"), weights_only=True)
        model_weights_offline = torch.load(os.path.join(model_path, f"model_step_200000.pt"), weights_only=True)
    else:
        model_weights_offline = torch.load(os.path.join(model_path, f"best_det_offline.pt"), weights_only=True)



    if "mean_fea_j" not in model_weights_offline:
        mean_fea_j = torch.tensor(0).to(device)
        std_fea_j = torch.tensor(1).to(device)
        mean_fea_m = torch.tensor(0).to(device)
        std_fea_m = torch.tensor(1).to(device)
        mean_fea_pairs = torch.tensor(0).to(device)
        std_fea_pairs = torch.tensor(1).to(device)
    else:
        mean_fea_j = model_weights_offline["mean_fea_j"].to(device)
        std_fea_j = model_weights_offline["std_fea_j"].to(device)
        mean_fea_m = model_weights_offline["mean_fea_m"].to(device)
        std_fea_m = model_weights_offline["std_fea_m"].to(device)
        mean_fea_pairs = model_weights_offline["mean_fea_pairs"].to(device)
        std_fea_pairs = model_weights_offline["std_fea_pairs"].to(device)

    net.load_state_dict(model_weights_offline["actor_net"])
    name, jobLength, opt = load_data_from_files_jsp(eval_instance_path)
    env_func = FJSPEnvForVariousOpNums


    res_df_deterministic_offline = eval_func(net, name, jobLength, opt, env_func, benchmark_df, device=device,
                                              deterministic=True, num_runs=1, use_mask=model_config["use_mask"],
                                             mean_fea_j=mean_fea_j,
                                             std_fea_j=std_fea_j, mean_fea_m=mean_fea_m, std_fea_m=std_fea_m,
                                             mean_fea_pairs=mean_fea_pairs, std_fea_pairs=std_fea_pairs
                                             )
    # print("Mean Gap Deterministic: ", res_df_deterministic_offline["gap"].mean())
    print("\nMean Makespan Deterministic: ", res_df_deterministic_offline["makespan"].mean())
    print("Mean Gap Deterministic: ", res_df_deterministic_offline["gap"].mean())
    # if print_gap:
    #     print("Mean Gap Deterministic: ", res_df_deterministic_offline["gap"].mean())
    # print("\n")
    # exit()
    res_df_deterministic_offline.rename(columns={
        'makespan': 'makespan_det_offline',
        'runtime': 'runtime_det_offline',
        'gap': "gap_det_offline"}, inplace=True)


    res_df_stochastic_offline = eval_func(net, name, jobLength, opt, env_func, benchmark_df, device=device,
                                                   use_mask=model_config["use_mask"], num_runs=num_runs,
                                          mean_fea_j=mean_fea_j,
                                          std_fea_j=std_fea_j, mean_fea_m=mean_fea_m, std_fea_m=std_fea_m,
                                          mean_fea_pairs=mean_fea_pairs, std_fea_pairs=std_fea_pairs,
                                          num_stoch_runs=1
                                          )
    print("\nMean Makespan Samp: ", res_df_stochastic_offline["makespan"].mean())
    print("Mean Gap Samp: ", res_df_stochastic_offline["gap"].mean())
    # if print_gap:
    #     print("Mean Gap Samp: ", res_df_stochastic_offline["gap"].mean())
    # print("\n")
    # exit()
    res_df_stochastic_offline.rename(columns={
        'makespan': 'makespan_stoch_offline',
        'runtime': 'runtime_stoch_offline',
        'gap': "gap_stoch_offline"}, inplace=True)


    dfs = [res_df_deterministic_offline, res_df_stochastic_offline]
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="name", how='inner')
    if not os.path.exists(save_results_path):
        os.makedirs(save_results_path)
    if use_final:
        merged_df.to_csv(os.path.join(save_results_path, f"{name_file}_{num_runs}_final.csv"))
    else:
        merged_df.to_csv(os.path.join(save_results_path, f"{name_file}_{num_runs}.csv"))


# def do_folder(path, exps):
#     print("Doing folder: ", path)
#     for data_folder, bench_csv_path, name_exp in exps:
#
#         eval(data_folder, bench_csv_path, path, name_exp, path, "cuda")
#         # exit()
#
#
# def do_all_folder(path, exps):
#     list_dir = os.listdir(path)
#     for dir in list_dir:
#         do_folder(os.path.join(path, dir), exps)
#

@pyrallis.wrap()
def eval_all(config: TrainConfig):
    model_config = load_yml(os.path.join(config.model_path, "config.yml"))
    train_instance = model_config["train_instance"].split("_")

    print("Train instance: ", train_instance)
    type_run = "{}x{}".format(train_instance[2], train_instance[3])
    # print("Type run: ", type_run)
    exps = [


        (os.path.join(config.datapath, 'JSP/DMU'),
         os.path.join(config.datapath, 'JSP/benchmark_results.csv'), 'DMU', False),
        (os.path.join(config.datapath, 'JSP/LA'),
         os.path.join(config.datapath, 'JSP/benchmark_results.csv'), 'LA', False),
        (os.path.join(config.datapath, 'JSP/TA'),
         os.path.join(config.datapath, 'JSP/benchmark_results.csv'), 'TA', False),
    ]
    # print(model_config)

    # exps_sd2 = [
    #     ('.\\data\\SD2\\10x5+mix', '.\\data\\gen_data.csv', 'SD2'),
    #     ('.\\data\\BenchData\\Brandimarte', '.\\data\\BenchData\\BenchDataSolution.csv', 'hk'),
    #     ('.\\data\\BenchData\\Hurink_edata', '.\\data\\BenchData\\BenchDataSolution.csv', 'edata'),
    #     ('.\\data\\BenchData\\Hurink_rdata', '.\\data\\BenchData\\BenchDataSolution.csv', 'rdata'),
    #     ('.\\data\\BenchData\\Hurink_vdata', '.\\data\\BenchData\\BenchDataSolution.csv', 'vdata'),
    #
    # ]

    for data_folder, bench_csv_path, name_exp, use_final in exps:
        print(data_folder, bench_csv_path, config.model_path, name_exp, config.model_path, config.device)
        eval(data_folder, bench_csv_path, config.model_path, name_exp, config.model_path, config.device, use_final=False)
        # eval(data_folder, bench_csv_path, config.model_path, name_exp, config.model_path, config.device, use_final=True)


    # do_all_folder(main_folder1, exps_sd1)


# @pyrallis.wrap()
# def eval(config: TrainConfig):
#     if "BenchData" in config.eval_instance_path:
#         eval_func = eval_model_complex
#         benchmark_df = pd.read_csv(config.benchmark_result_path)
#         # print_gap = True
#     else:
#         eval_func = eval_model_complex
#         benchmark_df = pd.read_csv("./data/gen_data.csv")
#
#     # set_seed(config.seed)
#     # config.device = torch.device(config.device)
#     config.model_path = open_path(config.model_path)
#     config.eval_instance_path = open_path(config.eval_instance_path)
#     config.benchmark_result_path = open_path(config.benchmark_result_path)
#     config.save_results_path = open_path(config.save_results_path)
#     model_config = load_yml(os.path.join(config.model_path, "config.yml"))
#     # print("Mask,")
#     net = ActorNet(
#         fea_j_input_dim=model_config["fea_j_input_dim"],
#         fea_m_input_dim=model_config["fea_m_input_dim"],
#         layer_fea_output_dim=model_config["layer_fea_output_dim"],
#         num_heads_OAB=model_config["num_heads_OAB"],
#         num_heads_MAB=model_config["num_heads_MAB"],
#         num_mlp_layers_actor=model_config["num_mlp_layers_actor"],
#         hidden_dim_actor=model_config["hidden_dim_actor"],
#         dropout_prob=model_config["dropout_prob_actor"],
#     ).to(config.device)
#     model_weights_offline = torch.load(os.path.join(config.model_path, f"best_det_offline.pt"), weights_only=True)
#
#     net.load_state_dict(model_weights_offline["actor_net"])
#     name, jobLength, opt = load_data_from_files(config.eval_instance_path)
#     env_func = FJSPEnvForVariousOpNums
#
#
#     res_df_deterministic_offline = eval_func(net, name, jobLength, opt, env_func, benchmark_df, device=config.device,
#                                               deterministic=True, num_runs=1, use_mask=model_config["use_mask"])
#     # print("Mean Gap Deterministic: ", res_df_deterministic_offline["gap"].mean())
#     print("\nMean Makespan Deterministic: ", res_df_deterministic_offline["makespan"].mean())
#     print("Mean Gap Deterministic: ", res_df_deterministic_offline["gap"].mean())
#     # if print_gap:
#     #     print("Mean Gap Deterministic: ", res_df_deterministic_offline["gap"].mean())
#     # print("\n")
#     # exit()
#     res_df_deterministic_offline.rename(columns={
#         'makespan': 'makespan_det_offline',
#         'runtime': 'runtime_det_offline',
#         'gap': "gap_det_offline"}, inplace=True)
#
#
#     res_df_stochastic_offline = eval_func(net, name, jobLength, opt, env_func, benchmark_df, device=config.device,
#                                                    use_mask=model_config["use_mask"])
#     print("\nMean Makespan Samp: ", res_df_stochastic_offline["makespan"].mean())
#     print("Mean Gap Samp: ", res_df_stochastic_offline["gap"].mean())
#     # if print_gap:
#     #     print("Mean Gap Samp: ", res_df_stochastic_offline["gap"].mean())
#     # print("\n")
#     # exit()
#     res_df_stochastic_offline.rename(columns={
#         'makespan': 'makespan_stoch_offline',
#         'runtime': 'runtime_stoch_offline',
#         'gap': "gap_stoch_offline"}, inplace=True)
#
#
#     dfs = [res_df_deterministic_offline, res_df_stochastic_offline]
#     merged_df = dfs[0]
#     for df in dfs[1:]:
#         merged_df = pd.merge(merged_df, df, on="name", how='inner')
#     if not os.path.exists(config.save_results_path):
#         os.makedirs(config.save_results_path)
#     merged_df.to_csv(os.path.join(config.save_results_path, f"{config.name_file}.csv"))


eval_all()