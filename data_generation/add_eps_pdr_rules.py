from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums as FJSPEnv
import numpy as np
import torch
import os
import tqdm
import argparse

import numpy as np
import os
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from collections import defaultdict
import itertools

class FJSPDispatching:
    def __init__(self, job_rule="MWR", operation_rule="SPT", machine_rule="LPT"):
        """
        Initialize dispatching with separate rules for job, operation, and machine selection.

        Available rules:
        Job rules: MWR (Most Work Remaining), MWKR (Most Work), LOPNR (Least Operations Remaining),
                  MOPNR (Most Operations Remaining), RANDOM
        Operation rules: SPT (Shortest Processing Time), LPT (Longest Processing Time),
                        RANDOM
        Machine rules: FAM (First Available Machine), LUM (Least Utilized Machine),
                      SPT (Shortest Processing Time), RANDOM
        """
        self.job_rule = job_rule
        self.operation_rule = operation_rule
        self.machine_rule = machine_rule

    def set_rules(self, job_rule=None, operation_rule=None, machine_rule=None):
        """Update dispatching rules individually"""
        if job_rule is not None:
            self.job_rule = job_rule
        if operation_rule is not None:
            self.operation_rule = operation_rule
        if machine_rule is not None:
            self.machine_rule = machine_rule

    def select_action(self, env_state, dynamic_pair_mask, env, eps_random=0, env_idx=0):
        """
        Select action using the configured dispatching rules.

        Args:
            env_state: Current environment state
            dynamic_pair_mask: Boolean mask for invalid operation-machine pairs
            env: Environment instance for accessing additional info
            eps_random: Probability of selecting a random valid action (epsilon-greedy)
            env_idx: Index into the batch dimension

        Returns:
            action: Selected action (job_id * num_machines + machine_id)
        """
        num_jobs = dynamic_pair_mask.shape[1]
        num_machines = dynamic_pair_mask.shape[2]
        valid_pairs = ~(dynamic_pair_mask[env_idx, :, :].numpy())
        random_prob = np.random.rand()
        if random_prob < eps_random:
            # print("Selecting random action due to epsilon-greedy strategy")
            valid_actions = np.where(valid_pairs.flatten())[0]
            if len(valid_actions) == 0:
                raise ValueError("No valid actions available for random selection")
            return np.random.choice(valid_actions)
        # print("Selecting action based on dispatching rules")
        if not np.any(valid_pairs):
            raise ValueError("No valid job-machine pairs available")

        # 1. Job Selection
        selected_job = self._select_job(env_state, valid_pairs, env, env_idx=env_idx)

        # 2. Machine Selection
        selected_machine = self._select_machine(env_state, valid_pairs[selected_job], selected_job, env, env_idx=env_idx)

        # Convert to action format
        action = selected_job * num_machines + selected_machine
        return action

    def _select_job(self, env_state, valid_pairs, env, env_idx=0):
        """Select job based on chosen rule"""
        valid_jobs = np.any(valid_pairs, axis=1)
        job_scores = np.full(valid_pairs.shape[0], -np.inf)
        num_dim = env.state.dynamic_pair_mask_tensor[env_idx].shape[1]
        available_jobs = np.where(env.state.dynamic_pair_mask_tensor[env_idx].sum(1) != num_dim)[0]
        #

        available_ops = env.candidate[env_idx][available_jobs]


        if self.job_rule == "MWR":
            # Most Work Remaining
            job_scores[valid_jobs] = env.own_job_remain_work[env_idx, valid_jobs] # Remaining work feature
            # print(job_scores)
        elif self.job_rule == "LWR":
            job_scores[valid_jobs] = -env.own_job_remain_work[env_idx, valid_jobs]  # Remaining work feature

        elif self.job_rule == "MWKR":
            raise NotImplementedError("MWKR not implemented yet")
            # Most Work in current queue
            job_scores[valid_jobs] = env_state.fea_j[env_idx, valid_jobs, 6]  # Current work feature

        elif self.job_rule == "LOPNR":
            remain_ops = env.remaining_ops[env_idx]

            # print(valid_jobs)
            # exit()

            # Least Operations Remaining
            job_scores[valid_jobs] = -remain_ops[valid_jobs] # Operations remaining feature

        elif self.job_rule == "MOPNR":
            # Most Operations Remaining
            remain_ops = env.remaining_ops[env_idx]

            # Least Operations Remaining
            job_scores[valid_jobs] = remain_ops[valid_jobs] # Operations remaining feature

        elif self.job_rule == "RANDOM":
            # Random selection among valid jobs
            valid_indices = np.where(valid_jobs)[0]
            return np.random.choice(valid_indices)

        return np.argmax(job_scores)

    def _select_machine(self, env_state, valid_machines, selected_job, env, env_idx=0):
        """Select machine based on chosen rule"""
        machine_scores = np.full(len(valid_machines), np.inf)
        valid_indices = np.where(valid_machines)[0]

        if not len(valid_indices):
            raise ValueError("No valid machines for selected job")

        if self.machine_rule == "FAM":
            # First Available Machine
            waiting_time = env.mch_waiting_time
            # print(waiting_time)

            machine_scores[valid_indices] = waiting_time[env_idx][valid_indices]  # Machine waiting time

        # elif self.machine_rule == "LUM":
        #     raise ValueError("LUM not implemented yet")
        #     # Least Utilized Machine
        #     utilized = env.mch_remain_work[env_idx]
        #     print(utilized[valid_indices])
        #     # exit()
        #     machine_scores[valid_indices] = utilized[valid_indices]  # Machine workload

        elif self.machine_rule == "SPT":
            # Shortest Processing Time
            processing_times = env_state.fea_pairs_tensor[env_idx, selected_job, :, 0].numpy()
            # print(processing_times)
            # print(len(processing_times[valid_indices]))
            # exit()
            machine_scores[valid_indices] = processing_times[valid_indices]
        elif self.machine_rule == "LPT": #
            processing_times = env_state.fea_pairs_tensor[env_idx, selected_job, :, 0].numpy()
            machine_scores[valid_indices] = -processing_times[valid_indices]
            # print(machine_scores)
        elif self.machine_rule == "EST":
            machine_scores[valid_indices] = env.mch_free_time[env_idx, valid_indices]

        elif self.machine_rule == "LST":
            machine_scores[valid_indices] = -env.mch_free_time[env_idx, valid_indices]
        elif self.machine_rule == "MCT":
            job_free = env.candidate_free_time[env_idx, selected_job]
            mch_free = env.mch_free_time[env_idx, valid_indices]
            start_times = np.maximum(job_free, mch_free)
            proc_times = env.true_op_pt[env_idx, env.candidate[env_idx, selected_job]][valid_indices]
            comp_times = start_times + proc_times
            machine_scores[valid_indices] = comp_times

        elif self.machine_rule == "LCT":
            job_free = env.candidate_free_time[env_idx, selected_job]
            mch_free = env.mch_free_time[env_idx, valid_indices]
            start_times = np.maximum(job_free, mch_free)
            proc_times = env.true_op_pt[env_idx, env.candidate[env_idx, selected_job]][valid_indices]
            comp_times = start_times + proc_times
            machine_scores[valid_indices] = -comp_times
        elif self.machine_rule == "RANDOM":
            # Random selection among valid machines
            return np.random.choice(valid_indices)
        else:
            print(self.machine_rule)
            raise ValueError("Invalid machine selection rule")

        return np.argmin(machine_scores)

    def get_rule_description(self):
        """Get current rule configuration"""
        return {
            "job_rule": self.job_rule,
            "operation_rule": self.operation_rule,
            "machine_rule": self.machine_rule
        }

    def get_available_rules(self):
        """Get list of all available rules"""
        return {
            "job_rules": ["MWR", "MWKR", "LOPNR", "MOPNR", "RANDOM"],
            "operation_rules": ["SPT", "LPT", "RANDOM"],
            "machine_rules": ["FAM", "LPT", "SPT", "RANDOM"]
        }


class FJSPMetrics:
    """Helper class to calculate various scheduling metrics"""

    @staticmethod
    def calculate_machine_utilization(env_state):
        """Calculate current machine utilization"""
        return np.mean(env_state.fea_m[0, :, 7])  # Machine working flag feature

    @staticmethod
    def calculate_work_balance(env_state):
        """Calculate work balance across machines"""
        machine_loads = env_state.fea_m[0, :, 5]  # Machine remaining work
        return np.std(machine_loads)

    @staticmethod
    def calculate_job_progress(env_state):
        """Calculate progress of each job"""
        job_remaining = env_state.fea_j[0, :, 8]  # Remaining work feature
        job_total = env_state.fea_j[0, :, 9]  # Total work feature
        return 1 - (job_remaining / (job_total + 1e-8))


# def run_trajectory(curr_instance, num_random):
#     jobLength = np.array([curr_instance['JobLength']]).repeat(num_random, axis=0)
#     OpPT = np.array([curr_instance['OpPT']]).repeat(num_random, axis=0)
    # n_m = OpPT.shape[-1]
    # n_o = OpPT.shape[1]
    # # print(n_m, n_o)
    # # exit()
    # n_j = jobLength.shape[-1]
#     action_array = np.zeros((n_o, num_random), dtype=int)
#     # print(OpPT.shape)

#     env = FJSPEnv(n_j, n_m, device="cpu", mask_actions=True)
#     state = env.set_initial_data(jobLength, OpPT)
#     done = env.done()
#     i = 0
#     while not done.all():
#         mask = (~state.dynamic_pair_mask_tensor.flatten(1))
#         actions = torch.multinomial(mask.float(), 1)

#         test = mask[0, actions[0].item()].item()
#         test2 = mask[1, actions[1].item()].item()
#         if not test or not test2:
#             print("ERROR")


#         actions = actions.squeeze(-1).cpu().numpy()

#         state, reward, done = env.step(actions)
#         action_array[i] = actions
#         i += 1

#     # print("DONE")
#     curr_makespan = env.current_makespan.astype(int)
#     return action_array, curr_makespan
def run_trajectory(env_data, num_runs, comb_dispatching, eps_random):
    jobLength = np.tile(np.array([env_data['JobLength']]), (num_runs, 1))  # [num_runs, J]
    OpPT = np.tile(np.array([env_data['OpPT']]), (num_runs, 1, 1))        # [num_runs, N, M]
    n_m = OpPT.shape[-1]
    n_j = jobLength.shape[-1]

    env = FJSPEnvForVariousOpNums(n_j, n_m, device="cpu", mask_actions=True)
    env.set_initial_data(jobLength, OpPT)

    dispatchers = []
    for i in range(num_runs):
        curr_rule = comb_dispatching[i % len(comb_dispatching)]
        d = FJSPDispatching()
        d.set_rules(job_rule=curr_rule[0], machine_rule=curr_rule[1])
        dispatchers.append(d)

    action_lists = [[] for _ in range(num_runs)]
    done = env.done_flag.copy()  # [num_runs] bool array

    while not done.all():
        incomplete = np.where(done == 0)[0]
        actions = np.empty(len(incomplete), dtype=int)
        for k, env_idx in enumerate(incomplete):
            action = dispatchers[env_idx].select_action(
                env.state, env.state.dynamic_pair_mask_tensor, env,
                eps_random=eps_random, env_idx=int(env_idx)
            )
            actions[k] = action
            action_lists[env_idx].append(int(action))
        _, _, done = env.step(actions)

    return action_lists
        


def main(seed, num_runs, path_file, eps_random):
    # SEED = 100
    # NUM_RANDON_TRAJECTORIES = 300
    # path_file = './dataset/SD1_train_10_5_1000.npy'
    torch.manual_seed(seed)
    np.random.seed(seed)

    data = np.load(path_file, allow_pickle=True)
    num_instances = len(data)
    job_selection_rules_masked = ["MWR", "LWR", "LOPNR", "MOPNR"]
    # machine_selection_rules_masked = ["FAM", "LPT", "SPT"]
    machine_selection_rules_masked = ["LCT", "MCT", "EST", "LST",  "LPT", "SPT"]
    machine_selection_rules_masked = ["SPT"]

    comb_all_masked = list(itertools.product(job_selection_rules_masked, machine_selection_rules_masked))
    for i in tqdm.tqdm(range(min(num_instances, 500))):
        curr_data = data[i]
        action_array = run_trajectory(curr_data, num_runs, comb_all_masked, eps_random)
        data[i][f"eps_random_{eps_random}"] = action_array
        # data[i]["random_makespan"] = curr_makespan.tolist()
    np.save(path_file, data, allow_pickle=True)
        # get


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_file", type=str, required=True)
    parser.add_argument("--num_runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--eps_random", type=float, default=0.1)
    args = parser.parse_args()

    main(args.seed, args.num_runs, args.path_file, args.eps_random)
