import numpy as np
import os
from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
from env.common_utils import *
from env.data_utils import SD2_instance_generator_no_config, CaseGenerator, create_jsp_instance
from collections import defaultdict
import itertools
from tqdm import tqdm

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

    def select_action(self, env_state, dynamic_pair_mask, env):
        """
        Select action using the configured dispatching rules.

        Args:
            env_state: Current environment state
            dynamic_pair_mask: Boolean mask for invalid operation-machine pairs

        Returns:
            action: Selected action (job_id * num_machines + machine_id)
        """
        num_jobs = dynamic_pair_mask.shape[1]
        num_machines = dynamic_pair_mask.shape[2]
        valid_pairs = ~(dynamic_pair_mask[0, :, :].numpy())

        if not np.any(valid_pairs):
            raise ValueError("No valid job-machine pairs available")

        # 1. Job Selection
        selected_job = self._select_job(env_state, valid_pairs, env)

        # 2. Machine Selection
        selected_machine = self._select_machine(env_state, valid_pairs[selected_job], selected_job, env)

        # Convert to action format
        action = selected_job * num_machines + selected_machine
        return action

    def _select_job(self, env_state, valid_pairs, env):
        """Select job based on chosen rule"""
        valid_jobs = np.any(valid_pairs, axis=1)
        job_scores = np.full(valid_pairs.shape[0], -np.inf)
        num_dim = env.state.dynamic_pair_mask_tensor[0].shape[1]
        available_jobs = np.where(env.state.dynamic_pair_mask_tensor[0].sum(1) != num_dim)[0]
        #

        available_ops = env.candidate[0][available_jobs]


        if self.job_rule == "MWR":
            # Most Work Remaining
            job_scores[valid_jobs] = env.own_job_remain_work[0, valid_jobs] # Remaining work feature
            # print(job_scores)
        elif self.job_rule == "LWR":
            job_scores[valid_jobs] = -env.own_job_remain_work[0, valid_jobs]  # Remaining work feature

        elif self.job_rule == "MWKR":
            raise NotImplementedError("MWKR not implemented yet")
            # Most Work in current queue
            job_scores[valid_jobs] = env_state.fea_j[0, valid_jobs, 6]  # Current work feature

        elif self.job_rule == "LOPNR":
            remain_ops = env.remaining_ops[0]

            # print(valid_jobs)
            # exit()

            # Least Operations Remaining
            job_scores[valid_jobs] = -remain_ops[valid_jobs] # Operations remaining feature

        elif self.job_rule == "MOPNR":
            # Most Operations Remaining
            remain_ops = env.remaining_ops[0]

            # Least Operations Remaining
            job_scores[valid_jobs] = remain_ops[valid_jobs] # Operations remaining feature

        elif self.job_rule == "RANDOM":
            # Random selection among valid jobs
            valid_indices = np.where(valid_jobs)[0]
            return np.random.choice(valid_indices)

        return np.argmax(job_scores)

    def _select_machine(self, env_state, valid_machines, selected_job, env):
        """Select machine based on chosen rule"""
        machine_scores = np.full(len(valid_machines), np.inf)
        valid_indices = np.where(valid_machines)[0]

        if not len(valid_indices):
            raise ValueError("No valid machines for selected job")

        if self.machine_rule == "FAM":
            # First Available Machine
            waiting_time = env.mch_waiting_time
            # print(waiting_time)

            machine_scores[valid_indices] = waiting_time[0][valid_indices]  # Machine waiting time

        # elif self.machine_rule == "LUM":
        #     raise ValueError("LUM not implemented yet")
        #     # Least Utilized Machine
        #     utilized = env.mch_remain_work[0]
        #     print(utilized[valid_indices])
        #     # exit()
        #     machine_scores[valid_indices] = utilized[valid_indices]  # Machine workload

        elif self.machine_rule == "SPT":
            # Shortest Processing Time
            processing_times = env_state.fea_pairs_tensor[0, selected_job, :, 0].numpy()
            # print(processing_times)
            # print(len(processing_times[valid_indices]))
            # exit()
            machine_scores[valid_indices] = processing_times[valid_indices]
        elif self.machine_rule == "LPT": #
            processing_times = env_state.fea_pairs_tensor[0, selected_job, :, 0].numpy()
            machine_scores[valid_indices] = -processing_times[valid_indices]
            # print(machine_scores)
        elif self.machine_rule == "EST":
            machine_scores[valid_indices] = env.mch_free_time[0, valid_indices]

        elif self.machine_rule == "LST":
            machine_scores[valid_indices] = -env.mch_free_time[0, valid_indices]
        elif self.machine_rule == "MCT":
            job_free = env.candidate_free_time[0, selected_job]
            mch_free = env.mch_free_time[0, valid_indices]
            start_times = np.maximum(job_free, mch_free)
            proc_times = env.true_op_pt[0, env.candidate[0, selected_job]][valid_indices]
            comp_times = start_times + proc_times
            machine_scores[valid_indices] = comp_times

        elif self.machine_rule == "LCT":
            job_free = env.candidate_free_time[0, selected_job]
            mch_free = env.mch_free_time[0, valid_indices]
            start_times = np.maximum(job_free, mch_free)
            proc_times = env.true_op_pt[0, env.candidate[0, selected_job]][valid_indices]
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


def main(seed=101):
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
    n_j = 10
    n_m = 10
    low = 1
    high = 99
    op_per_job_min = int(0.8 * n_m)
    op_per_job_max = int(1.2 * n_m)
    op_per_machine_min = 1
    op_per_machine_max = 1
    checkpoints = [500]
    num_runs = max(checkpoints)
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
        JobLength, OpPT = create_jsp_instance(n_j, n_m, low, high)
        curr_instance['JobLength'] = JobLength.tolist()

        curr_instance['OpPT'] = OpPT.tolist()
        curr_instance["rules"] = {}
        curr_instance["rules_info"] = defaultdict(dict)
        # curr_instance["info"] = info.tolist()
        data_list.append(curr_instance)

    dispatcher = FJSPDispatching()
    for curr_instance in tqdm(data_list, desc="Running Dispatching Rules"):
        jobLength = np.array([curr_instance['JobLength']])
        OpPT = np.array([curr_instance['OpPT']])
        env = FJSPEnvForVariousOpNums(n_j, n_m, device="cpu", mask_actions=True)
        for job_select, machine_select in comb_all_masked:
            dispatcher.set_rules(job_rule=job_select, machine_rule=machine_select)
            env.set_initial_data(jobLength, OpPT)
            done = False
            action_list = []
            while not done:
                action = dispatcher.select_action(env.state, env.state.dynamic_pair_mask_tensor, env)
                action_list.append(int(action))
                _, _, done = env.step(np.array([action]))
            name = f"{job_select}_{machine_select}_masked"
            # print(f"{name} got a makespan of {env.current_makespan[0]}")
            curr_instance["rules"][name] = action_list
            curr_instance["rules_info"][name]["makespan"] = int(env.current_makespan[0])
            curr_instance["rules_info"][name]["masked"] = True
            makespan_masked.append(env.current_makespan[0])
        # env = FJSPEnvForVariousOpNums(n_j, n_m, device="cpu", mask_actions=False)
        # for job_select, machine_select in comb_all_unmasked:
        #     dispatcher.set_rules(job_rule=job_select, machine_rule=machine_select)
        #     env.set_initial_data(jobLength, OpPT)
        #     done = False
        #     action_list = []
        #     while not done:
        #         action = dispatcher.select_action(env.state, env.state.dynamic_pair_mask_tensor, env)
        #         action_list.append(int(action))
        #         _, _, done = env.step(np.array([action]))
        #     name = f"{job_select}_{machine_select}_unmasked"
        #     # print(f"{name} got a makespan of {env.current_makespan[0]}")
        #     curr_instance["rules"][name] = action_list
        #     curr_instance["rules_info"][name]["makespan"] = int(env.current_makespan[0])
        #     curr_instance["rules_info"][name]["masked"] = False
        #
        #     makespan_unmasked.append(env.current_makespan[0])
        # print(curr_instance["rules_info"])
            # print(env.current_makespan)

    # exit()
    name_dataset = f'train_{n_j}_{n_m}_{num_runs}.npy'

    setup_seed(seed + 1000)
    np.save(os.path.join(dataset_folder, name_dataset), data_list)
    eval_instances = []
    for i in range(100):
        setup_seed(seed + 1000 + i)
        curr_instance = {}
        JobLength, OpPT = create_jsp_instance(n_j, n_m, low, high)
        curr_instance['JobLength'] = JobLength.tolist()

        curr_instance['OpPT'] = OpPT.tolist()
        eval_instances.append(curr_instance)
    name = f'eval_{n_j}_{n_m}.npy'
    np.save(os.path.join(dataset_folder, name), eval_instances)


main()




