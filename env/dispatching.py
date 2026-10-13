"""Priority dispatching rules (PDR) for the FJSP environments.

A rule is a job rule (which job's next operation to schedule) combined with a machine rule
(which compatible machine runs it). The offline datasets store the 16 combinations of
PDR_JOB_RULES x PDR_MACHINE_RULES under the names ``<job>_<machine>`` (cdqac/data). Rules
only choose among the pairs the environment's action mask allows.
"""
import numpy as np

PDR_JOB_RULES = ("MWR", "LWR", "LOPNR", "MOPNR")
PDR_MACHINE_RULES = ("EST", "LST", "LPT", "SPT")


def pdr_rule_name(job_rule: str, machine_rule: str) -> str:
    """Dataset name of a rule, e.g. MWR_SPT."""
    return f"{job_rule}_{machine_rule}"


def pdr_rules():
    """(job_rule, machine_rule) of the 16 dataset rules, in their stored order."""
    return [(j, m) for j in PDR_JOB_RULES for m in PDR_MACHINE_RULES]


class FJSPDispatching:
    def __init__(self, job_rule="MWR", operation_rule="SPT", machine_rule="LPT"):
        """
        Dispatching with separate rules for job and machine selection.

        Job rules: MWR / LWR (most / least work remaining), MOPNR / LOPNR (most / least
                   operations remaining), RANDOM
        Machine rules: SPT / LPT (shortest / longest processing time), EST / LST (earliest /
                       latest machine free time), MCT / LCT (minimum / largest completion
                       time), FAM (first available machine), RANDOM
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
        num_machines = dynamic_pair_mask.shape[2]
        valid_pairs = ~(dynamic_pair_mask[env_idx, :, :].numpy())
        # Drawn on every call (also for eps_random=0) so seeded runs reproduce the datasets
        random_prob = np.random.rand()
        if random_prob < eps_random:
            valid_actions = np.where(valid_pairs.flatten())[0]
            if len(valid_actions) == 0:
                raise ValueError("No valid actions available for random selection")
            return np.random.choice(valid_actions)
        if not np.any(valid_pairs):
            raise ValueError("No valid job-machine pairs available")

        # 1. Job Selection
        selected_job = self._select_job(env_state, valid_pairs, env, env_idx=env_idx)

        # 2. Machine Selection
        selected_machine = self._select_machine(env_state, valid_pairs[selected_job], selected_job,
                                                env, env_idx=env_idx)

        # Convert to action format
        action = selected_job * num_machines + selected_machine
        return action

    def _select_job(self, env_state, valid_pairs, env, env_idx=0):
        """Select job based on chosen rule"""
        valid_jobs = np.any(valid_pairs, axis=1)
        job_scores = np.full(valid_pairs.shape[0], -np.inf)

        if self.job_rule == "MWR":
            # Most Work Remaining
            job_scores[valid_jobs] = env.own_job_remain_work[env_idx, valid_jobs]
        elif self.job_rule == "LWR":
            job_scores[valid_jobs] = -env.own_job_remain_work[env_idx, valid_jobs]
        elif self.job_rule == "MWKR":
            raise NotImplementedError("MWKR not implemented yet")
        elif self.job_rule == "LOPNR":
            # Least Operations Remaining
            remain_ops = env.remaining_ops[env_idx]
            job_scores[valid_jobs] = -remain_ops[valid_jobs]
        elif self.job_rule == "MOPNR":
            # Most Operations Remaining
            remain_ops = env.remaining_ops[env_idx]
            job_scores[valid_jobs] = remain_ops[valid_jobs]
        elif self.job_rule == "RANDOM":
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
            machine_scores[valid_indices] = waiting_time[env_idx][valid_indices]
        elif self.machine_rule == "SPT":
            # Shortest Processing Time
            processing_times = env_state.fea_pairs_tensor[env_idx, selected_job, :, 0].numpy()
            machine_scores[valid_indices] = processing_times[valid_indices]
        elif self.machine_rule == "LPT":
            processing_times = env_state.fea_pairs_tensor[env_idx, selected_job, :, 0].numpy()
            machine_scores[valid_indices] = -processing_times[valid_indices]
        elif self.machine_rule == "EST":
            machine_scores[valid_indices] = env.mch_free_time[env_idx, valid_indices]
        elif self.machine_rule == "LST":
            machine_scores[valid_indices] = -env.mch_free_time[env_idx, valid_indices]
        elif self.machine_rule == "MCT":
            job_free = env.candidate_free_time[env_idx, selected_job]
            mch_free = env.mch_free_time[env_idx, valid_indices]
            start_times = np.maximum(job_free, mch_free)
            proc_times = env.true_op_pt[env_idx, env.candidate[env_idx, selected_job]][valid_indices]
            machine_scores[valid_indices] = start_times + proc_times
        elif self.machine_rule == "LCT":
            job_free = env.candidate_free_time[env_idx, selected_job]
            mch_free = env.mch_free_time[env_idx, valid_indices]
            start_times = np.maximum(job_free, mch_free)
            proc_times = env.true_op_pt[env_idx, env.candidate[env_idx, selected_job]][valid_indices]
            machine_scores[valid_indices] = -(start_times + proc_times)
        elif self.machine_rule == "RANDOM":
            return np.random.choice(valid_indices)
        else:
            raise ValueError(f"Invalid machine selection rule {self.machine_rule}")

        return np.argmin(machine_scores)

    def get_rule_description(self):
        """Get current rule configuration"""
        return {
            "job_rule": self.job_rule,
            "operation_rule": self.operation_rule,
            "machine_rule": self.machine_rule
        }


def run_dispatching(job_length, op_pt, job_rule: str, machine_rule: str):
    """Schedule one instance with one rule; returns (action list, makespan)."""
    from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums

    job_length, op_pt = np.asarray(job_length), np.asarray(op_pt)
    env = FJSPEnvForVariousOpNums(job_length.shape[0], op_pt.shape[1], device="cpu",
                                  mask_actions=True)
    env.set_initial_data(job_length[None], op_pt[None])
    dispatcher = FJSPDispatching(job_rule=job_rule, machine_rule=machine_rule)
    actions, done = [], False
    while not done:
        action = dispatcher.select_action(env.state, env.state.dynamic_pair_mask_tensor, env)
        actions.append(int(action))
        _, _, done = env.step(np.array([action]))
    return actions, int(env.current_makespan[0])
