import numpy as np
import torch

import tqdm


class Buffer:
    def __init__(self, size: int, n_j: int, n_m: int, n_op: int, f_j: int = 10, f_m: int = 8, device: str='cpu'):
        self.float_type = torch.float32
        self.size = size
        self.ptr = 0
        self.curr_size = 0
        self.device = device
        self.n_j = n_j
        self.n_m = n_m
        self.n_op = n_op
        self.f_j = f_j
        self.f_m = f_m
        self.fea_j = torch.zeros((size, self.n_op, self.f_j), device=self.device, dtype=self.float_type)
        self.op_mask = torch.zeros((size, self.n_op, 3), device=self.device, dtype=self.float_type)
        self.fea_m = torch.zeros((size, self.n_m, self.f_m), device=self.device, dtype=self.float_type)
        self.mch_mask = torch.zeros((size, self.n_m, self.n_m), device=self.device, dtype=self.float_type)
        self.dynamic_pair_mask = torch.zeros((size, self.n_j, self.n_m), device=self.device, dtype=torch.bool)
        self.comp_idx = torch.zeros((size, self.n_m, self.n_m, self.n_j), device=self.device, dtype=self.float_type)
        self.candidate = torch.zeros((size, self.n_j), device=self.device, dtype=torch.long)
        self.fea_pairs = torch.zeros((size, self.n_j, self.n_m, self.f_m), device=self.device, dtype=self.float_type)

        self.next_idx = torch.zeros(size, device=self.device, dtype=torch.long)


        self.reward = torch.zeros((size, 1), device=self.device, dtype=self.float_type)
        self.mc_return = torch.zeros((size, 1), device=self.device, dtype=self.float_type)
        self.done = torch.zeros((size, 1), device=self.device, dtype=self.float_type)
        self.action = torch.zeros((size, 1), device=self.device, dtype=torch.long)

    def validate_action(self, job_idx, machine_idx, dynamic_pair_mask):
        """Validate if the action respects the dynamic pair mask"""

        return ~dynamic_pair_mask[:, job_idx, machine_idx]

    def __len__(self):
        return self.curr_size

    def set_mc_return(self, mc_return: np.ndarray, start_idx: int, end_idx: int):
        self.mc_return[start_idx:end_idx] = torch.tensor(mc_return, device=self.device, dtype=torch.float32).unsqueeze(-1)


    def old_normalize_state(self):
        mean_fea_j = self.fea_j[:self.ptr].mean(dim=(0,1), keepdim=True)
        std_fea_j = self.fea_j[:self.ptr].std(dim=(0,1), keepdim=True)
        mean_fea_m = self.fea_m[:self.ptr].mean(dim=(0,1), keepdim=True)
        std_fea_m = self.fea_m[:self.ptr].std(dim=(0,1), keepdim=True)
        mean_fea_pairs = self.fea_pairs[:self.ptr].mean(dim=(0,1,2), keepdim=True)
        std_fea_pairs = self.fea_pairs[:self.ptr].std(dim=(0,1,2), keepdim=True)
        #
        # self.fea_m = (self.fea_m - mean_fea_m) / (std_fea_m + 1e-8)
        # self.fea_j = (self.fea_j - mean_fea_j) / (std_fea_j + 1e-8)
        # self.fea_pairs = (self.fea_pairs - mean_fea_pairs) / (std_fea_pairs + 1e-8)

        return mean_fea_j, mean_fea_m, std_fea_j, std_fea_m, mean_fea_pairs, std_fea_pairs

    def normalize_state(self):
        """
        Normalise the **filled part** of the buffer in‑place and without tracking
        autograd.  This keeps peak memory ≈ the size of the buffer itself.
        """
        with torch.no_grad():  # no grad graph, saves memory
            filled = slice(0, self.ptr)  # only valid transitions

            # 1. statistics (small tensors, negligible memory)
            mean_fj = self.fea_j[filled].mean(dim=(0, 1), keepdim=True)
            std_fj = self.fea_j[filled].std(dim=(0, 1), keepdim=True).clamp_min_(1e-8)

            mean_fm = self.fea_m[filled].mean(dim=(0, 1), keepdim=True)
            std_fm = self.fea_m[filled].std(dim=(0, 1), keepdim=True).clamp_min_(1e-8)

            mean_fp = self.fea_pairs[filled].mean(dim=(0, 1, 2), keepdim=True)
            std_fp = self.fea_pairs[filled].std(dim=(0, 1, 2), keepdim=True).clamp_min_(1e-8)

            # 2. **in‑place** whitening on the same storage – no second copy created
            self.fea_j[filled].sub_(mean_fj).div_(std_fj)
            self.fea_m[filled].sub_(mean_fm).div_(std_fm)
            self.fea_pairs[filled].sub_(mean_fp).div_(std_fp)

        # return stats for rescaling, if you need them later
        return mean_fj, mean_fm, std_fj, std_fm, mean_fp, std_fp

    # def push(self, state, next_state, action, reward, done):
    #     job_idx, machine_idx = action // self.n_m, action % self.n_m
    #     n_envs = state.fea_j_tensor.shape[0]
    #     s_i = self.ptr
    #     e_i = self.ptr + n_envs
    #     indices = torch.arange(s_i, e_i, device=self.device, dtype=torch.long) % self.size
    #     indices = indices.to(self.device)
    #     # print(indices)
    #     # exit()
    #     self.curr_size = min(self.size, self.curr_size + n_envs)
    #
    #     # self.fea_j[indices] = state.fea_j_tensor.clone().to(self.float_type).to(self.device)
    #     print(self.fea_j[0], state.fea_j_tensor)
    #     self.fea_j[indices].copy_(state.fea_j_tensor.to(self.device, dtype=self.float_type, non_blocking=True))
    #     # print()
    #
    #     # self.op_mask[indices] = state.op_mask_tensor.clone().to(self.float_type).to(self.device)
    #
    #     self.op_mask[indices].copy_(state.op_mask_tensor.to(self.device, dtype=self.float_type, non_blocking=True))
    #
    #     # self.fea_m[indices] = state.fea_m_tensor.clone().to(self.float_type).to(self.device)
    #     self.fea_m[indices].copy_(state.fea_m_tensor.to(self.device, dtype=self.float_type, non_blocking=True))
    #     # self.mch_mask[indices] = state.mch_mask_tensor.clone().to(self.float_type).to(self.device)
    #     self.mch_mask[indices].copy_(state.mch_mask_tensor.to(self.device, dtype=self.float_type, non_blocking=True))
    #     # self.dynamic_pair_mask[indices] = state.dynamic_pair_mask_tensor.clone().to(self.device)
    #     self.dynamic_pair_mask[indices].copy_(state.dynamic_pair_mask_tensor.to(self.device, dtype=torch.bool, non_blocking=True))
    #     # print(self.dynamic_pair_mask[self.ptr].shape)
    #     # print(state.dynamic_pair_mask_tensor.shape)
    #     # self.comp_idx[indices] = state.comp_idx_tensor.clone().to(self.float_type).to(self.device)
    #     self.comp_idx[indices].copy_(state.comp_idx_tensor.to(self.device, dtype=self.float_type, non_blocking=True))
    #     # print(self.candidate[indices].shape, state.candidate_tensor.shape)
    #     # self.candidate[indices] = state.candidate_tensor.clone().to(self.device)
    #     self.candidate[indices].copy_(state.candidate_tensor.to(self.device, dtype=torch.long, non_blocking=True))
    #
    #     # self.fea_pairs[indices] = state.fea_pairs_tensor.clone().to(self.float_type).to(self.device)
    #     self.fea_pairs[indices].copy_(state.fea_pairs_tensor.to(self.device, dtype=self.float_type, non_blocking=True))
    #
    #
    #     self.reward[indices] = torch.tensor(reward, device=self.device, dtype=self.float_type).unsqueeze(-1)
    #     self.done[indices] = torch.tensor(done, device=self.device, dtype=self.float_type).unsqueeze(-1)
    #     # print(action)
    #     # print(self.action[s_i:e_i])
    #     self.action[indices] = torch.tensor(action, device=self.device, dtype=torch.long).unsqueeze(-1)
    #
    #
    #     self.ptr = indices[-1].item() + 1
    #
    #     self.next_idx[indices] = indices + indices.shape[0]


    def push(self, state, next_state, action, reward, done):
        job_idx, machine_idx = action // self.n_m, action % self.n_m
        n_envs = state.fea_j_tensor.shape[0]
        s_i = self.ptr
        e_i = self.ptr + n_envs
        indices = torch.arange(s_i, e_i, device=self.device, dtype=torch.long) % self.size
        indices = indices.to(self.device)
        # print(indices)
        # exit()
        self.curr_size = min(self.size, self.curr_size + n_envs)

        self.fea_j[indices] = state.fea_j_tensor.clone().to(self.float_type).to(self.device)
        # print()

        self.op_mask[indices] = state.op_mask_tensor.clone().to(self.float_type).to(self.device)
        self.fea_m[indices] = state.fea_m_tensor.clone().to(self.float_type).to(self.device)
        self.mch_mask[indices] = state.mch_mask_tensor.clone().to(self.float_type).to(self.device)
        self.dynamic_pair_mask[indices] = state.dynamic_pair_mask_tensor.clone().to(self.device)
        # print(self.dynamic_pair_mask[self.ptr].shape)
        # print(state.dynamic_pair_mask_tensor.shape)
        self.comp_idx[indices] = state.comp_idx_tensor.clone().to(self.float_type).to(self.device)
        # print(self.candidate[indices].shape, state.candidate_tensor.shape)
        self.candidate[indices] = state.candidate_tensor.clone().to(self.device)
        self.fea_pairs[indices] = state.fea_pairs_tensor.clone().to(self.float_type).to(self.device)


        self.reward[indices] = torch.tensor(reward, device=self.device, dtype=self.float_type).unsqueeze(-1)
        self.done[indices] = torch.tensor(done, device=self.device, dtype=self.float_type).unsqueeze(-1)
        # print(action)
        # print(self.action[s_i:e_i])
        self.action[indices] = torch.tensor(action, device=self.device, dtype=torch.long).unsqueeze(-1)


        self.ptr = indices[-1].item() + 1

        self.next_idx[indices] = indices + indices.shape[0]


    def _n_step_helper(self, curr_idx: int, total_reward: torch.Tensor, remaining_n_step: int, steps_done: int, gamma: float):
        # print("being called")
        steps_done += 1
        remaining_n_step -= 1
        total_reward += (self.reward[curr_idx] * (gamma ** steps_done))

        # print(total_reward, self.reward[curr_idx])
        if self.done[curr_idx] or remaining_n_step == 0:
            return curr_idx, total_reward
        else:
            next_idx = self.next_idx[curr_idx].item()
            return self._n_step_helper(next_idx, total_reward, remaining_n_step, steps_done, gamma)


    def n_step_buffer(self, n_step: int, gamma: float):
        for i in tqdm.tqdm(range(self.ptr)):
            curr_reward = self.reward[i]
            curr_done = self.done[i]
            if curr_done:
                continue
            new_idx, new_reward = self._n_step_helper(self.next_idx[i].item(), curr_reward, n_step, 0, gamma)
            self.reward[i] = new_reward
            self.next_idx[i] = new_idx


    def epoch_generator(self, batch_size: int, device: str):
        batches = torch.randperm(self.ptr).split(batch_size)
        for batch_idx in batches:
            state = (
            self.fea_j[batch_idx].to(device), self.op_mask[batch_idx].to(device), self.candidate[batch_idx].to(device),
            self.fea_m[batch_idx].to(device),
            self.mch_mask[batch_idx].to(device), self.comp_idx[batch_idx].to(device),
            self.dynamic_pair_mask[batch_idx].to(device),
            self.fea_pairs[batch_idx].to(device))
            next_idx = self.next_idx[batch_idx].clamp(max=(self.curr_size - 1))

            next_state = (
                self.fea_j[next_idx].to(device),
                self.op_mask[next_idx].to(device),
                self.candidate[next_idx].to(device),
                self.fea_m[next_idx].to(device),
                self.mch_mask[next_idx].to(device),
                self.comp_idx[next_idx].to(device),
                self.dynamic_pair_mask[next_idx].to(device),
                self.fea_pairs[next_idx].to(device)
            )
            rewards = self.reward[batch_idx].to(device)
            dones = self.done[batch_idx].to(device)
            actions = self.action[batch_idx].to(device)
            mc_returns = self.mc_return[batch_idx].to(device)

            yield state, next_state, actions, rewards, dones, mc_returns




    def sample(self, batch_size: int, device: str = 'cuda'):
        batch_idx = torch.randint(0, self.curr_size, (batch_size,))
        # print(self.fea_j[batch_idx].shape)
        state = (self.fea_j[batch_idx].to(device), self.op_mask[batch_idx].to(device), self.candidate[batch_idx].to(device), self.fea_m[batch_idx].to(device),
                 self.mch_mask[batch_idx].to(device), self.comp_idx[batch_idx].to(device), self.dynamic_pair_mask[batch_idx].to(device),
                 self.fea_pairs[batch_idx].to(device))
        next_idx = self.next_idx[batch_idx].clamp(max=(self.curr_size-1))

        next_state = (
            self.fea_j[next_idx].to(device),
            self.op_mask[next_idx].to(device),
            self.candidate[next_idx].to(device),
            self.fea_m[next_idx].to(device),
            self.mch_mask[next_idx].to(device),
            self.comp_idx[next_idx].to(device),
            self.dynamic_pair_mask[next_idx].to(device),
            self.fea_pairs[next_idx].to(device)
        )
        rewards = self.reward[batch_idx].to(device)
        dones = self.done[batch_idx].to(device)
        actions = self.action[batch_idx].to(device)
        mc_returns = self.mc_return[batch_idx].to(device)

        return state, next_state, actions, rewards, dones, mc_returns

    def sample_td3(self, batch_size: int, device: str = 'cuda'):
        """Like sample(), but additionally returns the dataset action taken at the
        next state (needed for TD3-AWR's critic-side BC penalty). On terminal
        transitions the next action is arbitrary -- the target is gated by done."""
        state, next_state, actions, rewards, dones, mc_returns, next_actions = \
            self._sample_with_next_action(torch.randint(0, self.curr_size, (batch_size,)), device)
        return state, next_state, actions, next_actions, rewards, dones, mc_returns

    def epoch_generator_td3(self, batch_size: int, device: str = 'cuda'):
        batches = torch.randperm(self.ptr).split(batch_size)
        for batch_idx in batches:
            state, next_state, actions, rewards, dones, mc_returns, next_actions = \
                self._sample_with_next_action(batch_idx, device)
            yield state, next_state, actions, next_actions, rewards, dones, mc_returns

    def _sample_with_next_action(self, batch_idx: torch.Tensor, device: str):
        state = (self.fea_j[batch_idx].to(device), self.op_mask[batch_idx].to(device),
                 self.candidate[batch_idx].to(device), self.fea_m[batch_idx].to(device),
                 self.mch_mask[batch_idx].to(device), self.comp_idx[batch_idx].to(device),
                 self.dynamic_pair_mask[batch_idx].to(device),
                 self.fea_pairs[batch_idx].to(device))
        next_idx = self.next_idx[batch_idx].clamp(max=(self.curr_size - 1))

        next_state = (
            self.fea_j[next_idx].to(device),
            self.op_mask[next_idx].to(device),
            self.candidate[next_idx].to(device),
            self.fea_m[next_idx].to(device),
            self.mch_mask[next_idx].to(device),
            self.comp_idx[next_idx].to(device),
            self.dynamic_pair_mask[next_idx].to(device),
            self.fea_pairs[next_idx].to(device)
        )
        rewards = self.reward[batch_idx].to(device)
        dones = self.done[batch_idx].to(device)
        actions = self.action[batch_idx].to(device)
        mc_returns = self.mc_return[batch_idx].to(device)
        next_actions = self.action[next_idx].to(device)

        return state, next_state, actions, rewards, dones, mc_returns, next_actions


