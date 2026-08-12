import torch
import torch.nn as nn
from typing import Optional
from torch.optim.lr_scheduler import CosineAnnealingLR

def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def get_min_q(quantiles):
    # Stack the quantile tensors along a new dimension (dim=2)
    # Each tensor is assumed to have shape [B, ..., D]
    # The stacked tensor will have shape [B, ..., N, D] where N is the number of tensors.
    stacked_quantiles = torch.stack(quantiles, dim=2)

    # Compute the mean along the last dimension (D) for each tensor
    # This yields a tensor of shape [B, ..., N]
    q_means = stacked_quantiles.mean(dim=-1)

    # Find the index with the minimum mean for each element along the batch and other dimensions.
    # The result, arg_q, has shape [B, ..., 1]
    arg_q = torch.argmin(q_means, dim=2, keepdim=True)

    # Gather the quantile tensor corresponding to the minimum mean
    # We need to expand the index tensor to match the shape of the last dimension for gathering.
    min_quantile = torch.gather(
        stacked_quantiles,
        2,
        arg_q.unsqueeze(-1).expand(-1, -1, -1, stacked_quantiles.shape[-1])
    ).squeeze(2)  # Removing the extra dimension added for stacking

    return min_quantile



class IQL:
    def __init__(
            self,
            actor_net: nn.Module,
            q_net: nn.Module,
            target_net: nn.Module,
            value_net: nn.Module,
            value_optimizer: torch.optim.Optimizer,
            actor_optimizer: torch.optim.Optimizer,
            q_optimizer: torch.optim.Optimizer,
            target_update_freq: int,
            update_freq_policy: int,
            discount: float,
            tau: float,
            iql_tau: float = 0.7,
            beta: float = 5.0,
            max_grad_norm: float = 1.0,
            device: str = "cpu",

    ):
        self.actor_net = actor_net
        self.q_net = q_net

        self.target_net = target_net
        self.value_net = value_net

        self.tau = tau
        self.iql_tau = iql_tau
        self.beta = beta

        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.value_optimizer = value_optimizer

        self.target_update_freq = target_update_freq
        self.update_freq_policy = update_freq_policy
        self.max_grad_norm = max_grad_norm
        self.discount = discount

        # self.beta = beta
        # self.iql_tau = iql_tau
        self.device = device
        self.n_updates = 0
        self.n_updates_policy = 0


    def _q_loss(self, state, actions, rewards, dones, next_v, log_dict):
        target = rewards + (1 - dones) * self.discount * next_v.detach()
        q_val, _ = self.q_net(*state)
        q_loss = 0
        for i in range(len(q_val)):
            q_val_i = q_val[i].squeeze(-1).gather(1, actions)
            log_dict["q_val_" + str(i)] = q_val_i.mean().item()
            q_loss_i = torch.nn.functional.smooth_l1_loss(q_val_i, target)
            q_loss += q_loss_i
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.q_optimizer.step()
        log_dict["q_loss"] = q_loss.item()



    # def _policy_loss(self, state, adv, actions, log_dict):
    #     action_probs, _ = self.actor_net(*state)
    #     action_dist = torch.distributions.Categorical(action_probs)
    #     exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100)
    #     bc_loss = -action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
    #     p_loss = (bc_loss * exp_adv).mean()
    #     self.actor_optimizer.zero_grad()
    #     p_loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
    #     self.actor_optimizer.step()
    #     log_dict["policy_loss"] = p_loss.item()
    #     log_dict["bc_loss"] = bc_loss.mean().item()
    #     log_dict["exp_adv"] = exp_adv.mean().item()
    #     entropy = action_dist.entropy().mean()
    #     log_dict["entropy"] = entropy.item()
    def _policy_loss(self, state, adv, actions, log_dict):
        action_probs, _ = self.actor_net(*state)
        action_dist = torch.distributions.Categorical(action_probs)
        # print("action_probs", action_probs.shape)
        # print("adv", adv.shape)
        # adv = torch.nan_to_num(adv, 0)
        # p_loss = torch.sum(action_probs * -adv, dim=-1)

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100)
        bc_loss = -action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        p_loss = (bc_loss * exp_adv).mean()
        # print(p_loss)
        p_loss = torch.mean(p_loss)
        self.actor_optimizer.zero_grad()
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        log_dict["policy_loss"] = p_loss.item()
        # log_dict["bc_loss"] = bc_loss.mean().item()
        # log_dict["exp_adv"] = exp_adv.mean().item()
        entropy = action_dist.entropy().mean()
        log_dict["entropy"] = entropy.item()


    def _value_loss(self, state, actions, log_dict):
        with torch.no_grad():
            q_val, _ = self.target_net(*state)
            q_val = get_min_q(q_val).squeeze(-1)

            q_val_loss = q_val.gather(1, actions)
        val = self.value_net(*state)
        adv = q_val_loss - val


        val_loss = asymmetric_l2_loss(adv, self.iql_tau)
        self.value_optimizer.zero_grad()
        val_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()
        log_dict["value_loss"] = val_loss.item()
        log_dict["adv"] = adv.mean().item()
        return adv




    def train(self, batch):
        self.n_updates += 1
        log_dict = {}
        state, next_state, actions, rewards, dones, mc_returns = batch

        with torch.no_grad():
            next_v = self.value_net(*next_state)

        adv = self._value_loss(state, actions, log_dict)
        self._q_loss(state, actions, rewards, dones, next_v, log_dict)
        self._policy_loss(state, adv.detach(), actions, log_dict)
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()
        return log_dict


    def update_target(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_dict(self):
        return {
            "actor_net": self.actor_net.state_dict(),
        }
