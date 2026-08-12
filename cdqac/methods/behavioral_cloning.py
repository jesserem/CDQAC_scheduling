import torch
import torch.nn as nn
from typing import Optional
from torch.optim.lr_scheduler import CosineAnnealingLR


class Scalar(nn.Module):
    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant

def quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    cum_prob: Optional[torch.Tensor] = None,
    sum_over_quantiles: bool = True,
) -> torch.Tensor:
    """
    The quantile-regression loss, as described in the QR-DQN and TQC papers.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch.

    :param current_quantiles: current estimate of quantiles, must be either
        (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
    :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
        (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
    :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
        must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles).
        (if None, calculating unit quantiles)
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    if current_quantiles.ndim != target_quantiles.ndim:
        raise ValueError(
            f"Error: The dimension of curremt_quantile ({current_quantiles.ndim}) needs to match "
            f"the dimension of target_quantiles ({target_quantiles.ndim})."
        )
    if current_quantiles.shape[0] != target_quantiles.shape[0]:
        raise ValueError(
            f"Error: The batch size of current_quantile ({current_quantiles.shape[0]}) needs to match "
            f"the batch size of target_quantiles ({target_quantiles.shape[0]})."
        )
    if current_quantiles.ndim not in (2, 3):
        raise ValueError(f"Error: The dimension of current_quantiles ({current_quantiles.ndim}) needs to be either 2 or 3.")

    if cum_prob is None:
        n_quantiles = current_quantiles.shape[-1]
        # Cumulative probabilities to calculate quantiles.
        cum_prob = (torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5) / n_quantiles
        if current_quantiles.ndim == 2:
            # For QR-DQN, current_quantiles have a shape (batch_size, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, -1, 1)
        elif current_quantiles.ndim == 3:
            # For TQC, current_quantiles have a shape (batch_size, n_critics, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_critics, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, 1, -1, 1)

    # QR-DQN
    # target_quantiles: (batch_size, n_target_quantiles) -> (batch_size, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_target_quantiles, n_quantiles)
    # TQC
    # target_quantiles: (batch_size, 1, n_target_quantiles) -> (batch_size, 1, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_critics, n_quantiles) -> (batch_size, n_critics, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_critics, n_quantiles, n_target_quantiles)
    # Note: in both cases, the loss has the same shape as pairwise_delta
    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    if sum_over_quantiles:
        loss = loss.sum(dim=-2).mean()
    else:
        loss = loss.mean()
    return loss


def calculate_huber_loss(td_errors, kappa=1.0):
    return torch.where(
        td_errors.abs() <= kappa,
        0.5 * td_errors.pow(2),
        kappa * (td_errors.abs() - 0.5 * kappa))

def calculate_quantile_huber_loss(td_errors, taus, weights=None, kappa=1.0):
    assert not taus.requires_grad
    batch_size, N, N_dash = td_errors.shape

    # Calculate huber loss element-wisely.
    element_wise_huber_loss = calculate_huber_loss(td_errors, kappa)
    assert element_wise_huber_loss.shape == (
        batch_size, N, N_dash)

    # Calculate quantile huber loss element-wisely.
    element_wise_quantile_huber_loss = torch.abs(
        taus[..., None] - (td_errors.detach() < 0).float()
        ) * element_wise_huber_loss / kappa
    assert element_wise_quantile_huber_loss.shape == (
        batch_size, N, N_dash)

    # Quantile huber loss.
    batch_quantile_huber_loss = element_wise_quantile_huber_loss.sum(
        dim=1).mean(dim=1, keepdim=True)
    assert batch_quantile_huber_loss.shape == (batch_size, 1)

    if weights is not None:
        quantile_huber_loss = (batch_quantile_huber_loss * weights).mean()
    else:
        quantile_huber_loss = batch_quantile_huber_loss.mean()

    return quantile_huber_loss


def calculate_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Calculates the entropy for a batch of probability distributions.

    Handles distributions with zero probabilities and the case where all
    probabilities in a distribution might be zero (resulting in entropy 0).

    Args:
        probs: A tensor of shape [B, N] containing batch_size (B)
               probability distributions over N categories.
               Values should be non-negative. They don't strictly
               need to sum to 1 for the calculation, though entropy
               is typically defined for valid distributions.

    Returns:
        A tensor of shape [B] containing the entropy for each distribution
        in the batch.
    """
    # Clamp probabilities to avoid log(0) issues if any tiny negative values slipped in
    # Although log(0) is handled below, log(negative) would cause issues.
    # Use a small epsilon if you expect near-zero values but want to avoid exact zeros
    # for log calculation stability if not using nan_to_num. But with nan_to_num,
    # direct log is fine.
    # probs = torch.clamp(probs, min=0.0) # Ensure non-negative

    # Calculate log probabilities. log(0) will result in -inf.
    log_probs = torch.log(probs)

    # Calculate p * log(p).
    # Where p = 0, this becomes 0 * -inf = NaN.
    p_log_p = probs * log_probs

    # Use nan_to_num to replace NaN results (from 0 * log(0)) with 0.0.
    # This correctly implements the limit lim_{x->0} x*log(x) = 0.
    # It also handles the -inf from log_probs if probs was 0, ensuring 0 * -inf -> 0
    safe_p_log_p = torch.nan_to_num(p_log_p, nan=0.0, posinf=0.0, neginf=0.0)

    # Sum across the N dimension and negate to get entropy
    # Summing 0 for all terms in the all-zero case results in 0 entropy.
    entropy = -torch.sum(safe_p_log_p, dim=-1)

    return entropy



# def get_min_q(quantile1, quantile2):
#     q1 = quantile1.mean(-1, keepdim=True)
#     q2 = quantile2.mean(-1, keepdim=True)
#
#     q = torch.cat([q1, q2], dim=2)
#
#     quantile = torch.cat([quantile1.unsqueeze(2), quantile2.unsqueeze(2)], dim=2)
#
#     arg_q = torch.argmin(q, dim=2, keepdim=True)
#
#     min_quantile = torch.gather(quantile, 2, arg_q.unsqueeze(-1).expand(-1, -1, -1, quantile.shape[-1])).squeeze(2)
#
#     return min_quantile

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

# def get_min_q(quantiles):
#     stacked_quantiles = torch.stack(quantiles, dim=0)
#     min_quantiles = torch.min(stacked_quantiles, dim=0).values
#     # print(min_quantiles)
#     # exit()
#     return min_quantiles


def add_noise_to_distribution(probabilities, noise_level=0.1):
    noise = torch.randn_like(probabilities) * noise_level
    noisy_probs = probabilities + noise
    noisy_probs = torch.where(probabilities == 0, 0, noisy_probs)
    noisy_probs = noisy_probs.clamp(min=0.0)
    noisy_prob_sum = noisy_probs.sum(dim=-1, keepdim=True)
    return noisy_probs / noisy_prob_sum

import torch.nn.functional as F

def imitation_loss(logits, expert_actions):
    # logits: (N, C)
    # expert_actions: (N, 1) -> (N,)
    expert_actions = expert_actions.squeeze(-1)      # (N,)
    return F.cross_entropy(logits, expert_actions.long())

class Imitation_Learning:
    def __init__(
            self,
            actor_net: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            N: int = 50,
            max_grad_norm: float = 1.0,
            device: str = "cpu",

    ):
        self.actor_net = actor_net

        self.actor_optimizer = actor_optimizer

        self.max_grad_norm = max_grad_norm

        # self.beta = beta
        # self.iql_tau = iql_tau
        self.device = device
        self.N = N
        self.n_updates = 0
        self.n_updates_policy = 0


        # self.lr_scheduler = CosineAnnealingLR(self.p, max_steps, eta_min=0.0)




    def calculate_dqn_loss(self, state, next_state, actions, rewards, dones, log_dict):
        with torch.no_grad():
            nextQ, (next_fea_j, next_fea_m) = self.target_net(*next_state)
            nextQ = [nQ.squeeze(-1) for nQ in nextQ]
            nextQ_stack = torch.stack(nextQ, dim=0)

            nextQ = torch.min(nextQ_stack, dim=0).values

            nextProbs, _ = self.actor_net(*next_state)

            nextProbs = torch.nan_to_num(nextProbs, nan=0.0)

            # nextQ = get_min_q(nextQ)

            target_q = nextProbs * nextQ
            target_q = torch.nan_to_num(target_q, nan=0.0)


            target_q = target_q.sum(dim=-1, keepdim=True)


            target = rewards + (1.0 - dones) * self.discount * target_q


        currQ, (fea_j, fea_m) = self.q_net(*state)
        currQ = [cQ.squeeze(-1) for cQ in currQ]


        q_losses = []
        cql_losses = []

        for i in range(len(currQ)):
            q = currQ[i]
            q_action = q.gather(1, actions)


            q_loss = torch.nn.functional.mse_loss(target, q_action)


            q_losses.append(q_loss)
            if self.use_cql:
                # print()
                # cql_loss_q = self.cql_alpha * (
                #     -torch.log(torch.exp(q_action) / torch.sum(torch.exp(q), dim=1, keepdim=True)).mean())
                cql_loss_q = self.cql_alpha * (torch.logsumexp(q, dim=1, keepdim=True) - q_action).mean()

                cql_losses.append(cql_loss_q)

        # exit()
        q_loss = torch.stack(q_losses)

        log_dict["td_error_q"] = q_loss.mean().item()
        if self.use_cql:
            cql_loss = torch.stack(cql_losses)
            q_loss = q_loss + cql_loss
            log_dict["cql_loss"] = cql_loss.mean().item()
        q_loss = q_loss.sum()
        log_dict["mean_target_q"] = target_q.mean().item()
        log_dict["mean_target"] = target.mean().item()
        log_dict["std_target_q"] = target_q.std().item()
        log_dict["std_target"] = target.std().item()
        return q_loss

    def calculate_qrdqn_loss(self, state, next_state, actions, rewards, dones, log_dict):
        with torch.no_grad():
            nextQ, (next_fea_j, next_fea_m) = self.target_net(*next_state)
            nextQ = get_min_q(nextQ)
            if self.n_updates > self.q_pretrain_steps:
                nextProbs, _ = self.actor_net(*next_state)

                nextProbs = torch.nan_to_num(nextProbs, nan=0.0)


                target_q = nextProbs.unsqueeze(-1) * nextQ
                target_q = torch.nan_to_num(target_q, nan=0.0)

                target_q = target_q.sum(1)
            else:
                nextQ = torch.where(nextQ == float('-inf'), torch.nan, nextQ)
                target_q = nextQ.nanmean(dim=1)

            # entropy_bonus = (0.03 * entropy_bonus.unsqueeze(-1))
            # target_q = target_q.sum(1) + entropy_bonus
            # log_dict["entropy_bonus"] = entropy_bonus.mean().item()

            # print(target_q.shape, entropy_bonus.shape)
            # exit()

            target = rewards + (1.0 - dones) * self.discount * target_q

        currQ, (fea_j, fea_m) = self.q_net(*state)
        # fea_j_flatten = fea_j.flatten(1, 2)
        # next_fea_j_flatten = next_fea_j.flatten(1, 2)
        # dot_prod_j = fea_j_flatten.unsqueeze(1) @ next_fea_j_flatten.unsqueeze(2)
        # fea_m_flatten = fea_m.flatten(1, 2)
        # next_fea_m_flatten = next_fea_m.flatten(1, 2)
        # dot_prod_m = fea_m_flatten.unsqueeze(1) @ next_fea_m_flatten.unsqueeze(2)
        # dot_prod = (dot_prod_j + dot_prod_m).squeeze(2)
        # dot_prod = dot_prod[~(dones).bool()].mean()
        #
        # dr3_term = dot_prod * 0.001

        q_losses = []
        cql_losses = []
        for i in range(len(currQ)):
            q = currQ[i]
            q_action = q.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.N)).squeeze(1)

            q_loss = quantile_huber_loss(q_action, target, sum_over_quantiles=True)

            q_losses.append(q_loss)
            if self.use_cql:
                # q = q.mean(-1)
                # q_action = q_action.squeeze(-1).mean(-1)
                # cql_loss_q = self.cql_alpha * (
                #     -torch.log(torch.exp(q_action) / torch.sum(torch.exp(q), dim=1))).mean()
                # cql_losses.append(cql_loss_q)
                q = q.mean(-1)
                q_action = q_action.squeeze(-1).mean(-1)
                cql_loss_q = self.cql_alpha * ((torch.logsumexp(q / self.cql_temp, dim=1) * self.cql_temp) - q_action).mean()
                cql_losses.append(cql_loss_q)

        # exit()
        q_loss = torch.stack(q_losses)

        log_dict["td_error_q"] = q_loss.mean().item()
        if self.use_cql:
            cql_loss = torch.stack(cql_losses)
            q_loss = (q_loss) + cql_loss
            log_dict["cql_loss"] = cql_loss.mean().item()
        q_loss = q_loss.sum()
        log_dict["mean_target_q"] = target_q.mean().item()
        # log_dict["dr3_term"] = dr3_term.item()
        log_dict["mean_target"] = target.mean().item()
        # log_dict["dr3_term"] = dr3_term.item()
        log_dict["std_target_q"] = target_q.std().item()
        log_dict["std_target"] = target.std().item()
        return q_loss

    def get_q_values(self, state):
        with torch.no_grad():
            q_vals, _ = self.q_net(*state)
        if self.use_qrdqn:
            quantiles = get_min_q(q_vals)
            q_val = quantiles.mean(-1)
            q_val = q_val.masked_fill(q_val == float('-inf'), torch.nan)

        else:
            q_vals = [q.squeeze(-1) for q in q_vals]
            q_vals = torch.stack(q_vals, dim=0)

            q_val = torch.min(q_vals, dim=0).values
            q_val = q_val.masked_fill(q_val == float('-inf'), torch.nan)
        return q_val


    def train(self, batch):
        self.n_updates += 1
        log_dict = {}
        state, next_state, actions, rewards, dones, mc_returns = batch

        _, logits = self.actor_net(*state)
        imitation_loss_value = imitation_loss(logits, actions)
        self.actor_optimizer.zero_grad()
        imitation_loss_value.backward()
        nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()
        log_dict["imitation_loss"] = imitation_loss_value.item()






        return log_dict



    def update_target(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        # for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
        #     target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_dict(self):
        return {
            "actor_net": self.actor_net.state_dict(),
        }

    # def load_dict(self):
    #     return {
    #         "dqn_net": self.dqn_net.state_dict(),
    #         "target_net": self.target_net.state_dict(),
    #         "dqn_optimizer": self.dqn_optimizer.state_dict(),
    #         "dqn_lr_scheduler": self.dqn_lr_scheduler.state_dict(),
    #     }
        # pass