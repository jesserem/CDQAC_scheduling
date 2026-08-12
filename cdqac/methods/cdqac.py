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


def codac_discrete_penalty(
    z_pred: torch.Tensor,
    z_all: torch.Tensor,
    min_z_weight: float
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute the CODAC conservative penalty for discrete action spaces.
 
    Unlike the continuous version, we enumerate all actions exactly
    instead of relying on importance-sampled approximations.
 
    Args:
        z_pred: Quantile values for the dataset actions.
            Shape (batch_size, num_quantiles).
        z_all: Quantile values for ALL discrete actions at each state.
            Shape (batch_size, num_actions, num_quantiles).
        min_z_weight: ω in the paper — scales the penalty magnitude.
        with_lagrange: Whether to use the dual-variable (α') formulation.
        log_alpha_prime: Log of the Lagrange multiplier (learnable scalar).
            Required when with_lagrange=True.
        lagrange_thresh: ζ in the paper — threshold for the Lagrange term.
 
    Returns:
        penalty: Scalar conservative penalty to ADD to the critic loss.
        alpha_prime_loss: Scalar loss for updating α' (None if not using
            Lagrange).  Caller should call .backward() on this separately
            and step the α'-optimizer.
    """
    # ---- select a random quantile index (as in original code, line 246) ----
    num_quantiles = z_pred.shape[1]
    qi = torch.randint(0, num_quantiles, (1,)).item()
 
    # z_all_qi: (batch_size, num_actions)  — one quantile slice per action
    z_all_qi = z_all[:, :, qi]
 
    # ---- logsumexp over actions (exact for discrete) ----------------------
    # This is  log Σ_a exp(F^{-1}_{Z(s,a)}(τ_qi))  averaged over the batch.
    # In the continuous code this was approximated via importance sampling
    # (Eq. 15, Appendix B).  Here it is exact.
    logsumexp_term = torch.logsumexp(z_all_qi, dim=1).mean()
 
    # ---- dataset quantile mean (the subtracted term) ----------------------
    # E_{D(s,a)}[ F^{-1}_{Z(s,a)}(τ) ]  averaged over all quantiles.
    # The original code uses z_pred.mean() which averages over both the batch
    # and all N quantile indices (line 266-267).
    dataset_term = z_pred.mean()
 
    # ---- raw penalty -------------------------------------------------------
    penalty = (logsumexp_term - dataset_term) * min_z_weight
 
    # ---- optional Lagrange multiplier (α') ---------------------------------
    
    
 
    return penalty


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

class CDQAC:
    def __init__(
            self,
            actor_net: nn.Module,
            target_actor_net: nn.Module,
            q_net: nn.Module,
            target_net: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            q_optimizer: torch.optim.Optimizer,
            target_update_freq: int,
            update_freq_policy: int,
            discount: float,
            tau: float,
            max_steps: int,
            q_pretrain_steps: int = 0,
            use_calql: bool = False,
            use_cql: bool = True,
            alpha_lr: float = 3e-4,
            use_max_target: bool = False,
            target_entropy: float = 0.2,
            alpha: float= 2.5,
            alpha_multiplier: float = 1,
            cql_alpha: float = 1,
            cql_temp: float = 1,
            N: int = 50,
            max_grad_norm: float = 1.0,
            device: str = "cpu",
            anneal_entropy: bool = False,
            start_target_entropy: float = 0.98,
            end_target_entropy: float = 0.3,
            anneal_lr: bool = True,
            max_steps_lr: int = 1e6,
            noise_level: float = 0.1,
            kappa: float = 1,
            backup_entropy: bool = False,
            normalize_q: bool = True,
            use_qrdqn: bool = True,
            use_codac: bool = False,
            entropy_coef: float = 0.001,

    ):
        self.use_calql = use_calql
        self._calibration_enabled = True
        self.use_qrdqn = use_qrdqn
        self.actor_net = actor_net
        self.target_actor_net = target_actor_net
        self.q_net = q_net
        self.kappa = kappa
        self.use_codac = use_codac
        self.target_net = target_net

        self.tau = tau
        self.alpha = alpha
        self.noise_level = noise_level
        self.use_cql = use_cql
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp
        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.normalize_q = normalize_q
        self.target_update_freq = target_update_freq
        self.update_freq_policy = update_freq_policy
        self.max_grad_norm = max_grad_norm
        self.discount = discount
        self.alpha_lr = alpha_lr
        self.entropy_coef = entropy_coef

        self.log_alpha = Scalar(0.0)
        self.alpha_optimizer = torch.optim.Adam(
            self.log_alpha.parameters(),
            lr=self.alpha_lr,
        )

        # self.beta = beta
        # self.iql_tau = iql_tau
        self.device = device
        self.N = N
        self.n_updates = 0
        self.n_updates_policy = 0
        self.q_pretrain_steps = q_pretrain_steps
        self.target_entropy = target_entropy
        self.alpha_multiplier = alpha_multiplier
        self.start_target_entropy = start_target_entropy
        self.end_target_entropy = end_target_entropy
        self.use_max_target = use_max_target
        self.backup_entropy = backup_entropy
        if anneal_entropy:
            self.target_ent_func = lambda k: (start_target_entropy + (end_target_entropy - start_target_entropy) /
                                              (max_steps - 1) * k) if max_steps > 1 else start_target_entropy
            # self.target_entropy = start_target_entropy
            # self.entropy_anneal = (start_target_entropy - end_target_entropy) / max_steps
        else:
            self.target_ent_func = lambda k: target_entropy
            self.end_target_entropy = target_entropy
            self.start_target_entropy = target_entropy
        self.anneal_lr = anneal_lr
        if anneal_lr:
            self.lr_scheduler = CosineAnnealingLR(self.actor_optimizer, max_steps_lr, eta_min=0.0)
        else:
            self.lr_scheduler = None



        taus = torch.arange(
            0, N + 1, device=self.device, dtype=torch.float32) / N
        self.tau_hats = ((taus[1:] + taus[:-1]) / 2.0).view(1, N)
        self.tau_decay = lambda step: 10 - (9.9 * step / (100_000 - 1))

        # self.lr_scheduler = CosineAnnealingLR(self.p, max_steps, eta_min=0.0)


    def _alpha_and_alpha_loss(self, entropy: torch.Tensor, action_masks: torch.Tensor, target_entropy: float):
        assert not entropy.requires_grad
        action_masks = (~action_masks.flatten(1)).float()

        target_entropy = (-torch.log(1.0 / torch.sum(action_masks, dim=1)) * target_entropy).unsqueeze(-1)

        # print(target_entropy[0])
        alpha_loss = -(
            self.log_alpha() * (target_entropy - entropy).detach()
        ).mean()

        alpha = self.log_alpha().exp() * self.alpha_multiplier

        return alpha, alpha_loss

    def switch_calibration(self):
        self._calibration_enabled = not self._calibration_enabled


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
                if self.use_codac:
                    cql_loss_q = codac_discrete_penalty(q_action, q, self.cql_alpha)
                else:
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

        # print(actions)
        # exit()
        if self.use_qrdqn:
            q_loss = self.calculate_qrdqn_loss(state, next_state, actions, rewards, dones, log_dict)
        else:
            q_loss = self.calculate_dqn_loss(state, next_state, actions, rewards, dones, log_dict)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.q_optimizer.step()

        log_dict["q_loss"] = q_loss.item()



        # TODO: IMPROVE LOG PROBS
        if self.n_updates % self.update_freq_policy == 0 and self.n_updates > self.q_pretrain_steps:
            self.n_updates_policy += 1
            q_val = self.get_q_values(state)

            probs, log_probs = self.actor_net(*state)
            action_dist = torch.distributions.Categorical(probs)
            entropy = action_dist.entropy()

            if self.normalize_q:

                # q_val = q_val * lmbda
                mean_q = torch.nanmean(q_val.detach(), dim=-1, keepdim=True)
                q_val = (q_val - mean_q)
                log_dict["mean_q"] = mean_q.mean().item()




            q_val = torch.nan_to_num(q_val, nan=0.0)

            rl_loss = -(self.entropy_coef * entropy + (probs * q_val).sum(-1))

            policy_loss = rl_loss.mean()

            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()


            log_dict["policy_loss"] = policy_loss.item()

            log_dict["rl_loss"] = rl_loss.mean().item()
            log_dict["rl_loss_min"] = rl_loss.min().item()
            log_dict["rl_loss_max"] = rl_loss.max().item()
            log_dict["entropy_min"] = entropy.min().item()
            log_dict["entropy"] = entropy.mean().item()
            log_dict["entropy_max"] = entropy.max().item()

            # if self.n_updates_policy % self.target_update_freq == 0:
            #     self.update_target()

        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            log_dict["actor_lr"] = self.lr_scheduler.get_last_lr()[0]




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

    def get_dict_with_critic(self):
        # Same as get_dict but additionally stores the critic (and its target).
        # Backwards compatible: the extra keys are ignored by evaluation code
        # that only reads "actor_net".
        return {
            "actor_net": self.actor_net.state_dict(),
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
        }

    # def load_dict(self):
    #     return {
    #         "dqn_net": self.dqn_net.state_dict(),
    #         "target_net": self.target_net.state_dict(),
    #         "dqn_optimizer": self.dqn_optimizer.state_dict(),
    #         "dqn_lr_scheduler": self.dqn_lr_scheduler.state_dict(),
    #     }
        # pass