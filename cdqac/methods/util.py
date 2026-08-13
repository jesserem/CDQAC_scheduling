"""Shared building blocks for the offline-RL methods in ``cdqac.methods``.

Everything in this module is a pure function (or a tiny stateless module) that
was previously copy-pasted across the individual method files.  The training
classes themselves live in the per-method modules and share plumbing through
:class:`cdqac.methods.base.BaseMethod`.
"""
from typing import Optional

import torch
import torch.nn as nn


class Scalar(nn.Module):
    """A single learnable scalar wrapped in a module.

    Used for dual variables such as the entropy temperature ``log(alpha)`` so
    that a standard optimizer can be attached to it.
    """

    def __init__(self, init_value: float):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self) -> nn.Parameter:
        return self.constant


def soft_update(target_net: nn.Module, source_net: nn.Module, tau: float) -> None:
    """Polyak-average ``source_net`` into ``target_net`` in place.

    ``target <- tau * source + (1 - tau) * target`` for every parameter.

    Args:
        target_net: Network whose parameters are updated.
        source_net: Network providing the fresh parameters.
        tau: Averaging rate in [0, 1]; 1 copies the source verbatim.
    """
    for target_param, param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    """Expectile-regression loss used by IQL-style value updates.

    Weighs positive residuals by ``tau`` and negative ones by ``1 - tau``,
    so ``tau = 0.5`` reduces to a plain (halved) L2 loss.

    Args:
        u: Residuals (e.g. ``q - v``), any shape.
        tau: Expectile in (0, 1).

    Returns:
        Scalar loss (mean over all elements).
    """
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


def get_min_q(quantiles: list[torch.Tensor]) -> torch.Tensor:
    """Reduce an ensemble of quantile critics to the most pessimistic one.

    For every (batch, action) entry the critic with the lowest *expected*
    value (mean over quantiles) is selected and its **whole quantile vector**
    is returned, i.e. the reduction picks a critic per entry rather than
    taking an element-wise minimum.

    Args:
        quantiles: List of per-critic tensors of shape ``[B, A, D]`` where
            ``A`` is the number of actions and ``D`` the number of quantiles.

    Returns:
        Tensor of shape ``[B, A, D]`` with the selected quantile vectors.
    """
    # Stack to [B, A, n_critics, D], rank the critics by their mean value and
    # gather the full quantile vector of the winning critic.
    stacked_quantiles = torch.stack(quantiles, dim=2)
    q_means = stacked_quantiles.mean(dim=-1)
    arg_q = torch.argmin(q_means, dim=2, keepdim=True)
    min_quantile = torch.gather(
        stacked_quantiles,
        2,
        arg_q.unsqueeze(-1).expand(-1, -1, -1, stacked_quantiles.shape[-1])
    ).squeeze(2)

    return min_quantile


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


def codac_discrete_penalty(
    z_pred: torch.Tensor,
    z_all: torch.Tensor,
    min_z_weight: float
) -> torch.Tensor:
    """Compute the CODAC conservative penalty for discrete action spaces.

    Unlike the continuous version, all actions are enumerated exactly instead
    of relying on importance-sampled approximations: a random quantile index
    is drawn, the logsumexp over all actions at that quantile is the OOD term
    and the mean dataset quantile is subtracted from it.

    Args:
        z_pred: Quantile values for the dataset actions.
            Shape (batch_size, num_quantiles).
        z_all: Quantile values for ALL discrete actions at each state.
            Shape (batch_size, num_actions, num_quantiles).
        min_z_weight: Scales the penalty magnitude (omega in the paper).

    Returns:
        Scalar conservative penalty to ADD to the critic loss.
    """
    # ---- select a random quantile index -----------------------------------
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
    dataset_term = z_pred.mean()

    penalty = (logsumexp_term - dataset_term) * min_z_weight

    return penalty
