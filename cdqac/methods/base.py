"""Base classes shared by all offline-RL methods in ``cdqac.methods``.

Two levels of sharing:

* :class:`BaseMethod` — plumbing every method needs: update counters, the
  optimizer step (zero_grad / backward / clip / step), Polyak target updates
  and checkpoint dictionaries.
* :class:`QuantileActorCriticBase` — the full conservative quantile
  actor-critic machinery (critic losses with CQL/CODAC penalties, pessimistic
  Q-value extraction, entropy temperature) shared verbatim by
  :class:`cdqac.methods.cdqac.CDQAC` and
  :class:`cdqac.methods.d_msac.discrete_mSAC`, which differ only in their
  actor update (``train``).
"""
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from cdqac.methods.util import (Scalar, codac_discrete_penalty, get_min_q,
                                quantile_huber_loss, soft_update)


class BaseMethod(ABC):
    """Abstract base class for every training method in this package.

    Subclasses must implement :meth:`train`; the constructor only stores the
    handful of attributes every method uses.  Methods with a critic are
    expected to expose ``q_net`` / ``target_net`` so the default
    :meth:`update_target` works; methods with an actor are expected to expose
    ``actor_net`` so the default :meth:`get_dict` works.
    """

    def __init__(
            self,
            discount: float = 1.0,
            tau: float = 0.005,
            target_update_freq: int = 1,
            max_grad_norm: float = 1.0,
            device: str = "cpu",
    ):
        """
        Args:
            discount: Discount factor gamma for TD targets.
            tau: Polyak averaging rate for target-network updates.
            target_update_freq: Update the target network every n train steps.
            max_grad_norm: Gradient-norm clip applied in :meth:`_optimize`
                (``None`` disables clipping).
            device: Torch device string the tensors live on.
        """
        self.discount = discount
        self.tau = tau
        self.target_update_freq = target_update_freq
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.n_updates = 0
        self.n_updates_policy = 0

    @abstractmethod
    def train(self, batch) -> dict:
        """Perform one gradient update on ``batch`` and return a log dict."""

    def _optimize(self, optimizer: torch.optim.Optimizer, loss: torch.Tensor,
                  network: nn.Module) -> None:
        """One optimizer step: zero_grad, backward, clip gradients, step."""
        optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), self.max_grad_norm)
        optimizer.step()

    def update_target(self) -> None:
        """Polyak-average the online critic ``q_net`` into ``target_net``."""
        soft_update(self.target_net, self.q_net, self.tau)

    def get_dict(self) -> dict:
        """State dict used for checkpoints; evaluation only reads "actor_net"."""
        return {
            "actor_net": self.actor_net.state_dict(),
        }


class QuantileActorCriticBase(BaseMethod):
    """Conservative quantile actor-critic core shared by CDQAC and mSAC.

    Holds the actor/critic networks and optimizers, the learnable entropy
    temperature, optional entropy/learning-rate annealing schedules, and the
    critic losses (scalar DQN or distributional QR-DQN, both with an optional
    CQL — or, for QR-DQN, CODAC — conservative penalty).  Subclasses implement
    :meth:`train` with their own actor update.
    """

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
            alpha: float = 2.5,
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
    ):
        """
        Args:
            actor_net: Masked-softmax policy network.
            target_actor_net: Target copy of the actor (kept for API
                compatibility; the shared updates only touch the critic
                target).
            q_net: Critic ensemble; returns a list of per-critic tensors and
                the encoder features.
            target_net: Target copy of ``q_net``.
            actor_optimizer: Optimizer for ``actor_net``.
            q_optimizer: Optimizer for ``q_net``.
            target_update_freq: Polyak-update the critic target every n steps.
            update_freq_policy: Actor update delay (used by subclasses).
            discount: Discount factor gamma.
            tau: Polyak averaging rate.
            max_steps: Total train steps, used by the entropy anneal schedule.
            q_pretrain_steps: Steps during which only the critic is trained
                (the QR-DQN target then averages over feasible actions instead
                of using the actor distribution).
            use_calql: Accepted for API compatibility (unused).
            use_cql: Add the conservative (CQL) penalty to the critic loss.
            alpha_lr: Learning rate of the entropy temperature.
            use_max_target: Accepted for API compatibility (unused).
            target_entropy: Fixed target-entropy fraction when not annealing.
            alpha: Accepted for API compatibility (unused).
            alpha_multiplier: Multiplier on the learned entropy temperature.
            cql_alpha: CQL penalty strength.
            cql_temp: Temperature inside the CQL logsumexp.
            N: Number of quantiles of the critic.
            max_grad_norm: Gradient-norm clip (``None`` disables clipping).
            device: Torch device string.
            anneal_entropy: Linearly anneal the target entropy from
                ``start_target_entropy`` to ``end_target_entropy``.
            start_target_entropy: Anneal start value.
            end_target_entropy: Anneal end value.
            anneal_lr: Cosine-anneal the actor learning rate.
            max_steps_lr: Horizon of the cosine schedule.
            noise_level: Accepted for API compatibility (unused).
            kappa: Accepted for API compatibility (unused).
            backup_entropy: Accepted for API compatibility (unused).
            normalize_q: Center Q-values across actions in the actor loss
                (used by subclasses).
            use_qrdqn: Distributional QR-DQN critic (else scalar DQN critic).
            use_codac: Replace the CQL penalty by the CODAC distributional
                penalty in the QR-DQN loss.
        """
        super().__init__(discount=discount, tau=tau,
                         target_update_freq=target_update_freq,
                         max_grad_norm=max_grad_norm, device=device)
        self.actor_net = actor_net
        self.target_actor_net = target_actor_net
        self.q_net = q_net
        self.target_net = target_net
        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer

        self.update_freq_policy = update_freq_policy
        self.q_pretrain_steps = q_pretrain_steps
        self.N = N
        self.normalize_q = normalize_q
        self.use_qrdqn = use_qrdqn

        self.use_cql = use_cql
        self.use_codac = use_codac
        self.use_calql = use_calql
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp

        self.alpha = alpha
        self.alpha_lr = alpha_lr
        self.alpha_multiplier = alpha_multiplier
        self.use_max_target = use_max_target
        self.backup_entropy = backup_entropy
        self.noise_level = noise_level
        self.kappa = kappa

        self.log_alpha = Scalar(0.0)
        self.alpha_optimizer = torch.optim.Adam(
            self.log_alpha.parameters(),
            lr=self.alpha_lr,
        )

        self.target_entropy = target_entropy
        self.start_target_entropy = start_target_entropy
        self.end_target_entropy = end_target_entropy
        if anneal_entropy:
            self.target_ent_func = lambda k: (start_target_entropy + (end_target_entropy - start_target_entropy) /
                                              (max_steps - 1) * k) if max_steps > 1 else start_target_entropy
        else:
            self.target_ent_func = lambda k: target_entropy
            self.end_target_entropy = target_entropy
            self.start_target_entropy = target_entropy

        self.anneal_lr = anneal_lr
        if anneal_lr:
            self.lr_scheduler = CosineAnnealingLR(self.actor_optimizer, max_steps_lr, eta_min=0.0)
        else:
            self.lr_scheduler = None

    def _alpha_and_alpha_loss(self, entropy: torch.Tensor, action_masks: torch.Tensor,
                              target_entropy: float):
        """Compute the entropy temperature and its dual loss (SAC-style).

        The per-sample target entropy is ``target_entropy`` times the maximum
        entropy ``log(n_feasible_actions)`` of the state's feasibility mask.

        Args:
            entropy: Detached policy entropies, shape ``[B]``.
            action_masks: Boolean mask of infeasible actions (True = masked).
            target_entropy: Fraction of the maximum entropy to target.

        Returns:
            Tuple ``(alpha, alpha_loss)``: the current temperature (scaled by
            ``alpha_multiplier``) and the loss for the temperature optimizer.
        """
        assert not entropy.requires_grad
        action_masks = (~action_masks.flatten(1)).float()

        target_entropy = (-torch.log(1.0 / torch.sum(action_masks, dim=1)) * target_entropy).unsqueeze(-1)

        alpha_loss = -(
            self.log_alpha() * (target_entropy - entropy).detach()
        ).mean()

        alpha = self.log_alpha().exp() * self.alpha_multiplier

        return alpha, alpha_loss

    def calculate_dqn_loss(self, state, next_state, actions, rewards, dones, log_dict):
        """Critic loss for the scalar (non-distributional) DQN variant.

        The target is the actor-weighted expectation of the minimum over the
        target-critic ensemble; each critic gets an MSE TD loss plus an
        optional CQL penalty.  Returns the summed loss over critics.
        """
        with torch.no_grad():
            nextQ, (next_fea_j, next_fea_m) = self.target_net(*next_state)
            nextQ = [nQ.squeeze(-1) for nQ in nextQ]
            nextQ_stack = torch.stack(nextQ, dim=0)

            nextQ = torch.min(nextQ_stack, dim=0).values

            nextProbs, _ = self.actor_net(*next_state)

            nextProbs = torch.nan_to_num(nextProbs, nan=0.0)

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
                cql_loss_q = self.cql_alpha * (torch.logsumexp(q, dim=1, keepdim=True) - q_action).mean()

                cql_losses.append(cql_loss_q)

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
        """Critic loss for the distributional QR-DQN variant.

        The target distribution is the actor-weighted mixture of the
        pessimistic (per-entry minimum-mean) target-critic quantiles; during
        critic pretraining it is the mean over feasible actions instead.  Each
        critic gets a quantile Huber TD loss plus an optional conservative
        penalty (CQL on the quantile means, or CODAC when ``use_codac``).
        Returns the summed loss over critics.
        """
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

            target = rewards + (1.0 - dones) * self.discount * target_q

        currQ, (fea_j, fea_m) = self.q_net(*state)

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
                    q = q.mean(-1)
                    q_action = q_action.squeeze(-1).mean(-1)
                    cql_loss_q = self.cql_alpha * ((torch.logsumexp(q / self.cql_temp, dim=1) * self.cql_temp) - q_action).mean()
                cql_losses.append(cql_loss_q)

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

    def get_q_values(self, state):
        """Pessimistic Q(s, .) of the online critic ensemble, no gradients.

        Reduces the ensemble with :func:`get_min_q` (QR-DQN) or an
        element-wise minimum (scalar DQN) and replaces the ``-inf`` entries of
        infeasible actions by NaN so downstream reductions can use ``nanmean``.

        Returns:
            Tensor of shape ``[B, A]``.
        """
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
