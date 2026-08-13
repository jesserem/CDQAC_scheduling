"""IQL — Implicit Q-Learning (discrete, masked FJSP action space).

Standard IQL trio of updates per step: an expectile-regressed value net, a
smooth-L1 TD update of the critic ensemble toward ``r + gamma * V(s')``, and
an advantage-weighted-regression (AWR) actor update.
"""
import torch
import torch.nn as nn

from cdqac.methods.base import BaseMethod
from cdqac.methods.util import asymmetric_l2_loss, get_min_q


class IQL(BaseMethod):
    """Implicit Q-Learning trainer."""

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
        """
        Args:
            actor_net: Masked-softmax policy network.
            q_net: Critic ensemble (scalar Q per action).
            target_net: Target copy of ``q_net``.
            value_net: State-value network V(s).
            value_optimizer: Optimizer for ``value_net``.
            actor_optimizer: Optimizer for ``actor_net``.
            q_optimizer: Optimizer for ``q_net``.
            target_update_freq: Polyak-update the critic target every n steps.
            update_freq_policy: Accepted for API compatibility (the actor is
                updated every step).
            discount: Discount factor gamma.
            tau: Polyak averaging rate.
            iql_tau: Expectile of the value regression.
            beta: Inverse temperature of the AWR weight ``exp(beta * adv)``.
            max_grad_norm: Gradient-norm clip.
            device: Torch device string.
        """
        super().__init__(discount=discount, tau=tau,
                         target_update_freq=target_update_freq,
                         max_grad_norm=max_grad_norm, device=device)
        self.actor_net = actor_net
        self.q_net = q_net
        self.target_net = target_net
        self.value_net = value_net

        self.iql_tau = iql_tau
        self.beta = beta

        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.value_optimizer = value_optimizer

        self.update_freq_policy = update_freq_policy

    def _q_loss(self, state, actions, rewards, dones, next_v, log_dict):
        """Smooth-L1 TD update of every critic toward ``r + gamma * V(s')``."""
        target = rewards + (1 - dones) * self.discount * next_v.detach()
        q_val, _ = self.q_net(*state)
        q_loss = 0
        for i in range(len(q_val)):
            q_val_i = q_val[i].squeeze(-1).gather(1, actions)
            log_dict["q_val_" + str(i)] = q_val_i.mean().item()
            q_loss_i = torch.nn.functional.smooth_l1_loss(q_val_i, target)
            q_loss += q_loss_i
        self._optimize(self.q_optimizer, q_loss, self.q_net)
        log_dict["q_loss"] = q_loss.item()

    def _policy_loss(self, state, adv, actions, log_dict):
        """AWR actor update: log-likelihood weighted by ``exp(beta * adv)``."""
        action_probs, _ = self.actor_net(*state)
        action_dist = torch.distributions.Categorical(action_probs)

        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=100)
        bc_loss = -action_dist.log_prob(actions.squeeze(-1)).unsqueeze(-1)
        p_loss = (bc_loss * exp_adv).mean()
        self._optimize(self.actor_optimizer, p_loss, self.actor_net)
        log_dict["policy_loss"] = p_loss.item()
        entropy = action_dist.entropy().mean()
        log_dict["entropy"] = entropy.item()

    def _value_loss(self, state, actions, log_dict):
        """Expectile regression of V(s) toward the pessimistic target critic.

        Returns:
            The advantage ``min_i Q_i(s, a_data) - V(s)`` used by the actor.
        """
        with torch.no_grad():
            q_val, _ = self.target_net(*state)
            q_val = get_min_q(q_val).squeeze(-1)

            q_val_loss = q_val.gather(1, actions)
        val = self.value_net(*state)
        adv = q_val_loss - val

        val_loss = asymmetric_l2_loss(adv, self.iql_tau)
        self._optimize(self.value_optimizer, val_loss, self.value_net)
        log_dict["value_loss"] = val_loss.item()
        log_dict["adv"] = adv.mean().item()
        return adv

    def train(self, batch) -> dict:
        """One IQL step: value, critic and actor updates.

        Args:
            batch: ``(state, next_state, actions, rewards, dones, mc_returns)``
                as produced by ``Buffer.sample`` / ``Buffer.epoch_generator``.

        Returns:
            Dict of scalar training statistics.
        """
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
