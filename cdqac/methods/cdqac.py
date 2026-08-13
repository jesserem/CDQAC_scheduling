"""CDQAC — Conservative Discrete Quantile Actor-Critic.

The critic machinery (quantile critic ensemble, CQL/CODAC penalties,
pessimistic Q extraction, schedules) lives in
:class:`cdqac.methods.base.QuantileActorCriticBase`; this module adds the
CDQAC-specific actor update: a delayed policy step that maximizes the
probability-weighted (optionally centered) Q-values plus a fixed entropy
bonus ``entropy_coef``.
"""
import torch
import torch.nn as nn

from cdqac.methods.base import QuantileActorCriticBase


class CDQAC(QuantileActorCriticBase):
    """Conservative Discrete Quantile Actor-Critic trainer.

    Differences to :class:`cdqac.methods.d_msac.discrete_mSAC`:

    * the actor is updated every ``update_freq_policy`` critic steps and only
      after ``q_pretrain_steps``,
    * the entropy bonus uses the fixed coefficient ``entropy_coef`` instead of
      a learned SAC temperature.
    """

    def __init__(self, *args, use_codac: bool = False,
                 entropy_coef: float = 0.001, **kwargs):
        """See :class:`QuantileActorCriticBase` for the shared arguments.

        Args:
            use_codac: Use the CODAC distributional penalty instead of CQL in
                the QR-DQN critic loss.
            entropy_coef: Fixed entropy-bonus coefficient in the actor loss
                (lambda in Eq. 5 of the paper).
        """
        super().__init__(*args, use_codac=use_codac, **kwargs)
        self.entropy_coef = entropy_coef

    def train(self, batch) -> dict:
        """One critic update and (every ``update_freq_policy`` steps, after
        critic pretraining) one delayed actor update.

        Args:
            batch: ``(state, next_state, actions, rewards, dones, mc_returns)``
                as produced by ``Buffer.sample`` / ``Buffer.epoch_generator``.

        Returns:
            Dict of scalar training statistics.
        """
        self.n_updates += 1
        log_dict = {}
        state, next_state, actions, rewards, dones, mc_returns = batch

        if self.use_qrdqn:
            q_loss = self.calculate_qrdqn_loss(state, next_state, actions, rewards, dones, log_dict)
        else:
            q_loss = self.calculate_dqn_loss(state, next_state, actions, rewards, dones, log_dict)

        self._optimize(self.q_optimizer, q_loss, self.q_net)

        log_dict["q_loss"] = q_loss.item()

        if self.n_updates % self.update_freq_policy == 0 and self.n_updates > self.q_pretrain_steps:
            self.n_updates_policy += 1
            q_val = self.get_q_values(state)

            probs, log_probs = self.actor_net(*state)
            action_dist = torch.distributions.Categorical(probs)
            entropy = action_dist.entropy()

            if self.normalize_q:
                mean_q = torch.nanmean(q_val.detach(), dim=-1, keepdim=True)
                q_val = (q_val - mean_q)
                log_dict["mean_q"] = mean_q.mean().item()

            q_val = torch.nan_to_num(q_val, nan=0.0)

            rl_loss = -(self.entropy_coef * entropy + (probs * q_val).sum(-1))

            policy_loss = rl_loss.mean()

            self._optimize(self.actor_optimizer, policy_loss, self.actor_net)

            log_dict["policy_loss"] = policy_loss.item()

            log_dict["rl_loss"] = rl_loss.mean().item()
            log_dict["rl_loss_min"] = rl_loss.min().item()
            log_dict["rl_loss_max"] = rl_loss.max().item()
            log_dict["entropy_min"] = entropy.min().item()
            log_dict["entropy"] = entropy.mean().item()
            log_dict["entropy_max"] = entropy.max().item()

        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            log_dict["actor_lr"] = self.lr_scheduler.get_last_lr()[0]

        return log_dict

    def get_dict_with_critic(self) -> dict:
        """Checkpoint dict that additionally stores the critic and its target.

        Backwards compatible with :meth:`get_dict`: the extra keys are ignored
        by evaluation code that only reads ``"actor_net"``.
        """
        return {
            "actor_net": self.actor_net.state_dict(),
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
        }
