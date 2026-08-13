"""Discrete masked SAC (offline variant) for the FJSP action space.

Shares its whole critic side (quantile critic ensemble, CQL penalty,
pessimistic Q extraction, schedules) with CDQAC through
:class:`cdqac.methods.base.QuantileActorCriticBase` and differs only in the
actor update: the policy is updated every step with a SAC-style entropy bonus
whose temperature ``alpha`` is learned against an (optionally annealed)
target entropy.
"""
import torch

from cdqac.methods.base import QuantileActorCriticBase


class discrete_mSAC(QuantileActorCriticBase):
    """Discrete masked soft actor-critic trainer.

    See :class:`QuantileActorCriticBase` for the constructor arguments; this
    class adds no parameters of its own.
    """

    def train(self, batch) -> dict:
        """One critic update plus one SAC actor/temperature update.

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

        self.n_updates_policy += 1
        q_val = self.get_q_values(state)

        probs, log_probs = self.actor_net(*state)
        action_dist = torch.distributions.Categorical(probs)
        entropy = action_dist.entropy()

        curr_target_entropy = max(self.end_target_entropy, self.target_ent_func(self.n_updates - 1))
        log_dict["curr_target_entropy"] = curr_target_entropy
        # state[6] holds the feasibility mask of the batch.
        alpha, alpha_loss = self._alpha_and_alpha_loss(entropy.detach(), state[6], curr_target_entropy)

        q_val = torch.nan_to_num(q_val, nan=0.0)

        rl_loss = -(alpha * entropy + (probs * q_val).sum(-1))

        policy_loss = rl_loss.mean()

        self._optimize(self.actor_optimizer, policy_loss, self.actor_net)

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        log_dict["policy_loss"] = policy_loss.item()

        log_dict["rl_loss"] = rl_loss.mean().item()
        log_dict["rl_loss_min"] = rl_loss.min().item()
        log_dict["rl_loss_max"] = rl_loss.max().item()
        log_dict["alpha"] = alpha.item()
        log_dict["alpha_loss"] = alpha_loss.item()
        log_dict["entropy_min"] = entropy.min().item()
        log_dict["entropy"] = entropy.mean().item()
        log_dict["entropy_max"] = entropy.max().item()
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
            log_dict["actor_lr"] = self.lr_scheduler.get_last_lr()[0]

        return log_dict
