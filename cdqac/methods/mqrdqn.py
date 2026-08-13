"""mQRDQN — masked Quantile-Regression DQN baseline.

Pure value-based baseline without a separate actor: the greedy (masked)
argmax over the Q-network is the policy.  The critic is a single quantile
network trained with a double-DQN-style target (online net picks the argmax,
target net evaluates it) plus an optional CQL penalty on the quantile means.
"""
import torch
import torch.nn as nn

from cdqac.methods.base import BaseMethod
from cdqac.methods.util import quantile_huber_loss


class mQRDQN(BaseMethod):
    """Masked QR-DQN trainer.

    Note:
        Unlike the actor-critic methods, ``q_net``/``target_net`` here return
        a single quantile tensor of shape ``[B, A, N]`` (no feature tuple),
        and only the distributional path is implemented: ``use_qrdqn`` must
        stay True.
    """

    def __init__(
            self,
            q_net: nn.Module,
            target_net: nn.Module,
            q_optimizer: torch.optim.Optimizer,
            target_update_freq: int,
            discount: float,
            tau: float,
            max_steps: int,
            use_calql: bool = False,
            use_cql: bool = True,
            cql_alpha: float = 1,
            cql_temp: float = 1,
            N: int = 50,
            max_grad_norm: float = 1.0,
            device: str = "cpu",
            anneal_lr: bool = True,
            max_steps_lr: int = 1e6,
            noise_level: float = 0.1,
            kappa: float = 1,
            backup_entropy: bool = False,
            use_qrdqn: bool = True,
    ):
        """
        Args:
            q_net: Quantile Q-network, returns ``[B, A, N]``.
            target_net: Target copy of ``q_net``.
            q_optimizer: Optimizer for ``q_net``.
            target_update_freq: Polyak-update the target every n steps.
            discount: Discount factor gamma.
            tau: Polyak averaging rate.
            max_steps: Accepted for API compatibility (unused).
            use_calql: Accepted for API compatibility (unused).
            use_cql: Add the CQL penalty to the critic loss.
            cql_alpha: CQL penalty strength.
            cql_temp: Accepted for API compatibility (unused).
            N: Number of quantiles.
            max_grad_norm: Gradient-norm clip.
            device: Torch device string.
            anneal_lr: Accepted for API compatibility (unused).
            max_steps_lr: Accepted for API compatibility (unused).
            noise_level: Accepted for API compatibility (unused).
            kappa: Accepted for API compatibility (unused).
            backup_entropy: Accepted for API compatibility (unused).
            use_qrdqn: Must be True; the scalar-DQN path is not implemented
                for this method.
        """
        super().__init__(discount=discount, tau=tau,
                         target_update_freq=target_update_freq,
                         max_grad_norm=max_grad_norm, device=device)
        self.q_net = q_net
        self.target_net = target_net
        self.q_optimizer = q_optimizer

        self.N = N
        self.use_qrdqn = use_qrdqn
        self.use_cql = use_cql
        self.use_calql = use_calql
        self.cql_alpha = cql_alpha
        self.cql_temp = cql_temp

        self.anneal_lr = anneal_lr
        self.noise_level = noise_level
        self.kappa = kappa
        self.backup_entropy = backup_entropy

    def calculate_qrdqn_loss(self, state, next_state, actions, rewards, dones, log_dict):
        """Quantile Huber TD loss with a double-DQN target and CQL penalty.

        The online network picks the greedy next action; the target network's
        quantiles at that action form the target distribution.
        """
        with torch.no_grad():
            nextQ = self.target_net(*next_state)
            nextQAction = self.q_net(*next_state).mean(dim=-1)
            maxNextQ = nextQAction.argmax(dim=1, keepdim=True)

            target_q = nextQ.gather(1, maxNextQ.unsqueeze(-1).expand(-1, -1, self.N)).squeeze(1)

            target_q = torch.nan_to_num(target_q, nan=0.0)

            target = rewards + (1.0 - dones) * self.discount * target_q

        currQ = self.q_net(*state)
        q_action = currQ.gather(1, actions.unsqueeze(-1).expand(-1, -1, self.N)).squeeze(1)
        q_loss = quantile_huber_loss(q_action, target, sum_over_quantiles=True)
        log_dict["td_error_q"] = q_loss.mean().item()
        if self.use_cql:
            currQ = currQ.mean(-1)
            q_action = q_action.mean(-1)

            cql_loss_q = self.cql_alpha * (torch.logsumexp(currQ, dim=1) - q_action).mean()

            q_loss = q_loss + cql_loss_q
            log_dict["cql_loss"] = cql_loss_q.mean().item()

        log_dict["mean_target_q"] = target_q.mean().item()
        log_dict["mean_target"] = target.mean().item()
        return q_loss

    def train(self, batch) -> dict:
        """One QR-DQN critic update.

        Args:
            batch: ``(state, next_state, actions, rewards, dones, mc_returns)``
                as produced by ``Buffer.sample`` / ``Buffer.epoch_generator``.

        Returns:
            Dict of scalar training statistics.
        """
        self.n_updates += 1
        log_dict = {}
        state, next_state, actions, rewards, dones, mc_returns = batch

        if not self.use_qrdqn:
            raise NotImplementedError(
                "mQRDQN only implements the distributional (use_qrdqn=True) critic.")
        q_loss = self.calculate_qrdqn_loss(state, next_state, actions, rewards, dones, log_dict)

        self._optimize(self.q_optimizer, q_loss, self.q_net)

        log_dict["q_loss"] = q_loss.item()
        if self.n_updates % self.target_update_freq == 0:
            self.update_target()
        return log_dict

    def get_dict(self) -> dict:
        """Checkpoint dict; the greedy Q-network doubles as the policy, so it
        is stored under the "actor_net" key the evaluation code reads."""
        return {
            "actor_net": self.q_net.state_dict(),
        }
