"""Behavioral cloning (imitation learning) baseline.

Supervised cross-entropy training of the actor on the dataset actions; no
critic, no target networks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from cdqac.methods.base import BaseMethod


def imitation_loss(logits: torch.Tensor, expert_actions: torch.Tensor) -> torch.Tensor:
    """Cross-entropy between the policy logits and the expert actions.

    Args:
        logits: Unnormalized action scores, shape ``(N, C)``.
        expert_actions: Dataset action indices, shape ``(N, 1)`` or ``(N,)``.

    Returns:
        Scalar cross-entropy loss.
    """
    expert_actions = expert_actions.squeeze(-1)      # (N,)
    return F.cross_entropy(logits, expert_actions.long())


class Imitation_Learning(BaseMethod):
    """Behavioral-cloning trainer: cross-entropy on the dataset actions."""

    def __init__(
            self,
            actor_net: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            N: int = 50,
            max_grad_norm: float = 1.0,
            device: str = "cpu",
    ):
        """
        Args:
            actor_net: Masked-softmax policy network; its second output are
                the logits used for the cross-entropy loss.
            actor_optimizer: Optimizer for ``actor_net``.
            N: Accepted for API compatibility (unused).
            max_grad_norm: Gradient-norm clip.
            device: Torch device string.
        """
        super().__init__(max_grad_norm=max_grad_norm, device=device)
        self.actor_net = actor_net
        self.actor_optimizer = actor_optimizer
        self.N = N

    def train(self, batch) -> dict:
        """One supervised update on the dataset actions.

        Args:
            batch: ``(state, next_state, actions, rewards, dones, mc_returns)``
                as produced by ``Buffer.sample`` / ``Buffer.epoch_generator``;
                only ``state`` and ``actions`` are used.

        Returns:
            Dict with the imitation loss.
        """
        self.n_updates += 1
        log_dict = {}
        state, next_state, actions, rewards, dones, mc_returns = batch

        _, logits = self.actor_net(*state)
        imitation_loss_value = imitation_loss(logits, actions)
        self._optimize(self.actor_optimizer, imitation_loss_value, self.actor_net)
        log_dict["imitation_loss"] = imitation_loss_value.item()

        return log_dict
