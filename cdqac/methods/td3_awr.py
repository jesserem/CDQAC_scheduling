"""
TD3-AWR for the masked discrete FJSP action space (offline RL).

Combines:
  * TD3-AWR -- Jackson et al., "A Clean Slate for Offline RL" (arXiv:2504.11453):
    ReBRAC's critic (twin critics, target-policy smoothing, critic-side BC penalty
    on the *next* dataset action) with an IQL-style advantage-weighted regression
    (AWR) actor term:
      - value net V(s) trained by expectile regression toward min_i Q_i(s, a_data)
        using the ONLINE critics,
      - AWR weight  w = clip(exp(eta * (min_i Q_i(s, a_data) - V(s))), max=A_max),
      - critic target y = r + gamma * (1-d) * [min_i Q'_i(s', a~') - alpha_c * D(a~', a'_data)],
      - actor loss   L = beta_q * lambda * (-min_i Q_i(s, pi(s))) + beta_bc * w * D_bc,
        with lambda = 1 / mean|Q| (ReBRAC's Q-loss normalisation).
  * Eq. (18)/(20) of Fujimoto et al., MR.Q (arXiv:2501.16142) -- the unified TD3
    target-action rule for DISCRETE actions: clipped Gaussian noise is added to the
    target policy's (masked softmax) output, the feasibility-masked argmax is the
    target action; the actor-side DPG term goes through a straight-through
    Gumbel-softmax, plus a small pre-activation regulariser on the raw logits.

Networks are the repo's standard DAN-based ones (see train_td3_awr.py):
  actor_net / target_actor_net : ActorNet
  q_net / target_net           : QRDQNNet with num_quantiles=1 (list of per-critic
                                 Q(s, .) tensors [B, J*M, 1], -inf at masked pairs)
  value_net                    : IQL_value

Batch format (from Buffer.sample_td3 / Buffer.epoch_generator_td3):
  state, next_state, actions, next_actions, rewards, dones, mc_returns
where state is the usual 8-tuple and next_actions is the dataset action taken at
s' (arbitrary on terminal transitions -- the target is gated by done there).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u ** 2)


class TD3AWR:
    def __init__(
            self,
            actor_net: nn.Module,
            target_actor_net: nn.Module,
            q_net: nn.Module,
            target_net: nn.Module,
            value_net: nn.Module,
            actor_optimizer: torch.optim.Optimizer,
            q_optimizer: torch.optim.Optimizer,
            value_optimizer: torch.optim.Optimizer,
            target_update_freq: int = 1,
            update_freq_policy: int = 2,
            discount: float = 1.0,
            tau: float = 0.005,
            awr_temperature: float = 3.0,
            awr_adv_clip: float = 100.0,
            value_expectile: float = 0.7,
            beta_q: float = 1.0,
            beta_bc: float = 0.03,
            normalize_q_loss: bool = True,
            critic_bc_coef: float = 0.01,
            critic_bc_soft: bool = True,
            policy_noise: float = 0.2,
            noise_clip: float = 0.5,
            bc_distance: str = "ce",
            pre_activ_reg: float = 1e-5,
            gumbel_tau: float = 1.0,
            max_grad_norm: float = 1.0,
            device: str = "cpu",
    ):
        self.actor_net = actor_net
        self.target_actor_net = target_actor_net
        self.q_net = q_net
        self.target_net = target_net
        self.value_net = value_net

        self.actor_optimizer = actor_optimizer
        self.q_optimizer = q_optimizer
        self.value_optimizer = value_optimizer

        self.target_update_freq = target_update_freq
        self.update_freq_policy = update_freq_policy
        self.discount = discount
        self.tau = tau

        self.awr_temperature = awr_temperature
        self.awr_adv_clip = awr_adv_clip
        self.value_expectile = value_expectile
        self.beta_q = beta_q
        self.beta_bc = beta_bc
        self.normalize_q_loss = normalize_q_loss
        self.critic_bc_coef = critic_bc_coef
        self.critic_bc_soft = critic_bc_soft
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.bc_distance = bc_distance
        self.pre_activ_reg = pre_activ_reg
        self.gumbel_tau = gumbel_tau
        self.max_grad_norm = max_grad_norm

        self.device = device
        self.n_updates = 0

    # ------------------------------------------------------------------ #
    def _min_q_gather(self, q_net: nn.Module, state, actions: torch.Tensor) -> torch.Tensor:
        """min_i Q_i(s, a) for the given (feasible) dataset actions, shape [B, 1]."""
        q_list, _ = q_net(*state)
        q_a = torch.stack([q.squeeze(-1).gather(1, actions) for q in q_list], dim=0)
        return torch.min(q_a, dim=0).values

    def _actor_logits(self, state, mask_flat: torch.Tensor):
        """Raw pre-activations z_pi and feasibility-masked logits of the online actor."""
        candidate_feature = self.actor_net.get_embedding(
            state[0], state[1], state[2], state[3], state[4], state[5], state[7]
        )
        z_pre = self.actor_net.actor(candidate_feature).squeeze(-1)
        logits = z_pre.masked_fill(mask_flat, float("-inf"))
        return z_pre, logits

    # ------------------------------------------------------------------ #
    def _critic_loss(self, state, next_state, actions, next_actions, rewards, dones,
                     next_mask_flat, log_dict):
        with torch.no_grad():
            # MR.Q Eq. 18: a~' = argmax( pi'(s') + clip(eps, -c, c) ) over feasible pairs
            next_probs, _ = self.target_actor_net(*next_state)
            next_probs = torch.nan_to_num(next_probs, nan=0.0)
            eps = (torch.randn_like(next_probs) * self.policy_noise
                   ).clamp_(-self.noise_clip, self.noise_clip)
            noisy = (next_probs + eps).masked_fill(next_mask_flat, float("-inf"))
            next_a = noisy.argmax(dim=-1, keepdim=True)                     # [B, 1]

            next_q_list, _ = self.target_net(*next_state)
            next_q = torch.stack([q.squeeze(-1).gather(1, next_a) for q in next_q_list], dim=0)
            next_q = torch.min(next_q, dim=0).values                        # [B, 1]

            # ReBRAC critic-side BC penalty D(a~', a'_data)
            if self.critic_bc_soft:
                # ||pi'(s') - onehot(a'_data)||^2 = sum(pi^2) - 2 pi[a'_data] + 1
                p_a = next_probs.gather(1, next_actions)
                bc_pen = next_probs.pow(2).sum(dim=1, keepdim=True) - 2.0 * p_a + 1.0
            else:
                # ||onehot(a~') - onehot(a'_data)||^2 = 2 * disagreement
                bc_pen = 2.0 * (next_a != next_actions).float()

            target = rewards + (1.0 - dones) * self.discount * (
                next_q - self.critic_bc_coef * bc_pen)

        q_list, _ = self.q_net(*state)
        q_loss = 0
        for i, q in enumerate(q_list):
            q_a = q.squeeze(-1).gather(1, actions)
            log_dict["q_val_" + str(i)] = q_a.mean().item()
            q_loss = q_loss + F.mse_loss(q_a, target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.q_optimizer.step()

        log_dict["q_loss"] = q_loss.item()
        log_dict["mean_target"] = target.mean().item()
        log_dict["critic_bc_pen"] = bc_pen.mean().item()

    def _value_loss(self, state, q_data, log_dict):
        v = self.value_net(*state)
        adv_v = q_data - v
        value_loss = asymmetric_l2_loss(adv_v, self.value_expectile)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)
        self.value_optimizer.step()

        log_dict["value_loss"] = value_loss.item()
        log_dict["value_mean"] = v.mean().item()

    def _actor_loss(self, state, actions, q_data, mask_flat, log_dict):
        z_pre, logits = self._actor_logits(state, mask_flat)

        # DPG term through straight-through Gumbel-softmax (MR.Q Eq. 20).
        # Q(s, .) does not depend on the actor, so it enters as constants and the
        # gradient flows through the one-hot relaxation only.
        a_pi = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=True)
        with torch.no_grad():
            q_all_list, _ = self.q_net(*state)
            q_all = torch.min(torch.stack([q.squeeze(-1) for q in q_all_list], dim=0),
                              dim=0).values
            q_all = q_all.masked_fill(mask_flat, 0.0)
        q_pi = (a_pi * q_all).sum(dim=1, keepdim=True)
        if self.normalize_q_loss:
            lam = 1.0 / (q_pi.abs().mean().detach() + 1e-7)
        else:
            lam = 1.0
        q_loss = -(lam * q_pi).mean()

        # AWR weight w = clip(exp(eta * (min_i Q_i(s, a_data) - V(s))), A_max)
        with torch.no_grad():
            adv = q_data - self.value_net(*state)
            awr_w = torch.exp(self.awr_temperature * adv).clamp(max=self.awr_adv_clip)

        if self.bc_distance == "ce":
            log_pi = logits - torch.logsumexp(logits, dim=1, keepdim=True)
            bc_d = -log_pi.gather(1, actions)
        else:
            soft = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False)
            a_onehot = F.one_hot(actions.squeeze(-1), soft.shape[1]).float()
            bc_d = (soft - a_onehot).pow(2).sum(dim=1, keepdim=True)
        bc_loss = (awr_w * bc_d).mean()

        pre_reg = self.pre_activ_reg * z_pre.pow(2).mean()
        actor_loss = self.beta_q * q_loss + self.beta_bc * bc_loss + pre_reg

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
        self.actor_optimizer.step()

        log_dict["actor_loss"] = actor_loss.item()
        log_dict["actor_q"] = q_pi.mean().item()
        log_dict["bc_loss"] = bc_loss.item()
        log_dict["awr_weight_mean"] = awr_w.mean().item()
        log_dict["adv_mean"] = adv.mean().item()
        log_dict["pre_activ_reg"] = pre_reg.item()

    # ------------------------------------------------------------------ #
    def train(self, batch):
        self.n_updates += 1
        log_dict = {}
        state, next_state, actions, next_actions, rewards, dones, mc_returns = batch
        sz_b = actions.shape[0]
        mask_flat = state[6].reshape(sz_b, -1)             # True = infeasible
        next_mask_flat = next_state[6].reshape(sz_b, -1)

        # 1) critic update (ReBRAC target)
        self._critic_loss(state, next_state, actions, next_actions, rewards, dones,
                          next_mask_flat, log_dict)

        # 2) value update: expectile regression toward min_i Q_i(s, a_data) of the
        #    freshly-updated ONLINE critics (Unifloral)
        with torch.no_grad():
            q_data = self._min_q_gather(self.q_net, state, actions)
        self._value_loss(state, q_data, log_dict)

        # 3) delayed actor update (TD3-AWR)
        if self.n_updates % self.update_freq_policy == 0:
            self._actor_loss(state, actions, q_data, mask_flat, log_dict)
            self.update_actor_target()

        if self.n_updates % self.target_update_freq == 0:
            self.update_target()

        return log_dict

    # ------------------------------------------------------------------ #
    def update_target(self):
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def update_actor_target(self):
        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_dict(self):
        return {
            "actor_net": self.actor_net.state_dict(),
        }
