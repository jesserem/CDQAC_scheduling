"""Networks for routing-type problems with the same interfaces as cdqac.network.

The encoder follows the "light encoder, heavy decoder" idea of LEHD (Luo et al.,
NeurIPS 2023), which works well for constructive routing: a light encoder
embeds all nodes of the instance once, then a heavier decoder re-embeds only
the *open* stops together with a context token (current node, depot, global
features) at every step, conditioned on current->node relational features.
Optional zone pooling adds the mean embedding of the open stops of the same
zone to every stop (drivers route zone by zone in both datasets).

Every class mirrors its FJSP counterpart in ``cdqac/network/main_model.py``
(``ActorNet``, ``QRDQNNet``, ``IQL_value``, ``mQRDQNNet``): same forward
signature (the 8-tuple state of :mod:`realworld.routing.core` unpacked), same
outputs, same masking conventions (-inf at infeasible actions) and the same
``get_embedding`` / ``actor`` / ``get_action`` members, so all methods in
``cdqac.methods`` (CDQAC, IQL, TD3-AWR, Offline-LD, BC) train them unchanged.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from cdqac.network.sub_layers import (Actor, Critic, QRDQN_advantage_single,
                                      QRDQN_single)


class TransformerBlock(nn.Module):
    """Pre-LayerNorm transformer block with key-padding mask."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model))

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + a
        return x + self.ff(self.ln2(x))


class ZonePool(nn.Module):
    """h_i += W * mean{h_j : j open, zone_j == zone_i} (no-op for zone -1)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, zone: torch.Tensor, member: torch.Tensor) -> torch.Tensor:
        B, N, d = h.shape
        has_zone = zone >= 0
        idx = (zone + 1).clamp(min=0, max=N)
        w = (member & has_zone).float()
        sums = h.new_zeros(B, N + 1, d).scatter_add(1, idx[..., None].expand(-1, -1, d), h * w[..., None])
        cnt = h.new_zeros(B, N + 1).scatter_add(1, idx, w)
        mean = sums / cnt.clamp(min=1.0)[..., None]
        pooled = mean.gather(1, idx[..., None].expand(-1, -1, d)) * has_zone[..., None]
        return h + self.proj(pooled)


class RoutingEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, global_dim: int, d_model: int = 128,
                 n_heads: int = 8, n_enc_layers: int = 1, n_dec_layers: int = 3, ff_dim: int = 256,
                 zone_pool: bool = True, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.node_in = nn.Linear(node_dim, d_model)
        self.edge_in = nn.Linear(edge_dim, d_model)
        self.glob_in = nn.Linear(global_dim, d_model)
        self.cur_in = nn.Linear(d_model, d_model, bias=False)
        self.depot_in = nn.Linear(d_model, d_model, bias=False)
        self.enc = nn.ModuleList([TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_enc_layers)])
        self.dec = nn.ModuleList([TransformerBlock(d_model, n_heads, ff_dim, dropout) for _ in range(n_dec_layers)])
        self.zone_pools = nn.ModuleList([ZonePool(d_model) for _ in range(n_dec_layers)]) if zone_pool else None
        self.out_ln = nn.LayerNorm(d_model)

    def forward(self, node_feat, node_valid, current, global_feat, node_open, zone, edge_feat):
        """Returns per-node embeddings [B, N, d] and a global embedding [B, d]."""
        x = self.node_in(node_feat)
        for blk in self.enc:
            x = blk(x, ~node_valid)
        B, N, d = x.shape
        cur = x.gather(1, current.view(-1, 1, 1).expand(-1, 1, d)).squeeze(1)
        ctx = self.glob_in(global_feat) + self.cur_in(cur) + self.depot_in(x[:, 0])
        tokens = torch.cat([ctx[:, None], x + self.edge_in(edge_feat)], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~node_open], 1)
        for i, blk in enumerate(self.dec):
            tokens = blk(tokens, pad)
            if self.zone_pools is not None:
                nodes = self.zone_pools[i](tokens[:, 1:], zone, node_open)
                tokens = torch.cat([tokens[:, :1], nodes], 1)
        tokens = self.out_ln(tokens)
        return tokens[:, 1:], tokens[:, 0]


def _encoder_kwargs(kw: dict) -> dict:
    keys = ("node_dim", "edge_dim", "global_dim", "d_model", "n_heads", "n_enc_layers", "n_dec_layers",
            "ff_dim", "zone_pool")
    return {k: kw[k] for k in keys if k in kw}


class RoutingActorNet(nn.Module):
    """Masked-softmax policy over the next stop (counterpart of ``ActorNet``)."""

    def __init__(self, num_mlp_layers_actor: int = 3, hidden_dim_actor: int = 64,
                 dropout_prob: float = 0.0, **encoder_kw):
        super().__init__()
        self.feature_exact = RoutingEncoder(**_encoder_kwargs(encoder_kw))
        d = self.feature_exact.d_model
        self.actor = Actor(num_mlp_layers_actor, 2 * d, hidden_dim_actor, 1, dropout=dropout_prob)

    def get_embedding(self, node_feat, node_valid, current, global_feat, node_open, zone, edge_feat):
        hn, g = self.feature_exact(node_feat, node_valid, current, global_feat, node_open, zone, edge_feat)
        return torch.cat([hn, g[:, None].expand_as(hn)], -1)

    def forward(self, node_feat, node_valid, current, global_feat, node_open, zone, action_mask, edge_feat):
        cand = self.get_embedding(node_feat, node_valid, current, global_feat, node_open, zone, edge_feat)
        scores = self.actor(cand).squeeze(-1).masked_fill(action_mask, float("-inf"))
        log_pi = scores - torch.logsumexp(scores, dim=1, keepdim=True)
        return F.softmax(scores, dim=1), log_pi

    @torch.no_grad()
    def get_action(self, *state, deterministic: bool = True):
        pi, _ = self(*state)
        if deterministic:
            return pi.argmax(dim=1)
        return Categorical(pi).sample()


class RoutingQRDQNNet(nn.Module):
    """Ensemble of (dueling) quantile critics (counterpart of ``QRDQNNet``).

    Returns ``(list of [B, N, num_quantiles] tensors, (node_emb, global_emb))``
    with -inf at infeasible actions.
    """

    def __init__(self, num_mlp_layers_critic: int = 3, hidden_dim_critic: int = 64,
                 num_quantiles: int = 64, dropout_prob_q: float = 0.0, use_adv_net: bool = False,
                 n_critics: int = 2, layer_norm: bool = False, **encoder_kw):
        super().__init__()
        self.feature_exact = RoutingEncoder(**_encoder_kwargs(encoder_kw))
        d = self.feature_exact.d_model
        self.num_quantiles = num_quantiles
        self.use_adv_net = use_adv_net
        if use_adv_net:
            heads = [QRDQN_advantage_single(num_mlp_layers_critic, 2 * d, d, hidden_dim_critic, num_quantiles,
                                            dropout=dropout_prob_q, use_layer_norm=layer_norm)
                     for _ in range(n_critics)]
        else:
            heads = [QRDQN_single(num_mlp_layers_critic, 2 * d, hidden_dim_critic, num_quantiles,
                                  dropout=dropout_prob_q, use_layer_norm=layer_norm)
                     for _ in range(n_critics)]
        self.Q = nn.ModuleList(heads)

    def get_embedding(self, node_feat, node_valid, current, global_feat, node_open, zone, edge_feat):
        hn, g = self.feature_exact(node_feat, node_valid, current, global_feat, node_open, zone, edge_feat)
        return torch.cat([hn, g[:, None].expand_as(hn)], -1), g, (hn, g)

    def forward(self, node_feat, node_valid, current, global_feat, node_open, zone, action_mask, edge_feat):
        cand, g, feats = self.get_embedding(node_feat, node_valid, current, global_feat, node_open, zone,
                                            edge_feat)
        B = cand.shape[0]
        mask3 = action_mask.unsqueeze(-1)  # [B, N, 1]: per-state action counts in the dueling head
        q_val = [head(cand, g, mask3, B) for head in self.Q]
        q_val = [q.masked_fill(mask3.expand_as(q), float("-inf")) for q in q_val]
        return q_val, feats


class RoutingValueNet(nn.Module):
    """State-value network V(s) (counterpart of ``IQL_value``)."""

    def __init__(self, num_mlp_layers_critic: int = 3, hidden_dim_critic: int = 64, **encoder_kw):
        super().__init__()
        self.feature_exact = RoutingEncoder(**_encoder_kwargs(encoder_kw))
        self.critic = Critic(num_mlp_layers_critic, self.feature_exact.d_model, hidden_dim_critic, 1)

    def get_embedding(self, node_feat, node_valid, current, global_feat, node_open, zone, edge_feat):
        _, g = self.feature_exact(node_feat, node_valid, current, global_feat, node_open, zone, edge_feat)
        return g

    def forward(self, node_feat, node_valid, current, global_feat, node_open, zone, action_mask, edge_feat):
        return self.critic(self.get_embedding(node_feat, node_valid, current, global_feat, node_open, zone,
                                              edge_feat))


class RoutingMQRDQNNet(nn.Module):
    """Single quantile critic whose greedy argmax is the policy (``mQRDQNNet``)."""

    def __init__(self, num_mlp_layers_critic: int = 3, hidden_dim_critic: int = 64,
                 num_quantiles: int = 64, dropout_prob_q: float = 0.0, use_adv_net: bool = False,
                 layer_norm: bool = False, **encoder_kw):
        super().__init__()
        self.feature_exact = RoutingEncoder(**_encoder_kwargs(encoder_kw))
        d = self.feature_exact.d_model
        self.num_quantiles = num_quantiles
        if use_adv_net:
            self.Q = QRDQN_advantage_single(num_mlp_layers_critic, 2 * d, d, hidden_dim_critic, num_quantiles,
                                            dropout=dropout_prob_q, use_layer_norm=layer_norm)
        else:
            self.Q = QRDQN_single(num_mlp_layers_critic, 2 * d, hidden_dim_critic, num_quantiles,
                                  dropout=dropout_prob_q, use_layer_norm=layer_norm)

    def forward(self, node_feat, node_valid, current, global_feat, node_open, zone, action_mask, edge_feat):
        hn, g = self.feature_exact(node_feat, node_valid, current, global_feat, node_open, zone, edge_feat)
        cand = torch.cat([hn, g[:, None].expand_as(hn)], -1)
        mask3 = action_mask.unsqueeze(-1)
        q = self.Q(cand, g, mask3, cand.shape[0])
        return q.masked_fill(mask3.expand_as(q), float("-inf"))

    @torch.no_grad()
    def get_action(self, *state, deterministic: bool = True):
        q = self(*state).mean(-1)
        if deterministic:
            return q.argmax(dim=1)
        return Categorical(F.softmax(q, dim=1)).sample()
