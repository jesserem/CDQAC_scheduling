"""Network factory for the real-world routing datasets.

Plugs into the ``build_*`` functions of ``train.py`` (``nets=RoutingNets``), so
CDQAC, IQL, TD3-AWR, Offline-LD (d-mSAC / mQRDQN) and BC are built exactly as
for FJSP, only with the routing encoder and heads.
"""
from __future__ import annotations

from realworld.routing.network import (RoutingActorNet, RoutingMQRDQNNet, RoutingQRDQNNet,
                                       RoutingValueNet)


def encoder_kwargs(config) -> dict:
    """Encoder arguments from a real-world TrainConfig (dims are set by the data)."""
    return dict(node_dim=config.node_dim, edge_dim=config.edge_dim, global_dim=config.global_dim,
                d_model=config.d_model, n_heads=config.n_heads, n_enc_layers=config.n_enc_layers,
                n_dec_layers=config.n_dec_layers, ff_dim=config.ff_dim, zone_pool=config.zone_pool)


class RoutingNets:
    @staticmethod
    def actor(config) -> RoutingActorNet:
        return RoutingActorNet(num_mlp_layers_actor=config.num_mlp_layers_actor,
                               hidden_dim_actor=config.hidden_dim_actor,
                               dropout_prob=config.dropout_prob_actor,
                               **encoder_kwargs(config)).to(config.device)

    @staticmethod
    def qrdqn(config, num_quantiles: int, use_adv_net: bool) -> RoutingQRDQNNet:
        return RoutingQRDQNNet(num_mlp_layers_critic=config.num_mlp_layers_critic,
                               hidden_dim_critic=config.hidden_dim_critic,
                               num_quantiles=num_quantiles, use_adv_net=use_adv_net,
                               dropout_prob_q=config.dropout_prob_q, n_critics=config.n_critics,
                               layer_norm=config.layer_norm, **encoder_kwargs(config)).to(config.device)

    @staticmethod
    def value(config) -> RoutingValueNet:
        return RoutingValueNet(num_mlp_layers_critic=config.num_mlp_layers_value,
                               hidden_dim_critic=config.hidden_dim_value,
                               **encoder_kwargs(config)).to(config.device)

    @staticmethod
    def mqrdqn(config) -> RoutingMQRDQNNet:
        return RoutingMQRDQNNet(num_mlp_layers_critic=config.num_mlp_layers_actor,
                                hidden_dim_critic=config.hidden_dim_actor,
                                num_quantiles=config.num_quantiles, use_adv_net=config.use_adv_net,
                                dropout_prob_q=config.dropout_prob_q, layer_norm=config.layer_norm,
                                **encoder_kwargs(config)).to(config.device)


def build_eval_net(config, method: str):
    """Network that ``get_action``s for a checkpoint of ``method`` (actor or greedy Q)."""
    return RoutingNets.mqrdqn(config) if method == "mqrdqn" else RoutingNets.actor(config)
