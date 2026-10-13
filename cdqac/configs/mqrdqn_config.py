"""Training configuration for Offline-LD's maskable QR-DQN (mQRDQN).

Shared settings (dataset, dataset sources, encoder, buffer, evaluation) come from
base_config.py with CDQAC's defaults; this file holds mQRDQN's own. mQRDQN has no actor:
its greedy Q-network is the policy, and that network's MLP uses the actor-head sizes of the
base config (num_mlp_layers_actor, hidden_dim_actor).
"""
from typing import Optional
from dataclasses import dataclass

from cdqac.configs.base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    # Q-network
    num_quantiles: int = 64
    layer_norm: bool = False
    dropout_prob_q: float = 0

    # Ablation switches
    use_cql: bool = True
    use_adv_net: bool = False
    use_qrdqn: bool = True
    use_calql: bool = False

    # Optimization
    q_lr: float = 3e-4
    target_update_freq: int = 1
    tau: float = 0.005
    cql_alpha_offline: float = 0.05
    kappa: float = 1
    anneal_lr: bool = False

    # Output / logging
    save_folder: Optional[str] = "checkpoints_mqrdqn"
    project: str = "CDQAC-baselines"
    group: str = "mqrdqn"
    name: str = "mqrdqn"
