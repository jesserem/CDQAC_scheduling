"""Training configuration for Implicit Q-Learning (IQL).

Shared settings (dataset, dataset sources, encoder, buffer, evaluation) come from
base_config.py with CDQAC's defaults; this file holds IQL's own.
"""
from typing import Optional
from dataclasses import dataclass

from cdqac.configs.base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    # Critic and value network
    num_mlp_layers_critic: int = 3
    num_mlp_layers_value: int = 3
    hidden_dim_critic: int = 64
    hidden_dim_value: int = 64
    n_critics: int = 2
    layer_norm: bool = False
    dropout_prob_q: float = 0

    # IQL
    beta: float = 15
    iql_tau: float = 0.7
    mixing_rate: float = 0.25

    # Optimization
    q_lr: float = 3e-4
    p_lr: float = 3e-4
    v_lr: float = 3e-4
    update_freq_policy: int = 3
    target_update_freq: int = 1
    tau: float = 0.005

    # Output / logging
    save_folder: Optional[str] = "checkpoints_iql"
    project: str = "CDQAC-baselines"
    group: str = "iql"
    name: str = "iql"
