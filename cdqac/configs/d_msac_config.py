"""Training configuration for Offline-LD's discrete maskable SAC (d-mSAC).

Shared settings (dataset, dataset sources, encoder, buffer, evaluation) come from
base_config.py with CDQAC's defaults; this file holds d-mSAC's own.
"""
from typing import Optional
from dataclasses import dataclass

from cdqac.configs.base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    # Critic
    num_mlp_layers_critic: int = 3
    hidden_dim_critic: int = 64
    num_quantiles: int = 64
    n_critics: int = 2
    layer_norm: bool = False
    dropout_prob_q: float = 0

    # Ablation switches
    use_cql: bool = True
    use_adv_net: bool = False
    use_qrdqn: bool = False
    use_calql: bool = False

    # Optimization
    q_lr: float = 3e-4
    p_lr: float = 3e-5
    update_freq_policy: int = 4
    target_update_freq: int = 1
    tau: float = 0.005
    cql_alpha_offline: float = 0.1
    alpha_multiplier: float = 0.1
    target_entropy: float = 0.4
    anneal_entropy: bool = False
    anneal_lr: bool = False
    backup_entropy: bool = False
    kappa: float = 1
    max_grad_norm: Optional[float] = 1
    q_pretrain_steps: int = 0
    normalize_q: bool = False

    # Output / logging
    save_folder: Optional[str] = "checkpoints_dmsac"
    project: str = "CDQAC-baselines"
    group: str = "d_msac"
    name: str = "d_msac"
