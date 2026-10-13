"""Training configuration for CDQAC (Conservative Discrete Quantile Actor-Critic).

Defaults correspond to the hyperparameters reported in Table 11 of the paper. The settings
shared with the baselines (dataset, dataset sources, encoder, buffer, evaluation) live in
base_config.py; this file holds CDQAC's own. Every field can be overridden on the command
line via pyrallis, e.g.:

    python train.py --dataset fjsp/sd1_15x10 --seed 2
"""
from typing import Optional
from dataclasses import dataclass

from cdqac.configs.base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    # --- Critic (dueling quantile critic) -------------------------------------
    num_mlp_layers_critic: int = 3
    hidden_dim_critic: int = 64
    num_quantiles: int = 64             # N in Eq. 1
    n_critics: int = 2                  # twin critic heads, min over expected values
    layer_norm: bool = False
    dropout_prob_q: float = 0

    # --- CDQAC ablation switches (Sect. 5.5) ----------------------------------
    use_cql: bool = True                # CQL penalty (Eq. 4)
    use_adv_net: bool = True            # dueling architecture (Eq. 6)
    use_qrdqn: bool = True              # quantile critic (False -> scalar DQN critic)
    use_codac: bool = False             # CODAC distributional penalty instead of CQL (App. K.3)
    use_calql: bool = False

    # --- Optimization (Table 11) ----------------------------------------------
    q_lr: float = 2e-4                  # critic learning rate
    p_lr: float = 2e-5                  # policy learning rate
    update_freq_policy: int = 4         # policy delay eta
    target_update_freq: int = 1
    tau: float = 0.005                  # Polyak averaging rate rho
    cql_alpha_offline: float = 0.05     # CQL strength alpha_CQL
    entropy_coef: float = 0.001
    kappa: float = 1                    # quantile Huber loss threshold
    max_grad_norm: Optional[float] = 1
    q_pretrain_steps: int = 0
    normalize_q: bool = False
    target_entropy: float = 0.3         # only used when alpha_multiplier > 0
    alpha_multiplier: float = 0
    anneal_entropy: bool = False
    anneal_lr: bool = False
    backup_entropy: bool = False
