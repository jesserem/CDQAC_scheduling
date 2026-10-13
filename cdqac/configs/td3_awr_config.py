"""Training configuration for TD3-AWR.

Shared settings (dataset, dataset sources, encoder, buffer, evaluation) come from
base_config.py with CDQAC's defaults; this file holds TD3-AWR's own.
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
    layer_norm: bool = True          # ReBRAC: LayerNorm in the critic MLPs
    dropout_prob_q: float = 0

    # Target network / optimisation
    target_update_freq: int = 1
    tau: float = 0.005
    q_lr: float = 3e-4
    p_lr: float = 3e-4
    v_lr: float = 3e-4
    max_grad_norm: float = 1.0
    update_freq_policy: int = 4      # TD3 delayed actor update

    # --- TD3-AWR (arXiv:2504.11453) ---
    beta_q: float = 1.0              # Q-term coefficient in the actor loss
    beta_bc: float = 0.03            # AWR-weighted BC coefficient; tune in [5e-4, 1.0]
    normalize_q_loss: bool = True    # ReBRAC: scale the Q term by 1/mean|Q|
    awr_temperature: float = 10.0     # eta; tuned over {0.5, 3.0, 10.0}
    awr_adv_clip: float = 100.0      # A_max
    value_expectile: float = 0.7     # tuned over {0.5, 0.7, 0.9}
    bc_distance: str = "ce"          # "ce" (log-prob form) | "mse" (one-hot MSE)
    critic_bc_coef: float = 0.01     # alpha_c; tune in [0, 0.1]
    critic_bc_soft: bool = True      # penalty on target-policy probs vs hard disagreement

    # --- MR.Q discrete-action rules (arXiv:2501.16142, Eq. 18/20) ---
    policy_noise: float = 0.2        # sigma (action range is 1 for one-hot)
    noise_clip: float = 0.5          # c
    pre_activ_reg: float = 1e-5      # lambda_pre-activ on the raw actor logits
    gumbel_tau: float = 1.0          # Gumbel-softmax temperature

    # Output / logging
    save_folder: Optional[str] = "checkpoints_td3_awr"
    project: str = "CDQAC-baselines"
    group: str = "td3_awr"
    name: str = "td3_awr"
