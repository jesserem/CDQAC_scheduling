from typing import List, Dict, Any, Tuple, Optional
import pyrallis
from dataclasses import asdict, dataclass
import uuid

@dataclass
class TrainConfig:
    # JSSP Environment

    device: str = "cuda"


    # Data collection
    use_baseline: bool = False
    data_path: str = "./dataset"
    train_instance: str = "SD1_train_10_5_1000.npy"
    train_instance_2: str = "SD1_train_10_5_250.npy"
    eval_instance: str = "SD1_10_5_val.npy"
    eval_instance_2: str = "SD1_10_5_val.npy"

    normalize: bool = False
    type_div: Optional[str] = "scale"
    eval_every_epoch: int = 1
    # noisy_prob: float = 0

    train_epochs: int = 1000
    do_epoch: bool = False

    # Network Config
    fea_j_input_dim: int = 10
    fea_m_input_dim: int = 8
    layer_fea_output_dim: List = (32, 8)
    num_heads_OAB: List = (4, 4)
    num_heads_MAB: List = (4, 4, )
    num_mlp_layers_actor: int = 3
    num_mlp_layers_critic: int = 3
    num_mlp_layers_value: int = 3
    hidden_dim_actor: int = 64
    hidden_dim_critic: int = 64
    hidden_dim_value: int = 64

    n_critics: int = 2
    layer_norm: bool = False
    dropout_prob_actor: float = 0
    dropout_prob_q: float = 0
    dropout_prob_value: float = 0

    use_mask: bool = True

    # DQN Config
    n_step: int = 1
    target_update_freq: int = 1
    tau: float = 0.005

    batch_size: int = 256

    eval_freq: int = 1000
    normalize_q: bool = False
    warming_up_steps: int = 100
    target_entropy: float = 0.25
    q_lr: float = 3e-4
    p_lr: float = 3e-4
    v_lr: float = 3e-4

    beta: float = 15
    iql_tau: float = 0.7

    anneal_entropy: bool = False
    reward_scaling: Optional[str] = None


    n_envs: int = 16

    gamma: float = 1


    backup_entropy: bool = False

    precollect_steps: int = 5000


    update_freq_policy: int = 3
    num_train_step_offline: int = 200_000
    num_train_step_online: int = 0
    online_buffer_size: int = 1
    # n_envs: int = 4
    seed: int = 1
    reward_scale: float = 1
    reward_bias: float = 0
    mixing_rate: float = 0.25
    anneal_lr: bool = False

    remove_duplicate: bool = True

    save_folder: Optional[str] = "checkpoints_iql"

    num_instances: Optional[int] = 50

    use_dispatching: bool = False

    use_ga_hof: bool = False
    n_ga_hof: Optional[int] = 25
    ga_hof_random: bool = False

    use_ga_pop: bool = False
    n_ga_pop: Optional[int] = None
    ga_pop_random: bool = False

    use_random: bool = False
    n_random: Optional[int] = 50


    # Wandb Config
    use_wandb: bool = False
    project: str = "CDQAC-baselines"
    group: str = "iql"
    name: str = "iql"

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"