from typing import List, Optional
from dataclasses import dataclass
import uuid


@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 1

    # Data
    data_path: str = "./train_datasets/fjsp"
    train_instance: str = "train_10_5.npy"
    eval_instance: str = "val_10_5.npy"
    num_instances: Optional[int] = 50
    remove_duplicate: bool = True

    # Dataset sources
    use_dispatching: bool = False
    use_ga_hof: bool = False
    n_ga_hof: Optional[int] = 25
    ga_hof_random: bool = False
    use_ga_pop: bool = False
    n_ga_pop: Optional[int] = None
    ga_pop_random: bool = False
    use_random: bool = True
    n_random: Optional[int] = 50

    # Network Config
    fea_j_input_dim: int = 10
    fea_m_input_dim: int = 8
    layer_fea_output_dim: List = (32, 8)
    num_heads_OAB: List = (4, 4)
    num_heads_MAB: List = (4, 4)
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
    use_mask: bool = True

    # IQL
    beta: float = 15
    iql_tau: float = 0.7
    mixing_rate: float = 0.25

    # Optimization
    num_train_step_offline: int = 200_000
    batch_size: int = 256
    q_lr: float = 3e-4
    p_lr: float = 3e-4
    v_lr: float = 3e-4
    update_freq_policy: int = 3
    target_update_freq: int = 1
    tau: float = 0.005
    gamma: float = 1
    reward_scale: float = 1
    reward_bias: float = 0
    reward_scaling: Optional[str] = None

    # Disk-backed buffer / DataLoader (low-RAM dataset pipeline)
    use_disk_buffer: bool = False
    buffer_dir: Optional[str] = None
    cleanup_buffer_files: bool = True
    num_workers: int = 8
    prefetch_factor: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    sampler_chunk_len: int = 1

    # Evaluation / checkpointing
    eval_freq: int = 1000
    eval_every_epoch: int = 1
    do_epoch: bool = False
    train_epochs: int = 1000
    save_folder: Optional[str] = "checkpoints_iql"

    # Wandb Config
    use_wandb: bool = False
    project: str = "CDQAC-baselines"
    group: str = "iql"
    name: str = "iql"

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
