from typing import List, Optional
from dataclasses import dataclass
import uuid


@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 1

    # Data
    data_path: str = "./train_dataset/fjsp"
    train_instance: str = "train_10_5.npy"
    eval_instance: str = "val_10_5.npy"
    num_instances: Optional[int] = 500
    remove_duplicate: bool = True

    # Dataset sources
    use_dispatching: bool = True
    use_ga_hof: bool = True
    n_ga_hof: Optional[int] = None
    ga_hof_random: bool = False
    use_ga_pop: bool = False
    n_ga_pop: Optional[int] = None
    ga_pop_random: bool = False
    use_random: bool = False
    n_random: Optional[int] = 50

    # Network Config (Q-network only; the greedy Q-network is the policy)
    fea_j_input_dim: int = 10
    fea_m_input_dim: int = 8
    layer_fea_output_dim: List = (32, 8)
    num_heads_OAB: List = (4, 4)
    num_heads_MAB: List = (4, 4)
    num_mlp_layers_actor: int = 3
    hidden_dim_actor: int = 64
    num_quantiles: int = 64
    layer_norm: bool = False
    dropout_prob_q: float = 0
    use_mask: bool = True

    # Ablation switches
    use_cql: bool = True
    use_adv_net: bool = False
    use_qrdqn: bool = True
    use_calql: bool = False

    # Optimization
    num_train_step_offline: int = 200_000
    batch_size: int = 256
    q_lr: float = 3e-4
    target_update_freq: int = 1
    tau: float = 0.005
    cql_alpha_offline: float = 0.05
    gamma: float = 1
    kappa: float = 1
    anneal_lr: bool = False
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
    save_folder: Optional[str] = "checkpoints_mqrdqn"

    # Wandb Config
    use_wandb: bool = False
    project: str = "CDQAC-baselines"
    group: str = "mqrdqn"
    name: str = "mqrdqn"

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
