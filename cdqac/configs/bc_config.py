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
    num_instances: Optional[int] = 1
    normalize: bool = False
    remove_duplicate: bool = True

    # Dataset sources
    use_dispatching: bool = True
    use_ga_hof: bool = False
    n_ga_hof: Optional[int] = None
    ga_hof_random: bool = False
    use_ga_pop: bool = True
    n_ga_pop: Optional[int] = None
    ga_pop_random: bool = False
    use_random: bool = False
    n_random: Optional[int] = 100

    # Network Config (actor only)
    fea_j_input_dim: int = 10
    fea_m_input_dim: int = 8
    layer_fea_output_dim: List = (32, 8)
    num_heads_OAB: List = (4, 4)
    num_heads_MAB: List = (4, 4)
    num_mlp_layers_actor: int = 3
    hidden_dim_actor: int = 64
    dropout_prob_actor: float = 0
    use_mask: bool = True

    # Optimization
    num_train_step_offline: int = 200_000
    batch_size: int = 256
    p_lr: float = 3e-5
    n_step: int = 1
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
    train_epochs: int = 59
    save_folder: Optional[str] = "checkpoints_bc"

    # Wandb Config
    use_wandb: bool = False
    project: str = "CDQAC-baselines"
    group: str = "bc"
    name: str = "bc"

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
