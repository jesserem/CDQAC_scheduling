"""Settings shared by every training method, with CDQAC's defaults.

Each method's TrainConfig (cdqac_config.py, bc_config.py, ...) inherits from BaseConfig and
only adds its own hyperparameters (learning rates, critic design, losses, ...). Every method
therefore uses the same dataset, dataset sources, encoder, replay buffer and evaluation unless
a setting is overridden on the command line:

    python train.py --method iql --dataset fjsp/sd1_15x10 --use_dispatching true

Datasets are directories under ``data_root`` addressed by an id (see cdqac/data/README.md).
"""
from dataclasses import dataclass
from typing import List, Optional
import uuid


@dataclass
class BaseConfig:
    device: str = "cuda"
    seed: int = 1

    # --- Dataset ------------------------------------------------------------------
    data_root: str = "./train_datasets"   # folder holding the <problem>/<name>/ dataset directories
    dataset: str = "fjsp/sd1_10x5"        # training dataset id, e.g. fjsp/sd1_15x10, jsp/taillard_10x5
    eval_dataset: Optional[str] = None    # validation dataset id (default: same as dataset)
    eval_split: str = "val"               # split of the validation dataset to evaluate on
    # Trajectories are always replayed and evaluated with the action mask (operation-machine
    # pairs that cannot start at the current decision time are masked). Kept only so existing
    # configs parse: it must stay true.
    use_mask: bool = True

    # Legacy single-file .npy datasets (deprecated). Setting train_instance loads that file
    # from data_path instead of `dataset`; if the file is missing but its name is one of the
    # released files (e.g. train_15_10.npy), the converted dataset is used.
    data_path: Optional[str] = None       # default: ./train_datasets/legacy/fjsp
    train_instance: Optional[str] = None  # e.g. train_10_5.npy
    eval_instance: Optional[str] = None   # default: val_<size>.npy next to train_instance

    num_instances: Optional[int] = 500  # subset of training instances to use
    normalize: bool = False             # normalize state features with buffer statistics
    remove_duplicate: bool = True       # drop duplicate trajectories per instance

    # Dataset sources (Sect. 5, "Training dataset generation").
    # At least one of the following must be enabled.
    use_dispatching: bool = False       # PDR trajectories
    use_ga_hof: bool = False            # GA hall-of-fame trajectories
    n_ga_hof: Optional[int] = None
    ga_hof_random: bool = False
    use_ga_pop: bool = False            # GA final-population trajectories
    n_ga_pop: Optional[int] = None
    ga_pop_random: bool = False
    use_random: bool = True             # uniformly random trajectories (paper default)
    n_random: Optional[int] = 100
    use_eps_pdr: bool = False           # epsilon-perturbed PDR trajectories (App. F.3)
    eps_pdr_version: str = "0.1"
    n_eps_pdr: Optional[int] = 100

    # Dataset-property ablations (Table 3)
    use_proxy_reward: bool = False      # replace rewards by -1
    use_sparse_reward: bool = False     # makespan only at terminal state
    cutt_off_episode_prob: float = 0    # early-termination probability per step

    # --- Network: DAN encoder and actor head ----------------------------------------
    fea_j_input_dim: int = 10
    fea_m_input_dim: int = 8
    layer_fea_output_dim: List = (32, 8)
    num_heads_OAB: List = (4, 4)
    num_heads_MAB: List = (4, 4)
    num_mlp_layers_actor: int = 3
    hidden_dim_actor: int = 64
    dropout_prob_actor: float = 0

    # --- Training -------------------------------------------------------------------
    num_train_step_offline: int = 200_000
    batch_size: int = 256
    gamma: float = 1
    n_step: int = 1
    reward_scale: float = 1
    reward_bias: float = 0
    reward_scaling: Optional[str] = None

    # --- Disk-backed buffer / DataLoader (low-RAM dataset pipeline) -----------------
    use_disk_buffer: bool = True      # store transitions in memmaps on disk instead of RAM
    buffer_dir: Optional[str] = None  # memmap location; default: $SLURM_TMPDIR (node-local
    #                                   scratch) when running under SLURM, else
    #                                   <save_folder>/buffer_cache. For buffers larger than
    #                                   RAM this MUST be a local disk, not NFS/Lustre.
    cleanup_buffer_files: bool = True  # delete the memmap files when training finishes
    num_workers: int = 8             # DataLoader workers preparing batches in parallel
    prefetch_factor: int = 4         # batches prefetched per worker
    pin_memory: bool = True          # pinned host memory for async H2D copies
    persistent_workers: bool = True  # keep workers alive across epochs
    sampler_chunk_len: int = 1       # 1 = exact uniform sampling; >1 (e.g. 32) builds batches
    #                                   from contiguous index runs -> sequential disk reads.
    #                                   Set >1 for buffers larger than RAM (layout is env-major,
    #                                   so a chunk = same timestep of different trajectories).

    # --- Evaluation / checkpointing -------------------------------------------------
    eval_freq: int = 1000               # evaluate every n gradient steps
    eval_every_epoch: int = 1
    do_epoch: bool = False              # iterate epochs over the buffer instead of steps
    train_epochs: int = 50
    save_folder: Optional[str] = "checkpoints"

    # --- Logging (wandb is optional) ------------------------------------------------
    use_wandb: bool = False
    project: str = "CDQAC"
    group: str = "cdqac"
    name: str = "cdqac"

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
