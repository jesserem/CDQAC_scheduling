"""Training configuration for CDQAC (Conservative Discrete Quantile Actor-Critic).

Defaults correspond to the hyperparameters reported in Table 11 of the paper.
Every field can be overridden on the command line via pyrallis, e.g.:

    python train_cdqac.py --train_instance train_15_10.npy --seed 2
"""
from typing import List, Optional
from dataclasses import dataclass
import uuid


@dataclass
class TrainConfig:
    device: str = "cuda"
    seed: int = 1

    # --- Data -----------------------------------------------------------------
    data_path: str = "./train_dataset/fjsp"
    train_instance: str = "train_10_5.npy"
    eval_instance: str = "val_10_5.npy"
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

    # --- Network (DAN encoder + dueling quantile critic) ----------------------
    fea_j_input_dim: int = 10
    fea_m_input_dim: int = 8
    layer_fea_output_dim: List = (32, 8)
    num_heads_OAB: List = (4, 4)
    num_heads_MAB: List = (4, 4)
    num_mlp_layers_actor: int = 3
    num_mlp_layers_critic: int = 3
    hidden_dim_actor: int = 64
    hidden_dim_critic: int = 64
    num_quantiles: int = 64             # N in Eq. 1
    n_critics: int = 2                  # twin critic heads, min over expected values
    layer_norm: bool = False
    dropout_prob_actor: float = 0
    dropout_prob_q: float = 0
    use_mask: bool = True               # mask infeasible machine-operation pairs

    # --- CDQAC ablation switches (Sect. 5.5) ----------------------------------
    use_cql: bool = True                # CQL penalty (Eq. 4)
    use_adv_net: bool = True            # dueling architecture (Eq. 6)
    use_qrdqn: bool = True              # quantile critic (False -> scalar DQN critic)
    use_codac: bool = False             # CODAC distributional penalty instead of CQL (App. K.3)
    use_calql: bool = False

    # --- Optimization (Table 11) ----------------------------------------------
    num_train_step_offline: int = 200_000
    batch_size: int = 256
    q_lr: float = 2e-4                  # critic learning rate
    p_lr: float = 2e-5                  # policy learning rate
    update_freq_policy: int = 4         # policy delay eta
    target_update_freq: int = 1
    tau: float = 0.005                  # Polyak averaging rate rho
    cql_alpha_offline: float = 0.05     # CQL strength alpha_CQL
    # Entropy bonus in the policy loss (lambda in Eq. 5). NOTE: the paper's
    # Table 11 reports 0.005; the experiments were run with the value below.
    entropy_coef: float = 0.001
    gamma: float = 1
    kappa: float = 1                    # quantile Huber loss threshold
    max_grad_norm: Optional[float] = 1
    n_step: int = 1
    q_pretrain_steps: int = 0
    reward_scale: float = 1
    reward_bias: float = 0
    reward_scaling: Optional[str] = None
    normalize_q: bool = False
    target_entropy: float = 0.3         # only used when alpha_multiplier > 0
    alpha_multiplier: float = 0
    anneal_entropy: bool = False
    anneal_lr: bool = False
    backup_entropy: bool = False

    # --- Disk-backed buffer / DataLoader (low-RAM dataset pipeline) -----------
    use_disk_buffer: bool = False    # store transitions in memmaps on disk instead of RAM
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

    # --- Evaluation / checkpointing -------------------------------------------
    eval_freq: int = 1000               # evaluate every n gradient steps
    eval_every_epoch: int = 1
    do_epoch: bool = False              # iterate epochs over the buffer instead of steps
    train_epochs: int = 59
    save_folder: Optional[str] = "checkpoints"

    # --- Logging (wandb is optional) ------------------------------------------
    use_wandb: bool = False
    project: str = "CDQAC"
    group: str = "cdqac"
    name: str = "cdqac"

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
