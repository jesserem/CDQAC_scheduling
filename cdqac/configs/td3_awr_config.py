from typing import List, Optional
from dataclasses import dataclass
import uuid

@dataclass
class TrainConfig:
    # JSSP Environment

    device: str = "cuda"


    # Data collection
    data_path: str = "./train_dataset/fjsp"
    train_instance: str = "train_10_5.npy"
    eval_instance: str = "val_10_5.npy"

    eval_every_epoch: int = 1

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
    layer_norm: bool = True          # ReBRAC: LayerNorm in the critic MLPs
    dropout_prob_actor: float = 0
    dropout_prob_q: float = 0

    use_mask: bool = True

    # Target network / optimisation
    target_update_freq: int = 1
    tau: float = 0.005

    batch_size: int = 256

    eval_freq: int = 1000
    q_lr: float = 3e-4
    p_lr: float = 3e-4
    v_lr: float = 3e-4
    max_grad_norm: float = 1.0

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

    # --- Disk-backed buffer / DataLoader (low-RAM dataset pipeline) ---
    use_disk_buffer: bool = True     # store transitions in memmaps on disk instead of RAM
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

    reward_scaling: Optional[str] = None

    gamma: float = 1


    update_freq_policy: int = 4      # TD3 delayed actor update
    num_train_step_offline: int = 200_000
    seed: int = 1
    reward_scale: float = 1
    reward_bias: float = 0

    remove_duplicate: bool = True

    save_folder: Optional[str] = "checkpoints_td3_awr"

    num_instances: Optional[int] = 500

    use_dispatching: bool = False

    use_ga_hof: bool = False
    n_ga_hof: Optional[int] = 25
    ga_hof_random: bool = False

    use_ga_pop: bool = False
    n_ga_pop: Optional[int] = None
    ga_pop_random: bool = False

    use_random: bool = True
    n_random: Optional[int] = 100


    # Wandb Config
    use_wandb: bool = False
    project: str = "CDQAC-baselines"
    group: str = "td3_awr"
    name: str = "td3_awr"

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
