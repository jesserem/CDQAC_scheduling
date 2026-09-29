"""Configuration for training on the real-world datasets.

``make_config_class(MethodTrainConfig)`` returns a dataclass with every field of
the method's TrainConfig (so all hyperparameters, CLI overrides and the
checkpoint ``config.yml`` work exactly as for FJSP) plus the fields below.
Dataset-dependent defaults (reward weights, SMDP discounting, time source) are
``None`` / ``""`` here and filled in from the :class:`DatasetSpec` by
:func:`resolve`, so ``python train_realworld.py --dataset lade_p`` does the
right thing without further flags.
"""

from dataclasses import dataclass, make_dataclass
from typing import Optional

from realworld.registry import DatasetSpec
from realworld.routing.core import RewardConfig


@dataclass
class RealWorldFields:
    # --- data -------------------------------------------------------------------
    dataset: str = "amazon_lmrrc"        # registry name (python prepare_realworld.py list)
    data_root: str = "./realworld_data"  # prepared splits live in <data_root>/<dataset>/<split>.npz
    train_split: str = "train"
    eval_split: str = "val"
    sources: str = ""                    # behaviour sources, comma-separated ("" = the logged one,
    #                                      "all" = every source; "eps_nn*" prefix match)
    instance_filter: str = ""            # e.g. "route_score=High" or "city=Jilin,Yantai"
    num_instances: Optional[int] = None  # use the first n training instances (None = all)
    max_trajectories: Optional[int] = None
    time_source: str = ""                # logged | model ("" = dataset default)
    # --- reward (None = dataset default) -------------------------------------------
    reward_time: Optional[float] = None
    reward_travel: Optional[float] = None
    reward_lateness: Optional[float] = None
    reward_zone_leave: Optional[float] = None
    reward_latency: Optional[float] = None
    return_to_depot: Optional[bool] = None
    reward_scale: float = 1.0 / 60.0     # costs are in minutes -> rewards in hours
    # --- SMDP ------------------------------------------------------------------------
    smdp_discount: Optional[bool] = None  # discount gamma ** (tau / sojourn_unit) (None = dataset default)
    sojourn_unit: float = 1.0            # minutes per discount step when smdp_discount
    # --- network (routing encoder; MLP heads use the method's fields) -------------
    d_model: int = 128
    n_heads: int = 8
    n_enc_layers: int = 1
    n_dec_layers: int = 3
    ff_dim: int = 256
    zone_pool: bool = True
    # --- evaluation -------------------------------------------------------------------
    num_eval_instances: Optional[int] = 256
    eval_batch_size: int = 128
    eval_reference_metrics: bool = True  # route-similarity metrics vs. the logged routes
    select_metric: str = ""              # checkpoint selection ("" = dataset default: return)
    # --- pipeline ------------------------------------------------------------------------
    use_disk_buffer: bool = False        # not needed: routing states are rebuilt lazily from the episodes
    buffer_device: str = ""              # where instance tensors live ("" = device)
    stats_samples: int = 20000           # states used for feature normalization
    # --- set from the data at runtime (saved in config.yml for evaluation) ----------------
    node_dim: int = 0
    edge_dim: int = 0
    global_dim: int = 0


_CACHE = {}


def make_config_class(method_config_cls):
    """Dataclass combining a method's TrainConfig with :class:`RealWorldFields`."""
    if method_config_cls not in _CACHE:
        name = "RealWorld" + method_config_cls.__module__.split(".")[-1].title().replace("_", "")
        _CACHE[method_config_cls] = make_dataclass(name, [], bases=(RealWorldFields, method_config_cls))
    return _CACHE[method_config_cls]


def resolve(config, spec: DatasetSpec) -> None:
    """Fill dataset-dependent defaults in place."""
    for k, v in spec.reward.items():
        if getattr(config, f"reward_{k}") is None:
            setattr(config, f"reward_{k}", v)
    if config.return_to_depot is None:
        config.return_to_depot = spec.return_to_depot
    if config.smdp_discount is None:
        config.smdp_discount = spec.smdp_discount
    if not config.time_source:
        config.time_source = spec.time_source
    if not config.sources:
        config.sources = spec.logged_source
    if not config.select_metric:
        config.select_metric = spec.select_metric


def reward_config(config) -> RewardConfig:
    return RewardConfig(time=config.reward_time, travel=config.reward_travel, lateness=config.reward_lateness,
                        zone_leave=config.reward_zone_leave, latency=config.reward_latency,
                        return_to_depot=bool(config.return_to_depot), scale=config.reward_scale,
                        bias=config.reward_bias)


def source_list(sources: str) -> Optional[list]:
    s = [x.strip() for x in sources.split(",") if x.strip()]
    return None if (not s or s == ["all"]) else s
