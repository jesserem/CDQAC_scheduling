"""Dataset registry: one :class:`DatasetSpec` per real-world dataset.

Adding a dataset = writing an adapter module that turns the raw files into
:class:`realworld.episodes.RoutingEpisodeSet` splits and registering it here
(see "Adding a dataset" in realworld/README.md).  Everything downstream
(buffer, networks, all offline-RL methods, evaluation, CLIs) is shared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class DatasetSpec:
    name: str
    family: str                      # problem family; "routing" for now
    description: str
    mdp: str                         # decision process in one line
    license: str
    citation: str
    url: str
    prepare: Callable                # prepare(raw_dir, out_dir, **options) -> {split: path}
    download: Optional[Callable]     # download(raw_dir, **options) -> None
    logged_source: str               # name of the real behaviour source in the episode sets
    splits: List[str]                # splits written by prepare()
    reward: Dict[str, float]         # default RewardConfig weights
    return_to_depot: bool
    smdp_discount: bool              # default discounting mode
    time_source: str = "logged"      # default replay time source
    reference_metrics: List[str] = field(default_factory=lambda: ["hr3", "krc", "lsd", "ed"])
    amazon_score: bool = False       # compute the official Amazon challenge score
    select_metric: str = "return"    # checkpoint selection metric on the eval split


DATASETS: Dict[str, DatasetSpec] = {}

HIGHER_IS_BETTER = {"return", "krc", "hr3"}


def register(spec: DatasetSpec) -> DatasetSpec:
    DATASETS[spec.name] = spec
    return spec


def get_dataset(name: str) -> DatasetSpec:
    _load_builtin()
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; registered: {sorted(DATASETS)}")
    return DATASETS[name]


def list_datasets() -> List[str]:
    _load_builtin()
    return sorted(DATASETS)


def _load_builtin() -> None:
    # Importing the adapter modules registers them.
    from realworld.adapters import amazon_lmrrc, lade  # noqa: F401
