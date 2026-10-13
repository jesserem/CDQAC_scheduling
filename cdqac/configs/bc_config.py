"""Training configuration for Behavioral Cloning.

Shared settings (dataset, dataset sources, encoder, buffer, evaluation) come from
base_config.py with CDQAC's defaults; this file holds BC's own.
"""
from typing import Optional
from dataclasses import dataclass

from cdqac.configs.base_config import BaseConfig


@dataclass
class TrainConfig(BaseConfig):
    # Optimization
    p_lr: float = 3e-5

    # Output / logging
    save_folder: Optional[str] = "checkpoints_bc"
    project: str = "CDQAC-baselines"
    group: str = "bc"
    name: str = "bc"
