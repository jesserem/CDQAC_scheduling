"""Offline training datasets: on-disk format, loading, and trajectory selection.

    from cdqac.data import OfflineDataset
    ds = OfflineDataset.open("./train_datasets", "fjsp/sd1_10x5")
    train = ds.split("train")            # instances
    random_trajs = ds.source("random")   # one trajectory source

See cdqac/data/README.md for the format and ``python -m cdqac.data --help`` for the CLI.
"""
from cdqac.data.dataset import (InstanceSplit, OfflineDataset, TrajectorySource, list_datasets,
                                load_datasets, validate)
from cdqac.data.format import PDR_RULES, eps_pdr_source
from cdqac.data.legacy import load_instance_dicts

__all__ = ["InstanceSplit", "OfflineDataset", "TrajectorySource", "list_datasets", "load_datasets",
           "validate", "PDR_RULES", "eps_pdr_source", "load_instance_dicts"]
