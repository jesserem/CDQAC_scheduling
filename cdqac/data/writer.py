"""Writing datasets in the directory format (see cdqac/data/README.md).

Bring your own data in one call:

    from cdqac.data.writer import write_dataset
    write_dataset("./train_datasets", "fjsp/my_10x5",
                  train=(job_lengths, op_pts),          # per instance: [n_j], [n_ops, n_m]
                  val=(val_job_lengths, val_op_pts),
                  sources={"random": {"actions": actions, "makespans": makespans}})

or step by step with ``create_dataset`` and ``add_source``.
"""
import os
import shutil

import numpy as np

from cdqac.data import format as fmt
from cdqac.data.dataset import OfflineDataset, validate


def dataset_dir(data_root: str, dataset_id: str) -> str:
    return os.path.join(data_root, *dataset_id.split("/"))


def create_dataset(data_root: str, dataset_id: str, train, val=None, description: str = "",
                   generator: dict = None, provenance: dict = None, overwrite: bool = False) -> str:
    """Create a dataset directory with its instance splits; returns the directory.

    ``train`` and ``val`` are ``(job_lengths, op_pts)`` pairs of per-instance sequences.
    """
    if dataset_id.count("/") != 1:
        raise ValueError(f"dataset ids look like <problem>/<name>, got '{dataset_id}'")
    out = dataset_dir(data_root, dataset_id)
    if os.path.exists(os.path.join(out, fmt.MANIFEST_FILE)):
        if not overwrite:
            raise FileExistsError(f"{out} already holds a dataset (pass overwrite=True)")
        shutil.rmtree(out)
    job_lengths, op_pts = train
    n_j, n_m = len(job_lengths[0]), np.asarray(op_pts[0]).shape[1]
    manifest = fmt.new_manifest(dataset_id, n_j, n_m, description=description,
                                generator=generator, provenance=provenance)
    splits = {"train": train} if val is None else {"train": train, "val": val}
    for name, (jl, pt) in splits.items():
        entry = fmt.write_split(os.path.join(out, f"{name}.npz"), jl, pt)
        manifest["splits"][name] = dict(file=f"{name}.npz", **entry)
    fmt.write_manifest(out, manifest)
    return out


def add_source(path: str, name: str, actions, makespans, names=None, info: dict = None,
               overwrite: bool = False) -> dict:
    """Add one trajectory source to the dataset in directory ``path``; returns its manifest entry.

    ``actions[i]`` / ``makespans[i]`` are the trajectories of train instance ``i`` (covering
    the leading instances); ``info`` adds descriptive fields (description, order, seed, ...).
    """
    manifest = fmt.read_manifest(path)
    if name in manifest["sources"] and not overwrite:
        raise FileExistsError(f"{path} already has a '{name}' source (pass overwrite=True)")
    train = OfflineDataset.from_dir(path).split("train")
    if len(actions) > len(train):
        raise ValueError(f"source '{name}' covers {len(actions)} instances, the train split has "
                         f"{len(train)}")
    for i, a in enumerate(actions):
        if len(a) and np.asarray(a).shape[-1] != train.n_ops[i]:
            raise ValueError(f"source '{name}': trajectories of instance {i} have "
                             f"{np.asarray(a).shape[-1]} actions, the instance has "
                             f"{train.n_ops[i]} operations")
    file = f"sources/{name}.npz"
    entry = fmt.write_source(os.path.join(path, file), actions, makespans, names=names)
    manifest["sources"][name] = dict(file=file, **(info or {}), **entry)
    fmt.write_manifest(path, manifest)
    return manifest["sources"][name]


def write_dataset(data_root: str, dataset_id: str, train, val=None, sources: dict = None,
                  overwrite: bool = False, **kw) -> str:
    """Create a dataset with all its sources at once and validate it; returns the directory.

    ``sources`` maps a source name to ``{"actions", "makespans", ["names"], **info}``.
    """
    out = create_dataset(data_root, dataset_id, train, val=val, overwrite=overwrite, **kw)
    for name, spec in (sources or {}).items():
        spec = dict(spec)
        add_source(out, name, spec.pop("actions"), spec.pop("makespans"),
                   names=spec.pop("names", None), info=spec)
    problems = validate(OfflineDataset.from_dir(out))
    if problems:
        raise ValueError(f"{out} is not a valid dataset:\n  " + "\n  ".join(problems))
    return out
