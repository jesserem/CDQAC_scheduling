"""Reading offline datasets (format: cdqac/data/format.py, docs: cdqac/data/README.md)."""
import os
from typing import Dict, List, Optional

import numpy as np

from cdqac.data import format as fmt


class InstanceSplit:
    """The scheduling instances of one split, padded to the largest one (see format.py)."""

    def __init__(self, job_length, op_pt, n_ops):
        self.job_length = np.asarray(job_length, dtype=np.int64)
        self.op_pt = np.asarray(op_pt, dtype=np.int64)
        self.n_ops = np.asarray(n_ops, dtype=np.int64)

    @classmethod
    def from_lists(cls, job_lengths, op_pts) -> "InstanceSplit":
        op_pts = [np.asarray(p, dtype=np.int64) for p in op_pts]
        n_ops = np.array([len(p) for p in op_pts], dtype=np.int64)
        op_pt = np.zeros((len(op_pts), n_ops.max(), op_pts[0].shape[1]), dtype=np.int64)
        for i, p in enumerate(op_pts):
            op_pt[i, :len(p)] = p
        return cls(np.asarray([np.asarray(j) for j in job_lengths]), op_pt, n_ops)

    def __len__(self) -> int:
        return len(self.n_ops)

    def head(self, n: Optional[int]) -> "InstanceSplit":
        """The first ``n`` instances (all of them for ``None``)."""
        if n is None:
            return self
        return InstanceSplit(self.job_length[:n], self.op_pt[:n], self.n_ops[:n])

    def job_lengths(self) -> List[np.ndarray]:
        return list(self.job_length)

    def op_pts(self) -> List[np.ndarray]:
        """Per-instance [n_ops_i, n_m] processing times, without the padding."""
        return [pt[:n] for pt, n in zip(self.op_pt, self.n_ops)]

    def as_dicts(self) -> List[dict]:
        """The instances as ``{"JobLength", "OpPT"}`` dicts, the form eval_model_all takes."""
        return [{"JobLength": jl, "OpPT": pt} for jl, pt in zip(self.job_lengths(), self.op_pts())]


class TrajectorySource:
    """The trajectories of one source for the leading ``num_instances`` train instances."""

    def __init__(self, name: str, actions, makespan, counts=None, names=None):
        self.name = name
        self.actions = np.asarray(actions)
        self.makespan = np.asarray(makespan, dtype=np.float64)
        if counts is None:
            counts = np.full(len(self.actions), self.actions.shape[1])
        self.counts = np.asarray(counts, dtype=np.int64)
        self.names = None if names is None else [str(n) for n in names]

    @property
    def num_instances(self) -> int:
        return len(self.actions)

    def trajectories(self, i: int) -> np.ndarray:
        """[K_i, n_ops] action sequences of instance ``i``."""
        return self.actions[i, :self.counts[i]].astype(np.int64)

    def makespans(self, i: int) -> np.ndarray:
        return self.makespan[i, :self.counts[i]]

    def as_lists(self, n: int) -> List[List[List[int]]]:
        """Trajectories of the first ``n`` instances as nested Python lists."""
        rows = self.actions[:n].tolist()
        return [r[:c] for r, c in zip(rows, self.counts[:n].tolist())]

    def makespan_lists(self, n: int) -> List[List[float]]:
        rows = self.makespan[:n].tolist()
        return [r[:c] for r, c in zip(rows, self.counts[:n].tolist())]


class OfflineDataset:
    """One dataset: its manifest, instance splits and trajectory sources (loaded lazily).

    Open one from disk with ``OfflineDataset.open(data_root, "fjsp/sd1_10x5")``; legacy
    ``.npy`` files are wrapped in the same interface by ``cdqac.data.legacy``.
    """

    def __init__(self, manifest: dict, path: Optional[str] = None, split_loader=None,
                 source_loader=None):
        self.manifest = manifest
        self.path = path
        self._split_loader = split_loader or self._load_split_file
        self._source_loader = source_loader or self._load_source_file
        self._splits: Dict[str, InstanceSplit] = {}
        self._sources: Dict[str, TrajectorySource] = {}

    @classmethod
    def from_dir(cls, path: str) -> "OfflineDataset":
        return cls(fmt.read_manifest(path), path=path)

    @classmethod
    def open(cls, data_root: str, dataset_id: str) -> "OfflineDataset":
        path = os.path.join(data_root, *dataset_id.split("/"))
        if not os.path.isfile(os.path.join(path, fmt.MANIFEST_FILE)):
            available = list_datasets(data_root)
            hint = (f"available: {', '.join(available)}" if available else
                    "no datasets found there; download them with download_datasets.py")
            raise FileNotFoundError(f"dataset '{dataset_id}' not found under {data_root} ({hint})")
        return cls.from_dir(path)

    @property
    def id(self) -> str:
        return self.manifest["id"]

    @property
    def n_j(self) -> int:
        return self.manifest["n_j"]

    @property
    def n_m(self) -> int:
        return self.manifest["n_m"]

    @property
    def split_names(self) -> List[str]:
        return list(self.manifest["splits"])

    @property
    def source_names(self) -> List[str]:
        return list(self.manifest["sources"])

    def has_source(self, name: str) -> bool:
        return name in self.manifest["sources"]

    def source_downloaded(self, name: str) -> bool:
        """Whether the file of a listed source is present (a download can skip sources)."""
        if self.path is None:
            return True
        return os.path.isfile(os.path.join(self.path, self.manifest["sources"][name]["file"]))

    def coverage(self, name: str) -> int:
        """How many leading train instances source ``name`` covers (0 if it is absent)."""
        entry = self.manifest["sources"].get(name)
        return 0 if entry is None else entry["num_instances"]

    def split(self, name: str = "train") -> InstanceSplit:
        if name not in self.manifest["splits"]:
            raise KeyError(f"dataset '{self.id}' has no split '{name}' "
                           f"(available: {', '.join(self.split_names)})")
        if name not in self._splits:
            self._splits[name] = self._split_loader(name)
        return self._splits[name]

    def source(self, name: str) -> TrajectorySource:
        if not self.has_source(name):
            raise KeyError(f"dataset '{self.id}' has no source '{name}' "
                           f"(available: {', '.join(self.source_names) or 'none'})")
        if name not in self._sources:
            self._sources[name] = self._source_loader(name)
        return self._sources[name]

    def _load_split_file(self, name: str) -> InstanceSplit:
        arrays = fmt.read_split(os.path.join(self.path, self.manifest["splits"][name]["file"]))
        return InstanceSplit(arrays["job_length"], arrays["op_pt"], arrays["n_ops"])

    def _load_source_file(self, name: str) -> TrajectorySource:
        if not self.source_downloaded(name):
            raise FileNotFoundError(f"source '{name}' of dataset '{self.id}' is not downloaded; "
                                    f"get it with: python download_datasets.py {self.id} "
                                    f"--sources {name}")
        arrays = fmt.read_source(os.path.join(self.path, self.manifest["sources"][name]["file"]))
        return TrajectorySource(name, arrays["actions"], arrays["makespan"],
                                counts=arrays.get("counts"), names=arrays.get("names"))


def list_datasets(data_root: str) -> List[str]:
    """Ids of every dataset (directory with a dataset.json) under ``data_root``."""
    ids = []
    if not os.path.isdir(data_root):
        return ids
    for dirpath, dirnames, filenames in os.walk(data_root):
        if fmt.MANIFEST_FILE in filenames:
            ids.append(os.path.relpath(dirpath, data_root).replace(os.sep, "/"))
            dirnames[:] = []  # datasets do not nest
        dirnames.sort()
    return sorted(ids)


def validate(dataset: OfflineDataset, check_checksums: bool = True) -> List[str]:
    """Check a dataset against the format; returns the problems found (empty = valid).

    Sources whose files were not downloaded are skipped, not reported.
    """
    problems = []
    m = dataset.manifest
    for key in ("id", "problem", "n_j", "n_m", "splits", "sources"):
        if key not in m:
            problems.append(f"manifest is missing '{key}'")
    if problems:
        return problems
    if m.get("action_mask") is not True:
        problems.append("manifest must declare \"action_mask\": true (trajectories are always "
                        "generated and replayed with the action mask)")
    n_j, n_m = m["n_j"], m["n_m"]

    skipped = {e["file"] for name, e in m["sources"].items() if not dataset.source_downloaded(name)}
    if check_checksums and dataset.path is not None:
        for f, digest in m.get("sha256", {}).items():
            path = os.path.join(dataset.path, f)
            if f in skipped:
                continue
            if not os.path.isfile(path):
                problems.append(f"{f}: file is missing")
            elif fmt.sha256(path) != digest:
                problems.append(f"{f}: sha256 does not match the manifest")

    for name, entry in m["splits"].items():
        try:
            s = dataset.split(name)
        except Exception as e:  # noqa: BLE001 -- report and keep checking
            problems.append(f"split {name}: cannot be read ({e})")
            continue
        if s.job_length.shape[1:] != (n_j,) or s.op_pt.shape[2] != n_m:
            problems.append(f"split {name}: shapes {s.job_length.shape} / {s.op_pt.shape} "
                            f"do not match n_j={n_j}, n_m={n_m}")
            continue
        if len(s) != entry["num_instances"]:
            problems.append(f"split {name}: {len(s)} instances, manifest says {entry['num_instances']}")
        if not np.array_equal(s.n_ops, s.job_length.sum(1)):
            problems.append(f"split {name}: n_ops != job_length.sum(1)")
        if [int(s.n_ops.min()), int(s.n_ops.max())] != list(entry["n_ops"]):
            problems.append(f"split {name}: n_ops range differs from the manifest")
        for i, (pt, n) in enumerate(zip(s.op_pt, s.n_ops)):
            if (pt[n:] != 0).any():
                problems.append(f"split {name}: instance {i} has non-zero padding rows")
                break
            if ((pt[:n] > 0).sum(1) == 0).any():
                problems.append(f"split {name}: instance {i} has an operation no machine can process")
                break

    train = dataset.split("train") if "train" in m["splits"] else None
    for name, entry in m["sources"].items():
        if entry["file"] in skipped:
            continue
        try:
            src = dataset.source(name)
        except Exception as e:  # noqa: BLE001
            problems.append(f"source {name}: cannot be read ({e})")
            continue
        if src.num_instances != entry["num_instances"]:
            problems.append(f"source {name}: covers {src.num_instances} instances, "
                            f"manifest says {entry['num_instances']}")
        if src.makespan.shape != src.actions.shape[:2]:
            problems.append(f"source {name}: makespan shape {src.makespan.shape} does not match "
                            f"actions {src.actions.shape}")
        if src.actions.size and src.actions.max() >= n_j * n_m:
            problems.append(f"source {name}: actions exceed n_j * n_m = {n_j * n_m}")
        if src.names is not None and len(src.names) != src.actions.shape[1]:
            problems.append(f"source {name}: {len(src.names)} names for {src.actions.shape[1]} slots")
        if train is not None:
            if src.num_instances > len(train):
                problems.append(f"source {name}: covers more instances than the train split")
            elif src.actions.shape[2] != train.n_ops[:src.num_instances].max():
                problems.append(f"source {name}: trajectory length {src.actions.shape[2]} != "
                                f"operations per train instance")
    return problems


# ---------------------------------------------------------------------------
# Resolving a TrainConfig to datasets (used by train.py)
# ---------------------------------------------------------------------------

# Where download_datasets.py --legacy puts the legacy FJSP files
LEGACY_DATA_PATH = "./train_datasets/legacy/fjsp"

# Legacy single-file datasets -> the dataset id holding the same data. Keys are
# (problem folder, file name); the paper-era names are md5-identical copies.
LEGACY_TRAIN_FILES = {
    ("fjsp", "train_10_5.npy"): "fjsp/sd1_10x5",
    ("fjsp", "train_15_10.npy"): "fjsp/sd1_15x10",
    ("fjsp", "train_20_10.npy"): "fjsp/sd1_20x10",
    ("jsp", "train_10_5.npy"): "jsp/taillard_10x5",
    ("jsp", "train_15_10.npy"): "jsp/taillard_15x10",
    ("dataset", "SD1_train_10_5_1000.npy"): "fjsp/sd1_10x5",
    ("dataset", "SD1_train_15_10_1000.npy"): "fjsp/sd1_15x10",
    ("dataset", "SD1_train_20_10_500.npy"): "fjsp/sd1_20x10",
}
LEGACY_VAL_FILES = {
    ("fjsp", "val_10_5.npy"): "fjsp/sd1_10x5",
    ("fjsp", "val_15_10.npy"): "fjsp/sd1_15x10",
    ("fjsp", "val_20_10.npy"): "fjsp/sd1_20x10",
    ("jsp", "val_10_5.npy"): "jsp/taillard_10x5",
    ("jsp", "val_15_10.npy"): "jsp/taillard_15x10",
    ("dataset", "SD1_10_5_val.npy"): "fjsp/sd1_10x5",
    ("dataset", "SD1_15_10_val.npy"): "fjsp/sd1_15x10",
    ("dataset", "SD1_20_10_val.npy"): "fjsp/sd1_20x10",
}


def _legacy_key(data_path: str, file_name: str):
    return os.path.basename(os.path.normpath(data_path)), file_name


def load_datasets(config):
    """Training data and validation instances selected by a TrainConfig.

    ``--dataset`` (default) picks a dataset directory under ``--data_root`` and validates on
    its ``--eval_split`` (or on ``--eval_dataset``). Setting ``--train_instance`` switches to a
    legacy ``.npy`` file under ``--data_path``; when that file is missing but its name is a
    known legacy name, the matching dataset id is used instead.

    Returns ``(train_data, eval_instances)``: an OfflineDataset and a list of
    ``{"JobLength", "OpPT"}`` dicts.
    """
    data_path = config.data_path or LEGACY_DATA_PATH
    if config.train_instance is None:
        train_data = OfflineDataset.open(config.data_root, config.dataset)
        print(f"Training dataset: {train_data.id} ({train_data.path})")
    else:
        train_data = _open_legacy_train(config, data_path)

    from_legacy_file = train_data.path is None
    if config.eval_instance is not None:
        eval_instances = _open_legacy_val(config, data_path, config.eval_instance)
    elif from_legacy_file and config.eval_dataset is None:
        eval_instances = _open_legacy_val(config, data_path,
                                          _default_val_name(config.train_instance))
    else:
        eval_id = config.eval_dataset or train_data.id
        eval_data = train_data if (not from_legacy_file and eval_id == train_data.id) \
            else OfflineDataset.open(config.data_root, eval_id)
        eval_instances = eval_data.split(config.eval_split).as_dicts()
        print(f"Validation instances: {eval_id} [{config.eval_split}], {len(eval_instances)}")
    return train_data, eval_instances


def _open_legacy_train(config, data_path: str) -> OfflineDataset:
    from cdqac.data.legacy import load_legacy_train

    path = os.path.join(data_path, config.train_instance)
    mapped = LEGACY_TRAIN_FILES.get(_legacy_key(data_path, config.train_instance))
    if os.path.isfile(path):
        hint = f" (same data as --dataset {mapped})" if mapped else ""
        print(f"Training dataset: legacy file {path}{hint}")
        return load_legacy_train(path)
    if mapped is None:
        raise FileNotFoundError(f"legacy training file {path} not found; pick a dataset with "
                                f"--dataset instead (available under {config.data_root}: "
                                f"{', '.join(list_datasets(config.data_root)) or 'none'})")
    print(f"NOTE: {path} not found; --train_instance is deprecated, using the converted "
          f"dataset --dataset {mapped}")
    return OfflineDataset.open(config.data_root, mapped)


def _open_legacy_val(config, data_path: str, file_name: str) -> List[dict]:
    from cdqac.data.legacy import load_legacy_instances

    path = os.path.join(data_path, file_name)
    if os.path.isfile(path):
        instances = load_legacy_instances(path)
        print(f"Validation instances: legacy file {path}, {len(instances)}")
        return instances
    mapped = LEGACY_VAL_FILES.get(_legacy_key(data_path, file_name))
    if mapped is None:
        raise FileNotFoundError(f"legacy validation file {path} not found; use --eval_dataset / "
                                f"--eval_split instead")
    print(f"NOTE: {path} not found; --eval_instance is deprecated, using the validation split "
          f"of --dataset {mapped}")
    return OfflineDataset.open(config.data_root, mapped).split("val").as_dicts()


def _default_val_name(train_instance: str) -> str:
    """Validation file paired with a legacy training file: train_<size>.npy -> val_<size>.npy."""
    base = os.path.basename(train_instance)
    if base.startswith("train_"):
        return "val_" + base[len("train_"):]
    raise ValueError(f"cannot tell which validation file belongs to {train_instance}; "
                     f"pass --eval_instance")
