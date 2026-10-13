"""Legacy single-file datasets (``train_10_5.npy``, ``SD1_train_10_5_1000.npy``, ...).

A legacy file is a pickled numpy object array with one dict per instance. It is wrapped in
the OfflineDataset interface so train.py and the converter treat both formats alike. The
legacy keys map to sources as follows:

    rules, rules_info          -> pdr          (masked rules in PDR_RULES only, renamed
                                                MWR_EST_masked -> MWR_EST)
    random, random_makespan    -> random
    ga, ga_makespan            -> ga_hof       (hall of fame, sorted by makespan)
    ga_pop, ga_pop_makespan    -> ga_pop       (final GA population)
    eps_random_<eps>           -> eps_pdr_<eps> (no makespans stored: NaN)

A source covers the leading instances that hold its key (the eps keys exist only for the
first 500 instances of the 1000-instance files).

Loading needs ``allow_pickle=True``, which can execute code embedded in the file: only load
legacy files you trust. The directory format has no such risk.
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from cdqac.data import format as fmt
from cdqac.data.dataset import InstanceSplit, OfflineDataset, TrajectorySource

# source name -> (actions key, makespan key) for the sources stored as plain lists
LEGACY_LIST_SOURCES = {
    "random": ("random", "random_makespan"),
    "ga_hof": ("ga", "ga_makespan"),
    "ga_pop": ("ga_pop", "ga_pop_makespan"),
}
LEGACY_EPS_PREFIX = "eps_random_"
LEGACY_RULE_SUFFIX = "_masked"  # legacy rule names say the rule ran with the action mask


def read_legacy(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)


def load_legacy_instances(path: str) -> List[dict]:
    """The instances of a legacy file as ``{"JobLength", "OpPT"}`` dicts of arrays."""
    return [{"JobLength": np.asarray(d["JobLength"]), "OpPT": np.asarray(d["OpPT"])}
            for d in read_legacy(path)]


def legacy_split(instances) -> InstanceSplit:
    return InstanceSplit.from_lists([d["JobLength"] for d in instances],
                                    [d["OpPT"] for d in instances])


def legacy_rule_name(legacy_name: str) -> Optional[str]:
    """'MWR_EST_masked' -> 'MWR_EST'; None for a rule not run with the action mask."""
    if not legacy_name.endswith(LEGACY_RULE_SUFFIX):
        return None
    return legacy_name[:-len(LEGACY_RULE_SUFFIX)]


def kept_legacy_rules(inst) -> List[Tuple[str, str]]:
    """(legacy name, name) of the rules of one legacy instance that are kept: the masked
    ones in ``PDR_RULES``, in the file's order."""
    info = inst["rules_info"]
    out = []
    for legacy_name in inst["rules"]:
        name = legacy_rule_name(legacy_name)
        if name in fmt.PDR_RULES and info[legacy_name]["masked"]:
            out.append((legacy_name, name))
    return out


def legacy_pdr_rules(instances) -> List[Tuple[str, str]]:
    """(legacy name, name) of the PDR rules kept from a legacy file; every instance must
    hold the same ones."""
    rules = kept_legacy_rules(instances[0])
    for i, inst in enumerate(instances):
        if "rules" not in inst:
            break
        if kept_legacy_rules(inst) != rules:
            raise ValueError(f"instance {i} holds different PDR rules than instance 0")
    return rules


def legacy_coverage(instances) -> Dict[str, int]:
    """Source name -> number of leading instances that hold it."""
    keys = {"pdr": "rules"}
    keys.update({name: k for name, (k, _) in LEGACY_LIST_SOURCES.items()})
    for k in instances[0]:
        if k.startswith(LEGACY_EPS_PREFIX):
            keys[fmt.eps_pdr_source(k[len(LEGACY_EPS_PREFIX):])] = k
    coverage = {}
    for name, key in keys.items():
        n = 0
        while n < len(instances) and key in instances[n]:
            n += 1
        if n:
            coverage[name] = n
    return coverage


def legacy_source(instances, name: str, coverage: int) -> TrajectorySource:
    """One source of a legacy file as a TrajectorySource over its first ``coverage`` instances."""
    insts = instances[:coverage]
    names = None
    if name == "pdr":
        rules = legacy_pdr_rules(instances)
        names = [new for _, new in rules]
        actions = [[inst["rules"][old] for old, _ in rules] for inst in insts]
        makespans = [[inst["rules_info"][old]["makespan"] for old, _ in rules] for inst in insts]
    elif name in LEGACY_LIST_SOURCES:
        a_key, m_key = LEGACY_LIST_SOURCES[name]
        actions = [inst[a_key] for inst in insts]
        makespans = [inst[m_key] for inst in insts]
    elif name.startswith("eps_pdr_"):
        key = LEGACY_EPS_PREFIX + name[len("eps_pdr_"):]
        actions = [inst[key] for inst in insts]
        makespans = [[np.nan] * len(a) for a in actions]
    else:
        raise KeyError(name)

    counts = np.array([len(a) for a in actions])
    if counts.max() == 0:
        raise ValueError(f"legacy source {name} holds no trajectories")
    k_max, length = int(counts.max()), len(next(a for a in actions if len(a))[0])
    n_actions = len(insts[0]["JobLength"]) * len(insts[0]["OpPT"][0])
    act = np.zeros((len(actions), k_max, length), dtype=fmt.min_uint_dtype(n_actions))
    span = np.zeros((len(actions), k_max), dtype=np.float64)
    for i, (a, m) in enumerate(zip(actions, makespans)):
        if len(a):
            act[i, :len(a)] = np.asarray(a, dtype=np.int64)
            span[i, :len(m)] = np.asarray(m, dtype=np.float64)
    return TrajectorySource(name, act, span, counts=None if counts.min() == k_max else counts,
                            names=names)


def load_legacy_train(path: str) -> OfflineDataset:
    """Wrap a legacy training file in an OfflineDataset (sources are converted on first use)."""
    instances = read_legacy(path)
    split = legacy_split(instances)
    coverage = legacy_coverage(instances)
    manifest = fmt.new_manifest(
        dataset_id=f"legacy:{os.path.basename(path)}",
        n_j=split.job_length.shape[1], n_m=split.op_pt.shape[2],
        provenance={"legacy_file": os.path.abspath(path)})
    manifest["problem"] = os.path.basename(os.path.dirname(os.path.abspath(path)))
    manifest["splits"]["train"] = {"num_instances": len(split),
                                   "n_ops": [int(split.n_ops.min()), int(split.n_ops.max())]}
    manifest["sources"] = {name: {"num_instances": n} for name, n in coverage.items()}

    return OfflineDataset(
        manifest,
        split_loader=lambda name: split,
        source_loader=lambda name: legacy_source(instances, name, coverage[name]))


# ---------------------------------------------------------------------------
# Legacy dict view of a dataset (for code written against the legacy layout)
# ---------------------------------------------------------------------------

DICT_KEYS = {  # source name -> (actions key, makespan key or None)
    "random": ("random", "random_makespan"),
    "ga_hof": ("ga", "ga_makespan"),
    "ga_pop": ("ga_pop", "ga_pop_makespan"),
}


def as_instance_dicts(dataset: OfflineDataset, split: str = "train") -> List[dict]:
    """A dataset split as legacy-style instance dicts of Python lists.

    The train split carries every source under its legacy key (rules/rules_info,
    random/random_makespan, ga/ga_makespan, ga_pop/ga_pop_makespan, eps_random_<eps>); a
    source only appears on the instances it covers. Rules use the dataset names (MWR_EST).
    """
    s = dataset.split(split)
    insts = [{"JobLength": jl.tolist(), "OpPT": pt.tolist()}
             for jl, pt in zip(s.job_lengths(), s.op_pts())]
    if split != "train":
        return insts
    for name in dataset.source_names:
        src = dataset.source(name)
        n = src.num_instances
        actions, makespans = src.as_lists(n), src.makespan_lists(n)
        for i in range(n):
            inst = insts[i]
            if name == "pdr":
                inst["rules"] = dict(zip(src.names, actions[i]))
                inst["rules_info"] = {r: {"makespan": m} for r, m in zip(src.names, makespans[i])}
            elif name in DICT_KEYS:
                a_key, m_key = DICT_KEYS[name]
                inst[a_key], inst[m_key] = actions[i], makespans[i]
            elif name.startswith("eps_pdr_"):
                inst[LEGACY_EPS_PREFIX + name[len("eps_pdr_"):]] = actions[i]
    return insts


def load_instance_dicts(ref: str, split: str = "train", data_root: str = "./train_datasets"):
    """Instance dicts in the legacy layout from a dataset id, dataset directory or legacy file.

    A legacy file's rules are renamed and filtered like the converted datasets' (MWR_EST, the
    16 PDR_RULES), so code sees the same dicts from either. A missing legacy file with a known
    name (e.g. dataset/SD1_train_10_5_1000.npy) resolves to its converted dataset; for a
    legacy file the ``split`` argument is ignored.
    """
    from cdqac.data.dataset import LEGACY_TRAIN_FILES, LEGACY_VAL_FILES, _legacy_key
    if ref.endswith(".npy"):
        if os.path.isfile(ref):
            return [_rename_rules(d) for d in read_legacy(ref)]
        known = LEGACY_TRAIN_FILES if split == "train" else LEGACY_VAL_FILES
        mapped = known.get(_legacy_key(os.path.dirname(ref) or ".", os.path.basename(ref)))
        if mapped is None:
            raise FileNotFoundError(f"{ref} not found")
        print(f"NOTE: {ref} not found, using the converted dataset {mapped}")
        ref = mapped
    if os.path.isfile(os.path.join(ref, fmt.MANIFEST_FILE)):
        return as_instance_dicts(OfflineDataset.from_dir(ref), split)
    return as_instance_dicts(OfflineDataset.open(data_root, ref), split)


def _rename_rules(inst: dict) -> dict:
    """A legacy instance dict with only the kept rules, under their dataset names."""
    if "rules" not in inst:
        return inst
    inst = dict(inst)
    rules = kept_legacy_rules(inst)
    inst["rules_info"] = {new: {"makespan": inst["rules_info"][old]["makespan"]} for old, new in rules}
    inst["rules"] = {new: inst["rules"][old] for old, new in rules}
    return inst
