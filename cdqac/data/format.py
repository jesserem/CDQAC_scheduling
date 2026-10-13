"""On-disk format of the offline training datasets (documented in cdqac/data/README.md).

A dataset is one directory, addressed by its id ``<problem>/<name>`` under a data root:

    <data_root>/fjsp/sd1_10x5/
        dataset.json            manifest: what the dataset holds and where it came from
        train.npz, val.npz      instance splits
        sources/<source>.npz    one file per trajectory source (pdr, random, ga_hof, ...)

Every file holds plain numeric arrays, so ``np.load`` never needs ``allow_pickle``. All
trajectories are generated and replayed with the environment's action mask (only pairs that
can start at the current decision time are allowed); manifests declare ``"action_mask": true``.

Split files (``train.npz``, ``val.npz``):
    job_length  [N, n_j]         operations per job
    op_pt       [N, O_max, n_m]  processing time of each operation on each machine
                                 (0 = the machine cannot process it); rows past
                                 ``n_ops[i]`` are zero padding
    n_ops       [N]              operations of each instance (= job_length.sum(1))

Source files (``sources/<source>.npz``), covering instances ``0 .. N_s-1`` of the train split:
    actions     [N_s, K, O]      trajectories as action sequences, action = job * n_m + machine
    makespan    [N_s, K]         makespan each trajectory reaches
    counts      [N_s]            (optional) trajectories of each instance when it varies;
                                 rows past ``counts[i]`` are zero padding
    names       [K]              (optional) one label per trajectory slot, e.g. the PDR rule
"""
import hashlib
import json
import os

import numpy as np

FORMAT_VERSION = 1
MANIFEST_FILE = "dataset.json"
ACTION_ENCODING = "job * n_m + machine"

# The dispatching rules kept as the "pdr" source, in their stored order: job rule {MWR, LWR,
# LOPNR, MOPNR} x machine rule {EST, LST, LPT, SPT} (see env/dispatching.py). Like every
# trajectory, they are generated with the action mask, so the names carry no mask suffix
# (the legacy files call them e.g. MWR_EST_masked). The legacy files also hold LCT/MCT/FAM
# variants, which are dropped.
PDR_RULES = tuple(f"{job}_{machine}" for job in ("MWR", "LWR", "LOPNR", "MOPNR")
                  for machine in ("EST", "LST", "LPT", "SPT"))


def eps_pdr_source(eps) -> str:
    """Source name of the epsilon-PDR trajectories for one epsilon: 0.1 -> 'eps_pdr_0.1'."""
    return f"eps_pdr_{eps}"


def min_uint_dtype(max_value: int):
    """Smallest unsigned integer dtype that holds ``max_value``."""
    for dtype in (np.uint8, np.uint16, np.uint32):
        if max_value <= np.iinfo(dtype).max:
            return dtype
    return np.uint64


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def write_split(path: str, job_lengths, op_pts) -> dict:
    """Write one instance split; returns its manifest entry (without the file name).

    ``job_lengths`` is a sequence of [n_j] arrays and ``op_pts`` a sequence of [n_ops_i, n_m]
    arrays (instances may differ in their number of operations).
    """
    job_length = np.asarray([np.asarray(j) for j in job_lengths])
    op_pts = [np.asarray(p) for p in op_pts]
    if job_length.ndim != 2 or len(op_pts) != len(job_length):
        raise ValueError("job_lengths must be [N, n_j] and op_pts must hold N instances")
    n_ops = np.array([p.shape[0] for p in op_pts])
    if not np.array_equal(n_ops, job_length.sum(1)):
        raise ValueError("every instance needs job_length.sum() == number of op_pt rows")
    n_m = op_pts[0].shape[1]
    op_pt = np.zeros((len(op_pts), n_ops.max(), n_m), dtype=np.int64)
    for i, p in enumerate(op_pts):
        op_pt[i, :len(p)] = p
    _savez(path,
           job_length=job_length.astype(min_uint_dtype(job_length.max())),
           op_pt=op_pt.astype(min_uint_dtype(op_pt.max())),
           n_ops=n_ops.astype(min_uint_dtype(n_ops.max())))
    return {"num_instances": int(len(op_pts)), "n_ops": [int(n_ops.min()), int(n_ops.max())]}


def read_split(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in ("job_length", "op_pt", "n_ops")}


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------

def write_source(path: str, actions, makespans, names=None) -> dict:
    """Write one trajectory source; returns its manifest entry (without the file name).

    ``actions[i]`` holds the trajectories of instance ``i`` ([K_i, O] or a list of lists) and
    ``makespans[i]`` their makespans. When K_i varies per instance the arrays are zero-padded
    and the counts are stored.
    """
    per_inst = [np.asarray(a, dtype=np.int64).reshape(len(a), -1) if len(a)
                else np.zeros((0, 0), dtype=np.int64) for a in actions]
    spans = [np.asarray(m, dtype=np.float64) for m in makespans]
    if len(per_inst) != len(spans):
        raise ValueError("actions and makespans must cover the same instances")
    counts = np.array([len(a) for a in per_inst])
    if any(len(m) != c for m, c in zip(spans, counts)):
        raise ValueError("every trajectory needs exactly one makespan")
    traj_len = {a.shape[1] for a in per_inst if len(a)}
    if len(traj_len) != 1:
        raise ValueError(f"all trajectories of a source must have the same length, got {traj_len}")
    k_max, length = int(counts.max()), traj_len.pop()

    act = np.zeros((len(per_inst), k_max, length), dtype=np.int64)
    span = np.zeros((len(per_inst), k_max), dtype=np.float64)
    for i, (a, m) in enumerate(zip(per_inst, spans)):
        if len(a):
            act[i, :len(a)] = a
            span[i, :len(m)] = m
    arrays = {"actions": act.astype(min_uint_dtype(act.max())), "makespan": span}
    if counts.min() != k_max:
        arrays["counts"] = counts.astype(min_uint_dtype(k_max))
    if names is not None:
        if len(names) != k_max:
            raise ValueError(f"got {len(names)} names for {k_max} trajectory slots")
        arrays["names"] = np.asarray(names, dtype=str)
    _savez(path, **arrays)

    entry = {"num_instances": int(len(per_inst))}
    if "counts" in arrays:
        entry["per_instance"] = [int(counts.min()), k_max]
    else:
        entry["per_instance"] = k_max
    return entry


def read_source(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def read_manifest(dataset_dir: str) -> dict:
    path = os.path.join(dataset_dir, MANIFEST_FILE)
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    version = manifest.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: format_version {version} is not supported "
                         f"(this code reads version {FORMAT_VERSION})")
    return manifest


def write_manifest(dataset_dir: str, manifest: dict, update_checksums: bool = True) -> None:
    """Write ``dataset.json``; with ``update_checksums`` the sha256 of every listed file is
    recomputed first."""
    manifest = dict(manifest)
    manifest["format_version"] = FORMAT_VERSION
    if update_checksums:
        files = [e["file"] for e in manifest.get("splits", {}).values()]
        files += [e["file"] for e in manifest.get("sources", {}).values()]
        manifest["sha256"] = {f: sha256(os.path.join(dataset_dir, f)) for f in files}
    with open(os.path.join(dataset_dir, MANIFEST_FILE), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def new_manifest(dataset_id: str, n_j: int, n_m: int, description: str = "",
                 generator: dict = None, provenance: dict = None) -> dict:
    """An empty manifest for a new dataset; splits and sources are added as they are written."""
    problem = dataset_id.split("/")[0]
    return {
        "format_version": FORMAT_VERSION,
        "id": dataset_id,
        "problem": problem,
        "description": description,
        "n_j": int(n_j),
        "n_m": int(n_m),
        "action_encoding": ACTION_ENCODING,
        "action_mask": True,
        "generator": generator or {},
        "provenance": provenance or {},
        "splits": {},
        "sources": {},
    }


def _savez(path: str, **arrays) -> None:
    """Write a compressed npz atomically, so an interrupted write never leaves a broken file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)
