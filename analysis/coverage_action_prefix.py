"""State-action coverage (SACo) for offline-FJSP training datasets.

This is the counting-based counterpart of ``density_embedding_knn.py`` and the
script form of the coverage cells in ``analyze_training_dataset.ipynb``.  Where the
k-NN script measures how *densely* a source surrounds the model's optimal trajectory
in the encoder's embedding space, this one measures how *broadly* a source covers the
reachable state-action space, with no model involved:

  1. A partial schedule is fully determined by the sequence of actions taken so far,
     so an action *prefix* ``(a_0, ..., a_t)`` identifies a reachable state-action pair.
  2. For each dataset *type* (PDR, GA, PDR-GA, Random) we count the **unique action
     prefixes** over that type's trajectories for an instance.
  3. We report the count both raw and normalized by the instance's PDR count
     (``SACo``: PDR = 1.0 by construction), then aggregate across instances.

Higher SACo ==> that dataset type visits more distinct state-action pairs.

Prefixes are counted with a trie (one node per distinct prefix) rather than by
building ``"3,17,4,..."`` strings as the notebook does; the count is identical but the
cost is linear in the number of actions instead of quadratic in trajectory length.
Note that whole-trajectory dedup (the ``remove_duplicate_actions`` step in training)
cannot change a prefix count -- a duplicate trajectory contributes no new prefix -- so
unlike the k-NN script there is no ``--no_dedupe`` switch here.

Run (all instances of one dataset):

    python coverage_action_prefix.py --data dataset/SD1_train_10_5_1000.npy

All three SD1 sizes at once, writing every result to CSV:

    python coverage_action_prefix.py --data all --out results/coverage.csv

That produces one per-instance sheet per dataset (``results/coverage_10_5.csv``,
``_15_10``, ``_20_10``) plus one combined, aggregated sheet
``results/coverage_summary.csv`` with a row per (dataset, type).

Random-only sweep -- how much coverage does the Random source lose as it is thinned
from 100 to 1 trajectory per instance?  (Counting counterpart of the k-NN sweep.)

    python coverage_action_prefix.py --data all --random_sweep \
        --random_counts 1 5 10 25 50 100 --out results/coverage_random_sweep.csv

Same CSV layout, but rows are keyed by ``n_random`` instead of dataset type and only
Random is scored (still normalized by the same per-instance PDR denominator, so the
numbers stay comparable to the full run).

Full parameterisation via CLI (see ``build_argparser``) or import the functions
(``run_dataset``, ``do_all_coverage_datasets``, ``run_random_sweep_dataset``) into a
notebook.
"""

import argparse
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# Dataset-type definitions (identical to density_embedding_knn.py)
# ---------------------------------------------------------------------------

USED_PDR = [
    "MWR_SPT_masked", "MWR_LPT_masked", "LWR_SPT_masked", "LWR_LPT_masked",
    "MWR_EST_masked", "MWR_LST_masked", "LWR_EST_masked", "LWR_LST_masked",
    "MOPNR_SPT_masked", "MOPNR_LPT_masked", "LOPNR_SPT_masked", "LOPNR_LPT_masked",
    "MOPNR_EST_masked", "MOPNR_LST_masked", "LOPNR_EST_masked", "LOPNR_LST_masked",
]

DATASET_TYPES = ["PDR", "GA", "PDR-GA", "Random"]

DEFAULT_ALL_DATASETS = [
    "./dataset/SD1_train_10_5_1000.npy",
    "./dataset/SD1_train_15_10_1000.npy",
    "./dataset/SD1_train_20_10_500.npy",
]


# ---------------------------------------------------------------------------
# Prefix counting
# ---------------------------------------------------------------------------

def count_unique_prefixes(action_lists):
    """Number of distinct action prefixes over ``action_lists`` (and total actions).

    Each prefix ``(a_0, ..., a_t)`` names one reachable state-action pair, so this is
    the SACo numerator.  Implemented as a trie: every node below the root is exactly
    one distinct prefix, so the node count is the answer and each action costs one
    dict lookup.  Equivalent to the notebook's string-prefix set, minus the quadratic
    string building.
    """
    root = {}
    n_unique = 0
    n_actions = 0
    for action_list in action_lists:
        node = root
        for a in action_list:
            key = int(a)
            child = node.get(key)
            if child is None:
                child = {}
                node[key] = child
                n_unique += 1
            node = child
            n_actions += 1
    return n_unique, n_actions


def count_unique_prefixes_cumulative(action_lists, checkpoints):
    """Unique-prefix count after the first ``m`` trajectories, for each ``m``.

    The trie only ever grows as trajectories are added, so one pass over the longest
    prefix of the list yields every ``m`` at once -- the sweep costs the same as the
    single largest count.  ``checkpoints`` need not be sorted; counts beyond
    ``len(action_lists)`` are clipped to the full list.

    Returns ``({m: n_unique}, {m: n_actions})``.
    """
    wanted = sorted(set(int(m) for m in checkpoints))
    root = {}
    n_unique = 0
    n_actions = 0
    uniq_at, actions_at = {}, {}
    nxt = 0
    for i, action_list in enumerate(action_lists):
        node = root
        for a in action_list:
            key = int(a)
            child = node.get(key)
            if child is None:
                child = {}
                node[key] = child
                n_unique += 1
            node = child
            n_actions += 1
        while nxt < len(wanted) and wanted[nxt] == i + 1:
            uniq_at[wanted[nxt]] = n_unique
            actions_at[wanted[nxt]] = n_actions
            nxt += 1
    # Any checkpoint above the number of available trajectories sees the whole list.
    for m in wanted[nxt:]:
        uniq_at[m] = n_unique
        actions_at[m] = n_actions
    return uniq_at, actions_at


# ---------------------------------------------------------------------------
# Per-instance and per-dataset drivers
# ---------------------------------------------------------------------------

def gather_action_lists(instance, n_ga, n_random):
    """Return {type: list-of-action-lists} for one instance dict.

    Mirrors ``density_embedding_knn.gather_action_lists``; PDR-GA is the concatenation
    of PDR + GA (prefix counting merges them automatically -- shared prefixes are
    counted once).
    """
    rules = instance["rules"]
    pdr = [rules[r] for r in USED_PDR if r in rules]
    ga = list(instance["ga_pop"][:n_ga])
    rnd = list(instance["random"][:n_random])
    return {"PDR": pdr, "GA": ga, "Random": rnd}


def run_instance(instance, n_ga=200, n_random=100):
    """Score one instance.

    Returns (saco, unique, actions, n_traj): ``saco[type]`` is the unique-prefix count
    normalized by the instance's PDR count (so PDR == 1.0), ``unique[type]`` the raw
    count, ``actions[type]`` the total actions counted, ``n_traj[type]`` the number of
    trajectories.
    """
    raw = gather_action_lists(instance, n_ga=n_ga, n_random=n_random)
    raw["PDR-GA"] = raw["PDR"] + raw["GA"]

    unique, actions, n_traj = {}, {}, {}
    for t in DATASET_TYPES:
        unique[t], actions[t] = count_unique_prefixes(raw[t])
        n_traj[t] = len(raw[t])

    div = unique["PDR"]
    saco = {t: (unique[t] / div if div else float("nan")) for t in DATASET_TYPES}
    return saco, unique, actions, n_traj


def run_instance_random_sweep(instance, random_counts, n_ga=200, n_random=100):
    """Score one instance using ONLY the Random source, at several dataset sizes.

    For every ``m`` in ``random_counts`` we keep the *first* ``m`` stored random
    trajectories -- the same prefix a thinned training run would use -- and count their
    unique action prefixes.  The SACo denominator stays the instance's PDR count, so a
    sweep value is directly comparable to the Random column of the full run.

    Returns ({m: saco}, {m: unique}, {m: actions}), plus the PDR denominator.
    """
    raw = gather_action_lists(instance, n_ga=n_ga, n_random=n_random)
    pdr_unique, _ = count_unique_prefixes(raw["PDR"])

    rnd = list(instance["random"][:max(random_counts)])
    uniq_at, actions_at = count_unique_prefixes_cumulative(rnd, random_counts)
    saco_at = {m: (uniq_at[m] / pdr_unique if pdr_unique else float("nan"))
               for m in uniq_at}
    return saco_at, uniq_at, actions_at, pdr_unique


def _summarise(values, round_val=4):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(mean=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"))
    return dict(mean=round(float(a.mean()), round_val), std=round(float(a.std()), round_val),
                min=round(float(a.min()), round_val), max=round(float(a.max()), round_val))


def run_dataset(data_path, n_instances=None, n_ga=200, n_random=100, verbose=True,
                out=None):
    """Run the coverage experiment over one .npy dataset file.

    Returns (per_instance, summary): ``per_instance`` is a list of dicts (one per
    instance); ``summary`` is ``{type: {'saco': {mean,std,min,max}, 'unique': mean,
    'actions': mean, 'n_traj': mean}}`` aggregated across instances.  If ``out`` is
    given, the per-instance results are also written there as CSV.
    """
    data = np.load(data_path, allow_pickle=True)
    if n_instances is not None:
        data = data[:n_instances]

    per_instance = []
    saco_accum = {t: [] for t in DATASET_TYPES}
    uniq_accum = {t: [] for t in DATASET_TYPES}
    act_accum = {t: [] for t in DATASET_TYPES}
    traj_accum = {t: [] for t in DATASET_TYPES}

    t0 = time.time()
    for i in range(len(data)):
        saco, unique, actions, n_traj = run_instance(data[i], n_ga=n_ga, n_random=n_random)
        rec = {"instance": i}
        for t in DATASET_TYPES:
            rec[f"{t}_saco"] = saco[t]
            rec[f"{t}_unique"] = unique[t]
            rec[f"{t}_actions"] = actions[t]
            saco_accum[t].append(saco[t])
            uniq_accum[t].append(unique[t])
            act_accum[t].append(actions[t])
            traj_accum[t].append(n_traj[t])
        per_instance.append(rec)

    summary = {
        t: {"saco": _summarise(saco_accum[t]),
            "unique": float(np.mean(uniq_accum[t])),
            "actions": float(np.mean(act_accum[t])),
            "n_traj": float(np.mean(traj_accum[t]))}
        for t in DATASET_TYPES
    }

    if verbose:
        print(f"\n==== State-action coverage (SACo) for: {data_path} ====")
        print(f"(unique action prefixes per source, normalized by the instance's PDR "
              f"count; higher = broader coverage; {len(data)} instances, "
              f"{time.time() - t0:.1f}s)")
        for t in DATASET_TYPES:
            s = summary[t]["saco"]
            print(f"  {t:8s} SACo {s['mean']:6.2f}±{s['std']:.2f} "
                  f"[{s['min']:.2f},{s['max']:.2f}]  "
                  f"(uniq prefixes~{summary[t]['unique']:.0f} of "
                  f"{summary[t]['actions']:.0f} actions over "
                  f"{summary[t]['n_traj']:.0f} traj)")

    if out:
        write_csv(per_instance, out)

    return per_instance, summary


def run_random_sweep_dataset(data_path, random_counts=(1, 5, 10, 25, 50, 100),
                             n_instances=None, n_ga=200, n_random=100, verbose=True,
                             out=None):
    """Random-only coverage sweep over one .npy dataset: SACo vs. #random runs.

    Returns (per_instance, summary): ``per_instance`` is a list of rows, one per
    (instance, m); ``summary`` is ``{m: {'saco': {...}, 'unique': mean, 'actions': mean}}``.
    """
    random_counts = sorted(set(int(m) for m in random_counts))
    data = np.load(data_path, allow_pickle=True)
    if n_instances is not None:
        data = data[:n_instances]

    available = len(data[0]["random"]) if len(data) else 0
    if random_counts[-1] > available:
        print(f"WARNING: --random_counts asks for {random_counts[-1]} trajectories but the "
              f"dataset stores only {available} per instance; larger counts are clipped.")

    per_instance = []
    saco_accum = {m: [] for m in random_counts}
    uniq_accum = {m: [] for m in random_counts}
    act_accum = {m: [] for m in random_counts}

    t0 = time.time()
    for i in range(len(data)):
        saco, unique, actions, pdr_unique = run_instance_random_sweep(
            data[i], random_counts, n_ga=n_ga, n_random=n_random)
        for m in random_counts:
            per_instance.append({"instance": i, "n_random": m,
                                 "Random_saco": saco[m], "Random_unique": unique[m],
                                 "Random_actions": actions[m], "PDR_unique": pdr_unique})
            saco_accum[m].append(saco[m])
            uniq_accum[m].append(unique[m])
            act_accum[m].append(actions[m])

    summary = {
        m: {"saco": _summarise(saco_accum[m]),
            "unique": float(np.mean(uniq_accum[m])),
            "actions": float(np.mean(act_accum[m]))}
        for m in random_counts
    }

    if verbose:
        print(f"\n==== Random-only SACo vs. #random runs/instance: {data_path} ====")
        print(f"(unique action prefixes of the first m random trajectories, normalized by "
              f"the same PDR denominator; {len(data)} instances, {time.time() - t0:.1f}s)")
        for m in random_counts:
            s = summary[m]["saco"]
            print(f"  n_random={m:4d}  SACo {s['mean']:6.2f}±{s['std']:.2f} "
                  f"[{s['min']:.2f},{s['max']:.2f}]  "
                  f"(uniq prefixes~{summary[m]['unique']:.0f} of "
                  f"{summary[m]['actions']:.0f} actions)")

    if out:
        write_random_sweep_csv(per_instance, out)

    return per_instance, summary


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def write_csv(per_instance, out_path):
    """Per-instance CSV for one dataset (numeric-only, like the k-NN sheets)."""
    import csv
    fields = ["instance"] + [f"{t}_{f}" for t in DATASET_TYPES
                             for f in ("saco", "unique", "actions")]
    _ensure_dir(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_instance)
    print(f"Wrote per-instance coverage to {out_path}")


def summary_rows(dataset, per_instance, summary):
    """Long-format rows (one per dataset x type) for the cross-dataset sheet."""
    rows = []
    for t in DATASET_TYPES:
        s = summary[t]["saco"]
        rows.append({
            "dataset": dataset,
            "type": t,
            "n_instances": len(per_instance),
            "mean_trajectories": round(summary[t]["n_traj"], 2),
            "mean_unique_prefixes": round(summary[t]["unique"], 1),
            "mean_actions": round(summary[t]["actions"], 1),
            "saco_mean": s["mean"],
            "saco_std": s["std"],
            "saco_min": s["min"],
            "saco_max": s["max"],
        })
    return rows


def write_summary_csv(rows, out_path):
    """Aggregated coverage across all datasets/types (one sheet, human-readable)."""
    import csv
    fields = ["dataset", "type", "n_instances", "mean_trajectories",
              "mean_unique_prefixes", "mean_actions", "saco_mean", "saco_std",
              "saco_min", "saco_max"]
    _ensure_dir(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote combined coverage summary to {out_path}")


def write_random_sweep_csv(per_instance, out_path):
    """Per-(instance, n_random) sheet of the random-only sweep for one dataset."""
    import csv
    fields = ["instance", "n_random", "Random_saco", "Random_unique", "Random_actions",
              "PDR_unique"]
    _ensure_dir(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_instance)
    print(f"Wrote per-instance random sweep to {out_path}")


def random_sweep_summary_rows(dataset, per_instance, summary):
    """Long-format rows (one per dataset x n_random) for the cross-dataset sheet."""
    n_inst = len({rec["instance"] for rec in per_instance})
    rows = []
    for m in sorted(summary):
        s = summary[m]["saco"]
        rows.append({
            "dataset": dataset,
            "n_random": m,
            "n_instances": n_inst,
            "mean_unique_prefixes": round(summary[m]["unique"], 1),
            "mean_actions": round(summary[m]["actions"], 1),
            "saco_mean": s["mean"],
            "saco_std": s["std"],
            "saco_min": s["min"],
            "saco_max": s["max"],
        })
    return rows


def write_random_sweep_summary_csv(rows, out_path):
    """Aggregated random-only sweep across all datasets (one sheet)."""
    import csv
    fields = ["dataset", "n_random", "n_instances", "mean_unique_prefixes",
              "mean_actions", "saco_mean", "saco_std", "saco_min", "saco_max"]
    _ensure_dir(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote combined random-sweep summary to {out_path}")


# ---------------------------------------------------------------------------
# Convenience: all three SD1 sizes
# ---------------------------------------------------------------------------

def _size_tag(data_path):
    """'dataset/SD1_train_10_5_1000.npy' -> '10_5' (falls back to the file stem)."""
    stem = os.path.splitext(os.path.basename(data_path))[0]
    parts = stem.split("_")
    for i in range(len(parts) - 1):
        if parts[i].isdigit() and parts[i + 1].isdigit():
            return f"{parts[i]}_{parts[i + 1]}"
    return stem


def do_all_coverage_datasets(n_instances=None, out=None, datasets=None, **kw):
    """Run every dataset and, if ``out`` is given, write the CSVs.

    ``out`` is a base path (e.g. ``results/coverage.csv``): each dataset gets its own
    per-instance sheet ``results/coverage_10_5.csv`` and all datasets are aggregated
    into one ``results/coverage_summary.csv``.  Also prints the pooled SACo across all
    datasets, which is the number quoted in the paper.
    """
    datasets = datasets or DEFAULT_ALL_DATASETS
    base, ext = os.path.splitext(out) if out else (None, ".csv")

    results, all_rows = {}, []
    pooled = {t: [] for t in DATASET_TYPES}
    for path in datasets:
        print(f"\n########## {path} ##########")
        per_csv = f"{base}_{_size_tag(path)}{ext}" if out else None
        per_instance, summary = run_dataset(path, n_instances=n_instances, out=per_csv, **kw)
        results[path] = (per_instance, summary)
        all_rows += summary_rows(path, per_instance, summary)
        for t in DATASET_TYPES:
            pooled[t] += [rec[f"{t}_saco"] for rec in per_instance]

    print("\n==== SACo pooled over all datasets ====")
    for t in DATASET_TYPES:
        s = _summarise(pooled[t], round_val=2)
        print(f"  {t:8s} mean={s['mean']:.2f}  std={s['std']:.2f}")
        all_rows.append({
            "dataset": "ALL", "type": t, "n_instances": len(pooled[t]),
            "mean_trajectories": "", "mean_unique_prefixes": "", "mean_actions": "",
            "saco_mean": s["mean"], "saco_std": s["std"],
            "saco_min": s["min"], "saco_max": s["max"],
        })

    if out:
        write_summary_csv(all_rows, f"{base}_summary{ext}")
    return results


def do_all_random_sweep_datasets(random_counts=(1, 5, 10, 25, 50, 100), n_instances=None,
                                 out=None, datasets=None, **kw):
    """Random-only coverage sweep over every dataset; same CSV layout as the full run."""
    datasets = datasets or DEFAULT_ALL_DATASETS
    base, ext = os.path.splitext(out) if out else (None, ".csv")

    results, all_rows = {}, []
    for path in datasets:
        print(f"\n########## {path} ##########")
        per_csv = f"{base}_{_size_tag(path)}{ext}" if out else None
        per_instance, summary = run_random_sweep_dataset(
            path, random_counts=random_counts, n_instances=n_instances, out=per_csv, **kw)
        results[path] = (per_instance, summary)
        all_rows += random_sweep_summary_rows(path, per_instance, summary)

    if out:
        write_random_sweep_summary_csv(all_rows, f"{base}_summary{ext}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="dataset/SD1_train_20_10_500.npy",
                   help="Training dataset .npy (or 'all' for the three SD1 sizes)")
    p.add_argument("--n_instances", type=int, default=-1,
                   help="Number of instances to score (-1 = all; coverage is cheap)")
    p.add_argument("--n_ga", type=int, default=200, help="GA trajectories per instance (ga_pop)")
    p.add_argument("--n_random", type=int, default=100, help="Random trajectories per instance")
    p.add_argument("--out", default=None,
                   help="Results CSV path. With --data all it is a base path: one "
                        "per-instance sheet per dataset (<out>_10_5.csv, ...) plus a "
                        "combined <out>_summary.csv (default: coverage.csv)")
    p.add_argument("--random_sweep", action="store_true",
                   help="Score ONLY the Random source, at each dataset size in "
                        "--random_counts (ignores --n_random and the other types)")
    p.add_argument("--random_counts", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100],
                   help="Random trajectories/instance to sweep over with --random_sweep")
    return p


def main():
    args = build_argparser().parse_args()
    n_instances = None if (args.n_instances is None or args.n_instances < 0) else args.n_instances
    all_data = args.data.lower() == "all"

    if args.random_sweep:
        print(f"Random-only sweep over n_random in {args.random_counts}")
        if all_data:
            do_all_random_sweep_datasets(
                random_counts=args.random_counts, n_instances=n_instances,
                out=args.out or "coverage_random_sweep.csv", n_ga=args.n_ga)
        else:
            run_random_sweep_dataset(args.data, random_counts=args.random_counts,
                                     n_instances=n_instances, n_ga=args.n_ga,
                                     out=args.out)
        return

    if all_data:
        do_all_coverage_datasets(n_instances=n_instances, out=args.out or "coverage.csv",
                                 n_ga=args.n_ga, n_random=args.n_random)
    else:
        run_dataset(args.data, n_instances=n_instances, n_ga=args.n_ga,
                    n_random=args.n_random, out=args.out)


if __name__ == "__main__":
    main()
