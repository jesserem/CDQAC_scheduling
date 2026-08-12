"""
Summarize the average gap per experiment for one or more result folders.

A result folder holds *run* folders named
    <experiment>-<hash>_seed_<n>          e.g. p_freq_6-18e7cc61_seed_1
Several runs share the same <experiment> prefix (3 or 4 seeds each). Each run
folder holds the eval CSVs (SD1.csv / SD2.csv, mk.csv, edata.csv, rdata.csv,
vdata.csv), which have a `gap_det_offline` column (greedy / deterministic run)
and a `gap_stoch_offline` column (sampling / stochastic run).

Run folders are found at any depth, so both layouts work:
    ./p_freq/sd1/<run>/                      -- runs directly in the folder
    ./codac_experiments/<group>/<run>/       -- runs nested in a subfolder
In the nested case the subfolder is reported in a `group` column; the column is
omitted for flat layouts, where it would carry no information.

For every experiment and eval set this script:
  1. computes the mean gap over all instances within each seed's CSV, then
  2. averages those per-seed means across seeds (each seed weighted equally).

It then adds two aggregate columns:
  * generated  -- the generated instance set (SD1 or SD2)
  * benchmark  -- the mean over the benchmark sets (mk, edata, rdata, vdata),
                  each benchmark weighted equally (use --weight-by-instances to
                  weight each benchmark by its number of instances instead)

Every gap column comes with a matching `_std` column: the (sample) standard
deviation of the per-seed means across seeds, i.e. seed-to-seed variability, not
instance-to-instance. Pass --no-std to drop them.

Only the standard eval sets (SD1/SD2, mk, edata, rdata, vdata) are used; runs that
carry extra CSVs (e.g. mk_final.csv, SD1_30_10.csv) have those ignored so that all
experiments are compared on the same sets. Pass --all-evals or --evals to change that.

Usage:
    python summarize_gaps.py ./p_freq/sd1
    python summarize_gaps.py ./codac_experiments --out codac_gaps.csv
    python summarize_gaps.py ./p_freq/sd1 --out p_freq_sd1_gaps.csv
    python summarize_gaps.py ./p_freq/sd1 --per-seed          # + per-seed breakdown
    python summarize_gaps.py ./p_freq/sd1 --no-std            # means only
    python summarize_gaps.py ./p_freq/sd1 --evals SD1_final mk_final edata_final
    python summarize_gaps.py Freq_Exp_delay_6 Freq_Exp_delay_8   # old layout still works
"""
import os
import re
import glob
import argparse
import pandas as pd

GREEDY_COL = "gap_det_offline"      # deterministic / greedy rollout
SAMPLING_COL = "gap_stoch_offline"  # stochastic / sampling rollout

# What identifies one experiment: the top folder passed on the CLI, the subfolder
# it sits in (empty when runs live directly in the top folder), and the run-name
# prefix shared by its seeds.
KEYS = ["folder", "group", "experiment"]

# The eval sets that every run is expected to have. Some runs carry extra CSVs
# (e.g. mk_final.csv from the final instead of the best checkpoint, or an extra
# SD1_30_10.csv size); those are ignored by default so experiments stay
# comparable -- use --evals / --all-evals to pull them in.
DEFAULT_EVALS = ["SD1", "SD2", "mk", "edata", "rdata", "vdata"]

# Preferred display order for the eval sets; anything else is appended after.
EVAL_ORDER = DEFAULT_EVALS

# Eval sets that are generated instances rather than literature benchmarks
# (SD1, SD2, and variants such as SD1_30_10 or SD1_final).
GENERATED_RE = re.compile(r"^sd\d+(_|$)", re.IGNORECASE)

# Run dir name -> (experiment, seed). Matches "p_freq_6-18e7cc61_seed_1" and also
# the older "low_frequency_cql_td3_06_seed_1-18e7cc61_seed_1" (whose experiment
# part repeats the seed, which we strip so all seeds group together).
RUN_RE = re.compile(r"^(?P<exp>.+?)-[0-9a-f]{6,}_seed_(?P<seed>\d+)$")
TRAILING_SEED_RE = re.compile(r"_seed_\d+$")


def parse_run_dir(dirname):
    """Split a run folder name into (experiment_name, seed_label)."""
    m = RUN_RE.match(dirname)
    if not m:
        return dirname, dirname  # unrecognised layout: treat the dir as its own experiment
    exp = TRAILING_SEED_RE.sub("", m.group("exp"))
    return exp, f"seed_{m.group('seed')}"


def is_generated(eval_name):
    return bool(GENERATED_RE.match(eval_name))


def _eval_sort_key(name):
    return (EVAL_ORDER.index(name) if name in EVAL_ORDER else len(EVAL_ORDER), name)


def _num_token(tok):
    """Value of a digit run in a name.

    Plain integers sort numerically (p_freq_2 before p_freq_10). A leading zero
    means the digits encode a decimal fraction, as in cql_alpha_0025 = 0.025, so
    that alpha_0025 < alpha_005 < alpha_01 < alpha_1 rather than 25 < 5 < 1 < 1.
    """
    if tok.startswith("0") and len(tok) > 1:
        return float("0." + tok[1:])
    return float(tok)


def _exp_sort_key(name):
    """Natural sort on the digit runs inside an experiment / group name."""
    return tuple((1, _num_token(p), "") if p.isdigit() else (0, 0.0, p)
                 for p in re.split(r"(\d+)", name) if p)


def find_run_dirs(folder):
    """Every directory under `folder` that holds eval CSVs, at whatever depth.

    This covers both layouts: runs directly inside the folder (./p_freq/sd1/<run>)
    and runs nested one or more levels down (./codac_experiments/<group>/<run>).
    """
    runs = []
    for dirpath, _dirnames, filenames in os.walk(folder):
        if any(f.lower().endswith(".csv") for f in filenames):
            runs.append(dirpath)
    return sorted(runs)


def collect_per_seed(folder, keep_evals=None):
    """Return a long-form DataFrame: one row per (group, experiment, eval, seed).

    keep_evals: iterable of eval names to keep, or None to keep everything found.
    """
    rows, skipped = [], set()
    for run in find_run_dirs(folder):
        exp, seed = parse_run_dir(os.path.basename(run))
        # Where the run sits relative to `folder` -- "" when runs are directly
        # inside it, e.g. "abla_codac_alpha_0025" for the nested layout.
        group = os.path.relpath(os.path.dirname(run), folder).replace(os.sep, "/")
        if group == ".":
            group = ""
        for csv_path in sorted(glob.glob(os.path.join(run, "*.csv"))):
            eval_name = os.path.splitext(os.path.basename(csv_path))[0]
            df = pd.read_csv(csv_path)
            if GREEDY_COL not in df.columns or SAMPLING_COL not in df.columns:
                continue  # not an eval-result CSV
            if keep_evals is not None and eval_name not in keep_evals:
                skipped.add(eval_name)
                continue
            rows.append({
                "folder": os.path.basename(os.path.normpath(folder)),
                "group": group,
                "experiment": exp,
                "eval": eval_name,
                "seed": seed,
                "greedy_gap": df[GREEDY_COL].mean(),
                "sampling_gap": df[SAMPLING_COL].mean(),
                "n_instances": len(df),
            })
    if skipped:
        print(f"[INFO] Ignoring non-default eval CSVs in {os.path.normpath(folder)}: "
              f"{', '.join(sorted(skipped))} (use --evals/--all-evals to include them)")
    return pd.DataFrame(rows)


def add_aggregate_evals(per_seed_df, weight_by_instances=False):
    """Append pseudo-eval rows 'generated' and 'benchmark' per (experiment, seed)."""
    if per_seed_df.empty:
        return per_seed_df

    extra = []
    for key, grp in per_seed_df.groupby(KEYS + ["seed"]):
        gen = grp[grp["eval"].map(is_generated)]
        bench = grp[~grp["eval"].map(is_generated)]
        for label, sub in (("generated", gen), ("benchmark", bench)):
            if sub.empty:
                continue
            w = sub["n_instances"] if weight_by_instances else None
            row = dict(zip(KEYS + ["seed"], key))
            row.update({
                "eval": label,
                "greedy_gap": _mean(sub["greedy_gap"], w),
                "sampling_gap": _mean(sub["sampling_gap"], w),
                "n_instances": int(sub["n_instances"].sum()),
            })
            extra.append(row)
    return pd.concat([per_seed_df, pd.DataFrame(extra)], ignore_index=True)


def _mean(values, weights=None):
    if weights is None:
        return values.mean()
    return (values * weights).sum() / weights.sum()


def _sort_key(col):
    """Row sort: natural order on the name columns, eval-set order on `eval`."""
    if col.name == "eval":
        return col.map(_eval_sort_key)
    if col.name in ("group", "experiment"):
        return col.map(_exp_sort_key)
    return col


def summarize(per_seed_df):
    """Aggregate the per-seed means into one row per (experiment, eval)."""
    if per_seed_df.empty:
        return per_seed_df
    agg = (
        per_seed_df
        .groupby(KEYS + ["eval"])
        .agg(
            n_seeds=("seed", "nunique"),
            greedy_gap_mean=("greedy_gap", "mean"),
            greedy_gap_std=("greedy_gap", "std"),
            sampling_gap_mean=("sampling_gap", "mean"),
            sampling_gap_std=("sampling_gap", "std"),
        )
        .reset_index()
    )
    return agg.sort_values(by=KEYS + ["eval"], key=_sort_key).reset_index(drop=True)


def to_wide(summ, with_std=True):
    """One row per experiment; one column per (eval, metric)."""
    if summ.empty:
        return summ

    evals = sorted(summ["eval"].unique(), key=_eval_sort_key)
    # aggregates first, then the individual eval sets
    evals = [e for e in ("generated", "benchmark") if e in evals] + \
            [e for e in evals if e not in ("generated", "benchmark")]

    if with_std:
        metrics = ["greedy_gap_mean", "greedy_gap_std", "sampling_gap_mean", "sampling_gap_std"]
    else:
        metrics = ["greedy_gap_mean", "sampling_gap_mean"]
    short = {"greedy_gap_mean": "greedy", "greedy_gap_std": "greedy_std",
             "sampling_gap_mean": "sampling", "sampling_gap_std": "sampling_std"}

    wide = summ.pivot_table(index=KEYS, columns="eval", values=metrics)
    wide.columns = [f"{e}_{short[m]}" for m, e in wide.columns]
    wide = wide.reindex(columns=[f"{e}_{short[m]}" for e in evals for m in metrics])

    wide.insert(0, "n_seeds", summ.groupby(KEYS)["n_seeds"].max())
    wide = wide.reset_index().sort_values(by=KEYS, key=_sort_key).reset_index(drop=True)
    if (wide["group"] == "").all():  # flat layout: the column carries no information
        wide = wide.drop(columns=["group"])
    return wide


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="*", default=["./p_freq/sd1"],
                        help="Result folders to summarize (each holds run subfolders).")
    parser.add_argument("--out", default=None,
                        help="Path to write the wide summary table as CSV.")
    parser.add_argument("--out-long", default=None,
                        help="Path to write the long-form (folder, experiment, eval) table as CSV.")
    parser.add_argument("--per-seed", action="store_true",
                        help="Also print the per-seed breakdown.")
    parser.add_argument("--no-std", dest="with_std", action="store_false",
                        help="Drop the across-seed std columns (they are included by default).")
    parser.add_argument("--weight-by-instances", action="store_true",
                        help="Weight each benchmark by its instance count in the 'benchmark' "
                             "average (default: weight each benchmark set equally).")
    parser.add_argument("--evals", nargs="+", default=None,
                        help=f"Eval sets to include (default: {' '.join(DEFAULT_EVALS)}).")
    parser.add_argument("--all-evals", action="store_true",
                        help="Include every eval CSV found, including extras such as "
                             "*_final.csv or SD1_30_10.csv.")
    args = parser.parse_args()

    if args.all_evals:
        keep_evals = None
    else:
        keep_evals = set(args.evals) if args.evals else set(DEFAULT_EVALS)

    all_per_seed, all_summ = [], []
    for folder in args.folders:
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found, skipping: {folder}")
            continue
        per_seed = collect_per_seed(folder, keep_evals)
        if per_seed.empty:
            print(f"[WARN] No eval CSVs found in: {folder}")
            continue
        per_seed = add_aggregate_evals(per_seed, args.weight_by_instances)
        summ = summarize(per_seed)
        all_per_seed.append(per_seed)
        all_summ.append(summ)

        print("=" * 78)
        print(f"Folder: {os.path.normpath(folder)}")
        print("=" * 78)
        if args.per_seed:
            print("\n-- Per-seed mean gap --")
            cols = ["group", "experiment", "eval", "seed", "greedy_gap",
                    "sampling_gap", "n_instances"]
            if (per_seed["group"] == "").all():
                cols.remove("group")
            show = per_seed[cols].copy()
            show = show.sort_values(
                by=[c for c in ("group", "experiment", "eval", "seed") if c in cols],
                key=_sort_key,
            ).reset_index(drop=True)
            print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
            print()

        print("-- Average gap per experiment (mean across seeds) --")
        wide = to_wide(summ, with_std=args.with_std).drop(columns=["folder"])
        print(wide.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        print()

    if not all_summ:
        return

    combined = pd.concat(all_summ, ignore_index=True)
    if args.out:
        to_wide(combined, with_std=args.with_std).to_csv(args.out, index=False)
        print(f"Wide summary written to: {args.out}")
    if args.out_long:
        combined.to_csv(args.out_long, index=False)
        print(f"Long summary written to: {args.out_long}")


if __name__ == "__main__":
    main()
