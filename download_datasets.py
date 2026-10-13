"""Download the training datasets (and optionally evaluation data / checkpoints) from HuggingFace.

The datasets are hosted on the HuggingFace Hub because they are too large for git. Run this
script from the repository root:

    python download_datasets.py                                 # every training dataset
    python download_datasets.py fjsp/sd1_10x5                   # one dataset, all its sources
    python download_datasets.py fjsp/sd1_10x5 --sources random  # only what the paper default uses
    python download_datasets.py eval checkpoints                # + evaluation data and checkpoints
    python download_datasets.py --legacy --no_datasets          # only the old single-file .npy files
    python download_datasets.py fjsp/sd1_10x5 --dry_run         # show what would be downloaded

Datasets land in <output_dir>/train_datasets/<problem>/<name>/, where train.py looks for them
(``--data_root ./train_datasets``); the legacy files land in train_datasets/legacy/{fjsp,jsp}/.
A dataset's manifest and instance splits are always
downloaded; ``--sources`` limits which trajectory sources come along. Check a download with
``python -m cdqac.data list``.

Alternatively, every dataset can be regenerated locally with the scripts in
``data_generation/`` (see README).
"""
import argparse

HF_REPO_ID = "jesserem/cdqac_datasets"

EXTRAS = {
    "eval": "eval_datasets/*",
    "checkpoints": "checkpoints/*",
}
# The original single-file .npy datasets (same data as the dataset directories)
LEGACY_PATTERNS = ["train_datasets/legacy/*"]


def download_plan(dataset_ids, sources=None, extras=(), legacy=False, datasets=True):
    """List of (allow_patterns, ignore_patterns) passes for snapshot_download.

    HuggingFace patterns are fnmatch patterns whose ``*`` also matches ``/``, so selecting a
    subset of sources takes two passes: everything outside sources/, then the chosen sources.
    """
    roots = [f"train_datasets/{d}" for d in dataset_ids] or ["train_datasets/*/*"]
    passes = []
    if not datasets:
        pass
    elif sources is None:
        passes.append(([f"{r}/*" for r in roots], list(LEGACY_PATTERNS)))
    else:
        passes.append(([f"{r}/*" for r in roots],
                       [f"{r}/sources/*" for r in roots] + LEGACY_PATTERNS))
        if sources:
            passes.append(([f"{r}/sources/{s}.npz" for r in roots for s in sources], []))
    if legacy:
        passes.append((list(LEGACY_PATTERNS), []))
    if extras:
        passes.append(([EXTRAS[e] for e in extras], []))
    return passes


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "names",
        nargs="*",
        metavar="name",
        help=f"dataset ids (e.g. fjsp/sd1_10x5; default: all) and/or extras: {', '.join(EXTRAS)}",
    )
    parser.add_argument("--sources", nargs="*", default=None,
                        help="only these trajectory sources (e.g. random pdr); default: all")
    parser.add_argument("--legacy", action="store_true",
                        help="also download the legacy single-file .npy datasets")
    parser.add_argument("--no_datasets", action="store_true",
                        help="skip the training datasets (e.g. to fetch only eval/checkpoints)")
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to download into (default: current directory)",
    )
    parser.add_argument("--repo_id", default=HF_REPO_ID)
    parser.add_argument("--dry_run", action="store_true", help="print the download patterns only")
    args = parser.parse_args()

    extras = [n for n in args.names if n in EXTRAS]
    dataset_ids = [n for n in args.names if n not in EXTRAS]
    bad = [d for d in dataset_ids if d.count("/") != 1]
    if bad:
        parser.error(f"not a dataset id or extra: {', '.join(bad)} "
                     f"(ids look like fjsp/sd1_10x5; extras: {', '.join(EXTRAS)})")

    passes = download_plan(dataset_ids, args.sources, extras, args.legacy,
                           datasets=not args.no_datasets)
    for allow, ignore in passes:
        print(f"allow {allow}" + (f"  ignore {ignore}" if ignore else ""))
    if args.dry_run:
        return

    from huggingface_hub import snapshot_download
    for allow, ignore in passes:
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=args.output_dir,
            allow_patterns=allow,
            ignore_patterns=ignore or None,
        )
    print(f"Done. Downloaded into {args.output_dir}; list the datasets with "
          f"`python -m cdqac.data list`")


if __name__ == "__main__":
    main()
