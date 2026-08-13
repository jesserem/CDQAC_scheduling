"""Download the training/evaluation datasets from HuggingFace.

The datasets are hosted on the HuggingFace Hub because they are too large for
git. The training dataset is always downloaded; extra folders can be requested
as positional arguments. Run this script from the repository root:

    python scripts/download_datasets.py                    # train only
    python scripts/download_datasets.py eval               # train + eval
    python scripts/download_datasets.py eval checkpoints   # everything
    python scripts/download_datasets.py --output_dir data  # download into ./data

Alternatively, every dataset can be regenerated locally with the scripts in
``data_generation/`` (see README).
"""
import argparse

from huggingface_hub import snapshot_download

# TODO: set to the published dataset repository.
HF_REPO_ID = "jesserem/cdqac_datasets"

FOLDERS = {
    "train": "train_datasets/*",
    "eval": "eval_datasets/*",
    "checkpoints": "checkpoints/*",
}

OPTIONAL_FOLDERS = [name for name in FOLDERS if name != "train"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "extras",
        nargs="*",
        metavar="extra",
        help=f"Additional datasets to download besides train: {', '.join(OPTIONAL_FOLDERS)}",
    )
    parser.add_argument(
        "--output_dir",
        default=".",
        help="Directory to download the datasets into (default: current directory)",
    )
    parser.add_argument("--repo_id", default=HF_REPO_ID)
    args = parser.parse_args()

    unknown = [name for name in args.extras if name not in OPTIONAL_FOLDERS]
    if unknown:
        parser.error(
            f"unknown dataset(s): {', '.join(unknown)} (choose from {', '.join(OPTIONAL_FOLDERS)})"
        )

    names = ["train"] + list(dict.fromkeys(args.extras))
    patterns = [FOLDERS[name] for name in names]

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=args.output_dir,
        allow_patterns=patterns,
    )
    downloaded = ", ".join(FOLDERS[name].rstrip("/*") for name in names)
    print(f"Done. Downloaded {downloaded} into {args.output_dir}")


if __name__ == "__main__":
    main()
