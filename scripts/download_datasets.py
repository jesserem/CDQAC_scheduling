"""Download the training/evaluation datasets from HuggingFace.

The packed instance datasets (FJSP: ``dataset/``, JSP: ``jsp_dataset/``) and
the raw GA trajectory JSONs (``GA_data/``) are hosted on the HuggingFace Hub
because they are too large for git. Run this script from the repository root:

    python scripts/download_datasets.py            # FJSP datasets only
    python scripts/download_datasets.py --all      # FJSP + JSP + GA_data

Alternatively, every dataset can be regenerated locally with the scripts in
``data_generation/`` (see README).
"""
import argparse

from huggingface_hub import snapshot_download

# TODO: set to the published dataset repository.
HF_REPO_ID = "jvanremmerden/cdqac-fjsp-datasets"

FOLDERS = {
    "fjsp": "dataset/*",
    "jsp": "jsp_dataset/*",
    "ga": "GA_data/*",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Download JSP and GA data as well")
    parser.add_argument("--repo_id", default=HF_REPO_ID)
    args = parser.parse_args()

    patterns = [FOLDERS["fjsp"]]
    if args.all:
        patterns += [FOLDERS["jsp"], FOLDERS["ga"]]

    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=".",
        allow_patterns=patterns,
    )
    print("Done. Datasets are in ./dataset" + (", ./jsp_dataset and ./GA_data" if args.all else ""))


if __name__ == "__main__":
    main()
