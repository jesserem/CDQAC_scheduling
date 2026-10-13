"""download_datasets.py selects the right files (checked offline with huggingface_hub's filter).

Run from the repo root with:  python -m unittest discover tests
"""
import unittest

import download_datasets as dl

# The Hub during the transition: legacy files at their old paths and in legacy/, plus datasets
OLD_PATHS = ["train_datasets/fjsp/train_10_5.npy", "train_datasets/jsp/val_10_5.npy"]
LEGACY = ["train_datasets/legacy/fjsp/train_10_5.npy", "train_datasets/legacy/fjsp/val_20_10.npy",
          "train_datasets/legacy/jsp/val_10_5.npy"]
REPO_FILES = OLD_PATHS + LEGACY + [
    "train_datasets/fjsp/sd1_10x5/dataset.json", "train_datasets/fjsp/sd1_10x5/train.npz",
    "train_datasets/fjsp/sd1_10x5/val.npz", "train_datasets/fjsp/sd1_10x5/sources/random.npz",
    "train_datasets/fjsp/sd1_10x5/sources/pdr.npz",
    "train_datasets/jsp/taillard_10x5/dataset.json", "train_datasets/jsp/taillard_10x5/train.npz",
    "train_datasets/jsp/taillard_10x5/sources/random.npz",
    "eval_datasets/a.npy", "checkpoints/b.pt", "README.md",
]


def fetched(plan):
    try:
        from huggingface_hub.utils import filter_repo_objects
    except ImportError:  # pragma: no cover
        raise unittest.SkipTest("huggingface_hub not installed")
    out = set()
    for allow, ignore in plan:
        out |= set(filter_repo_objects(REPO_FILES, allow_patterns=allow,
                                       ignore_patterns=ignore or None))
    return out


class TestDownloadPlan(unittest.TestCase):
    def test_default_is_every_dataset_but_not_legacy_files(self):
        got = fetched(dl.download_plan([]))
        expected = {f for f in REPO_FILES if f.count("/") >= 3 and f not in LEGACY}
        self.assertEqual(got, expected)
        self.assertEqual(fetched(dl.download_plan([], sources=["random"])) & set(LEGACY), set())

    def test_one_dataset_with_selected_sources(self):
        got = fetched(dl.download_plan(["fjsp/sd1_10x5"], sources=["random"]))
        self.assertEqual(got, {"train_datasets/fjsp/sd1_10x5/dataset.json",
                               "train_datasets/fjsp/sd1_10x5/train.npz",
                               "train_datasets/fjsp/sd1_10x5/val.npz",
                               "train_datasets/fjsp/sd1_10x5/sources/random.npz"})

    def test_legacy_and_extras_without_datasets(self):
        got = fetched(dl.download_plan([], extras=["eval", "checkpoints"], legacy=True,
                                       datasets=False))
        self.assertEqual(got, set(LEGACY) | {"eval_datasets/a.npy", "checkpoints/b.pt"})


if __name__ == "__main__":
    unittest.main()
