"""The analysis scripts give the same results from a legacy .npy file and its dataset.

Run from the repo root with:  python -m unittest discover tests
"""
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fixtures as fx  # noqa: E402

from analysis import coverage_action_prefix as coverage  # noqa: E402
from analysis import trajectory_quality_return as tq  # noqa: E402
from cdqac.data import load_instance_dicts  # noqa: E402
from cdqac.data.legacy import read_legacy  # noqa: E402
from data_generation.convert_legacy import convert  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class TestAnalysisOnDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        legacy_dir = os.path.join(cls.tmp, "legacy", "fjsp")
        cls.train_path, val_path = fx.write_legacy_files(legacy_dir, unmasked_rule=None)
        cls.dataset_dir = convert(cls.train_path, val_path, "fjsp/toy_3x2",
                                  os.path.join(cls.tmp, "root"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_instance_dicts_match_the_legacy_file(self):
        legacy = list(read_legacy(self.train_path))
        view = load_instance_dicts(self.dataset_dir)
        self.assertEqual(len(view), len(legacy))
        for old, new in zip(legacy, view):
            self.assertEqual(new["JobLength"], old["JobLength"])
            self.assertEqual(new["OpPT"], old["OpPT"])
            for key in ("random", "random_makespan", "ga", "ga_makespan", "ga_pop",
                        "ga_pop_makespan"):
                self.assertEqual(new[key], old[key], key)
            old_rules = {r.removesuffix("_masked"): r for r in old["rules"]}  # legacy names
            self.assertEqual(set(new["rules"]), set(old_rules) & set(tq.USED_PDR))
            for r in new["rules"]:
                self.assertEqual(new["rules"][r], old["rules"][old_rules[r]])
                self.assertEqual(new["rules_info"][r]["makespan"],
                                 old["rules_info"][old_rules[r]]["makespan"])
            self.assertEqual("eps_random_0.1" in new, "eps_random_0.1" in old)
            if "eps_random_0.1" in old:
                self.assertEqual(new["eps_random_0.1"], old["eps_random_0.1"])

    def test_legacy_files_get_the_dataset_rule_names(self):
        legacy_view = load_instance_dicts(self.train_path)
        dataset_view = load_instance_dicts(self.dataset_dir)
        self.assertEqual(legacy_view[0]["rules"], dataset_view[0]["rules"])
        self.assertTrue(all("masked" not in r for r in legacy_view[0]["rules"]))

    def test_val_split_and_known_legacy_names(self):
        self.assertEqual(len(load_instance_dicts(self.dataset_dir, split="val")), 3)
        with self.assertRaises(FileNotFoundError):
            load_instance_dicts(os.path.join(self.tmp, "nope", "train_1_1.npy"))

    def test_coverage_results_identical(self):
        kw = dict(n_instances=3, n_ga=4, n_random=3, verbose=False)
        self.assertEqual(coverage.run_dataset(self.train_path, **kw),
                         coverage.run_dataset(self.dataset_dir, **kw))

    def test_trajectory_quality_results_identical(self):
        kw = dict(n_instances=3, n_ga=4, n_random=3, verbose=False, eps_versions=("0.1",),
                  n_eps=3, eps_ga_n=2)
        old = tq.run_dataset(self.train_path, torch.device("cpu"), **kw)
        new = tq.run_dataset(self.dataset_dir, torch.device("cpu"), **kw)
        np.testing.assert_equal(old, new)
        self.assertEqual(max(r["makespan_check_err"] for r in new[2]), 0)

    def test_size_tags_of_dataset_ids(self):
        self.assertEqual(tq._size_tag("fjsp/sd1_15x10"), "15_10")
        self.assertEqual(tq._size_tag("dataset/SD1_train_20_10_500.npy"), "20_10")
        self.assertEqual(coverage._size_tag("fjsp/sd1_10x5"), "10_5")


if __name__ == "__main__":
    unittest.main()
