"""Dataset format, legacy compatibility and trajectory selection (cdqac/data).

Run from the repo root with:  python -m unittest discover tests
"""
import os
import random
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fixtures as fx  # noqa: E402

from cdqac.data import format as fmt  # noqa: E402
from cdqac.data.dataset import (InstanceSplit, OfflineDataset, list_datasets,  # noqa: E402
                                load_datasets, validate)
from cdqac.data.legacy import load_legacy_train, read_legacy  # noqa: E402
from cdqac.data.select import select_action_lists  # noqa: E402
from data_generation.convert_legacy import convert  # noqa: E402

N_M = fx.N_M

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_selection(fn, data, config, seed=7):
    random.seed(seed)
    np.random.seed(seed)
    return fn(data, config)


def as_plain(selection):
    """(job_lengths, op_pts, action_lists) with arrays turned into lists, for comparison."""
    jl, pt, acts = selection
    return ([np.asarray(j).tolist() for j in jl], [np.asarray(p).tolist() for p in pt], acts)


class DataTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.legacy_dir = os.path.join(cls.tmp, "legacy", "fjsp")
        cls.train_path, cls.val_path = fx.write_legacy_files(cls.legacy_dir)
        cls.root = os.path.join(cls.tmp, "root")
        cls.dataset_dir = convert(cls.train_path, cls.val_path, "fjsp/toy_3x2", cls.root,
                                  replay_check=4)
        cls.legacy_instances = list(read_legacy(cls.train_path))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestFormat(DataTestCase):
    def test_split_round_trip_keeps_variable_op_counts(self):
        path = os.path.join(self.tmp, "split.npz")
        val = fx.make_legacy_val()
        entry = fmt.write_split(path, [v["JobLength"] for v in val], [v["OpPT"] for v in val])
        self.assertEqual(entry, {"num_instances": 3, "n_ops": [4, 6]})
        arrays = fmt.read_split(path)
        split = InstanceSplit(arrays["job_length"], arrays["op_pt"], arrays["n_ops"])
        for v, pt in zip(val, split.op_pts()):
            np.testing.assert_array_equal(pt, v["OpPT"])
        self.assertEqual(arrays["op_pt"].dtype, np.uint8)
        with np.load(path) as f:  # no pickles inside
            self.assertTrue(all(f[k].dtype != object for k in f.files))

    def test_ragged_source_stores_counts(self):
        path = os.path.join(self.tmp, "src.npz")
        entry = fmt.write_source(path, [[[0, 1], [2, 3]], [[4, 5]], []],
                                 [[5, 6], [7], []], names=["a", "b"])
        self.assertEqual(entry, {"num_instances": 3, "per_instance": [0, 2]})
        arrays = fmt.read_source(path)
        np.testing.assert_array_equal(arrays["counts"], [2, 1, 0])
        self.assertEqual(arrays["actions"].shape, (3, 2, 2))
        self.assertEqual(list(arrays["names"]), ["a", "b"])

    def test_write_source_rejects_mismatched_makespans(self):
        with self.assertRaises(ValueError):
            fmt.write_source(os.path.join(self.tmp, "bad.npz"), [[[0, 1]]], [[1, 2]])

    def test_converted_dataset_is_valid_and_listed(self):
        ds = OfflineDataset.from_dir(self.dataset_dir)
        self.assertEqual(validate(ds), [])
        self.assertEqual(list_datasets(self.root), ["fjsp/toy_3x2"])
        self.assertEqual(sorted(ds.source_names),
                         ["eps_pdr_0.1", "ga_hof", "ga_pop", "pdr", "random"])
        self.assertEqual(ds.coverage("eps_pdr_0.1"), 2)
        self.assertEqual(ds.coverage("random"), 4)

    def test_manifest_must_declare_the_action_mask(self):
        self.assertIs(OfflineDataset.from_dir(self.dataset_dir).manifest["action_mask"], True)
        bad_dir = os.path.join(self.tmp, "no_mask_copy")
        shutil.copytree(self.dataset_dir, bad_dir)
        manifest = fmt.read_manifest(bad_dir)
        manifest["action_mask"] = False
        fmt.write_manifest(bad_dir, manifest, update_checksums=False)
        problems = validate(OfflineDataset.from_dir(bad_dir))
        self.assertTrue(any("action_mask" in p for p in problems), problems)

    def test_validate_reports_corruption(self):
        bad_dir = os.path.join(self.tmp, "bad_copy")
        shutil.copytree(self.dataset_dir, bad_dir)
        src = fmt.read_source(os.path.join(bad_dir, "sources", "random.npz"))
        src["actions"][0, 0, 0] = 200  # outside n_j * n_m = 6
        np.savez_compressed(os.path.join(bad_dir, "sources", "random.npz"), **src)
        problems = validate(OfflineDataset.from_dir(bad_dir))
        self.assertTrue(any("sha256" in p for p in problems), problems)
        self.assertTrue(any("exceed" in p for p in problems), problems)

    def test_open_unknown_dataset_lists_available(self):
        with self.assertRaisesRegex(FileNotFoundError, "fjsp/toy_3x2"):
            OfflineDataset.open(self.root, "fjsp/nope")


class TestLegacy(DataTestCase):
    def test_pdr_keeps_only_masked_pdr_rules_in_file_order(self):
        ds = load_legacy_train(self.train_path)
        expected = [r.removesuffix("_masked") for r in fx.LEGACY_RULE_ORDER
                    if r.removesuffix("_masked") in fmt.PDR_RULES and r != fx.UNMASKED_RULE]
        self.assertTrue(all("masked" not in r for r in expected))
        self.assertEqual(ds.source("pdr").names, expected)
        self.assertEqual(len(expected), 15)
        converted = OfflineDataset.from_dir(self.dataset_dir)
        self.assertEqual(converted.source("pdr").names, expected)
        dropped = converted.manifest["provenance"]["dropped_pdr_rules"]
        self.assertEqual(len(dropped), 13)
        self.assertIn(fx.UNMASKED_RULE.removesuffix("_masked"), dropped)
        self.assertTrue(all("masked" not in r for r in dropped))

    def test_sources_match_the_legacy_lists(self):
        ds = OfflineDataset.from_dir(self.dataset_dir)
        for i, inst in enumerate(self.legacy_instances):
            self.assertEqual(ds.source("random").trajectories(i).tolist(), inst["random"])
            self.assertEqual(ds.source("random").makespans(i).tolist(), inst["random_makespan"])
            self.assertEqual(ds.source("ga_hof").trajectories(i).tolist(), inst["ga"])
            self.assertEqual(ds.source("ga_pop").trajectories(i).tolist(), inst["ga_pop"])
            self.assertEqual(ds.source("ga_pop").makespans(i).tolist(), inst["ga_pop_makespan"])
            if i < 2:
                self.assertEqual(ds.source("eps_pdr_0.1").trajectories(i).tolist(),
                                 inst["eps_random_0.1"])

    def test_eps_makespans_are_filled_by_replay(self):
        from cdqac.data.replay import replay_makespans
        ds = OfflineDataset.from_dir(self.dataset_dir)
        train, src = ds.split("train"), ds.source("eps_pdr_0.1")
        for i in range(src.num_instances):
            spans = replay_makespans(train.job_length[i], train.op_pts()[i], src.trajectories(i))
            np.testing.assert_array_equal(spans, src.makespans(i))


SELECTION_CASES = {
    "random_n": dict(use_random=True, n_random=3),
    "random_all": dict(use_random=True),
    "pdr_ga_pop": dict(use_dispatching=True, use_ga_pop=True),
    "ga_pop_weighted": dict(use_ga_pop=True, n_ga_pop=4, ga_pop_random=True),
    "ga_pop_first": dict(use_ga_pop=True, n_ga_pop=4),
    "ga_hof_n": dict(use_ga_hof=True, n_ga_hof=2),
    "ga_hof_more_than_stored": dict(use_ga_hof=True, n_ga_hof=50),
    "eps": dict(use_eps_pdr=True, n_eps_pdr=2, num_instances=2),
    "everything": dict(use_dispatching=True, use_random=True, n_random=2, use_ga_hof=True,
                       n_ga_hof=3, use_ga_pop=True, n_ga_pop=3, ga_pop_random=True,
                       use_eps_pdr=True, num_instances=2),
    "everything_dedup": dict(use_dispatching=True, use_random=True, use_ga_hof=True,
                             use_ga_pop=True, use_eps_pdr=True, num_instances=2,
                             remove_duplicate=True),
    "subset_dedup": dict(use_random=True, use_ga_pop=True, num_instances=3,
                         remove_duplicate=True),
}


class TestSelection(DataTestCase):
    def test_matches_original_train_logic_on_both_formats(self):
        legacy = load_legacy_train(self.train_path)
        converted = OfflineDataset.from_dir(self.dataset_dir)
        for case, overrides in SELECTION_CASES.items():
            with self.subTest(case=case):
                cfg = fx.selection_config(**overrides)
                expected = as_plain(run_selection(fx.original_train_selection,
                                                  self.legacy_instances, cfg))
                self.assertTrue(any(expected[2]), case)
                got_legacy = as_plain(run_selection(select_action_lists, legacy, cfg))
                got_new = as_plain(run_selection(select_action_lists, converted, cfg))
                self.assertEqual(got_legacy, expected)
                self.assertEqual(got_new, expected)

    def test_dedup_removes_ga_pop_duplicates(self):
        converted = OfflineDataset.from_dir(self.dataset_dir)
        _, _, acts = select_action_lists(converted, fx.selection_config(
            use_ga_pop=True, remove_duplicate=True))
        self.assertTrue(all(len(a) == 4 for a in acts))  # 7 stored, 3 are repeats

    def test_ga_hof_random_samples_without_replacement(self):
        converted = OfflineDataset.from_dir(self.dataset_dir)
        cfg = fx.selection_config(use_ga_hof=True, n_ga_hof=3, ga_hof_random=True)
        first = run_selection(select_action_lists, converted, cfg, seed=3)[2]
        again = run_selection(select_action_lists, converted, cfg, seed=3)[2]
        self.assertEqual(first, again)
        for i, acts in enumerate(first):
            hof = converted.source("ga_hof").trajectories(i).tolist()
            self.assertEqual(len(acts), 3)
            self.assertTrue(all(a in hof for a in acts))

    def test_partial_coverage_is_an_error(self):
        converted = OfflineDataset.from_dir(self.dataset_dir)
        with self.assertRaisesRegex(ValueError, "covers only the first 2"):
            select_action_lists(converted, fx.selection_config(use_eps_pdr=True))

    def test_missing_source_is_an_error(self):
        converted = OfflineDataset.from_dir(self.dataset_dir)
        with self.assertRaisesRegex(ValueError, "no 'eps_pdr_0.2' source"):
            select_action_lists(converted, fx.selection_config(use_eps_pdr=True,
                                                               eps_pdr_version="0.2"))

    def test_no_source_enabled_is_an_error(self):
        converted = OfflineDataset.from_dir(self.dataset_dir)
        with self.assertRaisesRegex(ValueError, "At least one of"):
            select_action_lists(converted, fx.selection_config())


def data_config(**overrides):
    cfg = dict(data_root=None, dataset="fjsp/toy_3x2", eval_dataset=None, eval_split="val",
               data_path=None, train_instance=None, eval_instance=None)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


class TestLoadDatasets(DataTestCase):
    def test_dataset_id_uses_its_val_split(self):
        train, evals = load_datasets(data_config(data_root=self.root))
        self.assertEqual(train.id, "fjsp/toy_3x2")
        self.assertEqual([len(e["OpPT"]) for e in evals], [5, 4, 6])

    def test_eval_dataset_override(self):
        _, evals = load_datasets(data_config(data_root=self.root, eval_dataset="fjsp/toy_3x2",
                                             eval_split="train"))
        self.assertEqual(len(evals), 4)

    def test_legacy_file_and_paired_val_file(self):
        train, evals = load_datasets(data_config(data_root=self.root, data_path=self.legacy_dir,
                                                 train_instance="train_3_2.npy"))
        self.assertIsNone(train.path)
        self.assertEqual(len(evals), 3)
        self.assertEqual(train.source("random").num_instances, 4)

    def test_missing_known_legacy_file_maps_to_dataset(self):
        from cdqac.data import dataset as dataset_mod
        key_train, key_val = ("fjsp", "train_9_9.npy"), ("fjsp", "val_9_9.npy")
        dataset_mod.LEGACY_TRAIN_FILES[key_train] = "fjsp/toy_3x2"
        dataset_mod.LEGACY_VAL_FILES[key_val] = "fjsp/toy_3x2"
        try:
            train, evals = load_datasets(data_config(
                data_root=self.root, data_path=self.legacy_dir, train_instance="train_9_9.npy"))
            self.assertEqual(train.id, "fjsp/toy_3x2")
            self.assertEqual(len(evals), 3)
            _, evals = load_datasets(data_config(data_root=self.root, eval_instance="val_9_9.npy",
                                                 data_path=self.legacy_dir))
            self.assertEqual(len(evals), 3)
        finally:
            del dataset_mod.LEGACY_TRAIN_FILES[key_train]
            del dataset_mod.LEGACY_VAL_FILES[key_val]

    def test_unknown_missing_legacy_file_is_an_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "not found"):
            load_datasets(data_config(data_root=self.root, data_path=self.legacy_dir,
                                      train_instance="train_8_8.npy"))


class TestCli(DataTestCase):
    def run_cli(self, *argv):
        from cdqac.data.__main__ import main
        return main(["--data_root", self.root, *argv])

    def test_list_info_validate(self):
        self.assertEqual(self.run_cli("list"), 0)
        self.assertEqual(self.run_cli("info", "fjsp/toy_3x2", "-v"), 0)
        self.assertEqual(self.run_cli("validate", "fjsp/toy_3x2", "--replay", "2"), 0)
        self.assertEqual(self.run_cli("validate", self.dataset_dir), 0)  # by directory

    def test_validate_rejects_actions_the_mask_forbids(self):
        from cdqac.data.replay import replay_makespans
        bad_dir = os.path.join(self.tmp, "forbidden_action")
        shutil.copytree(self.dataset_dir, bad_dir)
        train = OfflineDataset.from_dir(bad_dir).split("train")
        # an operation of job 0 that machine m cannot process: masked at step 0
        jl, pt = train.job_length[0], train.op_pts()[0]
        m = int(np.argmin(pt[0]))
        self.assertEqual(pt[0, m], 0, "fixture instance 0 needs an incompatible pair")
        src = fmt.read_source(os.path.join(bad_dir, "sources", "random.npz"))
        src["actions"][0, 0, 0] = 0 * N_M + m
        np.savez_compressed(os.path.join(bad_dir, "sources", "random.npz"), **src)
        with self.assertRaisesRegex(ValueError, "action mask forbids"):
            replay_makespans(jl, pt, src["actions"][0].astype(np.int64))
        self.assertEqual(self.run_cli("validate", bad_dir, "--no_checksums", "--replay", "1"), 1)

    def test_validate_fails_on_wrong_makespans(self):
        bad_dir = os.path.join(self.tmp, "bad_makespans")
        shutil.copytree(self.dataset_dir, bad_dir)
        src = fmt.read_source(os.path.join(bad_dir, "sources", "random.npz"))
        src["makespan"] = src["makespan"] + 1
        np.savez_compressed(os.path.join(bad_dir, "sources", "random.npz"), **src)
        self.assertEqual(self.run_cli("validate", bad_dir, "--no_checksums", "--replay", "1"), 1)


class TestPartialDownload(DataTestCase):
    """A download with --sources holds only some source files of the manifest."""

    def setUp(self):
        self.part = os.path.join(self.tmp, "partial_root", "fjsp", "toy_3x2")
        shutil.rmtree(self.part, ignore_errors=True)
        shutil.copytree(self.dataset_dir, self.part)
        os.remove(os.path.join(self.part, "sources", "pdr.npz"))

    def test_missing_sources_are_not_problems(self):
        ds = OfflineDataset.from_dir(self.part)
        self.assertEqual(validate(ds), [])
        self.assertFalse(ds.source_downloaded("pdr"))
        self.assertTrue(ds.source_downloaded("random"))
        from cdqac.data.__main__ import main
        self.assertEqual(main(["validate", self.part, "--replay", "1"]), 0)
        self.assertEqual(main(["info", self.part]), 0)

    def test_selecting_a_missing_source_says_how_to_get_it(self):
        ds = OfflineDataset.from_dir(self.part)
        with self.assertRaisesRegex(ValueError, "download_datasets.py fjsp/toy_3x2 --sources pdr"):
            select_action_lists(ds, fx.selection_config(use_dispatching=True))
        with self.assertRaisesRegex(FileNotFoundError, "not downloaded"):
            ds.source("pdr")
        _, _, acts = select_action_lists(ds, fx.selection_config(use_random=True, n_random=2))
        self.assertTrue(all(len(a) == 2 for a in acts))


REAL_LEGACY = os.path.join(REPO, "train_datasets", "fjsp", "train_10_5.npy")
REAL_DATASET = os.path.join(REPO, "train_datasets", "fjsp", "sd1_10x5")
REAL_MANIFEST = os.path.join(REAL_DATASET, "dataset.json")


@unittest.skipUnless(os.path.isfile(REAL_LEGACY) and os.path.isfile(REAL_MANIFEST),
                     "needs train_datasets/fjsp/train_10_5.npy and its converted fjsp/sd1_10x5")
class TestReleasedData(unittest.TestCase):
    """The converted 10x5 dataset selects exactly what the legacy file selects."""

    @classmethod
    def setUpClass(cls):
        cls.legacy_instances = list(read_legacy(REAL_LEGACY))
        cls.legacy = load_legacy_train(REAL_LEGACY)
        cls.converted = OfflineDataset.from_dir(REAL_DATASET)

    def test_selection_matches_original_train_logic(self):
        cases = {
            "paper_default": dict(use_random=True, n_random=100, num_instances=20,
                                  remove_duplicate=True),
            "dis_pop": dict(use_dispatching=True, use_ga_pop=True, num_instances=20,
                            remove_duplicate=True),
            "ga_pop_weighted": dict(use_ga_pop=True, n_ga_pop=50, ga_pop_random=True,
                                    num_instances=20),
            "eps": dict(use_eps_pdr=True, eps_pdr_version="0.2", n_eps_pdr=100,
                        num_instances=20, use_ga_hof=True, n_ga_hof=10),
        }
        for case, overrides in cases.items():
            with self.subTest(case=case):
                cfg = fx.selection_config(**overrides)
                expected = as_plain(run_selection(fx.original_train_selection,
                                                  self.legacy_instances, cfg))
                self.assertEqual(as_plain(run_selection(select_action_lists, self.legacy, cfg)),
                                 expected)
                self.assertEqual(as_plain(run_selection(select_action_lists, self.converted,
                                                        cfg)), expected)

    def test_sixteen_rules(self):
        self.assertEqual(sorted(self.converted.source("pdr").names), sorted(fmt.PDR_RULES))


if __name__ == "__main__":
    unittest.main()
