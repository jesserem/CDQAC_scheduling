"""Dataset generators (data_generation/) write valid datasets and reproduce the released data.

Run from the repo root with:  python -m unittest discover tests
"""
import ast
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fixtures as fx  # noqa: E402

from cdqac.data.dataset import OfflineDataset, validate  # noqa: E402
from cdqac.data.replay import replay_makespans  # noqa: E402
from cdqac.data.writer import add_source, dataset_dir  # noqa: E402
from data_generation import add_ga_data, generate_datasets as gen  # noqa: E402
from data_generation.add_random_trajectories import random_source  # noqa: E402
from data_generation.convert_legacy import convert  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASED_ROOT = os.path.join(REPO, "train_datasets")


def released(dataset_id):
    path = dataset_dir(RELEASED_ROOT, dataset_id)
    return OfflineDataset.from_dir(path) if os.path.isfile(os.path.join(path, "dataset.json")) \
        else None


def assert_replay_matches(test, ds):
    train = ds.split("train")
    for name in ds.source_names:
        src = ds.source(name)
        for i in range(src.num_instances):
            spans = replay_makespans(train.job_length[i], train.op_pts()[i], src.trajectories(i))
            np.testing.assert_array_equal(spans, src.makespans(i), err_msg=f"{name} instance {i}")


class TestGenerate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_sd1_dataset_with_all_generated_sources(self):
        path = gen.generate("fjsp/sd1_3x2", "sd1", n_j=3, n_m=2, num_train=3, num_val=2, seed=5,
                            val_seed=1005, sources=["pdr", "random:4", "eps_pdr:0.1:3:2"],
                            data_root=self.root)
        ds = OfflineDataset.from_dir(path)
        self.assertEqual(validate(ds), [])
        self.assertEqual(ds.coverage("eps_pdr_0.1"), 2)
        self.assertEqual(len(ds.source("pdr").names), 16)
        self.assertEqual(len(set(ds.split("train").n_ops)), 1)  # train shares one job length
        self.assertEqual(ds.manifest["generator"]["seed"], 5)
        self.assertIs(ds.manifest["action_mask"], True)
        assert_replay_matches(self, ds)

    def test_taillard_and_sd2(self):
        jsp = OfflineDataset.from_dir(gen.generate(
            "jsp/taillard_3x3", "taillard", n_j=3, n_m=3, num_train=2, num_val=1, seed=1,
            val_seed=2, sources=["pdr", "random:2"], data_root=self.root))
        self.assertEqual(validate(jsp), [])
        self.assertEqual(jsp.source("pdr").names, ["MWR_SPT", "LWR_SPT", "LOPNR_SPT", "MOPNR_SPT"])
        self.assertTrue(((jsp.split("train").op_pt > 0).sum(2) == 1).all())
        sd2 = OfflineDataset.from_dir(gen.generate(
            "fjsp/sd2_3x2", "sd2", n_j=3, n_m=2, num_train=2, num_val=0, seed=1, val_seed=None,
            sources=["random:2"], data_root=self.root))
        self.assertEqual(validate(sd2), [])
        self.assertEqual(sd2.split_names, ["train"])

    def test_existing_dataset_and_source_are_protected(self):
        args = dict(n_j=3, n_m=2, num_train=1, num_val=0, seed=1, val_seed=None, sources=[],
                    data_root=self.root)
        path = gen.generate("fjsp/protect_3x2", "sd1", **args)
        with self.assertRaises(FileExistsError):
            gen.generate("fjsp/protect_3x2", "sd1", **args)
        split = OfflineDataset.from_dir(path).split("train")
        actions, makespans = random_source(split, 2, seed=0)
        add_source(path, "random", actions, makespans)
        with self.assertRaises(FileExistsError):
            add_source(path, "random", actions, makespans)
        add_source(path, "random", actions, makespans, overwrite=True)
        with self.assertRaisesRegex(ValueError, "operations"):
            add_source(path, "bad", [[[0, 1]]], [[1.0]])

    def test_parse_source(self):
        self.assertEqual(gen.parse_source("eps_pdr:0.2:100:500"),
                         ("eps_pdr", {"eps": "0.2", "num_runs": 100, "num_instances": 500}))
        with self.assertRaises(ValueError):
            gen.parse_source("random")


class TestActionMask(unittest.TestCase):
    def test_environments_reject_unmasked_actions(self):
        from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums
        from env.fjsp_env_various_op_nums import FJSPEnvForVariousOpNums
        for env_cls in (FJSPEnvForSameOpNums, FJSPEnvForVariousOpNums):
            with self.subTest(env=env_cls.__name__):
                env_cls(3, 2, mask_actions=True)
                with self.assertRaisesRegex(ValueError, "action mask"):
                    env_cls(3, 2, mask_actions=False)


class TestGaFiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        train_path, val_path = fx.write_legacy_files(os.path.join(self.tmp, "legacy"),
                                                     eps_cover=0)
        self.root = os.path.join(self.tmp, "root")
        self.path = convert(train_path, val_path, "fjsp/toy_3x2", self.root)
        # GA-style JSON files built from stored trajectories, so makespans are genuine
        ds = OfflineDataset.from_dir(self.path)
        self.ga_dir = os.path.join(self.tmp, "ga")
        os.makedirs(self.ga_dir)
        for i in range(3):
            hof, pop = ds.source("ga_hof"), ds.source("random")
            data = {"hof": {"action_list": hof.trajectories(i).tolist(),
                            "makespan": hof.makespans(i).tolist()},
                    "pop": {"action_list": pop.trajectories(i).tolist(),
                            "makespan": pop.makespans(i).tolist()}, "runtime": 1.0}
            with open(os.path.join(self.ga_dir, f"toy_3x2_GA_{i}.json"), "w") as f:
                json.dump(data, f)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_add_ga_data_writes_both_sources(self):
        argv = sys.argv
        sys.argv = ["add_ga_data", "--dataset", "fjsp/toy_3x2", "--data_root", self.root,
                    "--ga_folder", self.ga_dir, "--num_instances", "3", "--overwrite"]
        try:
            add_ga_data.main()
        finally:
            sys.argv = argv
        ds = OfflineDataset.from_dir(self.path)
        self.assertEqual(validate(ds), [])
        self.assertEqual(ds.coverage("ga_hof"), 3)
        self.assertEqual(ds.manifest["sources"]["ga_hof"]["order"], "makespan_asc")
        assert_replay_matches(self, ds)

    def test_missing_ga_file_is_reported(self):
        with self.assertRaisesRegex(FileNotFoundError, "toy_3x2_GA_3.json"):
            add_ga_data.read_ga_folder(self.ga_dir, "toy_3x2", 4)

    def test_run_ga_custom_reads_dataset_dirs_and_legacy_files(self):
        # run_GA_custom.py imports the external GA repo at the top, so load just its helpers
        src = open(os.path.join(REPO, "data_generation", "run_GA_custom.py"), encoding="utf-8").read()
        funcs = [n for n in ast.parse(src).body
                 if isinstance(n, ast.FunctionDef) and n.name in ("load_data", "dataset_name")]
        ns = {"np": np, "os": os}
        exec(compile(ast.Module(body=funcs, type_ignores=[]), "run_GA_custom", "exec"), ns)
        self.assertEqual(ns["dataset_name"](self.path), "toy_3x2")
        from_dir = ns["load_data"](self.path)
        legacy = ns["load_data"](os.path.join(self.tmp, "legacy", "train_3_2.npy"))
        self.assertEqual(len(from_dir), len(legacy))
        for a, b in zip(from_dir, legacy):
            np.testing.assert_array_equal(a["JobLength"], b["JobLength"])
            np.testing.assert_array_equal(a["OpPT"], b["OpPT"])


class TestReproducesReleasedData(unittest.TestCase):
    """The generators rebuild the released data wherever its seeds are known."""

    def check_instances(self, dataset_id, generator, n_j, n_m, split, seed, k=5):
        ds = released(dataset_id)
        if ds is None:
            self.skipTest(f"{dataset_id} not converted")
        jl, pt = gen.make_instances(generator, n_j, n_m, k, seed, split)
        stored = ds.split(split)
        for i in range(k):
            np.testing.assert_array_equal(jl[i], stored.job_length[i])
            np.testing.assert_array_equal(pt[i], stored.op_pts()[i])

    def test_train_instances_seed_101(self):
        for dataset_id, generator, n_j, n_m in [("fjsp/sd1_10x5", "sd1", 10, 5),
                                                ("fjsp/sd1_15x10", "sd1", 15, 10),
                                                ("fjsp/sd1_20x10", "sd1", 20, 10),
                                                ("jsp/taillard_10x5", "taillard", 10, 5),
                                                ("jsp/taillard_15x10", "taillard", 15, 10)]:
            with self.subTest(dataset=dataset_id):
                self.check_instances(dataset_id, generator, n_j, n_m, "train", 101)

    def test_jsp_val_instances_seed_1101(self):
        for dataset_id, n_j, n_m in [("jsp/taillard_10x5", 10, 5), ("jsp/taillard_15x10", 15, 10)]:
            with self.subTest(dataset=dataset_id):
                self.check_instances(dataset_id, "taillard", n_j, n_m, "val", 1101)

    def test_pdr_source(self):
        for dataset_id, machine_rules in [("fjsp/sd1_10x5", ["EST", "LST", "LPT", "SPT"]),
                                          ("jsp/taillard_10x5", ["SPT"])]:
            with self.subTest(dataset=dataset_id):
                ds = released(dataset_id)
                if ds is None:
                    self.skipTest(f"{dataset_id} not converted")
                actions, makespans, names = gen.pdr_source(ds.split("train").head(3), machine_rules)
                src = ds.source("pdr")
                self.assertEqual(names, src.names)
                for i in range(3):
                    self.assertEqual(actions[i], src.trajectories(i).tolist())
                    self.assertEqual(makespans[i], src.makespans(i).tolist())

    def test_random_source_seed_100(self):
        # one random stream runs over all instances, so only instance 0 is cheap to check
        for dataset_id in ["fjsp/sd1_10x5", "jsp/taillard_10x5"]:
            with self.subTest(dataset=dataset_id):
                ds = released(dataset_id)
                if ds is None:
                    self.skipTest(f"{dataset_id} not converted")
                actions, makespans = random_source(ds.split("train").head(1), 300, seed=100)
                np.testing.assert_array_equal(actions[0], ds.source("random").trajectories(0))
                np.testing.assert_array_equal(makespans[0], ds.source("random").makespans(0))


if __name__ == "__main__":
    unittest.main()
