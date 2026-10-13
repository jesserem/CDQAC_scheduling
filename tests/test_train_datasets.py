"""train.py trains identically from a legacy .npy file and from its converted dataset.

Runs train.py in subprocesses on a tiny synthetic dataset (about a minute on CPU).
Run from the repo root with:  python -m unittest discover tests
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data_fixtures as fx  # noqa: E402

from data_generation.convert_legacy import convert  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMMON = ["--device", "cpu", "--num_train_step_offline", "20", "--eval_freq", "10",
          "--seed", "4", "--num_instances", "2"]


def flat(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from flat(v, f"{prefix}{k}/")
    else:
        yield prefix, obj


class TestTrainOnDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.legacy_dir = os.path.join(cls.tmp, "legacy", "fjsp")
        train_path, val_path = fx.write_legacy_files(cls.legacy_dir)
        cls.root = os.path.join(cls.tmp, "root")
        convert(train_path, val_path, "fjsp/toy_3x2", cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def train(self, tag, args):
        out = os.path.join(self.tmp, tag)
        proc = subprocess.run([sys.executable, "train.py", *COMMON, *args, "--save_folder", out],
                              cwd=REPO, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        (ckpt,) = glob.glob(os.path.join(out, "*", "model_final_offline.pt"))
        return dict(flat(torch.load(ckpt, map_location="cpu", weights_only=False))), proc.stdout

    def assert_same_training(self, method_args, tag):
        legacy, log = self.train(f"{tag}_legacy", method_args + [
            "--data_path", self.legacy_dir, "--train_instance", "train_3_2.npy"])
        self.assertIn("legacy file", log)
        converted, log = self.train(f"{tag}_dataset", method_args + [
            "--data_root", self.root, "--dataset", "fjsp/toy_3x2"])
        self.assertIn("Training dataset: fjsp/toy_3x2", log)
        self.assertEqual(legacy.keys(), converted.keys())
        for k in legacy:
            if torch.is_tensor(legacy[k]):
                self.assertTrue(torch.equal(legacy[k], converted[k]), k)

    def test_cdqac_with_every_source(self):
        self.assert_same_training(
            ["--use_dispatching", "true", "--use_random", "true", "--n_random", "3",
             "--use_ga_pop", "true", "--n_ga_pop", "3", "--ga_pop_random", "true",
             "--use_ga_hof", "true", "--use_eps_pdr", "true"], "cdqac")

    def test_td3_awr_disk_buffer(self):
        self.assert_same_training(["--method", "td3_awr", "--use_disk_buffer", "true",
                                   "--num_workers", "0", "--use_dispatching", "true"], "td3")

    def test_every_method_trains_with_the_shared_defaults(self):
        for method in ["cdqac", "bc", "iql", "dmsac", "mqrdqn", "td3_awr"]:
            with self.subTest(method=method):
                out = os.path.join(self.tmp, f"defaults_{method}")
                proc = subprocess.run([sys.executable, "train.py", "--method", method, *COMMON,
                                       "--n_random", "3", "--data_root", self.root,
                                       "--dataset", "fjsp/toy_3x2", "--save_folder", out],
                                      cwd=REPO, capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
                self.assertEqual(len(glob.glob(os.path.join(out, "*", "model_final_offline.pt"))), 1)

    def test_unmasked_training_is_rejected(self):
        proc = subprocess.run([sys.executable, "train.py", "--use_mask", "false", "--data_root",
                               self.root, "--dataset", "fjsp/toy_3x2", "--save_folder", ""],
                              cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("always uses the action mask", proc.stderr)

    def test_forbidden_action_in_the_data_is_rejected(self):
        import numpy as np
        from cdqac.data import format as fmt
        bad_root = os.path.join(self.tmp, "forbidden_root")
        bad = os.path.join(bad_root, "fjsp", "toy_3x2")
        shutil.copytree(os.path.join(self.root, "fjsp", "toy_3x2"), bad)
        with np.load(os.path.join(bad, "train.npz")) as f:
            pt = f["op_pt"][0]
        machine = int(np.argmin(pt[0]))  # a machine that cannot process job 0's first operation
        self.assertEqual(pt[0, machine], 0)
        src = fmt.read_source(os.path.join(bad, "sources", "random.npz"))
        src["actions"][0, 0, 0] = machine  # job 0 on that machine, at step 0
        np.savez_compressed(os.path.join(bad, "sources", "random.npz"), **src)
        proc = subprocess.run([sys.executable, "train.py", *COMMON, "--n_random", "1",
                               "--data_root", bad_root, "--dataset", "fjsp/toy_3x2",
                               "--save_folder", os.path.join(self.tmp, "forbidden")],
                              cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("action mask forbids", proc.stderr)

    def test_partial_source_fails_before_training(self):
        proc = subprocess.run([sys.executable, "train.py", *COMMON[:-2], "--num_instances", "4",
                               "--use_eps_pdr", "true", "--data_root", self.root,
                               "--dataset", "fjsp/toy_3x2", "--save_folder",
                               os.path.join(self.tmp, "partial")],
                              cwd=REPO, capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("covers only the first 2", proc.stderr)


if __name__ == "__main__":
    unittest.main()
