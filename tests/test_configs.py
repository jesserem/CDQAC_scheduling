"""Every method's TrainConfig inherits the shared settings of base_config.py (CDQAC's defaults).

Run from the repo root with:  python -m unittest discover tests
"""
import dataclasses
import importlib
import os
import tempfile
import unittest

import pyrallis

from cdqac.configs.base_config import BaseConfig

METHODS = ["cdqac", "bc", "d_msac", "iql", "mqrdqn", "td3_awr"]
IDENTITY = {"save_folder", "project", "group", "name"}  # where a method's runs are stored/logged


def config_class(method):
    return importlib.import_module(f"cdqac.configs.{method}_config").TrainConfig


def defaults(cls):
    return {f.name: f.default for f in dataclasses.fields(cls)}


class TestConfigs(unittest.TestCase):
    def test_every_method_shares_the_base_defaults(self):
        base = defaults(BaseConfig)
        for method in METHODS:
            with self.subTest(method=method):
                cls = config_class(method)
                self.assertTrue(issubclass(cls, BaseConfig))
                values = defaults(cls)
                for field, default in base.items():
                    if field not in IDENTITY:
                        self.assertEqual(values[field], default, field)

    def test_method_files_only_add_their_own_settings(self):
        shared = set(defaults(BaseConfig))
        for method in METHODS:
            with self.subTest(method=method):
                own = set(config_class(method).__dict__.get("__annotations__", {}))
                self.assertEqual(own & shared, own & IDENTITY)

    def test_dataset_options_work_for_every_method(self):
        args = ["--dataset", "jsp/taillard_10x5", "--use_dispatching", "true", "--n_random", "5",
                "--use_eps_pdr", "true", "--eval_dataset", "jsp/taillard_15x10"]
        for method in METHODS:
            with self.subTest(method=method):
                cfg = pyrallis.parse(config_class=config_class(method), args=args)
                self.assertEqual((cfg.dataset, cfg.use_dispatching, cfg.n_random, cfg.use_eps_pdr,
                                  cfg.eval_dataset),
                                 ("jsp/taillard_10x5", True, 5, True, "jsp/taillard_15x10"))
                self.assertTrue(cfg.name.startswith(defaults(config_class(method))["name"] + "-"))

    def test_configs_written_before_the_unification_still_load(self):
        old_bc = ("use_dispatching: true\nuse_ga_pop: true\nuse_random: false\ntrain_epochs: 59\n"
                  "save_folder: checkpoints_bc\ndata_path: ./train_datasets/fjsp\n"
                  "train_instance: train_10_5.npy\neval_instance: val_10_5.npy\nuse_mask: true\n"
                  "p_lr: 3.0e-05\nnormalize: false\nn_step: 1\n")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yml")
            with open(path, "w") as f:
                f.write(old_bc)
            cfg = pyrallis.parse(config_class=config_class("bc"), config_path=path, args=[])
        self.assertEqual((cfg.use_dispatching, cfg.use_random, cfg.train_epochs, cfg.p_lr),
                         (True, False, 59, 3e-5))


if __name__ == "__main__":
    unittest.main()
