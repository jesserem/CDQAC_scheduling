"""Tests for the real-world dataset extension (realworld/).

Run from the repo root with:  python -m unittest discover tests
All data are synthetic fixtures in the exact published raw layouts
(tests/realworld_fixtures.py); nothing is downloaded.
"""
import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

from realworld.adapters._io import _NaNToNull
from realworld.episodes import (InstanceRecord, RoutingEpisodeSet, TrajectoryRecord, concat_sets)
from realworld.registry import get_dataset
from realworld.routing import metrics
from realworld.routing.behaviors import generate_behavior, make_behavior
from realworld.routing.buffer import RoutingBuffer
from realworld.routing.core import RewardConfig, RoutingEnv
from tests.realworld_fixtures import write_amazon, write_lade


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def toy_set(mode="matrix", dynamic=False, E=12, seed=0):
    rng = np.random.default_rng(seed)
    insts, trajs = [], []
    for e in range(E):
        n = int(rng.integers(5, 12))
        xy = rng.normal(size=(n, 2)).astype(np.float32)
        xy[0] = 0
        travel = (np.linalg.norm(xy[:, None] - xy[None], axis=-1) * 3 + rng.random((n, n))).astype(np.float32)
        np.fill_diagonal(travel, 0)
        zone = rng.integers(0, 3, size=n).astype(np.int32)
        zone[0] = -1
        rel = np.zeros(n, np.float32)
        if dynamic:
            rel[1:] = rng.uniform(0, 60, n - 1) * (rng.random(n - 1) < 0.5)
        tw_end = np.full(n, np.inf, np.float32)
        tw_end[1::2] = rng.uniform(30, 200, len(tw_end[1::2]))
        insts.append(InstanceRecord(coords=xy, zone=zone, zone2=zone // 2, release=rel, tw_end=tw_end,
                                    service=np.full(n, 2.0, np.float32), travel=travel if mode == "matrix" else None,
                                    speed=0.3, start_tod=480.0, meta=dict(id=e, group=e % 3)))
        order = (np.argsort(rel[1:], kind="stable") + 1) if dynamic else rng.permutation(np.arange(1, n))
        times = np.maximum.accumulate(np.cumsum(rng.uniform(3, 10, n - 1)) + rel[order]).astype(np.float32)
        trajs.append(TrajectoryRecord(inst=e, actions=order, times=times if dynamic else None, source="logged"))
    return RoutingEpisodeSet.from_records("toy", "train", mode, insts, trajs, meta=dict(part="toy"))


# ----------------------------------------------------------------------------- metrics
def naive_erp(a, s, m, g=1000.0, memo=None):
    """Direct recursive ERP (distance, #edits) from its definition."""
    memo = {} if memo is None else memo
    key = (len(a), len(s))
    if key in memo:
        return memo[key]
    if not s:
        out = (g * len(a), len(a))
    elif not a:
        out = (g * len(s), len(s))
    else:
        d1, c1 = naive_erp(a[1:], s[1:], m, g, memo)
        d2, c2 = naive_erp(a[1:], s, m, g, memo)
        d3, c3 = naive_erp(a, s[1:], m, g, memo)
        o1, o2, o3 = d1 + m[a[0]][s[0]], d2 + g, d3 + g
        d = min(o1, o2, o3)
        out = (d, c1 + (a[0] != s[0])) if d == o1 else ((d, c2 + 1) if d == o2 else (d, c3 + 1))
    memo[key] = out
    return out


class TestMetrics(unittest.TestCase):
    def test_amazon_score_against_recursive_definition(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            n = int(rng.integers(3, 25))
            xy = rng.random((n + 1, 2))
            mat = np.round(np.abs(xy[:, None] - xy[None]).sum(-1) * 1000)
            actual = [0] + list(rng.permutation(np.arange(1, n + 1)))
            prop = [0] + list(rng.permutation(np.arange(1, n + 1)))
            norm = metrics._normalize_matrix(mat)
            a, s = actual + [0], prop + [0]
            d, c = naive_erp(a, s, norm.tolist())
            expected = metrics.sequence_deviation(a[1:-1], s[1:-1]) * (d / c if c else 0.0)
            self.assertAlmostEqual(metrics.amazon_route_score(actual, prop, mat), expected, places=12)
            self.assertEqual(metrics.amazon_route_score(actual, actual, mat), 0.0)
        with self.assertRaises(ValueError):
            metrics.amazon_route_score([0, 1, 2], [1, 0, 2], mat[:3, :3])

    def test_route_prediction_metrics(self):
        label = [3, 1, 4, 2]
        self.assertEqual(metrics.kendall_rank_correlation(label, label), 1.0)
        self.assertEqual(metrics.kendall_rank_correlation(label[::-1], label), -1.0)
        self.assertEqual(metrics.location_square_deviation(label, label), 0.0)
        self.assertEqual(metrics.edit_distance(label, label), 0)
        self.assertEqual(metrics.edit_distance([1, 3, 4, 2], label), 2)
        self.assertEqual(metrics.hit_rate([1, 3, 2, 4], label, 2), 1.0)
        z = np.array([-1, 0, 0, 1, 1])
        self.assertEqual(metrics.zone_statistics([0, 1, 3, 2, 4], z), dict(zone_switches=3, zone_revisits=2))


class TestIO(unittest.TestCase):
    def test_nan_to_null_across_chunk_boundaries(self):
        raw = b'{"a": NaN, "b": [NaN, 1.5, "NA"], "Nb": NaN, "c": {"NAN": 2}}'
        for chunk in range(1, 8):
            w = _NaNToNull(io.BytesIO(raw))
            out = b""
            while True:
                part = w.read(chunk)
                if not part:
                    break
                out += part
            self.assertEqual(json.loads(out), {"a": None, "b": [None, 1.5, "NA"], "Nb": None, "c": {"NAN": 2}})
        self.assertEqual(_NaNToNull(io.BytesIO(raw)).read(0), b"")


# ------------------------------------------------------------------------- episodes
class TestEpisodeSet(unittest.TestCase):
    def test_roundtrip_select_concat_split(self):
        eps = toy_set("matrix")
        with tempfile.TemporaryDirectory() as d:
            back = RoutingEpisodeSet.load(eps.save(os.path.join(d, "x.npz")))
        for f in ("n_nodes", "coords", "zone", "traj_actions", "travel_flat", "travel_offsets"):
            np.testing.assert_array_equal(getattr(eps, f), getattr(back, f))
        self.assertEqual(eps.inst_meta, back.inst_meta)
        sub = eps.select_instances([5, 2])
        for i, e in enumerate([5, 2]):
            np.testing.assert_array_equal(sub.travel_matrix(i), eps.travel_matrix(e))
            self.assertEqual(sub.inst_meta[i]["id"], e)
        both = concat_sets([sub, eps.select_instances([0])])
        np.testing.assert_array_equal(both.travel_matrix(2), eps.travel_matrix(0))
        self.assertEqual(both.num_trajectories, 3)
        parts = eps.split_by({"a": 0.5, "b": 0.5}, group_key="group")
        groups = [{m["group"] for m in p.inst_meta} for p in parts.values()]
        self.assertFalse(groups[0] & groups[1])
        with self.assertRaises(ValueError):
            eps.add_trajectories([0], [np.array([1, 1, 2])], "bad")


# ------------------------------------------------------------------ buffer / env
class TestStateReconstruction(unittest.TestCase):
    def check(self, mode, dynamic):
        eps = toy_set(mode, dynamic)
        with quiet():
            generate_behavior(eps, "eps_nn_0.3", per_instance=1)
        rc = RewardConfig(zone_leave=5.0, lateness=1.0, latency=0.01)
        buf = RoutingBuffer(eps, rc, time_source="model", gamma=0.99, smdp_discount=dynamic)
        stats = buf.normalize_state()
        env = RoutingEnv(buf.rt, rc)
        for k in range(buf.K):
            env.reset(buf.inst[k:k + 1])
            for t in range(int(buf.lengths[k])):
                s_env = env.state()
                s_buf = buf.states_at(torch.tensor([k]), torch.tensor([t]))
                for i, (a, b) in enumerate(zip(s_env, s_buf)):
                    if a.dtype.is_floating_point:
                        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5, msg=f"slot {i} step {t}")
                    else:
                        self.assertTrue(torch.equal(a, b), f"slot {i} step {t}")
                self.assertFalse(bool(s_buf[6][0, buf.actions[k, t]]), "logged action masked")
                _, r, _ = env.step(buf.actions[k, t].view(1))
                self.assertAlmostEqual(float(r), float(buf.reward_kt[k, t]), places=4)
            self.assertTrue(bool(env.done.all()))
        self.assertIsNotNone(stats)

    def test_matrix_static(self):
        self.check("matrix", False)

    def test_haversine_dynamic(self):
        self.check("haversine", True)

    def test_logged_times_and_smdp_discounts(self):
        eps = toy_set("haversine", dynamic=True)
        buf = RoutingBuffer(eps, RewardConfig(), time_source="logged", gamma=0.99, smdp_discount=True)
        for k in range(buf.K):
            T = int(buf.lengths[k])
            e = int(buf.inst[k])
            # the sojourns tile the episode: sum = last logged completion - start
            self.assertAlmostEqual(float(buf.sojourn_kt[k, :T].sum()),
                                   float(eps.traj_times[k, T - 1]) - float(buf.clock[k, 0]) +
                                   (float(buf.clock[k, 0]) - float(_start(eps, e))), places=3)
            torch.testing.assert_close(buf.disc_kt[k, :T], 0.99 ** buf.sojourn_kt[k, :T])
        (state, nstate, a, r, d, mc), disc = buf.sample(64)
        self.assertEqual(disc.shape, (64, 1))
        self.assertTrue(bool((disc < 1).all()))


def _start(eps, e):
    n = eps.n_nodes[e]
    rel = eps.release[e, 1:n]
    return 0.0 if (rel <= 1e-4).any() else rel.min()


class TestBehaviors(unittest.TestCase):
    def test_behaviors_valid_and_ordered(self):
        eps = toy_set("matrix", E=20)
        with quiet():
            for b in ("random", "nn", "zone_nn", "softmin_3", "eps_zone_nn_0.2"):
                generate_behavior(eps, b)
        self.assertEqual(eps.num_trajectories, 20 * 6)
        buf = RoutingBuffer(eps, RewardConfig(return_to_depot=True), sources=None)
        per = buf.describe()["per_source"]
        self.assertLess(per["nn"]["return_mean"] * -1, per["random"]["return_mean"] * -1)
        with self.assertRaises(ValueError):
            make_behavior("teleport")


# ----------------------------------------------------------------- SMDP targets
class TestSMDPTargets(unittest.TestCase):
    def test_per_transition_discount_reaches_td_target(self):
        import pyrallis
        from realworld.config import make_config_class
        from realworld.nets import RoutingNets
        from train import METHODS
        eps = toy_set("haversine", dynamic=True)
        buf = RoutingBuffer(eps, RewardConfig(), gamma=0.9, smdp_discount=True)
        buf.normalize_state()
        cfg = pyrallis.parse(config_class=make_config_class(METHODS["cdqac"]["config"]),
                             args=["--device", "cpu", "--d_model", "16", "--n_heads", "2",
                                   "--n_dec_layers", "1", "--ff_dim", "16", "--num_quantiles", "4"])
        cfg.node_dim, cfg.edge_dim, cfg.global_dim = buf.rt.node_dim, buf.rt.edge_dim, buf.rt.global_dim
        with quiet():
            trainer, _ = METHODS["cdqac"]["build"](cfg, nets=RoutingNets)
        batch, disc = buf.sample(32)
        state, nstate, a, r, d, mc = batch
        log = {}
        trainer.discount = torch.zeros_like(disc)
        trainer.calculate_qrdqn_loss(state, nstate, a, r, d, log)
        self.assertAlmostEqual(log["mean_target"], float(r.mean()), places=5)
        l_scalar, l_tensor = {}, {}
        trainer.discount = 0.9
        trainer.calculate_qrdqn_loss(state, nstate, a, r, d, l_scalar)
        trainer.discount = torch.full_like(disc, 0.9)
        trainer.calculate_qrdqn_loss(state, nstate, a, r, d, l_tensor)
        self.assertAlmostEqual(l_scalar["mean_target"], l_tensor["mean_target"], places=5)


# ------------------------------------------------------------------- adapters
class TestAdapters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.raw_amazon = write_amazon(os.path.join(cls.tmp, "amazon"), n_train=8, n_test=4)
        cls.raw_lade = write_lade(os.path.join(cls.tmp, "lade"), couriers=3, days=10)
        with quiet():
            cls.amazon = get_dataset("amazon_lmrrc").prepare(cls.raw_amazon, os.path.join(cls.tmp, "rw", "amazon_lmrrc"),
                                                             val_fraction=0.25)
            cls.lade = get_dataset("lade_p").prepare(cls.raw_lade, os.path.join(cls.tmp, "rw", "lade_p"),
                                                     cities="jl,yt", min_stops=3)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_amazon_matches_raw_json(self):
        base = os.path.join(self.raw_amazon, "almrrc2021-data-training", "model_build_inputs")
        def load(f):
            with open(os.path.join(base, f)) as fh:
                return json.load(fh)
        routes, seqs, travel = load("route_data.json"), load("actual_sequences.json"), load("travel_times.json")
        eps = RoutingEpisodeSet.load(self.amazon["train"])
        for e, m in enumerate(eps.inst_meta):
            rid = m["route_id"]
            stops = routes[rid]["stops"]
            station = [s for s, v in stops.items() if v["type"] == "Station"][0]
            order = [station] + sorted(s for s in stops if s != station)
            tt = eps.travel_matrix(e)
            i, j = 1, len(order) - 1
            self.assertAlmostEqual(float(tt[i, j]), travel[rid][order[i]][order[j]] / 60.0, places=3)
            k = int(np.nonzero(eps.traj_inst == e)[0][0])
            got = [order[a] for a in eps.traj_actions[k, :eps.traj_len[k]]]
            want = [s for s, _ in sorted(seqs[rid]["actual"].items(), key=lambda kv: kv[1])][1:]
            self.assertEqual(got, want)
            self.assertEqual(m["route_score"], routes[rid]["route_score"])
            nan_zone = [n for n, s in enumerate(order) if isinstance(stops[s]["zone_id"], float)]
            self.assertTrue(all(eps.zone[e, n] == -1 for n in nan_zone))
        test = RoutingEpisodeSet.load(self.amazon["test"])
        self.assertIsNone(test.inst_meta[0]["route_score"])
        self.assertTrue((test.tw_end[np.isfinite(test.tw_end)] > -24 * 60).all())

    def test_lade_episodes(self):
        eps = RoutingEpisodeSet.load(self.lade["train"])
        dates = {p: {m["date"] for m in RoutingEpisodeSet.load(path).inst_meta} for p, path in self.lade.items()}
        self.assertFalse(dates["train"] & dates["test"])
        identity = 0
        for k in range(eps.num_trajectories):
            e, T = int(eps.traj_inst[k]), int(eps.traj_len[k])
            acts, times = eps.traj_actions[k, :T], eps.traj_times[k, :T]
            self.assertEqual(sorted(acts.tolist()), list(range(1, T + 1)))
            self.assertTrue((np.diff(times) >= -1e-4).all())
            self.assertTrue((eps.release[e, acts] <= times + 1e-4).all())
            identity += int((acts == np.arange(1, T + 1)).all())
            speed = np.linalg.norm(np.diff(eps.coords[e, acts], axis=0), axis=-1) * 1000 / np.maximum(np.diff(times), 0.1)
            self.assertFalse(((speed[:-1] > 600) & (speed[1:] > 600)).any(), "GPS drift stop kept")
        self.assertLess(identity, eps.num_trajectories)          # node order does not leak the route
        self.assertTrue((eps.node_feats[..., 0] > np.log(3) - 1e-6).any())  # merged packages exist
        self.assertIn("jl", eps.meta["calibration"])

    def test_train_and_evaluate_all_methods(self):
        import eval_realworld
        import train_realworld
        from train import METHODS
        root = os.path.join(self.tmp, "rw")
        ckpt = os.path.join(self.tmp, "ckpt")
        common = ["--data_root", root, "--device", "cpu", "--d_model", "16", "--n_heads", "2", "--n_dec_layers", "1",
                  "--ff_dim", "32", "--num_train_step_offline", "6", "--eval_freq", "3", "--batch_size", "16",
                  "--num_eval_instances", "3", "--stats_samples", "200", "--save_folder", ckpt]
        quant = lambda m: ["--num_quantiles", "4"] if "num_quantiles" in METHODS[m]["config"].__dataclass_fields__ else []
        with quiet():
            for m in sorted(METHODS):
                train_realworld.main(["--method", m, "--dataset", "lade_p", "--gamma", "0.99", "--name", m]
                                     + common + quant(m))
            train_realworld.main(["--method", "cdqac", "--dataset", "amazon_lmrrc", "--name", "amz"] + common)
        runs = {d: sorted(os.listdir(os.path.join(ckpt, d))) for d in os.listdir(ckpt)}
        self.assertEqual(len(runs["lade_p"]), len(METHODS))
        amz_run = os.path.join(ckpt, "amazon_lmrrc", runs["amazon_lmrrc"][0])
        out = os.path.join(self.tmp, "results")
        with quiet():
            eval_realworld.main([amz_run, "logged", "nn", "--split", "test", "--device", "cpu", "--save_path", out])
        with open(os.path.join(out, "amazon_lmrrc_test_summary.json")) as fh:
            summary = json.load(fh)
        self.assertEqual(summary["logged:driver"]["amazon_score"], 0.0)
        self.assertTrue(all("amazon_score" in v for v in summary.values()))


if __name__ == "__main__":
    unittest.main()
