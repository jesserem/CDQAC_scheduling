"""Head input sizes must match the embeddings they are fed, for any embedding size d.

Run from the repo root with:  python -m unittest discover tests
"""
import unittest

import torch
import torch.nn as nn

from cdqac.network.main_model import ActorNet, IQL_value, QRDQNNet, mQRDQNNet

B, J, M = 3, 4, 5
N = 3 * J
FEA_J, FEA_M = 10, 8
NUM_QUANTILES = 7
EMBEDDING_DIMS = (4, 8, 16, 32)
DAN_KWARGS = dict(fea_j_input_dim=FEA_J, fea_m_input_dim=FEA_M, num_heads_OAB=(4, 4), num_heads_MAB=(4, 4))


def make_inputs():
    candidate = torch.stack([torch.randperm(N)[:J] for _ in range(B)])
    dynamic_pair_mask = torch.rand(B, J, M) > 0.6
    dynamic_pair_mask[:, 0, 0] = False  # keep at least one legal action per state
    return dict(
        fea_j=torch.randn(B, N, FEA_J),
        op_mask=torch.zeros(B, N, 3),
        candidate=candidate,
        fea_m=torch.randn(B, M, FEA_M),
        mch_mask=torch.ones(B, M, M),
        comp_idx=(torch.rand(B, M, M, J) > 0.5).float(),
        dynamic_pair_mask=dynamic_pair_mask,
        fea_pairs=torch.randn(B, J, M, 8),
    )


def embedding_inputs(inputs):
    return {k: v for k, v in inputs.items() if k != "dynamic_pair_mask"}


def in_features(module):
    return next(m for m in module.modules() if isinstance(m, nn.Linear)).in_features


def make_q_net(cls, d, use_adv_net):
    kwargs = dict(DAN_KWARGS, layer_fea_output_dim=(32, d), num_mlp_layers_critic=3, hidden_dim_critic=64,
                  num_quantiles=NUM_QUANTILES, use_adv_net=use_adv_net)
    if cls is QRDQNNet:
        kwargs["n_critics"] = 2
    return cls(**kwargs).eval()


class TestHeadInputDims(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_qrdqn_net(self):
        for d in EMBEDDING_DIMS:
            for use_adv_net in (True, False):
                with self.subTest(d=d, use_adv_net=use_adv_net):
                    net = make_q_net(QRDQNNet, d, use_adv_net)
                    inputs = make_inputs()
                    with torch.no_grad():
                        candidate_feature, global_feature, _ = net.get_embedding(**embedding_inputs(inputs))
                        q_values, _ = net(**inputs)
                    for head in net.Q:
                        self.assertEqual(in_features(head.q), candidate_feature.shape[-1])
                        if use_adv_net:
                            self.assertEqual(in_features(head.v), global_feature.shape[-1])
                    mask = inputs["dynamic_pair_mask"].reshape(B, -1)
                    for q in q_values:
                        self.assertEqual(tuple(q.shape), (B, J * M, NUM_QUANTILES))
                        self.assertTrue(torch.isneginf(q[mask]).all())
                        self.assertTrue(torch.isfinite(q[~mask]).all())

    def test_mqrdqn_net(self):
        for d in EMBEDDING_DIMS:
            for use_adv_net in (True, False):
                with self.subTest(d=d, use_adv_net=use_adv_net):
                    net = make_q_net(mQRDQNNet, d, use_adv_net)
                    inputs = make_inputs()
                    with torch.no_grad():
                        candidate_feature, global_feature, _ = net.get_embedding(**embedding_inputs(inputs))
                        q = net(**inputs)
                    self.assertEqual(in_features(net.Q.q), candidate_feature.shape[-1])
                    if use_adv_net:
                        self.assertEqual(in_features(net.Q.v), global_feature.shape[-1])
                    self.assertEqual(tuple(q.shape), (B, J * M, NUM_QUANTILES))

    def test_actor_net(self):
        for d in EMBEDDING_DIMS:
            with self.subTest(d=d):
                net = ActorNet(**DAN_KWARGS, layer_fea_output_dim=(32, d), num_mlp_layers_actor=3,
                               hidden_dim_actor=64).eval()
                inputs = make_inputs()
                with torch.no_grad():
                    candidate_feature = net.get_embedding(**embedding_inputs(inputs))
                    pi, _ = net(**inputs)
                self.assertEqual(in_features(net.actor), candidate_feature.shape[-1])
                self.assertEqual(tuple(pi.shape), (B, J * M))

    def test_iql_value(self):
        for d in EMBEDDING_DIMS:
            with self.subTest(d=d):
                net = IQL_value(**DAN_KWARGS, layer_fea_output_dim=(32, d), num_mlp_layers_critic=3,
                                hidden_dim_critic=64).eval()
                inputs = make_inputs()
                with torch.no_grad():
                    global_feature = net.get_embedding(**embedding_inputs(inputs))
                    v = net(**inputs)
                self.assertEqual(in_features(net.critic), global_feature.shape[-1])
                self.assertEqual(tuple(v.shape), (B, 1))


if __name__ == "__main__":
    unittest.main()
