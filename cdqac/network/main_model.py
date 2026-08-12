from torch.nn.functional import dropout

from .attention_layer import *
from .sub_layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from torch.distributions import Categorical

def nonzero_averaging(x):
    """
        remove zero vectors and then compute the mean of x
        (The deleted nodes are represented by zero vectors)
    :param x: feature vectors with shape [sz_b, node_num, d]
    :return:  the desired mean value with shape [sz_b, d]
    """
    b = x.sum(dim=-2)
    y = torch.count_nonzero(x, dim=-1)
    z = (y != 0).sum(dim=-1, keepdim=True)
    p = 1 / z
    p[z == 0] = 0
    return torch.mul(p, b)


def init_weights(m):
    if isinstance(m, nn.Linear):
        # Orthogonal init for hidden layers
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0.0)



class DualAttentionNetwork(nn.Module):
    def __init__(self, fea_j_input_dim: int, fea_m_input_dim: int, layer_fea_output_dim: List, num_heads_OAB: List,
                 num_heads_MAB: List, dropout_prob: float=0, normalize: bool=False):
        """
            The implementation of dual attention network (DAN)
        :param config: a package of parameters
        """
        super(DualAttentionNetwork, self).__init__()

        self.fea_j_input_dim = fea_j_input_dim
        self.fea_m_input_dim = fea_m_input_dim
        self.output_dim_per_layer = layer_fea_output_dim
        self.num_heads_OAB = num_heads_OAB
        self.num_heads_MAB = num_heads_MAB
        self.last_layer_activate = nn.ELU()

        self.num_dan_layers = len(self.num_heads_OAB)
        assert len(num_heads_MAB) == self.num_dan_layers
        assert len(self.output_dim_per_layer) == self.num_dan_layers
        self.alpha = 0.2
        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout_prob = dropout_prob

        num_heads_OAB_per_layer = [1] + list(self.num_heads_OAB)
        num_heads_MAB_per_layer = [1] + list(self.num_heads_MAB)

        # mid_dim = [self.embedding_output_dim] * (self.num_dan_layers - 1)
        mid_dim = list(self.output_dim_per_layer[:-1])

        j_input_dim_per_layer = [self.fea_j_input_dim] + mid_dim

        m_input_dim_per_layer = [self.fea_m_input_dim] + mid_dim

        self.op_attention_blocks = torch.nn.ModuleList()
        self.mch_attention_blocks = torch.nn.ModuleList()
        self.normalize = normalize

        for i in range(self.num_dan_layers):
            self.op_attention_blocks.append(
                MultiHeadOpAttnBlock(
                    input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_OAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )

        for i in range(self.num_dan_layers):
            self.mch_attention_blocks.append(
                MultiHeadMchAttnBlock(
                    node_input_dim=num_heads_MAB_per_layer[i] * m_input_dim_per_layer[i],
                    edge_input_dim=num_heads_OAB_per_layer[i] * j_input_dim_per_layer[i],
                    num_heads=self.num_heads_MAB[i],
                    output_dim=self.output_dim_per_layer[i],
                    concat=True if i < self.num_dan_layers - 1 else False,
                    activation=nn.ELU() if i < self.num_dan_layers - 1 else self.last_layer_activate,
                    dropout_prob=self.dropout_prob
                )
            )


    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx):
        """
        :param candidate: the index of candidates  [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :return:
            fea_j.shape = [sz_b, N, output_dim]
            fea_m.shape = [sz_b, M, output_dim]
            fea_j_global.shape = [sz_b, output_dim]
            fea_m_global.shape = [sz_b, output_dim]
        """
        sz_b, M, _, J = comp_idx.size()

        comp_idx_for_mul = comp_idx.reshape(sz_b, -1, J)

        for layer in range(self.num_dan_layers):
            candidate_idx = candidate.unsqueeze(-1). \
                repeat(1, 1, fea_j.shape[-1]).type(torch.int64)

            # fea_j_jc: candidate features with shape [sz_b, N, J]
            fea_j_jc = torch.gather(fea_j, 1, candidate_idx).type(torch.float32)
            comp_val_layer = torch.matmul(comp_idx_for_mul,
                                     fea_j_jc).reshape(sz_b, M, M, -1)
            fea_j = self.op_attention_blocks[layer](fea_j, op_mask)
            fea_m = self.mch_attention_blocks[layer](fea_m, mch_mask, comp_val_layer)

        # if self.normalize:
        #     # print(norm)
        #     norm_j = torch.linalg.norm(fea_j, ord=2, dim=(-2, -1), keepdim=True)
        #     norm_m = torch.linalg.norm(fea_m, ord=2, dim=(-2, -1), keepdim=True)
        #
        #     fea_j = fea_j / (norm_j + 1e-6)
        #     fea_m = fea_m / (norm_m + 1e-6)


        fea_j_global = nonzero_averaging(fea_j)
        fea_m_global = nonzero_averaging(fea_m)

        return fea_j, fea_m, fea_j_global, fea_m_global


class IQL_value(nn.Module):
    def __init__(
            self,
            fea_j_input_dim: int,
            fea_m_input_dim: int,
            layer_fea_output_dim: List,
            num_heads_OAB: List,
            num_heads_MAB: List,
            num_mlp_layers_critic: int,
            hidden_dim_critic: int,
            dropout_prob: float = 0,
    ):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(IQL_value, self).__init__()
        # device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(
            fea_j_input_dim=fea_j_input_dim,
            fea_m_input_dim=fea_m_input_dim,
            layer_fea_output_dim=layer_fea_output_dim,
            num_heads_OAB=num_heads_OAB,
            num_heads_MAB=num_heads_MAB,
            dropout_prob=dropout_prob,
        )
        self.critic = Critic(num_mlp_layers_critic, 2 * self.embedding_output_dim, hidden_dim_critic, 1)


    def get_embedding(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, fea_pairs):
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)

        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        return global_feature


    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        sz_b, M, _, J = comp_idx.size()

        global_feature = self.get_embedding(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                                                               fea_pairs)


        v = self.critic(global_feature)


        return v


class ActorNet(nn.Module):
    def __init__(
            self,
            fea_j_input_dim: int,
            fea_m_input_dim: int,
            layer_fea_output_dim: List,
            num_heads_OAB: List,
            num_heads_MAB: List,
            num_mlp_layers_actor: int,
            hidden_dim_actor: int,
            dropout_prob: float = 0,
    ):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(ActorNet, self).__init__()
        # device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(
            fea_j_input_dim=fea_j_input_dim,
            fea_m_input_dim=fea_m_input_dim,
            layer_fea_output_dim=layer_fea_output_dim,
            num_heads_OAB=num_heads_OAB,
            num_heads_MAB=num_heads_MAB,
            dropout_prob=0,
        )
        self.actor = Actor(num_mlp_layers_actor, 4 * self.embedding_output_dim + self.pair_input_dim,
                           hidden_dim_actor, 1, dropout=dropout_prob)


    def get_embedding(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, fea_pairs):
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs), dim=-1)
        return candidate_feature

    @torch.no_grad()
    def get_action(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                   deterministic=True):
        """
        :param fea_j:
        :param op_mask:
        :param candidate:
        :param fea_m:
        :param mch_mask:
        :param comp_idx:
        :param dynamic_pair_mask:
        :param fea_pairs:
        :param deterministic:
        :return:
        """
        sz_b, M, _, J = comp_idx.size()
        candidate_feature = self.get_embedding(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                                                  fea_pairs)
        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        # masking incompatible op-mch pairs
        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)
        if deterministic:
            action = pi.argmax(dim=1)
        else:
            action_dist = Categorical(pi)
            action = action_dist.sample()
        return action

    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        sz_b, M, _, J = comp_idx.size()

        candidate_feature = self.get_embedding(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                                                               fea_pairs)

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)

        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        log_pi = candidate_scores - torch.logsumexp(candidate_scores, dim=1, keepdim=True)


        pi = F.softmax(candidate_scores, dim=1)

        return pi, log_pi

    def noisy_action(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                     tau=1.5):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        sz_b, M, _, J = comp_idx.size()

        candidate_feature = self.get_embedding(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                                                               fea_pairs)

        candidate_scores = self.actor(candidate_feature)
        candidate_scores = candidate_scores.squeeze(-1)


        candidate_scores[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        log_pi = candidate_scores - torch.logsumexp(candidate_scores, dim=1, keepdim=True)


        pi = F.gumbel_softmax(candidate_scores, hard=False, tau=tau, dim=1)


        return pi, log_pi


class QRDQNNet(nn.Module):
    def __init__(
            self,
            fea_j_input_dim: int,
            fea_m_input_dim: int,
            layer_fea_output_dim: List,
            num_heads_OAB: List,
            num_heads_MAB: List,
            num_mlp_layers_critic: int,
            hidden_dim_critic: int,
            num_quantiles: int,
            dropout_prob: float=0,
            dropout_prob_q: float=0,
            use_adv_net: bool=False,
            use_global_state: bool=False,
            return_states: bool=False,
            n_critics: int = 5,
            layer_norm: bool = False
                 ):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(QRDQNNet, self).__init__()
        self.use_adv_net = use_adv_net
        # device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(
            fea_j_input_dim=fea_j_input_dim,
            fea_m_input_dim=fea_m_input_dim,
            layer_fea_output_dim=layer_fea_output_dim,
            num_heads_OAB=num_heads_OAB,
            num_heads_MAB=num_heads_MAB,
            dropout_prob=0,
            normalize=True
        )
        self.use_global_state = use_global_state
        self.use_adv_net = use_adv_net
        if self.use_adv_net:
            self.input_dim_candidate = (4 * self.embedding_output_dim + self.pair_input_dim)
        else:
            self.input_dim_candidate = (4 * self.embedding_output_dim + self.pair_input_dim)
        self.input_dim_global = 1 * self.embedding_output_dim
        if self.use_adv_net:
            self.Q = [QRDQN_advantage_single(
                            num_mlp_layers_critic,
                            self.input_dim_candidate,
                            self.input_dim_global + self.pair_input_dim,
                            hidden_dim_critic,
                            num_quantiles,
                            dropout=dropout_prob_q,
                            use_layer_norm=layer_norm,
                        ) for _ in range(n_critics)]


            # self.Q = QRDQN_advantage(num_mlp_layers_critic, self.input_dim_candidate,
            #                          self.input_dim_global, hidden_dim_critic, num_quantiles,
            #                          dropout=dropout_prob_q)
        else:
            self.Q = [QRDQN_single(num_mlp_layers_critic, self.input_dim_candidate,
                                hidden_dim_critic, num_quantiles, dropout=dropout_prob_q, use_layer_norm=layer_norm) for _ in range(n_critics)]
            # self.Q = QRDQN(num_mlp_layers_critic, self.input_dim_candidate,
            #                 hidden_dim_critic, num_quantiles, dropout=dropout_prob_q)
        self.Q = torch.nn.ModuleList(self.Q)
        self.num_quantiles = num_quantiles
        self.return_states = return_states


    def get_embedding(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, fea_pairs):
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        # Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        # Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # print(fea_pairs.shape)
        # exit()
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        if self.use_adv_net:

            # candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, fea_pairs), dim=-1)
            Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
            Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
            candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                           Fea_Gm_input, fea_pairs), dim=-1)
        else:
            Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
            Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
            candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                             Fea_Gm_input, fea_pairs), dim=-1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)

        return candidate_feature, global_feature, (fea_j, fea_m)



    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        sz_b, M, _, J = comp_idx.size()

        candidate_feature, global_feature, norm_feat = self.get_embedding(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                                                                          fea_pairs)

        # fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # fea_pairs[dynamic_pair_mask.reshape(sz_b, -1)] = 0
        #
        # fea_pairs_global_sum = fea_pairs.sum(dim=1)
        #
        # non_masked = (~dynamic_pair_mask).flatten(1,2).sum(dim=1)
        #
        # fea_pairs_global = fea_pairs_global_sum / non_masked.unsqueeze(-1)
        # # print(global_feature.shape, fea_pairs_global.shape)
        # global_feature = torch.cat((global_feature, fea_pairs_global), dim=-1)

        # q1, q2 = self.Q(candidate_feature, global_feature, dynamic_pair_mask, sz_b)
        q_val = [Qnet(candidate_feature, global_feature, dynamic_pair_mask, sz_b) for Qnet in self.Q]
        for i in range(len(q_val)):
            q_val[i][dynamic_pair_mask.reshape(sz_b, -1, 1).expand(-1, -1, self.num_quantiles)] = float('-inf')

        return q_val, norm_feat


class mQRDQNNet(nn.Module):
    def __init__(
            self,
            fea_j_input_dim: int,
            fea_m_input_dim: int,
            layer_fea_output_dim: List,
            num_heads_OAB: List,
            num_heads_MAB: List,
            num_mlp_layers_critic: int,
            hidden_dim_critic: int,
            num_quantiles: int,
            dropout_prob: float=0,
            dropout_prob_q: float=0,
            use_adv_net: bool=False,
            use_global_state: bool=False,
            return_states: bool=False,
            n_critics: int = 5,
            layer_norm: bool = False
                 ):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(mQRDQNNet, self).__init__()
        self.use_adv_net = use_adv_net
        # device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(
            fea_j_input_dim=fea_j_input_dim,
            fea_m_input_dim=fea_m_input_dim,
            layer_fea_output_dim=layer_fea_output_dim,
            num_heads_OAB=num_heads_OAB,
            num_heads_MAB=num_heads_MAB,
            dropout_prob=0,

        )
        self.use_global_state = use_global_state
        self.use_adv_net = use_adv_net
        if self.use_adv_net:
            self.input_dim_candidate = (2 * self.embedding_output_dim + self.pair_input_dim)
        else:
            self.input_dim_candidate = (4 * self.embedding_output_dim + self.pair_input_dim)
        self.input_dim_global = 2 * self.embedding_output_dim
        if self.use_adv_net:
            self.Q = QRDQN_advantage_single(
                            num_mlp_layers_critic,
                            self.input_dim_candidate,
                            self.input_dim_global,
                            hidden_dim_critic,
                            num_quantiles,
                            dropout=dropout_prob_q,
                            use_layer_norm=layer_norm,
                        )


            # self.Q = QRDQN_advantage(num_mlp_layers_critic, self.input_dim_candidate,
            #                          self.input_dim_global, hidden_dim_critic, num_quantiles,
            #                          dropout=dropout_prob_q)
        else:
            self.Q = QRDQN_single(num_mlp_layers_critic, self.input_dim_candidate,
                                hidden_dim_critic, num_quantiles, dropout=dropout_prob_q, use_layer_norm=layer_norm)
            # self.Q = QRDQN(num_mlp_layers_critic, self.input_dim_candidate,
            #                 hidden_dim_critic, num_quantiles, dropout=dropout_prob_q)

        self.num_quantiles = num_quantiles
        self.return_states = return_states


    def get_embedding(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, fea_pairs):
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        # Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        # Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        if self.use_adv_net:

            candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, fea_pairs), dim=-1)
        else:
            Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
            Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
            candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                             Fea_Gm_input, fea_pairs), dim=-1)

        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)

        return candidate_feature, global_feature, (fea_j, fea_m)



    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        sz_b, M, _, J = comp_idx.size()

        candidate_feature, global_feature, norm_feat = self.get_embedding(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                                                                          fea_pairs)

        # q1, q2 = self.Q(candidate_feature, global_feature, dynamic_pair_mask, sz_b)
        q_val = self.Q(candidate_feature, global_feature, dynamic_pair_mask, sz_b)

        q_val[dynamic_pair_mask.reshape(sz_b, -1, 1).expand(-1, -1, self.num_quantiles)] = float('-inf')

        return q_val

    @torch.no_grad()
    def get_action(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs,
                   deterministic=True):
        """
        :param fea_j:
        :param op_mask:
        :param candidate:
        :param fea_m:
        :param mch_mask:
        :param comp_idx:
        :param dynamic_pair_mask:
        :param fea_pairs:
        :param deterministic:
        :return:
        """
        q_values = self(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs).mean(-1)
        if deterministic:
            action = q_values.argmax(dim=1)
        else:
            q_soft = F.softmax(q_values, dim=1)
            action_dist = Categorical(q_soft)
            action = action_dist.sample()
        return action


# class QRDQNNet(nn.Module):
#     def __init__(
#             self,
#             fea_j_input_dim: int,
#             fea_m_input_dim: int,
#             layer_fea_output_dim: List,
#             num_heads_OAB: List,
#             num_heads_MAB: List,
#             num_mlp_layers_critic: int,
#             hidden_dim_critic: int,
#             num_quantiles: int,
#             dropout_prob: float=0,
#             dropout_prob_q: float=0,
#             use_adv_net: bool=False,
#             use_global_state: bool=False,
#             return_states: bool=False
#                  ):
#         """
#             The implementation of the proposed learning framework for fjsp
#         :param config: a package of parameters
#         """
#         super(QRDQNNet, self).__init__()
#         self.use_adv_net = use_adv_net
#         # device = torch.device(config.device)
#
#         # pair features input dim with fixed value
#         self.pair_input_dim = 8
#
#         self.embedding_output_dim = layer_fea_output_dim[-1]
#
#         self.feature_exact = DualAttentionNetwork(
#             fea_j_input_dim=fea_j_input_dim,
#             fea_m_input_dim=fea_m_input_dim,
#             layer_fea_output_dim=layer_fea_output_dim,
#             num_heads_OAB=num_heads_OAB,
#             num_heads_MAB=num_heads_MAB,
#             dropout_prob=0,
#
#         )
#         self.use_global_state = use_global_state
#         self.use_adv_net = use_adv_net
#         if self.use_adv_net:
#             self.input_dim_candidate = (2 * self.embedding_output_dim + self.pair_input_dim)
#         else:
#             self.input_dim_candidate = (4 * self.embedding_output_dim + self.pair_input_dim)
#         self.input_dim_global = 2 * self.embedding_output_dim
#         if self.use_adv_net:
#
#             self.Q = QRDQN_advantage(num_mlp_layers_critic, self.input_dim_candidate,
#                                      self.input_dim_global, hidden_dim_critic, num_quantiles,
#                                      dropout=dropout_prob_q)
#         else:
#             self.Q = QRDQN(num_mlp_layers_critic, self.input_dim_candidate,
#                             hidden_dim_critic, num_quantiles, dropout=dropout_prob_q)
#         self.num_quantiles = num_quantiles
#         self.return_states = return_states
#
#
#     def get_embedding(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, fea_pairs):
#         fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
#                                                                       comp_idx)
#         sz_b, M, _, J = comp_idx.size()
#         d = fea_j.size(-1)
#
#         # collect the input of decision-making network
#         candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
#         candidate_idx = candidate_idx.type(torch.int64)
#
#         Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)
#
#         Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
#         Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)
#
#         # Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
#         # Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
#
#         fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
#         # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
#         if self.use_adv_net:
#
#             candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, fea_pairs), dim=-1)
#         else:
#             Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
#             Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
#             candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
#                                              Fea_Gm_input, fea_pairs), dim=-1)
#
#         global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
#
#         return candidate_feature, global_feature, (fea_j, fea_m)
#
#
#
#     def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
#         """
#         :param candidate: the index of candidate operations with shape [sz_b, J]
#         :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
#         :param op_mask: used for masking nonexistent predecessors/successor
#                         (with shape [sz_b, N, 3])
#         :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
#         :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
#         :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
#                     the value of comp_idx[i, k, q, j] (any i) means whether
#                     machine $M_k$ and $M_q$ are competing for candidate[i,j]
#         :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
#                             incompatible op-mch pairs
#         :param fea_pairs: pair features with shape [sz_b, J, M, 8]
#         :return:
#             pi: scheduling policy with shape [sz_b, J*M]
#             v: the value of state with shape [sz_b, 1]
#         """
#         sz_b, M, _, J = comp_idx.size()
#
#         candidate_feature, global_feature, norm_feat = self.get_embedding(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
#                                                                           fea_pairs)
#
#         q1, q2 = self.Q(candidate_feature, global_feature, dynamic_pair_mask, sz_b)
#
#
#         q1[dynamic_pair_mask.reshape(sz_b, -1, 1).expand(-1, -1, self.num_quantiles)] = float('-inf')
#         q2[dynamic_pair_mask.reshape(sz_b, -1, 1).expand(-1, -1, self.num_quantiles)] = float('-inf')
#         if self.return_states:
#             return q1, q2, candidate_feature, global_feature
#         return q1, q2, norm_feat


class DQNNet(nn.Module):
    def __init__(
            self,
            fea_j_input_dim: int,
            fea_m_input_dim: int,
            layer_fea_output_dim: List,
            num_heads_OAB: List,
            num_heads_MAB: List,
            num_mlp_layers_critic: int,
            hidden_dim_critic: int,
            num_quantiles: int,
            dropout_prob: float=0,
            dropout_prob_q: float=0,
            use_adv_net: bool=False,
            use_global_state: bool=False
                 ):
        """
            The implementation of the proposed learning framework for fjsp
        :param config: a package of parameters
        """
        super(DQNNet, self).__init__()
        self.use_adv_net = use_adv_net
        # device = torch.device(config.device)

        # pair features input dim with fixed value
        self.pair_input_dim = 8

        self.embedding_output_dim = layer_fea_output_dim[-1]

        self.feature_exact = DualAttentionNetwork(
            fea_j_input_dim=fea_j_input_dim,
            fea_m_input_dim=fea_m_input_dim,
            layer_fea_output_dim=layer_fea_output_dim,
            num_heads_OAB=num_heads_OAB,
            num_heads_MAB=num_heads_MAB,
            dropout_prob=0,

        )
        self.use_global_state = use_global_state
        if self.use_global_state:
            input_dim_candidate = (4 * self.embedding_output_dim + self.pair_input_dim) + 2 * self.embedding_output_dim
        else:
            input_dim_candidate = (4 * self.embedding_output_dim + self.pair_input_dim)
        if self.use_adv_net:

            self.Q = DQN_advantage(num_mlp_layers_critic, input_dim_candidate,
                                     2 * self.embedding_output_dim, hidden_dim_critic, num_quantiles,
                                     dropout=dropout_prob_q)
        else:
            self.Q = QRDQN(num_mlp_layers_critic, input_dim_candidate,
                            hidden_dim_critic, 1, dropout=dropout_prob_q)
        self.num_quantiles = num_quantiles


    def get_embedding(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, fea_pairs):
        fea_j, fea_m, fea_j_global, fea_m_global = self.feature_exact(fea_j, op_mask, candidate, fea_m, mch_mask,
                                                                      comp_idx)
        sz_b, M, _, J = comp_idx.size()
        d = fea_j.size(-1)

        # collect the input of decision-making network
        candidate_idx = candidate.unsqueeze(-1).repeat(1, 1, d)
        candidate_idx = candidate_idx.type(torch.int64)

        Fea_j_JC = torch.gather(fea_j, 1, candidate_idx)

        Fea_j_JC_serialized = Fea_j_JC.unsqueeze(2).repeat(1, 1, M, 1).reshape(sz_b, M * J, d)
        Fea_m_serialized = fea_m.unsqueeze(1).repeat(1, J, 1, 1).reshape(sz_b, M * J, d)

        Fea_Gj_input = fea_j_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)
        Fea_Gm_input = fea_m_global.unsqueeze(1).expand_as(Fea_j_JC_serialized)

        fea_pairs = fea_pairs.reshape(sz_b, -1, self.pair_input_dim)
        # candidate_feature.shape = [sz_b, J*M, 4*output_dim + 8]
        candidate_feature = torch.cat((Fea_j_JC_serialized, Fea_m_serialized, Fea_Gj_input,
                                       Fea_Gm_input, fea_pairs), dim=-1)
        global_feature = torch.cat((fea_j_global, fea_m_global), dim=-1)
        if self.use_global_state:
            repeat_global_feature = global_feature.unsqueeze(1).repeat(1, candidate_feature.shape[1], 1)
        # print(repeat_global_feature.shape)
            candidate_feature = torch.cat((candidate_feature, repeat_global_feature), dim=-1)

        return candidate_feature, global_feature



    def forward(self, fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx, dynamic_pair_mask, fea_pairs):
        """
        :param candidate: the index of candidate operations with shape [sz_b, J]
        :param fea_j: input operation feature vectors with shape [sz_b, N, 8]
        :param op_mask: used for masking nonexistent predecessors/successor
                        (with shape [sz_b, N, 3])
        :param fea_m: input operation feature vectors with shape [sz_b, M, 6]
        :param mch_mask: used for masking attention coefficients (with shape [sz_b, M, M])
        :param comp_idx: a tensor with shape [sz_b, M, M, J] used for computing T_E
                    the value of comp_idx[i, k, q, j] (any i) means whether
                    machine $M_k$ and $M_q$ are competing for candidate[i,j]
        :param dynamic_pair_mask: a tensor with shape [sz_b, J, M], used for masking
                            incompatible op-mch pairs
        :param fea_pairs: pair features with shape [sz_b, J, M, 8]
        :return:
            pi: scheduling policy with shape [sz_b, J*M]
            v: the value of state with shape [sz_b, 1]
        """
        sz_b, M, _, J = comp_idx.size()

        candidate_feature, global_feature = self.get_embedding(fea_j, op_mask, candidate, fea_m, mch_mask, comp_idx,
                                                               fea_pairs)

        q1, q2 = self.Q(candidate_feature, global_feature, dynamic_pair_mask, sz_b)


        q1[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        q2[dynamic_pair_mask.reshape(sz_b, -1)] = float('-inf')
        q1 = q1.squeeze(-1)
        q2 = q2.squeeze(-1)

        return q1, q2

