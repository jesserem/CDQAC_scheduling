"""
Embedding-space k-NN density proxy for offline-FJSP training datasets.

This is the embedding-based counterpart of ``check_coverage.py``. Instead of
counting unique action-prefix strings, we:

  1. Load the pretrained DAN encoder from ``checkpoints_dumb_exp/best_det_offline.pt``
     (an ``ActorNet`` = DualAttentionNetwork feature extractor + actor head).
  2. For every training instance, define an "optimal" trajectory as the *best of
     ``n_samples`` stochastic rollouts* of that policy (lowest makespan).
  3. Embed every transition of a trajectory as
     ``cat(unit(state_emb), unit(next_state_emb)) / sqrt(2)`` (dim 4*d = 32), where
     ``state_emb = cat(fea_j_global, fea_m_global)`` (dim 2*d = 16) comes from the DAN
     encoder ``model.feature_exact``.  The state and next-state halves are L2-normalized
     **separately**, so only their directions matter (embedding magnitude grows with
     scheduling progress and would otherwise dominate the metric), and the ``1/sqrt(2)``
     leaves the transition vector itself on the unit sphere -- k-NN L2 distances are
     therefore bounded in ``[0, 2]`` and monotone in cosine similarity.
  4. For each dataset *type* (PDR, GA, PDR-GA, Random) embed all of its stored
     trajectories -- after removing exact-duplicate trajectories the same way
     training does (``remove_duplicate_actions``) -- then measure how densely
     those transitions surround the optimal
     trajectory with an **optimal-anchored k-NN distance** (for each optimal
     transition, mean L2 distance to its k nearest dataset-type transitions;
     averaged over the optimal transitions), for ``k in {1, 5, 10}``.

Lower distance  ==>  that dataset type has transitions lying close to (densely
covering) the model's optimal trajectory in the encoder's representation space.

We additionally report a **nearest-neighbour source competition**: pooling the
disjoint base sources {PDR, GA, Random}, we tally which source each optimal
transition's k nearest neighbours come from (PDR-GA = PDR+GA).  This answers
"which source most often provides the closest match" (e.g. is Random more often
closest than PDR-GA?), complementing the average distance.  (Caveat: a larger
pool wins more just by having more points, so read it alongside the unique-
trajectory counts.)

Everything is computed *per instance* and discarded before moving on, so memory
and disk stay tiny (we never persist raw states; only 16-d embedding vectors are
formed transiently, and only a small results CSV is written).

LOCAL vs GLOBAL reference pool
------------------------------
By default an instance's optimal transitions are only compared against *that same
instance's* stored trajectories ("local"): the k-NN measures within-instance
support density.  With ``--global_ref`` the reference pool is instead the union of
the stored trajectories of *all* reference instances, so a query transition may
match a state seen in a different instance -- answering "does the dataset, as a
whole, contain a state like this one?".

Global mode never materialises that pool (it would be 2-6 GB of embeddings).  It
runs two streaming passes:

  Pass 1  sample+embed every instance's optimal trajectory  -> the query set
          (small: ~20 MB even for 1000 instances of 15x10).
  Pass 2  for each reference instance in turn, embed its (deduped) trajectories,
          update a running per-query **top-k** table, and discard the embeddings.

Peak memory is therefore one instance's reference embeddings (<10 MB), a
``[n_query, k]`` table (~16 MB) and one ``--query_chunk x transitions-of-one-instance``
distance tile -- all independent of the number of instances.  Measured peak on
20x10 (the biggest dataset) is ~290 MiB of CUDA memory whether the pool is 40 or
500 instances; materialising the same pool would cost ~1.9 GB (and ~6 GB for
15x10 x 1000).  The tile dominates, so ``--query_chunk`` is the dial to turn if
memory is still tight.

The streamed result is the *exact* global k-NN, not an approximation: merging
per-source top-k lists is lossless because a global top-k over the union is always
contained in the union of the per-source top-k lists.  Verified against a
brute-force materialised pool -- identical to 6e-16 in float64.  In the production
float32 path both agree to <=1.1e-4, the difference being ``torch.cdist``'s
``|x|^2+|y|^2-2xy`` identity losing precision on the *zero* distances that duplicate
transitions produce (its rounding depends on the GEMM shape).  That noise is inherent
to the per-instance path too and is ~3 orders below the gaps between dataset types.

``--global_exclude_self`` drops each query instance's own trajectories from its
reference pool ("does some *other* instance cover this state?").

    python density_embedding_knn.py --global_ref --ref_instances 500
    python density_embedding_knn.py --global_ref --random_sweep --random_counts 1 5 10 25 50 100

Run (pilot default: 10x5, first 50 instances, 1000 samples, CUDA if available):

    python density_embedding_knn.py

All three SD1 sizes at once, writing every result to CSV:

    python density_embedding_knn.py --data all --out results/density_knn.csv

That produces one numeric per-instance sheet per dataset
(``results/density_knn_10_5.csv``, ``_15_10``, ``_20_10`` -- directly readable by
``density_coverage_figure.py --csv``) plus one combined, aggregated sheet
``results/density_knn_summary.csv`` with a row per (dataset, type, k).

The four ε-PDR datasets of the paper's Table 10 (ε-PDR and ε-PDR-GA at ε ∈ {0.1, 0.2})
can be scored alongside the standard four types with ``--eps``, in both the local and
``--global_ref`` pool modes:

    python density_embedding_knn.py --data all --eps --out results/density_knn_eps.csv
    python density_embedding_knn.py --data all --eps --global_ref --ref_instances 500 \
        --out results/density_knn_eps_global.csv

They are read from the ``eps_random_<v>`` keys that ``add_eps_pdr_rules.py`` stores in
the same .npy files (first 500 instances only -- the run is clipped to those), composed
exactly as in ``check_coverage.py`` (the script behind the table's SACo column):
``eps-PDR-<v> = eps_random_<v>[:100]`` and ``eps-PDR-GA-<v> = that + ga_pop[:50]``
(``--n_eps`` / ``--eps_ga_n``).  The ε types gain ``<type>_k<k>`` distance columns like
any other; the nearest-neighbour source competition keeps its original {PDR, GA, Random}
pool, so the ``nn_`` shares stay comparable to runs without ``--eps``.  Globally, each
``eps-PDR-GA-<v>`` is reconstructed the same way as PDR-GA: an ``eps-GA-extra-<v>``
accumulator streams the GA rows of the ``[:eps_ga_n]`` slice not already in that
instance's ε source, and its top-k table is merged losslessly with the ε source's.

Random-only sweep -- how far does the Random source's support density around the
optimal trajectory degrade as its dataset is thinned from 100 to 1 trajectory per
instance?  (Embedding-space counterpart of the dataset-thinning experiment.)

    python density_embedding_knn.py --data all --random_sweep \
        --random_counts 1 5 10 25 50 100 --out results/random_sweep.csv

Same CSV layout, but rows are keyed by ``n_random`` instead of dataset type and only
Random is scored.  The optimal anchor is sampled once per instance and shared across
all sizes, so the curve isolates the effect of dataset size alone.

Full parameterisation via CLI (see ``build_argparser``) or import the functions
(``run_dataset``, ``do_all_density_datasets``, ``run_random_sweep_dataset``) into a
notebook.
"""

import argparse
import math
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from cdqac.network.main_model import ActorNet
from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums


# ---------------------------------------------------------------------------
# Dataset-type definitions (mirrors check_coverage.py)
# ---------------------------------------------------------------------------

# The 16 PDR rules used as the "PDR" dataset type in check_coverage.py.
USED_PDR = [
    "MWR_SPT_masked", "MWR_LPT_masked", "LWR_SPT_masked", "LWR_LPT_masked",
    "MWR_EST_masked", "MWR_LST_masked", "LWR_EST_masked", "LWR_LST_masked",
    "MOPNR_SPT_masked", "MOPNR_LPT_masked", "LOPNR_SPT_masked", "LOPNR_LPT_masked",
    "MOPNR_EST_masked", "MOPNR_LST_masked", "LOPNR_EST_masked", "LOPNR_LST_masked",
]

# The dataset types we score.  Order is meaningful for printing.
DATASET_TYPES = ["PDR", "GA", "PDR-GA", "Random"]

# The disjoint base sources pooled for the nearest-neighbour competition.
# PDR-GA is derived (PDR + GA) so it is NOT a separate pool.
BASE_TYPES = ["PDR", "GA", "Random"]

# The four ε-PDR datasets of the paper's Table 10 (a PDR dispatcher that, per step,
# takes a uniformly random feasible action with probability ε).  They are stored in
# the same .npy files under ``eps_random_<v>`` (added by ``add_eps_pdr_rules.py``;
# only the first 500 instances carry them) and are enabled with ``--eps``, in both
# the local and ``--global_ref`` reference-pool modes.  The composition mirrors
# ``check_coverage.py``, which produced the table's SACo column: eps-PDR-<v> =
# eps_random_<v>[:n_eps] and eps-PDR-GA-<v> = that + ga_pop[:eps_ga_n] (defaults 100
# and 50).  The nearest-neighbour source competition keeps its original
# {PDR, GA, Random} pool, so those shares stay comparable to runs without --eps.
EPS_VERSIONS = ("0.1", "0.2")


def eps_dataset_key(version):
    """'0.1' -> 'eps_random_0.1', the instance-dict key add_eps_pdr_rules.py writes."""
    return f"eps_random_{version}"


def eps_types(versions):
    """The scored ε type names, in Table 10 order: plain then +GA, per version."""
    out = []
    for v in versions:
        out += [f"eps-PDR-{v}", f"eps-PDR-GA-{v}"]
    return out


def scored_types(eps_versions=()):
    """The dataset types a run scores: the standard four + Table 10's ε types."""
    return DATASET_TYPES + eps_types(eps_versions)

# The exact ActorNet architecture stored in best_det_offline.pt (reverse
# engineered from the checkpoint tensor shapes; strict load verified).
ACTORNET_KWARGS = dict(
    fea_j_input_dim=10,
    fea_m_input_dim=8,
    layer_fea_output_dim=(32, 8),
    num_heads_OAB=(4, 4),
    num_heads_MAB=(4, 4),
    num_mlp_layers_actor=3,
    hidden_dim_actor=64,
    dropout_prob=0,
)


# ---------------------------------------------------------------------------
# Model loading & input normalization
# ---------------------------------------------------------------------------

class Normalizer:
    """Applies the z-score normalization bundled inside the checkpoint.

    The encoder was trained on normalized features, so we reproduce that exact
    preprocessing before every forward pass.  All stats broadcast against the
    env state tensors: fea_j [E,N,10], fea_m [E,M,8], fea_pairs [E,J,M,8].
    """

    def __init__(self, ckpt, device):
        self.mean_fea_j = ckpt["mean_fea_j"].to(device)
        self.std_fea_j = ckpt["std_fea_j"].to(device)
        self.mean_fea_m = ckpt["mean_fea_m"].to(device)
        self.std_fea_m = ckpt["std_fea_m"].to(device)
        self.mean_fea_pairs = ckpt["mean_fea_pairs"].to(device)
        self.std_fea_pairs = ckpt["std_fea_pairs"].to(device)

    def fea_j(self, x):
        return (x - self.mean_fea_j) / (self.std_fea_j + 1e-8)

    def fea_m(self, x):
        return (x - self.mean_fea_m) / (self.std_fea_m + 1e-8)

    def fea_pairs(self, x):
        return (x - self.mean_fea_pairs) / (self.std_fea_pairs + 1e-8)


def load_model(checkpoint_path, device):
    """Build ActorNet, load the pretrained weights, return (model, normalizer)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = ActorNet(**ACTORNET_KWARGS)
    model.load_state_dict(ckpt["actor_net"], strict=True)
    model.to(device)
    model.eval()
    normalizer = Normalizer(ckpt, device)
    return model, normalizer


# ---------------------------------------------------------------------------
# Encoding & policy stepping
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_state(model, normalizer, state, normalize=True):
    """Return the pooled global state embedding ``cat(fea_j_global, fea_m_global)``.

    Shape [E, 2*d] (== [E, 16] for this checkpoint).  Uses the DAN encoder
    ``model.feature_exact`` directly (ActorNet.get_embedding only returns the
    per-pair candidate feature, not the global vector we want).
    """
    fea_j = normalizer.fea_j(state.fea_j_tensor) if normalize else state.fea_j_tensor
    fea_m = normalizer.fea_m(state.fea_m_tensor) if normalize else state.fea_m_tensor
    _, _, fea_j_global, fea_m_global = model.feature_exact(
        fea_j,
        state.op_mask_tensor,
        state.candidate_tensor,
        fea_m,
        state.mch_mask_tensor,
        state.comp_idx_tensor,
    )
    return torch.cat((fea_j_global, fea_m_global), dim=-1)


@torch.no_grad()
def policy_action(model, normalizer, state, deterministic, normalize=True):
    """Sample/greedy action from the actor (returns a numpy int array [E])."""
    fea_j = normalizer.fea_j(state.fea_j_tensor) if normalize else state.fea_j_tensor
    fea_m = normalizer.fea_m(state.fea_m_tensor) if normalize else state.fea_m_tensor
    fea_pairs = normalizer.fea_pairs(state.fea_pairs_tensor) if normalize else state.fea_pairs_tensor
    action = model.get_action(
        fea_j=fea_j,
        op_mask=state.op_mask_tensor,
        candidate=state.candidate_tensor,
        fea_m=fea_m,
        mch_mask=state.mch_mask_tensor,
        comp_idx=state.comp_idx_tensor,
        dynamic_pair_mask=state.dynamic_pair_mask_tensor,
        fea_pairs=fea_pairs,
        deterministic=deterministic,
    )
    return action.cpu().numpy()


# ---------------------------------------------------------------------------
# Step 2: best-of-N sampled trajectory ("optimal") per instance
# ---------------------------------------------------------------------------

def _tile_instance(job_length, op_pt, num_runs):
    jl = np.tile(np.expand_dims(job_length, 0), (num_runs, 1))
    pt = np.tile(np.expand_dims(op_pt, 0), (num_runs, 1, 1))
    return jl, pt


@torch.no_grad()
def sample_best_actions(model, normalizer, job_length, op_pt, n_samples, device,
                        sample_batch=None, normalize=True):
    """Run ``n_samples`` stochastic rollouts; return the best action list.

    Returns (best_action_list [n_op] int64, best_makespan float).  Only the
    per-step actions and final makespans are kept (both tiny), so memory is
    independent of embedding size.  ``sample_batch`` optionally splits the
    samples into chunks of independent envs to cap peak memory on large
    instances (the global argmin is tracked across chunks).
    """
    n_op = int(np.sum(job_length))
    if sample_batch is None:
        sample_batch = n_samples

    best_ms = np.inf
    best_actions = None
    remaining = n_samples
    while remaining > 0:
        r = min(sample_batch, remaining)
        remaining -= r
        jl, pt = _tile_instance(job_length, op_pt, r)
        env = FJSPEnvForSameOpNums(n_j=job_length.shape[0], n_m=op_pt.shape[1],
                                   device=device, mask_actions=True)
        state = env.set_initial_data(jl, pt)
        actions_per_step = np.empty((n_op, r), dtype=np.int64)
        for t in range(n_op):
            a = policy_action(model, normalizer, state, deterministic=False,
                              normalize=normalize)
            actions_per_step[t] = a
            state, _, _ = env.step(a)
        makespans = np.asarray(env.current_makespan)
        idx = int(np.argmin(makespans))
        if makespans[idx] < best_ms:
            best_ms = float(makespans[idx])
            best_actions = actions_per_step[:, idx].copy()
    return best_actions, best_ms


# ---------------------------------------------------------------------------
# Step 3: transition embeddings for a set of action lists
# ---------------------------------------------------------------------------

@torch.no_grad()
def replay_transition_embeddings(model, normalizer, job_length, op_pt, action_lists,
                                 device, batch_cap=256, normalize=True,
                                 out_dtype=torch.float32):
    """Replay ``action_lists`` and return their transition embeddings.

    Each transition is ``cat(u_t, u_{t+1}) / sqrt(2)`` (dim 4*d = 32), where
    ``u_t = state_emb_t / ||state_emb_t||`` is the **separately unit-normalized**
    state embedding.  Normalizing the state and next-state halves independently
    strips the embedding magnitude (which grows with scheduling progress and would
    otherwise dominate the L2 metric) and leaves only direction; the ``1/sqrt(2)``
    makes the concatenated transition itself unit-norm, so k-NN L2 distances live
    in ``[0, 2]`` and are a monotone function of cosine similarity.

    We embed only the ``n_op`` *pre-decision* states ``s_0..s_{n_op-1}`` and form
    ``n_op - 1`` transitions from consecutive pairs.  The fully-scheduled terminal
    state ``s_{n_op}`` is deliberately **not** encoded: with every node deleted it
    is degenerate for the DAN encoder (its pooled embedding blows up), and the
    policy never sees it either.  Output shape:
    ``[len(action_lists) * (n_op - 1), 4*d]`` on CPU.  Trajectories are replayed
    in batches of at most ``batch_cap``; the env is vectorized so a whole batch is
    stepped/encoded at once.
    """
    if len(action_lists) == 0:
        return torch.empty((0, 4 * model.embedding_output_dim), dtype=out_dtype)

    n_j = job_length.shape[0]
    n_m = op_pt.shape[1]
    chunks = []
    for start in range(0, len(action_lists), batch_cap):
        batch = action_lists[start:start + batch_cap]
        action_arr = np.asarray(batch, dtype=np.int64)          # [R, n_op]
        r, n_op = action_arr.shape
        jl, pt = _tile_instance(job_length, op_pt, r)
        env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m, device=device, mask_actions=True)
        state = env.set_initial_data(jl, pt)

        state_embs = []  # E_0..E_{n_op-1}, each [R, 2d] (pre-decision states)
        for t in range(n_op):
            state_embs.append(encode_state(model, normalizer, state, normalize=normalize))
            state, _, _ = env.step(action_arr[:, t])
        # E: [n_op, R, 2d] -> transitions [n_op-1, R, 4d] -> [R*(n_op-1), 4d].
        # The two halves are unit-normalized separately, then the pair is scaled by
        # 1/sqrt(2) so each transition vector has unit norm.
        E = torch.stack(state_embs, dim=0)
        current_emb = F.normalize(E[:-1], p=2, dim=-1)
        next_emb = F.normalize(E[1:], p=2, dim=-1)
        trans = torch.cat((current_emb, next_emb), dim=-1) / math.sqrt(2.0)
        emb = trans.permute(1, 0, 2).reshape(-1, 4 * model.embedding_output_dim)
        chunks.append(emb.to(out_dtype).cpu())
    return torch.cat(chunks, dim=0)


# ---------------------------------------------------------------------------
# Step 4: optimal-anchored k-NN distance
# ---------------------------------------------------------------------------

@torch.no_grad()
def knn_distance(query, reference, k, device, query_chunk=4096):
    """Mean over query rows of the mean L2 distance to their k nearest refs.

    query [Q, D], reference [R, D].  If ``R < k`` we clip k to R.  Non-finite
    rows in either set are dropped defensively (a single inf/nan would otherwise
    poison the average).  Returns a python float, or ``nan`` if either set is
    empty after filtering.
    """
    if query.numel() == 0 or reference.numel() == 0:
        return float("nan")
    query = query[torch.isfinite(query).all(dim=-1)]
    reference = reference[torch.isfinite(reference).all(dim=-1)]
    if query.numel() == 0 or reference.numel() == 0:
        return float("nan")
    k = min(k, reference.shape[0])
    reference = reference.to(device)
    totals = []
    for start in range(0, query.shape[0], query_chunk):
        q = query[start:start + query_chunk].to(device)
        d = torch.cdist(q, reference)                      # [q, R]
        knn, _ = torch.topk(d, k, dim=1, largest=False)    # [q, k]
        totals.append(knn.mean(dim=1).cpu())
    return float(torch.cat(totals).mean())


@torch.no_grad()
def neighbor_source_counts(query, reference, labels, k_list, n_types, device,
                           query_chunk=4096):
    """Tally which source each of the k nearest pooled neighbours came from.

    For every query row we find its ``max(k_list)`` nearest rows in the pooled
    ``reference`` (labelled 0..n_types-1 by ``labels``) and, for each ``k`` in
    ``k_list``, count how many of the top-k belong to each source type.  This is
    a per-transition "which source is closest" competition (k=1 = winner-takes-all;
    larger k = share of the neighbourhood).  Non-finite rows are dropped.

    Returns ``{k: np.int64[n_types]}`` (summed over query rows) and the number of
    finite query rows used.
    """
    result = {k: np.zeros(n_types, dtype=np.int64) for k in k_list}
    if query.numel() == 0 or reference.numel() == 0:
        return result, 0
    query = query[torch.isfinite(query).all(dim=-1)]
    rfin = torch.isfinite(reference).all(dim=-1)
    reference, labels = reference[rfin], labels[rfin]
    if query.numel() == 0 or reference.numel() == 0:
        return result, 0
    kmax = min(max(k_list), reference.shape[0])
    reference = reference.to(device)
    labels = labels.to(device)
    n_query = query.shape[0]
    for start in range(0, n_query, query_chunk):
        q = query[start:start + query_chunk].to(device)
        d = torch.cdist(q, reference)                          # [q, R]
        _, idx = torch.topk(d, kmax, dim=1, largest=False)     # [q, kmax]
        nn_labels = labels[idx]                                # [q, kmax]
        for k in k_list:
            kk = min(k, kmax)
            lk = nn_labels[:, :kk]
            result[k] += torch.bincount(lk.reshape(-1), minlength=n_types).cpu().numpy()
    return result, n_query


# ---------------------------------------------------------------------------
# Step 4b: streaming k-NN against a GLOBAL (cross-instance) reference pool
#
# The pool is never materialised.  We keep, per query transition, only the k
# smallest distances seen so far (a [n_query, k] table) and refresh it as each
# reference instance's embeddings stream past, so peak memory is independent of
# the number of instances.  The result is exact, not an approximation.
# ---------------------------------------------------------------------------

def _mean_topk(dist, k):
    """Per-row mean of the k smallest distances in an ascending-sorted table.

    ``dist`` is [Q, >=k] sorted ascending with ``inf`` padding where a row has
    fewer than k neighbours.  Padding is excluded from the mean (so a row with
    R < k neighbours averages over the R it has, matching ``knn_distance``'s
    ``k = min(k, R)``); rows with none yield nan.
    """
    d = dist[:, :k]
    finite = torch.isfinite(d)
    cnt = finite.sum(dim=1)
    total = torch.where(finite, d, torch.zeros_like(d)).sum(dim=1)
    out = total / cnt.clamp(min=1)
    return torch.where(cnt > 0, out, torch.full_like(out, float("nan")))


class TopKAccumulator:
    """Running k smallest distances per query row over a stream of reference chunks."""

    def __init__(self, n_query, k, device, dtype=torch.float32):
        self.k = k
        self.dist = torch.full((n_query, k), float("inf"), device=device, dtype=dtype)

    def update(self, q_start, d):
        """Fold in ``d`` [Qc, Rc]: distances from queries ``q_start...`` to a chunk."""
        if d.shape[1] == 0:
            return
        top, _ = torch.topk(d, min(self.k, d.shape[1]), dim=1, largest=False)
        q_end = q_start + d.shape[0]
        merged = torch.cat((self.dist[q_start:q_end], top), dim=1)
        self.dist[q_start:q_end] = torch.topk(merged, self.k, dim=1, largest=False)[0]

    def mean_topk(self, k):
        return _mean_topk(self.dist, k)


def merged_topk(accums, k):
    """Per-row mean k-NN distance over the UNION of several accumulators' pools.

    Exact: the k nearest neighbours in the union are always among the k nearest of
    each part, so merging the per-part top-k tables loses nothing.
    """
    dist = torch.cat([a.dist for a in accums], dim=1)
    kmax = min(max(a.k for a in accums), dist.shape[1])
    return _mean_topk(torch.topk(dist, kmax, dim=1, largest=False)[0], k)


def pooled_neighbor_counts(accums, k_list):
    """Per-query tally of which source the k nearest *pooled* neighbours came from.

    ``accums`` is one TopKAccumulator per source (order = BASE_TYPES).  Merging their
    top-k tables reproduces the global pooled top-k exactly.  Returns
    ``{k: [Q, n_sources] float}`` counts; ``inf`` padding contributes nothing.
    """
    dist = torch.cat([a.dist for a in accums], dim=1)                     # [Q, S*k]
    labels = torch.cat([torch.full_like(a.dist, j, dtype=torch.long)
                        for j, a in enumerate(accums)], dim=1)            # [Q, S*k]
    kmax = min(max(k_list), dist.shape[1])
    d, idx = torch.topk(dist, kmax, dim=1, largest=False)
    lab = torch.gather(labels, 1, idx)

    out = {}
    for k in k_list:
        kk = min(k, kmax)
        dk, lk = d[:, :kk], lab[:, :kk]
        # f64: counts are small integers, and the shares derived from them should not
        # inherit f32 rounding (~1e-7) on top of the tie-breaking noise cdist already has.
        counts = torch.zeros(dist.shape[0], len(accums), dtype=torch.float64,
                             device=dist.device)
        counts.scatter_add_(1, lk, torch.isfinite(dk).to(counts.dtype))
        out[k] = counts
    return out


def _group_mean(values, group_idx, n_groups):
    """Mean of ``values`` [Q] within each group, ignoring non-finite entries."""
    finite = torch.isfinite(values)
    v, g = values[finite], group_idx[finite]
    total = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    cnt = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    total.index_add_(0, g, v)
    cnt.index_add_(0, g, torch.ones_like(v))
    out = total / cnt.clamp(min=1)
    return torch.where(cnt > 0, out, torch.full_like(out, float("nan"))).cpu().numpy()


@torch.no_grad()
def stream_reference_instance(accums, queries, query_inst, blocks,
                              exclude_inst=None, query_chunk=1024):
    """Fold one reference instance's embeddings into the running top-k tables.

    ``blocks`` maps accumulator name -> reference rows [R_t, D] (already on ``device``).
    Distances are computed once per query chunk against the instance's concatenated
    rows, so the peak temporary is ``query_chunk x R_instance`` floats.  Rows of
    queries belonging to ``exclude_inst`` are ignored (self-exclusion).
    """
    names = [n for n in blocks if blocks[n].shape[0] > 0]
    if not names:
        return
    ref = torch.cat([blocks[n] for n in names], dim=0)
    sizes = [blocks[n].shape[0] for n in names]
    offs = np.cumsum([0] + sizes)

    for start in range(0, queries.shape[0], query_chunk):
        end = min(start + query_chunk, queries.shape[0])
        d = torch.cdist(queries[start:end], ref)                      # [Qc, R]
        if exclude_inst is not None:
            own = query_inst[start:end] == exclude_inst
            if bool(own.any()):
                d = d.masked_fill(own.unsqueeze(1), float("inf"))
        for j, name in enumerate(names):
            accums[name].update(start, d[:, offs[j]:offs[j + 1]])


# ---------------------------------------------------------------------------
# Per-instance and per-dataset drivers
# ---------------------------------------------------------------------------

def gather_action_lists(instance, n_ga, n_random, eps_versions=(), n_eps=100):
    """Return {type: list-of-action-lists} for one instance dict.

    PDR reuses the 16 USED_PDR rules; PDR-GA is built later as the (deduped)
    union of PDR + GA.  ``eps_versions`` adds one raw source ``eps-PDR-<v>`` per
    version, holding ``eps_random_<v>[:n_eps]``; the derived eps-PDR-GA sources
    are built later from these + the GA slice.
    """
    rules = instance["rules"]
    pdr = [rules[r] for r in USED_PDR if r in rules]
    ga = list(instance["ga_pop"][:n_ga])
    rnd = list(instance["random"][:n_random])
    out = {"PDR": pdr, "GA": ga, "Random": rnd}
    for v in eps_versions:
        out[f"eps-PDR-{v}"] = list(instance[eps_dataset_key(v)][:n_eps])
    return out


def dedupe_trajectories(action_lists):
    """Drop exact-duplicate trajectories (whole action lists), keeping the first
    occurrence.  Mirrors ``remove_duplicate_actions`` used during training, so
    the reference set here matches the offline buffer's (GA populations in
    particular contain many identical members).  Returns (list-of-tuples,
    first_idx): ``first_idx[j]`` is the raw index of the j-th unique trajectory's
    first occurrence, so raw-prefix membership stays recoverable.  Tuples are
    hashable and feed straight into ``np.asarray``.
    """
    seen = set()
    unique, first_idx = [], []
    for i, a in enumerate(action_lists):
        key = tuple(int(x) for x in a)
        if key not in seen:
            seen.add(key)
            unique.append(key)
            first_idx.append(i)
    return unique, np.asarray(first_idx, dtype=np.int64)


def instance_reference_embeddings(model, normalizer, instance, device, n_ga=200,
                                  n_random=100, batch_cap=256, normalize=True,
                                  dedupe=True, eps_versions=(), n_eps=100, eps_ga_n=50):
    """Embed one instance's stored trajectories, per base source.

    Returns (emb, keep, counts, eps_ga_keep): ``emb[t]`` is
    ``[n_unique_t * (n_op-1), 4d]`` for ``t`` in BASE_TYPES (+ one ``eps-PDR-<v>``
    entry per requested ε version); ``keep`` indexes the GA trajectories *not*
    already present in PDR (so PDR-GA = PDR rows + those GA rows);
    ``counts[t] = (n_unique, n_raw)``.  ``eps_ga_keep[v]`` indexes the GA
    trajectories contributing to eps-PDR-GA-<v> -- those whose first occurrence
    lies in the raw ``ga_pop[:eps_ga_n]`` prefix and (under dedup) that the ε
    source does not already contain, mirroring check_coverage.py's Table 10
    composition ``eps_random_<v>[:n_eps] + ga_pop[:eps_ga_n]``.  Trajectories are
    deduped like training (whole-trajectory equality) unless ``dedupe`` is False.
    """
    job_length = np.array(instance["JobLength"])
    op_pt = np.array(instance["OpPT"])

    raw = gather_action_lists(instance, n_ga=n_ga, n_random=n_random,
                              eps_versions=eps_versions, n_eps=n_eps)
    base = BASE_TYPES + [f"eps-PDR-{v}" for v in eps_versions]
    traj, first, counts = {}, {}, {}
    for t in base:
        if dedupe:
            u, fi = dedupe_trajectories(raw[t])
        else:
            u = [tuple(int(x) for x in a) for a in raw[t]]
            fi = np.arange(len(u), dtype=np.int64)
        traj[t], first[t] = u, fi
        counts[t] = (len(u), len(raw[t]))

    # embed each deduped type once (as lists for the env replay)
    emb = {}
    for t in base:
        emb[t] = replay_transition_embeddings(
            model, normalizer, job_length, op_pt, [list(x) for x in traj[t]], device,
            batch_cap=batch_cap, normalize=normalize)

    # PDR-GA = deduped union of PDR + GA; reuse embeddings by keeping only GA
    # trajectories not already present in PDR (each already internally unique).
    pdr_set = set(traj["PDR"]) if dedupe else set()
    keep = [i for i, g in enumerate(traj["GA"]) if g not in pdr_set]
    counts["PDR-GA"] = (len(traj["PDR"]) + len(keep), len(raw["PDR"]) + len(raw["GA"]))

    # eps-PDR-GA-<v> = ε source + the GA slice, reusing the GA embeddings the same way.
    eps_ga_keep = {}
    for v in eps_versions:
        et = f"eps-PDR-{v}"
        pos = np.nonzero(first["GA"] < eps_ga_n)[0]
        if dedupe:
            eps_set = set(traj[et])
            pos = np.asarray([p for p in pos if traj["GA"][p] not in eps_set],
                             dtype=np.int64)
        eps_ga_keep[v] = pos
        counts[f"eps-PDR-GA-{v}"] = (counts[et][0] + len(pos),
                                     len(raw[et]) + min(eps_ga_n, len(raw["GA"])))
    return emb, keep, counts, eps_ga_keep


def _ga_extra_rows(emb_ga, keep, n_ga_traj, d):
    """The GA embedding rows contributing to PDR-GA (GA trajectories not in PDR)."""
    if n_ga_traj > 0 and len(keep) < n_ga_traj:
        T = emb_ga.shape[0] // n_ga_traj                 # transitions per trajectory
        return emb_ga.reshape(n_ga_traj, T, d)[keep].reshape(-1, d)
    return emb_ga


def run_instance(model, normalizer, instance, k_list, n_samples, device,
                 n_ga=200, n_random=100, sample_batch=None, batch_cap=256,
                 normalize=True, dedupe=True, eps_versions=(), n_eps=100, eps_ga_n=50):
    """Score one instance against its OWN trajectories (local reference pool).

    Returns (result, best_makespan, counts, nn_counts) where ``result`` is
    ``{type: {k: knn_distance}}``, ``counts`` is ``{type: (n_unique, n_raw)}`` and
    ``nn_counts`` is ``{k: np.int64[len(BASE_TYPES)]}`` from the pooled
    nearest-neighbour-source competition over {PDR, GA, Random}.  ``eps_versions``
    additionally scores Table 10's eps-PDR-<v> / eps-PDR-GA-<v> sources; the
    nearest-neighbour competition deliberately keeps its original pool, so the
    reported shares stay comparable to runs without ε.
    """
    d = 4 * model.embedding_output_dim
    job_length = np.array(instance["JobLength"])
    op_pt = np.array(instance["OpPT"])

    # (2) optimal trajectory = best of n_samples stochastic rollouts
    best_actions, best_ms = sample_best_actions(
        model, normalizer, job_length, op_pt, n_samples, device,
        sample_batch=sample_batch, normalize=normalize)

    # (3a) optimal trajectory transition embeddings  [n_op-1, 4d]
    optimal_emb = replay_transition_embeddings(
        model, normalizer, job_length, op_pt, [best_actions], device,
        batch_cap=batch_cap, normalize=normalize)

    # (3b) this instance's stored trajectories, deduped like training
    emb, keep, counts, eps_ga_keep = instance_reference_embeddings(
        model, normalizer, instance, device, n_ga=n_ga, n_random=n_random,
        batch_cap=batch_cap, normalize=normalize, dedupe=dedupe,
        eps_versions=eps_versions, n_eps=n_eps, eps_ga_n=eps_ga_n)
    ga_extra = _ga_extra_rows(emb["GA"], keep, counts["GA"][0], d)
    emb["PDR-GA"] = torch.cat([emb["PDR"], ga_extra], dim=0)
    for v in eps_versions:
        ga_rows = _ga_extra_rows(emb["GA"], list(eps_ga_keep[v]), counts["GA"][0], d)
        emb[f"eps-PDR-GA-{v}"] = torch.cat([emb[f"eps-PDR-{v}"], ga_rows], dim=0)

    # (4) optimal-anchored k-NN distance, per type per k
    result = {t: {k: knn_distance(optimal_emb, emb[t], k, device) for k in k_list}
              for t in scored_types(eps_versions)}

    # (5) nearest-neighbour source competition: pool the disjoint base sources
    #     {PDR, GA, Random}, label each transition by source, and tally which
    #     source each optimal transition's k nearest neighbours came from.
    pooled = torch.cat([emb[t] for t in BASE_TYPES], dim=0)
    labels = torch.cat([torch.full((emb[t].shape[0],), j, dtype=torch.long)
                        for j, t in enumerate(BASE_TYPES)], dim=0)
    nn_counts, _ = neighbor_source_counts(
        optimal_emb, pooled, labels, k_list, len(BASE_TYPES), device)
    return result, best_ms, counts, nn_counts


def run_instance_random_sweep(model, normalizer, instance, k_list, n_samples, device,
                              random_counts, sample_batch=None, batch_cap=256,
                              normalize=True, dedupe=True):
    """Score one instance using ONLY the Random source, at several dataset sizes.

    For every ``m`` in ``random_counts`` we keep the *first* ``m`` stored random
    trajectories of the instance (the same prefix training would use when the
    Random dataset is thinned), dedupe them, and measure the optimal-anchored
    k-NN distance against just those.  The optimal anchor is sampled once and
    shared by all ``m``, so the curve isolates the effect of dataset size.

    Each unique trajectory is embedded exactly once: after deduping the longest
    prefix, the deduped set for the first ``m`` raw trajectories is precisely the
    uniques whose *first occurrence* lies below ``m``, so every smaller ``m`` is a
    row-subset of the embeddings we already have.

    Returns (dists, best_makespan, uniq_counts, trans_counts) keyed by ``m``
    (``dists[m][k]`` = k-NN distance).
    """
    d = 4 * model.embedding_output_dim
    job_length = np.array(instance["JobLength"])
    op_pt = np.array(instance["OpPT"])

    max_m = max(random_counts)
    raw = list(instance["random"][:max_m])

    best_actions, best_ms = sample_best_actions(
        model, normalizer, job_length, op_pt, n_samples, device,
        sample_batch=sample_batch, normalize=normalize)
    optimal_emb = replay_transition_embeddings(
        model, normalizer, job_length, op_pt, [best_actions], device,
        batch_cap=batch_cap, normalize=normalize)

    seen = set()
    uniq, first_idx = [], []
    for i, a in enumerate(raw):
        key = tuple(int(x) for x in a)
        if dedupe:
            if key in seen:
                continue
            seen.add(key)
        uniq.append(key)
        first_idx.append(i)
    first_idx = np.asarray(first_idx, dtype=np.int64)

    emb = replay_transition_embeddings(
        model, normalizer, job_length, op_pt, [list(x) for x in uniq], device,
        batch_cap=batch_cap, normalize=normalize)
    U = len(uniq)
    emb3 = emb.reshape(U, -1, d) if U else emb        # [U, n_op-1, 4d]

    dists, uniq_counts, trans_counts = {}, {}, {}
    for m in random_counts:
        sel = np.nonzero(first_idx < m)[0]
        sub = emb3[sel].reshape(-1, d) if len(sel) else emb[:0]
        dists[m] = {k: knn_distance(optimal_emb, sub, k, device) for k in k_list}
        uniq_counts[m] = len(sel)
        trans_counts[m] = int(sub.shape[0])
    return dists, best_ms, uniq_counts, trans_counts


def _summarise(values, round_val=4):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(mean=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"))
    return dict(mean=round(float(a.mean()), round_val), std=round(float(a.std()), round_val),
                min=round(float(a.min()), round_val), max=round(float(a.max()), round_val))


def _clamp_to_eps_instances(data, data_path, eps_versions):
    """Keep the leading prefix of instances that store every requested ε key.

    ``add_eps_pdr_rules.py`` only annotates the first 500 instances of each dataset,
    so a 1000-instance file must be clipped before ε scoring.
    """
    if not eps_versions:
        return data
    keys = [eps_dataset_key(v) for v in eps_versions]
    n_ok = 0
    for inst in data:
        if not all(k in inst for k in keys):
            break
        n_ok += 1
    if n_ok == 0:
        raise SystemExit(f"{data_path} stores none of {keys}; generate them with "
                         f"add_eps_pdr_rules.py before using --eps")
    if n_ok < len(data):
        print(f"NOTE: only the first {n_ok}/{len(data)} instances of {data_path} store "
              f"{keys} (add_eps_pdr_rules.py caps at 500); scoring those.")
    return data[:n_ok]


def run_dataset(model, normalizer, data_path, k_list, n_samples, device,
                n_instances=None, n_ga=200, n_random=100, sample_batch=None,
                batch_cap=256, normalize=True, dedupe=True, verbose=True,
                out=None, eps_versions=(), n_eps=100, eps_ga_n=50):
    """Run the density experiment over one .npy dataset file.

    Returns (per_instance, summary): ``per_instance`` is a list of dicts (one
    per instance) each ``{type: {k: dist}, 'best_makespan': ms}``; ``summary``
    is ``{type: {'unique_traj': mean, 'k': {k: {mean,std,min,max,nn_share}}}}``
    aggregated across instances.  If ``out`` is given, the per-instance results
    are also written there as CSV.

    ``eps_versions`` (e.g. ``EPS_VERSIONS``) additionally scores the paper's Table 10
    ε-PDR datasets: ``eps-PDR-<v>`` = ``eps_random_<v>[:n_eps]`` and ``eps-PDR-GA-<v>``
    = that + ``ga_pop[:eps_ga_n]`` (check_coverage.py's composition).  Their k-NN
    distances gain columns/rows like any other type; the nearest-neighbour source
    competition keeps its original {PDR, GA, Random} pool (ε types get no ``nn_``
    columns), so those shares stay comparable to runs without ε.
    """
    data = np.load(data_path, allow_pickle=True)
    if n_instances is not None:
        data = data[:n_instances]
    data = _clamp_to_eps_instances(data, data_path, eps_versions)
    types = scored_types(eps_versions)

    per_instance = []
    # accum[type][k] -> list of per-instance distances
    accum = {t: {k: [] for k in k_list} for t in types}
    uniq_accum = {t: [] for t in types}           # per-instance unique traj counts
    # nn_accum[type][k] -> list of per-instance nearest-neighbour source shares
    nn_accum = {t: {k: [] for k in k_list} for t in DATASET_TYPES}

    t0 = time.time()
    for i in range(len(data)):
        res, best_ms, counts, nn_counts = run_instance(
            model, normalizer, data[i], k_list, n_samples, device,
            n_ga=n_ga, n_random=n_random, sample_batch=sample_batch,
            batch_cap=batch_cap, normalize=normalize, dedupe=dedupe,
            eps_versions=eps_versions, n_eps=n_eps, eps_ga_n=eps_ga_n)
        rec = {"best_makespan": best_ms}
        for t in types:
            uniq_accum[t].append(counts[t][0])
            for k in k_list:
                accum[t][k].append(res[t][k])
                rec[f"{t}_k{k}"] = res[t][k]
        # per-instance nearest-neighbour source shares (base sources -> fraction;
        # PDR-GA = PDR + GA share).  Shares sum to 1 across BASE_TYPES.
        for k in k_list:
            c = nn_counts[k].astype(float)
            tot = c.sum() if c.sum() > 0 else 1.0
            share = {BASE_TYPES[j]: c[j] / tot for j in range(len(BASE_TYPES))}
            share["PDR-GA"] = share["PDR"] + share["GA"]
            for t in DATASET_TYPES:
                nn_accum[t][k].append(share[t])
                rec[f"nn_{t}_k{k}"] = share[t]
        per_instance.append(rec)
        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(data) - i - 1)
            print(f"[{i + 1}/{len(data)}] best_ms={best_ms:.1f} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    summary = {}
    for t in types:
        per_k = {}
        for k in k_list:
            s = dict(_summarise(accum[t][k]))
            if t in DATASET_TYPES:
                s["nn_share"] = float(np.mean(nn_accum[t][k]))
            per_k[k] = s
        summary[t] = {"unique_traj": float(np.mean(uniq_accum[t])), "k": per_k}

    if verbose:
        print(f"\n==== Optimal-anchored k-NN L2 distance for: {data_path} ====")
        print(f"(lower = dataset type's transitions lie closer to the model's optimal "
              f"trajectory; {len(data)} instances, {n_samples} samples/instance, "
              f"dedupe={dedupe})")
        if eps_versions:
            print(f"(eps sources: eps-PDR-<v> = eps_random_<v>[:{n_eps}], eps-PDR-GA-<v> "
                  f"= that + ga_pop[:{eps_ga_n}], as in check_coverage.py)")
        for t in types:
            parts = []
            for k in k_list:
                s = summary[t]["k"][k]
                parts.append(f"k={k}: {s['mean']:.4f}±{s['std']:.4f} "
                             f"[{s['min']:.4f},{s['max']:.4f}]")
            print(f"  {t:15s} (uniq traj~{summary[t]['unique_traj']:.0f}) " + " | ".join(parts))

        print(f"\n==== Nearest-neighbour source share among pooled {{PDR, GA, Random}} ====")
        print("(mean % of each optimal transition's k nearest pooled neighbours coming "
              "from each source; PDR-GA = PDR+GA. NOTE: larger pools win more just by "
              "having more points -- see uniq traj counts above)")
        for t in DATASET_TYPES:
            parts = [f"k={k}: {100 * summary[t]['k'][k]['nn_share']:5.1f}%" for k in k_list]
            print(f"  {t:15s} " + " | ".join(parts))

    if out:
        write_csv(per_instance, k_list, out, types=types)

    return per_instance, summary


def run_random_sweep_dataset(model, normalizer, data_path, k_list, n_samples, device,
                             random_counts=(1, 5, 10, 25, 50, 100), n_instances=None,
                             sample_batch=None, batch_cap=256, normalize=True,
                             dedupe=True, verbose=True, out=None):
    """Random-only density sweep over one .npy dataset: k-NN distance vs. #random runs.

    Answers "how much does the Random dataset's support density around the model's
    optimal trajectory degrade as we thin it from 100 to 1 trajectory per instance?"
    -- the embedding-space counterpart of the dataset-thinning experiment.

    Returns (per_instance, summary): ``per_instance`` is a list of rows, one per
    (instance, m); ``summary`` is ``{m: {'unique_traj','transitions','k': {k: {...}}}}``.
    """
    random_counts = sorted(set(int(m) for m in random_counts))
    data = np.load(data_path, allow_pickle=True)
    if n_instances is not None:
        data = data[:n_instances]

    available = len(data[0]["random"]) if len(data) else 0
    if random_counts[-1] > available:
        print(f"WARNING: --random_counts asks for {random_counts[-1]} trajectories but the "
              f"dataset stores only {available} per instance; larger counts are clipped.")

    per_instance = []
    accum = {m: {k: [] for k in k_list} for m in random_counts}
    uniq_accum = {m: [] for m in random_counts}
    trans_accum = {m: [] for m in random_counts}

    t0 = time.time()
    for i in range(len(data)):
        dists, best_ms, uniq, trans = run_instance_random_sweep(
            model, normalizer, data[i], k_list, n_samples, device, random_counts,
            sample_batch=sample_batch, batch_cap=batch_cap, normalize=normalize,
            dedupe=dedupe)
        for m in random_counts:
            rec = {"instance": i, "n_random": m, "best_makespan": best_ms,
                   "n_unique_traj": uniq[m], "n_transitions": trans[m]}
            for k in k_list:
                rec[f"Random_k{k}"] = dists[m][k]
                accum[m][k].append(dists[m][k])
            uniq_accum[m].append(uniq[m])
            trans_accum[m].append(trans[m])
            per_instance.append(rec)
        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(data) - i - 1)
            print(f"[{i + 1}/{len(data)}] best_ms={best_ms:.1f} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    summary = {
        m: {"unique_traj": float(np.mean(uniq_accum[m])),
            "transitions": float(np.mean(trans_accum[m])),
            "k": {k: _summarise(accum[m][k]) for k in k_list}}
        for m in random_counts
    }

    if verbose:
        print(f"\n==== Random-only k-NN density vs. #random runs/instance: {data_path} ====")
        print(f"(lower = the thinned Random dataset still lies close to the model's optimal "
              f"trajectory; {len(data)} instances, {n_samples} samples/instance, "
              f"dedupe={dedupe})")
        for m in random_counts:
            parts = [f"k={k}: {summary[m]['k'][k]['mean']:.4f}±{summary[m]['k'][k]['std']:.4f}"
                     for k in k_list]
            print(f"  n_random={m:4d} (uniq traj~{summary[m]['unique_traj']:.1f}) "
                  + " | ".join(parts))

    if out:
        write_random_sweep_csv(per_instance, k_list, out)

    return per_instance, summary


# ---------------------------------------------------------------------------
# Global-reference drivers (two streaming passes; see the module docstring)
# ---------------------------------------------------------------------------

def _group_sum(values, group_idx, n_groups):
    out = torch.zeros(n_groups, dtype=values.dtype, device=values.device)
    out.index_add_(0, group_idx, values)
    return out


@torch.no_grad()
def build_query_set(model, normalizer, data, n_query_inst, n_samples, device,
                    sample_batch=None, batch_cap=256, normalize=True, verbose=True):
    """Pass 1: sample + embed every query instance's optimal trajectory.

    Returns (queries [Q, 4d] on ``device``, query_inst [Q] long, best_makespans).
    Only the anchors are kept (a few MB); nothing else from pass 1 survives.  Replay
    and encoding draw no randomness, so these anchors are identical to the ones the
    per-instance path samples with the same seed.
    """
    embs, inst, makespans = [], [], []
    t0 = time.time()
    for i in range(n_query_inst):
        job_length = np.array(data[i]["JobLength"])
        op_pt = np.array(data[i]["OpPT"])
        best_actions, best_ms = sample_best_actions(
            model, normalizer, job_length, op_pt, n_samples, device,
            sample_batch=sample_batch, normalize=normalize)
        e = replay_transition_embeddings(
            model, normalizer, job_length, op_pt, [best_actions], device,
            batch_cap=batch_cap, normalize=normalize)
        e = e[torch.isfinite(e).all(dim=-1)]
        embs.append(e)
        inst.append(torch.full((e.shape[0],), i, dtype=torch.long))
        makespans.append(best_ms)
        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (n_query_inst - i - 1)
            print(f"  [pass 1: anchors] [{i + 1}/{n_query_inst}] best_ms={best_ms:.1f} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    queries = torch.cat(embs, dim=0).to(device)
    query_inst = torch.cat(inst, dim=0).to(device)
    return queries, query_inst, makespans


def run_global_dataset(model, normalizer, data_path, k_list, n_samples, device,
                       n_instances=None, ref_instances=None, n_ga=200, n_random=100,
                       sample_batch=None, batch_cap=256, query_chunk=1024,
                       exclude_self=False, normalize=True, dedupe=True, verbose=True,
                       out=None, eps_versions=(), n_eps=100, eps_ga_n=50):
    """Like ``run_dataset``, but each instance's optimal transitions are scored against
    the pooled trajectories of ALL reference instances (see the module docstring).

    ``n_instances`` sets how many instances are *queried* (get a CSV row);
    ``ref_instances`` how many contribute trajectories to the pool (default: the same
    instances; it may be larger to score few instances against a big pool).  Same
    return value and CSV schema as ``run_dataset``, so ``density_coverage_figure.py``
    reads either interchangeably.

    ``eps_versions`` scores Table 10's ε sources against the same kind of global pool:
    each ``eps-PDR-<v>`` gets its own top-k accumulator, and ``eps-PDR-GA-<v>`` is
    reconstructed exactly like PDR-GA -- an ``eps-GA-extra-<v>`` accumulator holds the
    GA rows of the ``ga_pop[:eps_ga_n]`` slice not already contained in that instance's
    ε source, and the two top-k tables are merged losslessly.  The nearest-neighbour
    source competition keeps its original {PDR, GA, Random} pool.
    """
    data = np.load(data_path, allow_pickle=True)
    data = _clamp_to_eps_instances(data, data_path, eps_versions)
    types = scored_types(eps_versions)
    n_query_inst = len(data) if n_instances is None else min(n_instances, len(data))
    n_ref_inst = n_query_inst if ref_instances is None else min(ref_instances, len(data))
    d = 4 * model.embedding_output_dim
    kmax = max(k_list)

    queries, query_inst, makespans = build_query_set(
        model, normalizer, data, n_query_inst, n_samples, device,
        sample_batch=sample_batch, batch_cap=batch_cap, normalize=normalize,
        verbose=verbose)
    n_query = queries.shape[0]

    # GA_extra = the GA rows that PDR does not already contain; PDR-GA is the union
    # of the PDR and GA_extra pools, merged losslessly from their top-k tables.  The
    # eps-GA-extra-<v> accumulators play the same role for the eps-PDR-GA-<v> unions.
    sources = ["PDR", "GA", "Random", "GA_extra"]
    for v in eps_versions:
        sources += [f"eps-PDR-{v}", f"eps-GA-extra-{v}"]
    accums = {s: TopKAccumulator(n_query, kmax, device, dtype=queries.dtype)
              for s in sources}
    uniq_accum = {t: [] for t in types}
    pool_rows = 0

    t0 = time.time()
    for j in range(n_ref_inst):
        emb, keep, counts, eps_ga_keep = instance_reference_embeddings(
            model, normalizer, data[j], device, n_ga=n_ga, n_random=n_random,
            batch_cap=batch_cap, normalize=normalize, dedupe=dedupe,
            eps_versions=eps_versions, n_eps=n_eps, eps_ga_n=eps_ga_n)
        pairs = [("PDR", emb["PDR"]), ("GA", emb["GA"]), ("Random", emb["Random"]),
                 ("GA_extra", _ga_extra_rows(emb["GA"], keep, counts["GA"][0], d))]
        for v in eps_versions:
            pairs += [(f"eps-PDR-{v}", emb[f"eps-PDR-{v}"]),
                      (f"eps-GA-extra-{v}",
                       _ga_extra_rows(emb["GA"], list(eps_ga_keep[v]), counts["GA"][0], d))]
        blocks = {}
        for name, e in pairs:
            blocks[name] = e[torch.isfinite(e).all(dim=-1)].to(device)
        stream_reference_instance(
            accums, queries, query_inst, blocks,
            exclude_inst=(j if exclude_self else None), query_chunk=query_chunk)

        pool_rows += sum(int(blocks[t].shape[0]) for t in BASE_TYPES)
        for t in types:
            uniq_accum[t].append(counts[t][0])
        del emb, blocks
        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / (j + 1) * (n_ref_inst - j - 1)
            print(f"  [pass 2: pool] [{j + 1}/{n_ref_inst}] pool_transitions={pool_rows} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    # Collapse the per-query top-k tables into per-instance numbers (same schema as local)
    per_instance = [{"best_makespan": ms} for ms in makespans]
    accum = {t: {k: [] for k in k_list} for t in types}
    nn_accum = {t: {k: [] for k in k_list} for t in DATASET_TYPES}
    nn_counts = pooled_neighbor_counts([accums[t] for t in BASE_TYPES], k_list)

    for k in k_list:
        per_query = {
            "PDR": accums["PDR"].mean_topk(k),
            "GA": accums["GA"].mean_topk(k),
            "Random": accums["Random"].mean_topk(k),
            "PDR-GA": merged_topk([accums["PDR"], accums["GA_extra"]], k),
        }
        for v in eps_versions:
            et = f"eps-PDR-{v}"
            per_query[et] = accums[et].mean_topk(k)
            per_query[f"eps-PDR-GA-{v}"] = merged_topk(
                [accums[et], accums[f"eps-GA-extra-{v}"]], k)
        for t in types:
            vals = _group_mean(per_query[t], query_inst, n_query_inst)
            for i in range(n_query_inst):
                per_instance[i][f"{t}_k{k}"] = float(vals[i])
                accum[t][k].append(float(vals[i]))

        c = nn_counts[k]                                            # [Q, len(BASE_TYPES)]
        den = _group_sum(c.sum(dim=1), query_inst, n_query_inst).clamp(min=1)
        share = {t: (_group_sum(c[:, j], query_inst, n_query_inst) / den).cpu().numpy()
                 for j, t in enumerate(BASE_TYPES)}
        share["PDR-GA"] = share["PDR"] + share["GA"]
        for t in DATASET_TYPES:
            for i in range(n_query_inst):
                per_instance[i][f"nn_{t}_k{k}"] = float(share[t][i])
                nn_accum[t][k].append(float(share[t][i]))

    summary = {}
    for t in types:
        per_k = {}
        for k in k_list:
            s = dict(_summarise(accum[t][k]))
            if t in DATASET_TYPES:
                s["nn_share"] = float(np.mean(nn_accum[t][k]))
            per_k[k] = s
        summary[t] = {"unique_traj": float(np.mean(uniq_accum[t])), "k": per_k}

    if verbose:
        print(f"\n==== GLOBAL optimal-anchored k-NN L2 distance for: {data_path} ====")
        print(f"(reference pool = {n_ref_inst} instances / {pool_rows} transitions, "
              f"shared by all queries; {n_query_inst} instances queried "
              f"({n_query} optimal transitions), {n_samples} samples/instance, "
              f"dedupe={dedupe}, exclude_self={exclude_self})")
        if eps_versions:
            print(f"(eps sources: eps-PDR-<v> = eps_random_<v>[:{n_eps}], eps-PDR-GA-<v> "
                  f"= that + ga_pop[:{eps_ga_n}], as in check_coverage.py; pooled over "
                  f"the same {n_ref_inst} reference instances)")
        for t in types:
            parts = []
            for k in k_list:
                s = summary[t]["k"][k]
                parts.append(f"k={k}: {s['mean']:.4f}±{s['std']:.4f} "
                             f"[{s['min']:.4f},{s['max']:.4f}]")
            print(f"  {t:15s} (uniq traj~{summary[t]['unique_traj']:.0f}/ref instance) "
                  + " | ".join(parts))

        print(f"\n==== Nearest-neighbour source share among the pooled {{PDR, GA, Random}} ====")
        print("(mean % of each optimal transition's k nearest neighbours -- drawn from the "
              "WHOLE pool, any instance -- coming from each source; PDR-GA = PDR+GA. NOTE: "
              "larger pools win more just by having more points)")
        for t in DATASET_TYPES:
            parts = [f"k={k}: {100 * summary[t]['k'][k]['nn_share']:5.1f}%" for k in k_list]
            print(f"  {t:15s} " + " | ".join(parts))

    if out:
        write_csv(per_instance, k_list, out, types=types)

    return per_instance, summary


def run_global_random_sweep_dataset(model, normalizer, data_path, k_list, n_samples, device,
                                    random_counts=(1, 5, 10, 25, 50, 100), n_instances=None,
                                    ref_instances=None, sample_batch=None, batch_cap=256,
                                    query_chunk=1024, exclude_self=False, normalize=True,
                                    dedupe=True, verbose=True, out=None):
    """Random-only sweep against a GLOBAL pool: k-NN distance vs. #random runs *per
    reference instance* (so the pool at ``m`` holds ``m`` random trajectories from each
    of the ``ref_instances`` instances).

    ``n_unique_traj`` / ``n_transitions`` in the CSV are the POOL totals (identical on
    every row), not per-instance counts as in the local sweep.
    """
    random_counts = sorted(set(int(m) for m in random_counts))
    data = np.load(data_path, allow_pickle=True)
    n_query_inst = len(data) if n_instances is None else min(n_instances, len(data))
    n_ref_inst = n_query_inst if ref_instances is None else min(ref_instances, len(data))
    kmax = max(k_list)
    max_m = max(random_counts)

    available = len(data[0]["random"]) if len(data) else 0
    if max_m > available:
        print(f"WARNING: --random_counts asks for {max_m} trajectories but the dataset "
              f"stores only {available} per instance; larger counts are clipped.")

    queries, query_inst, makespans = build_query_set(
        model, normalizer, data, n_query_inst, n_samples, device,
        sample_batch=sample_batch, batch_cap=batch_cap, normalize=normalize,
        verbose=verbose)
    n_query = queries.shape[0]

    accums = {m: TopKAccumulator(n_query, kmax, device, dtype=queries.dtype)
              for m in random_counts}
    pool_uniq = {m: 0 for m in random_counts}
    pool_trans = {m: 0 for m in random_counts}

    t0 = time.time()
    for j in range(n_ref_inst):
        job_length = np.array(data[j]["JobLength"])
        op_pt = np.array(data[j]["OpPT"])
        raw = list(data[j]["random"][:max_m])

        # Uniques in first-occurrence order, so the deduped set for the first m raw
        # trajectories is exactly the PREFIX of `uniq` whose first index is < m.
        seen = set()
        uniq, first_idx = [], []
        for i, a in enumerate(raw):
            key = tuple(int(x) for x in a)
            if dedupe:
                if key in seen:
                    continue
                seen.add(key)
            uniq.append(key)
            first_idx.append(i)
        first_idx = np.asarray(first_idx, dtype=np.int64)

        emb = replay_transition_embeddings(
            model, normalizer, job_length, op_pt, [list(x) for x in uniq], device,
            batch_cap=batch_cap, normalize=normalize)
        U = len(uniq)
        T = emb.shape[0] // U if U else 0                    # transitions per trajectory
        ref = emb.to(device)
        bad = ~torch.isfinite(ref).all(dim=-1)               # mask, not drop: keeps the
        rows = {}                                            # prefix->row mapping intact
        for m in random_counts:
            n_traj = int(np.count_nonzero(first_idx < m))
            rows[m] = n_traj * T
            pool_uniq[m] += n_traj
            pool_trans[m] += int((~bad[:rows[m]]).sum())

        for start in range(0, n_query, query_chunk):
            end = min(start + query_chunk, n_query)
            dist = torch.cdist(queries[start:end], ref)
            dist = dist.masked_fill(bad.unsqueeze(0), float("inf"))
            if exclude_self:
                own = query_inst[start:end] == j
                if bool(own.any()):
                    dist = dist.masked_fill(own.unsqueeze(1), float("inf"))
            for m in random_counts:
                accums[m].update(start, dist[:, :rows[m]])

        del emb, ref
        if verbose:
            elapsed = time.time() - t0
            eta = elapsed / (j + 1) * (n_ref_inst - j - 1)
            print(f"  [pass 2: pool] [{j + 1}/{n_ref_inst}] "
                  f"pool_transitions@{max_m}={pool_trans[max_m]} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    per_instance = []
    accum = {m: {k: [] for k in k_list} for m in random_counts}
    per_inst_dist = {m: {k: _group_mean(accums[m].mean_topk(k), query_inst, n_query_inst)
                         for k in k_list} for m in random_counts}
    for i in range(n_query_inst):
        for m in random_counts:
            rec = {"instance": i, "n_random": m, "best_makespan": makespans[i],
                   "n_unique_traj": pool_uniq[m], "n_transitions": pool_trans[m]}
            for k in k_list:
                v = float(per_inst_dist[m][k][i])
                rec[f"Random_k{k}"] = v
                accum[m][k].append(v)
            per_instance.append(rec)

    summary = {
        m: {"unique_traj": float(pool_uniq[m]),
            "transitions": float(pool_trans[m]),
            "k": {k: _summarise(accum[m][k]) for k in k_list}}
        for m in random_counts
    }

    if verbose:
        print(f"\n==== GLOBAL Random-only k-NN density vs. #random runs/instance: {data_path} ====")
        print(f"(pool = m random trajectories from each of {n_ref_inst} reference instances; "
              f"{n_query_inst} instances queried, {n_samples} samples/instance, "
              f"dedupe={dedupe}, exclude_self={exclude_self})")
        for m in random_counts:
            parts = [f"k={k}: {summary[m]['k'][k]['mean']:.4f}±{summary[m]['k'][k]['std']:.4f}"
                     for k in k_list]
            print(f"  n_random={m:4d} (pool: {pool_uniq[m]} uniq traj / "
                  f"{pool_trans[m]} transitions) " + " | ".join(parts))

    if out:
        write_random_sweep_csv(per_instance, k_list, out)

    return per_instance, summary


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def write_csv(per_instance, k_list, out_path, types=None):
    """Per-instance CSV for one dataset.

    Every column is numeric on purpose: ``density_coverage_figure.py`` casts all
    cells to float, so no dataset-name column may be added here (the dataset is
    identified by the file name instead).  ``types`` extends the distance columns
    (e.g. with the ε types); the ``nn_`` columns always cover the original four.
    """
    import csv
    dist_fields = [f"{t}_k{k}" for t in (types or DATASET_TYPES) for k in k_list]
    nn_fields = [f"nn_{t}_k{k}" for t in DATASET_TYPES for k in k_list]
    fields = ["instance", "best_makespan"] + dist_fields + nn_fields
    _ensure_dir(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, rec in enumerate(per_instance):
            row = {"instance": i, "best_makespan": rec["best_makespan"]}
            for name in dist_fields + nn_fields:
                row[name] = rec[name]
            w.writerow(row)
    print(f"Wrote per-instance results to {out_path}")


def summary_rows(dataset, per_instance, summary, k_list):
    """Long-format rows (one per dataset x type x k) for the cross-dataset sheet.

    Iterates the summary's own keys, so ε types (when scored) get rows too; they
    carry no ``nn_source_share`` (the competition pool stays {PDR, GA, Random}).
    """
    ms = [rec["best_makespan"] for rec in per_instance]
    rows = []
    for t in summary:
        for k in k_list:
            s = summary[t]["k"][k]
            rows.append({
                "dataset": dataset,
                "type": t,
                "k": k,
                "n_instances": len(per_instance),
                "mean_unique_traj": round(summary[t]["unique_traj"], 2),
                "mean_best_makespan": round(float(np.mean(ms)), 2) if ms else float("nan"),
                "knn_dist_mean": s["mean"],
                "knn_dist_std": s["std"],
                "knn_dist_min": s["min"],
                "knn_dist_max": s["max"],
                "nn_source_share": round(s["nn_share"], 4) if "nn_share" in s else "",
            })
    return rows


def write_summary_csv(rows, out_path):
    """Aggregated results across all datasets/types/k (one sheet, human-readable)."""
    import csv
    fields = ["dataset", "type", "k", "n_instances", "mean_unique_traj",
              "mean_best_makespan", "knn_dist_mean", "knn_dist_std", "knn_dist_min",
              "knn_dist_max", "nn_source_share"]
    _ensure_dir(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote combined summary to {out_path}")


def write_random_sweep_csv(per_instance, k_list, out_path):
    """Per-(instance, n_random) sheet of the random-only sweep for one dataset."""
    import csv
    fields = (["instance", "n_random", "best_makespan", "n_unique_traj", "n_transitions"]
              + [f"Random_k{k}" for k in k_list])
    _ensure_dir(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_instance)
    print(f"Wrote per-instance random sweep to {out_path}")


def random_sweep_summary_rows(dataset, per_instance, summary, k_list):
    """Long-format rows (one per dataset x n_random x k) for the cross-dataset sheet."""
    n_inst = len({rec["instance"] for rec in per_instance})
    ms = [rec["best_makespan"] for rec in per_instance if rec["n_random"] == min(summary)]
    rows = []
    for m in sorted(summary):
        for k in k_list:
            s = summary[m]["k"][k]
            rows.append({
                "dataset": dataset,
                "n_random": m,
                "k": k,
                "n_instances": n_inst,
                "mean_unique_traj": round(summary[m]["unique_traj"], 2),
                "mean_transitions": round(summary[m]["transitions"], 1),
                "mean_best_makespan": round(float(np.mean(ms)), 2) if ms else float("nan"),
                "knn_dist_mean": s["mean"],
                "knn_dist_std": s["std"],
                "knn_dist_min": s["min"],
                "knn_dist_max": s["max"],
            })
    return rows


def write_random_sweep_summary_csv(rows, out_path):
    """Aggregated random-only sweep across all datasets (one sheet)."""
    import csv
    fields = ["dataset", "n_random", "k", "n_instances", "mean_unique_traj",
              "mean_transitions", "mean_best_makespan", "knn_dist_mean", "knn_dist_std",
              "knn_dist_min", "knn_dist_max"]
    _ensure_dir(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote combined random-sweep summary to {out_path}")


# ---------------------------------------------------------------------------
# Convenience: all three SD1 sizes (mirrors check_coverage.do_all_coverage_datasets)
# ---------------------------------------------------------------------------

DEFAULT_ALL_DATASETS = [
    "./dataset/SD1_train_10_5_1000.npy",
    "./dataset/SD1_train_15_10_1000.npy",
    "./dataset/SD1_train_20_10_500.npy",
]


def _size_tag(data_path):
    """'dataset/SD1_train_10_5_1000.npy' -> '10_5' (falls back to the file stem)."""
    stem = os.path.splitext(os.path.basename(data_path))[0]
    parts = stem.split("_")
    for i in range(len(parts) - 1):
        if parts[i].isdigit() and parts[i + 1].isdigit():
            return f"{parts[i]}_{parts[i + 1]}"
    return stem


def do_all_density_datasets(model, normalizer, k_list, n_samples, device,
                            n_instances=500, out=None, datasets=None, global_ref=False,
                            **kw):
    """Run every dataset and, if ``out`` is given, write the CSVs.

    ``out`` is a base path (e.g. ``results/density_knn.csv``): each dataset gets
    its own per-instance sheet ``results/density_knn_10_5.csv`` (numeric-only, so
    ``density_coverage_figure.py --csv`` can read it directly) and all datasets
    are aggregated into one ``results/density_knn_summary.csv``.  ``global_ref``
    switches to the cross-instance reference pool (the pool stays within a dataset;
    sizes are never mixed).
    """
    datasets = datasets or DEFAULT_ALL_DATASETS
    base, ext = os.path.splitext(out) if out else (None, ".csv")
    runner = run_global_dataset if global_ref else run_dataset

    results, all_rows = {}, []
    for path in datasets:
        print(f"\n########## {path} ##########")
        per_csv = f"{base}_{_size_tag(path)}{ext}" if out else None
        per_instance, summary = runner(
            model, normalizer, path, k_list, n_samples, device,
            n_instances=n_instances, out=per_csv, **kw)
        results[path] = (per_instance, summary)
        all_rows += summary_rows(path, per_instance, summary, k_list)

    if out:
        write_summary_csv(all_rows, f"{base}_summary{ext}")
    return results


def do_all_random_sweep_datasets(model, normalizer, k_list, n_samples, device,
                                 random_counts=(1, 5, 10, 25, 50, 100), n_instances=500,
                                 out=None, datasets=None, global_ref=False, **kw):
    """Random-only density sweep over every dataset; same CSV layout as the full run."""
    datasets = datasets or DEFAULT_ALL_DATASETS
    base, ext = os.path.splitext(out) if out else (None, ".csv")
    runner = run_global_random_sweep_dataset if global_ref else run_random_sweep_dataset

    results, all_rows = {}, []
    for path in datasets:
        print(f"\n########## {path} ##########")
        per_csv = f"{base}_{_size_tag(path)}{ext}" if out else None
        per_instance, summary = runner(
            model, normalizer, path, k_list, n_samples, device,
            random_counts=random_counts, n_instances=n_instances, out=per_csv, **kw)
        results[path] = (per_instance, summary)
        all_rows += random_sweep_summary_rows(path, per_instance, summary, k_list)

    if out:
        write_random_sweep_summary_csv(all_rows, f"{base}_summary{ext}")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def set_seed(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="dataset/SD1_train_20_10_500.npy",
                   help="Training dataset .npy (or 'all' for the three SD1 sizes)")
    p.add_argument("--checkpoint", default="checkpoints_dumb_exp/best_det_offline.pt")
    p.add_argument("--n_instances", type=int, default=50,
                   help="Number of instances to score (None/-1 = all)")
    p.add_argument("--n_samples", type=int, default=1000,
                   help="Stochastic rollouts per instance; best (min makespan) = optimal")
    p.add_argument("--k_list", type=int, nargs="+", default=[1, 5, 10])
    p.add_argument("--n_ga", type=int, default=200, help="GA trajectories per instance (ga_pop)")
    p.add_argument("--n_random", type=int, default=100, help="Random trajectories per instance")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--sample_batch", type=int, default=None,
                   help="Cap concurrent envs during sampling (memory); default = n_samples")
    p.add_argument("--batch_cap", type=int, default=256,
                   help="Cap concurrent envs when replaying dataset trajectories")
    p.add_argument("--no_normalize", action="store_true",
                   help="Skip checkpoint feature normalization (NOT recommended)")
    p.add_argument("--no_dedupe", action="store_true",
                   help="Keep duplicate trajectories (default: dedupe like training)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None,
                   help="Results CSV path. With --data all it is a base path: one "
                        "per-instance sheet per dataset (<out>_10_5.csv, ...) plus a "
                        "combined <out>_summary.csv (default: density_knn.csv)")
    p.add_argument("--random_sweep", action="store_true",
                   help="Score ONLY the Random source, at each dataset size in "
                        "--random_counts (ignores --n_random and the other types)")
    p.add_argument("--random_counts", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100],
                   help="Random trajectories/instance to sweep over with --random_sweep")
    p.add_argument("--global_ref", action="store_true",
                   help="Score each instance's optimal transitions against the POOLED "
                        "trajectories of all reference instances instead of only its own "
                        "(streamed, so RAM stays flat; exact, not approximate)")
    p.add_argument("--ref_instances", type=int, default=None,
                   help="Instances contributing to the global pool (default: --n_instances). "
                        "May exceed it: score few instances against a big pool")
    p.add_argument("--global_exclude_self", action="store_true",
                   help="With --global_ref, drop each query instance's OWN trajectories from "
                        "its reference pool ('does some other instance cover this state?')")
    p.add_argument("--query_chunk", type=int, default=1024,
                   help="Query rows per distance tile in global mode (peak temporary = "
                        "query_chunk x transitions-of-one-instance floats)")
    p.add_argument("--eps", action="store_true",
                   help="Also score the four eps-PDR datasets of the paper's Table 10 "
                        "(eps-PDR-0.1/0.2 and eps-PDR-GA-0.1/0.2), read from the "
                        "eps_random_<v> keys.  Composition mirrors check_coverage.py, "
                        "which produced the table's SACo column: eps_random_<v>[:n_eps] "
                        "and + ga_pop[:eps_ga_n].  Works with the local and --global_ref "
                        "pools (not with --random_sweep)")
    p.add_argument("--n_eps", type=int, default=100,
                   help="eps-PDR trajectories per instance (as stored: 100)")
    p.add_argument("--eps_ga_n", type=int, default=50,
                   help="GA trajectories pooled into each eps-PDR-GA source "
                        "(check_coverage.py's pdr_ga_n)")
    return p


def main():
    args = build_argparser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_instances = None if (args.n_instances is None or args.n_instances < 0) else args.n_instances
    normalize = not args.no_normalize
    dedupe = not args.no_dedupe
    if args.eps and args.random_sweep:
        raise SystemExit("--eps is not supported with --random_sweep "
                         "(the sweep scores only the Random source)")

    set_seed(args.seed, device)
    print(f"Device: {device} | checkpoint: {args.checkpoint} | normalize: {normalize} "
          f"| dedupe: {dedupe}")
    model, normalizer = load_model(args.checkpoint, device)
    print(f"Loaded ActorNet (embedding_output_dim d={model.embedding_output_dim}, "
          f"transition-embedding dim={4 * model.embedding_output_dim})")

    common = dict(sample_batch=args.sample_batch, batch_cap=args.batch_cap,
                  normalize=normalize, dedupe=dedupe)
    all_data = args.data.lower() == "all"
    if args.global_ref:
        common.update(ref_instances=args.ref_instances, query_chunk=args.query_chunk,
                      exclude_self=args.global_exclude_self)
        print(f"Reference pool: GLOBAL (pooled over "
              f"{args.ref_instances or n_instances or 'all'} instances, streamed; "
              f"exclude_self={args.global_exclude_self})")
    else:
        print("Reference pool: LOCAL (each instance scored against its own trajectories)")

    if args.random_sweep:
        print(f"Random-only sweep over n_random in {args.random_counts}")
        sweep = (run_global_random_sweep_dataset if args.global_ref
                 else run_random_sweep_dataset)
        if all_data:
            do_all_random_sweep_datasets(
                model, normalizer, args.k_list, args.n_samples, device,
                random_counts=args.random_counts, global_ref=args.global_ref,
                n_instances=(n_instances if n_instances is not None else 500),
                out=args.out or "density_knn_random_sweep.csv", **common)
        else:
            sweep(model, normalizer, args.data, args.k_list, args.n_samples, device,
                  random_counts=args.random_counts,
                  n_instances=n_instances, out=args.out, **common)
        return

    common.update(n_ga=args.n_ga, n_random=args.n_random)
    if args.eps:
        common.update(eps_versions=EPS_VERSIONS, n_eps=args.n_eps, eps_ga_n=args.eps_ga_n)
    if all_data:
        do_all_density_datasets(model, normalizer, args.k_list, args.n_samples, device,
                                global_ref=args.global_ref,
                                n_instances=(n_instances if n_instances is not None else 500),
                                out=args.out or "density_knn.csv", **common)
    else:
        runner = run_global_dataset if args.global_ref else run_dataset
        runner(model, normalizer, args.data, args.k_list, args.n_samples, device,
               n_instances=n_instances, out=args.out, **common)


if __name__ == "__main__":
    main()
