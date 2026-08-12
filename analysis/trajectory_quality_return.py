"""Trajectory Quality (TQ) for offline-FJSP training datasets.

Third leg of the dataset-characterisation trio, alongside ``coverage_action_prefix.py``
(state-action coverage, SACo) and ``density_embedding_knn.py`` (support density around
the model's optimal trajectory).  Where SACo asks *how broadly* a source covers the
reachable state-action space, TQ asks *how good* the trajectories in it are.

Following Schweighofer et al., "Understanding the Effects of Dataset Characteristics on
Offline Reinforcement Learning" (NeurIPS 2021 Deep RL workshop), the relative Trajectory
Quality of a dataset D is its average return, min-max normalized::

    TQ(D) = (g_D - g_min) / (g_max - g_min)

where ``g_D`` is the average return of D's trajectories.  TQ = 0 means "as bad as the
worst trajectory available for this instance", TQ = 1 means "as good as the best".

Returns, not makespans
----------------------
Trajectory returns are the *environment's own* undiscounted returns: every trajectory is
replayed through ``FJSPEnvForSameOpNums`` and its per-step rewards
``R_t = C_LB(s_t) - C_LB(s_{t+1})`` are accumulated (``--gamma`` discounts them if you want
the discounted return instead).  This is the quantity the offline algorithms actually learn
to maximise, so it -- not the makespan -- is what a dataset-quality metric should average.
The replay also recovers each trajectory's makespan for free, which the script cross-checks
against the makespan stored in the dataset (``rules_info[rule]["makespan"]``,
``ga_pop_makespan``, ``random_makespan``) and reports as ``max|replay - stored|``; it is 0
on the SD1 datasets.

Per-instance anchors
--------------------
The paper normalizes against an online DQN expert and a random policy, both global to the
environment.  That does not transfer here, because an FJSP return is on a scale the
instance itself sets.  The env's completion-time bounds run on *normalized* processing
times (``op_pt`` is divided by the instance's largest processing time,
``fjsp_env_same_op_nums.py:144``), so the undiscounted return works out to

    G = C_LB(s_0) - makespan / max_pt        (verified to 5e-9 on SD1 10x5)

and *both* ``C_LB(s_0)`` and ``max_pt`` vary from instance to instance.  A return of -1.8
means something different on every instance, so a single global anchor pair would mostly
measure which instance you were looking at.

The anchors are therefore taken **per instance**: ``g_max`` is the best and ``g_min`` the
worst return among all the trajectories stored for that instance (PDR, GA and Random
pooled).  Every source then scores in [0, 1] on that instance's own scale, and only then
are instances averaged.

The anchors are written to their own CSV (``--out`` base + ``_references_<size>.csv``):
one row per instance with the best/worst return, which source produced each, their
makespans, and ``C_LB(s_0)``.

Quality spread, not just the average
------------------------------------
TQ, being an average, says nothing about how a source's quality is *distributed*.  So
alongside it every source also reports the TQ of its own return quantiles -- by default
``q10`` and ``q90`` (``--quantiles``), the edges of the middle 80% of its trajectories --
and of its single best trajectory (``tq_best``).  Each is computed per instance and then
averaged, so the reported ``q10``/``q90`` are means (with std) over instances, not
quantiles taken across instances.  Read together they separate "uniformly decent" (tight
q10-q90 band) from "mostly mediocre with a few gems" (low q10, high tq_best) -- the second
being what a Random source looks like, and the distinction that matters when the offline
algorithm gets to be selective about which trajectories it learns from.

Makespans are read from the dataset only for the cross-check; the metric itself never
uses them.

Run (all instances of one dataset):

    python trajectory_quality_return.py --data dataset/SD1_train_10_5_1000.npy

All three SD1 sizes at once, writing every result to CSV:

    python trajectory_quality_return.py --data all --out results/tq.csv

That produces, per dataset, a per-instance sheet (``results/tq_10_5.csv``) and an anchor
sheet (``results/tq_references_10_5.csv``), plus one combined ``results/tq_summary.csv``
with a row per (dataset, type).  The per-instance sheets are keyed by ``instance`` exactly
like the coverage sheets, so joining the two on that column gives the (SACo, TQ) plane of
the paper's Fig. 3.

The four ε-PDR datasets of the paper's Table 10 (ε-PDR and ε-PDR-GA at ε ∈ {0.1, 0.2})
can be scored alongside the standard four types with ``--eps``:

    python trajectory_quality_return.py --data all --eps --out results/tq_eps.csv

They are read from the ``eps_random_<v>`` keys that ``add_eps_pdr_rules.py`` stores in the
same .npy files (first 500 instances only -- the run is clipped to those), composed exactly
as in ``check_coverage.py`` (the script behind the table's SACo column):
``eps-PDR-<v> = eps_random_<v>[:100]`` and ``eps-PDR-GA-<v> = that + ga_pop[:50]``
(``--n_eps`` / ``--eps_ga_n``).  The per-instance anchors remain the {PDR, GA, Random}
pool, so every TQ value stays on the same scale as a run without ``--eps`` (ε sources can
therefore score outside [0, 1] on instances where they beat/undercut that pool).

Random-only sweep -- how does the Random source's quality move as it is thinned from 100
to 1 trajectory per instance?  (Quality counterpart of the coverage/density sweeps.)

    python trajectory_quality_return.py --data all --random_sweep \
        --random_counts 1 5 10 25 50 100 --out results/tq_random_sweep.csv

The anchors stay at their full-size values, so the curve isolates the effect of size alone.

Full parameterisation via CLI (see ``build_argparser``) or import the functions
(``run_dataset``, ``do_all_tq_datasets``, ``run_random_sweep_dataset``) into a notebook.
"""

import argparse
import os
import time

import numpy as np
import torch

from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums


# ---------------------------------------------------------------------------
# Dataset-type definitions (identical to coverage_action_prefix.py)
# ---------------------------------------------------------------------------

USED_PDR = [
    "MWR_SPT_masked", "MWR_LPT_masked", "LWR_SPT_masked", "LWR_LPT_masked",
    "MWR_EST_masked", "MWR_LST_masked", "LWR_EST_masked", "LWR_LST_masked",
    "MOPNR_SPT_masked", "MOPNR_LPT_masked", "LOPNR_SPT_masked", "LOPNR_LPT_masked",
    "MOPNR_EST_masked", "MOPNR_LST_masked", "LOPNR_EST_masked", "LOPNR_LST_masked",
]

DATASET_TYPES = ["PDR", "GA", "PDR-GA", "Random"]

# The disjoint base sources.  PDR-GA is derived (PDR + GA), and the per-instance anchors
# are the best/worst return over these three pooled.
BASE_TYPES = ["PDR", "GA", "Random"]

# The four ε-PDR datasets of the paper's Table 10 (a PDR dispatcher that, per step,
# takes a uniformly random feasible action with probability ε).  They are stored in
# the same .npy files under ``eps_random_<v>`` (added by ``add_eps_pdr_rules.py``;
# only the first 500 instances carry them) and are enabled with ``--eps``.  The
# composition mirrors ``check_coverage.py``, which produced the table's SACo column:
# eps-PDR-<v> = eps_random_<v>[:n_eps] and eps-PDR-GA-<v> = that + ga_pop[:eps_ga_n]
# (defaults 100 and 50).  The TQ anchors stay the {PDR, GA, Random} pool, so all
# values remain on the same scale as a run without --eps (an ε source holding a
# trajectory better/worse than that pool can therefore score outside [0, 1]).
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

DEFAULT_ALL_DATASETS = [
    "./dataset/SD1_train_10_5_1000.npy",
    "./dataset/SD1_train_15_10_1000.npy",
    "./dataset/SD1_train_20_10_500.npy",
]

# Quantiles of a source's per-instance return distribution, reported in TQ units alongside
# the mean: the lower/upper edges of the middle 80% of its trajectories.
DEFAULT_QUANTILES = (0.1, 0.9)


# ---------------------------------------------------------------------------
# Replaying trajectories to get their environment returns
# ---------------------------------------------------------------------------

def replay_returns(job_length, op_pt, action_lists, device, batch_cap=512, gamma=1.0):
    """Replay ``action_lists`` in the env; return their (returns, makespans, init_quality).

    The return of a trajectory is the accumulated environment reward,
    ``G = sum_t gamma^t * (C_LB(s_t) - C_LB(s_{t+1}))`` -- exactly what the offline
    algorithms are trained to maximise.  Undiscounted it telescopes to
    ``C_LB(s_0) - makespan / max_pt`` (the env's bounds run on normalized processing times),
    an instance-wise affine image of the makespan; that is a property of the reward shaping,
    not an assumption made here, and it stops holding as soon as ``gamma < 1``.

    The env is vectorized, so a whole batch of trajectories is stepped at once (at most
    ``batch_cap`` at a time).  Returns ``(G [R], makespan [R], C_LB(s_0))``; the makespans
    are the env's *true-clock* makespans, directly comparable to the stored ones.
    """
    if len(action_lists) == 0:
        return (np.zeros(0, dtype=float), np.zeros(0, dtype=float), float("nan"))

    n_j, n_m = job_length.shape[0], op_pt.shape[1]
    rets, spans = [], []
    init_quality = float("nan")

    for start in range(0, len(action_lists), batch_cap):
        batch = np.asarray(action_lists[start:start + batch_cap], dtype=np.int64)  # [R, n_op]
        r, n_op = batch.shape
        env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m, device=device, mask_actions=True)
        env.set_initial_data(np.tile(job_length[None], (r, 1)),
                             np.tile(op_pt[None], (r, 1, 1)))
        init_quality = float(np.asarray(env.init_quality)[0])

        g = np.zeros(r, dtype=float)
        discount = 1.0
        for t in range(n_op):
            _, reward, _ = env.step(batch[:, t])
            g += discount * np.asarray(reward, dtype=float)
            discount *= gamma
        rets.append(g)
        spans.append(np.asarray(env.current_makespan, dtype=float))

    return np.concatenate(rets), np.concatenate(spans), init_quality


def gather_action_lists(instance, n_ga, n_random, dedupe=False, eps_versions=(),
                        n_eps=100):
    """Return ({type: list-of-action-lists}, {type: stored makespans}) for one instance.

    Mirrors ``coverage_action_prefix.gather_action_lists``: PDR = the 16 USED_PDR rules,
    GA = ``ga_pop[:n_ga]``, Random = ``random[:n_random]``.  The stored makespans come
    along for the cross-check (``rules_info``, ``ga_pop_makespan``, ``random_makespan``);
    they are aligned index-for-index with the action lists.

    ``eps_versions`` adds one base source ``eps-PDR-<v>`` per version, holding
    ``eps_random_<v>[:n_eps]``.  ``add_eps_pdr_rules.py`` stores no makespans for these,
    so they do not appear in the returned ``stored`` dict (and are excluded from the
    makespan cross-check; the replay recovers their makespans regardless).

    With ``dedupe`` the whole-trajectory dedup that training applies
    (``remove_duplicate_actions``) is reproduced, keeping the first occurrence of each
    action list.  Unlike prefix *counting*, an average return does change under dedup --
    GA populations hold many identical members, which pull the raw GA mean toward whichever
    solution the GA converged on.
    """
    rules, rules_info = instance["rules"], instance["rules_info"]
    names = [r for r in USED_PDR if r in rules and r in rules_info]

    actions = {
        "PDR": [list(rules[r]) for r in names],
        "GA": [list(a) for a in instance["ga_pop"][:n_ga]],
        "Random": [list(a) for a in instance["random"][:n_random]],
    }
    stored = {
        "PDR": [float(rules_info[r]["makespan"]) for r in names],
        "GA": [float(m) for m in instance["ga_pop_makespan"][:n_ga]],
        "Random": [float(m) for m in instance["random_makespan"][:n_random]],
    }
    for v in eps_versions:
        actions[f"eps-PDR-{v}"] = [list(a) for a in instance[eps_dataset_key(v)][:n_eps]]

    if dedupe:
        for t in BASE_TYPES:
            actions[t], stored[t] = _dedupe(actions[t], stored[t])
        for v in eps_versions:
            t = f"eps-PDR-{v}"
            actions[t], _ = _dedupe(actions[t], [None] * len(actions[t]))
    return actions, {t: np.asarray(stored[t], dtype=float) for t in BASE_TYPES}


def _dedupe(action_lists, values):
    """Keep the first occurrence of each distinct action list (and its paired value)."""
    seen = set()
    a_keep, v_keep = [], []
    for a, v in zip(action_lists, values):
        key = tuple(int(x) for x in a)
        if key not in seen:
            seen.add(key)
            a_keep.append(a)
            v_keep.append(v)
    return a_keep, v_keep


def _first_occurrence_indices(action_lists):
    """Raw index of each distinct action list's first occurrence, in order.

    Aligned with what ``_dedupe`` keeps, so ``idx[j] < m`` says whether the j-th
    deduped trajectory came from the raw prefix ``[:m]``.
    """
    seen = set()
    idx = []
    for i, a in enumerate(action_lists):
        key = tuple(int(x) for x in a)
        if key not in seen:
            seen.add(key)
            idx.append(i)
    return np.asarray(idx, dtype=np.int64)


def instance_returns(instance, device, n_ga=200, n_random=100, batch_cap=512, gamma=1.0,
                     dedupe=False, eps_versions=(), n_eps=100, eps_ga_n=50):
    """Replay every stored trajectory of one instance, grouped by source.

    All sources are replayed in one pooled batch (the env is vectorized, so batching them
    together is strictly cheaper than separate envs) and split again afterwards.

    Returns ``(returns, makespans, stored, init_quality)``: the first three are
    ``{type: np.array}`` over BASE_TYPES + "PDR-GA" (the pooled PDR+GA source) and, per
    requested ε version, ``eps-PDR-<v>`` and ``eps-PDR-GA-<v>``; ``stored`` holds the
    makespans read from the dataset, for the cross-check (base sources only -- the ε
    trajectories carry no stored makespans).
    """
    job_length = np.array(instance["JobLength"])
    op_pt = np.array(instance["OpPT"])

    actions, stored = gather_action_lists(instance, n_ga=n_ga, n_random=n_random,
                                          dedupe=dedupe, eps_versions=eps_versions,
                                          n_eps=n_eps)
    base = BASE_TYPES + [f"eps-PDR-{v}" for v in eps_versions]
    pooled = []
    for t in base:
        pooled += actions[t]
    g, ms, init_quality = replay_returns(job_length, op_pt, pooled, device,
                                         batch_cap=batch_cap, gamma=gamma)

    sizes = [len(actions[t]) for t in base]
    offs = np.cumsum([0] + sizes)
    returns = {t: g[offs[j]:offs[j + 1]] for j, t in enumerate(base)}
    makespans = {t: ms[offs[j]:offs[j + 1]] for j, t in enumerate(base)}

    # PDR-GA is the pooled source.  Under dedup, GA trajectories already present in PDR
    # must not be counted twice, so rebuild the union rather than concatenating.
    if dedupe:
        keep = [i for i, a in enumerate(actions["GA"])
                if tuple(int(x) for x in a) not in {tuple(int(x) for x in p)
                                                    for p in actions["PDR"]}]
        returns["PDR-GA"] = np.concatenate([returns["PDR"], returns["GA"][keep]])
        makespans["PDR-GA"] = np.concatenate([makespans["PDR"], makespans["GA"][keep]])
    else:
        returns["PDR-GA"] = np.concatenate([returns["PDR"], returns["GA"]])
        makespans["PDR-GA"] = np.concatenate([makespans["PDR"], makespans["GA"]])

    # eps-PDR-GA-<v> = eps-PDR-<v> + ga_pop[:eps_ga_n], check_coverage.py's Table 10
    # composition.  The GA slice reuses the GA replays: its entries are the (possibly
    # deduped) GA rows whose ORIGINAL index lies in the raw [:eps_ga_n] prefix; under
    # dedup, GA trajectories already present in the ε source are dropped, as for PDR-GA.
    if eps_versions:
        if dedupe:
            ga_first = _first_occurrence_indices(instance["ga_pop"][:n_ga])
            slice_pos = np.nonzero(ga_first < eps_ga_n)[0]
        else:
            slice_pos = np.arange(min(eps_ga_n, len(actions["GA"])))
        for v in eps_versions:
            et = f"eps-PDR-{v}"
            pos = slice_pos
            if dedupe:
                eps_set = {tuple(int(x) for x in a) for a in actions[et]}
                pos = np.asarray([p for p in slice_pos
                                  if tuple(int(x) for x in actions["GA"][p]) not in eps_set],
                                 dtype=np.int64)
            returns[f"eps-PDR-GA-{v}"] = np.concatenate([returns[et], returns["GA"][pos]])
            makespans[f"eps-PDR-GA-{v}"] = np.concatenate(
                [makespans[et], makespans["GA"][pos]])

    return returns, makespans, stored, init_quality


# ---------------------------------------------------------------------------
# Per-instance anchors and TQ
# ---------------------------------------------------------------------------

def instance_anchors(returns, makespans):
    """Best and worst return among ALL of one instance's trajectories.

    These are the instance's own TQ scale.  Returns are not comparable across instances --
    ``G = C_LB(s_0) - makespan / max_pt`` and both constants are per-instance -- so a global
    anchor pair, as the paper uses, would mostly measure which instance you are looking at.
    Anchoring on the best and worst trajectory *of the instance* puts every source in [0, 1]
    on a scale the instance itself defines, which is what makes the per-instance numbers
    averageable.

    The pool is the disjoint base sources {PDR, GA, Random}, i.e. exactly the trajectories
    being scored, so both anchors are attainable by some measured source.  Being a min and a
    max, they are unaffected by ``--dedupe``.  The ε-PDR sources (``--eps``) are
    deliberately NOT pooled: keeping the anchors fixed leaves every TQ value on the same
    scale as a run without them (an ε trajectory beating the pool's best therefore scores
    above 1, and below 0 for the converse).

    Returns a dict with the best/worst return, the source and makespan of each, and the
    span used as the TQ denominator.
    """
    g = np.concatenate([returns[t] for t in BASE_TYPES])
    ms = np.concatenate([makespans[t] for t in BASE_TYPES])
    src = np.concatenate([np.full(returns[t].shape, j) for j, t in enumerate(BASE_TYPES)])
    if g.size == 0:
        return dict(best_return=float("nan"), worst_return=float("nan"),
                    best_source="", worst_source="", best_makespan=float("nan"),
                    worst_makespan=float("nan"), span=float("nan"))

    hi, lo = int(np.argmax(g)), int(np.argmin(g))
    return dict(
        best_return=float(g[hi]), worst_return=float(g[lo]),
        best_source=BASE_TYPES[int(src[hi])], worst_source=BASE_TYPES[int(src[lo])],
        best_makespan=float(ms[hi]), worst_makespan=float(ms[lo]),
        span=float(g[hi] - g[lo]),
    )


def normalized_tq(g, anchors):
    """The paper's Eq. 1 against this instance's anchors: (g - g_min) / (g_max - g_min).

    Returns nan on a degenerate instance whose trajectories all earn the same return
    (no spread to normalize by).
    """
    span = anchors["span"]
    if not np.isfinite(span) or span <= 0:
        return float("nan")
    return float((g - anchors["worst_return"]) / span)


def quantile_key(q):
    """0.1 -> 'tq_q10' -- the column/summary name of a quantile's TQ."""
    return f"tq_q{q * 100:g}".replace(".", "_")


def source_stats(returns, makespans, anchors, quantiles):
    """All TQ statistics of one source on one instance.

    ``tq`` is the paper's metric (the normalized *average* return).  ``tq_best`` and the
    ``tq_q<p>`` entries normalize the source's best and its p-th-percentile return with the
    same anchors, describing the *spread* of quality inside the source rather than only its
    centre: a source can hold one excellent schedule amid mediocre ones (high ``tq_best``,
    middling ``tq``) or be uniformly decent (a tight q10-q90 band).

    Because ``normalized_tq`` is an increasing affine map, the TQ of the p-th quantile
    return is exactly the p-th quantile of the source's per-trajectory TQ -- taking the
    quantile before or after normalizing is the same number.
    """
    n = returns.size
    mean_return = float(np.mean(returns)) if n else float("nan")
    best_return = float(np.max(returns)) if n else float("nan")

    st = {
        "n_traj": int(n),
        "mean_return": mean_return,
        "best_return": best_return,
        "mean_makespan": float(np.mean(makespans)) if makespans.size else float("nan"),
        "tq": normalized_tq(mean_return, anchors),
        "tq_best": normalized_tq(best_return, anchors),
    }
    for q in quantiles:
        gq = float(np.quantile(returns, q)) if n else float("nan")
        st[quantile_key(q)] = normalized_tq(gq, anchors)
    return st


def tq_keys(quantiles):
    """The TQ-valued stat names, in report order: mean, quantiles (ascending), best."""
    return ["tq"] + [quantile_key(q) for q in sorted(quantiles)] + ["tq_best"]


def run_instance(instance, device, n_ga=200, n_random=100, batch_cap=512, gamma=1.0,
                 dedupe=False, quantiles=DEFAULT_QUANTILES, eps_versions=(), n_eps=100,
                 eps_ga_n=50):
    """Score one instance.

    Returns (stats, anchors, ms_err), where ``stats[type]`` is the ``source_stats`` dict of
    that source and ``ms_err`` is the max |replayed makespan - makespan stored in the
    dataset| over this instance's trajectories: 0 confirms the replayed returns describe
    the same trajectories the dataset claims to hold.  The cross-check covers the base
    sources only (the ε trajectories store no makespans); the ε sources are scored against
    the SAME {PDR, GA, Random} anchors, so their TQ may fall outside [0, 1].
    """
    returns, makespans, stored, init_quality = instance_returns(
        instance, device, n_ga=n_ga, n_random=n_random, batch_cap=batch_cap, gamma=gamma,
        dedupe=dedupe, eps_versions=eps_versions, n_eps=n_eps, eps_ga_n=eps_ga_n)
    anchors = instance_anchors(returns, makespans)
    anchors["init_quality"] = init_quality

    ms_err = max((float(np.max(np.abs(makespans[t] - stored[t]))) if stored[t].size else 0.0)
                 for t in BASE_TYPES)

    stats = {t: source_stats(returns[t], makespans[t], anchors, quantiles)
             for t in scored_types(eps_versions)}
    return stats, anchors, ms_err


def run_instance_random_sweep(instance, device, random_counts, n_ga=200, n_random=100,
                              batch_cap=512, gamma=1.0, dedupe=False,
                              quantiles=DEFAULT_QUANTILES):
    """Score one instance using ONLY the Random source, at several dataset sizes.

    For every ``m`` in ``random_counts`` we keep the *first* ``m`` stored random
    trajectories -- the same prefix a thinned training run would use -- and average their
    returns.  The anchors stay the instance's full best/worst (computed over all sources,
    exactly as in the full run), so a sweep value is directly comparable to the Random
    column there.  Deduping is applied within each prefix, as training would.

    Returns ({m: source_stats}, anchors).
    """
    job_length = np.array(instance["JobLength"])
    op_pt = np.array(instance["OpPT"])

    # One replay covers both jobs: the anchors need PDR + GA + Random[:n_random], the sweep
    # needs Random[:max_m].  Replay PDR + GA + Random[:max(n_random, max_m)] once and slice.
    max_m = max(random_counts)
    actions, _ = gather_action_lists(instance, n_ga=n_ga,
                                     n_random=max(n_random, max_m), dedupe=False)
    pooled = actions["PDR"] + actions["GA"] + actions["Random"]
    g, ms, init_quality = replay_returns(job_length, op_pt, pooled, device,
                                         batch_cap=batch_cap, gamma=gamma)

    n_pdr, n_ga_traj = len(actions["PDR"]), len(actions["GA"])
    split = n_pdr + n_ga_traj
    raw = actions["Random"]
    g_raw, ms_raw = g[split:], ms[split:]

    # Anchors are min/max, so deduping cannot move them -- take them from the raw pool.
    # The Random slice is capped at n_random so the anchors match the full run exactly,
    # even when the sweep reaches past it.
    anchor_returns = {"PDR": g[:n_pdr], "GA": g[n_pdr:split], "Random": g_raw[:n_random]}
    anchor_makespans = {"PDR": ms[:n_pdr], "GA": ms[n_pdr:split],
                        "Random": ms_raw[:n_random]}
    anchors = instance_anchors(anchor_returns, anchor_makespans)
    anchors["init_quality"] = init_quality

    # First-occurrence order, so the deduped set of the first m raw trajectories is exactly
    # the prefix whose first index is < m (one replay serves every m).
    seen = set()
    first_idx = []
    for i, a in enumerate(raw[:max_m]):
        key = tuple(int(x) for x in a)
        if dedupe and key in seen:
            continue
        seen.add(key)
        first_idx.append(i)
    first_idx = np.asarray(first_idx, dtype=np.int64)

    stats = {}
    for m in random_counts:
        sel = first_idx[first_idx < m]
        stats[m] = source_stats(g_raw[sel], ms_raw[sel], anchors, quantiles)
    return stats, anchors


def _summarise(values, round_val=4):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return dict(mean=float("nan"), std=float("nan"), min=float("nan"), max=float("nan"))
    return dict(mean=round(float(a.mean()), round_val), std=round(float(a.std()), round_val),
                min=round(float(a.min()), round_val), max=round(float(a.max()), round_val))


# ---------------------------------------------------------------------------
# Dataset drivers
# ---------------------------------------------------------------------------

def _aggregate(stat_rows, quantiles):
    """Aggregate a list of per-instance ``source_stats`` dicts into one summary dict.

    Every TQ-valued stat (the mean, each quantile, the best) is summarised across instances
    as mean/std/min/max; the raw return/makespan/count stats are averaged.
    """
    summary = {k: _summarise([r[k] for r in stat_rows]) for k in tq_keys(quantiles)}
    for k in ("mean_return", "mean_makespan", "n_traj"):
        summary[k] = float(np.mean([r[k] for r in stat_rows]))
    return summary


def _print_group(label, s, quantiles):
    """One report line: TQ mean +- std, the quantile band, and the raw scales behind it."""
    band = "  ".join(f"q{q * 100:g} {s[quantile_key(q)]['mean']:5.2f}"
                     f"±{s[quantile_key(q)]['std']:.2f}" for q in sorted(quantiles))
    print(f"  {label:15s} TQ {s['tq']['mean']:6.2f}±{s['tq']['std']:.2f} "
          f"[{s['tq']['min']:.2f},{s['tq']['max']:.2f}]  |  {band}  |  "
          f"best {s['tq_best']['mean']:5.2f}±{s['tq_best']['std']:.2f}  "
          f"(return mean~{s['mean_return']:.1f}, makespan mean~{s['mean_makespan']:.1f} "
          f"over {s['n_traj']:.0f} traj)")


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


def run_dataset(data_path, device, n_instances=None, n_ga=200, n_random=100, batch_cap=512,
                gamma=1.0, dedupe=False, quantiles=DEFAULT_QUANTILES, verbose=True,
                out=None, refs_out=None, eps_versions=(), n_eps=100, eps_ga_n=50):
    """Run the trajectory-quality experiment over one .npy dataset file.

    Returns (per_instance, summary, references): ``per_instance`` is a list of dicts (one
    per instance); ``references`` is the per-instance anchor rows; ``summary`` is
    ``{type: {<tq stat>: {mean,std,min,max}, 'mean_return': mean, 'mean_makespan': mean,
    'n_traj': mean}}`` aggregated across instances, where ``<tq stat>`` runs over ``tq``,
    each ``tq_q<p>`` and ``tq_best``.  ``out`` / ``refs_out`` write the per-instance and
    anchor CSVs.

    ``eps_versions`` (e.g. ``EPS_VERSIONS``) additionally scores the paper's Table 10
    ε-PDR datasets: ``eps-PDR-<v>`` = ``eps_random_<v>[:n_eps]`` and ``eps-PDR-GA-<v>``
    = that + ``ga_pop[:eps_ga_n]`` (check_coverage.py's composition), against the same
    per-instance anchors as the standard four types.
    """
    data = np.load(data_path, allow_pickle=True)
    if n_instances is not None:
        data = data[:n_instances]
    data = _clamp_to_eps_instances(data, data_path, eps_versions)
    types = scored_types(eps_versions)

    per_instance, references = [], []
    accum = {t: [] for t in types}
    worst_ms_err = 0.0

    t0 = time.time()
    for i in range(len(data)):
        stats, anchors, ms_err = run_instance(
            data[i], device, n_ga=n_ga, n_random=n_random, batch_cap=batch_cap,
            gamma=gamma, dedupe=dedupe, quantiles=quantiles, eps_versions=eps_versions,
            n_eps=n_eps, eps_ga_n=eps_ga_n)
        worst_ms_err = max(worst_ms_err, ms_err)

        rec = {"instance": i,
               "best_return": anchors["best_return"],
               "worst_return": anchors["worst_return"],
               "return_span": anchors["span"]}
        for t in types:
            for k, v in stats[t].items():
                rec[f"{t}_{k}"] = v
            accum[t].append(stats[t])
        per_instance.append(rec)
        references.append(reference_row(i, anchors, ms_err))

        if verbose and (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(data) - i - 1)
            print(f"[{i + 1}/{len(data)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    summary = {t: _aggregate(accum[t], quantiles) for t in types}

    if verbose:
        print(f"\n==== Trajectory Quality (TQ) for: {data_path} ====")
        print(f"(normalized env return of each source; 0 = the instance's worst stored "
              f"trajectory, 1 = its best.  TQ = the paper's metric (normalized *mean* "
              f"return); q<p> = the p-th percentile of the source's own trajectories, so "
              f"the band shows how spread its quality is; best = its single best "
              f"trajectory.  Every number is computed per instance, then mean±std across "
              f"the {len(data)} instances; gamma={gamma}, dedupe={dedupe}, "
              f"{time.time() - t0:.1f}s)")
        if eps_versions:
            print(f"(eps sources: eps-PDR-<v> = eps_random_<v>[:{n_eps}], eps-PDR-GA-<v> "
                  f"= that + ga_pop[:{eps_ga_n}]; anchored on the same {{PDR, GA, Random}} "
                  f"pool, so their TQ may fall outside [0, 1])")
        for t in types:
            _print_group(t, summary[t], quantiles)
        print(f"  cross-check: max|replayed makespan - stored makespan| = {worst_ms_err:g}")

    if out:
        write_csv(per_instance, out, quantiles, types=types)
    if refs_out:
        write_references_csv(references, refs_out)

    return per_instance, summary, references


def run_random_sweep_dataset(data_path, device, random_counts=(1, 5, 10, 25, 50, 100),
                             n_instances=None, n_ga=200, n_random=100, batch_cap=512,
                             gamma=1.0, dedupe=False, quantiles=DEFAULT_QUANTILES,
                             verbose=True, out=None, refs_out=None):
    """Random-only quality sweep over one .npy dataset: TQ vs. #random runs.

    Returns (per_instance, summary, references), same shape as ``run_dataset`` but keyed
    by ``n_random`` instead of dataset type.
    """
    random_counts = sorted(set(int(m) for m in random_counts))
    data = np.load(data_path, allow_pickle=True)
    if n_instances is not None:
        data = data[:n_instances]

    available = len(data[0]["random"]) if len(data) else 0
    if random_counts[-1] > available:
        print(f"WARNING: --random_counts asks for {random_counts[-1]} trajectories but the "
              f"dataset stores only {available} per instance; larger counts are clipped.")

    per_instance, references = [], []
    accum = {m: [] for m in random_counts}

    t0 = time.time()
    for i in range(len(data)):
        stats, anchors = run_instance_random_sweep(
            data[i], device, random_counts, n_ga=n_ga, n_random=n_random,
            batch_cap=batch_cap, gamma=gamma, dedupe=dedupe, quantiles=quantiles)
        for m in random_counts:
            rec = {"instance": i, "n_random": m,
                   "best_return": anchors["best_return"],
                   "worst_return": anchors["worst_return"],
                   "return_span": anchors["span"]}
            for k, v in stats[m].items():
                rec[f"Random_{k}"] = v
            per_instance.append(rec)
            accum[m].append(stats[m])
        references.append(reference_row(i, anchors))

        if verbose and (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(data) - i - 1)
            print(f"[{i + 1}/{len(data)}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    summary = {m: _aggregate(accum[m], quantiles) for m in random_counts}

    if verbose:
        print(f"\n==== Random-only TQ vs. #random runs/instance: {data_path} ====")
        print(f"(the first m random trajectories, scored against the same per-instance "
              f"anchors as the full run; TQ = normalized mean return, q<p> = the p-th "
              f"percentile of those m trajectories, best = the best of them; mean±std over "
              f"{len(data)} instances, gamma={gamma}, dedupe={dedupe}, "
              f"{time.time() - t0:.1f}s)")
        for m in random_counts:
            _print_group(f"n_random={m}", summary[m], quantiles)

    if out:
        write_random_sweep_csv(per_instance, out, quantiles)
    if refs_out:
        write_references_csv(references, refs_out)

    return per_instance, summary, references


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _ensure_dir(path):
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _write(rows, fields, out_path, what):
    import csv
    _ensure_dir(out_path)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {what} to {out_path}")


def reference_row(i, anchors, ms_err=None):
    """One instance's TQ anchors, as written to the references CSV."""
    row = {
        "instance": i,
        "best_return": anchors["best_return"],
        "worst_return": anchors["worst_return"],
        "return_span": anchors["span"],
        "best_source": anchors["best_source"],
        "worst_source": anchors["worst_source"],
        "best_makespan": anchors["best_makespan"],
        "worst_makespan": anchors["worst_makespan"],
        "init_quality": anchors["init_quality"],
    }
    if ms_err is not None:
        row["makespan_check_err"] = ms_err
    return row


REFERENCE_FIELDS = ["instance", "best_return", "worst_return", "return_span",
                    "best_source", "worst_source", "best_makespan", "worst_makespan",
                    "init_quality", "makespan_check_err"]


def write_references_csv(references, out_path):
    """Per-instance TQ anchors: the best/worst return that define that instance's scale.

    ``init_quality`` is ``C_LB(s_0)``, the constant the env's rewards are measured against;
    with ``gamma = 1`` a trajectory's return is ``init_quality - makespan / max_pt``, so the
    row is self-checking.  ``makespan_check_err`` is 0 when the replayed makespans agree
    with the ones stored in the dataset.
    """
    rows = [{f: r.get(f, "") for f in REFERENCE_FIELDS} for r in references]
    _write(rows, REFERENCE_FIELDS, out_path, "per-instance TQ references (anchors)")


def write_csv(per_instance, out_path, quantiles=DEFAULT_QUANTILES, types=None):
    """Per-instance CSV for one dataset (numeric-only; joins to the coverage sheets on
    ``instance``, giving the paper's (SACo, TQ) plane)."""
    types = types or DATASET_TYPES
    stat_fields = tq_keys(quantiles) + ["mean_return", "best_return", "mean_makespan",
                                        "n_traj"]
    fields = (["instance", "best_return", "worst_return", "return_span"]
              + [f"{t}_{f}" for t in types for f in stat_fields])
    _write(per_instance, fields, out_path, "per-instance trajectory quality")


def summary_rows(dataset, per_instance, summary, quantiles=DEFAULT_QUANTILES):
    """Long-format rows (one per dataset x type) for the cross-dataset sheet.

    Iterates the summary's own keys, so ε types (when scored) get rows too.
    """
    return [dict({"dataset": dataset, "type": t, "n_instances": len(per_instance),
                  "mean_trajectories": round(summary[t]["n_traj"], 2),
                  "mean_return": round(summary[t]["mean_return"], 2),
                  "mean_makespan": round(summary[t]["mean_makespan"], 2)},
                 **_tq_columns(summary[t], quantiles))
            for t in summary]


def _tq_columns(entry, quantiles):
    """mean/std of every TQ-valued stat (mean TQ, each quantile, best), + min/max for TQ."""
    cols = {}
    for k in tq_keys(quantiles):
        s = entry[k]
        cols[f"{k}_mean"] = s["mean"]
        cols[f"{k}_std"] = s["std"]
    cols["tq_min"] = entry["tq"]["min"]
    cols["tq_max"] = entry["tq"]["max"]
    return cols


def _tq_column_names(quantiles):
    names = []
    for k in tq_keys(quantiles):
        names += [f"{k}_mean", f"{k}_std"]
        if k == "tq":
            names += ["tq_min", "tq_max"]
    return names


def summary_fields(quantiles):
    return (["dataset", "type", "n_instances", "mean_trajectories", "mean_return",
             "mean_makespan"] + _tq_column_names(quantiles))


def write_summary_csv(rows, out_path, quantiles=DEFAULT_QUANTILES):
    """Aggregated TQ across all datasets/types (one sheet, human-readable)."""
    _write(rows, summary_fields(quantiles), out_path,
           "combined trajectory-quality summary")


def write_random_sweep_csv(per_instance, out_path, quantiles=DEFAULT_QUANTILES):
    """Per-(instance, n_random) sheet of the random-only sweep for one dataset."""
    stat_fields = tq_keys(quantiles) + ["mean_return", "best_return", "mean_makespan",
                                        "n_traj"]
    fields = (["instance", "n_random"] + [f"Random_{f}" for f in stat_fields]
              + ["best_return", "worst_return", "return_span"])
    _write(per_instance, fields, out_path, "per-instance random sweep")


def random_sweep_summary_rows(dataset, per_instance, summary, quantiles=DEFAULT_QUANTILES):
    """Long-format rows (one per dataset x n_random) for the cross-dataset sheet."""
    n_inst = len({rec["instance"] for rec in per_instance})
    return [dict({"dataset": dataset, "n_random": m, "n_instances": n_inst,
                  "mean_trajectories": round(summary[m]["n_traj"], 2),
                  "mean_return": round(summary[m]["mean_return"], 2)},
                 **_tq_columns(summary[m], quantiles))
            for m in sorted(summary)]


def sweep_summary_fields(quantiles):
    return (["dataset", "n_random", "n_instances", "mean_trajectories", "mean_return"]
            + _tq_column_names(quantiles))


def write_random_sweep_summary_csv(rows, out_path, quantiles=DEFAULT_QUANTILES):
    """Aggregated random-only sweep across all datasets (one sheet)."""
    _write(rows, sweep_summary_fields(quantiles), out_path,
           "combined random-sweep summary")


# ---------------------------------------------------------------------------
# Convenience: all three SD1 sizes
# ---------------------------------------------------------------------------

def _size_tag(data_path):
    """'dataset/SD1_train_10_5_1000.npy' -> '10_5' (falls back to the file stem)."""
    stem = os.path.splitext(os.path.basename(data_path))[0]
    parts = stem.split("_")
    for i in range(len(parts) - 1):
        if parts[i].isdigit() and parts[i + 1].isdigit():
            return f"{parts[i]}_{parts[i + 1]}"
    return stem


def _out_paths(out, data_path):
    """(per-instance sheet, references sheet) for one dataset under the ``--out`` base."""
    if not out:
        return None, None
    base, ext = os.path.splitext(out)
    tag = _size_tag(data_path)
    return f"{base}_{tag}{ext}", f"{base}_references_{tag}{ext}"


def do_all_tq_datasets(device, n_instances=None, out=None, datasets=None,
                       quantiles=DEFAULT_QUANTILES, eps_versions=(), **kw):
    """Run every dataset and, if ``out`` is given, write the CSVs.

    ``out`` is a base path (e.g. ``results/tq.csv``): each dataset gets a per-instance
    sheet ``results/tq_10_5.csv`` and an anchor sheet ``results/tq_references_10_5.csv``,
    and all datasets are aggregated into ``results/tq_summary.csv``.  Also prints the
    pooled TQ across all datasets (an ``ALL`` row in the summary sheet).  ``eps_versions``
    is forwarded to ``run_dataset`` and extends the pooled report with the ε types.
    """
    datasets = datasets or DEFAULT_ALL_DATASETS
    base, ext = os.path.splitext(out) if out else (None, ".csv")
    keys = tq_keys(quantiles)
    types = scored_types(eps_versions)

    results, all_rows = {}, []
    pooled = {t: {k: [] for k in keys} for t in types}
    for path in datasets:
        print(f"\n########## {path} ##########")
        per_csv, refs_csv = _out_paths(out, path)
        per_instance, summary, references = run_dataset(
            path, device, n_instances=n_instances, out=per_csv, refs_out=refs_csv,
            quantiles=quantiles, eps_versions=eps_versions, **kw)
        results[path] = (per_instance, summary, references)
        all_rows += summary_rows(path, per_instance, summary, quantiles)
        for t in types:
            for k in keys:
                pooled[t][k] += [rec[f"{t}_{k}"] for rec in per_instance]

    print("\n==== TQ pooled over all datasets ====")
    for t in types:
        entry = {k: _summarise(pooled[t][k], round_val=2) for k in keys}
        band = "  ".join(f"q{q * 100:g} {entry[quantile_key(q)]['mean']:5.2f}"
                         f"±{entry[quantile_key(q)]['std']:.2f}" for q in sorted(quantiles))
        print(f"  {t:15s} TQ {entry['tq']['mean']:5.2f}±{entry['tq']['std']:.2f}  |  {band}"
              f"  |  best {entry['tq_best']['mean']:5.2f}±{entry['tq_best']['std']:.2f}")
        all_rows.append(dict(
            {"dataset": "ALL", "type": t, "n_instances": len(pooled[t]["tq"]),
             "mean_trajectories": "", "mean_return": "", "mean_makespan": ""},
            **_tq_columns(entry, quantiles)))

    if out:
        write_summary_csv(all_rows, f"{base}_summary{ext}", quantiles)
    return results


def do_all_random_sweep_datasets(device, random_counts=(1, 5, 10, 25, 50, 100),
                                 n_instances=None, out=None, datasets=None,
                                 quantiles=DEFAULT_QUANTILES, **kw):
    """Random-only quality sweep over every dataset; same CSV layout as the full run."""
    datasets = datasets or DEFAULT_ALL_DATASETS
    base, ext = os.path.splitext(out) if out else (None, ".csv")

    results, all_rows = {}, []
    for path in datasets:
        print(f"\n########## {path} ##########")
        per_csv, refs_csv = _out_paths(out, path)
        per_instance, summary, references = run_random_sweep_dataset(
            path, device, random_counts=random_counts, n_instances=n_instances,
            out=per_csv, refs_out=refs_csv, quantiles=quantiles, **kw)
        results[path] = (per_instance, summary, references)
        all_rows += random_sweep_summary_rows(path, per_instance, summary, quantiles)

    if out:
        write_random_sweep_summary_csv(all_rows, f"{base}_summary{ext}", quantiles)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="dataset/SD1_train_20_10_500.npy",
                   help="Training dataset .npy (or 'all' for the three SD1 sizes)")
    p.add_argument("--n_instances", type=int, default=-1,
                   help="Number of instances to score (-1 = all)")
    p.add_argument("--n_ga", type=int, default=200, help="GA trajectories per instance (ga_pop)")
    p.add_argument("--n_random", type=int, default=100, help="Random trajectories per instance")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Discount for the trajectory return (1.0 = the paper's undiscounted "
                        "return, which telescopes to C_LB(s_0) - makespan/max_pt)")
    p.add_argument("--dedupe", action="store_true",
                   help="Drop duplicate trajectories before averaging, as training does "
                        "(GA populations hold many identical members, which bias the raw mean)")
    p.add_argument("--quantiles", type=float, nargs="+", default=list(DEFAULT_QUANTILES),
                   help="Quantiles of each source's per-instance return distribution to "
                        "report in TQ units alongside the mean (default: 0.1 0.9, the edges "
                        "of the middle 80%% of its trajectories)")
    p.add_argument("--batch_cap", type=int, default=512,
                   help="Max trajectories replayed in one vectorized env (the default fits "
                        "an instance's whole pool -- 16 PDR + 200 GA + 100 Random -- in one)")
    p.add_argument("--device", default="cpu",
                   help="Device for the env replay. CPU is the right default: the env's "
                        "step is numpy-bound, so CUDA only adds transfer overhead (measured "
                        "slightly slower on SD1 10x5)")
    p.add_argument("--out", default=None,
                   help="Results CSV path. Also used as a base path: each dataset gets a "
                        "per-instance sheet (<out>_10_5.csv), an anchor sheet "
                        "(<out>_references_10_5.csv) and, with --data all, a combined "
                        "<out>_summary.csv (default: tq.csv)")
    p.add_argument("--random_sweep", action="store_true",
                   help="Score ONLY the Random source, at each dataset size in "
                        "--random_counts (ignores the other types)")
    p.add_argument("--random_counts", type=int, nargs="+", default=[1, 5, 10, 25, 50, 100],
                   help="Random trajectories/instance to sweep over with --random_sweep")
    p.add_argument("--eps", action="store_true",
                   help="Also score the four eps-PDR datasets of the paper's Table 10 "
                        "(eps-PDR-0.1/0.2 and eps-PDR-GA-0.1/0.2), read from the "
                        "eps_random_<v> keys.  Composition mirrors check_coverage.py, "
                        "which produced the table's SACo column: eps_random_<v>[:n_eps] "
                        "and + ga_pop[:eps_ga_n].  Anchors stay the {PDR, GA, Random} "
                        "pool, so all TQ values stay comparable to a run without --eps")
    p.add_argument("--n_eps", type=int, default=100,
                   help="eps-PDR trajectories per instance (as stored: 100)")
    p.add_argument("--eps_ga_n", type=int, default=50,
                   help="GA trajectories pooled into each eps-PDR-GA source "
                        "(check_coverage.py's pdr_ga_n)")
    return p


def main():
    args = build_argparser().parse_args()
    n_instances = None if (args.n_instances is None or args.n_instances < 0) else args.n_instances
    all_data = args.data.lower() == "all"
    device = torch.device(args.device)
    quantiles = tuple(sorted(args.quantiles))
    if any(not 0.0 <= q <= 1.0 for q in quantiles):
        raise SystemExit(f"--quantiles must lie in [0, 1]; got {list(args.quantiles)}")
    kw = dict(n_ga=args.n_ga, n_random=args.n_random, batch_cap=args.batch_cap,
              gamma=args.gamma, dedupe=args.dedupe, quantiles=quantiles)

    if args.random_sweep:
        if args.eps:
            raise SystemExit("--eps is not supported with --random_sweep "
                             "(the sweep scores only the Random source)")
        print(f"Random-only sweep over n_random in {args.random_counts} (device={device})")
        out = args.out or "tq_random_sweep.csv"
        if all_data:
            do_all_random_sweep_datasets(device, random_counts=args.random_counts,
                                         n_instances=n_instances, out=out, **kw)
        else:
            per_csv, refs_csv = _out_paths(out if args.out else None, args.data)
            run_random_sweep_dataset(args.data, device, random_counts=args.random_counts,
                                     n_instances=n_instances, out=per_csv,
                                     refs_out=refs_csv, **kw)
        return

    kw.update(eps_versions=EPS_VERSIONS if args.eps else (), n_eps=args.n_eps,
              eps_ga_n=args.eps_ga_n)
    print(f"Trajectory quality (device={device}, eps={args.eps})")
    if all_data:
        do_all_tq_datasets(device, n_instances=n_instances, out=args.out or "tq.csv", **kw)
    else:
        per_csv, refs_csv = _out_paths(args.out, args.data)
        run_dataset(args.data, device, n_instances=n_instances, out=per_csv,
                    refs_out=refs_csv, **kw)


if __name__ == "__main__":
    main()
