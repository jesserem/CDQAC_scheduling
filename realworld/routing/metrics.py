"""Route-level metrics.

* :func:`amazon_route_score` re-implements the official 2021 Amazon Last Mile
  Routing Research Challenge score (``scoring/score.py`` of MIT-CAVE/rc-cli):
  sequence deviation x (ERP with normalized travel times / number of ERP
  edits), 0 = identical to the driver, ~0.8-1.2 = random shuffle.  The
  recursive memoized original is replaced by an equivalent bottom-up DP (same
  recurrence, same tie-breaking), which is exact and fast for 250-stop routes.
* :func:`hit_rate`, :func:`kendall_rank_correlation`,
  :func:`location_square_deviation`, :func:`edit_distance` follow the
  definitions of LaDe's route-prediction benchmark (``route_prediction/utils/
  eval.py`` of wenhaomin/LaDe), applied to complete routes.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

ERP_GAP = 1000.0


def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Official normalization: z-score over all entries, then shift min to 0."""
    mat = np.asarray(mat, dtype=np.float64)
    z = (mat - mat.mean()) / mat.std()
    return z - z.min()


def sequence_deviation(actual: Sequence[int], proposed: Sequence[int]) -> float:
    """SD with both sequences given as full stop lists *without* the depot."""
    pos = {s: i for i, s in enumerate(actual)}
    comp = [pos[s] for s in proposed]
    n = len(actual)
    if n < 2:
        return 0.0
    total = sum(abs(comp[i] - comp[i - 1]) - 1 for i in range(1, len(comp)))
    return 2.0 / (n * (n - 1)) * total


def erp_per_edit(actual: Sequence[int], proposed: Sequence[int], norm_mat: np.ndarray,
                 g: float = ERP_GAP) -> float:
    """ERP / #edits, bottom-up version of the official memoized recursion."""
    a, s = list(actual), list(proposed)
    la, ls = len(a), len(s)
    # D[i][j] = (distance, edits) for suffixes a[i:], s[j:]
    dist = [[0.0] * (ls + 1) for _ in range(la + 1)]
    cnt = [[0] * (ls + 1) for _ in range(la + 1)]
    for j in range(ls + 1):
        dist[la][j] = g * (ls - j)
        cnt[la][j] = ls - j
    for i in range(la + 1):
        dist[i][ls] = g * (la - i)
        cnt[i][ls] = la - i
    rows = norm_mat.tolist()
    for i in range(la - 1, -1, -1):
        ha = a[i]
        row = rows[ha]
        di, dn = dist[i], dist[i + 1]
        ci, cn = cnt[i], cnt[i + 1]
        for j in range(ls - 1, -1, -1):
            hs = s[j]
            o1 = dn[j + 1] + row[hs]
            o2 = dn[j] + g
            o3 = di[j + 1] + g
            d = min(o1, o2, o3)
            if d == o1:
                c = cn[j + 1] if ha == hs else cn[j + 1] + 1
            elif d == o2:
                c = cn[j] + 1
            else:
                c = ci[j + 1] + 1
            di[j] = d
            ci[j] = c
    return 0.0 if cnt[0][0] == 0 else dist[0][0] / cnt[0][0]


def amazon_route_score(actual: Sequence[int], proposed: Sequence[int], travel: np.ndarray) -> float:
    """Score of a proposed route against the driver's route (lower is better).

    Args:
        actual, proposed: node sequences starting at the depot (node 0), e.g.
            ``[0, 5, 2, ...]``; the return to the depot is appended here, as in
            the official scorer.
        travel: [n, n] travel-time matrix of the route (any unit; it is
            normalized).
    """
    actual = list(actual)
    proposed = list(proposed)
    if len(actual) != len(proposed) or set(actual) != set(proposed) or actual[0] != proposed[0]:
        raise ValueError("invalid route: must be a permutation of the actual stops starting at the depot")
    a = actual + [actual[0]]
    s = proposed + [proposed[0]]
    return sequence_deviation(a[1:-1], s[1:-1]) * erp_per_edit(a, s, _normalize_matrix(travel))


def hit_rate(pred: Sequence[int], label: Sequence[int], k: int = 3) -> float:
    n = min(k, len(label))
    return len(set(pred[:n]) & set(label[:n])) / n if n else 1.0


def kendall_rank_correlation(pred: Sequence[int], label: Sequence[int]) -> float:
    n = len(label)
    if n <= 1:
        return 1.0
    pos = {v: i for i, v in enumerate(pred)}
    p = np.array([pos[v] for v in label])
    upper = np.triu_indices(n, 1)
    diff = np.sign(p[:, None] - p[None, :])[upper]
    # label order is 0..n-1, so pair (i<j) is concordant iff p_i < p_j
    concordant = int((diff < 0).sum())
    return (2 * concordant - len(diff)) / len(diff)


def location_square_deviation(pred: Sequence[int], label: Sequence[int]) -> float:
    pos = {v: i for i, v in enumerate(pred)}
    return float(np.mean([(i - pos[v]) ** 2 for i, v in enumerate(label)])) if len(label) else 0.0


def edit_distance(pred: Sequence[int], label: Sequence[int]) -> int:
    """Levenshtein distance between pred (restricted to label items) and label."""
    keep = set(label)
    a = [x for x in pred if x in keep]
    b = list(label)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def zone_statistics(seq: Sequence[int], zone: np.ndarray) -> Dict[str, int]:
    """Zone switches and revisits (re-entering a zone already left) along a route."""
    zs = [int(zone[v]) for v in seq if zone[v] >= 0]
    switches, revisits, seen = 0, 0, set()
    for i, z in enumerate(zs):
        if i and z != zs[i - 1]:
            switches += 1
            if z in seen:
                revisits += 1
        seen.add(z)
    return dict(zone_switches=switches, zone_revisits=revisits)


def route_prediction_metrics(pred: Sequence[int], label: Sequence[int]) -> Dict[str, float]:
    return dict(hr3=hit_rate(pred, label, 3), krc=kendall_rank_correlation(pred, label),
                lsd=location_square_deviation(pred, label), ed=float(edit_distance(pred, label)))


def summarize(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float, np.floating))})
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r and r[k] is not None], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            out[k] = float(vals.mean())
    return out
