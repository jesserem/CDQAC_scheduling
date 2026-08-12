"""
Return-distribution portraits of a distributional (QR-DQN) critic for offline FJSP.

This is a qualitative, presentation-oriented companion to ``density_embedding_knn.py``.
Where that script measures how densely the offline dataset covers the model's optimal
trajectory in embedding space, this one opens up the *return distribution* the learned
quantile critic assigns to individual state-action pairs and turns three of them into a
single publication-quality figure for the main paper.

What the critic gives us
------------------------
The checkpoint ``Quantile_critic/Check-*/latest_check.pt`` stores a ``QRDQNNet`` (key
``q_net``): a Dual-Attention encoder feeding ``n_critics=2`` dueling **quantile** heads,
each emitting ``num_quantiles=64`` monotone quantiles of the return ``Z(s,a)`` for every
op-machine candidate pair.  Following the control rule used at test time
(``mqrdqn.get_min_q``) we reduce the two critics per (s,a) by taking the *whole quantile
vector of the critic with the lower mean* (conservative estimate).

The return is ``Z(s,a) = C_LB(s) - makespan`` (the FJSP reward telescopes:
``r_t = C_LB(s_t) - C_LB(s_{t+1})`` and ``gamma=1``), so **higher Z == lower makespan ==
better**, and lower Z == worse.  Note Z is heavily confounded by episode progress (early
states have very negative Z, near-terminal states Z ~ 0), which is exactly why the
"excellent vs bad" contrast below is made *within a single state*.

The three archetypes (all from ONE decision state)
--------------------------------------------------
Reaching a state by replaying an offline-dataset trajectory (the same PDR / GA-pop /
Random action lists used for training, deduplicated as in training), we evaluate every
*valid* candidate action at that state and pick three of them:

  * **Excellent return** - the candidate with the highest mean Z (confidently good).
  * **Low return**       - the candidate with the lowest mean Z (confidently poor).
  * **High-variance / heavy-tailed (risky)** - a third candidate at the same state whose
    top-decile outcomes reach the excellent action's expected return (CVaR_90% >=
    E[Z]_excellent, plus an optional ``--risky_upside_margin``) but whose mean is clearly
    lower (``--risky_mean_gap`` x the excellent-vs-bad spread below the excellent E[Z]),
    carrying the heaviest downside among the qualifiers: the gamble whose best case is as
    good as the safe pick's typical outcome, but that usually ends much worse.  (The
    upside condition is anchored to the excellent action's *mean* rather than its own
    upper tail: a scan showed no candidate ever strictly dominates the max-mean action's
    CVaR_90% - upper tails track means under the conservative min-critic reduction.)

Because all three share the same state, the x-axis (return) is directly comparable and
the differences reflect the *action*, not the clock.  (An extensive scan of >200k
state-action pairs from this checkpoint found the learned return distributions to be
essentially unimodal; genuine multi-modality does not occur, so the third archetype is a
faithful high-variance/heavy-tailed example rather than a manufactured bimodal one.)

Run (defaults: 10x5 training set, first 500 instances, auto CUDA):

    python quantile_distribution_experiment.py

Outputs (into ``--out_dir``, default ``quantile_dist_figures/``):
    return_distributions.pdf / .png           - the main-paper figure (all three)
    return_distributions_good_bad.pdf / .png  - two-panel variant (excellent + low only)
    return_distribution_excellent.pdf / .png  - the excellent panel alone, no title/stats
                                                text (bare drop-in figure element)
    return_distributions.npz                  - the selected quantile vectors + metadata
"""

import argparse
import os
import random

import numpy as np
import torch

from cdqac.network.main_model import QRDQNNet
from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums


# ---------------------------------------------------------------------------
# Dataset-type definitions (mirrors density_embedding_knn.py / check_coverage.py)
# ---------------------------------------------------------------------------

USED_PDR = [
    "MWR_SPT_masked", "MWR_LPT_masked", "LWR_SPT_masked", "LWR_LPT_masked",
    "MWR_EST_masked", "MWR_LST_masked", "LWR_EST_masked", "LWR_LST_masked",
    "MOPNR_SPT_masked", "MOPNR_LPT_masked", "LOPNR_SPT_masked", "LOPNR_LPT_masked",
    "MOPNR_EST_masked", "MOPNR_LST_masked", "LOPNR_EST_masked", "LOPNR_LST_masked",
]

# QRDQNNet architecture stored in latest_check.pt (verified by strict load; matches the
# Quantile_critic/Check-*/config.yml -> use_adv_net, n_critics=2, num_quantiles=64).
QRDQN_KWARGS = dict(
    fea_j_input_dim=10,
    fea_m_input_dim=8,
    layer_fea_output_dim=(32, 8),
    num_heads_OAB=(4, 4),
    num_heads_MAB=(4, 4),
    num_mlp_layers_critic=3,
    hidden_dim_critic=64,
    num_quantiles=64,
    use_adv_net=True,
    n_critics=2,
    layer_norm=False,
    dropout_prob_q=0,
)


# ---------------------------------------------------------------------------
# Model loading & input normalization (same z-scoring the critic was trained with)
# ---------------------------------------------------------------------------

class Normalizer:
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


def load_critic(checkpoint_path, device):
    """Build QRDQNNet, load pretrained ``q_net`` weights, return (model, normalizer)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = QRDQNNet(**QRDQN_KWARGS)
    model.load_state_dict(ckpt["q_net"], strict=True)
    model.to(device)
    model.eval()
    return model, Normalizer(ckpt, device)


@torch.no_grad()
def candidate_quantiles(model, normalizer, state, device, normalize=True):
    """Return the per-candidate return-quantile matrix at a *single-env* state.

    Output ``Z`` has shape [J*M, num_quantiles]; rows for masked/invalid candidate pairs
    are all ``-inf``.  The two critics are reduced per candidate with the ``get_min_q``
    rule used for control: keep the full quantile vector of whichever critic has the lower
    mean (a conservative return estimate).
    """
    fj = normalizer.fea_j(state.fea_j_tensor) if normalize else state.fea_j_tensor
    fm = normalizer.fea_m(state.fea_m_tensor) if normalize else state.fea_m_tensor
    fp = normalizer.fea_pairs(state.fea_pairs_tensor) if normalize else state.fea_pairs_tensor
    q_list, _ = model(fj, state.op_mask_tensor, state.candidate_tensor, fm,
                      state.mch_mask_tensor, state.comp_idx_tensor,
                      state.dynamic_pair_mask_tensor, fp)
    q0, q1 = q_list[0][0], q_list[1][0]            # [J*M, N] each (single env)
    take_second = q0.mean(-1) > q1.mean(-1)        # pick lower-mean critic per candidate
    Z = torch.where(take_second[:, None], q1, q0)
    return Z.cpu().numpy()


# ---------------------------------------------------------------------------
# Dataset trajectories (deduped like training) -> reachable states
# ---------------------------------------------------------------------------

def gather_trajectories(instance, n_ga, n_random):
    rules = instance["rules"]
    trajs = [rules[r] for r in USED_PDR if r in rules]
    trajs += list(instance.get("ga_pop", [])[:n_ga])
    trajs += list(instance.get("random", [])[:n_random])
    seen, unique = set(), []
    for a in trajs:
        key = tuple(int(x) for x in a)
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    return unique


# ---------------------------------------------------------------------------
# Per-candidate distribution statistics used for archetype selection
# ---------------------------------------------------------------------------

def candidate_stats(z_sorted):
    """Summaries of one (already ascending-sorted) 64-quantile return vector."""
    n = len(z_sorted)
    mean = float(z_sorted.mean())
    std = float(z_sorted.std())
    k = max(1, int(round(0.10 * n)))
    downside = mean - float(z_sorted[:k].mean())   # >=0; heavier left tail -> larger
    upside = float(z_sorted[-k:].mean())           # CVaR_90%: mean of the top-10% tail
    median = float(np.median(z_sorted))
    skew_left = median - mean                       # >0 -> left-skewed (mass pulled low)
    return mean, std, downside, upside, skew_left


def select_states(model, normalizer, data, device, n_instances, n_ga, n_random,
                  prog_lo, prog_hi, min_valid, min_spread, risky_min_spread,
                  risky_mean_gap, risky_upside_margin, normalize, verbose):
    """One pass over reachable offline-dataset states, returning the single mid-episode
    state from which ALL THREE archetypes are drawn:

    * *excellent* / *bad* - the best / worst candidate action by mean return (the state
      must offer a mean spread of at least ``min_spread``).
    * *risky* - a third candidate at the SAME state whose top-decile outcomes reach the
      excellent action's expected return (``CVaR_90% >= E[Z]_excellent +
      risky_upside_margin x spread``) but whose mean is clearly lower (at least
      ``risky_mean_gap`` x the state's excellent-vs-bad spread below the excellent
      E[Z], so the two panels cannot be near-identical), spanning at least
      ``risky_min_spread``; among the qualifiers, the heaviest-downside one wins
      (``(E[Z] - CVaR_10%) * (1 + 6 * max(0, median-mean))``).

    Among all states offering the three archetypes, the one with the largest
    excellent-vs-bad mean spread is returned.
    """
    kept = []            # states offering all three archetypes
    for ii in range(min(n_instances, len(data))):
        inst = data[ii]
        jl = np.array(inst["JobLength"])
        pt = np.array(inst["OpPT"])
        n_op = int(jl.sum())
        trajs = gather_trajectories(inst, n_ga, n_random)
        R = len(trajs)
        if R == 0:
            continue
        A = np.array(trajs, dtype=np.int64)
        env = FJSPEnvForSameOpNums(n_j=jl.shape[0], n_m=pt.shape[1], device=device,
                                   mask_actions=True)
        state = env.set_initial_data(np.tile(jl[None], (R, 1)),
                                     np.tile(pt[None], (R, 1, 1)))
        for t in range(n_op):
            prog = t / n_op
            if prog_lo <= prog <= prog_hi:
                fj = normalizer.fea_j(state.fea_j_tensor) if normalize else state.fea_j_tensor
                fm = normalizer.fea_m(state.fea_m_tensor) if normalize else state.fea_m_tensor
                fp = normalizer.fea_pairs(state.fea_pairs_tensor) if normalize else state.fea_pairs_tensor
                with torch.no_grad():
                    q_list, _ = model(fj, state.op_mask_tensor, state.candidate_tensor, fm,
                                      state.mch_mask_tensor, state.comp_idx_tensor,
                                      state.dynamic_pair_mask_tensor, fp)
                    q0, q1 = q_list[0], q_list[1]                 # [R, JM, N]
                    take_second = q0.mean(-1) > q1.mean(-1)
                    Zb = torch.where(take_second[..., None], q1, q0).cpu().numpy()
                for r in range(R):
                    Z = Zb[r]
                    means = Z.mean(-1)
                    valid = np.where(np.isfinite(means))[0]
                    if valid.size < min_valid:
                        continue
                    stats = np.array([candidate_stats(np.sort(Z[c])) for c in valid])
                    mean_v, std_v, down_v, up_v, skew_v = stats.T
                    spread = float(mean_v.max() - mean_v.min())
                    if spread < min_spread:
                        continue
                    exc_li = int(np.argmax(mean_v))
                    bad_li = int(np.argmin(mean_v))
                    # risky: a third candidate at the SAME state whose top-decile
                    # outcomes reach the excellent action's expected return while its
                    # mean sits clearly below the excellent mean, and that spans a
                    # real range; the heaviest-downside qualifier wins
                    risky_li, risky_score = None, -np.inf
                    for li, c in enumerate(valid):
                        if li in (exc_li, bad_li):
                            continue
                        z = np.sort(Z[c])
                        if z[-1] - z[0] < risky_min_spread:
                            continue
                        if up_v[li] < mean_v[exc_li] + risky_upside_margin * spread:
                            continue
                        if mean_v[li] > mean_v[exc_li] - risky_mean_gap * spread:
                            continue
                        score = down_v[li] * (1.0 + max(0.0, skew_v[li]) * 6.0)
                        if score > risky_score:
                            risky_li, risky_score = li, float(score)
                    if risky_li is None:
                        continue
                    kept.append(dict(inst=ii, t=t, r=r, prog=prog, Z=Z, valid=valid,
                                     spread=spread, exc=int(valid[exc_li]),
                                     bad=int(valid[bad_li]), risky=int(valid[risky_li]),
                                     risky_score=risky_score))
            state, _, _ = env.step(A[:, t])
        if verbose:
            print(f"[scan] instance {ii + 1}/{min(n_instances, len(data))} "
                  f"kept={len(kept)}", flush=True)

    if not kept:
        raise RuntimeError(
            "No state offered all three archetypes (excellent/bad mean spread plus a "
            "risky candidate whose CVaR_90% reaches the excellent action's mean while "
            "its own mean is clearly lower); relax --min_spread / --min_valid / "
            "--risky_min_spread / --risky_mean_gap / --risky_upside_margin or widen "
            "the progress band.")

    spreads = np.array([k["spread"] for k in kept])
    best = kept[int(np.argmax(spreads))]
    if verbose:
        print(f"[scan] {len(kept)} eligible states; archetypes from instance "
              f"{best['inst']} t={best['t']} prog={best['prog']:.2f} "
              f"spread={best['spread']:.3f} risky pair={best['risky']} "
              f"(score={best['risky_score']:.3f})")
    return best


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
    })


def _density(z_sorted, grid, bw):
    from scipy.stats import gaussian_kde
    return gaussian_kde(z_sorted, bw_method=bw)(grid)


def make_figure(panels, out_dir, bw=0.25, stem="return_distributions",
                suptitle=("Return distributions learned by the distributional (QR-DQN) "
                          "critic for FJSP state-action pairs"),
                show_panel_titles=True, show_stats=True):
    """``panels``: ordered list of dicts, each with keys
    key, label, color, z (sorted 64), backdrop (list of sorted-64 arrays), prov (str),
    mean, std, down, up.  ``stem`` names the output files (<stem>.pdf/.png).
    Pass ``suptitle=None`` / ``show_panel_titles=False`` / ``show_stats=False`` for a
    bare drop-in panel (density + fill + mean line + rug only, no text)."""
    import matplotlib.pyplot as plt
    _style()

    ncol = len(panels)
    fig = plt.figure(figsize=(2.35 * ncol + 0.4, 3.0))
    gs = fig.add_gridspec(1, ncol, wspace=0.20)

    for j, P in enumerate(panels):
        ax = fig.add_subplot(gs[0, j])
        z = P["z"]
        lo = min(b.min() for b in P["backdrop"] + [z])
        hi = max(b.max() for b in P["backdrop"] + [z])
        pad = 0.10 * (hi - lo + 1e-9)
        grid = np.linspace(lo - pad, hi + pad, 512)
        backdrop = [_density(b, grid, bw) for b in P["backdrop"]]
        dens = _density(z, grid, bw)
        ymax = max([d.max() for d in backdrop] + [dens.max()]) * 1.20
        for d in backdrop:
            ax.plot(grid, d, color="0.82", lw=0.6, zorder=1)
        ax.fill_between(grid, dens, color=P["color"], alpha=0.28, zorder=2)
        ax.plot(grid, dens, color=P["color"], lw=1.9, zorder=3)
        ax.plot(z, np.full_like(z, -0.05 * ymax), "|", color=P["color"], ms=5,
                mew=0.7, zorder=4)
        ax.axvline(P["mean"], color=P["color"], ls="--", lw=1.0, zorder=3)
        ax.set_ylim(-0.10 * ymax, ymax)
        ax.set_xlim(grid[0], grid[-1])
        if show_panel_titles:
            ax.set_title(P["label"], color=P["color"], pad=6)
        if show_stats:
            txt = "\n".join([
                r"$\mathbb{E}[Z]=$" + f"{P['mean']:.2f}",
                r"$\mathrm{std}=$" + f"{P['std']:.2f}",
                r"$\mathrm{CVaR}_{10\%}=$" + f"{P['mean'] - P['down']:.2f}",
                r"$\mathrm{CVaR}_{90\%}=$" + f"{P['up']:.2f}",
            ])
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.6, color="0.15",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6))
        # densities are self-scaled per panel; hide the (non-comparable) y numbers off panel 1
        ax.set_ylabel("Density" if j == 0 else "")
        if j != 0:
            ax.set_yticklabels([])
        ax.set_xlabel(r"Return $Z(s,a)$")

    if suptitle:
        fig.suptitle(suptitle, y=1.06, fontsize=10.5)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=500, transparent=True)
        print(f"Wrote {path}")
    plt.close(fig)


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
    p.add_argument("--checkpoint",
                   default="Quantile_critic/Check-fe9af749_seed_1/latest_check.pt")
    p.add_argument("--data", default="dataset/SD1_train_10_5_1000.npy",
                   help="Training dataset .npy (states reached by replaying its trajectories)")
    p.add_argument("--n_instances", type=int, default=500)
    p.add_argument("--n_ga", type=int, default=200, help="GA-pop trajectories per instance")
    p.add_argument("--n_random", type=int, default=100, help="Random trajectories per instance")
    p.add_argument("--prog_lo", type=float, default=0.25,
                   help="Lower episode-progress bound for candidate states")
    p.add_argument("--prog_hi", type=float, default=0.70,
                   help="Upper episode-progress bound for candidate states")
    p.add_argument("--min_valid", type=int, default=8,
                   help="Minimum valid candidate actions a state must offer")
    p.add_argument("--min_spread", type=float, default=0.50,
                   help="Minimum best-vs-worst mean-return spread a state must offer")
    p.add_argument("--risky_min_spread", type=float, default=0.30,
                   help="Minimum quantile spread for the risky candidate")
    p.add_argument("--risky_mean_gap", type=float, default=0.15,
                   help="Minimum E[Z] gap between the excellent and risky candidates, "
                        "as a fraction of the state's excellent-vs-bad mean spread")
    p.add_argument("--risky_upside_margin", type=float, default=0.0,
                   help="Extra headroom for the risky upside condition: its CVaR90 must "
                        "reach E[Z]_excellent + margin x spread")
    p.add_argument("--bw", type=float, default=0.25, help="KDE bandwidth factor for density panels")
    p.add_argument("--device", default=None, help="cuda|cpu (default: auto)")
    p.add_argument("--out_dir", default="quantile_dist_figures")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_argparser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, device)
    print(f"Device: {device} | checkpoint: {args.checkpoint}")

    model, normalizer = load_critic(args.checkpoint, device)
    print(f"Loaded QRDQNNet (d={model.embedding_output_dim}, "
          f"num_quantiles={model.num_quantiles}, n_critics={len(model.Q)})")

    data = np.load(args.data, allow_pickle=True)
    best = select_states(
        model, normalizer, data, device,
        n_instances=args.n_instances, n_ga=args.n_ga, n_random=args.n_random,
        prog_lo=args.prog_lo, prog_hi=args.prog_hi, min_valid=args.min_valid,
        min_spread=args.min_spread, risky_min_spread=args.risky_min_spread,
        risky_mean_gap=args.risky_mean_gap,
        risky_upside_margin=args.risky_upside_margin, normalize=True, verbose=True)

    # all three archetypes = candidate actions at the SAME state (within-state contrast)
    Zb, vb = best["Z"], best["valid"]
    exc_c, bad_c, risky_c = best["exc"], best["bad"], best["risky"]
    backdrop = [np.sort(Zb[c]) for c in vb]
    n_op = int(np.array(data[best["inst"]]["JobLength"]).sum())
    prov = f"instance {best['inst']}, step {best['t']}/{n_op}"

    def panel(key, label, color, cand):
        z = np.sort(Zb[cand])
        mean, std, down, up, _ = candidate_stats(z)
        return dict(key=key, label=label, color=color, z=z, backdrop=backdrop,
                    prov=prov, mean=mean, std=std, down=down, up=up, cand=int(cand))

    P_exc = panel("excellent", "Excellent return", "#009E73", exc_c)
    P_bad = panel("bad", "Low return", "#D55E00", bad_c)
    P_risky = panel("risky", "High-variance (risky)", "#CC79A7", risky_c)
    panels = [P_exc, P_bad, P_risky]

    print(f"\nSelected archetype state-action pairs ({prov}):")
    for P in panels:
        print(f"  {P['key']:10s} pair={P['cand']:3d}  E[Z]={P['mean']:+.3f}  "
              f"std={P['std']:.3f}  CVaR10%={P['mean'] - P['down']:+.3f}  "
              f"CVaR90%={P['up']:+.3f}")

    make_figure(panels, out_dir=args.out_dir, bw=args.bw)
    # reviewer-facing variant without the risky archetype: just the good/bad contrast
    make_figure([P_exc, P_bad], out_dir=args.out_dir, bw=args.bw,
                stem="return_distributions_good_bad")
    # excellent panel alone, no title/stats text (bare drop-in paper figure element)
    make_figure([P_exc], out_dir=args.out_dir, bw=args.bw,
                stem="return_distribution_excellent",
                suptitle=None, show_panel_titles=False, show_stats=False)

    # persist the raw selected quantiles + metadata for reproducibility / re-plotting
    os.makedirs(args.out_dir, exist_ok=True)
    npz = os.path.join(args.out_dir, "return_distributions.npz")
    np.savez(npz,
             taus=(np.arange(Zb.shape[1]) + 0.5) / Zb.shape[1],
             Z_excellent=P_exc["z"], Z_bad=P_bad["z"], Z_risky=P_risky["z"],
             backdrop=np.stack(backdrop),
             meta=np.array([best["inst"], best["t"], n_op, len(vb)]),
             pairs=np.array([exc_c, bad_c, risky_c]))
    print(f"Wrote {npz}")


if __name__ == "__main__":
    main()
