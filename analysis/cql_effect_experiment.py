"""
What does the CQL(alpha=0.05) regularizer actually do to the learned return quantiles?

Companion to ``quantile_distribution_experiment.py``.  That script *portrays* the return
distribution the trained QR-DQN critic assigns to three archetype actions (excellent /
low / high-variance) at one offline-dataset state.  This one asks the follow-up question:
**how much of that shape is the conservatism term, and which way does it push?**

The regularizer
---------------
The Quantile_critic checkpoints were trained with ``use_cql: true`` and
``cql_alpha_offline: 0.05``.  The term is the standard CQL(H) penalty
(``cdqac/methods/td3_qrdqn.py::calculate_qrdqn_loss``), applied per critic:

    L_CQL(s) = alpha * ( logsumexp_a' Q(s,a')  -  Q(s, a_data) ),
    with  Q(s,a) = (1/N) * sum_tau Z_tau(s,a)          # the MEAN of the 64 quantiles

so it pushes *down* the value of every candidate action (weighted by its softmax share)
and pulls *up* the value of the action the offline dataset actually took at ``s``.

The analytic consequence is worth stating, because it is exactly derivable:

    dL_CQL / dZ_tau(s,a) = (alpha / N) * ( softmax(Q)(a) - 1{a = a_data} )

which is *independent of tau*.  In the tabular/free-quantile limit CQL is therefore a pure
**translation** of the return distribution - it shifts Z up or down without reshaping it.
Through a shared function approximator the realized update need not stay exactly uniform,
so the script *measures* the deviation rather than assuming it (see ``--out_dir``'s
cql_quantile_shift figure and the printed "uniformity" diagnostics).

How the effect is shown (no alpha=0 checkpoint exists to ablate against)
-----------------------------------------------------------------------
Both available checkpoints already train with alpha=0.05, so there is nothing to diff.
Instead we isolate the term and let it act on its own:

  1. Select the SAME archetype state + excellent/low/risky candidate actions as
     ``quantile_distribution_experiment.py`` (identical defaults & seed -> identical state).
  2. Read the three quantile vectors off the trained critic  -> **before**.
  3. Deep-copy the critic and take ``--cql_steps`` Adam steps on the CQL term *alone*
     (no TD/Bellman loss), over real offline-dataset (state, dataset-action) batches.
  4. Replay back to the same state and re-read the same three candidates -> **after**.

The before/after gap is the force CQL exerts, in return units.  Magnitude is a function of
``--cql_steps`` x ``--cql_lr`` (it is a force, not a fixed point - with the TD loss removed
nothing balances it), so read the figure for *direction and relative magnitude across
actions*, not for an absolute number of makespan units.  The over-valued candidate falls
hardest, the dataset action is held up: that asymmetry is the whole point of CQL.

Run (defaults mirror quantile_distribution_experiment.py, so the state matches):

    python cql_effect_experiment.py

Outputs (into ``--out_dir``, default ``cql_effect_figures/``):
    cql_effect.pdf / .png          - before/after return distributions, the 3 archetypes
    cql_effect_bare.pdf / .png     - same, no titles/annotations (drop-in figure element)
    cql_quantile_shift.pdf / .png  - Delta Z_tau vs tau: is the push a pure translation?
    cql_effect.npz                 - before/after quantiles, pushes and metadata
"""

import argparse
import copy
import os

import numpy as np
import torch

from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums
from quantile_distribution_experiment import (
    _density,
    _style,
    candidate_quantiles,
    candidate_stats,
    gather_trajectories,
    load_critic,
    select_states,
    set_seed,
)


# ---------------------------------------------------------------------------
# Reaching a specific dataset state, and the CQL term itself
# ---------------------------------------------------------------------------

def replay_to_state(inst, traj, t, device):
    """Single-env replay of ``traj`` for ``t`` steps -> the state at step ``t``.

    Used for both the before- and after-critic reads so the two forward passes see
    identical tensor shapes (batched vs single-env forwards differ in the last ulps).
    """
    jl = np.array(inst["JobLength"])
    pt = np.array(inst["OpPT"])
    env = FJSPEnvForSameOpNums(n_j=jl.shape[0], n_m=pt.shape[1], device=device,
                               mask_actions=True)
    state = env.set_initial_data(jl[None], pt[None])
    for k in range(t):
        state, _, _ = env.step(np.array([traj[k]], dtype=np.int64))
    return state


def _forward(model, normalizer, state, normalize=True):
    fj = normalizer.fea_j(state.fea_j_tensor) if normalize else state.fea_j_tensor
    fm = normalizer.fea_m(state.fea_m_tensor) if normalize else state.fea_m_tensor
    fp = normalizer.fea_pairs(state.fea_pairs_tensor) if normalize else state.fea_pairs_tensor
    q_list, _ = model(fj, state.op_mask_tensor, state.candidate_tensor, fm,
                      state.mch_mask_tensor, state.comp_idx_tensor,
                      state.dynamic_pair_mask_tensor, fp)
    return q_list


def cql_term(model, normalizer, state, actions, alpha, cql_temp=1.0, normalize=True):
    """The CQL(H) penalty exactly as trained (td3_qrdqn.calculate_qrdqn_loss), summed
    over the ``n_critics`` heads:

        alpha * ( logsumexp_a'( Q(s,a')/temp ) * temp  -  Q(s, a_data) ),  Q = mean_tau Z

    Masked (invalid) candidates carry ``-inf`` quantiles, so they drop out of the
    logsumexp with zero softmax weight - the same masking the trainer relied on.
    """
    q_list = _forward(model, normalizer, state, normalize)
    total = 0.0
    for q in q_list:                                    # each [B, J*M, N]
        qa = q.gather(1, actions.view(-1, 1, 1).expand(-1, 1, q.shape[-1])).squeeze(1)
        Q = q.mean(-1)                                  # [B, J*M]  = E[Z] per candidate
        Qa = qa.mean(-1)                                # [B]       = E[Z] of dataset action
        total = total + alpha * (
            torch.logsumexp(Q / cql_temp, dim=1) * cql_temp - Qa).mean()
    return total


def run_cql_steps(model, normalizer, data, device, alpha, lr, n_steps, batch_size,
                  n_instances, n_ga, n_random, max_grad_norm, seed, verbose,
                  force_instance=None):
    """Take ``n_steps`` Adam steps on the CQL term ALONE over offline-dataset batches.

    Each batch is a set of dataset trajectories replayed to a common timestep, so the
    (state, dataset-action) pairs are exactly the ones the offline learner was trained on.

    ``force_instance`` is visited first, guaranteeing the instance the archetype state
    lives in actually receives gradient steps.  Without it the measured before/after gap
    at that state would be pure generalization from other instances, which is a much
    weaker (and easily misread) statement about what the penalty does.

    Returns the per-step loss trace.
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    losses = []
    order = [int(i) for i in rng.permutation(min(n_instances, len(data)))]
    if force_instance is not None:
        order = [int(force_instance)] + [i for i in order if i != int(force_instance)]
    for ii in order:
        inst = data[int(ii)]
        trajs = gather_trajectories(inst, n_ga, n_random)
        if not trajs:
            continue
        A = np.array(trajs, dtype=np.int64)
        rows = rng.choice(A.shape[0], size=min(batch_size, A.shape[0]), replace=False)
        A = A[rows]
        B = A.shape[0]
        jl = np.array(inst["JobLength"])
        pt = np.array(inst["OpPT"])
        n_op = int(jl.sum())
        env = FJSPEnvForSameOpNums(n_j=jl.shape[0], n_m=pt.shape[1], device=device,
                                   mask_actions=True)
        state = env.set_initial_data(np.tile(jl[None], (B, 1)),
                                     np.tile(pt[None], (B, 1, 1)))
        for t in range(n_op):
            acts = torch.as_tensor(A[:, t], device=device, dtype=torch.long)
            loss = cql_term(model, normalizer, state, acts, alpha)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            losses.append(float(loss.detach()))
            if len(losses) >= n_steps:
                if verbose:
                    print(f"[cql] step {len(losses)}/{n_steps} loss={losses[-1]:+.4f}")
                return losses
            if verbose and len(losses) % 25 == 0:
                print(f"[cql] step {len(losses)}/{n_steps} loss={losses[-1]:+.4f}",
                      flush=True)
            state, _, _ = env.step(A[:, t])
    return losses


def cql_push(Z, valid, a_data, alpha):
    """Analytic per-candidate CQL pressure at one state, in the free-quantile limit:

        dL/dZ_tau(a) * N  =  alpha * ( softmax(Q)(a) - 1{a = a_data} )

    (independent of tau).  Negative => gradient descent RAISES that candidate's return;
    positive => it LOWERS it.  Returns dict candidate -> push, plus the softmax weights.
    """
    Q = np.array([Z[c].mean() for c in valid])
    w = np.exp(Q - Q.max())
    w = w / w.sum()
    push, weight = {}, {}
    for li, c in enumerate(valid):
        g = alpha * (w[li] - (1.0 if int(c) == int(a_data) else 0.0))
        push[int(c)] = float(g)
        weight[int(c)] = float(w[li])
    return push, weight


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_cql_figure(panels, out_dir, bw=0.25, stem="cql_effect",
                    suptitle=("Effect of the CQL conservatism term "
                              r"($\alpha=0.05$) on the learned return distributions"),
                    show_panel_titles=True, show_stats=True):
    """Before/after return densities for each archetype.

    ``panels``: dicts with key, label, color, z_before, z_after (sorted 64-vectors),
    mean_before, mean_after, std_before, std_after, is_data_action.
    """
    import matplotlib.pyplot as plt
    _style()

    ncol = len(panels)
    fig = plt.figure(figsize=(2.35 * ncol + 0.4, 3.0))
    gs = fig.add_gridspec(1, ncol, wspace=0.12)

    # All archetypes are candidates at the SAME state, so return is directly comparable:
    # share one x-grid (and one density scale) across panels, or the eye would read a
    # small shift on a zoomed panel as being as large as a big one on a wide panel.
    lo = min(min(P["z_before"].min(), P["z_after"].min()) for P in panels)
    hi = max(max(P["z_before"].max(), P["z_after"].max()) for P in panels)
    pad = 0.08 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, 512)
    dens = [(_density(P["z_before"], grid, bw), _density(P["z_after"], grid, bw))
            for P in panels]
    ymax = max(max(db.max(), da.max()) for db, da in dens) * 1.28

    for j, (P, (db, da)) in enumerate(zip(panels, dens)):
        ax = fig.add_subplot(gs[0, j])
        ax.plot(grid, db, color="0.55", lw=1.4, ls="--", zorder=2,
                label="before (trained critic)")
        ax.fill_between(grid, da, color=P["color"], alpha=0.28, zorder=3)
        ax.plot(grid, da, color=P["color"], lw=1.9, zorder=4,
                label="after CQL-only steps")
        ax.axvline(P["mean_before"], color="0.55", ls=":", lw=1.0, zorder=2)
        ax.axvline(P["mean_after"], color=P["color"], ls="--", lw=1.0, zorder=4)
        # arrow: which way, and how far, CQL moved the expected return
        ax.annotate("", xy=(P["mean_after"], 0.72 * ymax),
                    xytext=(P["mean_before"], 0.72 * ymax),
                    arrowprops=dict(arrowstyle="-|>", color=P["color"], lw=1.3,
                                    mutation_scale=11, shrinkA=0, shrinkB=0), zorder=5)
        ax.set_ylim(0, ymax)
        ax.set_xlim(grid[0], grid[-1])
        if show_panel_titles:
            title = P["label"] + ("\n(dataset action)" if P["is_data_action"] else "")
            ax.set_title(title, color=P["color"], pad=6, fontsize=9.5)
        if show_stats:
            d_mean = P["mean_after"] - P["mean_before"]
            txt = "\n".join([
                r"$\Delta\mathbb{E}[Z]=$" + f"{d_mean:+.3f}",
                r"$\Delta\mathrm{std}=$" + f"{P['std_after'] - P['std_before']:+.3f}",
            ])
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", ha="left",
                    fontsize=7.6, color="0.15",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.8", lw=0.6))
        ax.set_ylabel("Density" if j == 0 else "")
        if j != 0:
            ax.set_yticklabels([])
        ax.set_xlabel(r"Return $Z(s,a)$")

    if show_panel_titles:
        # figure-level legend: one grey "before" key, one neutral "after" key, placed
        # under the axes so it cannot collide with the per-panel stats boxes
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], color="0.55", lw=1.4, ls="--",
                          label="before (trained critic)"),
                   Line2D([], [], color="0.25", lw=1.9,
                          label=r"after CQL-only steps ($\alpha=0.05$)")]
        fig.legend(handles=handles, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.14), fontsize=8.5)

    if suptitle:
        fig.suptitle(suptitle, y=1.06, fontsize=10.5)

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=500, transparent=True)
        print(f"Wrote {path}")
    plt.close(fig)


def make_shift_figure(panels, out_dir, stem="cql_quantile_shift"):
    """Delta Z_tau vs tau, one line per archetype.

    A flat line == CQL acted as a pure translation of the return distribution (what the
    analytic gradient predicts); slope/curvature == the shared network reshaped it too.
    """
    import matplotlib.pyplot as plt
    _style()

    fig, ax = plt.subplots(figsize=(3.8, 3.0))
    for P in panels:
        d = P["z_after"] - P["z_before"]
        taus = (np.arange(len(d)) + 0.5) / len(d)
        ax.plot(taus, d, color=P["color"], lw=1.8, label=P["label"])
    ax.axhline(0.0, color="0.6", lw=0.8, ls="--", zorder=1)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"Quantile fraction $\tau$")
    ax.set_ylabel(r"$\Delta Z_\tau$  (after $-$ before)")
    ax.legend(loc="best")

    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=500, transparent=True)
        print(f"Wrote {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",
                   default="Quantile_critic/Check-fe9af749_seed_1/latest_check.pt")
    p.add_argument("--data", default="dataset/SD1_train_10_5_1000.npy")
    # --- state selection: keep these identical to quantile_distribution_experiment.py
    #     so the SAME state / archetypes are picked and the two figures line up ---
    p.add_argument("--n_instances", type=int, default=500)
    p.add_argument("--n_ga", type=int, default=200)
    p.add_argument("--n_random", type=int, default=100)
    p.add_argument("--prog_lo", type=float, default=0.25)
    p.add_argument("--prog_hi", type=float, default=0.70)
    p.add_argument("--min_valid", type=int, default=8)
    p.add_argument("--min_spread", type=float, default=0.50)
    p.add_argument("--risky_min_spread", type=float, default=0.30)
    p.add_argument("--risky_mean_gap", type=float, default=0.15)
    p.add_argument("--risky_upside_margin", type=float, default=0.0)
    # --- the CQL-only optimisation ---
    p.add_argument("--cql_alpha", type=float, default=0.05,
                   help="Conservatism weight (matches cql_alpha_offline in config.yml)")
    p.add_argument("--cql_steps", type=int, default=300,
                   help="Adam steps on the CQL term alone")
    p.add_argument("--cql_lr", type=float, default=3e-4,
                   help="Learning rate (matches q_lr in config.yml)")
    p.add_argument("--cql_batch", type=int, default=64,
                   help="Trajectories (= states) per CQL gradient step")
    p.add_argument("--cql_instances", type=int, default=50,
                   help="Instances to draw the CQL gradient-step batches from")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bw", type=float, default=0.25)
    p.add_argument("--device", default=None)
    p.add_argument("--out_dir", default="cql_effect_figures")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_argparser().parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed, device)
    print(f"Device: {device} | checkpoint: {args.checkpoint} | alpha={args.cql_alpha}")

    model, normalizer = load_critic(args.checkpoint, device)
    data = np.load(args.data, allow_pickle=True)

    # 1. the same archetype state as quantile_distribution_experiment.py
    best = select_states(
        model, normalizer, data, device,
        n_instances=args.n_instances, n_ga=args.n_ga, n_random=args.n_random,
        prog_lo=args.prog_lo, prog_hi=args.prog_hi, min_valid=args.min_valid,
        min_spread=args.min_spread, risky_min_spread=args.risky_min_spread,
        risky_mean_gap=args.risky_mean_gap,
        risky_upside_margin=args.risky_upside_margin, normalize=True, verbose=True)

    inst = data[best["inst"]]
    traj = gather_trajectories(inst, args.n_ga, args.n_random)[best["r"]]
    t = best["t"]
    a_data = int(traj[t])          # the action the offline dataset took at this state
    valid = best["valid"]
    exc_c, bad_c, risky_c = best["exc"], best["bad"], best["risky"]

    # 2. BEFORE: read the trained critic at that state (single-env replay)
    state = replay_to_state(inst, traj, t, device)
    Z_before = candidate_quantiles(model, normalizer, state, device)

    push, weight = cql_push(Z_before, valid, a_data, args.cql_alpha)
    print(f"\nState: instance {best['inst']}, step {t}, trajectory {best['r']} | "
          f"dataset action = pair {a_data} | {len(valid)} valid candidates")

    # 3. CQL-only gradient steps on a copy of the critic
    model_cql = copy.deepcopy(model)
    losses = run_cql_steps(
        model_cql, normalizer, data, device, alpha=args.cql_alpha, lr=args.cql_lr,
        n_steps=args.cql_steps, batch_size=args.cql_batch,
        n_instances=args.cql_instances, n_ga=args.n_ga, n_random=args.n_random,
        max_grad_norm=args.max_grad_norm, seed=args.seed, verbose=True,
        force_instance=best["inst"])
    print(f"[cql] took {len(losses)} steps | loss {losses[0]:+.4f} -> {losses[-1]:+.4f}")

    # 4. AFTER: replay to the identical state, read the updated critic
    state = replay_to_state(inst, traj, t, device)
    Z_after = candidate_quantiles(model_cql, normalizer, state, device)

    def panel(key, label, color, cand):
        zb, za = np.sort(Z_before[cand]), np.sort(Z_after[cand])
        mb, sb, *_ = candidate_stats(zb)
        ma, sa, *_ = candidate_stats(za)
        return dict(key=key, label=label, color=color, cand=int(cand),
                    z_before=zb, z_after=za, mean_before=mb, mean_after=ma,
                    std_before=sb, std_after=sa,
                    is_data_action=(int(cand) == a_data),
                    push=push[int(cand)], weight=weight[int(cand)])

    panels = [panel("excellent", "Excellent return", "#009E73", exc_c),
              panel("bad", "Low return", "#D55E00", bad_c),
              panel("risky", "High-variance (risky)", "#CC79A7", risky_c)]

    print("\nCQL(alpha=%.3g) effect on the three archetypes:" % args.cql_alpha)
    print(f"  {'archetype':<11} {'pair':>4} {'softmax':>8} {'push':>8} "
          f"{'E[Z] before':>12} {'E[Z] after':>11} {'dE[Z]':>8} {'dstd':>8} {'uniform':>8}")
    for P in panels:
        d = P["z_after"] - P["z_before"]
        # how close was the realized change to a pure translation?
        nonuniform = float(np.abs(d - d.mean()).max())
        print(f"  {P['key']:<11} {P['cand']:>4} {P['weight']:>8.3f} {P['push']:>+8.4f} "
              f"{P['mean_before']:>+12.3f} {P['mean_after']:>+11.3f} "
              f"{P['mean_after'] - P['mean_before']:>+8.3f} "
              f"{P['std_after'] - P['std_before']:>+8.3f} {nonuniform:>8.3f}"
              + ("   <- dataset action" if P["is_data_action"] else ""))
    print("  (push > 0 => CQL lowers this action's return; push < 0 => it raises it.")
    print("   'uniform' = max_tau |dZ_tau - mean(dZ)|; ~0 means a pure translation.)")

    # 5. figures
    make_cql_figure(panels, out_dir=args.out_dir, bw=args.bw)
    make_cql_figure(panels, out_dir=args.out_dir, bw=args.bw, stem="cql_effect_bare",
                    suptitle=None, show_panel_titles=False, show_stats=False)
    make_shift_figure(panels, out_dir=args.out_dir)

    os.makedirs(args.out_dir, exist_ok=True)
    npz = os.path.join(args.out_dir, "cql_effect.npz")
    np.savez(npz,
             taus=(np.arange(Z_before.shape[1]) + 0.5) / Z_before.shape[1],
             Z_before=np.stack([P["z_before"] for P in panels]),
             Z_after=np.stack([P["z_after"] for P in panels]),
             pairs=np.array([P["cand"] for P in panels]),
             pushes=np.array([P["push"] for P in panels]),
             weights=np.array([P["weight"] for P in panels]),
             losses=np.array(losses),
             meta=np.array([best["inst"], t, best["r"], a_data, args.cql_alpha,
                            args.cql_steps, args.cql_lr], dtype=object))
    print(f"Wrote {npz}")


if __name__ == "__main__":
    main()
