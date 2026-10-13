"""Small synthetic datasets in the legacy .npy layout, for the cdqac.data tests.

Test files import this by file location (``sys.path.insert(0, <tests dir>)``) because a
stray ``tests`` package in some environments shadows ``from tests import ...``.
"""
import os

import numpy as np
import torch

from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums

# The rule names of the released legacy FJSP files, in their stored order (the datasets drop
# the _masked suffix).
LEGACY_RULE_ORDER = [f"{j}_{m}_masked" for j in ("MWR", "LWR", "LOPNR", "MOPNR")
                     for m in ("LCT", "MCT", "EST", "LST", "FAM", "LPT", "SPT")]
UNMASKED_RULE = "LWR_EST_masked"  # flagged masked=False in the fixture, so it must be dropped

JOB_LENGTH = [2, 2, 1]
N_M = 2


def random_instance(rng, job_length, n_m=N_M):
    n_ops = int(sum(job_length))
    pt = rng.randint(1, 10, size=(n_ops, n_m))
    allowed = rng.rand(n_ops, n_m) < 0.6
    allowed[np.arange(n_ops), rng.randint(0, n_m, size=n_ops)] = True
    return [int(j) for j in job_length], (pt * allowed).tolist()


def rollouts(job_length, op_pt, k, seed):
    """``k`` uniformly random feasible trajectories and their makespans."""
    torch.manual_seed(seed)
    jl, pt = np.array(job_length), np.array(op_pt)
    env = FJSPEnvForSameOpNums(n_j=len(jl), n_m=pt.shape[1], device="cpu", mask_actions=True)
    state = env.set_initial_data(np.tile(jl[None], (k, 1)), np.tile(pt[None], (k, 1, 1)))
    actions = []
    for _ in range(pt.shape[0]):
        mask = ~state.dynamic_pair_mask_tensor.flatten(1)
        a = torch.multinomial(mask.float(), 1).squeeze(-1).numpy()
        state, _, _ = env.step(a)
        actions.append(a)
    return np.stack(actions, 1).tolist(), env.current_makespan.astype(int).tolist()


def make_legacy_instances(n_instances=4, eps_cover=2, seed=0, unmasked_rule=UNMASKED_RULE):
    rng = np.random.RandomState(seed)
    data = []
    for i in range(n_instances):
        jl, pt = random_instance(rng, JOB_LENGTH)
        inst = {"JobLength": jl, "OpPT": pt, "rules": {}, "rules_info": {}}
        acts, spans = rollouts(jl, pt, len(LEGACY_RULE_ORDER), seed=100 + i)
        for r, a, s in zip(LEGACY_RULE_ORDER, acts, spans):
            inst["rules"][r] = a
            inst["rules_info"][r] = {"makespan": s, "masked": r != unmasked_rule}
        inst["random"], inst["random_makespan"] = rollouts(jl, pt, 6, seed=200 + i)
        hof, hof_ms = rollouts(jl, pt, 4, seed=300 + i)
        order = np.argsort(hof_ms, kind="stable")
        inst["ga"] = [hof[j] for j in order]
        inst["ga_makespan"] = [float(hof_ms[j]) for j in order]
        pop, pop_ms = rollouts(jl, pt, 4, seed=400 + i)
        inst["ga_pop"] = pop + pop[:2] + pop[:1]  # duplicates, like the real GA populations
        inst["ga_pop_makespan"] = [float(m) for m in pop_ms + pop_ms[:2] + pop_ms[:1]]
        if i < eps_cover:
            inst["eps_random_0.1"], _ = rollouts(jl, pt, 3, seed=500 + i)
        data.append(inst)
    return data


def make_legacy_val(seed=1):
    rng = np.random.RandomState(seed)
    return [dict(zip(("JobLength", "OpPT"), random_instance(rng, jl)))
            for jl in ([2, 2, 1], [2, 1, 1], [3, 2, 1])]


def write_legacy_files(folder, train_name="train_3_2.npy", val_name="val_3_2.npy", **kw):
    """Write a legacy train/val pair into ``folder``; returns (train_path, val_path)."""
    os.makedirs(folder, exist_ok=True)
    train_path, val_path = os.path.join(folder, train_name), os.path.join(folder, val_name)
    np.save(train_path, make_legacy_instances(**kw), allow_pickle=True)
    np.save(val_path, make_legacy_val(), allow_pickle=True)
    return train_path, val_path


def original_train_selection(train_instances, config):
    """train.py's trajectory selection before cdqac.data existed (verbatim, minus prints),
    with the PDR rules restricted to PDR_RULES as both loading paths now do."""
    import random
    from cdqac.data.format import PDR_RULES
    from cdqac.utils import remove_duplicate_actions

    def weighted_choice(items, scores, amount):
        weights = [1 / score for score in scores]
        return random.choices(items, weights=weights, k=amount)

    job_lenght_list, opt_list, action_list_list = [], [], []
    eps_random_ver = f"eps_random_{getattr(config, 'eps_pdr_version', '0.1')}"
    use_eps_pdr = getattr(config, "use_eps_pdr", False)
    if config.num_instances is not None:
        train_instances = train_instances[:config.num_instances]
    for instance in train_instances:
        curr_action_list = []
        if config.use_dispatching:
            for rule in instance["rules"]:
                if rule.removesuffix("_masked") not in PDR_RULES:
                    continue
                is_masked = instance["rules_info"][rule]["masked"]
                if config.use_mask and not is_masked:
                    continue
                curr_action_list.append(instance["rules"][rule])
        if config.use_random and "random" in instance:
            if config.n_random is None:
                curr_action_list += instance["random"]
            else:
                curr_action_list += instance["random"][:config.n_random]
        job_lenght_list.append(instance["JobLength"])
        opt_list.append(instance["OpPT"])
        if "ga" in instance and config.use_ga_hof:
            if config.n_ga_hof is None:
                curr_action_list += instance["ga"]
            else:
                if config.n_ga_hof > len(instance["ga"]):
                    curr_action_list += instance["ga"]
                else:
                    curr_action_list += instance["ga"][:config.n_ga_hof]
        if "ga_pop" in instance and config.use_ga_pop:
            if config.n_ga_pop is None:
                curr_action_list += instance["ga_pop"]
            else:
                if config.n_ga_pop > len(instance["ga_pop"]):
                    curr_action_list += instance["ga_pop"]
                else:
                    if config.ga_pop_random:
                        makespans = instance['ga_pop_makespan']
                        curr_action_list += weighted_choice(instance["ga_pop"], makespans,
                                                            config.n_ga_pop)
                    else:
                        curr_action_list += instance["ga_pop"][:config.n_ga_pop]
        if eps_random_ver in instance and use_eps_pdr:
            if config.n_eps_pdr is None:
                curr_action_list += instance[eps_random_ver]
            else:
                if config.n_eps_pdr > len(instance[eps_random_ver]):
                    curr_action_list += instance[eps_random_ver]
                else:
                    curr_action_list += instance[eps_random_ver][:config.n_eps_pdr]
        if config.remove_duplicate:
            curr_action_list, _ = remove_duplicate_actions(curr_action_list)
        action_list_list.append(curr_action_list)
    return job_lenght_list, opt_list, action_list_list


def selection_config(**overrides):
    """A namespace with every option select_action_lists reads, all sources off."""
    from types import SimpleNamespace
    cfg = dict(num_instances=None, remove_duplicate=False, use_mask=True,
               use_dispatching=False, use_random=False, n_random=None,
               use_ga_hof=False, n_ga_hof=None, ga_hof_random=False,
               use_ga_pop=False, n_ga_pop=None, ga_pop_random=False,
               use_eps_pdr=False, eps_pdr_version="0.1", n_eps_pdr=None)
    cfg.update(overrides)
    return SimpleNamespace(**cfg)
