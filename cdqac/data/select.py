"""Choose which stored trajectories become training data (the ``use_*`` / ``n_*`` options)."""
import random
from typing import List, Tuple

import numpy as np

from cdqac.data import format as fmt
from cdqac.data.dataset import OfflineDataset
from cdqac.utils import remove_duplicate_actions


def weighted_choice(items, scores, amount):
    # Convert scores to probabilities (lower score -> higher probability)
    weights = [1 / score for score in scores]
    return random.choices(items, weights=weights, k=amount)


def enabled_sources(config) -> List[str]:
    """Source names switched on by a TrainConfig, in the order they are added per instance."""
    names = []
    if config.use_dispatching:
        names.append("pdr")
    if config.use_random:
        names.append("random")
    if config.use_ga_hof:
        names.append("ga_hof")
    if config.use_ga_pop:
        names.append("ga_pop")
    if getattr(config, "use_eps_pdr", False):
        names.append(fmt.eps_pdr_source(getattr(config, "eps_pdr_version", "0.1")))
    return names


def select_action_lists(data: OfflineDataset, config) -> Tuple[list, list, list]:
    """The training instances and, per instance, the trajectories to replay into the buffer.

    Per instance the enabled sources are added in a fixed order -- PDR, random, GA hall of
    fame, GA population, epsilon-PDR -- each cut to its ``n_*`` limit; duplicates are then
    removed if ``remove_duplicate`` is set.  The same seed picks the same trajectories as
    train.py did before this module existed.

    Returns ``(job_lengths, op_pts, action_lists)``, one entry per instance.
    """
    train = data.split("train").head(config.num_instances)
    n_inst = len(train)
    sources = enabled_sources(config)
    if not sources:
        raise ValueError("At least one of use_dispatching, use_ga_hof, use_ga_pop, "
                         "use_random, use_eps_pdr must be True")
    for name in sources:
        if not data.has_source(name):
            raise ValueError(f"dataset '{data.id}' has no '{name}' source (available: "
                             f"{', '.join(data.source_names) or 'none'}); disable it or pick "
                             f"another dataset")
        if not data.source_downloaded(name):
            raise ValueError(f"source '{name}' of dataset '{data.id}' is not downloaded; get it "
                             f"with: python download_datasets.py {data.id} --sources {name}")
        if data.coverage(name) < n_inst:
            raise ValueError(f"source '{name}' of dataset '{data.id}' covers only the first "
                             f"{data.coverage(name)} instances, but num_instances={n_inst}; "
                             f"lower --num_instances or disable the source")

    lists = {name: data.source(name).as_lists(n_inst) for name in sources}
    if config.use_ga_pop and config.ga_pop_random:
        ga_pop_makespans = data.source("ga_pop").makespan_lists(n_inst)
    eps_name = sources[-1] if getattr(config, "use_eps_pdr", False) else None

    job_lengths, op_pts = train.job_lengths(), train.op_pts()
    action_lists = []
    for i in range(n_inst):
        curr_action_list = []
        if config.use_dispatching:
            curr_action_list += lists["pdr"][i]
        if config.use_random:
            if config.n_random is None:
                curr_action_list += lists["random"][i]
            else:
                curr_action_list += lists["random"][i][:config.n_random]
        if config.use_ga_hof:
            ga = lists["ga_hof"][i]
            if config.n_ga_hof is None or config.n_ga_hof > len(ga):
                curr_action_list += ga
            elif config.ga_hof_random:
                idx = np.random.choice(len(ga), size=config.n_ga_hof, replace=False)
                curr_action_list += [ga[j] for j in idx]
            else:
                curr_action_list += ga[:config.n_ga_hof]
        if config.use_ga_pop:
            ga_pop = lists["ga_pop"][i]
            if config.n_ga_pop is None or config.n_ga_pop > len(ga_pop):
                curr_action_list += ga_pop
            elif config.ga_pop_random:
                curr_action_list += weighted_choice(ga_pop, ga_pop_makespans[i], config.n_ga_pop)
            else:
                curr_action_list += ga_pop[:config.n_ga_pop]
        if eps_name is not None:
            eps = lists[eps_name][i]
            if config.n_eps_pdr is None or config.n_eps_pdr > len(eps):
                curr_action_list += eps
            else:
                curr_action_list += eps[:config.n_eps_pdr]

        if config.remove_duplicate:
            orig_len = len(curr_action_list)
            curr_action_list, n_removed = remove_duplicate_actions(curr_action_list)
            print(f"Removed {n_removed} duplicate actions from {orig_len} actions")
        action_lists.append(curr_action_list)
    return job_lengths, op_pts, action_lists
