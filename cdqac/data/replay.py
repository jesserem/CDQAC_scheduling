"""Replay stored trajectories through the environment to recover their makespans."""
import numpy as np


def replay_makespans(job_length, op_pt, actions, batch_cap: int = 512) -> np.ndarray:
    """Makespans of the trajectories ``actions`` [R, n_ops] on one instance.

    Uses the same environment train.py replays the training data with
    (FJSPEnvForSameOpNums, action mask on), stepping up to ``batch_cap`` trajectories at once.
    Raises ValueError if a trajectory takes an action the action mask forbids.
    """
    from env.fjsp_env_same_op_nums import FJSPEnvForSameOpNums

    job_length = np.asarray(job_length)
    op_pt = np.asarray(op_pt)
    actions = np.asarray(actions, dtype=np.int64)
    n_j, n_m = job_length.shape[0], op_pt.shape[1]
    spans = []
    for start in range(0, len(actions), batch_cap):
        batch = actions[start:start + batch_cap]
        r = len(batch)
        env = FJSPEnvForSameOpNums(n_j=n_j, n_m=n_m, device="cpu", mask_actions=True)
        state = env.set_initial_data(np.tile(job_length[None], (r, 1)),
                                     np.tile(op_pt[None], (r, 1, 1)))
        for t in range(batch.shape[1]):
            forbidden = state.dynamic_pair_mask_tensor.flatten(1).cpu().numpy()[np.arange(r), batch[:, t]]
            if forbidden.any():
                k = int(np.argmax(forbidden))
                raise ValueError(f"trajectory {start + k} takes action {int(batch[k, t])} at step "
                                 f"{t}, which the action mask forbids")
            state, _, _ = env.step(batch[:, t])
        spans.append(np.asarray(env.current_makespan, dtype=np.float64))
    return np.concatenate(spans) if spans else np.zeros(0)

