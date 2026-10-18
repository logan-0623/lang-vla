"""Rollouts that can start from any saved simulator state (E3 grasp-state branching, E4 instruction switching).

LIBERO's sim state is `env.get_sim_state()` (MuJoCo time/qpos/qvel flattened); `env.set_init_state(s)` restores it
and returns a fresh observation. That is not the whole state: robosuite grippers keep `current_action`, which moves
toward the commanded open/close by a small step every sim substep. A stable grasp has it saturated closed, while
a reset leaves it at 0 (half open), so a restore without it loosens the fingers for the first few steps.
And every hard reset re-samples the fixtures' placement (cabinet, stove, rack move by up to ~2 cm), which lives in
the model's `body_pos` / `body_quat`, not in qpos (scripts/diag_restore.py). `snapshot` / `restore` carry both. The OSC controller takes delta actions from the current pose, so it holds no
state to save; the policy's open-loop action queue is cleared at every (re)start, so a branch always begins with
a fresh query. Whether a restore reproduces a trajectory is measured at run time (`divergence`), never assumed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def settle(env, init_state, wait_steps: int = 10, seed: int | None = None):
    """Upstream episode start: restore an init state, then let objects settle with no-op actions.

    The reset re-samples fixture placement from numpy's global RNG, so without `seed` the same init state gives a
    different scene depending on how many resets ran before it in the process. Pass a seed per (task, init) to make
    source rollouts identical across runs with different designs."""
    if seed is not None:
        np.random.seed(seed)
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(wait_steps):
        obs, *_ = env.step(DUMMY_ACTION)
    return obs


def _grippers(env) -> list:
    return [r.gripper for r in env.env.robots]


def snapshot(env) -> dict:
    """Everything `restore` needs: MuJoCo state, fixture placement and each gripper's `current_action`."""
    m = env.env.sim.model
    return {"state": np.asarray(env.get_sim_state()).copy(),
            "body_pos": np.array(m.body_pos), "body_quat": np.array(m.body_quat),
            "gripper": np.stack([np.asarray(g.current_action, float).copy() for g in _grippers(env)])}


def restore(env, snap) -> dict:
    """Reset, put the fixtures back where they were, restore the MuJoCo state and the grippers'
    `current_action`; returns the observation."""
    env.reset()
    m = env.env.sim.model
    for f in ("body_pos", "body_quat"):
        if np.shape(snap[f]) != np.shape(getattr(m, f)):
            raise ValueError(f"{f}: saved {np.shape(snap[f])} vs model {np.shape(getattr(m, f))}")
        getattr(m, f)[:] = snap[f]
    obs = env.set_init_state(snap["state"])            # set_init_state runs mj_forward with the restored model
    for g, a in zip(_grippers(env), snap["gripper"]):
        g.current_action = np.array(a, float)
    return obs


def rollout(env, policy, instruction: str, obs, *, steps: int,
            on_step: Callable[[int, dict], bool] | None = None) -> dict:
    """Run `policy` for up to `steps` env steps from the current state.

    policy.act(obs, instruction) returns the env actions for one open-loop chunk. on_step(t, obs) is called after
    every env step; returning True stops the rollout. Exceptions are caught and reported, never swallowed into a
    plain failure (IMPLEMENTATION_PLAN §0.6).
    """
    actions, queue, t = [], [], 0
    if hasattr(policy, "begin"):            # stochastic policies: same noise sequence for every (re)start
        policy.begin()
    try:
        while t < steps:
            if not queue:
                queue = list(policy.act(obs, instruction))
            a = queue.pop(0)
            actions.append(np.asarray(a, float))
            obs, *_ = env.step(a.tolist() if hasattr(a, "tolist") else list(a))
            t += 1
            if on_step is not None and on_step(t, obs):
                break
        return {"steps": t, "actions": actions, "obs": obs, "error": None}
    except Exception as e:  # reported per episode and counted separately
        return {"steps": t, "actions": actions, "obs": obs, "error": f"{type(e).__name__}: {e}"}


def same_actions(a: list, b: list) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def divergence(a: list, b: list) -> dict:
    """Where two action sequences first differ and by how much (None: identical over the common length)."""
    n = min(len(a), len(b))
    diffs = [float(np.max(np.abs(np.asarray(x) - np.asarray(y)))) for x, y in zip(a[:n], b[:n])]
    first = next((i for i, d in enumerate(diffs) if d > 0), None)
    return {"len_a": len(a), "len_b": len(b), "first_diff_step": first, "max_abs_diff": max(diffs, default=0.0)}


def same_obs(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)
