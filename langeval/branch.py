"""Rollouts that can start from any saved simulator state (E3 grasp-state branching, E4 instruction switching).

LIBERO's full state is `env.get_sim_state()` (MuJoCo qpos/qvel flattened); `env.set_init_state(s)` restores it
and returns a fresh observation. The OSC controller takes delta actions from the current pose, so no hidden
controller state needs saving; the policy's open-loop action queue is cleared at every (re)start, so a branch
always begins with a fresh query. Whether a restore reproduces a trajectory exactly is checked at run time
(`same_actions`), never assumed.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]


def settle(env, init_state, wait_steps: int = 10):
    """Upstream episode start: restore an init state, then let objects settle with no-op actions."""
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(wait_steps):
        obs, *_ = env.step(DUMMY_ACTION)
    return obs


def rollout(env, policy, instruction: str, obs, *, steps: int,
            on_step: Callable[[int, dict], bool] | None = None) -> dict:
    """Run `policy` for up to `steps` env steps from the current state.

    policy.act(obs, instruction) returns the env actions for one open-loop chunk. on_step(t, obs) is called after
    every env step; returning True stops the rollout. Exceptions are caught and reported, never swallowed into a
    plain failure (IMPLEMENTATION_PLAN §0.6).
    """
    actions, queue, t = [], [], 0
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
