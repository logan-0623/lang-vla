"""One entry point for the experiment scripts: policy, env factory, init states and episode length per backend.

  upstream: VLA-Adapter / OpenVLA-OFT (repo first on PYTHONPATH, `vla` env)
  lerobot:  pi0.5 / SmolVLA / X-VLA (`lerobot` env)
  starvla:  StarVLA (Qwen3-VL) via its policy server; `checkpoint` is host:port (`vla` env, StarVLA repo on PYTHONPATH)
"""
from __future__ import annotations


def load(backend: str, checkpoint: str, suite: str, seed: int):
    """Returns (policy, make_env(task_suite, task_id) -> (env, language), init_states(task_suite, i), max_steps)."""
    if backend == "upstream":
        from experiments.robot.libero.libero_utils import get_libero_env
        from experiments.robot.libero.run_libero_eval import TASK_MAX_STEPS
        from experiments.robot.robot_utils import set_seed_everywhere
        from langeval.adapters.upstream import UpstreamPolicy
        set_seed_everywhere(seed)
        policy = UpstreamPolicy(checkpoint, suite)
        return (policy, lambda s, i: get_libero_env(s.get_task(i), "openvla", resolution=256),
                lambda s, i: s.get_task_init_states(i), TASK_MAX_STEPS[suite])
    if backend == "lerobot":
        from lerobot.envs.libero import TASK_SUITE_MAX_STEPS, get_task_init_states
        from lerobot.utils.random_utils import set_seed
        from langeval.adapters.lerobot_policy import LeRobotPolicy
        set_seed(seed)
        policy = LeRobotPolicy(checkpoint, suite, seed=seed)
        # LeRobot's loader passes weights_only=False (init states are numpy pickles; torch >= 2.6 refuses them).
        return (policy, lambda s, i: policy.make_env(s, i, suite), lambda s, i: get_task_init_states(s, i),
                TASK_SUITE_MAX_STEPS[suite])
    if backend == "starvla":
        from langeval.adapters.starvla import StarVLAPolicy, make_env
        np_seed(seed)
        return (StarVLAPolicy(checkpoint, suite), make_env, lambda s, i: s.get_task_init_states(i), MAX_STEPS[suite])
    raise ValueError(f"unknown backend {backend!r}")


# Upstream LIBERO eval episode lengths (TASK_MAX_STEPS in OpenVLA-OFT / VLA-Adapter, StarVLA's eval_libero.py).
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def np_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
