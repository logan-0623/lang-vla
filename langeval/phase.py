"""Manipulation phases and post-grasp failure taxonomy (RESEARCH_DESIGN E3).

A *stable grasp* of an object: lifted at least `lift_m` above its rest height while the gripper's finger pads
both touch it, for `k` consecutive steps. Thresholds are calibration knobs; check them on a few rollout videos.
"""
from __future__ import annotations

import numpy as np

OUTCOMES = ("success", "wrong_goal", "wrong_object", "stalled", "holding", "dropped", "misplaced", "error")


class GraspDetector:
    def __init__(self, rest_z: float, lift_m: float = 0.03, k: int = 5):
        self.rest_z, self.lift_m, self.k = rest_z, lift_m, k
        self.count = 0
        self.stable_step: int | None = None

    def update(self, z: float, grasped: bool, t: int) -> bool:
        """Returns True once the grasp has been stable for k steps (and stays True)."""
        if self.stable_step is not None:
            return True
        self.count = self.count + 1 if (grasped and z - self.rest_z >= self.lift_m) else 0
        if self.count >= self.k:
            self.stable_step = t
        return self.stable_step is not None


def classify_outcome(*, target_goal: int, executed: int | None, target_grasped_end: bool, target_z_end: float,
                     rest_z: float, other_grasped: bool, eef_path_end: float,
                     table_tol: float = 0.02, still_tol: float = 0.005) -> str:
    """Post-grasp outcome of a branch, checked in this priority order:
    success / wrong_goal (another goal held) / wrong_object (another movable object was grasped) /
    stalled (still holding, end effector moved < still_tol over the final window) / holding (still holding, moving) /
    dropped (released, back at table height) / misplaced (released, resting somewhere else).
    """
    if executed == target_goal:
        return "success"
    if executed is not None:
        return "wrong_goal"
    if other_grasped:
        return "wrong_object"
    if target_grasped_end:
        return "stalled" if eef_path_end < still_tol else "holding"
    return "dropped" if abs(target_z_end - rest_z) < table_tol else "misplaced"


def path_length(points: list) -> float:
    p = np.asarray(points, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(p) > 1 else 0.0


# --- simulator readouts (LIBERO OffScreenRenderEnv) ---------------------------------------------------------

def object_z(env, name: str) -> float:
    inner = env.env
    return float(inner.sim.data.body_xpos[inner.obj_body_id[name]][2])


def is_grasped(env, name: str) -> bool:
    inner = env.env
    return bool(inner._check_grasp(gripper=inner.robots[0].gripper, object_geoms=inner.objects_dict[name]))


def movable_objects(env) -> list[str]:
    return list(env.env.objects_dict)
