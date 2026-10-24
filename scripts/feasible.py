"""Is a goal's end state physically stable? For each goal of a goals json and each of --n_init init states: put the
target object 5 cm above its destination (a region's site, or the destination object), let it fall and settle for
--steps no-op steps (robot still, gripper open), then check the goal's predicates. End-state feasibility only: it says
nothing about whether the arm can carry the object there.

  python scripts/feasible.py --goals_json configs/heldout_goals.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals_json", required=True)
    ap.add_argument("--n_init", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from libero.libero import benchmark
    from langeval.adapters.starvla import make_env
    from langeval.branch import DUMMY_ACTION, settle, shift_object
    from langeval.goal_predicates import load_novel_goals, satisfied
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    env, _ = make_env(suite, 0)
    inits = suite.get_task_init_states(0)
    specs = json.load(open(a.goals_json))["goals"]
    for spec, goal in zip(specs, load_novel_goals(a.goals_json)):
        kind, obj, dest = goal.predicates[0]
        ok = 0
        for k in range(a.n_init):
            settle(env, inits[k], seed=a.seed * 100_000 + k)
            sim = env.env.sim
            to = (np.array(sim.data.get_site_xpos(dest)) if dest in env.env.object_sites_dict
                  else np.array(sim.data.body_xpos[env.env.obj_body_id[dest]]))
            here = np.array(sim.data.body_xpos[env.env.obj_body_id[obj]])
            shift_object(env, obj, list(to[:2] - here[:2]) + [to[2] + 0.05 - here[2]])
            for _ in range(a.steps):
                env.step(DUMMY_ACTION)
            ok += satisfied(env, [goal])[0]
        print(f"{spec['id']} {spec['language']!r}: predicate holds after the drop in {ok}/{a.n_init}")
    env.close()


if __name__ == "__main__":
    main()
