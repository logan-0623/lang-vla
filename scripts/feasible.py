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
    ap.add_argument("--drop", type=float, default=0.01, help="release height above the destination (m); 0.05 tipped "
                    "the wine bottle out of the bowl although the arm can place it (batch_1030)")
    ap.add_argument("--sites", default=None, help="also write the region sites' world positions (init 0) to this json")
    a = ap.parse_args()
    from libero.libero import benchmark
    from langeval.adapters.starvla import make_env
    from langeval.branch import DUMMY_ACTION, settle, shift_object
    from langeval.goal_predicates import load_novel_goals, satisfied
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    env, _ = make_env(suite, 0)
    inits = suite.get_task_init_states(0)
    if a.sites:
        settle(env, inits[0], seed=a.seed * 100_000)
        sim = env.env.sim
        pos = {n: np.array(sim.data.get_site_xpos(n)).round(4).tolist() for n in env.env.object_sites_dict
               if n.endswith("_region")}
        json.dump(pos, open(a.sites, "w"), indent=1)
        print(f"{len(pos)} region sites -> {a.sites}")
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
            shift_object(env, obj, list(to[:2] - here[:2]) + [to[2] + a.drop - here[2]])
            for _ in range(a.steps):
                env.step(DUMMY_ACTION)
            ok += satisfied(env, [goal])[0]
        print(f"{spec['id']} {spec['language']!r}: predicate holds after the drop in {ok}/{a.n_init}")
    env.close()


if __name__ == "__main__":
    main()
