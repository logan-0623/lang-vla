"""RESEARCH_PLAN §11.7 stage 2: once the target object is already held, does patching the destination improve the
placement, and does the robot keep the right object?

  collect : roll out each goal *unpatched* from its init states until a stable grasp of its target object
            (langeval.phase.GraspDetector) and save the simulator state (langeval.branch.snapshot). Run without
            LV_PATCH. Scenes are seeded as in scripts/eval_goals.py.
  branch  : from every saved state, continue with the goal's own instruction. The condition is the server's
            LV_PATCH_U plus the client's LV_PATCH donor map (scripts/erase_action.py, langeval/adapters/starvla.py):
            one condition per job, all from the same states. Run `none` with the same donor map as the patches.
  report  : success, post-grasp outcome (langeval.phase.classify_outcome) and donor goal completed, per run and goal.

  python scripts/patch_branch.py collect --backend starvla --checkpoint SERVER --goals n0,n3,n4 --out $D/states
  LV_PATCH=n0=s8 python scripts/patch_branch.py branch --backend starvla --checkpoint SERVER --states $D/states --out $D/dest
  python scripts/patch_branch.py report $D/none $D/dest ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parents[1]


def setup(args):
    from libero.libero import benchmark
    from langeval.adapters.factory import load
    from langeval.goal_predicates import load_goals, load_novel_goals
    policy, make_env, get_init_states, max_steps = load(args.backend, args.checkpoint, "libero_goal", args.seed)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    seen, novel = load_goals("libero_goal"), load_novel_goals(args.novel_json)
    names = [f"s{i}" for i in range(len(seen))] + [g["id"] for g in json.load(open(args.novel_json))["goals"]]
    return policy, make_env, get_init_states, max_steps, suite, seen + novel, names


def config(args, out: Path, **extra) -> None:
    out.mkdir(parents=True, exist_ok=False)                      # never overwrite an earlier run
    (out / "run_config.json").write_text(json.dumps({
        **vars(args), **extra, "env": {k: v for k, v in os.environ.items() if k.startswith("LV_")},
        "git_langvla": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                      text=True).stdout.strip(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))


def collect(args) -> None:
    from langeval.branch import rollout, settle, snapshot
    from langeval.phase import GraspDetector, is_grasped, object_z
    policy, make_env, get_init_states, max_steps, suite, goals, names = setup(args)
    if os.environ.get("LV_PATCH"):
        raise SystemExit("collect runs unpatched: unset LV_PATCH")
    out = Path(args.out)
    config(args, out, max_steps=max_steps, goal_names=names)
    env, _ = make_env(suite, 0)                                  # LIBERO-Goal: one shared scene
    inits = get_init_states(suite, 0)
    log = open(out / "states.jsonl", "w")
    for g in args.goals.split(","):
        gi, n = names.index(g), 0
        target = goals[gi].target
        for k in range(min(args.max_attempts, len(inits))):
            obs = settle(env, inits[k], seed=args.seed * 100_000 + gi * 1000 + k)
            det = GraspDetector(object_z(env, target), args.lift_m, args.grasp_k)
            r = rollout(env, policy, goals[gi].language, obs, steps=max_steps,
                        on_step=lambda t, o: det.update(object_z(env, target), is_grasped(env, target), t))
            if r["error"] is None and det.stable_step is not None:
                path = out / f"{g}-init{k}.npz"
                np.savez(path, **snapshot(env), grasp_step=det.stable_step, rest_z=det.rest_z)
                log.write(json.dumps({"goal": g, "init": k, "path": path.name, "grasp_step": det.stable_step,
                                      "rest_z": det.rest_z}) + "\n")
                log.flush()
                n += 1
            if n >= args.n_states:
                break
        print(f"{g}: {n} stable grasps of {target} from {k + 1} init states", flush=True)
    env.close()


def branch(args) -> None:
    from langeval.branch import restore, rollout, same_obs
    from langeval.goal_matrix import GoalTracker
    from langeval.goal_predicates import satisfied
    from langeval.phase import classify_outcome, is_grasped, movable_objects, object_z, path_length
    policy, make_env, get_init_states, max_steps, suite, goals, names = setup(args)
    sdir, out = Path(args.states), Path(args.out)
    config(args, out, max_steps=max_steps, goal_names=names)
    states = [json.loads(l) for l in open(sdir / "states.jsonl")]
    env, _ = make_env(suite, 0)
    eps = open(out / "episodes.jsonl", "w")
    for j, s in enumerate(states):
        z = np.load(sdir / s["path"])
        snap = {k: z[k] for k in ("state", "body_pos", "body_quat", "gripper")}
        gi = names.index(s["goal"])
        target = goals[gi].target
        if j == 0 and not same_obs(restore(env, snap), restore(env, snap)):
            raise SystemExit("two restores of one state give different observations: restore is broken")
        tracker = GoalTracker(len(goals), args.hold_steps, prefer=gi)
        others = [o for o in movable_objects(env) if o != target]
        other_grasped, eef = [False], []

        def on_step(t, o):
            eef.append(np.asarray(o["robot0_eef_pos"], float))
            other_grasped[0] |= any(is_grasped(env, n) for n in others)
            sat = satisfied(env, goals)
            tracker.update(sat, t)
            return sat[gi]                                       # LIBERO done, as in eval_goals
        obs = restore(env, snap)
        r = rollout(env, policy, goals[gi].language, obs, steps=max_steps - s["grasp_step"], on_step=on_step)
        success = r["error"] is None and tracker.last[gi]
        executed = gi if success else tracker.executed
        outcome = "error" if r["error"] else classify_outcome(
            target_goal=gi, executed=executed, target_grasped_end=is_grasped(env, target),
            target_z_end=object_z(env, target), rest_z=s["rest_z"], other_grasped=other_grasped[0],
            eef_path_end=path_length(eef[-args.stall_window:]))
        eps.write(json.dumps({
            "goal": s["goal"], "init": s["init"], "state": s["path"], "success": success, "outcome": outcome,
            "executed": None if executed is None else names[executed],
            "completed_other": {names[i]: t for i, t in tracker.completed.items() if i != gi},
            "final_satisfied": [names[i] for i, v in enumerate(tracker.last) if v],
            "final_pos": {k: np.asarray(v).tolist() for k, v in (r["obs"] or {}).items() if k.endswith("pos")},
            "steps": r["steps"], "error": r["error"]}) + "\n")
        eps.flush()
    env.close()


def report(args) -> None:
    for d in args.runs:
        d = Path(d)
        if not (d / "episodes.jsonl").exists():
            print(f"{d.name:<14} not finished")
            continue
        cfg = json.load(open(d / "run_config.json"))
        donors = dict(kv.split("=") for kv in cfg["env"].get("LV_PATCH", "").split(",") if kv)
        eps = [json.loads(l) for l in open(d / "episodes.jsonl")]
        print(f"{d.name}  ({cfg['env'].get('LV_PATCH_U', '-').rsplit('/', 1)[-1]})")
        for g in dict.fromkeys(e["goal"] for e in eps):
            es = [e for e in eps if e["goal"] == g]
            b = donors.get(g)
            print(f"  {g}<-{b or '-':<3} success {sum(e['success'] for e in es)}/{len(es)}  donor goal done "
                  f"{sum(b in e['completed_other'] for e in es) if b else '-'}  "
                  f"{dict(Counter(e['outcome'] for e in es).most_common())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("collect", "branch"):
        p = sub.add_parser(nm)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--backend", default="starvla", choices=["upstream", "lerobot", "starvla"])
        p.add_argument("--novel_json", default=str(ROOT / "configs/novel_goals.json"))
        p.add_argument("--out", required=True)
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--hold_steps", type=int, default=10)
    c = sub.choices["collect"]
    c.add_argument("--goals", required=True)
    c.add_argument("--n_states", type=int, default=20)
    c.add_argument("--max_attempts", type=int, default=40)
    c.add_argument("--lift_m", type=float, default=0.03)
    c.add_argument("--grasp_k", type=int, default=5)
    b = sub.choices["branch"]
    b.add_argument("--states", required=True)
    b.add_argument("--stall_window", type=int, default=30)
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+")
    a = ap.parse_args()
    {"collect": collect, "branch": branch, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
