"""E4: when does switching the instruction stop working? (docs/RESEARCH_PLAN.md E4)

E3 showed that after a stable grasp, a new goal on the same object redirects the bowl (stove / plate / cabinet
top: ~85-100%) but almost never the wine bottle (rack <-> cabinet top: 0-33%). E4 asks *when* each becomes
committed: roll out with the original instruction, snapshot the full state at every step, and from the states
at several switch times tau (relative to the stable-grasp step) continue with another goal on the same object,
the empty instruction, and the original (twice, as a check). Outcome = which goal got executed.

  python scripts/e4_switch_time.py --checkpoint $CKPT/LIBERO-Goal-Pro --tasks 2,4 --out $RESULTS/e4/va-run1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path


from langeval.branch import divergence, restore, rollout, settle, snapshot
from langeval.goal_matrix import GoalTracker
from langeval.goal_predicates import load_goals, satisfied
from langeval.phase import GraspDetector, is_grasped, object_z

# Switch times relative to the stable-grasp step g. "start" is tau = 0 (right after settling).
DEFAULT_TAUS = "start,-40,-20,-10,0,20"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--tasks", default="1,2,4,8,9", help="benchmark-order task ids (default: bowl and wine tasks)")
    ap.add_argument("--n_states", type=int, default=20, help="source rollouts with a stable grasp per task")
    ap.add_argument("--max_attempts", type=int, default=40)
    ap.add_argument("--taus", default=DEFAULT_TAUS)
    ap.add_argument("--hold_steps", type=int, default=10)
    ap.add_argument("--lift_m", type=float, default=0.03)
    ap.add_argument("--grasp_k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", default="upstream", choices=["upstream", "lerobot"],
                    help="upstream: VLA-Adapter/OFT (vla env, repo on PYTHONPATH); lerobot: pi0.5 etc. (lerobot env)")
    args = ap.parse_args()
    if args.suite != "libero_goal":
        raise SystemExit("E4 needs the shared LIBERO-Goal scene (other goals on the same object)")

    from libero.libero import benchmark
    from langeval.adapters.factory import load

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    taus = [t if t == "start" else int(t) for t in args.taus.split(",")]
    post = max([t for t in taus if t != "start"] + [0])

    policy, make_env, get_init_states, max_steps = load(args.backend, args.checkpoint, args.suite, args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    goals = load_goals(args.suite)
    (out / "run_config.json").write_text(json.dumps({
        **vars(args), "max_steps": max_steps,
        "git_langvla": subprocess.run(["git", "-C", str(Path(__file__).parents[1]), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip(),
        "model_repo": os.environ.get("PYTHONPATH", "").split(":")[0],
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    episodes = open(out / "episodes.jsonl", "w")
    checks, collection = [], []

    for tid in [int(t) for t in args.tasks.split(",")]:
        task = suite.get_task(tid)
        target = goals[tid].target
        alts = [j for j, g in enumerate(goals) if j != tid and g.target == target]
        if target is None or not alts:
            print(f"task {tid} '{task.language}': no other goal on the same object, skipped")
            continue
        env, _ = make_env(suite, tid)
        init_states = get_init_states(suite, tid)
        conts = [("orig", tid, task.language), ("orig_rep", tid, task.language), ("empty", None, "")]
        conts += [("other", j, goals[j].language) for j in alts]

        n_ok, k = 0, -1
        for k in range(min(args.max_attempts, len(init_states))):
            obs = settle(env, init_states[k])
            snaps = [snapshot(env)]                                   # snaps[t] = state after t policy steps
            det = GraspDetector(object_z(env, target), args.lift_m, args.grasp_k)
            stop_at = [None]

            def on_step(t, o):
                snaps.append(snapshot(env))
                if det.update(object_z(env, target), is_grasped(env, target), t) and stop_at[0] is None:
                    stop_at[0] = t + post
                return stop_at[0] is not None and t >= stop_at[0]

            res = rollout(env, policy, task.language, obs, steps=max_steps, on_step=on_step)
            g = det.stable_step
            if res["error"] or g is None or len(snaps) <= g + post:
                continue
            n_ok += 1
            for tau in taus:
                t_abs = 0 if tau == "start" else g + tau
                if t_abs < 0:
                    continue
                snap = snaps[t_abs]
                orig_actions = {}
                for kind, goal_id, text in conts:
                    tracker = GoalTracker(len(goals), args.hold_steps, prefer=goal_id if goal_id is not None else tid)
                    obs = restore(env, snap)
                    r = rollout(env, policy, text, obs, steps=max_steps - t_abs,
                                on_step=lambda t, o: tracker.update(satisfied(env, goals), t))
                    if kind in ("orig", "orig_rep"):
                        orig_actions[kind] = r["actions"]
                    episodes.write(json.dumps({
                        "task_id": tid, "task": task.language, "init": k, "grasp_step": g, "tau": tau, "t_abs": t_abs,
                        "cont": kind, "cont_goal": goal_id, "cont_text": text, "executed": tracker.executed,
                        "executed_step": tracker.executed_step,
                        "steps": r["steps"], "error": r["error"]}) + "\n")
                episodes.flush()
                checks.append({"task_id": tid, "init": k, "tau": tau, **divergence(orig_actions["orig"], orig_actions["orig_rep"])})
            print(f"task {tid} init {k}: grasp at {g}, {len(taus)} switch times x {len(conts)} continuations", flush=True)
            if n_ok >= args.n_states:
                break
        collection.append({"task_id": tid, "attempts": k + 1, "grasped": n_ok})
        env.close()

    episodes.close()
    (out / "collection.json").write_text(json.dumps(collection, indent=2))
    (out / "determinism.json").write_text(json.dumps({"orig_vs_orig_rep": checks,
                                                      "n_identical": sum(c["first_diff_step"] is None for c in checks)}, indent=2))
    eps = [json.loads(l) for l in open(out / "episodes.jsonl")]
    summary = {}
    for e in eps:
        key = f't{e["task_id"]} {e["cont"]}{"" if e["cont_goal"] is None else e["cont_goal"]} tau={e["tau"]}'
        res = "error" if e["error"] else "none" if e["executed"] is None else \
            "orig" if e["executed"] == e["task_id"] else "new" if e["executed"] == e["cont_goal"] else f'g{e["executed"]}'
        summary.setdefault(key, Counter())[res] += 1
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for key, c in summary.items():
        print(key, dict(c))


if __name__ == "__main__":
    main()
