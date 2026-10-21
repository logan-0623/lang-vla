"""Closed-loop success on the LIBERO-Goal tasks and on novel combinations of their objects and destinations
(configs/novel_goals.json), recording which goal each rollout executed.

Success and stopping follow LIBERO's own `done`: the episode ends as soon as the *instructed* goal's predicates
hold, or at the step limit. Every other goal of both sets is only recorded (held for `hold_steps` ->
`completed`), never a reason to stop: an intermediate step can satisfy another goal (opening the top drawer is
n7 on the way to n6 and to seen task 3). `executed` = the first other goal completed, for reading failures: the
seen goal on the same object (default destination / name retrieval), another object, or nothing.

  python scripts/eval_goals.py --checkpoint $CKPT/LIBERO-Goal-Pro --n_init 20 --out $RESULTS/novel/va-...
  python scripts/eval_goals.py --backend starvla --checkpoint 127.0.0.1:6694 --sets novel --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path

from langeval.branch import rollout, settle
from langeval.goal_matrix import GoalTracker
from langeval.goal_predicates import load_goals, load_novel_goals, satisfied

ROOT = Path(__file__).parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", default="upstream", choices=["upstream", "lerobot", "starvla"])
    ap.add_argument("--sets", default="seen,novel", help="seen = the 10 LIBERO-Goal tasks, novel = configs/novel_goals.json")
    ap.add_argument("--novel_json", default=str(ROOT / "configs/novel_goals.json"))
    ap.add_argument("--goals", default=None, help="only these goal names, e.g. s4,n2 (default: all of --sets)")
    ap.add_argument("--n_init", type=int, default=20)
    ap.add_argument("--hold_steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    suite_name = "libero_goal"

    from libero.libero import benchmark
    from langeval.adapters.factory import load

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    policy, make_env, get_init_states, max_steps = load(args.backend, args.checkpoint, suite_name, args.seed)
    suite = benchmark.get_benchmark_dict()[suite_name]()
    seen, novel = load_goals(suite_name), load_novel_goals(args.novel_json)
    goals = seen + novel                                  # goal index: 0-9 seen (benchmark order), 10+ novel
    names = [f"s{i}" for i in range(len(seen))] + [g["id"] for g in json.load(open(args.novel_json))["goals"]]
    (out / "run_config.json").write_text(json.dumps({
        **vars(args), "max_steps": max_steps, "goal_names": names, "goal_language": [g.language for g in goals],
        "env": {k: v for k, v in os.environ.items() if k.startswith("LV_")},
        "git_langvla": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))

    sets = args.sets.split(",")
    todo = ([(i, i) for i in range(len(seen))] if "seen" in sets else []) + \
           ([(len(seen) + j, 0) for j in range(len(novel))] if "novel" in sets else [])   # (goal, env task); scene is shared
    if args.goals:
        todo = [(gi, t) for gi, t in todo if names[gi] in args.goals.split(",")]
    episodes = open(out / "episodes.jsonl", "w")
    for gi, task_id in todo:
        env, _ = make_env(suite, task_id)
        inits = get_init_states(suite, task_id)
        for k in range(min(args.n_init, len(inits))):
            obs = settle(env, inits[k], seed=args.seed * 100_000 + gi * 1000 + k)
            tracker = GoalTracker(len(goals), args.hold_steps, prefer=gi)

            def on_step(t, o):
                sat = satisfied(env, goals)
                tracker.update(sat, t)
                return sat[gi]                        # LIBERO done: instructed goal holds now
            r = rollout(env, policy, goals[gi].language, obs, steps=max_steps, on_step=on_step)
            success = r["error"] is None and tracker.last[gi]
            others = sorted((t, i) for i, t in tracker.completed.items() if i != gi)
            episodes.write(json.dumps({
                "goal": names[gi], "language": goals[gi].language, "init": k, "success": success,
                "success_step": r["steps"] if success else None,
                "executed": names[gi] if success else (names[others[0][1]] if others else None),
                "completed_other": {names[i]: t for t, i in others},
                "first_satisfied": {names[i]: t for i, t in tracker.first_satisfied.items()},
                "steps": r["steps"], "error": r["error"]}) + "\n")
            episodes.flush()
        env.close()
    episodes.close()

    eps = [json.loads(l) for l in open(out / "episodes.jsonl")]
    summary = {}
    for name in dict.fromkeys(e["goal"] for e in eps):
        es = [e for e in eps if e["goal"] == name and not e["error"]]
        summary[name] = {"language": es[0]["language"] if es else None, "n": len(es),
                         "success": sum(e["success"] for e in es) / max(len(es), 1),
                         "executed": dict(Counter(str(e["executed"]) for e in es)),
                         "errors": sum(bool(e["error"]) for e in eps if e["goal"] == name)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    for name, s in summary.items():
        print(f"{name:<4} {s['success']:.2f} (n={s['n']}, err={s['errors']})  {s['language']}  executed: {s['executed']}")


if __name__ == "__main__":
    main()
