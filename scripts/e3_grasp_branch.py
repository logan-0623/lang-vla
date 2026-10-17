"""E3: does instruction wording change what happens AFTER a stable grasp? (docs/RESEARCH_DESIGN.md)

1. Collect: roll out from init states with the original instruction and with paraphrases; save the simulator
   state at the first stable grasp of the task's target object.
2. Branch: from every saved state, continue with each continuation instruction:
     orig | LIBERO-Para paraphrases | meaning-preserving perturbations | empty | other goals on the same object
   and classify the outcome (langeval.phase.classify_outcome) and the executed goal (langeval.goal_matrix).

The source x continuation design separates "the paraphrase produced a worse grasp" from "the paraphrase breaks
post-grasp control". Every state is also continued twice with the original instruction (`orig`, `orig_rep`):
their disagreement is the noise floor that instruction effects are compared against. determinism.json records,
per state, where the two orig action sequences first differ; for the first state it also checks that two
restores give identical observations (if not, restoring is broken and the script stops) and that the policy
returns identical actions for one observation twice.

Run with the model repo first on PYTHONPATH (bridges2/run.sbatch REPO=vla-adapter|oft):
  python scripts/e3_grasp_branch.py --checkpoint $CKPT/LIBERO-Goal-Pro --out $RESULTS/e3/va-run1
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np

from langeval.branch import divergence, restore, rollout, same_obs, settle, snapshot
from langeval.goal_matrix import GoalTracker
from langeval.goal_predicates import load_goals, satisfied
from langeval.instructions import PERTURBATIONS, load_paraphrases, perturb
from langeval.phase import GraspDetector, classify_outcome, is_grasped, movable_objects, object_z, path_length


def git_rev(path: str) -> str:
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def pick_paraphrases(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """n paraphrases spread over categories (object-lexical first: LIBERO-Para's largest drop)."""
    order = sorted({(r["high"], r["mid"]) for r in rows}, key=lambda c: (c != ("obj", "lexical"), c))
    picked = []
    for cat in order:
        pool = [r for r in rows if (r["high"], r["mid"]) == cat]
        if pool and len(picked) < n:
            picked.append(rng.choice(pool))
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--tasks", default="", help="comma-separated task ids (benchmark order); default: every pick-and-place (\"put\") task")
    ap.add_argument("--n_states", type=int, default=20, help="grasp states per (task, source)")
    ap.add_argument("--max_attempts", type=int, default=50, help="init states tried per (task, source)")
    ap.add_argument("--source_paraphrases", type=int, default=1)
    ap.add_argument("--cont_paraphrases", type=int, default=3)
    ap.add_argument("--para_csv", default=None, help="default: $PARA/metrics/libero_para_metadata.csv")
    ap.add_argument("--paraphrase_json", default=None,
                    help='{"instruction": ["para", ...] or [["category", "para"], ...]} instead of LIBERO-Para')
    ap.add_argument("--hold_steps", type=int, default=10)
    ap.add_argument("--lift_m", type=float, default=0.03)
    ap.add_argument("--grasp_k", type=int, default=5)
    ap.add_argument("--stall_window", type=int, default=30, help="final steps over which 'stalled' is judged")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", default="upstream", choices=["upstream", "lerobot"],
                    help="upstream: VLA-Adapter/OFT (vla env, repo on PYTHONPATH); lerobot: pi0.5 etc. (lerobot env)")
    args = ap.parse_args()

    from libero.libero import benchmark
    from langeval.adapters.factory import load

    out = Path(args.out)
    (out / "states").mkdir(parents=True, exist_ok=False)          # never overwrite an earlier run
    rng = random.Random(args.seed)

    policy, make_env, get_init_states, max_steps = load(args.backend, args.checkpoint, args.suite, args.seed)
    suite = benchmark.get_benchmark_dict()[args.suite]()
    shared = args.suite == "libero_goal"                         # all goals share one scene only here
    all_goals = load_goals(args.suite) if shared else None

    if args.paraphrase_json:
        # Entries are "text" or ["category", "text"]; each category is picked separately, in sorted order.
        para_rows = {k: [{"high": "user", "mid": p[0], "low": "", "new_instruction": p[1]} if isinstance(p, list)
                         else {"high": "user", "mid": "user", "low": "", "new_instruction": p} for p in v]
                     for k, v in json.load(open(args.paraphrase_json)).items()}
    else:
        csv_path = args.para_csv or Path(os.environ["PARA"]) / "metrics/libero_para_metadata.csv"
        para_rows = load_paraphrases(csv_path, per_category=10**6, seed=args.seed)

    task_ids = [int(t) for t in args.tasks.split(",")] if args.tasks else range(suite.n_tasks)
    (out / "run_config.json").write_text(json.dumps({
        **vars(args), "max_steps": max_steps, "policy_deterministic": policy.deterministic,
        "git_langvla": git_rev(str(Path(__file__).parents[1])), "git_model_repo": git_rev(os.environ.get("PYTHONPATH", ".").split(":")[0]),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    episodes = open(out / "episodes.jsonl", "w")
    collection = []
    checked_restore, restore_check, replays = False, None, []

    for tid in task_ids:
        task = suite.get_task(tid)
        goals = all_goals if shared else load_goals(args.suite, [tid])
        g_self = tid if shared else 0
        target = goals[g_self].target
        if target is None or (not args.tasks and "put" not in task.language.lower()):
            # No object to grasp (drawer, stove), or the object is pushed, never lifted (push the plate).
            print(f"task {tid} '{task.language}': not a pick-and-place task, skipped")
            continue
        rows = para_rows.get(task.language, [])
        if not rows:
            print(f"task {tid}: no paraphrases for '{task.language}', skipped")
            continue
        env, _ = make_env(suite, tid)
        init_states = get_init_states(suite, tid)
        emb_orig = policy.instruction_embedding(task.language)

        # ---- 1. collect grasp states ----
        src_paras = pick_paraphrases(rows, args.source_paraphrases, rng)
        sources = [("orig", "", task.language)] + [("para", f'{r["high"]}/{r["mid"]}', r["new_instruction"]) for r in src_paras]
        states = []
        for src_kind, src_cat, src_text in sources:
            n_grasp = 0
            attempts = min(args.max_attempts, len(init_states))
            for k in range(attempts):
                obs = settle(env, init_states[k])
                det = GraspDetector(object_z(env, target), args.lift_m, args.grasp_k)
                res = rollout(env, policy, src_text, obs, steps=max_steps,
                              on_step=lambda t, o: det.update(object_z(env, target), is_grasped(env, target), t))
                if res["error"] is None and det.stable_step is not None:
                    path = out / "states" / f"t{tid}-{src_kind}{len(states)}-init{k}.npz"
                    snap = snapshot(env)
                    np.savez(path, **snap, grasp_step=det.stable_step, rest_z=det.rest_z)
                    states.append({"path": path, "src_kind": src_kind, "src_cat": src_cat, "src_text": src_text,
                                   "init": k, "grasp_step": det.stable_step, "rest_z": det.rest_z})
                    n_grasp += 1
                if n_grasp >= args.n_states:
                    break
            collection.append({"task_id": tid, "source": src_kind, "source_text": src_text,
                               "attempts": k + 1, "grasps": n_grasp})
            print(f"task {tid} source {src_kind}: {n_grasp} grasp states from {k + 1} attempts", flush=True)

        # ---- 2. branch ----
        # The source paraphrases are always continuations too: (para state, same para) is the 2x2's fourth cell.
        conts = [("orig", "", task.language), ("orig_rep", "", task.language)]
        conts += [("para_source", f'{r["high"]}/{r["mid"]}/{r["low"]}', r["new_instruction"]) for r in src_paras]
        fresh = [r for r in rows if r["new_instruction"] not in {p["new_instruction"] for p in src_paras}]
        conts += [("para", f'{r["high"]}/{r["mid"]}/{r["low"]}', r["new_instruction"])
                  for r in pick_paraphrases(fresh, args.cont_paraphrases, rng)]
        conts += [("perturb", p, perturb(p, task.language)) for p in PERTURBATIONS]
        conts += [("empty", "", "")]
        if shared:
            conts += [("other_same_object", str(j), g.language) for j, g in enumerate(goals)
                      if j != g_self and g.target == target]

        for s in states:
            npz = np.load(s["path"])
            snap = {k: npz[k] for k in ("state", "body_pos", "body_quat", "gripper")}
            budget = max_steps - s["grasp_step"]
            if not checked_restore:
                begin = getattr(policy, "begin", lambda: None)     # stochastic policies: replay the same noise
                o1 = restore(env, snap)
                begin(); a1 = policy.act(o1, task.language)
                begin(); a1_again = policy.act(o1, task.language)
                o2 = restore(env, snap)
                begin()
                restore_check = {"state": s["path"].name, "restore_obs_equal": same_obs(o1, o2),
                                 "policy_repeatable": all(np.array_equal(x, y) for x, y in zip(a1, a1_again)),
                                 "first_chunk_equal_after_two_restores": all(
                                     np.array_equal(x, y) for x, y in zip(a1, policy.act(o2, task.language)))}
                print("restore check:", restore_check, flush=True)
                if not restore_check["restore_obs_equal"]:
                    (out / "determinism.json").write_text(json.dumps({"restore_check": restore_check}, indent=2))
                    raise SystemExit("two restores of one state give different observations: restore is broken")
                checked_restore = True
            orig_actions = {}
            for cont_kind, cont_cat, cont_text in conts:
                tracker = GoalTracker(len(goals), args.hold_steps, prefer=g_self)
                others = [o for o in movable_objects(env) if o != target]
                other_grasped, eef = [False], []

                def on_step(t, o):
                    eef.append(np.asarray(o["robot0_eef_pos"], float))
                    other_grasped[0] |= any(is_grasped(env, n) for n in others)
                    return tracker.update(satisfied(env, goals), t)

                obs = restore(env, snap)
                res = rollout(env, policy, cont_text, obs, steps=budget, on_step=on_step)
                if cont_kind in ("orig", "orig_rep"):
                    orig_actions[cont_kind] = res["actions"]

                outcome = "error" if res["error"] else classify_outcome(
                    target_goal=g_self, executed=tracker.executed, target_grasped_end=is_grasped(env, target),
                    target_z_end=object_z(env, target), rest_z=s["rest_z"], other_grasped=other_grasped[0],
                    eef_path_end=path_length(eef[-args.stall_window:]))
                ep = {"task_id": tid, "task": task.language, "init": s["init"], "state": s["path"].name,
                      "source": s["src_kind"], "source_cat": s["src_cat"], "source_text": s["src_text"],
                      "grasp_step": s["grasp_step"], "cont": cont_kind, "cont_cat": cont_cat, "cont_text": cont_text,
                      "embed_dist": float(np.linalg.norm(policy.instruction_embedding(cont_text) - emb_orig)),
                      "outcome": outcome, "executed": tracker.executed, "executed_step": tracker.executed_step,
                      "first_satisfied": tracker.first_satisfied, "final_satisfied": [k for k, v in enumerate(tracker.last) if v],
                      "steps": res["steps"], "error": res["error"]}
                episodes.write(json.dumps(ep) + "\n")
                episodes.flush()
            replays.append({"task_id": tid, "state": s["path"].name,
                            **divergence(orig_actions["orig"], orig_actions["orig_rep"])})
            print(f"task {tid} state {s['path'].name}: {len(conts)} branches, orig vs orig_rep: {replays[-1]}",
                  flush=True)
        env.close()

    episodes.close()
    (out / "collection.json").write_text(json.dumps(collection, indent=2))
    (out / "determinism.json").write_text(json.dumps({"restore_check": restore_check, "orig_vs_orig_rep": replays,
                                                      "n_identical": sum(r["first_diff_step"] is None for r in replays)},
                                                     indent=2))
    eps = [json.loads(l) for l in open(out / "episodes.jsonl")]
    table = Counter((e["source"], e["cont"], e["outcome"]) for e in eps)
    summary = {}
    for (src, cont, outc), n in sorted(table.items()):
        summary.setdefault(f"{src}->{cont}", {})[outc] = n
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    n_err = sum(e["outcome"] == "error" for e in eps)
    if n_err:
        print(f"WARNING: {n_err} branches raised exceptions (outcome=error, excluded from rates)")


if __name__ == "__main__":
    main()
