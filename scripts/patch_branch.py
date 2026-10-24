"""RESEARCH_PLAN §11.7 stage 2: once the target object is already held, does patching the destination improve the
placement, and does the robot keep the right object?

  collect : roll out each goal *unpatched* from its init states until a stable grasp of its target object
            (langeval.phase.GraspDetector) and save the simulator state (langeval.branch.snapshot). Run without
            LV_PATCH. Scenes are seeded as in scripts/eval_goals.py.
  branch  : from every saved state, continue with the goal's own instruction. The condition is the server's
            LV_PATCH_U plus the client's LV_PATCH donor map (scripts/erase_action.py, langeval/adapters/starvla.py):
            one condition per job, all from the same states. Run `none` with the same donor map as the patches.
  report  : success, post-grasp outcome (langeval.phase.classify_outcome) and donor goal completed, per run and goal;
            where the held target ended (every on/in destination of any goal) and whether another object was grasped (other_grasped: the grasp check, not contact).
            --lang on branch continues with another instruction (parse_lang).
  probe   : one call per (state, instruction), nothing executed, for the server's LV_ATTN attention dumps; `attn`
            reads them (camera / instruction shares, token weights, image overlays).
  norms   : ||U U^T d|| / ||d|| of each patch on the branch states, from a LV_RECORD run with LV_PATCH.

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


def parse_lang(spec: str, goals, names) -> dict:
    """--lang "n3=s1;n4=put the wine bottle on the dax;s8=" : the instruction a goal's states continue with, a goal
    name (that goal's instruction) or literal text (may be empty). Success is still the state's own goal."""
    out = {}
    for kv in filter(None, (spec or "").split(";")):
        g, text = kv.split("=", 1)
        out[g] = goals[names.index(text)].language if text in names else text
    return out


def probe(args) -> None:
    """One policy call per (state, instruction), nothing executed: for the server's LV_ATTN attention dumps.
    --instr is ';'-separated, each a goal name, literal text, or `own` (the state's goal)."""
    from langeval.branch import restore
    policy, make_env, get_init_states, max_steps, suite, goals, names = setup(args)
    if os.environ.get("LV_PATCH"):
        raise SystemExit("probe sends one instruction per call: unset LV_PATCH")
    sdir, out = Path(args.states), Path(args.out)
    config(args, out, goal_names=names)
    states = [json.loads(l) for l in open(sdir / "states.jsonl")]
    env, _ = make_env(suite, 0)
    log = open(out / "probe.jsonl", "w")
    for s in [s for g in args.goals.split(",") for s in [x for x in states if x["goal"] == g][:args.n_per_goal]]:
        z = np.load(sdir / s["path"])
        obs = restore(env, {k: z[k] for k in ("state", "body_pos", "body_quat", "gripper")})
        for name in args.instr.split(";"):
            key = s["goal"] if name == "own" else name
            text = goals[names.index(key)].language if key in names else name
            policy.act(obs, text)
            log.write(json.dumps({"goal": s["goal"], "init": s["init"], "instr": name, "lang": text}) + "\n")
            log.flush()
    env.close()


def attn_split(tokens, grid, merge: int) -> dict:
    """Positions of each image's tokens and of the instruction (between the last image and the OFT prompt suffix)."""
    img = [i for i, t in enumerate(tokens) if t == "<|image_pad|>"]
    sizes = [int(np.prod(g)) // merge ** 2 for g in grid]
    assert sum(sizes) == len(img), f"{len(img)} image tokens vs grid {grid.tolist()}"
    cut = np.cumsum([0] + sizes)
    start = img[-1] + 2                                          # skip <|vision_end|>
    end = next(i for i in range(start, len(tokens)) if tokens[i].lstrip("\u0120") == "Please")
    return {"images": [img[a:b] for a, b in zip(cut[:-1], cut[1:])], "instr": list(range(start, end))}


def attn(args) -> None:
    """Attention of the action tokens (mean over the 8 tokens and over --layers) from the server's LV_ATTN dumps of
    a probe run: share on each camera's image tokens / the instruction / the rest (template, suffix, action tokens),
    the instruction tokens' weights (text heat map) and, with matplotlib, an image overlay per (goal, instruction).
    Attention weights are descriptive: where the action tokens read from, not what changes the action."""
    from collections import defaultdict
    d = Path(args.run)
    rows = [json.loads(l) for l in open(d / "probe.jsonl")]
    files = sorted(Path(args.dumps).glob("call*_0.npz"))
    assert len(files) == len(rows), f"{len(files)} dumps vs {len(rows)} probe calls: not the same run?"
    a, b = (int(x) if x else None for x in args.layers.split(":"))
    groups = defaultdict(list)
    for r, f in zip(rows, files):
        z = np.load(f)
        w = z["att"][a:b].astype(np.float32).mean((0, 1))                     # [L]
        sp = attn_split(list(z["tokens"]), z["grid"], int(z["merge"]))
        groups[(r["goal"], r["instr"])].append((w, sp, z))
    print(f"layers {args.layers}; share of attention: agentview / wrist / instruction / rest")
    for (g, ins), xs in groups.items():
        sh = np.mean([[w[sp["images"][0]].sum(), w[sp["images"][1]].sum(), w[sp["instr"]].sum()] for w, sp, _ in xs], 0)
        w0, sp0, z0 = xs[0]
        toks = [z0["tokens"][i].replace("\u0120", " ") for i in sp0["instr"]]
        same = all(len(sp["instr"]) == len(sp0["instr"]) for _, sp, _ in xs)
        tw = np.mean([w[sp["instr"]] for w, sp, _ in xs], 0) if same else w0[sp0["instr"]]
        tw = tw / max(tw.sum(), 1e-12)
        print(f"{g} [{ins}] n={len(xs)}  {sh[0]:.2f} / {sh[1]:.2f} / {sh[2]:.3f} / {1 - sh.sum():.2f}")
        print("    " + " ".join(f"{t.strip() or '_'}:{v:.2f}" for t, v in zip(toks, tw)))
    if args.png:
        per_state_figures(args, rows, files, a, b)


def per_state_figures(args, rows, files, a, b) -> None:
    """One figure per probed state: one row per instruction (same observation), overlays on that state's own
    images, one colour scale per camera for the whole figure (rows are comparable, figures are not). With --word,
    a last column shows that instruction token's attention to the agentview patches (dumps with att_text only)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from collections import defaultdict
    states = defaultdict(list)
    for r, f in zip(rows, files):
        states[(r["goal"], r["init"])].append((r, np.load(f)))
    out = Path(args.png)
    out.mkdir(parents=True, exist_ok=True)
    for (g, init), xs in states.items():
        heats, texts, words = [], [], []
        for r, z in xs:
            sp = attn_split(list(z["tokens"]), z["grid"], int(z["merge"]))
            w = z["att"][a:b].astype(np.float32).mean((0, 1))
            m = int(z["merge"])
            heats.append([w[sp["images"][c]].reshape(z["grid"][c][1] // m, z["grid"][c][2] // m) for c in range(2)])
            toks = [z["tokens"][i].replace("\u0120", " ").strip() or "_" for i in sp["instr"]]
            tw = w[sp["instr"]]
            texts.append((toks, tw / max(tw.sum(), 1e-12)))
            hit = None
            if args.word and "att_text" in z and args.word in toks:
                k = sp["instr"][toks.index(args.word)] - int(z["text_start"])            # row in att_text
                ww = z["att_text"][a:b, k].astype(np.float32).mean(0)
                hit = ww[sp["images"][0]].reshape(z["grid"][0][1] // m, z["grid"][0][2] // m)
            words.append(hit)
        vmax = [max(h[c].max() for h in heats) for c in range(2)]
        wmax = max([h.max() for h in words if h is not None], default=None)
        ncol = 4 if wmax is not None else 3
        fig, ax = plt.subplots(len(xs), ncol, figsize=(4.2 * ncol, 3.4 * len(xs)), squeeze=False)
        for i, ((r, z), h, (toks, tw), wh) in enumerate(zip(xs, heats, texts, words)):
            for c in range(2):
                img = z["imgs"][c]
                ax[i, c].imshow(img)
                im = ax[i, c].imshow(h[c], cmap="magma", alpha=0.55, vmin=0, vmax=vmax[c],
                                     extent=(0, img.shape[1], img.shape[0], 0))
                ax[i, c].set_title(f"{['agentview', 'wrist'][c]}  sum {h[c].sum():.3f}", fontsize=9)
                ax[i, c].axis("off")
                if i == 0:
                    fig.colorbar(im, ax=ax[:, c], location="bottom", shrink=0.8, pad=0.02,
                                 label=f"{['agentview', 'wrist'][c]}: action-token attention (shared in figure)")
            ax[i, 2].barh(range(len(toks)), tw, color="tab:orange")
            ax[i, 2].set_yticks(range(len(toks)), toks, fontsize=8)
            ax[i, 2].invert_yaxis()
            ax[i, 2].set_xlim(0, 1)
            ax[i, 2].set_title(f"[{r['instr'] or 'empty'}] text share (incl. prompt template)", fontsize=9)
            if ncol == 4:
                ax[i, 3].axis("off")
                if wh is not None:
                    ax[i, 3].imshow(z["imgs"][0])
                    ax[i, 3].imshow(wh, cmap="viridis", alpha=0.55, vmin=0, vmax=wmax,
                                    extent=(0, z["imgs"][0].shape[1], z["imgs"][0].shape[0], 0))
                    ax[i, 3].set_title(f"'{args.word}' -> agentview", fontsize=9)
        fig.suptitle(f"{g} init {init}  layers {args.layers}  (same observation in every row)")
        fig.savefig(out / f"{g}_init{init}.png", dpi=100, bbox_inches="tight")
        plt.close(fig)
    print(f"{len(states)} figures in {out}")


def norms(args) -> None:
    """How much of the actual donor difference d = q_B - q_A each patch moves, on the branch states themselves:
    ||U U^T d|| / ||d|| (median over calls and the 8 tokens) per subspace and rank, from a run recorded with
    LV_RECORD and LV_PATCH (server rows alternate A, donor B; recorded before any patch)."""
    shape = json.load(open(args.rec + ".json"))["shape"]
    X = np.fromfile(args.rec, np.float16).reshape(-1, 2, *shape).astype(np.float32)           # [calls, A/B, 8, H]
    d = X[:, 1] - X[:, 0]
    print(f"{len(X)} calls; ||d|| / ||q_A|| median {np.median(np.linalg.norm(d, axis=-1) / np.linalg.norm(X[:, 0], axis=-1)):.2f}")
    z = np.load(args.subspaces)
    for c in args.conds.split(","):
        name, k = c.split(":")
        U = z[f"{name}.U"][:, :, :int(k)]                                                  # [8, H, k]
        p = np.einsum("thk,ntk->nth", U, np.einsum("thk,nth->ntk", U, d))                # as the server applies it
        print(f"  {c:<10} ||UU^T d|| / ||d|| median {np.median(np.linalg.norm(p, axis=-1) / np.linalg.norm(d, axis=-1)):.2f}")


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
    from langeval.branch import restore, rollout, same_obs, shift_object
    from langeval.goal_matrix import GoalTracker
    from langeval.goal_predicates import satisfied
    from langeval.phase import classify_outcome, is_grasped, movable_objects, object_z, path_length
    policy, make_env, get_init_states, max_steps, suite, goals, names = setup(args)
    sdir, out = Path(args.states), Path(args.out)
    config(args, out, max_steps=max_steps, goal_names=names)
    states = [json.loads(l) for l in open(sdir / "states.jsonl")
              if not args.goals or json.loads(l)["goal"] in args.goals.split(",")]
    lang = parse_lang(args.lang, goals, names)
    if lang and os.environ.get("LV_PATCH"):
        raise SystemExit("--lang with LV_PATCH: the client finds donors by the goal's own instruction text")
    # every place a goal puts an object, to see where the held target ends up (whatever the instruction asked)
    dests = list({p[2]: [p[0].lower(), p[2]] for g in goals for p in g.predicates
                  if p[0].lower() in ("on", "in")}.values())
    dests = [[k, None, d] for k, d in dests]
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
        other_grasped, eef, on_ever = [False], [], {}
        places = [[p[0], target, p[2]] for p in dests if p[2] != target]

        def on_step(t, o):
            eef.append(np.asarray(o["robot0_eef_pos"], float))
            other_grasped[0] |= any(is_grasped(env, n) for n in others)
            for p in places:
                if p[2] not in on_ever and env.env._eval_predicate(p):
                    on_ever[p[2]] = t
            sat = satisfied(env, goals)
            tracker.update(sat, t)
            return sat[gi]                                       # LIBERO done, as in eval_goals
        obs = restore(env, snap)
        shift = None
        if args.shift:
            name, d = args.shift.split(":")
            before = np.asarray(obs[f"{name}_pos"], float)
            obs = shift_object(env, name, [float(x) for x in d.split(",")])
            after = np.asarray(obs[f"{name}_pos"], float)
            clear = min(np.linalg.norm((np.asarray(obs[f"{o}_pos"]) - after)[:2])
                        for o in movable_objects(env) if o not in (name, target))      # the held target moves anyway
            shift = {"object": name, "before": before.tolist(), "after": after.tolist(), "clearance_xy": clear}
        r = rollout(env, policy, lang.get(s["goal"], goals[gi].language), obs, steps=max_steps - s["grasp_step"],
                    on_step=on_step)
        success = r["error"] is None and tracker.last[gi]
        executed = gi if success else tracker.executed
        outcome = "error" if r["error"] else classify_outcome(
            target_goal=gi, executed=executed, target_grasped_end=is_grasped(env, target),
            target_z_end=object_z(env, target), rest_z=s["rest_z"], other_grasped=other_grasped[0],
            eef_path_end=path_length(eef[-args.stall_window:]))
        eps.write(json.dumps({
            "goal": s["goal"], "init": s["init"], "state": s["path"], "success": success, "outcome": outcome,
            "lang": lang.get(s["goal"], goals[gi].language), "target": target, "other_grasped": other_grasped[0],
            "shift": shift,
            "eef_path": np.round(np.asarray(eef[::2]), 4).tolist(),
            "target_on_end": [p[2] for p in places if env.env._eval_predicate(p)], "target_on_ever": on_ever,
            "executed": None if executed is None else names[executed],
            "completed_other": {names[i]: t for i, t in tracker.completed.items() if i != gi},
            "final_satisfied": [names[i] for i, v in enumerate(tracker.last) if v],
            "final_pos": {k: np.asarray(v).tolist() for k, v in (r["obs"] or {}).items() if k.endswith("pos")},
            "steps": r["steps"], "error": r["error"]}) + "\n")
        eps.flush()
    env.close()


def landing_sites(es: list, sites: dict) -> Counter:
    """Per episode: the object that moved most (> 5 cm) and the nearest region site / movable object to its final xy
    within 8 cm ('elsewhere' beyond; 'lifted' if it ends > 5 cm above its start, i.e. still held)."""
    c = Counter()
    for e in es:
        objs = [k for k in e["start_pos"] if k in e["final_pos"] and not k.startswith("robot") and "_to_" not in k]
        k, m = max(((k, np.linalg.norm(np.subtract(e["final_pos"][k], e["start_pos"][k]))) for k in objs),
                   key=lambda kv: kv[1])
        if m <= 0.05:
            c["none moved"] += 1
            continue
        end = np.asarray(e["final_pos"][k])
        if end[2] - e["start_pos"][k][2] > 0.05:
            c[f"{k[:-4]} lifted"] += 1
            continue
        cand = {**sites, **{o[:-4]: e["final_pos"][o] for o in objs if o != k}}
        n, d = min(((n, np.linalg.norm((end - np.asarray(p))[:2])) for n, p in cand.items()), key=lambda x: x[1])
        c[f"{k[:-4]} -> {n if d < 0.08 else 'elsewhere'}"] += 1
    return c


def moved(args) -> None:
    """For an eval_goals run (start_pos / final_pos per episode): per goal, the executed goal and which object moved
    most (> 5 cm, else none) - for the key probes, which never succeed by construction."""
    for d in args.runs:
        eps = [json.loads(l) for l in open(Path(d) / "episodes.jsonl")]
        print(Path(d).name)
        for g in dict.fromkeys(e["goal"] for e in eps):
            es = [e for e in eps if e["goal"] == g]
            mv = Counter()
            for e in es:
                objs = [k for k in e["start_pos"] if k in e["final_pos"] and not k.startswith("robot")
                        and "_to_" not in k]                    # obj_to_robot0_eef_pos moves with the arm
                k, m = max(((k, np.linalg.norm(np.subtract(e["final_pos"][k], e["start_pos"][k]))) for k in objs),
                           key=lambda kv: kv[1])
                mv[k[:-4] if m > 0.05 else "none"] += 1
            print(f"  {g:<3} {es[0]['language']!r:<45} executed {dict(Counter(str(e['executed']) for e in es).most_common())}"
                  f"  moved {dict(mv.most_common())}")
            if args.sites:
                print(f"      moved object ends near (xy < 8 cm): {dict(landing_sites(es, json.load(open(args.sites))).most_common())}")


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
                  f"{sum(any(x in e['completed_other'] for x in b.split('+')) for e in es) if b else '-'}  "
                  f"{dict(Counter(e['outcome'] for e in es).most_common())}")
            if "target_on_end" in es[0]:
                end = Counter(d for e in es for d in e["target_on_end"])
                ever = Counter(d for e in es for d in e["target_on_ever"])
                print(f"           target on (end) {dict(end.most_common())}  (ever) {dict(ever.most_common())}  "
                      f"grasped another object {sum(e['other_grasped'] for e in es)}")
            if es[0].get("shift"):
                # where the held target ended: near the moved object's new place, or where it used to be (6 cm, xy)
                t = es[0]["target"]
                near = lambda e, k: np.linalg.norm((np.asarray(e["final_pos"][f"{t}_pos"]) -
                                                    np.asarray(e["shift"][k]))[:2]) < 0.06
                print(f"           shift {es[0]['shift']['object']}: {t} near new place {sum(near(e, 'after') for e in es)}"
                      f"  near old place {sum(near(e, 'before') for e in es)}  min clearance "
                      f"{min(e['shift']['clearance_xy'] for e in es) * 100:.0f} cm  moved object displaced > 3 cm "
                      f"{sum(np.linalg.norm(np.subtract(e['final_pos'][e['shift']['object'] + '_pos'], e['shift']['after'])) > 0.03 for e in es)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("collect", "branch", "probe"):
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
    b.add_argument("--goals", default="", help="only the states of these goals")
    b.add_argument("--lang", default="", help="instruction overrides, see parse_lang")
    b.add_argument("--shift", default="", help="obj:dx,dy  move a free object after the restore, e.g. plate_1:0,0.1")
    pr = sub.choices["probe"]
    pr.add_argument("--states", required=True)
    pr.add_argument("--goals", required=True)
    pr.add_argument("--instr", required=True)
    pr.add_argument("--n_per_goal", type=int, default=3)
    at = sub.add_parser("attn")
    at.add_argument("run", help="the probe run (probe.jsonl)")
    at.add_argument("--dumps", required=True, help="the server's LV_ATTN directory")
    at.add_argument("--layers", default="-8:", help="python slice over layers, e.g. -8: or 0:12")
    at.add_argument("--png", default=None, help="write one figure per probed state here")
    at.add_argument("--word", default=None, help="also plot this instruction token's attention to the agentview")
    nm = sub.add_parser("norms")
    nm.add_argument("--rec", required=True)
    nm.add_argument("--subspaces", required=True)
    nm.add_argument("--conds", default="dest:4,random:4,dest_c:16,random:16,dest_c:64,random:64")
    mo = sub.add_parser("moved")
    mo.add_argument("runs", nargs="+")
    mo.add_argument("--sites", default=None, help="region positions from scripts/feasible.py --sites")
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+")
    a = ap.parse_args()
    {"collect": collect, "branch": branch, "probe": probe, "attn": attn, "norms": norms, "moved": moved, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
