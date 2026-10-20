"""Spatial / state QA from LIBERO simulator ground truth, as LLaVA-json for StarVLA VLM training and co-training.

Scenes: fresh random placements of the shared LIBERO-Goal scene (env.reset(), never the eval init states),
driven by a policy on the 10 seen tasks; a frame every `--every` steps. Images are the agentview camera flipped
[::-1, ::-1], as in the VLA training data. From each frame's simulator state the same facts are written in three
styles with identical questions (one conversation per image, same number of turns):

  lang : natural-language answers (arm B: spatial language)
  num  : the same facts as compact JSON (arm D: same information, not language)
  ctrl : non-spatial questions about the same image (arm C: same images and turns, no spatial content)

Facts: each object's image position (Qwen3-VL's 0-1000 coordinates), left/right of object pairs, distances,
gripper-to-object offset, what the gripper holds, what each object rests on, drawer / stove state, and the next
step of the instructed task (a fixed rule: grasp / transport / ...; it does not say why). v1 tests spatial-state
supervision only; action preconditions, consequences, step dependencies and failure reasons are later versions.

Two steps. `collect` runs the rollouts and writes images + frames.jsonl (simulator facts, with each object's raw
robosuite projection `rc` = (row, col)); `build` turns frames.jsonl into the llava jsons, so templates, the image
coordinate convention and the train / held-out split can change without new rollouts. `collect` ends with a
`build` at the defaults.

Image coordinates: robosuite's projection is (row, col) in its own image convention, and the saved training
images are the observation flipped [::-1, ::-1]; which axes to flip is checked on the saved images, not assumed:
`build --overlay 12` draws every convention on 12 frames (overlay/). Checked 2026-10-18 on goal-v1: `col` (flip
the column only) sits on the bowl, plate, cream cheese and wine bottle in every overlay; it is the default.

Held-out split by scene (all frames of a scene on one side): scene % heldout_every == heldout_every - 1.

  python scripts/make_spatial_qa.py collect --checkpoint $CKPT/LIBERO-Goal-Pro --n_scenes 50 --out $STORE/qa/goal-v1
  python scripts/make_spatial_qa.py build --dir $STORE/qa/goal-v1 --overlay 12 --px_convention col
  python scripts/make_spatial_qa.py build --dir $STORE/qa/goal-v1 --balance --build_out $STORE/qa/goal-v2 --overlay 0

Balanced builds (--balance, goal-v2): in v1 most answers are predictable without the image ("No." for 95% of
drawer and 98% of stove questions, "Nothing." for 78% of held, "the table" for ~92% of supports). A candidate
question is kept with probability minority-frequency / its-class-frequency and redrawn otherwise, so each binary
fact comes out about even; facts with one class only (bottom drawer never open) are never asked.
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np

from langeval.branch import DUMMY_ACTION, rollout
from langeval.goal_predicates import load_goals, satisfied
from langeval.phase import is_grasped

ROOT = Path(__file__).parents[1]
H = W = 256
OBJECTS = {"akita_black_bowl_1": "black bowl", "cream_cheese_1": "cream cheese", "wine_bottle_1": "wine bottle",
           "plate_1": "plate"}
FIXTURES = {"wooden_cabinet_1": "cabinet", "flat_stove_1": "stove", "wine_rack_1": "wine rack"}
SUPPORTS = [("on", "plate_1", "the plate"), ("on", "akita_black_bowl_1", "the black bowl"),
            ("on", "flat_stove_1_cook_region", "the stove"), ("on", "wooden_cabinet_1_top_side", "top of the cabinet"),
            ("on", "wine_rack_1_top_region", "the wine rack"),
            ("in", "wooden_cabinet_1_top_region", "the top drawer"), ("in", "wooden_cabinet_1_middle_region", "the middle drawer"),
            ("in", "wooden_cabinet_1_bottom_region", "the bottom drawer")]
DRAWERS = ("top", "middle", "bottom")
NOT_IN_SCENE = ["ketchup bottle", "moka pot", "basket", "frying pan", "milk carton", "alphabet soup can"]


def facts(env, obs, task_goal) -> dict:
    """Ground truth for one frame. task_goal: (Goal, its index) of the instructed task."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
    inner = env.env
    pos = {n: np.array(inner.sim.data.body_xpos[inner.obj_body_id[n]]) for n in {**OBJECTS, **FIXTURES}}
    m = get_camera_transform_matrix(inner.sim, "agentview", H, W)
    rc = {n: [int(v) for v in project_points_from_world_to_camera(p[None], m, H, W)[0]] for n, p in pos.items()}
    eef = np.array(obs["robot0_eef_pos"])
    held = next((n for n in OBJECTS if is_grasped(env, n)), None)
    rests = {}
    for n in OBJECTS:
        if n == held:
            rests[n] = "held by the gripper"
            continue
        rests[n] = next((txt for pred, place, txt in SUPPORTS
                         if place != n and inner._eval_predicate([pred, n, place])), "the table")
    drawers = {d: bool(inner._eval_predicate(["open", f"wooden_cabinet_1_{d}_region"])) for d in DRAWERS}
    stove_on = bool(inner._eval_predicate(["turnon", "flat_stove_1"]))
    goal, gi = task_goal
    if satisfied(env, [goal])[0]:
        nxt = "done"
    elif goal.target is None:                                         # drawer / stove tasks
        nxt = "operate"
    elif "push" in goal.language:
        nxt = "push"
    elif "drawer" in goal.language and not drawers["top"] and held != goal.target:
        nxt = "open_drawer"
    elif held == goal.target:
        nxt = "transport"
    else:
        nxt = "grasp"
    return {"rc": rc, "pos": {n: [round(float(v), 3) for v in p] for n, p in pos.items()},
            "eef": [round(float(v), 3) for v in eef], "held": held, "rests": rests, "drawers": drawers,
            "stove_on": stove_on, "next": nxt, "task": goal.language, "target": goal.target}


def name(n):
    return OBJECTS.get(n) or FIXTURES.get(n)


CONVENTIONS = {"none": (False, False), "row": (True, False), "col": (False, True), "both": (True, True)}


def rc_of(f: dict) -> dict:
    """Raw projections; frames from the first collect (before 2026-10-18 evening) stored px with both axes flipped."""
    if "rc" in f:
        return f["rc"]
    return {n: [H - 1 - round(y * H / 1000), W - 1 - round(x * W / 1000)] for n, (x, y) in f["px"].items()}


def px_of(f: dict, convention: str) -> dict:
    """(x, y) in 0-1000 on the saved image, flipping the axes the convention names."""
    flip_r, flip_c = CONVENTIONS[convention]
    out = {}
    for n, (r, c) in rc_of(f).items():
        r, c = (H - 1 - r if flip_r else r), (W - 1 - c if flip_c else c)
        out[n] = (int(round(1000 * c / W)), int(round(1000 * r / H)))
    return out


def balance_key(v: dict, q: str):
    """The (fact, answer class) a question is balanced on, or None. Binary facts with a dominant answer only."""
    if "held" in v:
        return ("held", v["held"] is None)
    if "rests" in v:
        return ("support", v["rests"] == "the table")
    if "on" in v:
        return ("stove", v["on"])
    if "open" in v:
        return (q.split()[2] + "_drawer", v["open"])           # "Is the top drawer ..." -> top_drawer
    return None


def balance_rates(frames: list[dict]) -> dict:
    """Acceptance probability per (fact, class) = minority frequency / this class's frequency, so each fact's
    answers come out about even (a class absent from the data makes its fact never asked)."""
    cnt = {}
    for f in frames:
        x = f["facts"]
        keys = [("held", x["held"] is None), ("stove", x["stove_on"])] + [(f"{d}_drawer", x["drawers"][d]) for d in DRAWERS]
        keys += [("support", r == "the table") for r in x["rests"].values()]
        for k in keys:
            cnt[k] = cnt.get(k, 0) + 1
    rates = {}
    for fact in {k[0] for k in cnt}:
        n = {c: cnt.get((fact, c), 0) for c in (True, False)}
        for c in (True, False):
            rates[(fact, c)] = min(n.values()) / n[c] if n[c] else 0.0
    return rates


def conversation(f: dict, rng: random.Random, style: str, n_turns: int, rates: dict | None = None) -> list[tuple[str, str]]:
    """n_turns (question, answer) pairs. lang and num share questions (num adds "Answer in JSON.").
    With `rates` (balance_rates), a candidate question is kept with its (fact, class) probability and redrawn
    otherwise, so a model cannot score by always answering "No." / "Nothing." / "on the table"."""
    if style == "ctrl":
        present = list(OBJECTS.values()) + list(FIXTURES.values())
        pool = []
        for _ in range(n_turns):
            k = rng.randrange(3)
            # Full sentences, so answer length is close to lang's (C controls extra text QA on the same images,
            # without spatial content; its answers barely depend on the image).
            if k == 0:
                o = rng.choice(present + NOT_IN_SCENE)
                pool.append((f"Is there a {o} in this image?", f"Yes, there is a {o} in this image." if o in present
                             else f"No, there is no {o} in this image."))
            elif k == 1:
                o = rng.choice(list(OBJECTS.values()))
                pool.append(("Name one object the robot could pick up here.", f"The robot could pick up the {o} here."))
            else:
                pool.append(("Is this a kitchen or a living room?", "This is a kitchen, not a living room."))
        return pool
    qs = []
    # Positions only for the four movable objects: fixture body origins are not where they look (the stove's
    # sits on its back knob; overlay check 2026-10-18), so fixtures appear only as supports.
    objs = every = list(OBJECTS)
    for _ in range(n_turns):
        for _attempt in range(200):
            n_before = len(qs)
            _one_question(f, rng, qs, objs, every)
            key = balance_key(qs[-1][2], qs[-1][0]) if rates else None
            if key is None or rng.random() < rates.get(key, 1.0):
                break
            del qs[n_before:]
    if style == "num":
        return [(q + " Answer in JSON.", json.dumps(v, separators=(",", ":"))) for q, _, v in qs]
    return [(q, a) for q, a, _ in qs]


def _one_question(f: dict, rng: random.Random, qs: list, objs: list, every: list) -> None:
    k = rng.randrange(7)
    if k == 0:
        n = rng.choice(every)
        x, y = f["px"][n]
        qs.append((f"Where is the {name(n)} in the image?", f"The {name(n)} is at ({x}, {y}).", {"x": x, "y": y}))
    elif k == 1:
        a, b = rng.sample(every, 2)
        side = "left" if f["px"][a][0] < f["px"][b][0] else "right"
        d = round(100 * float(np.linalg.norm(np.array(f["pos"][a][:2]) - np.array(f["pos"][b][:2]))))
        qs.append((f"Is the {name(a)} to the left or right of the {name(b)}, and how far apart are they?",
                   f"The {name(a)} is to the {side} of the {name(b)}, about {d} cm away.", {"side": side, "cm": d}))
    elif k == 2:
        n = rng.choice(objs)
        dx, dy, dz = (100 * (np.array(f["pos"][n]) - np.array(f["eef"]))).round().astype(int).tolist()
        d = round(float(np.linalg.norm([dx, dy, dz])))
        vert = "below" if dz < 0 else "above"
        qs.append((f"How far is the gripper from the {name(n)}?",
                   f"About {d} cm. The {name(n)} is {abs(dz)} cm {vert} the gripper.", {"cm": d, "dz_cm": dz}))
    elif k == 3:
        h = f["held"]
        qs.append(("What is the gripper holding?", f"The {name(h)}." if h else "Nothing.", {"held": h}))
    elif k == 4:
        n = rng.choice(objs)
        r = f["rests"][n]
        prep = "" if r == "held by the gripper" else "in " if "drawer" in r else "on "
        ans = f"The {name(n)} is {prep}{r}."
        qs.append((f"Where is the {name(n)} resting?", ans, {"rests": r}))
    elif k == 5:
        d = rng.choice(DRAWERS + ("stove",))
        if d == "stove":
            qs.append(("Is the stove turned on?", "Yes." if f["stove_on"] else "No.", {"on": f["stove_on"]}))
        else:
            qs.append((f"Is the {d} drawer of the cabinet open?", "Yes." if f["drawers"][d] else "No.", {"open": f["drawers"][d]}))
    else:
        t, tgt, nx = f["task"], f["target"], f["next"]
        ans = {"done": "The task is already done.", "operate": "Move the gripper to the handle or knob and operate it.",
               "push": f"Put the gripper behind the {name(tgt)} and push it forward." if tgt else "",
               "open_drawer": "Open the top drawer first.",
               "transport": f"Carry the {name(tgt)} to its destination and release it." if tgt else "",
               "grasp": f"Move to the {name(tgt)} and grasp it." if tgt else ""}[nx]
        qs.append((f"The task is: {t}. What should the robot do next?", ans,
                   {"next": nx, "object": name(tgt) if nx in ("grasp", "transport", "push") else None}))


def build(d: Path, convention: str, turns: int, seed: int, heldout_every: int, overlay: int,
          balance: bool = False, out: Path | None = None) -> None:
    frames = [json.loads(l) for l in open(d / "frames.jsonl")]
    rates = balance_rates(frames) if balance else None
    out = out or d                             # a balanced build can go to its own dir; images stay in d
    (out / "llava_jsons").mkdir(parents=True, exist_ok=True)
    for shared in ("images", "frames.jsonl"):  # llava jsons and eval/probe scripts resolve these under the QA dir
        if out != d and not (out / shared).exists():
            (out / shared).symlink_to((d / shared).resolve())
    for style in ("lang", "num", "ctrl"):
        rng = random.Random(seed)              # same seed per style: lang and num ask the same questions
        split = {"train": [], "heldout": []}
        for f in frames:
            fx = {**f["facts"], "px": px_of(f["facts"], convention)}
            conv = []
            for i, (q, a) in enumerate(conversation(fx, rng, style, turns, rates)):
                conv += [{"from": "human", "value": ("<image>\n" if i == 0 else "") + q}, {"from": "gpt", "value": a}]
            side = "heldout" if f["scene"] % heldout_every == heldout_every - 1 else "train"
            split[side].append({"image": f["image"], "conversations": conv, "task_id": f["task_id"], "scene": f["scene"]})
        for side, data in split.items():
            suffix = "" if side == "train" else "_heldout"
            (out / "llava_jsons" / f"libero_qa_{style}{suffix}.json").write_text(json.dumps(data))
        n_chars = sum(len(t["value"]) for t in [x for v in split.values() for e in v for x in e["conversations"]] if t["from"] == "gpt")
        print(f"{style}: {len(split['train'])} train / {len(split['heldout'])} held-out images, answer chars {n_chars}")
    (out / "build_config.json").write_text(json.dumps({"px_convention": convention, "turns": turns, "seed": seed,
                                                       "heldout_every": heldout_every, "balance": balance,
                                                       "frames_dir": str(d)}, indent=2))
    if overlay:
        from PIL import Image, ImageDraw
        (d / "overlay").mkdir(exist_ok=True)
        colors = {"none": "red", "row": "blue", "col": "lime", "both": "yellow"}
        for f in frames[:: max(1, len(frames) // overlay)][:overlay]:
            img = Image.open(d / f["image"]).convert("RGB").resize((4 * W, 4 * H), Image.NEAREST)
            draw = ImageDraw.Draw(img)
            for conv_name, color in colors.items():
                for n, (x, y) in px_of(f["facts"], conv_name).items():
                    u, v = 4 * W * x / 1000, 4 * H * y / 1000
                    draw.ellipse([u - 6, v - 6, u + 6, v + 6], outline=color, width=3)
                    draw.text((u + 8, v - 6), name(n), fill=color)
            draw.text((8, 8), "red none / blue row / lime col / yellow both", fill="white")
            img.save(d / "overlay" / Path(f["image"]).name)
        print(f"overlay: {overlay} images in {d / 'overlay'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--checkpoint", required=True, help="policy that drives the scenes (any backend)")
    c.add_argument("--backend", default="upstream", choices=["upstream", "lerobot", "starvla"])
    c.add_argument("--out", required=True)
    c.add_argument("--n_scenes", type=int, default=50, help="random scenes per task")
    c.add_argument("--every", type=int, default=20, help="env steps between sampled frames")
    b = sub.add_parser("build")
    b.add_argument("--dir", required=True)
    b.add_argument("--balance", action="store_true", help="even answers per binary fact (balance_rates)")
    b.add_argument("--build_out", default=None, help="write the jsons here (images, frames.jsonl symlinked)")
    for a in (c, b):
        a.add_argument("--turns", type=int, default=6, help="QA turns per image")
        a.add_argument("--seed", type=int, default=7)
        a.add_argument("--px_convention", default="col", choices=list(CONVENTIONS))
        a.add_argument("--heldout_every", type=int, default=5)
        a.add_argument("--overlay", type=int, default=12)
    args = ap.parse_args()
    if args.cmd == "build":
        build(Path(args.dir), args.px_convention, args.turns, args.seed, args.heldout_every, args.overlay,
              args.balance, Path(args.build_out) if args.build_out else None)
        return
    collect(args)
    build(Path(args.out), args.px_convention, args.turns, args.seed, args.heldout_every, args.overlay)


def collect(args) -> None:
    from PIL import Image
    from libero.libero import benchmark
    from langeval.adapters.factory import load

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=False)
    (out / "llava_jsons").mkdir()
    policy, make_env, _, max_steps = load(args.backend, args.checkpoint, "libero_goal", args.seed)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    goals = load_goals("libero_goal")
    frames = []
    for tid in range(suite.n_tasks):
        env, lang = make_env(suite, tid)
        for s in range(args.n_scenes):
            np.random.seed(9_000_000 + args.seed * 100_000 + tid * 1000 + s)   # disjoint from eval seeds (settle)
            env.reset()
            obs = None
            for _ in range(10):
                obs, *_ = env.step(DUMMY_ACTION)

            def grab(t, o, tid=tid, s=s):
                if t % args.every == 0:
                    f = facts(env, o, (goals[tid], tid))
                    path = f"images/t{tid}-s{s}-{t:03d}.png"
                    Image.fromarray(np.ascontiguousarray(o["agentview_image"][::-1, ::-1])).save(out / path)
                    frames.append({"image": path, "task_id": tid, "scene": s, "t": t, "facts": f})
                return bool(satisfied(env, [goals[tid]])[0]) and t % args.every == 0
            r = rollout(env, policy, lang, obs, steps=max_steps, on_step=grab)
            if r["error"]:
                print(f"task {tid} scene {s}: {r['error']}", flush=True)
        env.close()
        print(f"task {tid}: {sum(f['task_id'] == tid for f in frames)} frames", flush=True)

    with open(out / "frames.jsonl", "w") as fh:
        for f in frames:
            fh.write(json.dumps(f) + "\n")
    (out / "run_config.json").write_text(json.dumps({
        **vars(args), "n_frames": len(frames),
        "git_langvla": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))


if __name__ == "__main__":
    main()
