"""Counter-heuristic state pairs for the gripper-target relation (RESEARCH_PLAN §11.6.2-11.6.3).

In the policy-driven QA frames, simple image cues and the true relation agree: a gripper drawn over an object is
near it, a closed gripper usually holds something. A probe trained on those frames, or a VLM answering, can score
by the cue. Here the cue is held fixed and the relation is varied, with the arm moved by short OSC actions (no
teleporting) from one settled parent state per scene; every label is read back from the simulator afterwards,
never taken from the intended manipulation.

  ray  : the gripper (open) is put on the camera ray through one movable object, at RAY_T metres from it toward
         the camera. All points of the ray project to the object's pixel, so the 2D gripper-object distance stays
         ~0 while the 3D distance grows. consistent = drawn over the object and near (<= 10 cm); counter = drawn
         over it but far (>= 17 cm). Within a parent, a 2D shortcut cannot order the frames by distance.
  grip : the gripper descends to the object (offset 0) or beside it (offset GRIP_DX), closes, lifts LIFT. Closed in
         every frame; consistent = holding the object (is_grasped after the lift), counter = closed and empty.

Every frame keeps the full `facts` of make_spatial_qa (so probe_vlm.py extract / eval_qa.py run on the dir as
is) plus `cf`: kind, parent, object, ray t, reach error, 3D and 2D gripper-object distance, contact, held.
All frames are test frames (scene = parent): probes are trained on goal-v1 and only applied here.

  python scripts/make_counter_pairs.py collect --n_parents 80 --out $STORE/qa/counter-v1       (vla env, ~20 min)
  python scripts/probe_vlm.py extract --base ... --qa_dir $STORE/qa/counter-v1 --out $RESULTS/probe/counter-raw.npz
  python scripts/eval_qa.py --base ... --qa_dir $STORE/qa/counter-v1 --max_images 100000 --out $RESULTS/qa/counter-raw
  python scripts/make_counter_pairs.py analyze --train_dir $STORE/qa/goal-v1 --test_dir $STORE/qa/counter-v1 \\
      --feats raw=$RESULTS/probe/raw.npz,$RESULTS/probe/counter-raw.npz --qa $RESULTS/qa/counter-raw --pixels
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parents[1])]      # scripts/, and langeval for make_spatial_qa
from make_spatial_qa import OBJECTS, H, W, name                                    # noqa: E402

RAY_T = (0.06, 0.12, 0.19, 0.26)        # metres from the object toward the camera
GRIP_DX = 0.06                          # lateral miss for the closed-but-empty grasp
LIFT = 0.06
NEAR, FAR, OVER_PX = 0.10, 0.17, 12     # group thresholds (written before any result)
SHORT = {"akita_black_bowl_1": "bowl", "cream_cheese_1": "cheese", "wine_bottle_1": "bottle", "plate_1": "plate"}


# ------------------------------------------------------------------ simulation (vla env, LIBERO)

def _eef(obs):
    return np.array(obs["robot0_eef_pos"], float)


def move_to(env, obs, target, grip: float, steps: int = 150, tol: float = 0.008):
    """P-control of the OSC position delta toward `target`; returns (obs, final distance to target)."""
    for _ in range(steps):
        d = np.asarray(target) - _eef(obs)
        if np.linalg.norm(d) < tol:
            break
        obs, *_ = env.step([*np.clip(10 * d, -1, 1), 0, 0, 0, grip])
    return obs, float(np.linalg.norm(np.asarray(target) - _eef(obs)))


def hold(env, obs, grip: float, n: int):
    for _ in range(n):
        obs, *_ = env.step([0, 0, 0, 0, 0, 0, grip])
    return obs


def frame_extras(env, obs, obj: str) -> dict:
    from robosuite.utils.camera_utils import get_camera_transform_matrix, project_points_from_world_to_camera
    from langeval.phase import is_grasped
    inner = env.env
    m = get_camera_transform_matrix(inner.sim, "agentview", H, W)
    eef, p = _eef(obs), np.array(inner.sim.data.body_xpos[inner.obj_body_id[obj]])
    rc = project_points_from_world_to_camera(np.stack([eef, p]), m, H, W)
    return {"eef_rc": [int(v) for v in rc[0]], "px2d": float(np.linalg.norm(rc[0] - rc[1])),
            "dist3d": float(np.linalg.norm(eef - p)), "held": bool(is_grasped(env, obj)),
            "contact": bool(inner.check_contact(inner.robots[0].gripper, inner.objects_dict[obj]))}


def collect(a) -> None:
    from PIL import Image
    from libero.libero import benchmark
    from langeval.adapters.starvla import make_env
    from langeval.branch import DUMMY_ACTION, restore, snapshot
    from langeval.goal_predicates import load_goals
    from make_spatial_qa import facts

    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=False)
    (out / "llava_jsons").mkdir()
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    goals = load_goals("libero_goal")
    env, _ = make_env(suite, 0)                      # LIBERO-Goal tasks share one scene; task 0's goal is unused
    cam = env.env.sim.model.camera_name2id("agentview")
    rng = np.random.default_rng(a.seed)
    frames = []

    def save(obs, parent, kind, obj, **kw):
        path = f"images/p{parent:03d}-{kind}-{len(frames):04d}.png"
        Image.fromarray(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])).save(out / path)
        cf = {"kind": kind, "parent": parent, "object": obj, **kw, **frame_extras(env, obs, obj)}
        frames.append({"image": path, "task_id": 0, "scene": parent, "t": len(frames),
                       "facts": facts(env, obs, (goals[0], 0)), "cf": cf})

    for parent in range(a.n_parents):
        np.random.seed(8_000_000 + a.seed * 100_000 + parent)                  # disjoint from QA / eval seeds
        env.reset()
        for _ in range(10):
            obs, *_ = env.step(DUMMY_ACTION)
        snap = snapshot(env)
        inner = env.env
        obj = list(OBJECTS)[rng.integers(len(OBJECTS))]
        p0 = np.array(inner.sim.data.body_xpos[inner.obj_body_id[obj]])
        u = np.array(inner.sim.data.cam_xpos[cam]) - p0
        u /= np.linalg.norm(u)
        for t in RAY_T:
            obs = restore(env, snap)
            obs, err = move_to(env, obs, p0 + t * u, -1)
            obs = hold(env, obs, -1, 5)
            save(obs, parent, "ray", obj, ray_t=t, reach_err=err)
        for dx in (0.0, GRIP_DX):
            obs = restore(env, snap)
            above = p0 + [0, dx, 0.12]
            obs, e1 = move_to(env, obs, above, -1)
            obs, e2 = move_to(env, obs, p0 + [0, dx, 0.01], -1, steps=80)
            obs = hold(env, obs, 1, 15)
            obs, _ = move_to(env, obs, _eef(obs) + [0, 0, LIFT], 1, steps=40)
            obs = hold(env, obs, 1, 5)
            save(obs, parent, "grip", obj, grip_dx=dx, reach_err=max(e1, e2))
        if parent % 10 == 0:
            print(f"parent {parent}: {len(frames)} frames", flush=True)
    env.close()
    with open(out / "frames.jsonl", "w") as fh:
        for f in frames:
            fh.write(json.dumps(f) + "\n")
    write_qa(out, frames)
    overlay(out, frames[:: max(1, len(frames) // 12)][:12])
    (out / "run_config.json").write_text(json.dumps({**vars(a), "n_frames": len(frames), "RAY_T": RAY_T,
                                                     "GRIP_DX": GRIP_DX, "LIFT": LIFT}, indent=2))
    print(f"wrote {len(frames)} frames to {out}")


def write_qa(out: Path, frames: list[dict]) -> None:
    """One question per frame, in make_spatial_qa's lang templates, so eval_qa.py scores it unchanged."""
    data = []
    for f in frames:
        x, cf = f["facts"], f["cf"]
        o = cf["object"]
        if cf["kind"] == "ray":
            dx, dy, dz = (100 * (np.array(x["pos"][o]) - np.array(x["eef"]))).round().astype(int).tolist()
            d = round(float(np.linalg.norm([dx, dy, dz])))
            q, ans = (f"How far is the gripper from the {name(o)}?",
                      f"About {d} cm. The {name(o)} is {abs(dz)} cm {'below' if dz < 0 else 'above'} the gripper.")
        else:
            q, ans = "What is the gripper holding?", f"The {name(x['held'])}." if x["held"] else "Nothing."
        data.append({"image": f["image"], "scene": f["scene"], "task_id": 0,
                     "conversations": [{"from": "human", "value": "<image>\n" + q}, {"from": "gpt", "value": ans}]})
    (out / "llava_jsons" / "libero_qa_lang_heldout.json").write_text(json.dumps(data))


def overlay(out: Path, frames: list[dict]) -> None:
    """Gripper (red) and object (lime) projections on the saved images, in the `col` convention: check that a ray
    frame has the two on top of each other."""
    from PIL import Image, ImageDraw
    (out / "overlay").mkdir(exist_ok=True)
    for f in frames:
        img = Image.open(out / f["image"]).convert("RGB").resize((4 * W, 4 * H), Image.NEAREST)
        dr = ImageDraw.Draw(img)
        for (r, c), col in ((f["cf"]["eef_rc"], "red"), (f["facts"]["rc"][f["cf"]["object"]], "lime")):
            u, v = 4 * (W - 1 - c), 4 * r
            dr.ellipse([u - 8, v - 8, u + 8, v + 8], outline=col, width=3)
        dr.text((8, 8), f"{f['cf']['kind']} {f['cf']['object']} 3d {f['cf']['dist3d']:.2f} m held {f['cf']['held']}", fill="white")
        img.save(out / "overlay" / Path(f["image"]).name)


# ------------------------------------------------------------------ analysis (CPU)

def group(cf: dict) -> str | None:
    """consistent / counter (cue agrees / disagrees with the truth), or None (in between, or the arm missed)."""
    if cf["reach_err"] > 0.02:
        return None
    if cf["kind"] == "ray":
        if cf["px2d"] > OVER_PX:
            return None
        return "consistent" if cf["dist3d"] <= NEAR else "counter" if cf["dist3d"] >= FAR else None
    return "consistent" if cf["held"] else "counter"


def pair_order(parent: np.ndarray, true: np.ndarray, pred: np.ndarray) -> float:
    """Within-parent pairwise ordering accuracy of `pred` against `true` (0.5 = no ordering, e.g. a 2D cue)."""
    ok = []
    for p in np.unique(parent):
        i = np.flatnonzero(parent == p)
        ok += [(pred[j] - pred[k]) * (true[j] - true[k]) > 0 for j, k in combinations(i, 2) if true[j] != true[k]]
    return float(np.mean(ok)) if ok else float("nan")


def fit_predict(Xtr, ytr, kind, va, Xte):
    """Ridge on the goal-v1 frames (penalty picked on the validation scenes `va`); returns (val score, predictions)."""
    from probe_vlm import ALPHAS, Ridge, score, target
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Y = target(kind, ytr)
    r_fit, r_all = Ridge(Xtr[~va]), Ridge(Xtr)
    m_fit, m_all = Y[~va].mean(0), Y.mean(0)
    vs = {al: score(kind, Xtr[va] @ r_fit.coef(Y[~va] - m_fit, al) + m_fit, ytr[va], m_fit) for al in ALPHAS}
    best = max(vs, key=vs.get)
    return vs[best], Xte @ r_all.coef(Y - m_all, best) + m_all


def analyze(a) -> None:
    from probe_vlm import frames as load_frames, labels
    tr_fr, te_fr = load_frames(Path(a.train_dir)), load_frames(Path(a.test_dir))
    tr_lab = labels(tr_fr)
    scene = np.array([f["scene"] for f in tr_fr])
    va = scene % 5 == 3                               # same validation scenes as probe_vlm fit; all scenes train
    cf = [f["cf"] for f in te_fr]
    grp = np.array([group(c) or "" for c in cf])
    kind = np.array([c["kind"] for c in cf])
    parent = np.array([c["parent"] for c in cf])
    d3 = np.array([c["dist3d"] for c in cf])
    held = np.array([c["held"] for c in cf])
    objs = np.array([c["object"] for c in cf])
    for k in ("ray", "grip"):
        print(f"{k}: " + "  ".join(f"{g} {int(((kind == k) & (grp == g)).sum())}" for g in ("consistent", "counter", ""))
              + "  (blank = between thresholds or reach error > 2 cm)")

    feats = {}
    if a.pixels:
        from PIL import Image
        px = lambda d, fr: np.stack([np.asarray(Image.open(Path(d) / f["image"]).convert("RGB").resize((32, 32), Image.BILINEAR), float).ravel() for f in fr])
        feats["pixels"] = [("-", px(a.train_dir, tr_fr), px(a.test_dir, te_fr))]
    for spec in a.feats:
        nm, paths = spec.split("=", 1)
        ztr, zte = (np.load(p) for p in paths.split(","))
        assert list(zte["images"]) == [f["image"] for f in te_fr], f"{paths}: test frames differ"
        for pool in ("mean", "grid"):
            A, B = ztr[pool], zte[pool]
            lay = ztr["grid_layers" if pool == "grid" else "layers"]
            feats[f"{nm}.{pool}"] = [(int(l), A[:, i].astype(float), B[:, i].astype(float)) for i, l in enumerate(lay)]

    res = {}
    for fn, per_layer in feats.items():
        best = {}
        for layer, Xtr, Xte in per_layer:              # layer picked per label on goal-v1 validation scenes
            for o in OBJECTS:
                ln = f"{SHORT[o]}_gripper_dist"
                v, P = fit_predict(Xtr, tr_lab[ln][1], "reg", va, Xte)
                if v > best.get(ln, (-np.inf,))[0]:
                    best[ln] = (v, layer, P[:, 0])
            y = tr_lab["held"][1]
            v, P = fit_predict(Xtr, y, "cls", va, Xte)
            if v > best.get("held", (-np.inf,))[0]:
                best["held"] = (v, layer, P.argmax(1))
        pred_d = np.array([best[f"{SHORT[o]}_gripper_dist"][2][i] for i, o in enumerate(objs)])
        pred_held = best["held"][2] == np.array([1 + list(OBJECTS).index(o) for o in objs])
        row = {}
        for g in ("consistent", "counter"):
            r = (kind == "ray") & (grp == g)
            gm = (kind == "grip") & (grp == g)
            row[g] = {"ray_n": int(r.sum()), "ray_true_cm": 100 * float(d3[r].mean()) if r.any() else None,
                      "ray_pred_cm": 100 * float(pred_d[r].mean()) if r.any() else None,
                      "ray_mae_cm": 100 * float(np.abs(pred_d[r] - d3[r]).mean()) if r.any() else None,
                      "grip_n": int(gm.sum()), "grip_acc": float((pred_held[gm] == held[gm]).mean()) if gm.any() else None}
        ray = (kind == "ray") & (np.array([c["reach_err"] for c in cf]) <= 0.02)
        row["ray_pair_order"] = pair_order(parent[ray], d3[ray], pred_d[ray])
        row["layers"] = {k: v[1] for k, v in best.items()}
        res[fn] = row
        c, x = row["consistent"], row["counter"]
        fmt = lambda v, s=".1f": "-" if v is None else format(v, s)
        print(f"{fn:<12} ray: consistent true {fmt(c['ray_true_cm'])} pred {fmt(c['ray_pred_cm'])} cm (mae {fmt(c['ray_mae_cm'])})"
              f" | counter true {fmt(x['ray_true_cm'])} pred {fmt(x['ray_pred_cm'])} cm (mae {fmt(x['ray_mae_cm'])})"
              f" | pair order {row['ray_pair_order']:.2f}   grip: held acc {fmt(c['grip_acc'], '.2f')} / closed-empty acc {fmt(x['grip_acc'], '.2f')}")

    if a.qa:
        from eval_qa import content_facts, qtype
        img_i = {f["image"]: i for i, f in enumerate(te_fr)}
        for d in a.qa:
            ok = {"ray": {}, "grip": {}}
            for l in open(Path(d) / "answers.jsonl"):
                x = json.loads(l)
                i = img_i[x["image"]]
                if grp[i]:
                    v = all(content_facts(x["pred"], x["ref"], qtype(x["q"])).values())
                    ok[kind[i]].setdefault(grp[i], []).append(v)
            s = {k: {g: (float(np.mean(v)), len(v)) for g, v in gs.items()} for k, gs in ok.items()}
            res[f"qa:{d}"] = s
            print(f"QA {Path(d).name}: " + "  ".join(f"{k}.{g} {m:.2f} (n={n})" for k, gs in s.items() for g, (m, n) in sorted(gs.items())))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("collect")
    c.add_argument("--n_parents", type=int, default=80)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", required=True)
    n = sp.add_parser("analyze")
    n.add_argument("--train_dir", required=True, help="QA dir the probes are trained on (goal-v1)")
    n.add_argument("--test_dir", required=True, help="the counter-pair dir")
    n.add_argument("--feats", nargs="*", default=[], help="name=train.npz,test.npz (probe_vlm extract outputs)")
    n.add_argument("--pixels", action="store_true")
    n.add_argument("--qa", nargs="*", default=[], help="eval_qa.py --out dirs run on the counter-pair dir")
    n.add_argument("--out", default=None)
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else analyze(a)


if __name__ == "__main__":
    main()
