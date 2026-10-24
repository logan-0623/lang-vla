"""RESEARCH_PLAN §11.8 A: does a rollout re-execute a demonstration? (V3; the trajectory-similarity idea of
ALT, arXiv 2505.05787, adapted to many demos per task.)

Post-grasp end-effector paths of branch runs (scripts/patch_branch.py records eef_path every 5 steps) vs the
post-grasp part of every LIBERO-Goal demonstration (LeRobot dataset: observation.state[:3] = eef xyz, same frame as
robot0_eef_pos; the grasp is the first gripper-close action, the segment starts once the eef has risen 3 cm).
Both are resampled to N points by arc length; d = mean point distance.

  per episode: nearest demo's task, d1, and ratio = d1 / (that task's typical demo-to-nearest-other-demo distance).
  ratio <= ~1: as close to a demo as demos are to each other (replay-like); >> 1: a path no demo took.
  S = 1 - d1 / d2 (ALT's score, d2 = second-nearest demo) is printed too, but with ~50 near-identical demos per task
  it stays near 0 even for replays: read the ratio.

  python scripts/traj_sim.py --demos $STORE/data/starvla_libero/libero_goal_no_noops_1.0.0_lerobot runs...
  python scripts/traj_sim.py plate runs...      plate xy at the episode starts / after a shift (no demos needed)
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

N = 50
LIFT = 0.03


def resample(p: np.ndarray, n: int = N) -> np.ndarray:
    """n points equally spaced in arc length along the polyline p [T, 3]."""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(p[:1], n, 0)
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, p[:, k]) for k in range(3)], 1)


def post_grasp(xyz: np.ndarray, grip_cmd: np.ndarray) -> np.ndarray | None:
    """Demo segment from 3 cm above the eef height at the first close command to the end. Demos start open, so a
    close is the first command far from the first one (the dataset's gripper convention does not matter)."""
    close = np.flatnonzero(np.abs(grip_cmd - grip_cmd[0]) > 0.5 * (np.ptp(grip_cmd) + 1e-9))
    if not len(close):
        return None
    c = close[0]
    up = np.flatnonzero(xyz[c:, 2] > xyz[c, 2] + LIFT)
    return xyz[c + up[0]:] if len(up) and len(xyz) - c - up[0] > 5 else None


def load_demos(root: Path) -> tuple[np.ndarray, list[str], dict[str, float]]:
    import pandas as pd
    tasks = {json.loads(l)["task_index"]: json.loads(l)["task"] for l in open(root / "meta/tasks.jsonl")}
    paths, names = [], []
    for f in sorted(root.glob("data/chunk-*/episode_*.parquet")):
        df = pd.read_parquet(f, columns=["observation.state", "action", "task_index"])
        xyz = np.stack(df["observation.state"].to_numpy())[:, :3]
        seg = post_grasp(xyz, np.stack(df["action"].to_numpy())[:, -1])
        if seg is not None:
            paths.append(resample(seg))
            names.append(tasks[int(df["task_index"].iloc[0])])
    P = np.stack(paths)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1).mean(-1)                 # demo x demo
    np.fill_diagonal(D, np.inf)
    typical = {t: float(np.median(D[[i for i, n in enumerate(names) if n == t]].min(1))) for t in set(names)}
    return P, names, typical


def sim(args) -> None:
    P, names, typical = load_demos(Path(args.demos))
    print(f"{len(P)} demos with a post-grasp segment; typical demo-to-demo distance (cm): "
          + ", ".join(f"{t[:28]} {v * 100:.1f}" for t, v in sorted(typical.items())))
    for d in args.runs:
        eps = [json.loads(l) for l in open(Path(d) / "episodes.jsonl")]
        eps = [e for e in eps if len(e.get("eef_path") or []) > 5]
        if not eps:
            print(f"{Path(d).name}: no eef_path")
            continue
        print(Path(d).name)
        for g in dict.fromkeys(e["goal"] for e in eps):
            rows = defaultdict(list)
            for e in (e for e in eps if e["goal"] == g):
                dist = np.linalg.norm(P - resample(np.asarray(e["eef_path"]))[None], axis=-1).mean(-1)
                i, j = np.argsort(dist)[:2]
                key = "success" if e["success"] else "fail"
                rows[key].append((names[i], dist[i], dist[i] / typical[names[i]], 1 - dist[i] / dist[j]))
            for key, r in rows.items():
                near = Counter(n for n, *_ in r).most_common(2)
                print(f"  {g} {key:<7} n={len(r):<2} nearest demo task {[(t[:30], c) for t, c in near]}  "
                      f"d1 {np.median([x[1] for x in r]) * 100:.1f} cm  ratio {np.median([x[2] for x in r]):.2f}  "
                      f"S {np.median([x[3] for x in r]):.2f}")


def plate(args) -> None:
    """Plate xy at the start of every episode (eval runs: start_pos) and where shift runs moved it."""
    for d in args.runs:
        eps = [json.loads(l) for l in open(Path(d) / "episodes.jsonl")]
        start = np.array([e["start_pos"]["plate_1_pos"][:2] for e in eps if "start_pos" in e])
        moved = np.array([e["shift"]["after"][:2] for e in eps if e.get("shift")])
        for name, x in (("start", start), ("after shift", moved)):
            if len(x):
                print(f"{Path(d).name} {name}: n={len(x)}  x [{x[:, 0].min():.3f}, {x[:, 0].max():.3f}]  "
                      f"y [{x[:, 1].min():.3f}, {x[:, 1].max():.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    if __import__("sys").argv[1:2] == ["plate"]:
        ap.add_argument("cmd")
        ap.add_argument("runs", nargs="+")
        plate(ap.parse_args())
        return
    ap.add_argument("--demos", required=True)
    ap.add_argument("runs", nargs="+")
    sim(ap.parse_args())


if __name__ == "__main__":
    main()
