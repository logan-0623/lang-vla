"""Does the StarVLA-4in1 policy *use* the gripper-object geometry that probes can read (RESEARCH_PLAN §11.6.3b)?
Linear erasure at the only features the OFT action head reads: the last-layer hidden states at the action tokens
(Qwenvl_OFT._gather_action_token_embeddings, [B, chunk, H]); closed-loop success on the seen LIBERO-Goal tasks
with the concept erased vs a random subspace of the same rank.

  serve  : `erase_action.py <server_policy.py> <its args>` wraps the StarVLA policy server (SV_WRAP in
           bridges2/run_starvla.sbatch). LV_RECORD=f appends every call's action queries (float16) to f;
           LV_ERASE=f.npz:name applies that eraser. The client (langeval/adapters/starvla.py) writes the matching
           labels with LV_LABELS, in the same call order.
  fit    : erasers from a recorded run. Per action-token position t, U_t = orthonormal basis of the cross-covariance
           Cov(x_t, z); x_t <- x_t - U_t U_t^T (x_t - mu_t) leaves no linear covariance with z (so no linear probe,
           on one token or all of them concatenated, beats the mean). Random control: variance-matched principal
           directions not aligned with the concept (`matched`), same rank. Concepts:
             eef  : gripper xyz (rank 3; OFT gets no proprio, so it must come from the images)
             dist : |eef - object| for the 4 movable objects (rank 4; the quantity of the counter-heuristic test)
             geom : eef xyz and eef - object vectors (rank 15)
             grip : finger opening (rank 1; action-relevant but not a spatial relation)
           Prints held-out-episode R^2 (linear and kNN) before / after each eraser and the fraction of ||x - mu|| removed.

  sbatch --export=ALL,SV_CKPT=...,SV_WRAP=$PWD/scripts/erase_action.py,LV_RECORD=$P/rec.f16,LV_LABELS=$P/rec.jsonl \
    bridges2/run_starvla.sbatch python scripts/eval_goals.py --backend starvla --checkpoint SERVER --sets seen ...
  python scripts/erase_action.py fit --rec $P/rec.f16 --labels $P/rec.jsonl --run_config $P/none/run_config.json --out $P/erasers.npz
  binding : (RESEARCH_PLAN §11.7) with LV_SWAP, the recorded run also sent every 10th observation with all 19
           instructions; for each novel instruction, the nearest seen goal at the action-head input / in the actions,
           vs the goal its failed episodes executed.
  python scripts/erase_action.py binding --rec $P/rec.f16 --labels $P/rec.jsonl --run_dir $P/none
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys

import numpy as np

OBJ = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1"]
CONCEPTS = ("eef", "dist", "geom", "grip")
N_RANDOM = 2
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]


def serve(argv: list[str]) -> None:
    import torch
    from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT

    gather = Qwenvl_OFT._gather_action_token_embeddings
    rec, erase = os.environ.get("LV_RECORD"), os.environ.get("LV_ERASE")
    state = {}

    def hooked(self, *a, **k):
        q = gather(self, *a, **k)                               # [B, chunk, H]
        if rec:
            if "f" not in state:
                state["f"] = open(rec, "ab")
                json.dump({"shape": list(q.shape[1:])}, open(rec + ".json", "w"))
            state["f"].write(q.float().cpu().numpy().astype(np.float16).tobytes())
            state["f"].flush()
        if erase:
            if "U" not in state:
                path, name = erase.rsplit(":", 1)
                z = np.load(path)
                state["U"] = torch.tensor(z[f"{name}.U"], device=q.device)       # [chunk, H, k]
                state["mu"] = torch.tensor(z["mu"], device=q.device)            # [chunk, H]
                print(f"erasing {name}: U {tuple(state['U'].shape)}", flush=True)
            U, mu = state["U"], state["mu"]
            x = q.float() - mu
            q = (q.float() - torch.einsum("thk,btk->bth", U, torch.einsum("thk,bth->btk", U, x))).to(q.dtype)
        return q

    Qwenvl_OFT._gather_action_token_embeddings = hooked
    sys.argv = argv
    runpy.run_path(argv[0], run_name="__main__")


def load(rec: str, labels: str) -> tuple[np.ndarray, list[dict], dict[str, np.ndarray]]:
    shape = json.load(open(rec + ".json"))["shape"]
    X = np.fromfile(rec, np.float16).reshape(-1, *shape).astype(np.float32)
    L = [json.loads(l) for l in open(labels)]
    assert len(L) == len(X), f"{len(X)} server records vs {len(L)} label lines: not the same run?"
    eef = np.array([l["robot0_eef_pos"] for l in L])
    rel = np.stack([np.array([l[f"{o}_pos"] for l in L]) - eef for o in OBJ], 1)          # [n, 4, 3]
    grip = np.array([l["robot0_gripper_qpos"] for l in L])
    Z = {"eef": eef, "dist": np.linalg.norm(rel, axis=2), "geom": np.concatenate([eef, rel.reshape(len(L), -1)], 1),
         "grip": (grip[:, :1] - grip[:, 1:2])}                                               # finger opening
    return X, L, Z


def basis(C: np.ndarray) -> np.ndarray:
    """[chunk, H, k] cross-covariances -> orthonormal bases of their column spans."""
    return np.stack([np.linalg.qr(c)[0] for c in C])


def matched(Xc: np.ndarray, U: np.ndarray, rng) -> np.ndarray:
    """Random control for one token position: for each concept direction, a principal direction of the records with
    nearly the same variance (one of the 5 nearest in log variance) that is not aligned with the concept subspace
    (|cos| < 0.3 to every concept direction). Same rank and about the same removed variance as the concept eraser."""
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    lv = np.log(S ** 2 / len(Xc) + 1e-12)
    ok = np.abs(Vt @ U).max(1) < 0.3
    pick = []
    for v in np.log(((Xc @ U) ** 2).mean(0) + 1e-12):
        cand = [j for j in np.argsort(np.abs(lv - v)) if ok[j] and j not in pick][:5]
        pick.append(int(rng.choice(cand)))
    return Vt[pick].T


def erasers(X: np.ndarray, z: np.ndarray, rng) -> tuple[np.ndarray, list[np.ndarray]]:
    Xc, zc = X - X.mean(0), z - z.mean(0)
    U = basis(np.einsum("nth,nk->thk", Xc, zc) / len(X))
    R = [np.stack([matched(Xc[:, t], U[t], rng) for t in range(U.shape[0])]) for _ in range(N_RANDOM)]
    return U, R


def apply(X: np.ndarray, mu: np.ndarray, U: np.ndarray) -> np.ndarray:
    x = X - mu
    return X - np.einsum("thk,ntk->nth", U, np.einsum("thk,nth->ntk", U, x))


def r2(Xtr, ztr, Xte, zte) -> float:
    """Ridge on all action tokens concatenated (dual form, n << chunk*H); R^2 against the train mean, at the best
    penalty *on the held-out episodes*: optimistic for every row, so conservative for "the eraser removed it"."""
    A, B = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    m, s = A.mean(0), ztr.mean(0)
    A, B = A - m, B - m
    s2, E = np.linalg.eigh(A @ A.T)
    Ey, Bk = E.T @ (ztr - s), B @ A.T @ E
    best = -np.inf
    for alpha in ALPHAS:
        pred = Bk @ (Ey / (np.clip(s2, 0, None) + alpha)[:, None]) + s
        best = max(best, float(1 - ((pred - zte) ** 2).sum() / ((zte - s) ** 2).sum()))
    return best


def knn_r2(Xtr, ztr, Xte, zte, k=10) -> float:
    """Nonlinear check: k-nearest-neighbour regression on the concatenated tokens. Linear erasure leaves nonlinear
    information; a high value here means the action head could still recover the concept."""
    A, B = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    d = (B ** 2).sum(1)[:, None] - 2 * B @ A.T + (A ** 2).sum(1)[None]
    pred = ztr[np.argsort(d, 1)[:, :k]].mean(1)
    return float(1 - ((pred - zte) ** 2).sum() / ((zte - ztr.mean(0)) ** 2).sum())


def fit(a) -> None:
    X, L, Z = load(a.rec, a.labels)
    cfg = json.load(open(a.run_config))
    seen = {l for l, n in zip(cfg["goal_language"], cfg["goal_names"]) if n.startswith("s")}
    keep = np.array(["swap_of" not in l and l["lang"] in seen for l in L])       # executed calls, seen goals only
    X, ep, Z = X[keep], np.array([l["episode"] for l in L])[keep], {c: z[keep] for c, z in Z.items()}
    rng = np.random.default_rng(a.seed)
    te = ep % 5 == 4                                            # readability check: held-out episodes
    mu = X.mean(0)
    out = {"mu": mu}
    print(f"{len(X)} calls, {len(set(ep))} episodes, action queries {X.shape[1:]}")
    for c in CONCEPTS:
        z = Z[c]
        U, R = erasers(X[~te], z[~te], rng)                     # check: erasers from train episodes only
        mtr = X[~te].mean(0)
        line = (f"{c:<5} rank {z.shape[1]:>2}  held-out R2 linear/kNN: none {r2(X[~te], z[~te], X[te], z[te]):.3f}/"
                f"{knn_r2(X[~te], z[~te], X[te], z[te]):.3f}")
        for nm, V in [("concept", U)] + [(f"random{i}", V) for i, V in enumerate(R)]:
            Xe = apply(X, mtr, V)
            frac = np.linalg.norm(X - Xe, axis=2).mean() / np.linalg.norm(X - mtr, axis=2).mean()
            line += (f" | {nm} {r2(Xe[~te], z[~te], Xe[te], z[te]):.3f}/{knn_r2(Xe[~te], z[~te], Xe[te], z[te]):.3f}"
                     f" (removes {frac:.3f} of |x-mu|)")
        print(line, flush=True)
        U, R = erasers(X, z, rng)                               # served erasers: all episodes
        out[f"{c}.U"] = U.astype(np.float32)
        for i, V in enumerate(R):
            out[f"{c}-random{i}.U"] = V.astype(np.float32)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {sorted(k for k in out if k != 'mu')}")


def nearest_seen(V: np.ndarray, names: list[str], seen: list[str]) -> dict[str, str]:
    """V [goals, ...] at one observation: for each goal, the most similar *other* seen goal (cosine after removing
    the mean over all instructions, i.e. what the instruction adds at this observation)."""
    F = V.reshape(len(V), -1)
    F = F - F.mean(0)
    F = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    S = F @ F.T
    si = [names.index(s) for s in seen]
    return {n: seen[int(np.argmax([S[i, j] if j != i else -np.inf for j in si]))] for i, n in enumerate(names)}


def binding(a) -> None:
    """RESEARCH_PLAN §11.7 B1-B2: at the same observation, which seen goal does a novel instruction look like at the
    action-head input (and in the predicted actions), and is that the goal the novel episode actually executed?"""
    from collections import Counter, defaultdict
    from pathlib import Path
    X, L, _ = load(a.rec, a.labels)
    cfg = json.load(open(Path(a.run_dir) / "run_config.json"))
    name = dict(zip(cfg["goal_language"], cfg["goal_names"]))
    names, seen = cfg["goal_names"], [n for n in cfg["goal_names"] if n.startswith("s")]
    novel = [n for n in names if n not in seen]
    eps = [json.loads(l) for l in open(Path(a.run_dir) / "episodes.jsonl")]
    groups = defaultdict(dict)                                   # (episode, call) -> goal -> row
    for i, l in enumerate(L):
        if l["call"] % 10 == 0:                                  # SWAP_EVERY in langeval/adapters/starvla.py
            groups[l["episode"], l["call"]][name[l["lang"]]] = i
    groups = {k: g for k, g in groups.items() if len(g) == len(names)}
    words = {n: set(l.split()) for l, n in name.items()}
    by_text = {n: max((s for s in seen if s != n), key=lambda s: len(words[n] & words[s]) / len(words[n] | words[s]))
               for n in names}
    print(f"{len(groups)} complete instruction-swap groups ({len(names)} instructions each)")
    for space in ("state", "action"):
        near = {k: nearest_seen(np.stack([X[g[n]] if space == "state" else np.array(L[g[n]]["action"]) for n in names]),
                                names, seen) for k, g in groups.items()}
        print(f"\n[{space}] novel goal: most common nearest seen goal over all observations | text-nearest | "
              f"failed episodes of that goal: nearest at its first call == executed")
        hit = tot = 0
        for n in novel:
            mode = Counter(nr[n] for nr in near.values()).most_common(2)
            fails = [(i, e) for i, e in enumerate(eps) if e["goal"] == n and not e["success"] and (i, 0) in near]
            h = sum(near[i, 0][n] == e["executed"] for i, e in fails)
            hit, tot = hit + h, tot + len(fails)
            print(f"  {n:<4} {' '.join(f'{s} {c / len(near):.2f}' for s, c in mode):<20} | {by_text[n]:<3} | "
                  f"{h}/{len(fails)}  executed: {dict(Counter(str(e['executed']) for _, e in fails))}")
        print(f"  B{1 if space == 'state' else 2}: nearest == executed in {hit}/{tot} failed novel episodes"
              f" ({hit / max(tot, 1):.2f}; chance about 0.1)")


def main() -> None:
    if sys.argv[1:2] and sys.argv[1].endswith(".py"):      # wrapping the policy server
        return serve(sys.argv[1:])
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--rec", required=True)
    f.add_argument("--labels", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--run_config", required=True, help="run_config.json of the recorded eval_goals run")
    b = sub.add_parser("binding")
    b.add_argument("--rec", required=True)
    b.add_argument("--labels", required=True)
    b.add_argument("--run_dir", required=True, help="the recorded eval_goals --out dir")
    a = ap.parse_args()
    fit(a) if a.cmd == "fit" else binding(a)


if __name__ == "__main__":
    main()
