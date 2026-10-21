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
           on one token or all of them concatenated, beats the mean). Random control: U_t from random combinations
           of the centred records (covariance-weighted directions, same rank). Concepts:
             dist : |eef - object| for the 4 movable objects (rank 4; the quantity of the counter-heuristic test)
             geom : eef xyz and eef - object vectors (rank 15)
           Prints held-out-episode R^2 before / after each eraser and the fraction of ||x - mu|| it removes.

  sbatch --export=ALL,SV_CKPT=...,SV_WRAP=$PWD/scripts/erase_action.py,LV_RECORD=$P/rec.f16,LV_LABELS=$P/rec.jsonl \
    bridges2/run_starvla.sbatch python scripts/eval_goals.py --backend starvla --checkpoint SERVER --sets seen ...
  python scripts/erase_action.py fit --rec $P/rec.f16 --labels $P/rec.jsonl --out $P/erasers.npz
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys

import numpy as np

OBJ = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1"]
CONCEPTS = ("dist", "geom")
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


def load(rec: str, labels: str) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    shape = json.load(open(rec + ".json"))["shape"]
    X = np.fromfile(rec, np.float16).reshape(-1, *shape).astype(np.float32)
    L = [json.loads(l) for l in open(labels)]
    assert len(L) == len(X), f"{len(X)} server records vs {len(L)} label lines: not the same run?"
    eef = np.array([l["robot0_eef_pos"] for l in L])
    rel = np.stack([np.array([l[f"{o}_pos"] for l in L]) - eef for o in OBJ], 1)          # [n, 4, 3]
    Z = {"dist": np.linalg.norm(rel, axis=2), "geom": np.concatenate([eef, rel.reshape(len(L), -1)], 1)}
    return X, np.array([l["episode"] for l in L]), Z


def basis(C: np.ndarray) -> np.ndarray:
    """[chunk, H, k] cross-covariances -> orthonormal bases of their column spans."""
    return np.stack([np.linalg.qr(c)[0] for c in C])


def erasers(X: np.ndarray, z: np.ndarray, rng) -> tuple[np.ndarray, list[np.ndarray]]:
    Xc, zc = X - X.mean(0), z - z.mean(0)
    U = basis(np.einsum("nth,nk->thk", Xc, zc) / len(X))
    R = [basis(np.einsum("nth,nk->thk", Xc, rng.standard_normal(zc.shape))) for _ in range(N_RANDOM)]
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


def fit(a) -> None:
    X, ep, Z = load(a.rec, a.labels)
    rng = np.random.default_rng(a.seed)
    te = ep % 5 == 4                                            # readability check: held-out episodes
    mu = X.mean(0)
    out = {"mu": mu}
    print(f"{len(X)} calls, {len(set(ep))} episodes, action queries {X.shape[1:]}")
    for c in CONCEPTS:
        z = Z[c]
        U, R = erasers(X[~te], z[~te], rng)                     # check: erasers from train episodes only
        mtr = X[~te].mean(0)
        line = f"{c:<5} rank {z.shape[1]:>2}  R2 held-out: none {r2(X[~te], z[~te], X[te], z[te]):.3f}"
        for nm, V in [("concept", U)] + [(f"random{i}", V) for i, V in enumerate(R)]:
            Xe = apply(X, mtr, V)
            frac = np.linalg.norm(X - Xe, axis=2).mean() / np.linalg.norm(X - mtr, axis=2).mean()
            line += f" | {nm} {r2(Xe[~te], z[~te], Xe[te], z[te]):.3f} (removes {frac:.3f} of |x-mu|)"
        print(line, flush=True)
        U, R = erasers(X, z, rng)                               # served erasers: all episodes
        out[f"{c}.U"] = U.astype(np.float32)
        for i, V in enumerate(R):
            out[f"{c}-random{i}.U"] = V.astype(np.float32)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {sorted(k for k in out if k != 'mu')}")


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
    fit(ap.parse_args())


if __name__ == "__main__":
    main()
