"""Does the StarVLA-4in1 policy *use* the gripper-object geometry that probes can read (RESEARCH_PLAN §11.6.3b)?
Linear erasure at the only features the OFT action head reads: the last-layer hidden states at the action tokens
(Qwenvl_OFT._gather_action_token_embeddings, [B, chunk, H]); closed-loop success on the seen LIBERO-Goal tasks
with the concept erased vs a random subspace of the same rank.

  serve  : `erase_action.py <server_policy.py> <its args>` wraps the StarVLA policy server (SV_WRAP in
           bridges2/run_starvla.sbatch). LV_RECORD=f appends every call's action queries (float16) to f;
           LV_ERASE=f.npz:name[:alpha] applies that eraser (alpha < 1: only that fraction of the projection). The
           client (langeval/adapters/starvla.py) writes the matching labels with LV_LABELS, in the same call order.
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
    rec, erase, patch = os.environ.get("LV_RECORD"), os.environ.get("LV_ERASE"), os.environ.get("LV_PATCH_U")
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
                path, name, *alpha = erase.split(":")
                state["alpha"] = float(alpha[0]) if alpha else 1.0          # partial erasure (dose matching)
                z = np.load(path)
                state["U"] = torch.tensor(z[f"{name}.U"], device=q.device)       # [chunk, H, k]
                state["mu"] = torch.tensor(z["mu"], device=q.device)            # [chunk, H]
                print(f"erasing {name} x{state['alpha']}: U {tuple(state['U'].shape)}", flush=True)
            U, mu = state["U"], state["mu"]
            x = q.float() - mu
            proj = torch.einsum("thk,btk->bth", U, torch.einsum("thk,bth->btk", U, x))
            q = (q.float() - state["alpha"] * proj).to(q.dtype)
        if patch and q.shape[0] == 2:                           # batch [A, donor B] from the client's LV_PATCH
            if "P" not in state:
                # path:none | path:full:strength | path:name:rank:strength (subspaces2.npz keeps rank 64: no default)
                path, name, *opt = patch.split(":")
                if name != "none" and len(opt) != (1 if name == "full" else 2):
                    raise SystemExit(f"LV_PATCH_U={patch}: give {'strength' if name == 'full' else 'rank:strength'}")
                state["beta"] = float(opt[-1]) if opt else 1.0
                state["P"] = name if name in ("none", "full") else \
                    torch.tensor(np.load(path)[f"{name}.U"][:, :, :int(opt[0])], device=q.device)
                print(f"patching {name} {':'.join(opt)}", flush=True)
            P, q, beta = state["P"], q.clone(), state["beta"]
            d = q[1].float() - q[0].float()
            if P == "full" and beta == 1.0:
                q[0] = q[1]
            elif isinstance(P, str):
                q[0] = (q[0].float() + beta * d).to(q.dtype) if P == "full" else q[0]
            else:
                q[0] = (q[0].float() + beta * torch.einsum("thk,tk->th", P, torch.einsum("thk,th->tk", P, d))).to(q.dtype)
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


def seen_calls(a) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """The calibration states: executed calls of the seen goals (no instruction swaps, no novel goals)."""
    X, L, Z = load(a.rec, a.labels)
    cfg = json.load(open(a.run_config))
    seen = {l for l, n in zip(cfg["goal_language"], cfg["goal_names"]) if n.startswith("s")}
    keep = np.array(["swap_of" not in l and l["lang"] in seen for l in L])
    return X[keep], np.array([l["episode"] for l in L])[keep], {c: z[keep] for c, z in Z.items()}


def alphas(a) -> None:
    """Dose matching for batch_1022: alpha = mean removed fraction of the two random erasers / that of the concept
    eraser, with the *served* erasers (erasers.npz: U and mu fit on all seen calls) on those same calls. The removed
    norm is linear in alpha, so alpha x concept then removes what a random eraser removes."""
    X, _, _ = seen_calls(a)
    z = np.load(a.erasers)
    norm = np.linalg.norm(X - z["mu"], axis=2).mean()
    frac = lambda nm: float(np.linalg.norm(X - apply(X, z["mu"], z[f"{nm}.U"]), axis=2).mean() / norm)
    out = {}
    for c in CONCEPTS:
        r = np.mean([frac(f"{c}-random{i}") for i in range(N_RANDOM)])
        out[c] = round(r / frac(c), 3)
        print(f"{c:<5} concept removes {frac(c):.3f}, random {r:.3f} -> alpha {out[c]}")
    json.dump(out, open(a.out, "w"))


def fit(a) -> None:
    X, ep, Z = seen_calls(a)
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
        reach = sum(e["executed"] in seen for i, e in enumerate(eps) if e["goal"] in novel and not e["success"]
                    and (i, 0) in near)
        print(f"  B{1 if space == 'state' else 2}: nearest == executed in {hit}/{tot} failed novel episodes"
              f" ({hit / max(tot, 1):.2f}; chance about 0.1; ceiling {reach}/{tot}: only seen goals can match)")
        # Exploratory (after the B1-B2 results, 2026-10-22): failures that completed a *seen* goal only, and the
        # nearest seen goal over each novel episode's own observations by progress (call 0 / 10-20 / 30+).
        own = [(i, e) for i, e in enumerate(eps) if e["goal"] in novel and not e["success"] and e["executed"] in seen]
        h = sum(near[i, 0][e["goal"]] == e["executed"] for i, e in own if (i, 0) in near)
        top = Counter(e["executed"] for _, e in own).most_common(1)[0][0] if own else None
        print(f"  conditional (follow-up, not B1/B2): failures that completed a seen goal: nearest at first call =="
              f" executed {h}/{len(own)}; baselines on the same episodes: text-nearest (Jaccard)"
              f" {sum(by_text[e['goal']] == e['executed'] for _, e in own)}/{len(own)}, most frequent executed goal"
              f" ({top}, in-sample) {sum(e['executed'] == top for _, e in own)}/{len(own)}")
        for n in novel:
            bins = {b: Counter() for b in ("0", "10-20", "30+")}
            for (i, c), nr in near.items():
                if eps[i]["goal"] == n:
                    bins["0" if c == 0 else "10-20" if c <= 20 else "30+"][nr[n]] += 1
            print(f"    {n:<4} by progress: " + " | ".join(
                f"{b}: " + " ".join(f"{s} {k}" for s, k in cnt.most_common(2)) for b, cnt in bins.items() if cnt))
    moved = defaultdict(list)                                    # which movable object each novel episode moved
    for ep_i in sorted({l["episode"] for l in L}):
        rows = [l for l in L if l["episode"] == ep_i and "swap_of" not in l]
        d = {o: np.linalg.norm(np.subtract(rows[-1][f"{o}_pos"], rows[0][f"{o}_pos"])) for o in OBJ}
        if eps[ep_i]["goal"] in novel:
            o, m = max(d.items(), key=lambda kv: kv[1])
            moved[eps[ep_i]["goal"]].append(o.split("_")[0] if m > 0.05 else "none")
    print("\nobject moved > 5 cm (largest displacement, first to last call) in each novel goal's episodes:")
    for n in novel:
        print(f"  {n:<4} {cfg['goal_language'][names.index(n)]:<52} {dict(Counter(moved[n]))}")


# §11.7 P0-P3: instruction sets that differ only in the object (same destination) / only in the destination (same
# object). n2 (cream cheese -> cabinet top, executed as bowl -> cabinet top) is the test case, never used to fit.
OBJ_SETS = [["s2", "s4"], ["s1", "n1", "n4"], ["s8", "n0", "n3"]]
DEST_SETS = [["s1", "s3", "s4", "s8", "n5"], ["s6", "n0", "n1"], ["s2", "s9", "n3", "n4"]]
K_MAX = 64
RANDOM_BG = 128


def swap_groups(L: list[dict], names: dict[str, str]) -> list[dict[str, int]]:
    """(episode, call) groups of the er-record run in which every instruction was sent with one observation."""
    from collections import defaultdict
    g = defaultdict(dict)
    for i, l in enumerate(L):
        if l["call"] % 10 == 0:                                  # SWAP_EVERY in langeval/adapters/starvla.py
            g[l["episode"], l["call"]][names[l["lang"]]] = i
    return [v for v in g.values() if len(v) == len(set(names.values()))]


def top_basis(D: np.ndarray, k: int) -> np.ndarray:
    """D [n, chunk, H] -> per token position the top-k left singular directions, [chunk, H, k] (eigenvectors of the
    H x H Gram matrix: the same directions as the SVD, without computing all n right singular vectors)."""
    out = []
    for t in range(D.shape[1]):
        A = D[:, t].astype(np.float64)
        w, V = np.linalg.eigh(A.T @ A)
        out.append(V[:, ::-1][:, :k])
    return np.stack(out)


def contrast_basis(Dp: np.ndarray, Dn: np.ndarray, k: int, lam: float = 0.1) -> np.ndarray:
    """Per token, the k directions with the most variance of Dp relative to Dn (generalized eigenvectors of
    Cov(Dp) v = l (Cov(Dn) + lam * mean eig * I) v), orthonormalised: e.g. what changes with the object but not with
    the destination. [chunk, H, k]."""
    out = []
    for t in range(Dp.shape[1]):
        A, B = Dp[:, t].astype(np.float64), Dn[:, t].astype(np.float64)
        Cp, Cn = A.T @ A / len(A), B.T @ B / len(B)
        w, V = np.linalg.eigh(Cn)
        W = V / np.sqrt(np.clip(w, 0, None) + lam * w.clip(0).mean())               # Cn_reg^{-1/2}
        g, G = np.linalg.eigh(W.T @ Cp @ W)
        out.append(np.linalg.qr(W @ G[:, ::-1][:, :k])[0])
    return np.stack(out)


def subspaces(a) -> None:
    """Object and destination subspaces of the action queries from the instruction-swap groups: what changes when only
    the object (only the destination) of the instruction changes at one observation. Saved at rank K_MAX; the server
    takes the first k columns (nested ranks) and must be told k:
      obj, dest     : top principal directions of the object (destination) differences
      obj_c, dest_c : contrastive, object differences relative to destination differences and vice versa (a soft
                      preference, not orthogonality: check the printed leakage per rank)
      random        : a random rotation of the top RANDOM_BG principal directions of all instruction-induced variation
                      (batch_1023 used 32 with rank 4)
    --exclude: goals left out of fitting entirely, both their instructions and the observations of their episodes
    (held-out *combinations*). The remaining groups are split by episode (% 5 == 4 held out); the saved subspaces are
    the ones fit on the train split, i.e. the ones whose held-out shares are printed. Split recorded in the npz."""
    from pathlib import Path
    X, L, _ = load(a.rec, a.labels)
    cfg = json.load(open(Path(a.run_dir) / "run_config.json"))
    goal_of = [e["goal"] for e in map(json.loads, open(Path(a.run_dir) / "episodes.jsonl"))]
    name = dict(zip(cfg["goal_language"], cfg["goal_names"]))
    excl = set(a.exclude.split(",")) - {""}
    G = [g for g in swap_groups(L, name) if goal_of[L[next(iter(g.values()))]["episode"]] not in excl]
    ep = lambda g: L[next(iter(g.values()))]["episode"]
    keep = lambda sets: [st for st in ([n for n in st if n not in excl] for st in sets) if len(st) > 1]
    osets, dsets = keep(OBJ_SETS), keep(DEST_SETS)
    alln = keep([[n for n in cfg["goal_names"] if n != "n2"]])
    diffs = lambda G, sets: np.concatenate([X[[g[n] for n in st]] - X[[g[n] for n in st]].mean(0)
                                            for g in G for st in sets])
    rng = np.random.default_rng(a.seed)
    tr, te = [g for g in G if ep(g) % 5 != 4], [g for g in G if ep(g) % 5 == 4]
    Do, Dd, Da = diffs(tr, osets), diffs(tr, dsets), diffs(tr, alln)
    Ua = top_basis(Da, RANDOM_BG)
    B = {"obj": top_basis(Do, K_MAX), "dest": top_basis(Dd, K_MAX),
         "obj_c": contrast_basis(Do, Dd, K_MAX), "dest_c": contrast_basis(Dd, Do, K_MAX),
         "random": np.stack([np.linalg.qr(Ua[t] @ rng.standard_normal((RANDOM_BG, K_MAX)))[0]
                             for t in range(Ua.shape[0])])}
    share = lambda D, U: float((np.einsum("thk,nth->ntk", U, D) ** 2).sum() / (D ** 2).sum())
    Do, Dd, Da = diffs(te, osets), diffs(te, dsets), diffs(te, alln)
    print(f"excluded goals {sorted(excl) or '-'}; object sets {osets}; destination sets {dsets}")
    print(f"{len(tr)} train / {len(te)} held-out groups (by episode); held-out share of object / destination / all "
          f"instruction-induced variation, by rank:")
    for nm, U in B.items():
        print(f"  {nm:<7}" + "  ".join(f"k={k}: {share(Do, U[:, :, :k]):.2f}/{share(Dd, U[:, :, :k]):.2f}/"
                                       f"{share(Da, U[:, :, :k]):.2f}" for k in (1, 4, 16, 64)))
    for k in (1, 4, 16):
        cos = lambda u, v: np.mean([np.linalg.svd(u[t, :, :k].T @ v[t, :, :k], compute_uv=False).max()
                                    for t in range(u.shape[0])])
        print(f"  largest cosine at k={k}: obj/dest {cos(B['obj'], B['dest']):.2f}, obj_c/dest_c "
              f"{cos(B['obj_c'], B['dest_c']):.2f}")
    np.savez(a.out, **{f"{nm}.U": U.astype(np.float32) for nm, U in B.items()},
             exclude=np.array(sorted(excl)), train_episodes=np.array(sorted({ep(g) for g in tr})),
             heldout_episodes=np.array(sorted({ep(g) for g in te})), random_bg=RANDOM_BG, seed=a.seed)
    print(f"wrote {a.out} (train-split fit)")


def moved_object(rows: list[dict]) -> tuple[str, np.ndarray | None]:
    """The movable object displaced most (> 5 cm) between an episode's first and last call, and its last position."""
    d = {o: np.linalg.norm(np.subtract(rows[-1][f"{o}_pos"], rows[0][f"{o}_pos"])) for o in OBJ}
    o, m = max(d.items(), key=lambda kv: kv[1])
    return (o.split("_")[0], np.array(rows[-1][f"{o}_pos"])) if m > 0.05 else ("none", None)


# Destination and target object of every place goal; plate and bowl are movable, so their own position is the anchor.
DEST = {"s1": "stove", "s2": "cabinet", "s3": "drawer", "s4": "cabinet", "s5": "stove front", "s6": "bowl",
        "s8": "plate", "s9": "rack", "n0": "plate", "n1": "stove", "n2": "cabinet", "n3": "plate", "n4": "stove",
        "n5": "stove front", "n6": "drawer"}
TARGET = {"s1": "akita", "s2": "wine", "s3": "akita", "s4": "akita", "s5": "plate", "s6": "cream", "s8": "akita",
          "s9": "wine", "n0": "cream", "n1": "cream", "n2": "cream", "n3": "wine", "n4": "wine", "n5": "akita",
          "n6": "cream"}
MOVABLE_DEST = {"plate": "plate_1", "bowl": "akita_black_bowl_1"}


def landing(a) -> None:
    """Where were the objects at the end? Positions are the episode's final state when eval_goals recorded it
    (`final_pos`), else the last policy call (the last chunk not yet executed: "last-decision position").
    Anchors for fixed destinations: mean final position of the target object in successful *seen* (s*) episodes of
    the anchor run; plate / bowl: their own position in the episode. Per run and goal: the moved object -> nearest
    destination (within 12 cm, else "elsewhere"), and separately the target object's median distance to its own
    destination. Position only: says where things were, not whether the policy "meant" it (no grasp / stability
    check)."""
    from collections import Counter, defaultdict
    from pathlib import Path

    def episodes(run, labels):
        eps = [json.loads(l) for l in open(Path(run) / "episodes.jsonl")]
        rows = defaultdict(list)
        for l in map(json.loads, open(labels)):
            if "swap_of" not in l:
                rows[l["episode"]].append(l)
        out = []
        for i, e in enumerate(eps):
            start = {o: np.array(rows[i][0][f"{o}_pos"]) for o in OBJ}
            end = {o: np.array((e.get("final_pos") or rows[i][-1])[f"{o}_pos"]) for o in OBJ}
            out.append((e, start, end))
        return out

    pts = defaultdict(list)
    for e, _, end in episodes(a.anchor_run, a.anchor_labels):
        g = e["goal"]
        if e["success"] and g.startswith("s") and g in DEST and DEST[g] not in MOVABLE_DEST:
            pts[DEST[g]].append(next(v for o, v in end.items() if o.startswith(TARGET[g])))
    anchor = {k: np.mean(v, 0) for k, v in pts.items()}
    print("anchors (successful seen episodes): " + ", ".join(f"{k} n={len(pts[k])}" for k in anchor))
    for d in a.runs:
        if not (Path(d) / "episodes.jsonl").exists():
            print(f"{Path(d).name:<16} not finished")
            continue
        eps = episodes(d, Path(d) / "labels.jsonl")
        print(f"{Path(d).name}: {'final state' if eps and 'final_pos' in eps[0][0] else 'last-decision position'}")
        for goal in dict.fromkeys(e["goal"] for e, _, _ in eps):
            if goal not in DEST or (a.goals and goal not in a.goals.split(",")):
                continue
            where, tgt = Counter(), []
            for e, start, end in (x for x in eps if x[0]["goal"] == goal):
                dests = lambda o: {**anchor, **{k: end[m] for k, m in MOVABLE_DEST.items() if not m.startswith(o)}}
                t = next(o for o in OBJ if o.startswith(TARGET[goal]))
                tgt.append(np.linalg.norm(end[t] - dests(TARGET[goal])[DEST[goal]]))
                o, m = max(((o, np.linalg.norm(end[o] - start[o])) for o in OBJ), key=lambda kv: kv[1])
                if m <= 0.05:
                    where["none"] += 1
                    continue
                k, dk = min(((k, np.linalg.norm(end[o] - v)) for k, v in dests(o.split("_")[0]).items()),
                            key=lambda kv: kv[1])
                where[f"{o.split('_')[0]}->{k if dk < 0.12 else 'elsewhere'}"] += 1
            print(f"  {goal} {TARGET[goal]}->{DEST[goal]:<11} target median {np.median(tgt) * 100:4.0f} cm"
                  f" (<12 cm {sum(x < 0.12 for x in tgt)}/{len(tgt)})  moved: {dict(where)}")


def patch_report(a) -> None:
    """One line per patch run: requested-goal success, donor goal completed, object moved > 5 cm."""
    from collections import Counter
    from pathlib import Path
    for d in a.runs:
        if not (Path(d) / "episodes.jsonl").exists():
            print(f"{Path(d).name:<16} not finished")
            continue
        cfg = json.load(open(Path(d) / "run_config.json"))
        eps = [json.loads(l) for l in open(Path(d) / "episodes.jsonl")]
        L = [json.loads(l) for l in open(Path(d) / "labels.jsonl")]
        donors = dict(kv.split("=") for kv in cfg.get("env", {}).get("LV_PATCH", "").split(",") if kv)
        for goal in dict.fromkeys(e["goal"] for e in eps):
            idx = [i for i, e in enumerate(eps) if e["goal"] == goal]
            moved = Counter()
            for i in idx:
                moved[moved_object([l for l in L if l["episode"] == i])[0]] += 1
            b = donors.get(goal)
            don = sum(b in eps[i]["completed_other"] for i in idx) if b else 0
            print(f"{Path(d).name:<16} {goal}<-{b or '-':<3} success {sum(eps[i]['success'] for i in idx)}/{len(idx)}"
                  f"  donor goal done {don}/{len(idx)}  moved {dict(moved)}")


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
    sp = sub.add_parser("subspaces")
    sp.add_argument("--rec", required=True)
    sp.add_argument("--labels", required=True)
    sp.add_argument("--run_dir", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--exclude", default="", help="comma-separated goals left out of fitting (held-out combinations)")
    al = sub.add_parser("alphas")
    al.add_argument("--rec", required=True)
    al.add_argument("--labels", required=True)
    al.add_argument("--run_config", required=True)
    al.add_argument("--erasers", required=True)
    al.add_argument("--out", required=True)
    pr = sub.add_parser("patch_report")
    pr.add_argument("runs", nargs="+", help="eval_goals --out dirs of the patch runs (with labels.jsonl)")
    la = sub.add_parser("landing")
    la.add_argument("--anchor_run", required=True, help="the recorded eval_goals --out dir (seen successes)")
    la.add_argument("--anchor_labels", required=True)
    la.add_argument("--goals", default="", help="comma-separated goal filter")
    la.add_argument("runs", nargs="+", help="eval_goals --out dirs with labels.jsonl")
    a = ap.parse_args()
    {"fit": fit, "binding": binding, "subspaces": subspaces, "alphas": alphas, "patch_report": patch_report,
     "landing": landing}[a.cmd](a)


if __name__ == "__main__":
    main()
