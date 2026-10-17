"""Apply the pre-registered rules of docs/RESEARCH_PLAN.md §3.2 (written 2026-10-17, before batch_1017/1017b results).

  python scripts/judge_h1h3.py                       # all $RESULTS/e4/* and $RESULTS/e3/nonce-*

Units for bootstrap: source states, i.e. (task, init) for E4 and (task, state file) for E3. "Clearly different"
means non-overlapping 95% CIs. TV thresholds are judged on point estimates; CIs are printed alongside.
Bowl goals: 1 stove, 4 cabinet top, 8 plate (3 = top drawer + bowl is infeasible while holding the bowl: excluded).
Wine goals: 2 cabinet top, 9 rack.
"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

R = Path(os.environ["RESULTS"])
BOWL, WINE = {1, 4, 8}, {2, 9}
RNG = np.random.default_rng(0)
B = 2000


def model_of(name):
    """Model from a result dir name: e4/oft-t1-..., e3/nonce-oft-..., .../pi05-..."""
    parts = name.split("-")
    return "oft" if "oft" in parts else "pi05" if "pi05" in parts else "vla-adapter"


def boot(units, stat):
    """units: dict unit -> list of items; stat(list of items) -> float. Returns (point, lo, hi)."""
    keys = list(units)
    point = stat([x for k in keys for x in units[k]])
    vals = []
    for _ in range(B):
        pick = RNG.integers(0, len(keys), len(keys))
        v = stat([x for i in pick for x in units[keys[i]]])
        if v == v:
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return point, lo, hi


def frac(items):
    return float(np.mean(items)) if items else float("nan")


def fmt(p):
    return f"{p[0]:.2f} [{p[1]:.2f}, {p[2]:.2f}]"


def apart(a, b):
    return a[2] < b[1] or b[2] < a[1]


def tv(p, q):
    keys = set(p) | set(q)
    n_p, n_q = sum(p.values()), sum(q.values())
    if not n_p or not n_q:
        return float("nan")
    return 0.5 * sum(abs(p[k] / n_p - q[k] / n_q) for k in keys)


def judge_h1():
    print("\n================ H1' (E4 switch time) ================")
    by_model = defaultdict(list)
    for d in sorted(R.glob("e4/*")):
        f = d / "episodes.jsonl"
        if f.exists() and not d.name.startswith("frac-"):
            by_model[model_of(d.name)] += [json.loads(l) for l in open(f)]
    for model, eps in sorted(by_model.items()):
        eps = [e for e in eps if e["cont"] == "other" and not e["error"]]
        f = {}
        for obj, goals in (("bowl", BOWL), ("wine", WINE)):
            for tau in ("start", -40, -20, -10, 0, 20):
                units = defaultdict(list)
                for e in eps:
                    if e["task_id"] in goals and e["cont_goal"] in goals and e["tau"] == tau:
                        units[(e["task_id"], e["init"])].append(e["executed"] == e["cont_goal"])
                f[obj, tau] = boot(units, frac) if units else (np.nan,) * 3
        print(f"--- {model}")
        for obj in ("bowl", "wine"):
            print(f"  f({obj}): " + "  ".join(f"{t}: {fmt(f[obj, t])}" for t in ("start", -40, -20, -10, 0, 20)))
        p1 = f["bowl", "start"][0] >= 0.7 and f["wine", "start"][0] >= 0.7
        p2 = f["bowl", 20][0] >= 0.6
        p3 = f["wine", -20][2] < f["wine", "start"][1] and apart(f["wine", 0], f["bowl", 0]) \
            and f["wine", 0][0] < f["bowl", 0][0]
        ra = all(not apart(f["bowl", t], f["wine", t]) for t in (-10, 0, 20))
        rb = f["bowl", 20][0] <= 0.3
        rc = f["bowl", "start"][0] <= 0.3 and f["wine", "start"][0] <= 0.3
        print(f"  P1 f(start) >= 0.7 both: {p1} | P2 bowl f(+20) >= 0.6: {p2} | "
              f"P3 wine f(-20) < f(start) and f(0) < bowl f(0), CIs apart: {p3}")
        print(f"  refute (a) no bowl/wine difference at every tau >= -10: {ra} | (b) bowl f(+20) <= 0.3: {rb} | "
              f"(c) both f(start) <= 0.3: {rc}")
        print(f"  => H1' {'REFUTED' if ra or rb or rc else 'SUPPORTED' if p1 and p2 and p3 else 'NOT SUPPORTED (no refutation)'}")


def judge_h1pp():
    """H1'' (RESEARCH_PLAN §3.3): switch times as fractions q of the grasp -> placement interval, dirs e4/frac-*."""
    print("\n================ H1'' (E4, placement-relative switch times) ================")
    by_model = defaultdict(list)
    for d in sorted(R.glob("e4/frac-*")):
        f = d / "episodes.jsonl"
        if f.exists():
            by_model[model_of(d.name)] += [json.loads(l) for l in open(f)]
    for model, eps in sorted(by_model.items()):
        f, base = {}, {}
        for obj, goals in (("bowl", BOWL), ("wine", WINE)):
            for tau in (0, "q0.25", "q0.5", "q0.75"):
                units, ou = defaultdict(list), defaultdict(list)
                for e in eps:
                    if e["error"] or e["task_id"] not in goals or e["tau"] != tau:
                        continue
                    if e["cont"] == "other" and e["cont_goal"] in goals:
                        units[(e["task_id"], e["init"])].append(e["executed"] == e["cont_goal"])
                    if e["cont"] == "orig":
                        ou[(e["task_id"], e["init"])].append(e["executed"] == e["task_id"])
                f[obj, tau] = boot(units, frac) if units else (np.nan,) * 3
                base[obj, tau] = boot(ou, frac) if ou else (np.nan,) * 3
        print(f"--- {model}")
        for obj in ("bowl", "wine"):
            print(f"  f({obj}):    " + "  ".join(f"{t}: {fmt(f[obj, t])}" for t in (0, "q0.25", "q0.5", "q0.75")))
            print(f"  orig({obj}): " + "  ".join(f"{t}: {fmt(base[obj, t])}" for t in (0, "q0.25", "q0.5", "q0.75")))
        p1 = f["bowl", "q0.25"][0] >= 0.5
        p2 = f["bowl", "q0.75"][2] < f["bowl", "q0.25"][1]
        p3 = f["wine", "q0.25"][0] <= 0.2
        ra = f["bowl", "q0.25"][0] <= 0.2
        rb = not apart(f["bowl", "q0.25"], f["wine", "q0.25"])
        print(f"  P1 bowl f(q0.25) >= 0.5: {p1} | P2 bowl f(q0.75) < f(q0.25), CIs apart: {p2} | P3 wine f(q0.25) <= 0.2: {p3}")
        print(f"  refute (a) bowl f(q0.25) <= 0.2: {ra} | (b) bowl and wine f(q0.25) CIs overlap: {rb}")
        print(f"  => H1'' {'REFUTED' if ra or rb else 'SUPPORTED' if p1 and p2 and p3 else 'NOT SUPPORTED (no refutation)'}")


def judge_h3():
    print("\n================ H3' (nonce words, E3) ================")
    by_model = defaultdict(list)
    for d in sorted(R.glob("e3/nonce-*")):
        f = d / "episodes.jsonl"
        if f.exists():
            by_model[model_of(d.name)] += [json.loads(l) for l in open(f)]

    def kind(e):
        c = e["cont"]
        if c in ("para_source", "para"):
            mid = e["cont_cat"].split("/")[1]
            return {"a_target_nonce": "target_blicket", "b_target_nonce": "target_dax",
                    "c_object_nonce": "object_nonce", "d_both_nonce": "both_nonce"}[mid]
        return c

    for model, eps in sorted(by_model.items()):
        print(f"--- {model}")
        post = [e for e in eps if e["source"] == "orig" and not e["error"]]       # post-grasp: orig-sourced states
        bowl = [e for e in post if e["task_id"] in BOWL]
        dist = defaultdict(Counter)
        units = defaultdict(lambda: defaultdict(list))
        for e in bowl:
            k = kind(e)
            ex = e["executed"] if e["executed"] is not None else "none"
            dist[k][(e["task_id"], ex)] += 1
            units[k][(e["task_id"], e["state"])].append(e)
        tgt = dist["target_blicket"] + dist["target_dax"]
        tv_te, tv_to, tv_bd = tv(tgt, dist["empty"]), tv(tgt, dist["orig"]), tv(dist["target_blicket"], dist["target_dax"])
        succ = lambda k: boot(units[k], lambda xs: frac([x["outcome"] == "success" for x in xs]))
        s_orig, s_obj = succ("orig"), succ("object_nonce")
        for t in sorted(BOWL):
            per = {k: Counter({g: n for (tt, g), n in dist[k].items() if tt == t}) for k in ("orig", "empty", "target_blicket", "target_dax", "object_nonce")}
            print(f"  bowl t{t}: " + " | ".join(f"{k} {dict(v)}" for k, v in per.items()))
        print(f"  TV(target nonce, empty) = {tv_te:.2f}   TV(target nonce, orig) = {tv_to:.2f}   TV(blicket, dax) = {tv_bd:.2f}")
        print(f"  bowl success: orig {fmt(s_orig)}  object nonce {fmt(s_obj)}")
        p1 = tv_te <= 0.2 and tv_to >= 0.4
        p2 = s_obj[0] >= s_orig[0] - 0.10
        p3 = tv_bd <= 0.2
        # P4, wine t2 only (t9's original goal is the rack itself): rack (9) after an orig continuation,
        # from target-nonce-sourced vs orig-sourced grasp states.
        wu = {src: defaultdict(list) for src in ("orig", "para")}
        for e in eps:
            if e["task_id"] == 2 and e["cont"] == "orig" and not e["error"]:
                wu[e["source"]][(e["task_id"], e["state"])].append(e["executed"] == 9)
        r_orig = boot(wu["orig"], frac) if wu["orig"] else (np.nan,) * 3
        r_para = boot(wu["para"], frac) if wu["para"] else (np.nan,) * 3
        p4 = apart(r_para, r_orig) and r_para[0] > r_orig[0]
        print(f"  wine t2, orig continuation, executed rack: orig-sourced {fmt(r_orig)}  target-nonce-sourced {fmt(r_para)}")
        def mode(c, t):
            m = Counter({g: n for (tt, g), n in c.items() if tt == t}).most_common(1)
            return m[0][0] if m else None
        # (b) per bowl task, the target-nonce mode is neither the task's own goal nor the empty-instruction mode
        other_dest = [mode(tgt, t) not in (t, mode(dist["empty"], t), None) for t in sorted(BOWL) if mode(tgt, t) is not None]
        ra = tv_to <= 0.2
        rb = tv_te >= 0.4 and bool(other_dest) and all(other_dest)
        rc = s_obj[0] < s_orig[0] - 0.30
        print(f"  P1 target nonce ~ empty, != orig: {p1} | P2 object nonce >= orig - 10pt: {p2} | P3 blicket ~ dax: {p3} | "
              f"P4 wine nonce-sourced -> rack more: {p4}")
        print(f"  refute (a) target nonce ~ orig: {ra} | (b) far from empty, consistent other destination: {rb} | "
              f"(c) object nonce drops > 30pt: {rc}")
        print(f"  => H3' {'REFUTED' if ra or rb or rc else 'SUPPORTED' if p1 and p2 and p3 and p4 else 'NOT SUPPORTED (no refutation)'}")


if __name__ == "__main__":
    judge_h1()
    judge_h1pp()
    judge_h3()
