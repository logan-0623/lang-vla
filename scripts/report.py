"""One compact text report of everything under $RESULTS (E0/E1 eval, E3, E6, LIBERO-Para), for pasting.

  python scripts/report.py [--since 10161500]     # only result dirs whose name ends with a timestamp >= since
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

R = Path(os.environ["RESULTS"])
GOALS = ["open middle drawer", "bowl->stove", "wine->cabinet top", "top drawer+bowl", "bowl->cabinet top",
         "push plate", "cheese->bowl", "turn on stove", "bowl->plate", "wine->rack"]


def ts_ok(d, since):
    m = re.search(r"-(\d{8})$", d.name)
    return since is None or (m is not None and m.group(1) >= since)


def eval_section():
    print("\n######## E0 / E1 (upstream eval) ########")
    for d in sorted(R.glob("eval/*")):
        f = d / "summary.json"
        if f.exists():
            s = json.loads(f.read_text())
            print(f"{d.name:<80} {s['successes']:>4}/{s['episodes']:<4} {s['success_rate'] or 0:6.1%} err={s['episode_errors']}")
        elif (d / "eval_info.json").exists():
            s = json.loads((d / "eval_info.json").read_text())
            o = s.get("overall", s.get("aggregated", {}))
            print(f"{d.name:<80} lerobot {json.dumps(o)[:120]}")


def e3_section(since):
    print("\n######## E3 ########")
    by_model = defaultdict(list)
    for d in sorted(R.glob("e3/*")):
        if d.name.startswith("pilot") or not ts_ok(d, since):
            continue
        ep = d / "episodes.jsonl"
        if not ep.exists():
            print(f"{d.name}: no episodes.jsonl (failed or still running)")
            continue
        det = json.loads((d / "determinism.json").read_text()) if (d / "determinism.json").exists() else {}
        col = json.loads((d / "collection.json").read_text()) if (d / "collection.json").exists() else []
        eps = [json.loads(l) for l in open(ep)]
        n_rep = len(det.get("orig_vs_orig_rep", []))
        print(f"{d.name}: {len(eps)} branches, orig==orig_rep {det.get('n_identical', '?')}/{n_rep}, "
              f"collection " + "; ".join(f"t{c['task_id']} {c['source']} {c['grasps']}/{c['attempts']}" for c in col))
        model = "oft" if "-oft" in d.name else "vla-adapter"
        kind = d.name.split("-")[0]                     # para / wording / src3 ...: never pooled
        by_model[(model, kind)].extend(eps)

    for (model, kind), eps in sorted(by_model.items()):
        print(f"\n=== {model} / {kind}: {len(eps)} branches, {len({(e['task_id'], e['state']) for e in eps})} states")
        tab = defaultdict(Counter)
        for e in eps:
            tab[(e["source"], e["cont"])][e["outcome"]] += 1
        print(f"  {'source->cont':<28}{'n':>5}{'success':>9}  other outcomes")
        for (src, cont), c in sorted(tab.items()):
            n = sum(c.values())
            rest = ", ".join(f"{k} {v}" for k, v in c.most_common() if k != "success")
            print(f"  {src + '->' + cont:<28}{n:>5}{c['success'] / n:>9.0%}  {rest}")
        # per task success for the key contrasts
        print("  per task success: orig->orig | orig->para_source | orig->para | para->orig | para->para_source")
        for t in sorted({e["task_id"] for e in eps}):
            row = []
            for src, cont in (("orig", "orig"), ("orig", "para_source"), ("orig", "para"), ("para", "orig"), ("para", "para_source")):
                xs = [e["outcome"] == "success" for e in eps if e["task_id"] == t and e["source"] == src and e["cont"] == cont]
                row.append(f"{sum(xs)}/{len(xs)}" if xs else "-")
            print(f"    t{t} {GOALS[t]:<20} " + " | ".join(row))
        # redirection after grasp: did the robot execute the newly instructed goal?
        red = defaultdict(Counter)
        for e in eps:
            if e["cont"] == "other_same_object":
                j = int(e["cont_cat"])
                red[(e["task_id"], j)]["new" if e["executed"] == j else "orig" if e["outcome"] == "success" else "other"] += 1
        if red:
            print("  redirect after grasp (new instructed goal executed / original kept / other):")
            for (t, j), c in sorted(red.items()):
                print(f"    t{t} {GOALS[t]:<18} -> {GOALS[j]:<18} new {c['new']:>3}  orig {c['orig']:>3}  other {c['other']:>3}")
        print("  source paraphrase per task: text | orig->para_source executed | para->orig executed")
        for t in sorted({e["task_id"] for e in eps}):
            src = [e for e in eps if e["task_id"] == t and e["cont"] == "para_source"]
            for text in sorted({e["cont_text"] for e in src}):
                ex = lambda sc, cont: Counter(GOALS[e["executed"]] if e["executed"] is not None else e["outcome"]
                                              for e in eps if e["task_id"] == t and e["source"] == sc and e["cont"] == cont
                                              and (cont != "para_source" or e["cont_text"] == text))
                print(f"    t{t} [{src[0]['cont_cat']}] \"{text}\"\n        orig->para_source {dict(ex('orig', 'para_source'))}"
                      f"\n        para->orig        {dict(ex('para', 'orig'))}")
        emp = defaultdict(Counter)
        for e in eps:
            if e["cont"] == "empty":
                emp[e["task_id"]][GOALS[e["executed"]] if e["executed"] is not None else e["outcome"]] += 1
        if emp:
            print("  empty instruction after grasp, executed:")
            for t, c in sorted(emp.items()):
                print(f"    t{t} {GOALS[t]:<18} " + ", ".join(f"{k} {v}" for k, v in c.most_common()))


def e4_section(since):
    print("\n######## E4 (switch time; 'new' = the switched-to goal was executed) ########")
    for d in sorted(R.glob("e4/*")):
        if not ts_ok(d, since) or not (d / "episodes.jsonl").exists():
            continue
        eps = [json.loads(l) for l in open(d / "episodes.jsonl")]
        det = json.loads((d / "determinism.json").read_text()) if (d / "determinism.json").exists() else {}
        print(f"{d.name}: {len(eps)} branches, orig==orig_rep {det.get('n_identical', '?')}/{len(det.get('orig_vs_orig_rep', []))}")
        taus = sorted({e["tau"] for e in eps},
                      key=lambda t: -10**6 if t == "start" else 10**5 + float(t[1:]) if isinstance(t, str) else t)
        print(f"  {'task -> switched-to goal':<44}" + "".join(f"{str(t):>9}" for t in taus))
        keys = sorted({(e["task_id"], e["cont"], e["cont_goal"]) for e in eps if e["cont"] in ("other", "empty", "orig")},
                      key=lambda k: (k[0], k[1], -1 if k[2] is None else k[2]))
        for t, cont, j in keys:
            cells = []
            for tau in taus:
                xs = [e for e in eps if (e["task_id"], e["cont"], e["cont_goal"], e["tau"]) == (t, cont, j, tau)]
                hit = sum(e["executed"] == (t if cont in ("orig", "empty") else j) for e in xs)
                cells.append(f"{hit}/{len(xs)}" if xs else "-")
            label = f"t{t} {GOALS[t]} -> " + (GOALS[j] if cont == "other" else cont + " (keeps orig)")
            print(f"  {label:<44}" + "".join(f"{c:>9}" for c in cells))


def e6_section(since):
    print("\n######## E6 ########")
    for d in sorted(R.glob("e6/*")):
        if not ts_ok(d, since) or not (d / "rows.csv").exists():
            continue
        rows = list(csv.DictReader(open(d / "rows.csv")))
        last = max(int(r["layer"]) for r in rows)
        ract, rlast = defaultdict(dict), defaultdict(list)
        for r in rows:
            g = "empty" if r["kind"] == "empty" else r["high"]
            ract[g][(r["task_id"], r["init"], r["para_id"])] = float(r["R_action"])
            if r["cls"] == "action" and int(r["layer"]) == last and r["R"] != "":
                rlast[g].append(float(r["R"]))
        print(f"{d.name}  (mean, 95% CI over probes; last layer {last})")
        for g in sorted(ract):
            v = np.array(list(ract[g].values()))
            se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
            print(f"  {g:<6} R_action {v.mean():.3f} [{v.mean() - 1.96 * se:.3f}, {v.mean() + 1.96 * se:.3f}] n={len(v):<4}"
                  f" action-slot R {np.nanmean(rlast[g]):.3f}")


def para_section():
    print("\n######## LIBERO-Para (own eval) ########")
    for d in sorted(R.glob("para/*")):
        files = [Path(f) for f in glob.glob(f"{d}/**/*", recursive=True) if Path(f).is_file() and not f.endswith(".bddl")]
        kinds = Counter(f.suffix for f in files)
        print(f"{d.name}: {len(files)} files {dict(kinds)}")
        for f in [f for f in files if f.suffix in (".json", ".csv", ".txt", ".log")][:3]:
            print(f"  -- {f.relative_to(d)}")
            print("   " + "\n   ".join(f.read_text(errors="replace").splitlines()[-6:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="MMDDHHMM: only E3/E6 dirs with a timestamp >= this")
    a = ap.parse_args()
    eval_section()
    e3_section(a.since)
    e4_section(a.since)
    e6_section(a.since)
    para_section()
