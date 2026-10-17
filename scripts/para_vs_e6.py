"""Does E6's R_action (how far a paraphrase moves the action chunk, at the first frame) predict whether that
paraphrase fails in LIBERO-Para's own rollouts?

  python scripts/para_vs_e6.py --para $RESULTS/para/vla-adapter-LIBERO-Goal-Pro-49317676 --e6 $RESULTS/e6/vla-adapter-all-...

Joins by paraphrase text (lowercased). R_action is averaged over E6's init states. Reports the AUC of R_action for
predicting failure (0.5 = no information), overall and per LIBERO-Para category, and success by R_action quintile.
"""
import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def auc(pos, neg):
    """P(score of a random positive > random negative), ties count half (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--para", required=True, help="LIBERO-Para result dir (shard*/logs/eval*.json)")
    ap.add_argument("--e6", required=True, help="E6 result dir (rows.csv)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--para_csv", default=None, help="LIBERO-Para metadata (text-only baselines); default $PARA/metrics/...")
    a = ap.parse_args()

    succ = defaultdict(list)   # text -> [success]
    cat = {}
    for f in glob.glob(f"{a.para}/shard*/logs/eval*.json"):
        for e in json.load(open(f))["episodes"]:
            t = e["paraphrased_instruction"].strip().lower()
            succ[t].append(bool(e["success"]))
            cat[t] = e["paraphrase_type"]
    ract = defaultdict(list)
    for r in csv.DictReader(open(Path(a.e6) / "rows.csv")):
        if r["kind"] == "para" and r["cls"] == "action":
            ract[r["instruction"].strip().lower()].append(float(r["R_action"]))
    ract = {t: np.mean(v) for t, v in ract.items()}     # R_action repeats across layers; the mean is that value

    keys = sorted(set(succ) & set(ract))
    print(f"LIBERO-Para paraphrases: {len(succ)}, in E6: {len(ract)}, joined: {len(keys)}")
    if not keys:
        return
    x = np.array([ract[k] for k in keys])
    y = np.array([np.mean(succ[k]) for k in keys])           # usually one episode per paraphrase
    fail = y < 0.5
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.boot):
        i = rng.integers(0, len(keys), len(keys))
        boots.append(auc(x[i][fail[i]], x[i][~fail[i]]))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    print(f"success rate {y.mean():.3f}; AUC(R_action -> failure) = {auc(x[fail], x[~fail]):.3f} [{lo:.3f}, {hi:.3f}]")
    for c in sorted({cat[k] for k in keys}):
        m = np.array([cat[k] == c for k in keys])
        print(f"  {c:<6} n={m.sum():<5} success {y[m].mean():.3f}  AUC {auc(x[m & fail], x[m & ~fail]):.3f}")
    # Text-only baselines from LIBERO-Para's own metadata: does a rollout-free *text* score predict failure as well?
    import os
    meta = Path(a.para_csv or Path(os.environ.get("PARA", ".")) / "metrics/libero_para_metadata.csv")
    if meta.exists():
        sim = {}
        for r in csv.DictReader(open(meta)):
            sim[r["new_instruction"].strip().lower()] = (float(r["structural_similarity"]), float(r["keyword_similarity"]))
        ok = [i for i, k in enumerate(keys) if k in sim]
        print(f"text-only baselines (n={len(ok)}; score = 1 - similarity, higher should mean more failure):")
        for j, name in enumerate(("structural_similarity", "keyword_similarity")):
            z = np.array([1 - sim[keys[i]][j] for i in ok])
            f_ok = fail[ok]
            print(f"  {name:<22} AUC {auc(z[f_ok], z[~f_ok]):.3f}")
        xo = x[ok]
        print(f"  R_action on the same rows  AUC {auc(xo[fail[ok]], xo[~fail[ok]]):.3f}")
    qs = np.quantile(x, [0, .2, .4, .6, .8, 1])
    print("success by R_action quintile:")
    for lo_, hi_ in zip(qs[:-1], qs[1:]):
        m = (x >= lo_) & (x <= hi_)
        print(f"  R_action [{lo_:.2f}, {hi_:.2f}]  n={m.sum():<5} success {y[m].mean():.3f}")


if __name__ == "__main__":
    main()
