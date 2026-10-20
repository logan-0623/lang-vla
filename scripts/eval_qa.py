"""Held-out QA accuracy of a VLM, at the three points of RESEARCH_PLAN §11: the raw Qwen3-VL, after stage-1 QA
SFT, after stage-2 VLA fine-tuning. Without it, "B did not help" cannot be split into "never learned", "lost in
VLA fine-tuning" and "learned but not used for action".

Runs in the starvla env. `--ckpt` is a StarVLA framework checkpoint (.pt); its `qwen_vl_interface.model.*`
weights are loaded into the Hugging Face model built from `--base` (StarVLA adds no tokens, so shapes match).
Every question is asked on its own, single turn, greedy decoding.

  python scripts/eval_qa.py --base $STORE/ckpt/Qwen3-VL-4B-Instruct --qa_dir $STORE/qa/goal-v1 --style lang --out ...
  python scripts/eval_qa.py --base ... --ckpt $STORE/runs/starvla/vlm-lang-<job>/checkpoints/steps_2000_pytorch_model.pt ...

Scoring per question type: the answer's words must match the reference's after numbers are removed, and every
number must be within tolerance (image coordinates +-50 on the 0-1000 scale, about 13 px; other numbers
max(2, 20%)). JSON answers (style num) are compared field by field with the same tolerances.

That exact score also requires our answer template, so a model never trained on it (the raw VLM) scores ~0 even
when right. `content_facts` therefore scores only the fact each answer carries, whatever the phrasing: yes/no,
the held object, the support, left/right (subject order handled), distances, image position (a box counts by its
centre). The summary reports both, plus each type's majority-answer rate (e.g. "Nothing." for most held
questions), the accuracy of always giving the commonest answer. next_step has no content score (rule label).
Rescore saved answers without a GPU: python scripts/eval_qa.py --rescore $RESULTS/qa/<tag>
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NUM = re.compile(r"-?\d+(?:\.\d+)?")
TYPES = [("do next", "next_step"), ("in the image", "position"), ("left or right", "side_distance"), ("How far is the gripper", "gripper_offset"),
         ("gripper holding", "held"), ("resting", "support"), ("drawer", "drawer"), ("stove turned on", "stove"),
         ("Is there a", "ctrl_presence"), ("pick up here", "ctrl_name"), ("kitchen", "ctrl_room")]


def qtype(q: str) -> str:
    return next((t for key, t in TYPES if key in q), "other")


def close(p: float, r: float, kind: str) -> bool:
    return abs(p - r) <= (50 if kind == "position" else max(2.0, 0.2 * abs(r)))


def score(pred: str, ref: str, kind: str) -> bool:
    try:
        rj = json.loads(ref)
    except ValueError:
        rj = None
    if isinstance(rj, dict):
        try:
            pj = json.loads(pred[pred.index("{"): pred.rindex("}") + 1])
        except ValueError:
            return False
        for k, v in rj.items():
            if isinstance(v, bool) or v is None or isinstance(v, str):
                if pj.get(k) != v or type(pj.get(k)) is not type(v):     # 1 == True in Python
                    return False
            elif not isinstance(pj.get(k), (int, float)) or isinstance(pj.get(k), bool) or not close(pj[k], v, kind):
                return False
        return True
    words = lambda s: NUM.sub("#", s.lower()).replace(".", "").split()
    if words(pred) != words(ref):
        return False
    pn, rn = [float(x) for x in NUM.findall(pred)], [float(x) for x in NUM.findall(ref)]
    return len(pn) == len(rn) and all(close(p, r, kind) for p, r in zip(pn, rn))


OBJ = ["black bowl", "cream cheese", "wine bottle", "plate"]
CM = re.compile(r"(-?\d+(?:\.\d+)?)\s*(cm|centimet|m\b|meter|metre|mm)")


def _yes(t: str, kind: str):
    m = re.search(r"\b(yes|no)\b", t)
    if m:
        return m.group(1) == "yes"
    if kind == "drawer":
        return None if "closed" not in t and "open" not in t else ("closed" not in t and "not open" not in t)
    return True if re.search(r"\bon\b", t) and not re.search(r"\boff\b|not on", t) else False if re.search(r"\boff\b|not on", t) else None


def _cm(t: str):
    m = CM.search(t)
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    return v * 100 if u in ("m", "meter", "metre") else v / 10 if u == "mm" else v


def content_facts(pred: str, ref: str, kind: str) -> dict:
    """{fact: correct} for the facts in `ref`; empty when the type has no content score."""
    p, r = pred.lower(), ref.lower().rstrip(".")
    if kind in ("drawer", "stove"):
        return {kind: _yes(p, kind) == _yes(r, kind)}
    if kind == "held":
        names = [o for o in OBJ if o in p or o.split()[-1] in p]
        if r == "nothing":
            return {"held": not names and bool(re.search(r"nothing|not holding|empty|no object|isn't holding|is not holding", p))}
        obj = r.removeprefix("the ")
        return {"held": obj in p or obj.split()[-1] in p}
    if kind == "support":
        sup = r.split(" is ", 1)[1]
        if "held" in sup:
            return {"support": "gripper" in p or "held" in p}
        key = sup.split(" the ")[-1]
        return {"support": key in p or key.split()[-1] in p}
    if kind == "side_distance":
        m = re.match(r"the (.+?) is to the (left|right) of the (.+?), about (\d+) cm away", r)
        a, d, b, dist = m.group(1), m.group(2), m.group(3), float(m.group(4))
        ia, ib = p.find(a.split()[-1]), p.find(b.split()[-1])
        dirs = re.findall(r"\b(left|right)\b", p)
        pd = dirs[0] if dirs else None
        if pd and ib != -1 and (ia == -1 or ib < ia):       # "the B is to the left of the A" = A right of B
            pd = {"left": "right", "right": "left"}[pd]
        pc = _cm(p)
        return {"side": pd == d, "distance": pc is not None and close(pc, dist, "distance")}
    if kind == "gripper_offset":
        pc = _cm(p)
        return {"distance": pc is not None and close(pc, float(NUM.findall(r)[0]), "distance")}
    if kind == "position":
        rn, pn = [float(x) for x in NUM.findall(r)], [float(x) for x in NUM.findall(p)]
        if len(pn) >= 4:
            pn = [(pn[0] + pn[2]) / 2, (pn[1] + pn[3]) / 2]
        return {"position": len(pn) >= 2 and close(pn[0], rn[0], kind) and close(pn[1], rn[1], kind)}
    return {}


def summarize(rows: list[dict]) -> dict:
    exact, content, refs = defaultdict(list), defaultdict(list), defaultdict(list)
    for x in rows:
        x = {**x, "type": qtype(x["q"])}          # recomputed, so --rescore also fixes types saved by older runs
        exact[x["type"]].append(x["ok"])
        refs[x["type"]].append(x["ref"])
        for f, ok in content_facts(x["pred"], x["ref"], x["type"]).items():
            content[f"{x['type']}.{f}"].append(ok)
    out = {}
    for k in sorted(exact):
        top = max(refs[k].count(v) for v in set(refs[k])) / len(refs[k])
        out[k] = {"n": len(exact[k]), "exact": sum(exact[k]) / len(exact[k]), "majority_answer": top}
    for k in sorted(content):
        out.setdefault(k, {"n": len(content[k])})["content"] = sum(content[k]) / len(content[k])
    allc = [v for vs in content.values() for v in vs]
    out["_all"] = {"n": len(rows), "exact": sum(x["ok"] for x in rows) / max(1, len(rows)),
                   "content": sum(allc) / max(1, len(allc))}
    return out


def report(summary: dict) -> None:
    for k, v in summary.items():
        cols = "  ".join(f"{m}={v[m]:.3f}" for m in ("exact", "content", "majority_answer") if m in v)
        print(f"{k:<24} n={v['n']:<5} {cols}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescore", default=None, help="an earlier --out dir: rescore its answers.jsonl, no model")
    ap.add_argument("--base", help="Hugging Face Qwen3-VL directory")
    ap.add_argument("--ckpt", default=None, help="StarVLA .pt; omit for the raw VLM")
    ap.add_argument("--qa_dir")
    ap.add_argument("--style", default="lang", choices=["lang", "num", "ctrl"])
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.rescore:
        d = Path(args.rescore)
        summary = summarize([json.loads(l) for l in open(d / "answers.jsonl")])
        prev = json.loads((d / "summary.json").read_text())
        (d / "summary.json").write_text(json.dumps({"args": prev["args"], "accuracy": summary}, indent=1))
        return report(summary)
    if not (args.base and args.out and args.qa_dir):
        ap.error("--base, --qa_dir and --out are required unless --rescore")

    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(args.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa")
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location="cpu")
        pre = "qwen_vl_interface.model."
        sub = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        missing, unexpected = model.load_state_dict(sub, strict=False)
        print(f"loaded {len(sub)} tensors from {args.ckpt}; missing {len(missing)}, unexpected {len(unexpected)}")
        if unexpected or len(missing) > 0.01 * len(sub):
            raise SystemExit(f"checkpoint does not match {args.base}: missing {missing[:5]} unexpected {unexpected[:5]}")
    model = model.cuda().eval()
    proc = AutoProcessor.from_pretrained(args.base)

    qa_dir = Path(args.qa_dir)
    data = json.load(open(qa_dir / "llava_jsons" / f"libero_qa_{args.style}_heldout.json"))[: args.max_images]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    rows, saved = open(out / "answers.jsonl", "w"), []
    for item in data:
        img = Image.open(qa_dir / item["image"]).convert("RGB")
        conv = item["conversations"]
        for h, g in zip(conv[::2], conv[1::2]):
            q = h["value"].replace("<image>\n", "")
            msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": q}]}]
            inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to(model.device)
            with torch.inference_mode():
                ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
            pred = proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            kind = qtype(q)
            ok = score(pred, g["value"], kind)
            saved.append({"image": item["image"], "type": kind, "q": q, "ref": g["value"], "pred": pred, "ok": ok})
            rows.write(json.dumps(saved[-1]) + "\n")
    rows.close()
    summary = summarize(saved)
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "accuracy": summary}, indent=1))
    report(summary)


if __name__ == "__main__":
    main()
