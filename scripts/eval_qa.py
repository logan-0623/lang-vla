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
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

NUM = re.compile(r"-?\d+(?:\.\d+)?")
TYPES = [("in the image", "position"), ("left or right", "side_distance"), ("How far is the gripper", "gripper_offset"),
         ("gripper holding", "held"), ("resting", "support"), ("drawer", "drawer"), ("stove turned on", "stove"),
         ("do next", "next_step"), ("Is there a", "ctrl_presence"), ("pick up here", "ctrl_name"), ("kitchen", "ctrl_room")]


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Hugging Face Qwen3-VL directory")
    ap.add_argument("--ckpt", default=None, help="StarVLA .pt; omit for the raw VLM")
    ap.add_argument("--qa_dir", required=True)
    ap.add_argument("--style", default="lang", choices=["lang", "num", "ctrl"])
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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
    rows, acc = open(out / "answers.jsonl", "w"), defaultdict(list)
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
            acc[kind].append(ok)
            rows.write(json.dumps({"image": item["image"], "type": kind, "q": q, "ref": g["value"], "pred": pred, "ok": ok}) + "\n")
    rows.close()
    summary = {k: {"n": len(v), "acc": sum(v) / len(v)} for k, v in sorted(acc.items())}
    summary["_all"] = {"n": sum(len(v) for v in acc.values()), "acc": sum(map(sum, acc.values())) / max(1, sum(map(len, acc.values())))}
    (out / "summary.json").write_text(json.dumps({"args": vars(args), "accuracy": summary}, indent=1))
    for k, v in summary.items():
        print(f"{k:<16} n={v['n']:<5} acc={v['acc']:.3f}")


if __name__ == "__main__":
    main()
