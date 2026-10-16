"""E6: layer-wise paraphrase invariance (forward passes only, no rollouts).

For each LIBERO-Goal task, start frame, decoder layer and position class:

    R = d(orig, x) / mean_j d(orig, other task j)

where x is a LIBERO-Para paraphrase (or the empty instruction). R ~ 0 means the model treats x as the same
task as the original instruction; R ~ 1 means it treats x like a different task. The same ratio on the
predicted action chunk (R_action) ties the representation to the output.

Position classes (no pooling across positions except where lengths differ):
  vision  -- vision patch positions, per-position distance averaged
  lang    -- instruction tokens; paraphrases differ in length, so the span is mean-pooled first
  action  -- action-query slots, per-position distance averaged
The action head of VLA-Adapter reads the hidden states shifted one position left of the query slots
(see docs/DECISIONS.md); `action` here is the slots themselves.

Works for VLA-Adapter (Qwen prompt) and OpenVLA-OFT (Llama prompt): run with the matching repo first on
PYTHONPATH. Layout assumptions are asserted on the first forward pass.

  PYTHONPATH=$VLAA python scripts/paraphrase_invariance.py --checkpoint $CKPT/LIBERO-Goal-Pro --out $RESULTS/e6/va
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from langeval.instructions import load_paraphrases

QWEN_PROMPT = ("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
               "<|im_start|>user\nWhat action should the robot take to {}?<|im_end|>\n<|im_start|>assistant\n")
LLAMA_PROMPT = "In: What action should the robot take to {}?\nOut:"
CLASSES = ("vision", "lang", "action")


def instruction_span(tokenize, template: str, instruction: str) -> tuple[int, int]:
    """[start, end) of the instruction's tokens within tokenize(template.format(instruction)).

    The prefix is tokenized without its trailing space, because BPE / sentencepiece attach that space to the
    instruction's first word. The start is checked against the full prompt's tokens. At the end, the
    instruction's last token can merge with the template text after it (a paraphrase ending in "?" followed by
    the template's "?" becomes "??"); the span then stops before the merged token.
    """
    pre = template.split("{}")[0]
    full = tokenize(template.format(instruction))
    head = tokenize(pre.rstrip(" "))
    upto = tokenize(pre + instruction) if instruction else head
    if full[: len(head)] != head:
        raise ValueError(f"prompt does not tokenize at the instruction start: {instruction!r}")
    e = len(head)
    while e < min(len(full), len(upto)) and full[e] == upto[e]:
        e += 1
    return len(head), e


def ratio(d: np.ndarray, d_other: np.ndarray) -> np.ndarray:
    """d / d_other, NaN where the reference distance is 0 (identical inputs)."""
    d, d_other = np.asarray(d, float), np.asarray(d_other, float)
    out = np.full(np.broadcast(d, d_other).shape, np.nan)
    np.divide(d, d_other, out=out, where=d_other > 0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--para_csv", default=None, help="default: $PARA/metrics/libero_para_metadata.csv")
    ap.add_argument("--n_init", type=int, default=3, help="start frames (init states) per task")
    ap.add_argument("--per_category", type=int, default=5, help="paraphrases per (high, mid) category per task")
    ap.add_argument("--layer_stride", type=int, default=2)
    ap.add_argument("--num_steps_wait", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from libero.libero import benchmark
    from experiments.robot.libero.libero_utils import get_libero_env
    from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, prepare_observation
    from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere
    import prismatic.vla.constants as C

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)          # never overwrite an earlier run
    para_csv = Path(args.para_csv or Path(os.environ["PARA"]) / "metrics/libero_para_metadata.csv")
    set_seed_everywhere(args.seed)

    is_adapter = "use_minivlm" in GenerateConfig.__dataclass_fields__
    kw = {"use_pro_version": True} if is_adapter else {"center_crop": True}
    cfg = GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name="libero_goal",
                         use_proprio=True, num_images_in_input=2, **kw)
    model, action_head, proprio_projector, _, processor = initialize_model(cfg)
    template = QWEN_PROMPT if is_adapter and cfg.use_minivlm else LLAMA_PROMPT
    tokenize = lambda s: processor.tokenizer(s).input_ids
    n_vision = model.vision_backbone.get_num_patches() * model.vision_backbone.get_num_images_in_input()
    n_slots = getattr(C, "NUM_TOKENS", C.ACTION_DIM * C.NUM_ACTIONS_CHUNK)

    decoder = model.language_model.model.layers
    layer_ids = list(range(args.layer_stride - 1, len(decoder), args.layer_stride))
    captured: dict[int, torch.Tensor] = {}
    for li in layer_ids:
        decoder[li].register_forward_hook(lambda m, i, o, li=li: captured.__setitem__(li, o[0].detach().float()))

    def run(observation, instruction: str):
        """Per-class representations (layers stacked) and the action chunk for one forward pass."""
        captured.clear()
        actions = get_action(cfg, model, copy.deepcopy(observation), instruction, processor=processor,
                             action_head=action_head, proprio_projector=proprio_projector, use_film=cfg.use_film,
                             **({"use_minivlm": cfg.use_minivlm} if is_adapter else {}))
        h = torch.stack([captured[li][0] for li in layer_ids])           # (layers, L, D)
        ids = tokenize(template.format(instruction.lower()))
        # OFT's predict_action appends the empty token 29871 when the prompt does not end with it.
        T = len(ids) + (not is_adapter and ids[-1] != 29871)
        P = h.shape[1] - T - n_slots - 1                                # vision (+ proprio token for OFT)
        if P not in (n_vision, n_vision + 1):
            raise RuntimeError(f"unexpected layout: L={h.shape[1]} T={T} slots={n_slots} -> P={P}, "
                               f"expected {n_vision} or {n_vision + 1}")
        s, e = instruction_span(tokenize, template, instruction.lower())
        lang = h[:, P + s: P + e].mean(1) if e > s else None             # input_ids[j] sits at P + j for j >= 1
        return {"vision": h[:, 1:1 + n_vision], "lang": lang, "action": h[:, P + T: P + T + n_slots]}, \
            np.asarray(actions, dtype=np.float64)

    def dist(a, b) -> np.ndarray:
        """Per-layer distance; NaN when a class is missing (empty instruction has no tokens)."""
        if a is None or b is None:
            return np.full(len(layer_ids), np.nan)
        diff = (a - b).norm(dim=-1)
        return (diff.mean(-1) if diff.dim() == 2 else diff).cpu().numpy()

    paraphrases = load_paraphrases(para_csv, args.per_category, args.seed)
    suite = benchmark.get_benchmark_dict()["libero_goal"]()
    tasks = [suite.get_task(i) for i in range(suite.n_tasks)]
    missing = [t.language for t in tasks if t.language not in paraphrases]
    if missing:
        raise RuntimeError(f"no LIBERO-Para rows for: {missing}")

    (out / "run_config.json").write_text(json.dumps({
        **vars(args), "para_csv": str(para_csv), "layers": layer_ids, "is_adapter": is_adapter,
        "git": subprocess.run(["git", "-C", os.environ.get("VLAA", "."), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))

    rows, action_rows = [], []
    for tid, task in enumerate(tasks):
        env, _ = get_libero_env(task, "openvla", resolution=256)
        init_states = suite.get_task_init_states(tid)
        for k in range(args.n_init):
            env.reset()
            obs = env.set_init_state(init_states[k])
            for _ in range(args.num_steps_wait):
                obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
            observation, _ = prepare_observation(obs, get_image_resize_size(cfg))

            ref, a_ref = run(observation, task.language)
            others = [run(observation, t.language) for t in tasks if t is not task]
            d_other = {c: np.nanmean([dist(ref[c], o[0][c]) for o in others], axis=0) for c in CLASSES}
            a_other = float(np.mean([np.linalg.norm(o[1] - a_ref) for o in others]))

            probes = [("empty", "", "", "", "")] + [
                ("para", f'{p["high"]}/{p["mid"]}/{p["low"]}/{p["batch_idx"]}', p["high"], p["mid"],
                 p["new_instruction"]) for p in paraphrases[task.language]]
            for kind, pid, high, mid, text in probes:
                rep, a = run(observation, text)
                r_action = float(ratio(np.linalg.norm(a - a_ref), a_other))
                action_rows.append({"group": "empty" if kind == "empty" else high, "R_action": r_action})
                for c in CLASSES:
                    d = dist(ref[c], rep[c])
                    for li, dv, do, rv in zip(layer_ids, d, d_other[c], ratio(d, d_other[c])):
                        rows.append({"task_id": tid, "init": k, "kind": kind, "para_id": pid, "high": high,
                                     "mid": mid, "instruction": text, "layer": li, "cls": c,
                                     "d": dv, "d_other": do, "R": rv, "R_action": r_action})
            print(f"task {tid} init {k}: {len(probes)} probes", flush=True)
        env.close()

    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # Summary: mean R per (kind/high, class, layer), plus R_action.
    agg = defaultdict(list)
    for r in rows:
        agg[(r["kind"] if r["kind"] == "empty" else r["high"], r["cls"], r["layer"])].append(r["R"])
    for r in action_rows:
        agg[(r["group"], "R_action", -1)].append(r["R_action"])
    summary = [{"group": g, "cls": c, "layer": l, "mean_R": float(np.nanmean(v)), "n": len(v)}
               for (g, c, l), v in sorted(agg.items())]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    for s in summary:
        if s["cls"] in ("action", "R_action"):
            print(s)


if __name__ == "__main__":
    main()
