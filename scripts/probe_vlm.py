"""Linear probes: are the spatial / state facts of the QA data linearly decodable from a VLM's hidden states, even
when the model cannot say them? Compares the raw Qwen3-VL with the VLM inside a VLA checkpoint (the released
StarVLA 4in1, fine-tuned for actions from the same base): knowledge in the representation vs in language.

  extract (GPU, ~10 min): one forward per frame of frames.jsonl with a neutral prompt ("Describe the scene."),
    hidden states of every 4th layer, mean over the image tokens and at the last token. Layer 0 is the input
    embedding, i.e. the vision encoder's output at the image positions.
  fit (CPU, login node): ridge probes trained on the QA train scenes, scored on the held-out scenes (the QA split,
    scene % 5 == 4); the ridge penalty is picked on train scenes with scene % 5 == 3. Classification by argmax of a
    one-hot ridge fit (accuracy, vs the majority class); regression by R^2 (0 = predicting the train mean).
    `--pixels` adds a probe on the 32x32 downsampled image, the floor for "it is in the pixels anyway".

  python scripts/probe_vlm.py extract --base $STORE/ckpt/Qwen3-VL-4B-Instruct --qa_dir $STORE/qa/goal-v1 --out $RESULTS/probe/raw.npz
  python scripts/probe_vlm.py extract ... --ckpt $CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt --out $RESULTS/probe/4in1.npz
  python scripts/probe_vlm.py fit --qa_dir $STORE/qa/goal-v1 --feats raw=$RESULTS/probe/raw.npz 4in1=$RESULTS/probe/4in1.npz --pixels
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
MOVABLE = ["akita_black_bowl_1", "cream_cheese_1", "wine_bottle_1", "plate_1"]
PROMPT = "Describe the scene."
ALPHAS = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5]


def frames(qa_dir: Path) -> list[dict]:
    return [json.loads(l) for l in open(qa_dir / "frames.jsonl")]


def extract(a) -> None:
    import torch
    from eval_qa import load_image, load_vlm

    model, proc = load_vlm(a.base, a.ckpt)
    fr = frames(Path(a.qa_dir))
    img_tok = model.config.image_token_id
    layers = None
    mean, last = [], []
    for i, f in enumerate(fr):
        img = load_image(Path(a.qa_dir) / f["image"], a.upscale)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt").to(model.device)
        with torch.inference_mode():
            hs = model(**inp, output_hidden_states=True).hidden_states
        layers = layers or list(range(0, len(hs), 4))
        is_img = inp["input_ids"][0] == img_tok
        mean.append(torch.stack([hs[l][0, is_img].float().mean(0) for l in layers]).half().cpu().numpy())
        last.append(torch.stack([hs[l][0, -1].float() for l in layers]).half().cpu().numpy())
        if i % 500 == 0:
            print(f"{i}/{len(fr)} frames; image tokens {int(is_img.sum())}; layers {layers}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, mean=np.stack(mean), last=np.stack(last), layers=np.array(layers),
             images=np.array([f["image"] for f in fr]), args=json.dumps(vars(a)))
    print(f"wrote {a.out}: mean {np.stack(mean).shape}")


def labels(fr: list[dict]) -> dict[str, tuple[str, np.ndarray]]:
    """name -> (kind, y): kind 'cls' (int classes) or 'reg' (float, n x k)."""
    F = [f["facts"] for f in fr]
    out = {"held": ("cls", np.array([0 if x["held"] is None else 1 + MOVABLE.index(x["held"]) for x in F]))}
    for d in ("top", "middle", "bottom"):
        out[f"drawer_{d}_open"] = ("cls", np.array([int(x["drawers"][d]) for x in F]))
    out["gripper_xyz"] = ("reg", np.array([x["eef"] for x in F], float))
    for o in MOVABLE:
        short = o.split("_")[-2] if o != "akita_black_bowl_1" else "bowl"
        out[f"{short}_image_rc"] = ("reg", np.array([x["rc"][o] for x in F], float))
        out[f"{short}_world_xy"] = ("reg", np.array([x["pos"][o][:2] for x in F], float))
        out[f"{short}_gripper_dist"] = ("reg", np.array([[np.linalg.norm(np.subtract(x["pos"][o], x["eef"]))] for x in F]))
    for a, b in combinations(MOVABLE, 2):                 # image left/right of each pair (column order)
        out[f"lr_{a.split('_')[-2]}_{b.split('_')[-2]}"] = ("cls", np.array([int(x["rc"][a][1] < x["rc"][b][1]) for x in F]))
    return out


class Ridge:
    """Ridge for many targets and penalties from one SVD of the (standardized) training rows."""

    def __init__(self, X):
        self.U, self.s, self.Vt = np.linalg.svd(X, full_matrices=False)

    def coef(self, Y, alpha):
        return self.Vt.T @ ((self.s / (self.s ** 2 + alpha))[:, None] * (self.U.T @ Y))


def target(kind, y):
    return np.eye(int(y.max()) + 1)[y] if kind == "cls" else y.reshape(len(y), -1).astype(float)


def score(kind, P, y, Ytr_mean):
    if kind == "cls":
        return float((P.argmax(1) == y).mean())
    Y = y.reshape(len(y), -1)
    return float(1 - ((P - Y) ** 2).sum() / ((Y - Ytr_mean) ** 2).sum())


def probes(X, labs, tr, va, te) -> dict[str, float]:
    """Held-out score of a ridge probe per label; penalty chosen on `va` (a subset of `tr`)."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    X = (X - mu) / sd
    fit_tr = tr & ~va
    r_fit, r_all = Ridge(X[fit_tr]), Ridge(X[tr])
    out = {}
    for ln, (kind, y) in labs.items():
        Y = target(kind, y)
        m_fit, m_all = Y[fit_tr].mean(0), Y[tr].mean(0)        # centring stands in for the intercept
        best = max(ALPHAS, key=lambda al: score(kind, X[va] @ r_fit.coef(Y[fit_tr] - m_fit, al) + m_fit, y[va], m_fit))
        out[ln] = score(kind, X[te] @ r_all.coef(Y[tr] - m_all, best) + m_all, y[te], m_all)
    return out


def fit(a) -> None:
    qa = Path(a.qa_dir)
    fr = frames(qa)
    scene = np.array([f["scene"] for f in fr])
    te, va = scene % 5 == 4, scene % 5 == 3
    tr = ~te
    labs = {}
    for ln, (kind, y) in labels(fr).items():
        if kind == "cls" and (len(np.unique(y)) < 2 or min(np.bincount(y[te], minlength=int(y.max()) + 1)[np.unique(y)]) < 5):
            print(f"skip {ln}: one class only, or a class has <5 held-out frames")
            continue
        labs[ln] = (kind, y)
    res = {ln: {"kind": k, "floor": float(np.bincount(y[te]).max() / te.sum()) if k == "cls" else 0.0}
           for ln, (k, y) in labs.items()}
    if a.pixels:
        from PIL import Image
        px = np.stack([np.asarray(Image.open(qa / f["image"]).convert("RGB").resize((32, 32), Image.BILINEAR), float).ravel()
                       for f in fr])
        for ln, v in probes(px, labs, tr, va, te).items():
            res[ln]["pixels"] = v
    for spec in a.feats:
        name, path = spec.split("=", 1)
        z = np.load(path)
        assert list(z["images"]) == [f["image"] for f in fr], f"{path}: frames differ from {qa}/frames.jsonl"
        for pool in ("mean", "last"):
            for i, layer in enumerate(z["layers"]):
                for ln, v in probes(z[pool][:, i].astype(float), labs, tr, va, te).items():
                    res[ln].setdefault(f"{name}.{pool}", {})[int(layer)] = v
            print(f"done {name}.{pool}", flush=True)
    for ln, row in res.items():
        cols = [f"floor {row['floor']:.3f}"] + ([f"pixels {row['pixels']:.3f}"] if "pixels" in row else [])
        for k, v in row.items():
            if isinstance(v, dict):
                L = max(v, key=v.get)
                cols.append(f"{k} {v[L]:.3f}@L{L}")
        print(f"{ln:<24} {row['kind']}  " + "  ".join(cols))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--base", required=True)
    e.add_argument("--ckpt", default=None)
    e.add_argument("--qa_dir", required=True)
    e.add_argument("--upscale", type=int, default=1)
    e.add_argument("--out", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--qa_dir", required=True)
    f.add_argument("--feats", nargs="+", required=True, help="name=path.npz")
    f.add_argument("--pixels", action="store_true")
    f.add_argument("--out", default=None, help="json with every layer's score")
    a = ap.parse_args()
    extract(a) if a.cmd == "extract" else fit(a)


if __name__ == "__main__":
    main()
