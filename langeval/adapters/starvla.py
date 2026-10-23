"""StarVLA policies (Qwen3-VL backbones) through StarVLA's own policy server.

The model runs in the `starvla` env behind `deployment/model_server/server_policy.py` (websocket, returns
un-normalized action chunks); this client runs in the `vla` env next to LIBERO, so the simulator is the same one
the other models use. Observation and action handling copy StarVLA's LIBERO client exactly
(examples/simBenchmarks/LIBERO/eval_files/{eval_libero,model2libero_interface}.py @ 985c223): agentview and wrist
images flipped [::-1, ::-1] and resized to 224 (PIL bilinear), no proprio state, the whole chunk executed open
loop, gripper open = action > 0.5 -> -1.

`checkpoint` for this backend is the server address, host:port.
"""
from __future__ import annotations

import json

import numpy as np

LIBERO_ENV_RESOLUTION = 256
IMAGE_SIZE = (224, 224)
SWAP_EVERY = 10


def _goal_languages() -> dict[str, str]:
    """Goal name (s0-s9 in benchmark order, then the novel ids) -> instruction, as in scripts/eval_goals.py."""
    import json
    from pathlib import Path
    from langeval.goal_predicates import load_goals
    novel = json.load(open(Path(__file__).parents[2] / "configs/novel_goals.json"))["goals"]
    return {**{f"s{i}": g.language for i, g in enumerate(load_goals("libero_goal"))},
            **{g["id"]: g["language"] for g in novel}}


def _resize(img: np.ndarray) -> np.ndarray:
    from PIL import Image
    if img.shape[:2] == IMAGE_SIZE:
        return img
    return np.asarray(Image.fromarray(img).resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR))


class StarVLAPolicy:
    def __init__(self, address: str, suite: str, unnorm_key: str | None = None):
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
        host, port = address.rsplit(":", 1)
        self.client = WebsocketClientPolicy(host, int(port))
        meta = self.client.get_server_metadata()
        self.chunk = int(meta["action_chunk_size"])
        keys = meta.get("available_unnorm_keys") or []
        # Single-dataset checkpoints have one stats key (the released 4in1 ones: "franka"); else pick the suite's.
        self.unnorm_key = unnorm_key or meta.get("default_unnorm_key") or \
            (keys[0] if len(keys) == 1 else next((k for k in keys if suite in k), None))
        self.meta = meta
        self.deterministic = True     # OFT head, do_sample=False; checked at run time by orig vs orig_rep
        # LV_LABELS: one json line of the observation's *pos entries (positions, gripper qpos) per server call, in
        # call order, so that the server-side record of scripts/erase_action.py (same order) can be labelled.
        # LV_SWAP (with LV_LABELS): every SWAP_EVERY-th call of an episode, the same observation is also sent with
        # each other LIBERO-Goal / novel instruction (answers only recorded, never executed; RESEARCH_PLAN §11.7).
        import os
        self.labels = open(os.environ["LV_LABELS"], "a") if os.environ.get("LV_LABELS") else None
        self.swap = list(_goal_languages().values()) if self.labels and os.environ.get("LV_SWAP") else []
        # LV_PATCH="s4=s2,n2=s6+s6p0": when the instruction is goal A, the same observation goes to the server as a
        # batch [A, donor B1, B2, ...]; the server (scripts/erase_action.py, LV_PATCH_U) patches A's action queries
        # toward the donors' mean and only A's actions are executed (RESEARCH_PLAN §11.7). LV_PATCH_EXTRA: a json
        # {name: instruction} of extra donor names (e.g. paraphrases of a seen instruction).
        gl = _goal_languages() if os.environ.get("LV_PATCH") else {}
        if os.environ.get("LV_PATCH_EXTRA"):
            gl.update(json.load(open(os.environ["LV_PATCH_EXTRA"])))
        pairs = [kv.split("=") for kv in os.environ.get("LV_PATCH", "").split(",") if kv]
        self.patch = {gl[a]: [gl[x] for x in b.split("+")] for a, b in pairs}
        self.episode, self.call = -1, 0

    def begin(self) -> None:
        self.episode, self.call = self.episode + 1, 0

    def _query(self, imgs: list, lang: str) -> np.ndarray:
        examples = [{"image": imgs, "lang": l} for l in [lang, *getattr(self, "patch", {}).get(lang, [])]]
        resp = self.client.predict_action({"examples": examples,
                                           "unnorm_key": self.unnorm_key, "do_sample": False,
                                           "use_ddim": True, "num_ddim_steps": 10})
        return np.asarray(resp["data"]["actions"])[0][: self.chunk]

    def act(self, obs: dict, instruction: str) -> list[np.ndarray]:
        imgs = [_resize(np.ascontiguousarray(obs[k][::-1, ::-1])) for k in ("agentview_image", "robot0_eye_in_hand_image")]
        chunk = self._query(imgs, instruction)
        if self.labels:
            pos = {k: np.asarray(v).tolist() for k, v in obs.items() if k.endswith("pos")}
            rec = {"episode": self.episode, "call": self.call, "lang": instruction, "action": chunk.tolist(), **pos}
            self.labels.write(json.dumps(rec) + "\n")
            if self.call % SWAP_EVERY == 0:
                for alt in self.swap:
                    if alt != instruction:
                        self.labels.write(json.dumps({**rec, "lang": alt, "swap_of": instruction,
                                                      "action": self._query(imgs, alt).tolist()}) + "\n")
            self.labels.flush()
        self.call += 1
        out = []
        for a in chunk:
            g = 1.0 - 2.0 * float(a[6] > 0.5)
            out.append(np.concatenate([a[:6], [g]]).astype(np.float32))
        return out

    def instruction_embedding(self, text: str) -> np.ndarray:
        return np.full(1, np.nan)     # not exposed by the server; E3 perturbation matching falls back like pi0.5's


def make_env(task_suite, task_id: int):
    import pathlib
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task = task_suite.get_task(task_id)
    env = OffScreenRenderEnv(bddl_file_name=pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
                             camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION)
    env.seed(0)
    return env, task.language
