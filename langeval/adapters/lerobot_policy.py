"""LeRobot policies (pi0.5, SmolVLA, X-VLA) through lerobot-eval's own pipeline (lerobot @ ae2b57f).

Per inference: LiberoEnv._format_raw_obs -> preprocess_observation -> task -> env preprocessor (LiberoProcessorStep:
180-degree image flip, eef pos + axis-angle + gripper state) -> policy preprocessor -> select_action x n_action_steps
-> policy postprocessor -> env postprocessor, exactly as lerobot_eval.rollout. The env is LeRobot's own LiberoEnv
(360x360 cameras, relative control); scripts drive its inner OffScreenRenderEnv, like the upstream adapter's env.

Sampling noise: flow policies draw noise from torch's global RNG. `begin()` (called by langeval.branch.rollout at
every (re)start) resets a call counter, and the k-th inference of a rollout uses seed (seed, k). Two continuations
from one state therefore share their noise draws (common random numbers): orig vs orig_rep is a real check, and
instruction contrasts are not diluted by sampling noise. Different `seed`s give independent replicates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _batch(x):
    """Add a leading batch dim of 1 to every array in a nested dict."""
    return {k: _batch(v) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x)[None]


class LeRobotPolicy:
    def __init__(self, checkpoint: str, suite: str, *, device: str = "cuda", n_action_steps: int = 10,
                 compile_model: bool = False, seed: int = 7):
        import torch
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies import make_policy, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(checkpoint)
        cfg.pretrained_path = Path(checkpoint)
        cfg.device = device
        if hasattr(cfg, "n_action_steps"):
            cfg.n_action_steps = n_action_steps       # pi0.5 on LIBERO: 10, as in eval_lerobot.sbatch
        if hasattr(cfg, "compile_model"):
            cfg.compile_model = compile_model         # max-autotune took ~10 min/episode on L40S
        self.env_cfg = LiberoEnvConfig(task=suite)
        self.policy = make_policy(cfg=cfg, env_cfg=self.env_cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg, pretrained_path=cfg.pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(self.policy.config.device)},
                                    "rename_observations_processor": {"rename_map": {}}})
        self.env_pre, self.env_post = make_env_pre_post_processors(env_cfg=self.env_cfg, policy_cfg=cfg)
        self.n = getattr(self.policy.config, "n_action_steps", n_action_steps)
        self.seed, self._calls = seed, 0
        self.deterministic = False
        self._torch = torch
        self._gym = None

    def make_env(self, task_suite, task_id: int, suite_name: str):
        """LeRobot's LiberoEnv for (suite, task); returns its inner OffScreenRenderEnv and the instruction."""
        from lerobot.envs.libero import LiberoEnv
        c = self.env_cfg
        self._gym = LiberoEnv(task_suite, task_id, suite_name, camera_name=c.camera_name, obs_type=c.obs_type,
                              observation_width=c.observation_width, observation_height=c.observation_height,
                              control_mode=c.control_mode)
        self._gym._ensure_env()
        return self._gym._env, task_suite.get_task(task_id).language

    def begin(self) -> None:
        self._calls = 0

    def act(self, obs: dict, instruction: str) -> list[np.ndarray]:
        from lerobot.envs.utils import preprocess_observation
        from lerobot.utils.constants import ACTION
        torch = self._torch
        # lerobot-eval steps a gym vector env, so every leaf arrives with a batch dim; LiberoProcessorStep
        # (e.g. _quat2axisangle) requires it. Add it here, as the vector env's stacking would.
        o = preprocess_observation(_batch(self._gym._format_raw_obs(obs)))
        o["task"] = [instruction]
        o = self.pre(self.env_pre(o))
        self.policy.reset()                                       # a (re)start always queries afresh
        torch.manual_seed(self.seed * 100_003 + self._calls)
        self._calls += 1
        out = []
        with torch.inference_mode():
            for _ in range(self.n):                               # first call fills the queue, the rest pop it
                a = self.post(self.policy.select_action(o))
                a = self.env_post({ACTION: a})[ACTION]
                out.append(a.float().cpu().numpy()[0])
        return out

    def instruction_embedding(self, text: str) -> np.ndarray:
        """Mean input embedding of the instruction's tokens (PaliGemma vocabulary); NaN if not reachable."""
        torch = self._torch
        try:
            from transformers import AutoTokenizer
            tok = getattr(self, "_tok", None) or AutoTokenizer.from_pretrained(self.policy.config.text_tokenizer_name)
            self._tok = tok
            emb = self.policy.model.paligemma_with_expert.paligemma.get_input_embeddings()
            ids = tok(text.lower(), add_special_tokens=False).input_ids
            if not ids:
                return np.zeros(emb.weight.shape[1])
            with torch.inference_mode():
                return emb(torch.tensor(ids, device=emb.weight.device)).float().mean(0).cpu().numpy()
        except Exception:
            return np.full(1, np.nan)
