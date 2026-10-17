"""VLA-Adapter or OpenVLA-OFT through their own LIBERO eval code.

Whichever repo is first on PYTHONPATH is used (both expose the same `experiments.robot` API). Observation
preparation, action un-normalization and gripper post-processing are upstream's functions, unchanged.
"""
from __future__ import annotations

import copy

import numpy as np


class UpstreamPolicy:
    def __init__(self, checkpoint: str, suite: str):
        from experiments.robot.libero.run_libero_eval import (
            GenerateConfig, initialize_model, prepare_observation, process_action)
        from experiments.robot.robot_utils import get_action, get_image_resize_size

        self.is_adapter = "use_minivlm" in GenerateConfig.__dataclass_fields__
        kw = {"use_pro_version": True} if self.is_adapter else {"center_crop": True}
        self.cfg = GenerateConfig(pretrained_checkpoint=checkpoint, task_suite_name=suite,
                                  use_proprio=True, num_images_in_input=2, **kw)
        # Upstream's get_vla rewrites <checkpoint>/config.json in place (update_auto_map). Jobs starting together on one
        # checkpoint raced on it (JSONDecodeError on an empty file, E4 jobs 49344275/7): serialize loading per checkpoint.
        # The lock is an O_EXCL-created file (atomic on Lustre/Ocean, where flock may be unsupported); a lock older than
        # 15 min is treated as left behind by a killed job.
        import os
        import time
        from pathlib import Path
        lock = Path(checkpoint) / ".langvla-load.lock"
        while True:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                break
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > 900:
                        lock.unlink()
                except FileNotFoundError:
                    pass
                time.sleep(5)
        try:
            self.model, self.action_head, self.proprio_projector, _, self.processor = initialize_model(self.cfg)
        finally:
            lock.unlink(missing_ok=True)
        self._prepare, self._process, self._get_action = prepare_observation, process_action, get_action
        self._resize = get_image_resize_size(self.cfg)
        self.deterministic = True     # L1-regression heads: no sampling

    def act(self, obs: dict, instruction: str) -> list[np.ndarray]:
        observation, _ = self._prepare(obs, self._resize)
        extra = {"use_minivlm": self.cfg.use_minivlm} if self.is_adapter else {}
        # get_vla_action normalizes observation["state"] in place: never hand it a dict we reuse.
        actions = self._get_action(self.cfg, self.model, copy.deepcopy(observation), instruction,
                                   processor=self.processor, action_head=self.action_head,
                                   proprio_projector=self.proprio_projector, use_film=self.cfg.use_film, **extra)
        # Upstream queues the chunk in deque(maxlen=num_open_loop_steps), which keeps the last n actions.
        return [self._process(np.asarray(a), self.cfg.model_family) for a in list(actions)[-self.cfg.num_open_loop_steps:]]

    def instruction_embedding(self, text: str) -> np.ndarray:
        """Mean input-embedding of the instruction's tokens (no prompt template); used to match perturbation size."""
        import torch
        ids = self.processor.tokenizer(text.lower(), add_special_tokens=False).input_ids
        if not ids:
            return np.zeros(self.model.get_input_embeddings().weight.shape[1])
        with torch.inference_mode():
            e = self.model.get_input_embeddings()(torch.tensor(ids, device=self.model.device))
        return e.float().mean(0).cpu().numpy()
