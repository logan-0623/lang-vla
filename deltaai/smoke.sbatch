#!/bin/bash
# End-to-end platform check (~20 min). Steps run in order and stop at the first failure:
#   devices -> RLDS read -> EGL render -> one model forward -> 10 rollouts with the official Goal-Pro checkpoint.
# Usage: sbatch -A $ACCOUNT deltaai/smoke.sbatch
#SBATCH --job-name=lv-smoke
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/deltaai/config.sh"
export PYTHONPATH="$VLAA:${PYTHONPATH:-}"
export CKPT=$STORE/ckpt/LIBERO-Goal-Pro
OUT=$STORE/smoke-$SLURM_JOB_ID; mkdir -p "$OUT" && cd "$OUT"

echo "== [1/5] devices"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-<unset>}"
nvidia-smi -L

echo "== [2/5] RLDS read (full prismatic pipeline, identity batch transform)"
python - <<'PY'
import os, time
from pathlib import Path
from prismatic.vla.datasets import RLDSDataset
t = time.time()
ds = RLDSDataset(Path(os.environ["DATA_ROOT"]), "libero_goal_no_noops", lambda b: b,
                 resize_resolution=(224, 224), shuffle_buffer_size=1000, image_aug=True)
b = next(iter(ds))
print(f"RLDS ok in {time.time()-t:.1f}s: instruction={b['task']['language_instruction']!r}",
      {k: v.shape for k, v in b["observation"].items() if hasattr(v, "shape")}, "action", b["action"].shape)
PY

echo "== [3/5] LIBERO + EGL render, [4/5] one model forward on that observation"
python - <<'PY'
import os, time, torch
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, prepare_observation
from experiments.robot.robot_utils import get_action, get_image_resize_size

suite = benchmark.get_benchmark_dict()["libero_goal"]()
env, desc = get_libero_env(suite.get_task(0), "openvla", resolution=256)
env.reset(); obs = env.set_init_state(suite.get_task_init_states(0)[0])
t = time.time()
for _ in range(50): obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
assert obs["agentview_image"].std() > 1, "blank render: EGL is not working"
print(f"LIBERO ok: '{desc}', {50/(time.time()-t):.1f} sim steps/s")

cfg = GenerateConfig(pretrained_checkpoint=os.environ["CKPT"], task_suite_name="libero_goal",
                     use_pro_version=True, use_proprio=True, num_images_in_input=2)
model, action_head, proprio_projector, _, processor = initialize_model(cfg)
print("LLM attention implementation:", model.language_model.config._attn_implementation)
observation, _ = prepare_observation(obs, get_image_resize_size(cfg))
t = time.time()
actions = get_action(cfg, model, observation, desc, processor=processor, action_head=action_head,
                     proprio_projector=proprio_projector, use_film=cfg.use_film, use_minivlm=cfg.use_minivlm)
torch.cuda.synchronize()
print(f"forward ok in {time.time()-t:.2f}s: {len(actions)} actions x {len(actions[0])} dims, "
      f"peak mem {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
env.close()
PY

echo "== [5/5] rollouts: 1 trial x 10 tasks"
python "$VLAA/experiments/robot/libero/run_libero_eval.py" \
  --pretrained_checkpoint "$CKPT" --task_suite_name libero_goal \
  --use_proprio True --num_images_in_input 2 --use_film False --use_pro_version True \
  --num_trials_per_task 1 --local_log_dir "$OUT/logs"
grep -i "success rate" -r "$OUT/logs" | tail -3
