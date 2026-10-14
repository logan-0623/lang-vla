#!/bin/bash
# Platform check (~30 min, ~0.5 SU on L40S). Stops at the first failure:
#   devices -> LIBERO + EGL -> VLA-Adapter forward -> OFT forward -> 10 VLA-Adapter rollouts -> LeRobot pi0.5 1 episode
#   sbatch bridges2/smoke.sbatch
#SBATCH -A cis260349p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -J lv-smoke
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/bridges2/config.sh"
OUT=$RESULTS/smoke-${SLURM_JOB_ID:-local}; mkdir -p "$OUT" && cd "$OUT"

echo "== [1/6] devices"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>} MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-<unset>} nproc=$(nproc)"
nvidia-smi -L

use_env vla
# One forward pass on a real LIBERO-Goal observation. $1 = repo, $2 = checkpoint, $3 = extra GenerateConfig kwargs (JSON).
forward() {
  PYTHONPATH="$1" python - "$2" "$3" <<'PY'
import copy, json, sys, time, torch
from libero.libero import benchmark
from experiments.robot.libero.libero_utils import get_libero_env
from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, prepare_observation
from experiments.robot.robot_utils import get_action, get_image_resize_size

suite = benchmark.get_benchmark_dict()["libero_goal"]()
env, desc = get_libero_env(suite.get_task(0), "openvla", resolution=256)
env.reset(); obs = env.set_init_state(suite.get_task_init_states(0)[0])
t = time.time()
for _ in range(20): obs, *_ = env.step([0, 0, 0, 0, 0, 0, -1])
assert obs["agentview_image"].std() > 1, "blank render: EGL is not working"
print(f"LIBERO ok: '{desc}', {20/(time.time()-t):.1f} sim steps/s")

cfg = GenerateConfig(pretrained_checkpoint=sys.argv[1], task_suite_name="libero_goal",
                     use_proprio=True, num_images_in_input=2, **json.loads(sys.argv[2]))
model, action_head, proprio_projector, _, processor = initialize_model(cfg)
print("LLM attention implementation:", model.language_model.config._attn_implementation)
observation, _ = prepare_observation(obs, get_image_resize_size(cfg))
t = time.time()
# get_vla_action normalizes obs["state"] in place: always pass a copy.
actions = get_action(cfg, model, copy.deepcopy(observation), desc, processor=processor, action_head=action_head,
                     proprio_projector=proprio_projector, use_film=cfg.use_film,
                     **({"use_minivlm": cfg.use_minivlm} if hasattr(cfg, "use_minivlm") else {}))
torch.cuda.synchronize()
print(f"forward ok in {time.time()-t:.2f}s: {len(actions)} x {len(actions[0])}, "
      f"peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
env.close()
PY
}

echo "== [2-3/6] LIBERO + EGL, VLA-Adapter forward"
forward "$VLAA" "$CKPT/LIBERO-Goal-Pro" '{"use_pro_version": true}'
echo "== [4/6] OpenVLA-OFT forward"
forward "$OFT" "$CKPT/openvla-7b-oft-finetuned-libero-goal" '{"center_crop": true}'

echo "== [5/6] VLA-Adapter rollouts: 1 trial x 10 tasks"
PYTHONPATH="$VLAA" python "$VLAA/experiments/robot/libero/run_libero_eval.py" \
  --pretrained_checkpoint "$CKPT/LIBERO-Goal-Pro" --task_suite_name libero_goal \
  --use_proprio True --num_images_in_input 2 --use_film False --use_pro_version True \
  --num_trials_per_task 1 --local_log_dir "$OUT/logs"
grep -h "^Success\|Episode error" "$OUT"/logs/*.txt | sort | uniq -c

echo "== [6/6] LeRobot pi0.5: 1 episode of libero_goal task 0"
conda deactivate; use_env lerobot
lerobot-eval --output_dir="$OUT/lerobot" --env.type=libero --env.task=libero_goal --env.task_ids='[0]' \
  --eval.n_episodes=1 --eval.batch_size=1 --policy.path="$CKPT/pi05_libero_finetuned_v044" \
  --policy.device=cuda --policy.n_action_steps=10
echo "smoke done"
