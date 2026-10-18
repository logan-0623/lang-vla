#!/bin/bash
# StarVLA policy server (starvla env) + one client command (vla env, LIBERO) on one L40S. The literal argument
# SERVER is replaced by the server's host:port.
#   sbatch --export=ALL,SV_CKPT=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt \
#     bridges2/run_starvla.sbatch python scripts/eval_goals.py --backend starvla --checkpoint SERVER --out ...
#SBATCH -A cis260349p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -J lv-starvla
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/bridges2/config.sh"
SRC=$STORE/src/starVLA
PORT=$((20000 + SLURM_JOB_ID % 20000))
nvidia-smi -L

conda activate "$STORE/envs/starvla"
( cd "$SRC" && PYTHONPATH=$SRC python deployment/model_server/server_policy.py --ckpt_path "${SV_CKPT:?}" \
    --port "$PORT" --use_bf16 --seed 7 --idle_timeout -1 \
    --config_override framework.qwenvl.attn_implementation=sdpa ) > "logs/starvla-server-$SLURM_JOB_ID.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
for _ in $(seq 120); do                     # model load: up to 20 min
  (echo > /dev/tcp/127.0.0.1/$PORT) 2>/dev/null && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died:"; tail -40 "logs/starvla-server-$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done

use_env vla
export PYTHONPATH="$SRC"
cd "$LANGVLA_ROOT"
args=()
for a in "$@"; do [ "$a" = SERVER ] && args+=("127.0.0.1:$PORT") || args+=("$a"); done
"${args[@]}"
