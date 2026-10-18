#!/bin/bash
# Weights and data for StarVLA (login node is fine; needs `hf`, installed in the starvla env).
#   bash scripts/download_starvla_assets.sh bridges2   # base VLM + released 4in1 OFT checkpoint (adapter validation)
#   bash scripts/download_starvla_assets.sh deltaai    # base VLM + LIBERO-Goal LeRobot data (training)
set -euo pipefail
CLUSTER=${1:?bridges2 or deltaai}
source "$(dirname "$0")/../$CLUSTER/config.sh"
conda activate "$STORE/envs/starvla"
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "$STORE/ckpt/Qwen3-VL-4B-Instruct"
if [ "$CLUSTER" = bridges2 ]; then
  hf download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 --local-dir "$STORE/ckpt/Qwen3-VL-OFT-LIBERO-4in1"
else
  D=$STORE/data/starvla_libero/libero_goal_no_noops_1.0.0_lerobot
  hf download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot --repo-type dataset --local-dir "$D"
  cp "$STORE/src/starVLA/examples/simBenchmarks/LIBERO/train_files/modality.json" "$D/meta/"
fi
