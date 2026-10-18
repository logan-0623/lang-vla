#!/bin/bash
# Weights and data for StarVLA (login node is fine; needs `hf`, installed in the starvla env).
#   bash scripts/download_starvla_assets.sh bridges2   # base VLM, released 4in1 OFT checkpoint, LIBERO-Goal LeRobot data
#   bash scripts/download_starvla_assets.sh deltaai    # base VLM + LIBERO-Goal LeRobot data
set -euo pipefail
CLUSTER=${1:?bridges2 or deltaai}
source "$(dirname "$0")/../$CLUSTER/config.sh"
conda activate "$STORE/envs/starvla"
hf download Qwen/Qwen3-VL-4B-Instruct --local-dir "$STORE/ckpt/Qwen3-VL-4B-Instruct"
[ "$CLUSTER" = bridges2 ] && hf download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 --local-dir "$STORE/ckpt/Qwen3-VL-OFT-LIBERO-4in1"
D=$STORE/data/starvla_libero/libero_goal_no_noops_1.0.0_lerobot
# ~1k small per-episode video files: the Hub allows 1000 API requests per 5 min, so download with few workers and
# retry after the window (completed files are skipped on each retry).
for try in 1 2 3 4 5 6; do
  hf download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot --repo-type dataset --local-dir "$D" --max-workers 2 && break
  echo "download attempt $try failed (rate limit?); retrying in 6 min"; sleep 360
done
cp "$STORE/src/starVLA/examples/simBenchmarks/LIBERO/train_files/modality.json" "$D/meta/"
python - "$D" <<'PY'
import json, pathlib, sys
d = pathlib.Path(sys.argv[1]); n = json.load(open(d / "meta/info.json"))["total_episodes"]
vids = {p.parent.name: 0 for p in d.glob("videos/*/*/*.mp4")}
for p in d.glob("videos/*/*/*.mp4"):
    vids[p.parent.name] += 1
print(f"episodes {n}; videos per camera {vids}")
assert vids and all(v == n for v in vids.values()), "dataset incomplete: rerun this script"
PY
