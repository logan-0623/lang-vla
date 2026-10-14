#!/bin/bash
# E0 / E1: upstream LIBERO eval for VLA-Adapter or OpenVLA-OFT, plus a summary with error count and throughput.
#   sbatch --export=ALL,MODEL=vla-adapter,CKPT_NAME=LIBERO-Goal-Pro,SUITE=libero_goal bridges2/eval_libero.sbatch
#   sbatch --export=ALL,MODEL=oft,CKPT_NAME=openvla-7b-oft-finetuned-libero-goal,SUITE=libero_goal bridges2/eval_libero.sbatch
# VLA-Adapter only: MODE=orig|empty|other|shuffle (patched eval), PROCS=k processes sharing the GPU (tasks split round-robin).
# Also callable with plain `bash` inside an allocation (bench_procs.sbatch does this).
#SBATCH -A cis260349p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -J lv-eval
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/bridges2/config.sh"
use_env vla
MODEL=${MODEL:?vla-adapter|oft} SUITE=${SUITE:-libero_goal} TRIALS=${TRIALS:-50} PROCS=${PROCS:-1} MODE=${MODE:-orig}
CK=$CKPT/${CKPT_NAME:?}
OUT=$RESULTS/eval/${TAG:-$MODEL-${CKPT_NAME}-$SUITE-$MODE}-${SLURM_JOB_ID:-local}
mkdir -p "$OUT" && cd "$OUT"                     # upstream writes ./rollouts/*.mp4 relative to cwd
echo "nproc=$(nproc) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"; nvidia-smi -L

case $MODEL in
  vla-adapter)
    export PYTHONPATH="$VLAA"
    ARGS=(--use_proprio True --num_images_in_input 2 --use_film False --use_pro_version "${PRO:-True}"
          --instruction_mode "$MODE") ;;
  oft)
    [ "$MODE" = orig ] || { echo "MODE=$MODE needs the patched eval: VLA-Adapter only" >&2; exit 1; }
    [ "$PROCS" = 1 ] || { echo "PROCS>1 needs task_ids: VLA-Adapter only" >&2; exit 1; }
    export PYTHONPATH="$OFT"
    ARGS=(--use_proprio True --num_images_in_input 2 --center_crop True) ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac

START=$(date +%s)
for ((p = 0; p < PROCS; p++)); do
  IDS=()
  ((PROCS > 1)) && IDS=(--task_ids "$(seq -s, $p "$PROCS" 9)")
  python "$PYTHONPATH/experiments/robot/libero/run_libero_eval.py" \
    --pretrained_checkpoint "$CK" --task_suite_name "$SUITE" --num_trials_per_task "$TRIALS" \
    --local_log_dir "$OUT/logs" --run_id_note "p$p" "${ARGS[@]}" ${IDS[@]+"${IDS[@]}"} ${EXTRA_ARGS:-} \
    > "$OUT/proc$p.log" 2>&1 &
done
wait
ELAPSED=$(( $(date +%s) - START ))

python - "$OUT" "$ELAPSED" <<'PY'
import json, pathlib, re, sys
out, elapsed = pathlib.Path(sys.argv[1]), int(sys.argv[2])
text = "\n".join(p.read_text() for p in (out / "logs").glob("*.txt"))
succ = re.findall(r"^Success: (True|False)", text, re.M)
errors = len(re.findall(r"^Episode error", text, re.M))
n, k = len(succ), succ.count("True")
summary = {"episodes": n, "successes": k, "success_rate": k / n if n else None,
           "episode_errors": errors, "elapsed_s": elapsed,
           "rollouts_per_gpu_hour": round(n / elapsed * 3600, 1) if elapsed else None}
(out / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
# Errored episodes count as failures upstream; they must be reported, not silently averaged in.
if errors: print(f"WARNING: {errors} episodes raised exceptions (counted as failures by upstream)")
PY
