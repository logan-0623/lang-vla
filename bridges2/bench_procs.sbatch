#!/bin/bash
# E0 throughput: rollouts per GPU-hour for VLA-Adapter with 1/2/4/8 processes on one L40S (3 trials x 10 tasks each).
# Pick the best PROCS for configs/eval.yaml. CPU per GPU on GPU-shared is what limits this; nproc is printed.
#   sbatch bridges2/bench_procs.sbatch
#SBATCH -A cis260349p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -J lv-bench
set -euo pipefail
cd "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}"
for p in ${PROCS_LIST:-1 2 4 8}; do
  MODEL=vla-adapter CKPT_NAME=${CKPT_NAME:-LIBERO-Goal-Pro} TRIALS=3 PROCS=$p TAG=bench-p$p bash bridges2/eval_libero.sbatch
done
source bridges2/config.sh; grep -h rollouts_per_gpu_hour "$RESULTS"/eval/bench-p*-"$SLURM_JOB_ID"/summary.json 2>/dev/null || true
