#!/bin/bash
# Run any command on one L40S. ENV picks the conda env; REPO puts that model repo first on PYTHONPATH.
#   sbatch --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch \
#     python scripts/paraphrase_invariance.py --checkpoint /path/to/ckpt --out /path/to/out
#SBATCH -A cis260349p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -J lv-run
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/bridges2/config.sh"
use_env "${ENV:-vla}"
case ${REPO:-} in
  vla-adapter) export PYTHONPATH="$VLAA" ;;
  oft)         export PYTHONPATH="$OFT" ;;
  "")          ;;
  *) echo "unknown REPO=$REPO" >&2; exit 1 ;;
esac
cd "$LANGVLA_ROOT"
echo "nproc=$(nproc)"; nvidia-smi -L
"$@"
