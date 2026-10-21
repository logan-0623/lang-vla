#!/bin/bash
# Batch 2026-10-22: targeted activation patching at the action-head input (RESEARCH_PLAN §11.7, P0-P3 written before
# these results). Same observation, instruction A executed, its action queries patched with donor B's along the
# object subspace (instructions differing only in the object) or the destination subspace, from the batch_1021
# instruction-swap groups (scripts/erase_action.py subspaces). ~10 GPU-h.
#   a: s4 (bowl -> cabinet top) <- s2 (wine -> cabinet top);  n2 (cheese -> cabinet top, done as s4) <- s6 (cheese -> bowl)
#   b: s4 <- s1 (bowl -> stove)
#   patches: none (batched pair, no change), full (all of B's queries), obj, dest, random (rank 4 each)
#   git pull ; bash bridges2/batch_1023.sh       (afterwards: bash bridges2/batch_1023.sh report)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
E=$RESULTS/erase
P=$E/patch
PATCHES="none full obj dest random"
if [ "${1:-}" = report ]; then
  cat logs/er-subspaces-*.out | grep -v "^nproc\|^GPU"
  python scripts/erase_action.py patch_report $(for p in $PATCHES; do echo $P/$p-a $P/$p-b; done)
  exit
fi
[ -e $P ] && { echo "$P exists; move it away first" >&2; exit 1; }
mkdir -p $P
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 >> logs/batch_1023.txt; echo "$id"; }
s=$(sub -t 0:30:00 -J er-subspaces --export=ALL,ENV=starvla bridges2/run.sbatch \
  python scripts/erase_action.py subspaces --rec $E/rec.f16 --labels $E/rec.jsonl --run_dir $E/none --out $P/subspaces.npz)
for p in $PATCHES; do
  for x in "a s4=s2,n2=s6 s4,n2" "b s4=s1 s4"; do
    set -- $x
    sub -t 3:00:00 -J er-patch-$p-$1 --dependency=afterok:$s \
      --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,LV_PATCH_U=$P/subspaces.npz:$p,LV_PATCH=$2,LV_LABELS=$P/$p-$1/labels.jsonl \
      bridges2/run_starvla.sbatch python scripts/eval_goals.py --backend starvla --checkpoint SERVER --sets seen,novel \
      --goals $3 --n_init 10 --out $P/$p-$1 > /dev/null
  done
done
echo "submitted subspaces $s -> 10 patch runs; ids in logs/batch_1023.txt"
