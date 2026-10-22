#!/bin/bash
# Batch 2026-10-22: does the batch_1023 repair generalise? Every failing novel goal with a seen counterpart, patched
# along the object subspace from the seen goal with its object, or along the destination subspace from the seen goal
# with its destination (RESEARCH_PLAN §11.7 Q0-Q3, written before these results). Subspaces from batch_1023. ~4 GPU-h.
#   obj:  n0,n6 <- s6 (cheese -> bowl);  n3,n4 <- s2 (wine -> cabinet top)
#   dest: n0,n3 <- s8 (bowl -> plate);  n4 <- s1 (bowl -> stove);  n5 <- s5 (plate -> stove front);  n6 <- s3 (bowl -> top drawer)
#   git pull ; bash bridges2/batch_1024.sh       (afterwards: bash bridges2/batch_1024.sh report)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
E=$RESULTS/erase
U=$E/patch/subspaces.npz
P=$E/patch2
OBJ="n0=s6,n3=s2,n4=s2,n6=s6"
DEST="n0=s8,n3=s8,n4=s1,n5=s5,n6=s3"
RUNS="none:dest:$DEST obj:obj:$OBJ random-obj:random:$OBJ dest:dest:$DEST random-dest:random:$DEST"
if [ "${1:-}" = report ]; then
  python scripts/erase_action.py patch_report $(for r in $RUNS; do echo $P/${r%%:*}; done)
  exit
fi
[ -f $U ] || { echo "no $U; run batch_1023 first" >&2; exit 1; }
[ -e $P ] && { echo "$P exists; move it away first" >&2; exit 1; }
mkdir -p $P
for r in $RUNS; do
  IFS=: read -r name u map <<< "$r"
  export LV_PATCH="$map"      # not in --export: Slurm splits its list on commas
  [ $name = none ] && u=none
  id=$(sbatch --parsable -t 3:00:00 -J er-$name \
    --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,LV_PATCH_U=$U:$u,LV_LABELS=$P/$name/labels.jsonl \
    bridges2/run_starvla.sbatch python scripts/eval_goals.py --backend starvla --checkpoint SERVER --sets novel \
    --goals $(echo "$map" | sed 's/=s[0-9]//g') --n_init 20 --out $P/$name)
  echo "$id $name $u $map" | tee -a logs/batch_1024.txt
done
