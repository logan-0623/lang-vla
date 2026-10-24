#!/bin/bash
# RESEARCH_PLAN §11.8 D group (2026-10-24): after a stable grasp, with the object abstracted ("it" / "the object"),
# does the held object follow the destination word to a destination other than the original goal's?
# From the batch_1026 StarVLA states (n0, n3, n4). Controls: batch_1027 none-A, batch_1029 obj-it / obj-generic.
#   bash bridges2/batch_1032.sh run       4 jobs
#   bash bridges2/batch_1032.sh report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
S=$RESULTS/erase/branch/states
D=$RESULTS/erase/branch5
run() { sbatch --parsable -t 3:00:00 -J $1 --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,LV_PATCH_U= \
  bridges2/run_starvla.sbatch python scripts/patch_branch.py "${@:2}" --backend starvla --checkpoint SERVER; }
unset LV_PATCH
mkdir -p $D
case ${1:-} in
  run)
    [ -f $S/states.jsonl ] || { echo "no $S" >&2; exit 1; }
    for spec in "it-alt1|n0=put it on the stove;n3=put it on the stove;n4=put it on the plate" \
                "it-alt2|n0=put it on top of the cabinet;n3=put it on top of the cabinet;n4=put it on top of the cabinet" \
                "obj-alt1|n0=put the object on the stove;n3=put the object on the stove;n4=put the object on the plate" \
                "obj-alt2|n0=put the object on top of the cabinet;n3=put the object on top of the cabinet;n4=put the object on top of the cabinet"; do
      c=${spec%%|*}; l=${spec#*|}
      [ -e $D/$c ] && { echo "$D/$c exists; skipped" >&2; continue; }
      echo "$(run br5-$c branch --states $S --out $D/$c --lang "$l") $c" | tee -a logs/batch_1032.txt
    done ;;
  report)
    python scripts/patch_branch.py report $(ls -d $D/*/) $RESULTS/erase/branch2/none-A ;;
  *) echo "usage: $0 run | report" >&2; exit 1 ;;
esac
