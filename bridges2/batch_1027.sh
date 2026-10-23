#!/bin/bash
# RESEARCH_PLAN §11.7 stage 2b (2026-10-22). From stable-grasp states, nothing in the low-rank arm is re-run.
#   bash bridges2/batch_1027.sh swap       5 jobs on the batch_1026 states (n0, n3, n4): does the held object follow
#                                          the donor's destination? none-A also records q_A / q_B for `norms`
#   bash bridges2/batch_1027.sh collect    bowl grasp states for s4, s8, n5 (10 each)
#   bash bridges2/batch_1027.sh default    4 jobs on the bowl states: own / empty / nonce destination / no destination
#   bash bridges2/batch_1027.sh attn       2 jobs: attention dumps (eager) of one call per state x instruction
#   bash bridges2/batch_1027.sh report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
U=$RESULTS/erase/patch/subspaces2.npz
S=$RESULTS/erase/branch/states
B=$RESULTS/erase/branch/bowl
D=$RESULTS/erase/branch2
run() { sbatch --parsable -t 3:00:00 -J $1 --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,$2 \
  bridges2/run_starvla.sbatch python scripts/patch_branch.py "${@:3}" --backend starvla --checkpoint SERVER; }
log() { echo "$1 $2" | tee -a logs/batch_1027.txt; }
fresh() { [ -e $D/$1 ] && { echo "$D/$1 exists; skipped" >&2; return 1; }; return 0; }
mkdir -p $D
case ${1:-} in
  swap)
    [ -f $S/states.jsonl ] || { echo "no $S: batch_1026 collect first" >&2; exit 1; }
    for spec in "none-A n0=s8,n3=s8,n4=s1 none" "full-A n0=s8,n3=s8,n4=s1 full:1" \
                "full-B n0=s1,n3=s1,n4=s8 full:1" "full-C n0=s4,n3=s4,n4=s4 full:1"; do
      set -- $spec
      fresh $1 || continue
      export LV_PATCH=$2             # not in --export: Slurm splits its list on commas
      extra=; [ $1 = none-A ] && extra=,LV_RECORD=$D/none-A.rec
      log "$(run br2-$1 LV_PATCH_U=$U:$3$extra branch --states $S --out $D/$1)" $1
    done
    unset LV_PATCH
    fresh instr-A && log "$(run br2-instr-A LV_PATCH_U= branch --states $S --out $D/instr-A \
      --lang 'n0=s8;n3=s8;n4=s1')" instr-A ;;
  collect)
    [ -e $B ] && { echo "$B exists" >&2; exit 1; }
    unset LV_PATCH
    log "$(run br2-collect LV_PATCH_U= collect --goals s4,s8,n5 --n_states 10 --out $B)" collect ;;
  default)
    [ -f $B/states.jsonl ] || { echo "no $B: run collect first" >&2; exit 1; }
    unset LV_PATCH
    for spec in "own|" "empty|s4=;s8=;n5=" \
                "nonce|s4=put the bowl on the dax;s8=put the bowl on the dax;n5=put the bowl in front of the dax" \
                "nodest|s4=put the bowl;s8=put the bowl;n5=put the bowl"; do
      c=${spec%%|*}; l=${spec#*|}
      fresh bowl-$c || continue
      log "$(run br2-bowl-$c LV_PATCH_U= branch --states $B --out $D/bowl-$c --lang "$l")" bowl-$c
    done ;;
  attn)
    [ -f $B/states.jsonl ] || { echo "no $B: run collect first" >&2; exit 1; }
    unset LV_PATCH
    fresh attn-wine && log "$(run br2-attn-wine LV_PATCH_U=,LV_ATTN=$D/attn-wine.dumps probe --states $S \
      --goals n3,n4 --n_per_goal 5 --out $D/attn-wine \
      --instr 'own;s8;s1;s2;s9;;put the wine bottle on the dax')" attn-wine
    fresh attn-bowl && log "$(run br2-attn-bowl LV_PATCH_U=,LV_ATTN=$D/attn-bowl.dumps probe --states $B \
      --goals s4,s8,n5 --n_per_goal 5 --out $D/attn-bowl \
      --instr 'own;s1;s4;s8;;put the bowl on the dax;put the bowl in front of the dax')" attn-bowl ;;
  report)
    python scripts/patch_branch.py report $(ls -d $D/*/ | grep -v -e attn -e dumps)
    [ -f $D/none-A.rec ] && python scripts/patch_branch.py norms --rec $D/none-A.rec --subspaces $U
    for a in attn-wine attn-bowl; do
      [ -f $D/$a/probe.jsonl ] || continue
      python scripts/patch_branch.py attn $D/$a --dumps $D/$a.dumps --png $D/$a.png
      python scripts/patch_branch.py attn $D/$a --dumps $D/$a.dumps --layers 0:12
    done ;;
  *) echo "usage: $0 swap | collect | default | attn | report" >&2; exit 1 ;;
esac
