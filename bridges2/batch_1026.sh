#!/bin/bash
# Stage 2 of RESEARCH_PLAN §11.7 (2026-10-22): branch from "target object stably grasped" states of n0, n3, n4 and
# compare no patch / destination patches / random, all with the same donor map and from the same states.
#   bash bridges2/batch_1026.sh collect                                  unpatched grasp states, ~0.5 GPU-h
#   bash bridges2/batch_1026.sh branch none dest:4:1 dest_c:4:1 random:4:1   one job per condition (name:rank:strength
#                                                                        of subspaces2.npz, from batch_1025)
#   bash bridges2/batch_1026.sh report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
U=$RESULTS/erase/patch/subspaces2.npz
D=$RESULTS/erase/branch
DONORS="n0=s8,n3=s8,n4=s1"
run() { sbatch --parsable -t 3:00:00 -J $1 --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,$2 \
  bridges2/run_starvla.sbatch python scripts/patch_branch.py "${@:3}" --backend starvla --checkpoint SERVER; }
case ${1:-} in
  collect)
    [ -e $D/states ] && { echo "$D/states exists" >&2; exit 1; }
    mkdir -p $D
    unset LV_PATCH
    echo "$(run br-collect LV_PATCH_U= collect --goals n0,n3,n4 --n_states 20 --out $D/states) collect" | tee -a logs/batch_1026.txt ;;
  branch)
    [ -f $D/states/states.jsonl ] || { echo "no grasp states yet: run collect first" >&2; exit 1; }
    [ -f $U ] || { echo "no $U: run batch_1025 first" >&2; exit 1; }
    for c in "${@:2}"; do
      out=$D/${c//:/-}
      [ -e $out ] && { echo "$out exists; skipped" >&2; continue; }
      export LV_PATCH=$DONORS        # not in --export: Slurm splits its list on commas
      echo "$(run br-${c//:/-} LV_PATCH_U=$U:$c branch --states $D/states --out $out) $c" | tee -a logs/batch_1026.txt
    done ;;
  report)
    python scripts/patch_branch.py report $(ls -d $D/*/ | grep -v states) ;;
  *) echo "usage: $0 collect | branch <cond>... | report" >&2; exit 1 ;;
esac
