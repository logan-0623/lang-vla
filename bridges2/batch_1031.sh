#!/bin/bash
# RESEARCH_PLAN §11.8 (2026-10-24).
#   bash bridges2/batch_1031.sh paired     login node: paired (goal, init) statistics of the StarVLA branch runs
#   bash bridges2/batch_1031.sh collect    pi0.5: stable-grasp states for n0, n3, n4 (unpatched)
#   bash bridges2/batch_1031.sh branch [collect job id]   pi0.5: 5 jobs from those states (T1', W1'); with a job id
#                                          they wait for it (sbatch --dependency=afterok)
#   bash bridges2/batch_1031.sh all        collect + branch in one go
#   bash bridges2/batch_1031.sh report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
PI=$CKPT/pi05_libero_finetuned_v044
P=$RESULTS/pi05/branch
E=$RESULTS/erase
AFTER=                                  # sbatch dependency, set by `all` / `branch <collect job id>`
pi() { sbatch --parsable -t 8:00:00 -J $1 $AFTER --export=ALL,ENV=lerobot bridges2/run.sbatch \
  python scripts/patch_branch.py "${@:2}" --backend lerobot --checkpoint $PI; }
log() { echo "$1 $2" | tee -a logs/batch_1031.txt; }
case ${1:-} in
  paired)
    python scripts/patch_branch.py paired $E/branch2/none-A \
      $E/branch/dest-4-1 $E/branch/random-64-1 $E/branch2/full-A $E/branch2/instr-A \
      $E/branch3/full-paraA $E/branch3/full-paraonly $E/branch4/obj-it $E/branch4/obj-generic $E/branch4/obj-bottle ;;
  collect)
    [ -e $P/states ] && { echo "$P/states exists" >&2; exit 1; }
    mkdir -p $P
    log "$(pi pi-collect collect --goals n0,n3,n4 --n_states 20 --max_attempts 50 --out $P/states)" collect ;;
  all)                                  # collect, then the 5 branch jobs once it has finished OK
    [ -e $P/states ] && { echo "$P/states exists: use branch" >&2; exit 1; }
    mkdir -p $P
    cid=$(pi pi-collect collect --goals n0,n3,n4 --n_states 20 --max_attempts 50 --out $P/states)
    log $cid collect
    exec bash "$0" branch $cid ;;
  branch)
    if [ -n "${2:-}" ]; then
      AFTER="--dependency=afterok:$2"     # waits for that collect job; cancelled if it fails
    else
      [ -f $P/states/states.jsonl ] || { echo "no $P/states: run collect first, or branch <collect job id>" >&2; exit 1; }
    fi
    for spec in "own|" "seenA|n0=s8;n3=s8;n4=s1" "seenB|n0=s1;n3=s1;n4=s8" "seenC|n0=s4;n3=s4;n4=s4" \
                "it|n0=put it on the plate;n3=put it on the plate;n4=put it on the stove"; do
      c=${spec%%|*}; l=${spec#*|}
      [ -e $P/$c ] && { echo "$P/$c exists; skipped" >&2; continue; }
      log "$(pi pi-$c branch --states $P/states --out $P/$c ${l:+--lang "$l"})" $c
    done ;;
  report)
    python scripts/patch_branch.py report $(ls -d $P/*/ | grep -v states)
    [ -f $P/own/episodes.jsonl ] && python scripts/patch_branch.py paired $P/own $(ls -d $P/*/ | grep -v -e states -e own) ;;
  *) echo "usage: $0 paired | collect | branch [job id] | all | report" >&2; exit 1 ;;
esac
