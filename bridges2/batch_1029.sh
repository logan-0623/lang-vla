#!/bin/bash
# RESEARCH_PLAN §11.7 lookup-table tests (2026-10-23).
#   bash bridges2/batch_1029.sh collect   20 stable bowl grasps for s8 (bowl -> plate), unpatched
#   bash bridges2/batch_1029.sh shift     5 jobs from those states: plate moved 0 / +-10 cm in x / y after the grasp
#   bash bridges2/batch_1029.sh keys      1 job: key probes from the episode start (configs/key_probes.json) + s8, n0
#   bash bridges2/batch_1029.sh traj      1 job: batch_1027 full-C again (cheese / wine -> cabinet top) with eef paths
#   bash bridges2/batch_1029.sh objword   3 jobs on the batch_1026 states: the object word replaced (it / the object /
#                                         a shorter name), destination word kept
#   bash bridges2/batch_1029.sh nonce     2 jobs on the batch_1027 bowl states: "on the blicket" (all), "on the dax" (n5)
#   bash bridges2/batch_1029.sh attn      1 job: attention dumps with text -> image rows (n3, n4 states)
#   bash bridges2/batch_1029.sh report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
U=$RESULTS/erase/patch/subspaces2.npz
S=$RESULTS/erase/branch/states
P=$RESULTS/erase/branch/plate
D=$RESULTS/erase/branch4
run() { sbatch --parsable -t 3:00:00 -J $1 --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,$2 \
  bridges2/run_starvla.sbatch python "${@:3}" --backend starvla --checkpoint SERVER; }
log() { echo "$1 $2" | tee -a logs/batch_1029.txt; }
mkdir -p $D
unset LV_PATCH
case ${1:-} in
  collect)
    [ -e $P ] && { echo "$P exists" >&2; exit 1; }
    log "$(run br4-collect LV_PATCH_U= scripts/patch_branch.py collect --goals s8 --n_states 20 --out $P)" collect ;;
  shift)
    [ -f $P/states.jsonl ] || { echo "no $P: run collect first" >&2; exit 1; }
    for spec in "shift-0 0,0" "shift-xp 0.1,0" "shift-xm -0.1,0" "shift-yp 0,0.1" "shift-ym 0,-0.1"; do
      set -- $spec
      [ -e $D/$1 ] && { echo "$D/$1 exists; skipped" >&2; continue; }
      log "$(run br4-$1 LV_PATCH_U= scripts/patch_branch.py branch --states $P --out $D/$1 --shift plate_1:$2)" $1
    done ;;
  keys)
    [ -e $D/keys ] && { echo "$D/keys exists" >&2; exit 1; }
    log "$(run br4-keys LV_PATCH_U= scripts/eval_goals.py --sets seen,novel --novel_json configs/key_probes.json \
      --goals s8,n0,k0,k1,k2,k3,k4,k5,k6,k7,k8 --n_init 20 --out $D/keys)" keys ;;
  traj)
    [ -e $D/full-C ] && { echo "$D/full-C exists" >&2; exit 1; }
    export LV_PATCH=n0=s4,n3=s4,n4=s4      # not in --export: Slurm splits its list on commas
    log "$(run br4-full-C LV_PATCH_U=$U:full:1 scripts/patch_branch.py branch --states $S --out $D/full-C)" full-C ;;
  objword)
    [ -f $S/states.jsonl ] || { echo "no $S: batch_1026 collect first" >&2; exit 1; }
    for spec in "it|n0=put it on the plate;n3=put it on the plate;n4=put it on the stove" \
                "generic|n0=put the object on the plate;n3=put the object on the plate;n4=put the object on the stove" \
                "bottle|n0=put the cheese on the plate;n3=put the bottle on the plate;n4=put the bottle on the stove"; do
      c=${spec%%|*}; l=${spec#*|}
      [ -e $D/obj-$c ] && { echo "$D/obj-$c exists; skipped" >&2; continue; }
      log "$(run br4-obj-$c LV_PATCH_U= scripts/patch_branch.py branch --states $S --out $D/obj-$c --lang "$l")" obj-$c
    done ;;
  nonce)
    B=$RESULTS/erase/branch/bowl
    [ -f $B/states.jsonl ] || { echo "no $B: batch_1027 collect first" >&2; exit 1; }
    for spec in "bowl-blicket||put the bowl on the blicket" "bowl-dax-on|n5|put the bowl on the dax"; do
      IFS='|' read -r c g l <<< "$spec"
      [ -e $D/$c ] && { echo "$D/$c exists; skipped" >&2; continue; }
      lang="s4=$l;s8=$l;n5=$l"
      log "$(run br4-$c LV_PATCH_U= scripts/patch_branch.py branch --states $B --out $D/$c --lang "$lang" \
        ${g:+--goals $g})" $c
    done ;;
  attn)
    [ -e $D/attn-wine2 ] && { echo "$D/attn-wine2 exists" >&2; exit 1; }
    log "$(run br4-attn-wine2 LV_PATCH_U=,LV_ATTN=$D/attn-wine2.dumps scripts/patch_branch.py probe --states $S \
      --goals n3,n4 --n_per_goal 5 --out $D/attn-wine2 --instr 'own;s8;s1;put it on the plate')" attn-wine2 ;;
  report)
    python scripts/patch_branch.py report $(ls -d $D/*/ | grep -v -e keys -e attn)
    [ -f $D/keys/episodes.jsonl ] && python scripts/patch_branch.py moved $D/keys
    if [ -f $D/attn-wine2/probe.jsonl ]; then
      python scripts/patch_branch.py attn $D/attn-wine2 --dumps $D/attn-wine2.dumps --png $D/fig-wine2-wine --word wine
      python scripts/patch_branch.py attn $D/attn-wine2 --dumps $D/attn-wine2.dumps --png $D/fig-wine2-plate --word plate
    fi ;;
  *) echo "usage: $0 collect | shift | keys | traj | objword | nonce | attn | report" >&2; exit 1 ;;
esac
