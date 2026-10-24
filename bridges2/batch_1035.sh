#!/bin/bash
# RESEARCH_PLAN §11.9 (2026-10-24): complete the post-grasp 2 x 2 {object name, it} x {original destination,
# another destination} plus two controls (short name, "the object"), on both models' existing grasp states, and
# "it" under a moved destination (layout). Nothing here needs new grasp states.
#   name-alt  : object name kept, destination changed (the missing cell), StarVLA + pi0.5
#   short-alt : short name ("the cheese" / "the bottle"), destination changed, StarVLA + pi0.5
#   pi-d      : pi0.5 D group (it / the object x other destinations), as batch_1032 for StarVLA
#   layout    : StarVLA n0 / n3 states, plate moved +-10 cm in x: original instruction vs it
#   bash bridges2/batch_1035.sh all | name-alt | short-alt | pi-d | layout | report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
PI=$CKPT/pi05_libero_finetuned_v044
SS=$RESULTS/erase/branch/states
PS=$RESULTS/pi05/branch/states
D=$RESULTS/erase/branch6
P=$RESULTS/pi05/branch2
sv() { sbatch --parsable -t 4:00:00 -J $1 --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,LV_PATCH_U= \
  bridges2/run_starvla.sbatch python scripts/patch_branch.py branch --backend starvla --checkpoint SERVER "${@:2}"; }
pi() { sbatch --parsable -t 10:00:00 -J $1 --export=ALL,ENV=lerobot bridges2/run.sbatch \
  python scripts/patch_branch.py branch --backend lerobot --checkpoint $PI "${@:2}"; }
log() { echo "$1 $2" | tee -a logs/batch_1035.txt; }
sub() { local name=$1 dir=$2; shift 2; [ -e $dir ] && { echo "$dir exists; skipped" >&2; return 0; }; log "$("$@")" $name; }
NAME1="n0=put the cream cheese on the stove;n3=put the wine bottle on the stove;n4=put the wine bottle on the plate"
NAME2="n0=put the cream cheese on top of the cabinet;n3=put the wine bottle on top of the cabinet;n4=put the wine bottle on top of the cabinet"
SHORT1="n0=put the cheese on the stove;n3=put the bottle on the stove;n4=put the bottle on the plate"
SHORT2="n0=put the cheese on top of the cabinet;n3=put the bottle on top of the cabinet;n4=put the bottle on top of the cabinet"
IT1="n0=put it on the stove;n3=put it on the stove;n4=put it on the plate"
IT2="n0=put it on top of the cabinet;n3=put it on top of the cabinet;n4=put it on top of the cabinet"
OBJ1="n0=put the object on the stove;n3=put the object on the stove;n4=put the object on the plate"
OBJ2="n0=put the object on top of the cabinet;n3=put the object on top of the cabinet;n4=put the object on top of the cabinet"
ITA="n0=put it on the plate;n3=put it on the plate"
unset LV_PATCH
mkdir -p $D $P
namealt() {
  sub sv-name-alt1 $D/name-alt1 sv b6-name1 --states $SS --out $D/name-alt1 --lang "$NAME1"
  sub sv-name-alt2 $D/name-alt2 sv b6-name2 --states $SS --out $D/name-alt2 --lang "$NAME2"
  sub pi-name-alt1 $P/name-alt1 pi p2-name1 --states $PS --out $P/name-alt1 --lang "$NAME1"
  sub pi-name-alt2 $P/name-alt2 pi p2-name2 --states $PS --out $P/name-alt2 --lang "$NAME2"; }
shortalt() {
  sub sv-short-alt1 $D/short-alt1 sv b6-short1 --states $SS --out $D/short-alt1 --lang "$SHORT1"
  sub sv-short-alt2 $D/short-alt2 sv b6-short2 --states $SS --out $D/short-alt2 --lang "$SHORT2"
  sub pi-short-alt1 $P/short-alt1 pi p2-short1 --states $PS --out $P/short-alt1 --lang "$SHORT1"
  sub pi-short-alt2 $P/short-alt2 pi p2-short2 --states $PS --out $P/short-alt2 --lang "$SHORT2"; }
pid() {
  sub pi-it-alt1 $P/it-alt1 pi p2-it1 --states $PS --out $P/it-alt1 --lang "$IT1"
  sub pi-it-alt2 $P/it-alt2 pi p2-it2 --states $PS --out $P/it-alt2 --lang "$IT2"
  sub pi-obj-alt1 $P/obj-alt1 pi p2-obj1 --states $PS --out $P/obj-alt1 --lang "$OBJ1"
  sub pi-obj-alt2 $P/obj-alt2 pi p2-obj2 --states $PS --out $P/obj-alt2 --lang "$OBJ2"; }
layout() {
  for sh in "xp|0.1,0" "xm|-0.1,0"; do
    s=${sh%%|*}; v=${sh#*|}
    sub sv-shift-$s-own $D/shift-$s-own sv b6-$s-own --states $SS --goals n0,n3 --shift plate_1:$v --out $D/shift-$s-own
    sub sv-shift-$s-it $D/shift-$s-it sv b6-$s-it --states $SS --goals n0,n3 --shift plate_1:$v --out $D/shift-$s-it --lang "$ITA"
  done; }
case ${1:-} in
  name-alt) namealt ;;
  short-alt) shortalt ;;
  pi-d) pid ;;
  layout) layout ;;
  all) namealt; shortalt; pid; layout ;;
  report)
    python scripts/patch_branch.py report $(ls -d $D/*/) $RESULTS/erase/branch2/none-A $RESULTS/erase/branch5/*/
    python scripts/patch_branch.py report $(ls -d $P/*/) $RESULTS/pi05/branch/own
    for s in xp xm; do
      [ -f $D/shift-$s-it/episodes.jsonl ] && python scripts/patch_branch.py paired $D/shift-$s-own $D/shift-$s-it
    done ;;
  *) echo "usage: $0 all | name-alt | short-alt | pi-d | layout | report" >&2; exit 1 ;;
esac
