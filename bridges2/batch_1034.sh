#!/bin/bash
# RESEARCH_PLAN §11.9 follow-ups to the A2 pilot (2026-10-24), written before any batch_1033 result. Full episodes,
# init states 20-39 unless noted. Modes can be submitted together: `all`.
#   deploy  2 jobs : it-after with the deployable trigger (--trigger action), StarVLA + pi0.5, n0 / n3 / n4
#   goals   8 jobs : the four A2 conditions on n1 / n2, StarVLA + pi0.5
#   retain  4 jobs : seen pick-and-place tasks s1 s2 s4 s6 s8 s9, orig vs it-after, StarVLA + pi0.5
#   oft     6 jobs : a third model with the rule frozen: OpenVLA-OFT (4-suite) grasp-state branches own / it / seenA
#                    (collect, then branches chained with afterok) and full-episode orig / it-after
#   bash bridges2/batch_1034.sh all | deploy | goals | retain | oft | report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
PI=$CKPT/pi05_libero_finetuned_v044
OFT4=$CKPT/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
J=configs/stage_goals.json
D=$RESULTS/stage_a2
common=(--init_offset 20 --n_init 20)
sv() { sbatch --parsable -t 6:00:00 -J $1 --export=ALL,SV_CKPT=$SV,SV_WRAP= bridges2/run_starvla.sbatch \
  python scripts/eval_goals.py --backend starvla --checkpoint SERVER "${common[@]}" "${@:2}"; }
pi() { sbatch --parsable -t 12:00:00 -J $1 --export=ALL,ENV=lerobot bridges2/run.sbatch \
  python scripts/eval_goals.py --backend lerobot --checkpoint $PI "${common[@]}" "${@:2}"; }
oft() { sbatch --parsable -t 12:00:00 -J $1 $AFTER --export=ALL,ENV=vla,REPO=oft bridges2/run.sbatch \
  python "${@:2}"; }
AFTER=
log() { echo "$1 $2" | tee -a logs/batch_1034.txt; }
sub() { local name=$1 dir=$2; shift 2; [ -e $dir ] && { echo "$dir exists; skipped" >&2; return 0; }; log "$("$@")" $name; }
IT3="n0=put it on the plate;n3=put it on the plate;n4=put it on the stove"
IT12="n1=put it on the stove;n2=put it on top of the cabinet"
SH12="n1=put the cheese on the stove;n2=put the cheese on top of the cabinet"
ITS="s1=put it on the stove;s2=put it on top of the cabinet;s4=put it on top of the cabinet;s6=put it in the bowl;s8=put it on the plate;s9=put it on the rack"
unset LV_PATCH
mkdir -p $D
deploy() {
  for m in sv pi; do
    sub $m-it-after-action $D/$m-it-after-action $m a3-$m-itact --sets novel --novel_json $J --goals n0,n3,n4 \
      --switch "$IT3" --trigger action --out $D/$m-it-after-action
  done; }
goals() {
  for m in sv pi; do
    sub $m-n12-orig $D/$m-n12-orig $m a3-$m-n12o --sets novel --novel_json $J --goals n1,n2 --out $D/$m-n12-orig
    sub $m-n12-it-start $D/$m-n12-it-start $m a3-$m-n12s --sets novel --novel_json $J --goals a1,a2 --out $D/$m-n12-it-start
    sub $m-n12-short-after $D/$m-n12-short-after $m a3-$m-n12h --sets novel --novel_json $J --goals n1,n2 \
      --switch "$SH12" --out $D/$m-n12-short-after
    sub $m-n12-it-after $D/$m-n12-it-after $m a3-$m-n12i --sets novel --novel_json $J --goals n1,n2 \
      --switch "$IT12" --out $D/$m-n12-it-after
  done; }
retain() {
  for m in sv pi; do
    sub $m-seen-orig $D/$m-seen-orig $m a3-$m-seeno --sets seen --goals s1,s2,s4,s6,s8,s9 --out $D/$m-seen-orig
    sub $m-seen-it-after $D/$m-seen-it-after $m a3-$m-seeni --sets seen --goals s1,s2,s4,s6,s8,s9 \
      --switch "$ITS" --out $D/$m-seen-it-after
  done; }
oftrun() {
  local B=$RESULTS/oft4/branch
  mkdir -p $B
  local cid=
  if [ ! -e $B/states ]; then
    cid=$(oft oft-collect scripts/patch_branch.py collect --backend upstream --checkpoint $OFT4 --goals n0,n3,n4 \
      --n_states 20 --max_attempts 50 --out $B/states)
    log $cid oft-collect
  fi
  [ -n "$cid" ] && AFTER="--dependency=afterok:$cid"
  for spec in "own|" "it|$IT3" "seenA|n0=s8;n3=s8;n4=s1"; do
    c=${spec%%|*}; l=${spec#*|}
    sub oft-$c $B/$c oft oft-$c scripts/patch_branch.py branch --backend upstream --checkpoint $OFT4 \
      --states $B/states --out $B/$c ${l:+--lang "$l"}
  done
  AFTER=
  sub oft-orig $D/oft-orig oft oft-orig scripts/eval_goals.py --checkpoint $OFT4 "${common[@]}" --sets novel \
    --novel_json $J --goals n0,n3,n4 --out $D/oft-orig
  sub oft-it-after $D/oft-it-after oft oft-itafter scripts/eval_goals.py --checkpoint $OFT4 "${common[@]}" --sets novel \
    --novel_json $J --goals n0,n3,n4 --switch "$IT3" --out $D/oft-it-after; }
case ${1:-} in
  deploy) deploy ;;
  goals) goals ;;
  retain) retain ;;
  oft) oftrun ;;
  all) deploy; goals; retain; oftrun ;;
  report)
    bash bridges2/batch_1033.sh report
    B=$RESULTS/oft4/branch
    [ -d $B ] && python scripts/patch_branch.py report $(ls -d $B/*/ | grep -v states)
    [ -f $B/own/episodes.jsonl ] && python scripts/patch_branch.py paired $B/own $(ls -d $B/*/ | grep -v -e states -e own)
    python - <<'PY'
import json, glob, os
for d in sorted(glob.glob(os.path.expandvars("$RESULTS/stage_a2/*-action"))):
    eps = [json.loads(l) for l in open(f"{d}/episodes.jsonl")]
    sw = [e for e in eps if e.get("switch_step") is not None]
    print(f"{os.path.basename(d)}: switched {len(sw)}/{len(eps)}, target actually held at switch {sum(bool(e['target_held_at_switch']) for e in sw)}")
PY
    ;;
  *) echo "usage: $0 all | deploy | goals | retain | oft | report" >&2; exit 1 ;;
esac
