#!/bin/bash
# Batch 2026-10-20: counter-heuristic pairs for the gripper-target relation (RESEARCH_PLAN §11.6; ~2 SU).
#   cf-collect : scripted arm moves from 80 settled parent scenes (scripts/make_counter_pairs.py collect)
#   cf-feats   : hidden states of the raw VLM and 4in1 on those frames; raw-VLM QA on them (no format hint: the
#                hint made it invent held objects, findings 2026-10-20)
#   cf-analyze : probes trained on goal-v1, applied to the pairs; QA by consistent / counter group
#   git pull ; bash bridges2/batch_1020.sh       (afterwards: tail -20 logs/cf-analyze-*.out)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
BASE=$CKPT/Qwen3-VL-4B-Instruct
CF=$STORE/qa/counter-v1
P=$RESULTS/probe
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 >> logs/batch_1020.txt; echo "$id"; }
[ -e "$CF" ] && { echo "$CF exists; move it away first" >&2; exit 1; }

c=$(sub -t 1:00:00 -J cf-collect --export=ALL,ENV=vla bridges2/run.sbatch \
  python scripts/make_counter_pairs.py collect --n_parents 80 --out $CF)
f=$(sub -t 1:30:00 -J cf-feats --dependency=afterok:$c --export=ALL,ENV=starvla bridges2/run.sbatch bash -c "
  python scripts/probe_vlm.py extract --base $BASE --qa_dir $CF --out $P/counter-raw.npz;
  python scripts/probe_vlm.py extract --base $BASE --ckpt $SV --qa_dir $CF --out $P/counter-4in1.npz;
  python scripts/eval_qa.py --base $BASE --qa_dir $CF --max_images 100000 --out $RESULTS/qa/counter-raw")
a=$(sub -t 0:30:00 -J cf-analyze --dependency=afterok:$f --export=ALL,ENV=starvla bridges2/run.sbatch \
  python scripts/make_counter_pairs.py analyze --train_dir $STORE/qa/goal-v1 --test_dir $CF \
  --feats raw=$P/raw.npz,$P/counter-raw.npz 4in1=$P/4in1.npz,$P/counter-4in1.npz \
  --qa $RESULTS/qa/counter-raw --pixels --out $P/counter-analysis.json)
echo "submitted $c -> $f -> $a; ids in logs/batch_1020.txt"
