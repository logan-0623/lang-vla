#!/bin/bash
# Batch 2026-10-19: single-GPU diagnostics while 4-GPU training is blocked (RESEARCH_PLAN §11.5, predictions
# written before these results). ~7 SU in total.
#   qa-raw   : raw Qwen3-VL QA with an output-format hint, at 256 px and upscaled 2x; recognition (ctrl style) at both
#              -> is the raw score low because of format, resolution, or knowledge?
#   qa-4in1  : the same for the VLM inside the released StarVLA 4in1 VLA (same base, fine-tuned for actions)
#              -> what VLA fine-tuning does to the verbal QA
#   probe    : hidden states of both models for every QA frame (scripts/probe_vlm.py extract); the probes are fit on
#              the login node afterwards; also a 4x4 grid of the image tokens (D6) -> is the knowledge in the representation even where it is not said?
#   novel-oft-joint : OFT trained on the four suites, on the nine novel goals -> cross-suite data vs pretraining
#   git pull ; bash bridges2/batch_1019.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
BASE=$CKPT/Qwen3-VL-4B-Instruct
QA=$STORE/qa/goal-v1
Q="python scripts/eval_qa.py --base $BASE --qa_dir $QA"
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 | tee -a logs/batch_1019.txt; }
[ -f "$SV" ] || { echo "missing $SV" >&2; exit 1; }

sub -t 5:00:00 -J qa-raw --export=ALL,ENV=starvla bridges2/run.sbatch bash -c "
  $Q --style lang --suffix FORMAT_HINT --out $RESULTS/qa/raw-lang-hint;
  $Q --style lang --suffix FORMAT_HINT --upscale 2 --out $RESULTS/qa/raw-lang-hint-x2;
  $Q --style ctrl --out $RESULTS/qa/raw-ctrl;
  $Q --style ctrl --upscale 2 --out $RESULTS/qa/raw-ctrl-x2"
sub -t 4:00:00 -J qa-4in1 --export=ALL,ENV=starvla bridges2/run.sbatch bash -c "
  $Q --ckpt $SV --style lang --suffix FORMAT_HINT --out $RESULTS/qa/4in1-lang-hint;
  $Q --ckpt $SV --style ctrl --out $RESULTS/qa/4in1-ctrl"
sub -t 2:00:00 -J probe --export=ALL,ENV=starvla bridges2/run.sbatch bash -c "
  python scripts/probe_vlm.py extract --base $BASE --qa_dir $QA --out $RESULTS/probe/raw.npz;
  python scripts/probe_vlm.py extract --base $BASE --ckpt $SV --qa_dir $QA --out $RESULTS/probe/4in1.npz"
sub -t 8:00:00 -J novel-oft-joint --export=ALL,ENV=vla,REPO=oft bridges2/run.sbatch \
  python scripts/eval_goals.py --checkpoint $CKPT/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --n_init 20 --out $RESULTS/novel/oft-joint-$(date +%m%d%H%M)
echo "submitted; ids in logs/batch_1019.txt"
