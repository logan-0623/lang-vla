# DeltaAI (aarch64 GH200, 4 x H100 per node): training

> 2026-10-18: no DeltaAI allocation; all training currently runs on Bridges-2 (`bridges2/train_starvla.sbatch`, see
> `bridges2/README.md`). These scripts are kept for a future allocation.

Set `ACCOUNT` in `deltaai/config.sh` (`accounts` on a login node). Submit from the repo root.

## StarVLA arms (RESEARCH_PLAN §11; current priority)

```bash
srun -A $ACCOUNT -p ghx4-interactive --gpus=1 --cpus-per-task=16 --mem=64g -t 1:30:00 bash scripts/setup_starvla.sh deltaai
bash scripts/download_starvla_assets.sh deltaai        # Qwen3-VL-4B-Instruct + LIBERO-Goal LeRobot data
# copy the QA data generated on Bridges-2 to $STORE/qa/goal-v1
```
Smoke test first (measures s/step; sets the real step counts): `MAX_STEPS=50,SAVE=50` with each stage.

| arm | stage 1 (VLM SFT) | stage 2 (VLA, LIBERO-Goal) |
|---|---|---|
| A | — | `STAGE=vla` |
| B | `STAGE=vlm,QA=lang` | `STAGE=vla,TAG=lang,INIT=<B stage-1 .pt>` |
| C | `STAGE=vlm,QA=ctrl` | `STAGE=vla,TAG=ctrl,INIT=<C stage-1 .pt>` |
| D | `STAGE=vlm,QA=num` | `STAGE=vla,TAG=num,INIT=<D stage-1 .pt>` |
| A+co | — | `STAGE=vla,COTRAIN=lang` |
| B+co | as B | `STAGE=vla,TAG=lang,INIT=<B stage-1 .pt>,COTRAIN=lang` |

```bash
sbatch -A $ACCOUNT --export=ALL,STAGE=vlm,QA=lang deltaai/train_starvla.sbatch
sbatch -A $ACCOUNT --export=ALL,STAGE=vla,TAG=lang,INIT=$STORE/runs/starvla/vlm-lang-<job>/checkpoints/steps_2000_pytorch_model.pt deltaai/train_starvla.sbatch
```
Held-out QA accuracy at three points (raw VLM, after stage 1, after stage 2), required before reading T1-T4:
```bash
sbatch -A $ACCOUNT --export=ALL,STYLE=lang deltaai/eval_qa.sbatch
sbatch -A $ACCOUNT --export=ALL,STYLE=lang,CKPT=<stage-1 or stage-2 .pt> deltaai/eval_qa.sbatch
```
Stage 2 loads only the VLM of the stage-1 checkpoint (`reload_modules=qwen_vl_interface`). Defaults: stage 1 2000
steps, stage 2 20000 steps saved every 5000 (global batch 4 x 16 x 2 = 128). Evaluate checkpoints on Bridges-2
(`bridges2/README.md`, `run_starvla.sbatch`); copy `checkpoints/steps_N_pytorch_model.pt` together with the run's
`config.yaml` and `dataset_statistics.json`.

## Optimizer / LangGuard arms (RESEARCH_PLAN §6; backlog)

```bash
srun -A $ACCOUNT -p ghx4-interactive --gpus=1 --cpus-per-task=16 --mem=64g -t 1:00:00 bash deltaai/setup_env.sh
bash deltaai/download_assets.sh
sbatch -A $ACCOUNT deltaai/smoke.sbatch
sbatch -A $ACCOUNT --export=ALL,ARM=pion,SEED=7,DATA_NAME=libero_goal_no_noops deltaai/train.sbatch
```
