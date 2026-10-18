# lang-vla

Does language-expressed spatial and physical knowledge transfer into robot action? And, as a baseline: when do
current vision-language-action models actually use their instructions? Evaluation, interpretability and training on LIBERO.

- `docs/RESEARCH_PLAN.md`: **start here**. Questions, hypotheses with pre-registered rules, every experiment
  (design, implementation, status), budget. §11 is the current main question; §0–§10 are phase 1
- `docs/findings.md`: results, with their sources and verdicts
- `docs/DECISIONS.md`: engineering decisions and traps found in upstream code
- `docs/RESEARCH_DESIGN.md`: phase-1 design record (literature gaps, E0–E8); `IMPLEMENTATION_PLAN.md`: original training/probe build plan
- `langeval/`: model-agnostic evaluation code: goal predicates (incl. novel goals), executed-goal tracking, grasp
  phases, exact snapshot/restore branching; `adapters/` for VLA-Adapter / OpenVLA-OFT (`upstream`), pi0.5 (`lerobot`)
  and StarVLA Qwen3-VL (`starvla`)
- `scripts/`: experiment entry points

  | script | what |
  |---|---|
  | `e3_grasp_branch.py`, `e4_switch_time.py` | branch from grasp states / switch the instruction mid-rollout |
  | `paraphrase_invariance.py`, `para_vs_e6.py` | E6 layer-wise invariance, R_action vs LIBERO-Para failures |
  | `judge_h1h3.py` | apply the pre-registered H1′/H1″/H3′ rules |
  | `eval_goals.py` | success on the 10 Goal tasks and the novel combinations, with the executed goal |
  | `make_spatial_qa.py` | spatial QA from simulator ground truth (lang / num / ctrl styles, LLaVA json) |
  | `report.py`, `per_task.py`, `diag_restore.py` | reports and diagnostics |
  | `setup_pion.sh`, `setup_starvla.sh`, `download_starvla_assets.sh` | code and envs |
- `configs/`: paraphrase / nonce-word sets, novel goals, StarVLA training config
- `bridges2/`: PSC Bridges-2 (x86, L40S): evaluation. `deltaai/`: NCSA DeltaAI (aarch64 GH200): training
- `patches/`: our changes to Pion's VLA-Adapter copy and to LIBERO-Para

## Setup

```bash
git clone https://github.com/logan-0623/lang-vla.git && cd lang-vla
bash scripts/setup_pion.sh          # Pion/ = OPTML-Group/Pion @ 8796501 + patches/ on branch lang-spectral
```
Then follow `bridges2/README.md` (evaluation) or `deltaai/README.md` (training).

Local unit tests (no GPU, no LIBERO):
```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]" && .venv/bin/python -m pytest -q tests/
```
