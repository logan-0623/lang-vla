# lang-vla

Do vision-language-action models read their instructions, and when? Evaluation and interpretability on LIBERO.

- `docs/RESEARCH_DESIGN.md`: questions, gaps in the literature, hypotheses, experiments E0–E7
- `IMPLEMENTATION_PLAN.md`: the full build plan (training arms, metrics, probes)
- `docs/DECISIONS.md`: design decisions and traps found in upstream code
- `langeval/`: model-agnostic evaluation code (goal predicates, executed-goal tracking, grasp phases, branching rollouts)
- `scripts/`: experiment entry points (`e3_grasp_branch.py`, `paraphrase_invariance.py`, `setup_pion.sh`)
- `bridges2/`: PSC Bridges-2 (x86, L40S), zero-training experiments
- `deltaai/`: NCSA DeltaAI (aarch64 GH200), training
- `patches/`: our changes to Pion's VLA-Adapter copy

## Setup

```bash
git clone https://github.com/logan-0623/lang-vla.git && cd lang-vla
bash scripts/setup_pion.sh          # Pion/ = OPTML-Group/Pion @ 8796501 + patches/ on branch lang-spectral
```
Then follow `bridges2/README.md` (evaluation) or `deltaai/` (training).

Local unit tests (no GPU, no LIBERO):
```bash
python -m venv .venv && .venv/bin/pip install -e ".[test]" && .venv/bin/python -m pytest -q tests/
```
