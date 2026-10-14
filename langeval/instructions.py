"""Instruction transforms (IMPLEMENTATION_PLAN §4.2) and the meaning-preserving perturbations used as controls
in E3: each perturbation touches only a function word, so it changes the token sequence without changing the
task. LIBERO-Para supplies the paraphrases.
"""
from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path


def transform(mode: str, instruction: str, *, task_id: int, all_instructions: list[str],
              shift: int = 1, seed: int = 0) -> str:
    if mode == "orig":
        return instruction
    if mode == "empty":
        return ""
    if mode == "other":
        return all_instructions[(task_id + shift) % len(all_instructions)]
    if mode == "shuffle":
        words = instruction.split()
        random.Random(seed).shuffle(words)
        return " ".join(words)
    if mode in ("cipher", "key"):
        raise NotImplementedError(f"{mode} is stage 5")
    raise ValueError(f"unknown instruction mode: {mode}")


def _first_article(words: list[str]) -> int:
    return next((i for i, w in enumerate(words) if w == "the"), 0)


def perturb(kind: str, instruction: str) -> str:
    """Meaning-preserving token-level perturbation of a function word."""
    w = instruction.split()
    i = _first_article(w)
    if kind == "dup_article":            # "put the the bowl ..."
        return " ".join(w[: i + 1] + [w[i]] + w[i + 1:])
    if kind == "filler":                 # "put um the bowl ..."
        return " ".join(w[:1] + ["um"] + w[1:])
    if kind == "typo_article":           # "put teh bowl ..."
        return " ".join(w[:i] + ["teh" if w[i] == "the" else w[i][::-1]] + w[i + 1:])
    raise ValueError(f"unknown perturbation: {kind}")


PERTURBATIONS = ("dup_article", "filler", "typo_article")


def load_paraphrases(csv_path: str | Path, per_category: int, seed: int) -> dict[str, list[dict]]:
    """LIBERO-Para metadata -> {original instruction: rows}, `per_category` rows per (high, mid) category.
    Rows keep the CSV columns (high, mid, low, new_instruction, ...). Match by text: LIBERO-Para's `eval` index
    follows bddl filename order, not the benchmark's task order."""
    groups = defaultdict(list)
    for r in csv.DictReader(open(csv_path)):
        groups[(r["original_instruction"], r["high"], r["mid"])].append(r)
    rng = random.Random(seed)
    out = defaultdict(list)
    for (orig, _, _), rows in sorted(groups.items()):
        out[orig] += rng.sample(rows, min(per_category, len(rows)))
    return dict(out)
