"""LIBERO goal predicates: which of a suite's goals hold in the current simulator state.

In LIBERO-Goal all 10 bddl files share regions, fixtures, objects and init (only `obj_of_interest` differs),
so every goal can be evaluated in any task's env. `load_goals` asserts this before returning a shared goal set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SHARED_SECTIONS = ("regions", "fixtures", "objects", "init")
MOVE_PREDICATES = ("on", "in")


def bddl_section(text: str, name: str) -> str:
    """The balanced s-expression starting at `(:name`, whitespace-normalized."""
    i = text.index(f"(:{name}")
    depth = 0
    for j in range(i, len(text)):
        depth += text[j] == "("
        depth -= text[j] == ")"
        if depth == 0:
            return re.sub(r"\s+", " ", text[i: j + 1])
    raise ValueError(f"unbalanced section {name}")


def check_shared_scene(texts: dict[str, str]) -> None:
    """Raise if the bddl files differ in anything that defines the scene."""
    for name in SHARED_SECTIONS:
        variants = {bddl_section(t, name) for t in texts.values()}
        if len(variants) != 1:
            raise ValueError(f"bddl files differ in (:{name} ...): {sorted(texts)}")


def _names(group) -> set[str]:
    if isinstance(group, dict):                         # {category: [names]}
        return {n for names in group.values() for n in names}
    return set(group)


@dataclass
class Goal:
    language: str
    predicates: list = field(repr=False)   # parsed_problem["goal_state"], e.g. [["on", "akita_black_bowl_1", "plate_1"]]
    target: str | None                      # movable object the goal moves (first arg of On/In), else None


def goal_target(predicates: list, movable: set[str]) -> str | None:
    for p in predicates:
        if p[0].lower() in MOVE_PREDICATES and p[1] in movable:
            return p[1]
    return None


def load_goals(suite_name: str, task_ids: list[int] | None = None) -> list[Goal]:
    """Goals in benchmark task order. For libero_goal, also checks that the scene is shared."""
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.bddl_utils import robosuite_parse_problem
    import os

    suite = benchmark.get_benchmark_dict()[suite_name]()
    ids = range(suite.n_tasks) if task_ids is None else task_ids
    paths = {}
    for i in ids:
        t = suite.get_task(i)
        paths[i] = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    if suite_name == "libero_goal":
        check_shared_scene({p: open(p).read() for p in paths.values()})
    goals = []
    for i in ids:
        parsed = robosuite_parse_problem(paths[i])
        goals.append(Goal(suite.get_task(i).language, parsed["goal_state"],
                          goal_target(parsed["goal_state"], _names(parsed["objects"]))))
    return goals


def satisfied(env, goals: list[Goal]) -> list[bool]:
    """Whether each goal's predicates all hold now. `env` is LIBERO's OffScreenRenderEnv."""
    inner = env.env
    return [all(inner._eval_predicate(p) for p in g.predicates) for g in goals]
