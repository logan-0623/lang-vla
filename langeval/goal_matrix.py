"""Which goal did a rollout execute? (IMPLEMENTATION_PLAN §4.4)

The env's own `done` only covers the instructed task, so we track every goal: a goal counts as executed once
its predicates hold for `hold_steps` consecutive steps. Transient states (e.g. the bowl passing over the plate
on its way to the stove) therefore do not count.
"""
from __future__ import annotations


class GoalTracker:
    def __init__(self, n_goals: int, hold_steps: int = 10, prefer: int | None = None):
        self.hold_steps = hold_steps
        self.prefer = prefer                    # tie-break when several goals complete their hold on the same step
        self.run = [0] * n_goals
        self.first_satisfied: dict[int, int] = {}
        self.executed: int | None = None
        self.executed_step: int | None = None
        self.completed: dict[int, int] = {}     # every goal that ever held for hold_steps -> step it did
        self.last: list[bool] = [False] * n_goals

    def update(self, sat: list[bool], t: int) -> bool:
        """Feed one step's satisfaction vector; returns True once a goal has been executed."""
        self.last = list(sat)
        done = []
        for k, s in enumerate(sat):
            self.run[k] = self.run[k] + 1 if s else 0
            if s and k not in self.first_satisfied:
                self.first_satisfied[k] = t
            if self.run[k] >= self.hold_steps:
                done.append(k)
                self.completed.setdefault(k, t)
        if self.executed is None and done:
            self.executed = self.prefer if self.prefer in done else done[0]
            self.executed_step = t
        return self.executed is not None
