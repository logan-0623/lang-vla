import numpy as np
import pytest

from langeval.branch import rollout, same_actions
from langeval.goal_matrix import GoalTracker
from langeval.goal_predicates import bddl_section, check_shared_scene, goal_target
from langeval.instructions import load_paraphrases, perturb, transform
from langeval.phase import GraspDetector, classify_outcome, path_length

SCENE = """(define (problem P)
  (:language Put the bowl on the plate)
  (:regions (plate_region (:target main_table)))
  (:fixtures main_table - table)
  (:objects akita_black_bowl_1 - akita_black_bowl plate_1 - plate)
  (:obj_of_interest {ooi})
  (:init (On plate_1 main_table_plate_region))
  (:goal (And {goal}))
)"""


def test_shared_scene_ignores_goal_and_obj_of_interest():
    a = SCENE.format(ooi="plate_1", goal="(On akita_black_bowl_1 plate_1)")
    b = SCENE.format(ooi="akita_black_bowl_1", goal="(Open wooden_cabinet_1_middle_region)")
    check_shared_scene({"a": a, "b": b})
    with pytest.raises(ValueError, match="init"):
        check_shared_scene({"a": a, "c": a.replace("main_table_plate_region", "elsewhere")})
    assert bddl_section(a, "objects") == "(:objects akita_black_bowl_1 - akita_black_bowl plate_1 - plate)"


def test_goal_target_is_moved_movable_object():
    movable = {"akita_black_bowl_1", "plate_1"}
    assert goal_target([["on", "akita_black_bowl_1", "plate_1"]], movable) == "akita_black_bowl_1"
    assert goal_target([["open", "wooden_cabinet_1_middle_region"]], movable) is None
    assert goal_target([["turnon", "flat_stove_1"]], movable) is None


def test_goal_tracker_needs_consecutive_hold():
    g = GoalTracker(3, hold_steps=3)
    for t, sat in enumerate([[0, 1, 0], [0, 1, 0], [0, 0, 0], [1, 1, 0], [1, 1, 0], [1, 1, 0]]):
        g.update([bool(x) for x in sat], t)
    assert g.first_satisfied == {1: 0, 0: 3}
    assert g.executed == 0 and g.executed_step == 5      # goal 1's first run was interrupted; tie -> lower index


def test_goal_tracker_prefers_instructed_goal_on_tie():
    g = GoalTracker(2, hold_steps=1, prefer=1)
    g.update([True, True], 0)
    assert g.executed == 1


def test_grasp_detector():
    d = GraspDetector(rest_z=0.9, lift_m=0.03, k=2)
    seq = [(0.95, False), (0.95, True), (0.91, True), (0.95, True), (0.96, True)]
    flags = [d.update(z, g, t) for t, (z, g) in enumerate(seq)]
    assert flags == [False, False, False, False, True] and d.stable_step == 4


def test_classify_outcome_priority():
    base = dict(target_goal=3, executed=None, target_grasped_end=False, target_z_end=0.9, rest_z=0.9,
                other_grasped=False, eef_path_end=0.1)
    assert classify_outcome(**{**base, "executed": 3}) == "success"
    assert classify_outcome(**{**base, "executed": 4}) == "wrong_goal"
    assert classify_outcome(**{**base, "other_grasped": True}) == "wrong_object"
    assert classify_outcome(**{**base, "target_grasped_end": True, "eef_path_end": 0.001}) == "stalled"
    assert classify_outcome(**{**base, "target_grasped_end": True}) == "holding"
    assert classify_outcome(**base) == "dropped"
    assert classify_outcome(**{**base, "target_z_end": 1.1}) == "misplaced"
    assert path_length([[0, 0, 0], [3, 4, 0], [3, 4, 0]]) == 5.0


def test_transform_and_perturb():
    ins = ["a b", "c d", "e f"]
    assert transform("other", "a b", task_id=2, all_instructions=ins) == "a b"
    assert transform("empty", "a b", task_id=0, all_instructions=ins) == ""
    assert sorted(transform("shuffle", "put the bowl on it", task_id=0, all_instructions=ins).split()) == \
        sorted("put the bowl on it".split())
    with pytest.raises(NotImplementedError):
        transform("cipher", "a", task_id=0, all_instructions=ins)
    s = "put the bowl on the plate"
    assert perturb("dup_article", s) == "put the the bowl on the plate"
    assert perturb("filler", s) == "put um the bowl on the plate"
    assert perturb("typo_article", s) == "put teh bowl on the plate"
    assert perturb("dup_article", "turn on stove") == "turn turn on stove"


def test_load_paraphrases_samples_per_category(tmp_path):
    p = tmp_path / "m.csv"
    lines = ["high,mid,low,eval,batch_idx,new_instruction,original_instruction"]
    lines += [f"act,lexical,x,0,{i},para {i},orig a" for i in range(10)]
    lines += [f"obj,lexical,x,0,{i},objpara {i},orig a" for i in range(3)]
    p.write_text("\n".join(lines))
    got = load_paraphrases(p, per_category=4, seed=0)
    assert len(got["orig a"]) == 4 + 3


class FakeEnv:
    def __init__(self, fail_at=None):
        self.t, self.fail_at = 0, fail_at

    def step(self, a):
        self.t += 1
        if self.t == self.fail_at:
            raise RuntimeError("sim blew up")
        return {"t": self.t}, 0, False, {}


class FakePolicy:
    def __init__(self):
        self.queries = 0

    def act(self, obs, instruction):
        self.queries += 1
        return [np.full(7, self.queries, float)] * 8


def test_rollout_requeries_per_chunk_and_stops():
    p = FakePolicy()
    res = rollout(FakeEnv(), p, "x", {}, steps=20, on_step=lambda t, o: t == 12)
    assert res["steps"] == 12 and p.queries == 2 and res["error"] is None
    assert same_actions(res["actions"], rollout(FakeEnv(), FakePolicy(), "x", {}, steps=12)["actions"])


def test_rollout_reports_errors():
    res = rollout(FakeEnv(fail_at=5), FakePolicy(), "x", {}, steps=20)
    assert res["error"].startswith("RuntimeError") and res["steps"] == 4


def test_restore_carries_gripper_fixtures_and_divergence():
    from types import SimpleNamespace
    from langeval.branch import divergence, restore, snapshot

    g = SimpleNamespace(current_action=np.array([-1.0, 1.0]))
    model = SimpleNamespace(body_pos=np.zeros((3, 3)), body_quat=np.ones((3, 4)))
    inner = SimpleNamespace(robots=[SimpleNamespace(gripper=g)], sim=SimpleNamespace(model=model))

    class Env:
        env = inner
        sim = np.array([1.0, 2.0])
        def get_sim_state(self): return self.sim
        def reset(self):                                         # what a robosuite hard reset does
            g.current_action = np.zeros(2)
            model.body_pos[:] = np.random.rand(3, 3)
        def set_init_state(self, s): self.sim = np.array(s); return {"x": self.sim}

    env = Env()
    snap = snapshot(env)
    env.sim, g.current_action = np.array([9.0, 9.0]), np.array([0.3, -0.3])
    obs = restore(env, snap)
    assert np.array_equal(obs["x"], [1.0, 2.0]) and np.array_equal(g.current_action, [-1.0, 1.0])
    assert np.array_equal(model.body_pos, np.zeros((3, 3)))

    a = [np.zeros(7), np.ones(7)]
    assert divergence(a, a)["first_diff_step"] is None
    d = divergence(a, [np.zeros(7), np.ones(7) * 1.5])
    assert d["first_diff_step"] == 1 and d["max_abs_diff"] == 0.5
