import argparse
import importlib.util
import io
import json
import random
from pathlib import Path

import numpy as np

from langeval.adapters.lerobot_policy import _batch
from langeval.adapters.starvla import StarVLAPolicy

spec = importlib.util.spec_from_file_location("qa", Path(__file__).parents[1] / "scripts/make_spatial_qa.py")
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)

FACTS = {"px": {n: (100 + 50 * i, 500) for i, n in enumerate({**qa.OBJECTS, **qa.FIXTURES})},
         "pos": {n: [0.1 * i, 0.0, 0.9] for i, n in enumerate({**qa.OBJECTS, **qa.FIXTURES})},
         "eef": [0.0, 0.0, 1.1], "held": "akita_black_bowl_1",
         "rests": {n: "the table" for n in qa.OBJECTS} | {"akita_black_bowl_1": "held by the gripper"},
         "drawers": {"top": False, "middle": False, "bottom": True}, "stove_on": False,
         "next": "transport", "task": "put the bowl on the plate", "target": "akita_black_bowl_1"}


def test_lang_and_num_ask_the_same_questions_ctrl_same_turns():
    lang = qa.conversation(FACTS, random.Random(0), "lang", 300)
    num = qa.conversation(FACTS, random.Random(0), "num", 300)
    ctrl = qa.conversation(FACTS, random.Random(0), "ctrl", 300)
    assert len(lang) == len(num) == len(ctrl) == 300
    assert all(n[0] == l[0] + " Answer in JSON." for l, n in zip(lang, num))
    for _, a in num:
        json.loads(a)
    assert all(a for _, a in lang)
    assert ("What is the gripper holding?", "The black bowl.") in lang
    assert any(q.startswith("Is the bottom drawer") and a == "Yes." for q, a in lang)


def test_batch_adds_leading_dim_everywhere():
    b = _batch({"pixels": {"a": np.zeros((4, 4, 3), np.uint8)}, "s": {"q": np.zeros(4), "m": np.eye(3)}})
    assert b["pixels"]["a"].shape == (1, 4, 4, 3) and b["s"]["q"].shape == (1, 4) and b["s"]["m"].shape == (1, 3, 3)


def test_starvla_client_flips_images_and_binarizes_gripper():
    sent = {}

    class Fake:
        def predict_action(self, q):
            sent.update(q)
            return {"data": {"actions": np.array([[[.1] * 6 + [.9], [.2] * 6 + [.1], [.3] * 7]])}}

    p = object.__new__(StarVLAPolicy)
    p.client, p.chunk, p.unnorm_key, p.episode = Fake(), 2, "franka", -1
    p.labels, p.swap, p.call = io.StringIO(), ["put the bowl on the plate", "turn on the stove"], 0
    img = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    p.begin()
    acts = p.act({"agentview_image": img, "robot0_eye_in_hand_image": img, "robot0_eef_pos": np.ones(3)},
                 "put the bowl on the plate")
    main, swap = [json.loads(l) for l in p.labels.getvalue().splitlines()]
    assert main["lang"] == "put the bowl on the plate" and main["episode"] == 0 and main["robot0_eef_pos"] == [1.0] * 3
    assert swap["lang"] == "turn on the stove" and swap["swap_of"] == main["lang"] and "swap_of" not in main
    p.act({"agentview_image": img, "robot0_eye_in_hand_image": img}, "put the bowl on the plate")
    assert len(p.labels.getvalue().splitlines()) == 3                     # no swaps off the SWAP_EVERY grid
    assert len(acts) == 2 and acts[0][6] == -1.0 and acts[1][6] == 1.0      # open (>0.5) -> -1, close -> +1
    assert np.array_equal(sent["examples"][0]["image"][0], img[::-1, ::-1])
    assert sent["examples"][0]["lang"] == "put the bowl on the plate" and sent["do_sample"] is False


def test_px_conventions_and_old_frames():
    f = {"rc": {"plate_1": [10, 200]}}
    assert qa.px_of(f, "none")["plate_1"] == (round(1000 * 200 / 256), round(1000 * 10 / 256))
    assert qa.px_of(f, "col")["plate_1"] == (round(1000 * 55 / 256), round(1000 * 10 / 256))
    both = qa.px_of(f, "both")
    old = {"px": both}                               # first-collect frames stored the both-flipped px
    assert abs(qa.rc_of(old)["plate_1"][0] - 10) <= 1 and abs(qa.rc_of(old)["plate_1"][1] - 200) <= 1


def test_num_next_step_carries_the_object():
    num = qa.conversation(FACTS, random.Random(0), "num", 300)
    nxt = [json.loads(a) for q, a in num if "do next" in q]
    assert nxt and all(v == {"next": "transport", "object": "black bowl"} for v in nxt)


def test_eval_qa_scorer():
    spec = importlib.util.spec_from_file_location("eq", Path(__file__).parents[1] / "scripts/eval_qa.py")
    eq = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eq)
    s = eq.score
    assert s("The plate is at (430, 520).", "The plate is at (412, 530).", "position")
    assert not s("The plate is at (480, 520).", "The plate is at (412, 530).", "position")
    assert not s("The bowl is at (412, 530).", "The plate is at (412, 530).", "position")
    assert s("About 21 cm. The plate is 9 cm below the gripper.", "About 22 cm. The plate is 10 cm below the gripper.", "gripper_offset")
    assert not s("About 22 cm. The plate is 10 cm above the gripper.", "About 22 cm. The plate is 10 cm below the gripper.", "gripper_offset")
    assert s('{"held":null}', '{"held":null}', "held") and not s('{"held":"plate"}', '{"held":null}', "held")
    assert s('Sure: {"cm":23,"dz_cm":-9}', '{"cm":22,"dz_cm":-10}', "gripper_offset")
    assert not s('{"open":1}', '{"open":true}', "drawer") and not s("no json", '{"open":true}', "drawer")
    assert eq.qtype("Where is the plate in the image?") == "position" and eq.qtype("Is the top drawer of the cabinet open?") == "drawer"


def test_eval_qa_content_facts_accept_free_phrasing():
    spec = importlib.util.spec_from_file_location("eq", Path(__file__).parents[1] / "scripts/eval_qa.py")
    eq = importlib.util.module_from_spec(spec); spec.loader.exec_module(eq)
    cf = eq.content_facts
    assert cf("No, the drawer appears to be closed.", "No.", "drawer") == {"drawer": True}
    assert cf("The top drawer is open.", "No.", "drawer") == {"drawer": False}
    assert cf("The stove is off.", "No.", "stove") == {"stove": True}
    assert cf("The gripper is not holding anything; it is empty.", "Nothing.", "held") == {"held": True}
    assert cf("It is holding a black bowl.", "Nothing.", "held") == {"held": False}
    assert cf("The robot holds the wine bottle.", "The wine bottle.", "held") == {"held": True}
    assert cf("It sits on the wooden table.", "The plate is on the table.", "support") == {"support": True}
    ref = "The plate is to the right of the wine bottle, about 26 cm away."
    assert cf("The plate is on the right side of the wine bottle, roughly 0.25 m apart.", ref, "side_distance") == {"side": True, "distance": True}
    assert cf("The wine bottle is to the left of the plate, about 60 cm.", ref, "side_distance") == {"side": True, "distance": False}
    assert cf("The plate is left of the bottle.", ref, "side_distance") == {"side": False, "distance": False}
    assert cf("Roughly 23 cm away.", "About 24 cm. The black bowl is 24 cm below the gripper.", "gripper_offset") == {"distance": True}
    assert cf("[470, 580, 520, 630]", "The black bowl is at (492, 602).", "position") == {"position": True}
    assert cf("(100, 602)", "The black bowl is at (492, 602).", "position") == {"position": False}
    rows = [{"q": "What is the gripper holding?", "type": "held", "ref": "Nothing.", "pred": "Nothing.", "ok": True},
            {"q": "What is the gripper holding?", "type": "held", "ref": "The plate.", "pred": "Nothing.", "ok": False}]
    s = eq.summarize(rows)
    assert s["held"]["exact"] == 0.5 and s["held"]["majority_answer"] == 0.5 and s["held.held"]["content"] == 0.5
    g = [{"q": "How far is the gripper from the plate?", "type": "gripper_offset", "pred": "",  "ok": False,
          "ref": f"About {c} cm. The plate is {c} cm below the gripper."} for c in (10, 30, 31, 32)]
    assert eq.summarize(g)["gripper_offset.distance"]["constant_answer"] == 0.75     # best constant, not a tie pick


def test_eval_qa_next_step_type_wins_over_drawer():
    spec = importlib.util.spec_from_file_location("eq", Path(__file__).parents[1] / "scripts/eval_qa.py")
    eq = importlib.util.module_from_spec(spec); spec.loader.exec_module(eq)
    assert eq.qtype("The task is: open the middle drawer of the cabinet. What should the robot do next?") == "next_step"
    assert eq.qtype("Is the top drawer of the cabinet open?") == "drawer"
    assert eq.content_facts("They are approximately 0.25 meters apart. The plate is to the right of the wine bottle.",
                            "The plate is to the right of the wine bottle, about 26 cm away.", "side_distance") == {"side": True, "distance": True}


def test_eval_qa_ctrl_content():
    spec = importlib.util.spec_from_file_location("eq", Path(__file__).parents[1] / "scripts/eval_qa.py")
    eq = importlib.util.module_from_spec(spec); spec.loader.exec_module(eq)
    cf = eq.content_facts
    yes, no = "Yes, there is a black bowl in this image.", "No, there is no frying pan in this image."
    assert eq.qtype("Is there a black bowl in this image?") == "ctrl_presence"
    assert cf("There is no black bowl visible.", yes, "ctrl_presence") == {"presence": False}
    assert cf("Yes, a dark bowl sits on the table.", yes, "ctrl_presence") == {"presence": True}
    assert cf("I don't see a frying pan.", no, "ctrl_presence") == {"presence": True}
    assert cf("It could grab the wine bottle.", "The robot could pick up the plate here.", "ctrl_name") == {"name": True}


def test_probe_recovers_linear_signal_and_not_noise():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import probe_vlm as P
    rng = np.random.default_rng(0)
    n = 600
    y_cls, y_reg = rng.integers(0, 3, n), rng.normal(size=n)
    sig = np.c_[np.eye(3)[y_cls], y_reg]
    X = np.c_[sig, rng.normal(size=(n, 50))] @ rng.normal(size=(54, 80))
    scene = np.arange(n) // 10
    te, va = scene % 5 == 4, scene % 5 == 3
    labs = {"c": ("cls", y_cls), "r": ("reg", y_reg)}
    good = P.probes(X, labs, ~te, va, te)
    bad = P.probes(rng.normal(size=(n, 80)), labs, ~te, va, te)
    assert good["c"][1] > 0.95 and good["r"][1] > 0.9
    assert bad["c"][1] < 0.5 and bad["r"][1] < 0.1


def test_balanced_build_evens_held_answers():
    held_f = {**FACTS}
    empty_f = {**FACTS, "held": None, "rests": {n: "the table" for n in qa.OBJECTS}}
    frames = [{"facts": held_f}] * 20 + [{"facts": empty_f}] * 80          # 80% "Nothing."
    rates = qa.balance_rates(frames)
    assert abs(rates[("held", True)] - 0.25) < 1e-9 and rates[("held", False)] == 1.0
    assert rates[("top_drawer", False)] == 0.0                            # top drawer never open: never asked
    rng, held = random.Random(0), []
    for f in frames * 10:
        for q, a in qa.conversation(f["facts"], rng, "lang", 6, rates):
            assert "top drawer" not in q
            if q == "What is the gripper holding?":
                held.append(a == "Nothing.")
    assert 0.4 < np.mean(held) < 0.6


def test_counter_pairs_groups_order_and_qa(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
    import make_counter_pairs as C
    spec = importlib.util.spec_from_file_location("eq", Path(__file__).parents[1] / "scripts/eval_qa.py")
    eq = importlib.util.module_from_spec(spec); spec.loader.exec_module(eq)
    ray = {"kind": "ray", "reach_err": 0.005, "px2d": 3.0}
    assert C.group({**ray, "dist3d": 0.06}) == "consistent" and C.group({**ray, "dist3d": 0.25}) == "counter"
    assert C.group({**ray, "dist3d": 0.14}) is None and C.group({**ray, "dist3d": 0.25, "px2d": 40}) is None
    assert C.group({**ray, "dist3d": 0.06, "reach_err": 0.05}) is None
    assert C.group({"kind": "grip", "reach_err": 0.05, "held": False}) == "counter"      # blocked descent is fine
    parent, true = np.array([0, 0, 0, 1, 1, 1]), np.array([.06, .12, .19, .06, .12, .19])
    assert C.pair_order(parent, true, true * 2) == 1.0 and C.pair_order(parent, true, -true) == 0.0
    assert C.pair_order(parent, true, np.zeros(6)) == 0.0          # ties are not "ordered"
    fr = [{"image": "a.png", "scene": 0, "facts": FACTS, "cf": {"kind": "ray", "object": "plate_1"}},
          {"image": "b.png", "scene": 0, "facts": FACTS, "cf": {"kind": "grip", "object": "plate_1"}}]
    (tmp_path / "llava_jsons").mkdir()
    C.write_qa(tmp_path, fr)
    data = json.loads((tmp_path / "llava_jsons" / "libero_qa_lang_heldout.json").read_text())
    for item, k in zip(data, ("gripper_offset", "held")):
        q, ref = item["conversations"][0]["value"].replace("<image>\n", ""), item["conversations"][1]["value"]
        assert eq.qtype(q) == k and all(eq.content_facts(ref, ref, k).values())


def test_action_eraser_leaves_no_linear_covariance(tmp_path):
    spec = importlib.util.spec_from_file_location("ea", Path(__file__).parents[1] / "scripts/erase_action.py")
    ea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ea)
    rng = np.random.default_rng(1)
    n, T, H = 300, 4, 64
    pos = rng.uniform(-.3, .3, (n, 5, 3))
    X = rng.standard_normal((n, T, H)) * .1 + (pos.reshape(n, -1) @ rng.standard_normal((15, H)))[:, None, :]
    X.astype(np.float16).tofile(tmp_path / "rec.f16")
    json.dump({"shape": [T, H]}, open(tmp_path / "rec.f16.json", "w"))
    with open(tmp_path / "rec.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"episode": i // 10, "robot0_eef_pos": pos[i, 0].tolist(), "robot0_gripper_qpos": [0, 0],
                                **{f"{o}_pos": pos[i, j + 1].tolist() for j, o in enumerate(ea.OBJ)}}) + "\n")
    X, L, Z = ea.load(str(tmp_path / "rec.f16"), str(tmp_path / "rec.jsonl"))
    ep = np.array([l["episode"] for l in L])
    z = Z["geom"]
    U, R = ea.erasers(X, z, rng)
    cov = lambda A: np.abs(np.einsum("nth,nk->thk", A - A.mean(0), z - z.mean(0))).max()
    assert cov(ea.apply(X, X.mean(0), U)) < 1e-4 * cov(X)
    te = ep % 5 == 4
    assert ea.r2(X[~te], z[~te], X[te], z[te]) > 0.95
    assert ea.r2(*(lambda Xe: (Xe[~te], z[~te], Xe[te], z[te]))(ea.apply(X, X.mean(0), U))) < 0.05
    assert R[0].shape == U.shape and np.abs(np.einsum("thk,thj->tkj", U, R[0])).max() < 0.3
    assert ea.knn_r2(X[~te], z[~te], X[te], z[te]) > 0.5


def test_nearest_seen_ignores_the_shared_observation_and_itself():
    spec = importlib.util.spec_from_file_location("ea", Path(__file__).parents[1] / "scripts/erase_action.py")
    ea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ea)
    rng = np.random.default_rng(0)
    base, d = rng.standard_normal(50) * 100, rng.standard_normal((3, 50))      # big shared part, 3 goal directions
    V = np.stack([base + d[0], base + d[1], base + d[2], base + d[1] + .1 * d[2]])   # s0 s1 s2, n0 ~ s1
    near = ea.nearest_seen(V, ["s0", "s1", "s2", "n0"], ["s0", "s1", "s2"])
    assert near["n0"] == "s1" and near["s1"] != "s1"


def test_contrast_basis_drops_the_shared_direction():
    spec = importlib.util.spec_from_file_location("ea", Path(__file__).parents[1] / "scripts/erase_action.py")
    ea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ea)
    rng = np.random.default_rng(0)
    e = np.eye(20)
    shared, own = rng.standard_normal((500, 1)) * 10, rng.standard_normal((500, 1))   # shared dir 0 dominates both
    Dp = (shared * e[0] + own * e[1] + rng.standard_normal((500, 20)) * .05)[:, None]
    Dn = (rng.standard_normal((500, 1)) * 10 * e[0] + rng.standard_normal((500, 20)) * .05)[:, None]
    assert abs(ea.top_basis(Dp, 1)[0, 0, 0]) > .99 and abs(ea.contrast_basis(Dp, Dn, 1)[0, 1, 0]) > .99


def test_erase_action_commands_exist():
    spec = importlib.util.spec_from_file_location("ea", Path(__file__).parents[1] / "scripts/erase_action.py")
    ea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ea)
    assert all(callable(getattr(ea, c, None)) for c in ("fit", "binding", "subspaces", "alphas", "patch_report", "landing"))


def test_patch_branch_lang_override_and_attention_spans():
    spec = importlib.util.spec_from_file_location("pb", Path(__file__).parents[1] / "scripts/patch_branch.py")
    pb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pb)
    goals = [argparse.Namespace(language="put the bowl on the stove"), argparse.Namespace(language="x")]
    assert pb.parse_lang("n3=s1;s8=;n4=put it on the dax", goals, ["s1", "n3"]) == \
        {"n3": "put the bowl on the stove", "s8": "", "n4": "put it on the dax"}
    pad = "<|image_pad|>"
    toks = ["<|im_start|>", pad, pad, pad, pad, pad, "<|vision_end|>", "put", "\u0120the", "\u0120bowl", "\u0120Please"]
    sp = pb.attn_split(toks, np.array([[1, 4, 4], [1, 2, 2]]), 2)      # 4 tokens, then 1 token
    assert sp == {"images": [[1, 2, 3, 4], [5]], "instr": [7, 8, 9]}
    assert all(callable(getattr(pb, c)) for c in ("collect", "branch", "probe", "attn", "norms", "report"))


def test_probe_goals_never_hold_and_moved_object(tmp_path, capsys):
    from langeval.goal_predicates import Goal, satisfied
    env = argparse.Namespace(env=argparse.Namespace(_eval_predicate=lambda p: True))
    assert satisfied(env, [Goal("probe", [], None), Goal("g", [["on", "a", "b"]], "a")]) == [False, True]
    spec = importlib.util.spec_from_file_location("pb", Path(__file__).parents[1] / "scripts/patch_branch.py")
    pb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pb)
    run = tmp_path / "keys"
    run.mkdir()
    e = {"goal": "k1", "language": "put the plate on the bowl", "executed": "s8",
         "start_pos": {"plate_1_pos": [0, 0, 0], "akita_black_bowl_1_pos": [0, 0, 0], "robot0_eef_pos": [0, 0, 0],
                       "plate_1_to_robot0_eef_pos": [0, 0, 0]},
         "final_pos": {"plate_1_pos": [0, 0.01, 0], "akita_black_bowl_1_pos": [0.2, 0, 0], "robot0_eef_pos": [1, 1, 1],
                       "plate_1_to_robot0_eef_pos": [1, 1, 1]}}
    (run / "episodes.jsonl").write_text(json.dumps(e) + "\n")
    pb.moved(argparse.Namespace(runs=[str(run)]))
    assert "moved {'akita_black_bowl_1': 1}" in capsys.readouterr().out


def test_traj_sim_resample_and_post_grasp():
    spec = importlib.util.spec_from_file_location("ts", Path(__file__).parents[1] / "scripts/traj_sim.py")
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)
    line = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], float)
    assert np.allclose(ts.resample(line, 4)[:, 0], [0, 1, 2, 3])
    z = np.array([0, 0, 0, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1])
    xyz = np.stack([np.zeros_like(z), np.zeros_like(z), z], 1)
    grip = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)            # 1 = open here: closes at frame 2
    assert np.allclose(ts.post_grasp(xyz, grip)[0], [0, 0, 0.04])          # first frame 3 cm above the close
    assert ts.post_grasp(xyz, np.ones_like(z)) is None                       # never closes


def test_paired_counts_and_mcnemar(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("pb", Path(__file__).parents[1] / "scripts/patch_branch.py")
    pb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pb)
    assert pb.mcnemar_p(0, 0) == 1.0 and abs(pb.mcnemar_p(6, 0) - 2 / 64) < 1e-12
    for name, succ in (("a", [0, 0, 1, 1]), ("b", [1, 1, 1, 0])):
        (tmp_path / name).mkdir()
        (tmp_path / name / "episodes.jsonl").write_text(
            "".join(json.dumps({"goal": "n3", "init": i, "success": bool(v)}) + "\n" for i, v in enumerate(succ)))
    pb.paired(argparse.Namespace(base=str(tmp_path / "a"), runs=[str(tmp_path / "b")]))
    assert "rescued 2  lost 1" in capsys.readouterr().out


def test_landing_anchors_use_seen_successes_only(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("ea", Path(__file__).parents[1] / "scripts/erase_action.py")
    ea = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ea)
    run = tmp_path / "run"
    run.mkdir()
    start = {f"{o}_pos": [0, 0, 0] for o in ea.OBJ}
    # s1 (bowl -> stove) succeeds with the bowl at x=1; n1 (cheese -> stove) succeeds with the cheese at x=5
    ends = [("s1", {**start, "akita_black_bowl_1_pos": [1, 0, 0]}), ("n1", {**start, "cream_cheese_1_pos": [5, 0, 0]})]
    with open(run / "episodes.jsonl", "w") as f:
        for g, end in ends:
            f.write(json.dumps({"goal": g, "success": True, "final_pos": end}) + "\n")
    with open(run / "labels.jsonl", "w") as f:
        for i in range(2):
            f.write(json.dumps({"episode": i, **start}) + "\n")
    ea.landing(argparse.Namespace(anchor_run=str(run), anchor_labels=str(run / "labels.jsonl"), goals="", runs=[str(run)]))
    out = capsys.readouterr().out
    assert "stove n=1" in out and "n1 cream->stove" in out and "target median  400 cm" in out


def test_starvla_client_sends_donor_as_second_example():
    sent = {}

    class Fake:
        def predict_action(self, q):
            sent.update(q)
            return {"data": {"actions": np.array([[[.1] * 7] * 2, [[.9] * 7] * 2])}}

    p = object.__new__(StarVLAPolicy)
    p.client, p.chunk, p.unnorm_key, p.labels, p.call = Fake(), 2, "franka", None, 0
    p.patch = {"put the cream cheese on top of the cabinet": ["put the cream cheese in the bowl"]}
    img = np.zeros((224, 224, 3), np.uint8)
    acts = p.act({"agentview_image": img, "robot0_eye_in_hand_image": img}, "put the cream cheese on top of the cabinet")
    assert [e["lang"] for e in sent["examples"]] == ["put the cream cheese on top of the cabinet", "put the cream cheese in the bowl"]
    assert acts[0][0] == np.float32(.1)                                    # the requested goal's actions, not the donor's
    p.patch = {"a": ["b", "c"]}                                            # several donors: all after A, in order
    p.act({"agentview_image": img, "robot0_eye_in_hand_image": img}, "a")
    assert [e["lang"] for e in sent["examples"]] == ["a", "b", "c"]
