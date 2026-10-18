import importlib.util
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
    p.client, p.chunk, p.unnorm_key = Fake(), 2, "franka"
    img = np.arange(224 * 224 * 3, dtype=np.uint8).reshape(224, 224, 3)
    acts = p.act({"agentview_image": img, "robot0_eye_in_hand_image": img}, "put the bowl on the plate")
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
