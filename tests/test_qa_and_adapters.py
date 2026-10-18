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
