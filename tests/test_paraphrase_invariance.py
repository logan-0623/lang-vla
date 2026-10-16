import importlib.util
from pathlib import Path

import re

import numpy as np

spec = importlib.util.spec_from_file_location("pi", Path(__file__).parents[1] / "scripts/paraphrase_invariance.py")
pi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pi)


def ws_tokenize(s):
    # Attaches the leading space to the word and splits off "?", like BPE / sentencepiece.
    return [hash(w) for w in re.findall(r" ?[^ ?]+| ?\?", s)]


def test_instruction_span_marks_exactly_the_instruction():
    for template in (pi.LLAMA_PROMPT, "go to {} now"):
        instr = "put the bowl on the plate"
        full = ws_tokenize(template.format(instr))
        s, e = pi.instruction_span(ws_tokenize, template, instr)
        assert e - s == len(instr.split())
        assert full[s:e] == ws_tokenize(" " + instr)


def test_instruction_span_empty_instruction():
    s, e = pi.instruction_span(ws_tokenize, pi.LLAMA_PROMPT, "")
    assert s == e


def test_instruction_span_last_token_merges_with_template():
    # A tokenizer that merges "??" into one token, as BPE does.
    merge = lambda s: [hash(w) for w in re.findall(r" ?[^ ?]+| ?\?\??", s)]
    instr = "could you open the drawer?"
    s, e = pi.instruction_span(merge, pi.LLAMA_PROMPT, instr)
    full = merge(pi.LLAMA_PROMPT.format(instr))
    assert full[s:e] == merge(" could you open the drawer")


def test_ratio_handles_zero_reference():
    r = pi.ratio(np.array([1.0, 2.0, 0.0]), np.array([2.0, 0.0, 0.0]))
    assert r[0] == 0.5 and np.isnan(r[1]) and np.isnan(r[2])
