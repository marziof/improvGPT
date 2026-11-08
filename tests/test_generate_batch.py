
from gpt.generate import generate_cached, generate_batch
import torch
import pytest
from helpers import V, CTX, rand_ids


def test_rows_match_single_greedy(model):
    idx = rand_ids(4, 10)
    out, fin, n = generate_batch(model, CTX, idx, 30, 1.0, 1, None, [-1], 0)     # [-1] never matches a token
    for b in range(4):
        ref = generate_cached(model, CTX, idx[b:b + 1], 30, 1.0, 1, None, -1)
        assert torch.equal(out[b:b + 1], ref)
    assert not fin.any() and (n == 30).all()

def test_stopping_and_padding(model):
    idx = rand_ids(4, 10)
    first = model(idx[:1])[0][0, -1].argmax().item()                             # row 0 stops after one token
    out, fin, n = generate_batch(model, CTX, idx, 30, 1.0, 1, None, [first], 0)
    assert fin[0] and n[0] == 1
    for b in range(4):                                                           # every row equals its own single run, then padding
        ref = generate_cached(model, CTX, idx[b:b + 1], 30, 1.0, 1, None, first)
        L = ref.shape[1]
        assert torch.equal(out[b, :L], ref[0])
        assert (out[b, L:] == 0).all()
        assert n[b] == L - 10