"""KV-cache tests. They use a small randomly initialized model, so no checkpoint is needed.

Run from the repo root:  pytest -q tests/test_kv_cache.py
"""
import pytest
import torch

from gpt.config import Config
from gpt.generate import generate, generate_cached
from gpt.model import GPT
from helpers import V, CTX, rand_ids


# V, CTX = 100, 64


# @pytest.fixture(autouse=True)
# def no_grad():
#     with torch.no_grad():
#         yield


# @pytest.fixture(scope="module")
# def model():
#     torch.manual_seed(0)
#     cfg = Config(n_layer=2, n_head=2, d_model=64, context_len=CTX, batch_size=4, vocab_size=V)
#     m = GPT(cfg)
#     # the 0.02 init gives almost flat logits, so argmax gaps could sit near float noise;
#     # larger weights make the argmax comparisons meaningful
#     for p in m.parameters():
#         if p.dim() >= 2:
#             torch.nn.init.normal_(p, std=0.15)
#     return m.eval()





def run_cached(model, idx, t0, cache):
    """Prefill the first t0 tokens, then decode one token at a time; returns logits for every position."""
    out, _ = model(idx[:, :t0], start_pos=0, cache=cache)
    steps = [out]
    for t in range(t0, idx.shape[1]):
        out, _ = model(idx[:, t:t + 1], start_pos=t, cache=cache)
        steps.append(out)
    return torch.cat(steps, dim=1)


# ---------------------------------------------------------------- cache object

def test_make_cache(model):
    cfg = model.cfg
    cache = model.make_cache(2)
    assert len(cache) == cfg.n_layer
    ptrs = [t.data_ptr() for kv in cache for t in kv]
    assert len(set(ptrs)) == 2 * cfg.n_layer                      # no storage shared between layers or between K and V
    for k, v in cache:
        assert k.shape == v.shape == (2, cfg.n_head, cfg.context_len, cfg.d_model // cfg.n_head)
        assert k.dtype == model.tok_emb.weight.dtype
    assert model.make_cache(1, dtype=torch.float16)[0][0].dtype == torch.float16


# ---------------------------------------------------------------- logits level

@pytest.mark.parametrize("B,t0", [(1, 1), (1, 7), (1, CTX), (3, 1), (3, 16)])
def test_cached_matches_full(model, B, t0):
    idx = rand_ids(B, CTX)
    full, _ = model(idx)
    cached = run_cached(model, idx, t0, model.make_cache(B))
    assert torch.allclose(cached, full, atol=1e-4)
    assert torch.equal(cached.argmax(-1), full.argmax(-1))        # if this fails while allclose passes, look for near-ties


def test_cache_reuse(model):
    cache = model.make_cache(1)
    run_cached(model, rand_ids(1, 50, seed=1), 10, cache)         # fill the cache with a longer sequence first
    short = rand_ids(1, 30, seed=2)
    again = run_cached(model, short, 10, cache)                   # stale entries beyond position 30 must never be read
    full, _ = model(short)
    assert torch.allclose(again, full, atol=1e-4)


def test_rows_do_not_leak(model):
    idx = rand_ids(3, 40)
    batched = run_cached(model, idx, 5, model.make_cache(3))
    for b in range(3):
        alone = run_cached(model, idx[b:b + 1], 5, model.make_cache(1))
        assert torch.allclose(batched[b:b + 1], alone, atol=1e-4)


def test_guards(model):
    with pytest.raises(AssertionError):                           # several tokens into a non-empty cache
        model(rand_ids(1, 2), start_pos=5, cache=model.make_cache(1))
    with pytest.raises(AssertionError):                           # writing past the end of the cache
        model(rand_ids(1, 1), start_pos=CTX, cache=model.make_cache(1))
    with pytest.raises(AssertionError):                           # start_pos without a cache
        model(rand_ids(1, 1), start_pos=3)
    with pytest.raises(AssertionError):                           # batch size does not match the cache
        model(rand_ids(2, 1), start_pos=0, cache=model.make_cache(1))


# ---------------------------------------------------------------- generation level

@pytest.mark.parametrize("prompt_len", [1, 5, 20])
def test_greedy_generation_identical(model, prompt_len):
    idx = rand_ids(1, prompt_len)
    n = CTX - prompt_len                                          # stay inside the window: the cache cannot slide
    a = generate(model, CTX, idx, n, 1.0, 1, None, -1)            # eot_id=-1 never matches, so there is no early stop
    b = generate_cached(model, CTX, idx, n, 1.0, 1, None, -1)
    assert torch.equal(a, b)


def test_sampled_generation_identical(model):
    idx = rand_ids(1, 8)
    a = generate(model, CTX, idx, 40, 0.8, 20, torch.Generator().manual_seed(0), -1)
    b = generate_cached(model, CTX, idx, 40, 0.8, 20, torch.Generator().manual_seed(0), -1)
    assert torch.equal(a, b)                                      # same seed, same logits up to ~1e-5, so the same draws


def test_stops_at_eot(model):
    idx = rand_ids(1, 8)
    first = model(idx)[0][0, -1].argmax().item()                  # the token greedy decoding produces first
    for fn in (generate, generate_cached):
        out = fn(model, CTX, idx, 20, 1.0, 1, None, first)
        assert out.shape[1] == 9 and out[0, -1].item() == first  # stops right after appending the eot token


def test_length_cap(model):
    with pytest.raises(AssertionError):
        generate_cached(model, CTX, rand_ids(1, 10), CTX, 1.0, 1, None, -1)