"""Tests for the improv example format (src/gpt/improv_data.py).

Interfaces these tests assume:
  split_turn(turn) -> (tag, body)                      e.g. "JULIET:\nA gift?" -> ("JULIET:", "A gift?")
  build_example(context_turns, target_turn, tok, eot_id, eoturn_id) -> (ids, prompt_len)
      ids = [eot] + tokens(prompt) + tokens(reply) + [eoturn]; prompt_len counts the eot and the prompt tokens
  to_row(ids, row_len, pad_id) -> np.ndarray (uint16, length row_len); raises ValueError if ids are too long
  batch_from_rows(rows, prompt_lens, total_lens) -> (x, y)   x: (B, row_len-1), y: (B, row_len-1), -100 where masked

Run from the repo root:  pytest -q tests/test_improv_data.py
"""
import numpy as np
import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from gpt.improv_data import batch_from_rows, build_example, split_turn, to_row


ROW_LEN = 128
CTX_TURNS = ["ROMEO:\nI have a gift for thee.", "JULIET:\nA gift? What manner of gift?"]
TARGET = "ROMEO:\nA boat, that we may sail away tonight."
CORPUS = [
    "ROMEO:\nI have a gift for thee.\n\nJULIET:\nA gift? What manner of gift?\n\nROMEO:\nA boat, that we may sail away tonight.",
    "First Citizen:\nBefore we proceed any further, hear me speak.\n\nAll:\nSpeak, speak.",
    "PETRUCHIO:\nGood morrow, sweet Kate! Hast thou seen my hat?\n\nKATHARINA:\nNo, sir; I would not wear it if I had.",
] * 20


@pytest.fixture(scope="module")
def tok():
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=300,
        special_tokens=["<|endoftext|>", "<|endofturn|>", "<|scene|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    t.train_from_iterator(CORPUS, trainer=trainer)
    return t


@pytest.fixture(scope="module")
def ids_(tok):
    return tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|endofturn|>")


def test_split_turn():
    assert split_turn("JULIET:\nA gift?\nWhat gift?") == ("JULIET:", "A gift?\nWhat gift?")


def test_build_example_layout(tok, ids_):
    eot, eoturn = ids_
    ids, P = build_example(CTX_TURNS, TARGET, tok, eot, eoturn)
    assert ids[0] == eot and ids[-1] == eoturn
    assert eot not in ids[1:] and eoturn not in ids[:-1]
    assert tok.decode(ids[1:P]) == ("ROMEO:\nI have a gift for thee.\n\nJULIET:\nA gift? What manner of gift?\n\nROMEO:\n")
    assert tok.decode(ids[P:-1]) == "A boat, that we may sail away tonight."


def test_prompt_and_reply_tokenize_like_joint_text(tok, ids_):
    # inference tokenizes the prompt and then samples the reply; training must see the same token boundaries
    eot, eoturn = ids_
    ids, _ = build_example(CTX_TURNS, TARGET, tok, eot, eoturn)
    joint = tok.encode("\n\n".join(CTX_TURNS) + "\n\n" + TARGET).ids
    assert ids[1:-1] == joint


def test_to_row():
    row = to_row([5, 6, 7], 10, pad_id=0)
    assert row.shape == (10,) and row.dtype == np.uint16
    assert row[:3].tolist() == [5, 6, 7] and (row[3:] == 0).all()
    with pytest.raises(ValueError):
        to_row(list(range(11)), 10, pad_id=0)


def test_labels_are_masked_to_the_reply(tok, ids_):
    eot, eoturn = ids_
    ids, P = build_example(CTX_TURNS, TARGET, tok, eot, eoturn)
    row = to_row(ids, ROW_LEN, pad_id=eot)
    x, y = batch_from_rows(row[None], [P], [len(ids)])
    assert x.shape == y.shape == (1, ROW_LEN - 1)
    valid = (y[0] != -100).nonzero().flatten() + 1             # indices in the row that are real targets
    assert valid.tolist() == list(range(P, len(ids)))          # exactly the reply tokens and the end-of-turn
    assert y[0][valid - 1].tolist() == ids[P:]
    assert y[0][valid[-1] - 1].item() == eoturn                # the last target teaches the model to stop
    assert x[0, : len(ids) - 1].tolist() == ids[:-1]           # inputs are the row shifted by one


def test_batch_of_different_lengths(tok, ids_):
    eot, eoturn = ids_
    a, Pa = build_example(CTX_TURNS[:1], TARGET, tok, eot, eoturn)
    b, Pb = build_example(CTX_TURNS, TARGET, tok, eot, eoturn)
    rows = np.stack([to_row(a, ROW_LEN, eot), to_row(b, ROW_LEN, eot)])
    x, y = batch_from_rows(rows, [Pa, Pb], [len(a), len(b)])
    assert (y != -100).sum(dim=1).tolist() == [len(a) - Pa, len(b) - Pb]
    assert (y[0, len(a) - 1:] == -100).all()                   # padding after the sequence never counts



# addtional


import random

import pytest

from gpt.improv_data import split_turn
#from scripts.prepare_improv import build_improv_examples
from gpt.improv_data import build_improv_examples


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    """Simple deterministic tokenizer for dataset tests."""

    def encode(self, text):
        return FakeEncoding(list(text.encode("utf-8")))


    


@pytest.fixture
def fake_tok():
    return FakeTokenizer()


@pytest.fixture
def synthetic_turns():
    """Long enough to exercise multiple blocks and episodes."""
    speakers = ["ROMEO", "JULIET", "MERCUTIO"]

    return [
        f"{speakers[i % 3]}:\nLine number {i} with some dialogue."
        for i in range(1100)
    ]


def generate(turns, fake_tok, seed=42):
    return build_improv_examples(
        turns=turns,
        tok=fake_tok,
        eot_id=0,
        eoturn_id=1,
        block_size=50,
        val_every=10,
        min_episode=10,
        max_episode=15,
        min_context=1,
        max_context=3,
        max_reply_tokens=40,
        max_total_tokens=256,
        seed=seed,
    )


def test_block_boundaries(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    for example in train + val:
        block_start = example["block_index"] * 50
        block_end = min(block_start + 50, len(synthetic_turns))

        assert block_start <= example["target_index"] < block_end

        for idx in example["context_indices"]:
            assert block_start <= idx < block_end


def test_validation_blocks_are_isolated(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    train_blocks = {ex["block_index"] for ex in train}
    val_blocks = {ex["block_index"] for ex in val}

    assert train_blocks.isdisjoint(val_blocks)

    for ex in val:
        assert (ex["block_index"]+1) % 10 == 0

    for ex in train:
        assert (ex["block_index"]+1) % 10 != 0


def test_episodes_stay_inside_blocks(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    for ex in train + val:
        block_start = ex["block_index"] * 50
        block_end = min(block_start + 50, len(synthetic_turns))

        assert block_start <= ex["episode_start"]
        assert ex["episode_end"] <= block_end

        assert (
            ex["episode_start"]
            <= ex["target_index"]
            < ex["episode_end"]
        )

        for idx in ex["context_indices"]:
            assert ex["episode_start"] <= idx < ex["episode_end"]


def test_episode_lengths(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    episodes = {}

    for ex in train + val:
        key = (ex["block_index"], ex["episode_index"])
        episodes[key] = (
            ex["episode_end"] - ex["episode_start"]
        )

    for length in episodes.values():
        assert 1 <= length <= 15

        # Short final episodes are allowed.
        # All other episodes should be at least 10 turns.
        # The final-episode distinction is checked separately
        # by the episode-construction tests.


def test_user_character_is_persistent(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    episodes = {}

    for ex in train + val:
        key = (ex["block_index"], ex["episode_index"])

        if key not in episodes:
            episodes[key] = ex["user_speaker"]
        else:
            assert episodes[key] == ex["user_speaker"]


def test_target_is_not_user_character(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    for ex in train + val:
        target_speaker, _ = split_turn(
            synthetic_turns[ex["target_index"]]
        )

        assert target_speaker != ex["user_speaker"]


def test_user_character_appears_in_context(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    for ex in train + val:
        context_speakers = {
            split_turn(synthetic_turns[idx])[0]
            for idx in ex["context_indices"]
        }

        assert ex["user_speaker"] in context_speakers


def test_context_length(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    for ex in train + val:
        assert 1 <= len(ex["context_indices"]) <= 3


def test_target_follows_context(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    for ex in train + val:
        assert ex["context_indices"]
        assert max(ex["context_indices"]) < ex["target_index"]


def test_deterministic_generation(fake_tok, synthetic_turns):
    train1, val1, _ = generate(synthetic_turns, fake_tok, seed=42)
    train2, val2, _ = generate(synthetic_turns, fake_tok, seed=42)

    assert train1 == train2
    assert val1 == val2


def test_token_limits(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    for ex in train + val:
        assert ex["total_len"] <= 256
        assert ex["prompt_len"] < ex["total_len"]


def test_nonempty_dataset(fake_tok, synthetic_turns):
    train, val, stats = generate(synthetic_turns, fake_tok)

    assert len(train) > 0
    assert len(val) > 0
