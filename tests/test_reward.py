"""Tests that define src/gpt/reward.py.

Interface these tests assume:
  words(s) -> list[str]                  lowercase word tokens, apostrophes stripped from the ends
  content(s) -> set[str]                 words that are not stopwords and have more than 2 letters
  has_leak(reply) -> bool                True if the reply contains a line that looks like a new speaker tag ("NAME:")
  TERM_NAMES: tuple[str, ...]            names of the reward terms, each scored in [0, 1] with 1 = good
  DEFAULT_WEIGHTS: dict[str, float]      one non-negative weight per term name
  reward_terms(user_line, reply, ended, known_words) -> dict[str, float]
        ended is "turn", "doc" or "truncated"; known_words is a set of valid words (plays + story vocabulary)
  combine(terms, weights=DEFAULT_WEIGHTS) -> float          sum of weight * term; KeyError for an unknown weight name

Suggested terms: stopped, length_ok, no_leak, relevance, not_copy, novelty, known_words, no_repeat.
  stopped      1 if ended in ("turn", "doc") else 0 (both stop tokens are valid stops)
  length_ok    1 for 4..40 words, ramping down to 0 below 4 words and above ~60 words
  no_leak      0 if has_leak(reply) else 1
  relevance    min(shared content words with the user's line, cap) / cap, cap = min(2, content words in the user's line); 0 if cap == 0
  not_copy     falls from 1 to 0 as the share of the reply's content words copied from the user's line goes from 0.6 to 1
  novelty      min(content words not in the user's line, 3) / 3
  known_words  fraction of the reply's words found in known_words (0 for an empty reply)
  no_repeat    1 - share of repeated bigrams inside the reply

Run from the repo root:  pytest -q tests/test_reward.py
"""
import pytest

from gpt.reward import DEFAULT_WEIGHTS, TERM_NAMES, combine, content, has_leak, reward_terms, words

USER = "Hey, I think I left my keys in your garden."
GOOD = "What keys, sir? Search the garden, and I will help thee find them."
PARROT = "Hey, I think I left my keys in your garden."
RAMBLE = " ".join(["the king is dead and the queen is sad"] * 9)             # 81 words, many repeated bigrams
LEAKED = "No, I did not.\n\nJULIET:\nWhat keys?"
NONWORDS = "Wrelf aeterratulations holyountain bselves capoberts findeer!"
ONE_SHARED = "Keys are a heavy burden, sir, for the poor."
TWO_SHARED = "Your keys lie hidden somewhere in the garden, sir."
THREE_SHARED = "Your keys, your garden, and the left gate, sir, I know."

KNOWN = set(words(" ".join([USER, GOOD, RAMBLE, LEAKED, ONE_SHARED, TWO_SHARED, THREE_SHARED,
                            "what is it sir ay no yes my lord good night stay come go with me"])))


def score(reply, ended="turn", user=USER, known=KNOWN):
    return combine(reward_terms(user, reply, ended, known))


def test_helpers():
    assert words("Thou 'tis, SIR!") == ["thou", "tis", "sir"]
    assert "keys" in content(USER) and "the" not in content(GOOD) and "i" not in content(USER)
    assert has_leak(LEAKED) and not has_leak(GOOD)


def test_all_terms_present_and_in_range():
    for reply in (GOOD, PARROT, RAMBLE, LEAKED, NONWORDS, "", "Ay."):
        for ended in ("turn", "doc", "truncated"):
            terms = reward_terms(USER, reply, ended, KNOWN)
            assert set(terms) == set(TERM_NAMES)
            assert all(0.0 <= v <= 1.0 for v in terms.values()), (reply, ended, terms)


def test_default_weights_cover_the_terms():
    assert set(DEFAULT_WEIGHTS) == set(TERM_NAMES) and all(w >= 0 for w in DEFAULT_WEIGHTS.values())


def test_good_reply_beats_the_failure_modes():
    good = score(GOOD)
    for name, bad in {"parrot": PARROT, "ramble": RAMBLE, "leak": LEAKED, "nonwords": NONWORDS, "empty": ""}.items():
        assert good > score(bad), name


def test_stopping_terms():
    assert score(GOOD, "turn") == score(GOOD, "doc")                         # both stop tokens count as proper stops
    assert score(GOOD, "turn") > score(GOOD, "truncated")
    assert reward_terms(USER, GOOD, "doc", KNOWN)["stopped"] == 1.0
    assert reward_terms(USER, GOOD, "truncated", KNOWN)["stopped"] == 0.0


def test_leak_term_in_isolation():
    assert reward_terms(USER, LEAKED, "turn", KNOWN)["no_leak"] == 0.0
    assert reward_terms(USER, GOOD, "turn", KNOWN)["no_leak"] == 1.0


def test_relevance_is_monotone_and_capped():
    r = [reward_terms(USER, t, "turn", KNOWN)["relevance"] for t in (NONWORDS, ONE_SHARED, TWO_SHARED, THREE_SHARED)]
    assert r[0] == 0.0 and r[0] < r[1] < r[2]
    assert r[2] == r[3] == 1.0                                              # sharing more than the cap adds nothing


def test_copying_is_penalised_but_novelty_is_rewarded():
    parrot, good = reward_terms(USER, PARROT, "turn", KNOWN), reward_terms(USER, GOOD, "turn", KNOWN)
    assert parrot["not_copy"] == 0.0 and good["not_copy"] == 1.0
    assert parrot["novelty"] == 0.0 and good["novelty"] == 1.0


def test_length_and_vocabulary_terms():
    assert reward_terms(USER, "", "turn", KNOWN)["length_ok"] == 0.0
    assert reward_terms(USER, GOOD, "turn", KNOWN)["length_ok"] == 1.0
    assert reward_terms(USER, RAMBLE, "turn", KNOWN)["length_ok"] == 0.0
    assert reward_terms(USER, NONWORDS, "turn", KNOWN)["known_words"] == 0.0
    assert reward_terms(USER, GOOD, "turn", KNOWN)["known_words"] == 1.0
    assert reward_terms(USER, RAMBLE, "turn", KNOWN)["no_repeat"] < 0.5


def test_edge_cases_do_not_crash():
    assert reward_terms("Ay.", GOOD, "turn", KNOWN)["relevance"] == 0.0      # no content words in the user's line
    assert reward_terms("", GOOD, "turn", KNOWN)["relevance"] == 0.0
    assert reward_terms(USER, "", "truncated", set())["known_words"] == 0.0


def test_combine():
    terms = reward_terms(USER, GOOD, "turn", KNOWN)
    assert combine(terms, {k: 0.0 for k in TERM_NAMES}) == 0.0
    assert combine(terms, {"stopped": 1.0}) == terms["stopped"]
    assert combine(terms, DEFAULT_WEIGHTS) == pytest.approx(sum(DEFAULT_WEIGHTS[k] * terms[k] for k in TERM_NAMES))
    with pytest.raises(KeyError):
        combine(terms, {"not_a_term": 1.0})


def test_deterministic():
    assert reward_terms(USER, GOOD, "turn", KNOWN) == reward_terms(USER, GOOD, "turn", KNOWN)