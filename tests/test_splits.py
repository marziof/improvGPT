import inspect
from gpt.splits import BLOCK, VAL_EVERY, split_of_block
from gpt.improv_data import build_improv_examples

def test_val_blocks():
    assert [b for b in range(30) if split_of_block(b) == "val"] == [9, 19, 29]

def test_improv_uses_the_shared_rule():
    p = inspect.signature(build_improv_examples).parameters
    assert p["block_size"].default == BLOCK and p["val_every"].default == VAL_EVERY