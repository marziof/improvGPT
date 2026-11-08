import re


BLOCK = 50          # turns per block
VAL_EVERY = 10      # blocks with (b + 1) % 10 == 0 go to validation


def split_of_block(b):
    return "val" if (b + 1) % VAL_EVERY == 0 else "train"


def parse_turns(text):
    return [t.strip() for t in re.split(r"\n\s*\n", text) if t.strip()]