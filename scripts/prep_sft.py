
import re
    
import numpy as np
from pathlib import Path
import argparse
import json
import shutil
import urllib.request

from tokenizers import Tokenizer
from gpt.splits import BLOCK, VAL_EVERY, split_of_block, parse_turns
from gpt.utils import file_sha256

def pack_chunks(turns, tok, max_tokens, sep="\n\n"):
    # greedy packing; returns a list of chunk strings (turns joined by sep), never splitting a turn
    sep_len = len(tok.encode(sep).ids)
    chunks, current, total = [], [], 0
    n_oversized = 0
    for turn in turns:
        n = len(tok.encode(turn).ids)
        if n > max_tokens - 1:
            n_oversized += 1                     # one turn alone exceeds the budget; it gets its own chunk
        cost = n + (sep_len if current else 0)
        if current and total + cost > max_tokens - 1:    # -1 leaves room for the eot id in front
            chunks.append(sep.join(current))
            current, total, cost = [], 0, n
        current.append(turn)
        total += cost
    if current:
        chunks.append(sep.join(current))
    return chunks, n_oversized


def pack_blocks(turns, tok, max_tokens):
    # pack inside each BLOCK-turn block, so no chunk crosses a block boundary; then split by block
    train_chunks, val_chunks, n_oversized = [], [], 0
    n_blocks = (len(turns) + BLOCK - 1) // BLOCK
    for b in range(n_blocks):
        block = turns[b * BLOCK:(b + 1) * BLOCK]
        chunks, n_over = pack_chunks(block, tok, max_tokens)
        assert "\n\n".join(chunks) == "\n\n".join(block), f"turns lost or duplicated in block {b}"
        n_oversized += n_over
        (val_chunks if split_of_block(b) == "val" else train_chunks).extend(chunks)
    return train_chunks, val_chunks, n_blocks, n_oversized
        

# def split_chunks(chunks, val_every=10, mode="interleaved", block_size=10):
#     # returns (train_chunks, val_chunks), each keeping the original order
#     assert val_every >= 2
#     n = len(chunks)
#     if mode == "interleaved":
#         # whole blocks of consecutive chunks go to val, one block in every val_every
#         val_idx = {i for i in range(n) if (i // block_size) % val_every == val_every - 1}
#     elif mode == "contiguous":
#         n_val = max(1, n // val_every)
#         val_idx = set(range(n - n_val, n))
#     else:
#         raise ValueError(f"unknown mode {mode!r}")
#     train = [c for i, c in enumerate(chunks) if i not in val_idx]
#     val = [c for i, c in enumerate(chunks) if i in val_idx]
#     assert train and val, "empty split: too few chunks for these settings"
#     assert len(train) + len(val) == n
#     return train, val


def write_bin(chunks, tok, eot_id, path):
    # encodes each chunk as a whole, prepends the eot id, writes raw uint16 (same format as prepare_data.py)
    vocab = tok.get_vocab_size()
    assert vocab < 2**16, "vocab must fit in uint16"
    ids = []
    for chunk in chunks:
        ids.append(eot_id)
        ids.extend(tok.encode(chunk).ids)
    assert max(ids) < vocab, "token id out of range for this tokenizer"
    arr = np.asarray(ids, dtype=np.uint16)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(path)
    return {"chunks": len(chunks), "tokens": int(arr.size), "chars": sum(len(c) for c in chunks)}



URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download(url, path):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    urllib.request.urlretrieve(url, path)


def main(argv=None):
    p = argparse.ArgumentParser(description="Prepare Tiny Shakespeare for finetuning")
    p.add_argument("--tokenizer", default="runs/pretrain_v2/tokenizer.json")
    p.add_argument("--out_dir", default="data_sft_blocks")
    p.add_argument("--max_tokens", type=int, default=224)
    # p.add_argument("--val_every", type=int, default=10)
    # p.add_argument("--block_size", type=int, default=10)
    # p.add_argument("--split_mode", choices=["interleaved", "contiguous"], default="interleaved")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    train_bin, val_bin = out / "train.bin", out / "val.bin"
    if not args.overwrite and (train_bin.exists() or val_bin.exists()):
        raise SystemExit(f"{train_bin} or {val_bin} exists; pass --overwrite to redo.")
    out.mkdir(parents=True, exist_ok=True)

    # 1) raw text
    raw = out / "raw" / "input.txt"
    download(URL, raw)
    text = raw.read_text(encoding="utf-8")

    # 2) the pretrained tokenizer (never trained here)
    tok = Tokenizer.from_file(args.tokenizer)
    eot_id = tok.token_to_id("<|endoftext|>")
    assert eot_id is not None, "end-of-text token missing from tokenizer"
    meta_src = Path(args.tokenizer).with_name("meta.json")
    if meta_src.exists():       # extra guard: the tokenizer must match the pretraining run's metadata
        assert json.loads(meta_src.read_text())["vocab_size"] == tok.get_vocab_size(), \
            "tokenizer vocab size does not match the pretraining run's meta.json"
    src = json.loads(meta_src.read_text())
    if src.get("tokenizer_sha256"):
        assert src["tokenizer_sha256"] == file_sha256(args.tokenizer), "tokenizer.json does not match its own meta.json"

    # 3) turns -> chunks -> split
    turns = parse_turns(text)
    train_chunks, val_chunks, n_blocks, n_oversized = pack_blocks(turns, tok, args.max_tokens)
    assert train_chunks and val_chunks, "empty split"
    # turns = parse_turns(text)
    # chunks = pack_chunks(turns, tok, args.max_tokens)
    # assert "\n\n".join(chunks) == "\n\n".join(turns), "turns lost or duplicated while packing"
    # train_chunks, val_chunks = split_chunks(chunks, args.val_every, args.split_mode, args.block_size)

    # 4) write the .bin files
    train_info = write_bin(train_chunks, tok, eot_id, train_bin)
    val_info = write_bin(val_chunks, tok, eot_id, val_bin)

    # 5) metadata (load_config reads vocab_size from here) and a copy of the tokenizer
    meta = {
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot_id,
        "source": URL,
        "max_tokens": args.max_tokens,
        "split": {"rule": "block", "block_size": BLOCK, "val_every": VAL_EVERY},
        "turns": len(turns),
        "n_blocks": n_blocks,
        "val_blocks": [b for b in range(n_blocks) if split_of_block(b) == "val"],
        "oversized_turns": n_oversized,
        "train": train_info,
        "val": val_info,
        "tokenizer_sha256": file_sha256(args.tokenizer)
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    shutil.copy(args.tokenizer, out / "tokenizer.json")

    # 6) stats
    print(f"{len(turns)} turns in {n_blocks} blocks ({len(meta['val_blocks'])} val) -> "
          f"{len(train_chunks) + len(val_chunks)} chunks; {n_oversized} oversized turn(s)")
    for name, s in (("train", train_info), ("val", val_info)):
        print(f"{name}: {s['chunks']} chunks, {s['tokens']:,} tokens, "
              f"{s['tokens'] / s['chunks']:.1f} tokens/chunk, {s['chars'] / s['tokens']:.2f} chars/token")
        if s["tokens"] < 10_000:
            print(f"warning: {name} has under 10k tokens, so its loss estimate will be noisy")



if __name__ == "__main__":
    main()

    