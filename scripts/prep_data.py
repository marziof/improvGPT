"""Prepare TinyStories: train a byte-level BPE, tokenize to flat uint16 .bin files.
 
Usage:
    python scripts/prepare_data.py --streaming --num_stories 10000      # local dev
    python scripts/prepare_data.py --num_stories 500000                 # Kaggle / real run
 
Outputs (in --out_dir): tokenizer.json, train.bin, val.bin, meta.json
Read the .bin files later with: np.memmap(path, dtype=np.uint16, mode="r")
"""
import argparse
import json
from itertools import islice
from pathlib import Path


import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from gpt.utils import file_sha256

 
DATASET = "roneneldan/TinyStories"
# Reserved now so the vocab never has to change when you add the Star Wars stage.
BASE_SPECIAL_TOKENS = ["<|endoftext|>", "<|endofturn|>", "<|scene|>"]



def iter_stories(split, n, streaming, seed):
    """
    Load up to n shuffled, stripped, non-empty stories from a train/validation split.
    Args: 
    - split: train or validation
    - n: max number of stories to load
    - streaming: load one at a time or all at once
    - seed: reproducibility of split
    """
    from datasets import load_dataset

    if streaming:
        ds = load_dataset(DATASET, split=split, streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000).take(n)
    else:
        ds = load_dataset(DATASET, split=split)
        ds = ds.shuffle(seed=seed).select(range(min(n, len(ds))))

    for entry in ds:
        story = entry["text"].strip()
        if story:
            yield story


def train_tokenizer(texts, vocab_size, special_tokens):
    """
    Train a BPE tokenizer on data texts
    Args:
    - texts: data to tokenize
    - vocab_size: desired nb of tokens
    - special_tokens: EOS, etc
    """
    tok = Tokenizer(models.BPE()) # tokenizer (pipeline that receives a model)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False) # initial split of text into tokens
    tok.decoder = decoders.ByteLevel() # how to turn tokens back to text
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )   # trainer with target nb of tokens to produce, special tokens, initial alphabet
    tok.train_from_iterator(texts, trainer=trainer) # train tokenizer on text
    return tok 



def tokenize_to_bin(stories, tok, eot_id, path, batch_size=1000):
    """
    Tokenize stories in batches, append eot after each, write raw uint16 to path.
    Args:
    - stories: raw text
    - tok: trained tokenizer
    - eot_id: token id for end of sequences, to append to end of each story
    - path: save path
    - batch_size: number of stories per batch
    """
    n_stories = n_tokens = n_chars = 0
 
    def flush(batch, f):
        # encode batch efficiently
        nonlocal n_tokens
        ids = []
        for enc in tok.encode_batch(batch):
            # add eot token at end of stories
            ids.extend(enc.ids)
            ids.append(eot_id)
        arr = np.asarray(ids, dtype=np.uint16)
        f.write(arr.tobytes())
        n_tokens += len(arr)
 
    batch = []
    # write token encodings of stories to path
    with open(path, "wb") as f:
        for text in stories:
            batch.append(text)
            n_stories += 1
            n_chars += len(text)
            if len(batch) == batch_size:
                flush(batch, f)
                batch = []
        if batch:
            flush(batch, f)
    return {"stories": n_stories, "tokens": n_tokens, "chars": n_chars}



def main(argv=None):
    # parser args for data tokenization
    p = argparse.ArgumentParser(description="Prepare TinyStories data")
    p.add_argument("--num_stories", type=int, default=10_000)
    p.add_argument("--num_val_stories", type=int, default=2_000)
    p.add_argument("--vocab_size", type=int, default=4096)
    p.add_argument("--tokenizer_stories", type=int, default=100_000,
                   help="max stories used to train the BPE")
    p.add_argument("--extra_special_tokens", nargs="*", default=[],
                   help="e.g. LUKE: VADER: (added after the base special tokens)")
    p.add_argument("--out_dir", type=str, default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--streaming", action="store_true",
                   help="stream instead of downloading the full dataset")
    p.add_argument("--overwrite", action="store_true",
                   help="retrain tokenizer and rewrite .bin files")
    args = p.parse_args(argv)
 
    assert args.vocab_size < 2**16, "vocab must fit in uint16"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok_path = out / "tokenizer.json"
    train_bin, val_bin = out / "train.bin", out / "val.bin"
 
    if not args.overwrite and (train_bin.exists() or val_bin.exists()):
        raise SystemExit(f"{train_bin} or {val_bin} exists; pass --overwrite to redo.")
 
    # 1) tokenizer: train once, then reuse forever (never retrain between runs)
    if tok_path.exists() and not args.overwrite:
        print(f"Loading existing tokenizer from {tok_path}")
        tok = Tokenizer.from_file(str(tok_path))
    else:
        specials = BASE_SPECIAL_TOKENS + args.extra_special_tokens
        n_tok = min(args.tokenizer_stories, args.num_stories)
        print(f"Training BPE (vocab {args.vocab_size}) on {n_tok} stories...")
        texts = islice(iter_stories("train", n_tok, args.streaming, args.seed), n_tok)
        tok = train_tokenizer(texts, args.vocab_size, specials)
        tok.save(str(tok_path))
 
    eot_id = tok.token_to_id(BASE_SPECIAL_TOKENS[0])
    assert eot_id is not None, "end-of-text token missing from tokenizer"
 
    # 2) tokenize both splits
    print("Tokenizing train...")
    train_stats = tokenize_to_bin(
        iter_stories("train", args.num_stories, args.streaming, args.seed),
        tok, eot_id, train_bin)
    print("Tokenizing validation...")
    val_stats = tokenize_to_bin(
        iter_stories("validation", args.num_val_stories, args.streaming, args.seed),
        tok, eot_id, val_bin)
 
    # 3) stats + metadata (chars/token for bits-per-char later)
    meta = {
        "vocab_size": tok.get_vocab_size(),
        "eot_id": eot_id,
        "special_tokens": BASE_SPECIAL_TOKENS + args.extra_special_tokens,
        "train": train_stats,
        "val": val_stats,
        "tokenizer_sha256": file_sha256(tok_path)
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
 
    for name, s in (("train", train_stats), ("val", val_stats)):
        print(f"{name}: {s['stories']:,} stories, {s['tokens']:,} tokens, "
              f"{s['tokens'] / max(s['stories'], 1):.1f} tokens/story, "
              f"{s['chars'] / max(s['tokens'], 1):.2f} chars/token")
 
 
if __name__ == "__main__":
    main()
 