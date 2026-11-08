import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from gpt.improv_data import (build_improv_examples, get_example_batch,
                             get_val_example_batches, split_turn, to_row)
from gpt.splits import BLOCK, parse_turns, split_of_block
from gpt.utils import file_sha256

def load_turns(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Adapt this to your existing turn parser if you already have one.
    turns = parse_turns(text)

    return turns


def split_examples(examples, block_size=50, val_every=10):
    train, val = [], []

    for example in examples:
        block = example["target_index"] // block_size

        if block % val_every == 0:
            val.append(example)
        else:
            train.append(example)

    return train, val


def save_dataset(
    train,
    val,
    output_dir,
    tokenizer_path,
    vocab_size,
    eot_id,
    eoturn_id,
    row_len=256,
    meta_extra=None
):
    meta_extra = meta_extra or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split, examples in [("train", train), ("val", val)]:
        tokens = np.stack([
            to_row(ex["ids"], row_len, eot_id)
            for ex in examples
        ])

        lens = np.asarray([
            [ex["prompt_len"], ex["total_len"]]
            for ex in examples
        ], dtype=np.int32)

        assert np.all(lens[:, 1] - lens[:, 0] >= 2)           # at least one reply token plus the end-of-turn

        np.save(output_dir / f"{split}_tokens.npy", tokens)
        np.save(output_dir / f"{split}_lens.npy", lens)

    shutil.copy(tokenizer_path, output_dir / "tokenizer.json")

    meta = {
       "vocab_size": vocab_size,
       "eot_id": eot_id,
       "eoturn_id": eoturn_id,
       "pad_id": eot_id,
       "row_len": row_len,
       "n_train": len(train),
       "n_val": len(val),
       "tokenizer_sha256": file_sha256(tokenizer_path),
       **meta_extra,
   }

    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def save_jsonl(examples, turns, path):
    with open(path, "w", encoding="utf-8") as f:
        for e in examples:
            context = [turns[i] for i in e["context_indices"]]
            user_line = next(split_turn(t)[1] for t in reversed(context)
                             if split_turn(t)[0] == e["user_speaker"])    # the filter guarantees the user appears in the context
            rec = {
                "context_turns": context,
                "target_turn": turns[e["target_index"]],
                "user_speaker": e["user_speaker"],
                "user_line": user_line,
                "context_indices": e["context_indices"],
                "target_index": e["target_index"],
                "block_index": e["block_index"],
                "episode_index": e["episode_index"],
                "prompt_len": e["prompt_len"],
                "total_len": e["total_len"],
            }
            f.write(json.dumps(rec) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description="Build the improv (stage 2) dataset")
    p.add_argument("--raw", default="data_sft_blocks/raw/input.txt")
    p.add_argument("--tokenizer", default="data_sft_blocks/tokenizer.json")
    p.add_argument("--shake_meta", default="data_sft_blocks/meta.json")
    p.add_argument("--out_dir", default="data_improv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_reply_tokens", type=int, default=60)
    p.add_argument("--role_markers", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(argv)

    out = Path(args.out_dir)
    if out.exists() and any(out.glob("*.npy")) and not args.overwrite:
        raise SystemExit(f"{out} already holds a dataset; pass --overwrite to redo.")

    tok = Tokenizer.from_file(args.tokenizer)
    eot_id, eoturn_id = tok.token_to_id("<|endoftext|>"), tok.token_to_id("<|endofturn|>")
    assert eot_id is not None and eoturn_id is not None
    shake = json.loads(Path(args.shake_meta).read_text())
    assert tok.get_vocab_size() == shake["vocab_size"]
    if shake.get("tokenizer_sha256"):
        assert file_sha256(args.tokenizer) == shake["tokenizer_sha256"], "tokenizer differs from the Shakespeare SFT data's"

    turns = parse_turns(Path(args.raw).read_text(encoding="utf-8"))
    assert len(turns) == shake["turns"], "turn count differs from the Shakespeare SFT data"

    train, val, stats = build_improv_examples(
        turns, tok, eot_id=eot_id, eoturn_id=eoturn_id, seed=args.seed,
        max_reply_tokens=args.max_reply_tokens, role_markers=args.role_markers)
    assert train and val, "empty split"
    assert all(split_of_block(e["block_index"]) == "val" for e in val)
    assert all(split_of_block(e["block_index"]) == "train" for e in train)
    assert {e["block_index"] for e in val} <= set(shake["val_blocks"])

    extra = {"role_markers": args.role_markers, "turns": len(turns), "val_blocks": shake["val_blocks"],
             "block_size": BLOCK, "seed": args.seed, "max_reply_tokens": args.max_reply_tokens, "stats": stats}
    save_dataset(train, val, out, args.tokenizer, tok.get_vocab_size(), eot_id, eoturn_id, meta_extra=extra)
    save_jsonl(train, turns, out / "train_examples.jsonl")
    save_jsonl(val, turns, out / "val_examples.jsonl")

    # self-check: reload through the real loaders
    x, y = get_example_batch(out, "train", 8, torch.Generator().manual_seed(0), "cpu")
    labels = y[0][y[0] != -100]
    assert labels[-1].item() == eoturn_id, "the last label must be the end-of-turn token"
    batches = get_val_example_batches(out, 16, "cpu")
    assert sum(len(b[0]) for b in batches) == len(val)
    print("sample reply (decoded labels):", repr(tok.decode(labels.tolist())))

    print(json.dumps(stats, indent=1))
    for name, exs in (("train", train), ("val", val)):
        reply = [e["total_len"] - e["prompt_len"] - 1 for e in exs]
        print(f"{name}: {len(exs)} examples | mean reply tokens {sum(reply) / len(reply):.1f} | "
              f"{sum(reply):,} loss tokens | mean total length {sum(e['total_len'] for e in exs) / len(exs):.0f}")


if __name__ == "__main__":
    main()

# def is_speaker_turn(turn):
#     tag, body = split_turn(turn)
#     return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 .,'-]*:", tag))


# def get_speakers(turns):
#     return sorted({
#         split_turn(turn)[0]
#         for turn in turns
#         if is_speaker_turn(turn)
#     })


# def format_turn(turn, user_speaker):
#     tag, body = split_turn(turn)

#     role = "USER" if tag == user_speaker else "ASSISTANT"

#     return f"[{role}] {tag}\n{body}"


# def filter_example(
#     context_turns,
#     target_turn,
#     user_speaker,
#     tok,
#     max_reply_tokens=40,
#     max_total_tokens=256,
# ):
#     target_tag, target_body = split_turn(target_turn)

#     # Target must be an assistant turn.
#     if target_tag == user_speaker:
#         return False

#     if not target_body.strip():
#         return False

#     if not any(split_turn(t)[0] == user_speaker for t in context_turns):
#         return False

#     if len(tok.encode(target_body).ids) > max_reply_tokens:
#         return False

#     formatted_context = [
#         format_turn(t, user_speaker) for t in context_turns
#     ]
#     formatted_target = format_turn(target_turn, user_speaker)

#     ids, _ = build_example(
#         formatted_context,
#         formatted_target,
#         tok,
#         eot_id=0,
#         eoturn_id=1,
#     )

#     if len(ids) > max_total_tokens:
#         return False

#     return True


# import random

# def build_improv_examples(
#     turns,
#     tok,
#     eot_id,
#     eoturn_id,
#     min_context=1,
#     max_context=3,
#     max_reply_tokens=40,
#     max_total_tokens=256,
#     seed=42,
# ):
#     rng = random.Random(seed)
#     speakers = get_speakers(turns)

#     examples = []

#     for i in range(1, len(turns)):
#         target_turn = turns[i]
#         target_tag, _ = split_turn(target_turn)

#         # Need a target character that can be assigned to the assistant.
#         possible_users = [s for s in speakers if s != target_tag]

#         if not possible_users:
#             continue

#         user_speaker = rng.choice(possible_users)

#         n_context = rng.randint(
#             min_context,
#             min(max_context, i)
#         )

#         context_turns = turns[i - n_context:i]

#         if not filter_example(
#             context_turns,
#             target_turn,
#             user_speaker,
#             tok,
#             max_reply_tokens,
#             max_total_tokens,
#         ):
#             continue

#         formatted_context = [
#             format_turn(t, user_speaker) for t in context_turns
#         ]
#         formatted_target = format_turn(target_turn, user_speaker)

#         ids, prompt_len = build_example(
#             formatted_context,
#             formatted_target,
#             tok,
#             eot_id,
#             eoturn_id,
#         )

#         examples.append({
#             "ids": ids,
#             "prompt_len": prompt_len,
#             "total_len": len(ids),
#             "target_index": i,
#             "user_speaker": user_speaker,
#         })

#     return examples
