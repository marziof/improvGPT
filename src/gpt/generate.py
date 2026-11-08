import torch

from gpt.model import GPT
from gpt.config import Config

import argparse
from pathlib import Path
from tokenizers import Tokenizer
from gpt.utils import check_tokenizer, checkpoint_tokenizer_hash     # add to the imports at the top of generate.py

from dataclasses import fields

def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    saved = ckpt["cfg"]
    valid = {f.name for f in fields(Config)}
    unknown = sorted(set(saved) - valid)
    if unknown:
        print(f"note: ignoring saved config keys this Config doesn't have: {unknown}")
    cfg = Config(**{k: v for k, v in saved.items() if k in valid})
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg

# def load_model(ckpt_path, device="cpu"):
#     ckpt = torch.load(ckpt_path, map_location=device)
#     cfg = Config(**ckpt["cfg"])
#     model = GPT(cfg).to(device)
#     model.load_state_dict(ckpt["model"])
#     return model.eval(), cfg



def select_next_token(logits, temperature=1.0, k=None, generator=None):
    # logits: (V,)  - one row, over whole vocabulary of tokens
    assert temperature>0, "temperature must be positive"
    if k is not None:
        assert isinstance(k, int), "k must be an integer"
        k = min(k, logits.size(-1))
    else:
        k = logits.size(-1)
    top_values, top_indices = torch.topk(logits, k=k, dim=-1)
    probas = torch.softmax(top_values/temperature, dim=-1)
    sampled_index = torch.multinomial(probas, num_samples=1, generator=generator)
    selected_index = top_indices.gather(-1, sampled_index)
    return selected_index


@torch.no_grad()
def generate(model, context_len, idx, max_new_tokens, temperature, k, generator, eot_id):
    """
    Generates token sequence from a prompt through iterated sampling
    Args:
    - model: pretrained model; context_len: max window of tokens the model can see
    - idx: prompt as token ids, shape (1, T0), dtype long, on the model's device
    - max_new_tokens: max amount of tokens to generate, ending with eot_id
    - temperature, k, generator: sampling params
    Returns:
    - output: full sequence, including prompt, shape (1, T0 + n_new)
    """
    output = idx.clone()
    for _ in range(max_new_tokens):
        context = output[:, -context_len:] # shape (1, context_len)
        logits, _ = model(context, targets=None) # (1, T_window, V)
        next_tok = select_next_token(logits[0,-1], temperature, k, generator).view(1, 1)
        output = torch.cat([output, next_tok], dim=-1)
        if next_tok.item() == eot_id:
            break
    return output 



@torch.no_grad()
def generate_cached(model, context_len, idx, max_new_tokens, temperature, k, generator, eot_id):
    B, T0 = idx.shape
    assert B == 1, "batched generation comes later"
    assert T0 + max_new_tokens <= context_len, "the cache holds context_len positions; shorten the prompt or max_new_tokens"
    cache = model.make_cache(1)
    output = idx.clone()
    logits, _ = model(idx, start_pos=0, cache=cache)         # prefill: fills positions 0 .. T0-1
    pos = T0                                                 # the next token goes at position T0
    for _ in range(max_new_tokens):
        next_tok = select_next_token(logits[0, -1], temperature, k, generator)    # (1,)
        output = torch.cat([output, next_tok.view(1, 1)], dim=-1)
        if next_tok.item() == eot_id:
            break
        logits, _ = model(next_tok.view(1, 1), start_pos=pos, cache=cache)        # decode: one token at position pos
        pos += 1
    return output


@torch.no_grad()
def generate_batch(model, context_len, idx, max_new_tokens, temperature, k, generator, stop_ids, pad_id):
    # idx: (B, T0), all prompts the same length. Returns (output, finished, n_new).
    B, T0 = idx.shape
    assert T0 + max_new_tokens <= context_len, "the cache holds context_len positions"
    stop = torch.as_tensor(stop_ids, device=idx.device)                      # (S,) long
    cache = model.make_cache(B)                                              # change 1: B rows instead of 1
    output = idx.clone()
    logits, _ = model(idx, start_pos=0, cache=cache)                         # prefill, (B, T0, V)
    pos = T0
    finished = torch.zeros(B, dtype=torch.bool, device=idx.device)           # change 2: one flag per row
    n_new = torch.zeros(B, dtype=torch.long, device=idx.device)
    for _ in range(max_new_tokens):
        tok = select_next_token(logits[:, -1], temperature, k, generator)    # change 3: (B, V) in, (B, 1) out
        tok = torch.where(finished[:, None], torch.full_like(tok, pad_id), tok)    # rows that already stopped emit padding
        n_new += (~finished).long()                                          # count tokens from rows that were active, stop token included
        finished = finished | torch.isin(tok[:, 0], stop)                    # change 4: a set of stop ids, checked per row
        output = torch.cat([output, tok], dim=-1)
        if finished.all():                                                   # change 5: stop only when every row is done
            break
        logits, _ = model(tok, start_pos=pos, cache=cache)
        pos += 1
    return output, finished, n_new


def main(argv=None):
    p = argparse.ArgumentParser(description="Sample text from a checkpoint")
    p.add_argument("--run_dir", required=True)                 # folder with the checkpoint (and the tokenizer)
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--tokenizer", default=None)                # defaults to <run_dir>/tokenizer.json
    p.add_argument("--prompt", default="")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--no_cache", action="store_true")
    args = p.parse_args(argv)

    run = Path(args.run_dir)
    #model, cfg = load_model(run / args.ckpt)
    #tok = Tokenizer.from_file(args.tokenizer or str(run / "tokenizer.json"))
    # inside main(), replacing the two lines that load the model and the tokenizer:
    ckpt_path = run / args.ckpt
    model, cfg = load_model(ckpt_path)
    tok_path = args.tokenizer or str(run / "tokenizer.json")
    expected = cfg.tokenizer_sha256 or checkpoint_tokenizer_hash(ckpt_path)
    check_tokenizer(tok_path, expected)                                   # raises on a mismatch, warns if no hash is recorded
    tok = Tokenizer.from_file(tok_path)
    eot_id = tok.token_to_id("<|endoftext|>")
    idx = torch.tensor([[eot_id] + tok.encode(args.prompt).ids])     # every training document starts after an eot token
    g = torch.Generator().manual_seed(args.seed)
    k = 1 if args.greedy else args.top_k

    if args.no_cache:
        fn, max_new = generate, args.max_new_tokens
    else:
        fn, max_new = generate_cached, min(args.max_new_tokens, cfg.context_len - idx.shape[1])   # the cache can't slide past the window
    out = fn(model, cfg.context_len, idx, max_new, args.temperature, k, g, eot_id)
    print(tok.decode(out[0, 1:].tolist()))


if __name__ == "__main__":
    main()