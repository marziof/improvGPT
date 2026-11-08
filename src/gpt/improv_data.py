import random
import re
from collections import Counter

import numpy as np
import torch

from gpt.splits import BLOCK, VAL_EVERY      # one shared split rule (same values prepare_sft.py uses)


def split_turn(turn):
    lines = turn.split("\n")
    tag = lines[0]
    body = "\n".join(lines[1:])
    return tag, body


def build_example(context_turns, target_turn, tok, eot_id, eoturn_id):
    tag, body = split_turn(target_turn)

    context = "\n\n".join(context_turns)
    prompt = context + "\n\n" + tag + "\n"

    prompt_ids = tok.encode(prompt).ids
    reply_ids = tok.encode(body).ids

    ids = [eot_id] + prompt_ids + reply_ids + [eoturn_id]
    prompt_len = 1 + len(prompt_ids)

    return ids, prompt_len


def to_row(ids, row_len, pad_id):
    if len(ids) > row_len:
        raise ValueError(f"Example length {len(ids)} exceeds row length {row_len}")

    row = np.full(row_len, pad_id, dtype=np.uint16)
    row[:len(ids)] = ids

    return row


def batch_from_rows(rows, prompt_lens, total_lens):
    x = torch.as_tensor(rows[:, :-1].astype(np.int64))
    y = torch.as_tensor(rows[:, 1:].astype(np.int64))

    B, T = y.shape

    # y[j] is the token at row index j+1, so a reply token at row index t is the target at position t-1.
    # Real targets: prompt_len <= t < total_len  <=>  prompt_len-1 <= j < total_len-1
    positions = torch.arange(T).unsqueeze(0)
    prompt_lens = torch.as_tensor(np.array(prompt_lens, dtype=np.int64)).unsqueeze(1)    # np.array copies: memmaps are read-only
    total_lens = torch.as_tensor(np.array(total_lens, dtype=np.int64)).unsqueeze(1)

    mask = (positions >= prompt_lens - 1) & (positions < total_lens - 1)
    y[~mask] = -100

    assert (y != -100).any(dim=1).all(), "a row has no reply tokens, so its loss would be empty"
    return x, y


def get_example_batch(data_dir, split, B, generator, device):
    rows = np.load(
        f"{data_dir}/{split}_tokens.npy",
        mmap_mode="r"
    )
    lens = np.load(
        f"{data_dir}/{split}_lens.npy",
        mmap_mode="r"
    )

    indices = torch.randint(
        len(rows),
        (B,),
        generator=generator
    ).numpy()

    batch_rows = rows[indices]
    batch_lens = lens[indices]

    x, y = batch_from_rows(
        batch_rows,
        batch_lens[:, 0],
        batch_lens[:, 1]
    )

    return x.to(device), y.to(device)


def get_val_example_batches(data_dir, B, device, split="val"):
    # every row of the split exactly once, in order: an exact, fixed validation set (the split is small)
    rows = np.load(f"{data_dir}/{split}_tokens.npy", mmap_mode="r")
    lens = np.load(f"{data_dir}/{split}_lens.npy", mmap_mode="r")
    batches = []
    for start in range(0, len(rows), B):
        x, y = batch_from_rows(
            rows[start:start + B],
            lens[start:start + B, 0],
            lens[start:start + B, 1],
        )
        batches.append((x.to(device), y.to(device)))
    return batches


def is_speaker_turn(turn):
    tag, _ = split_turn(turn)
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z0-9 .,'&-]*:", tag))


def get_speakers(turns):
    return sorted({
        split_turn(turn)[0]
        for turn in turns
        if is_speaker_turn(turn)
    })


def format_turn(turn, user_speaker, role_markers=False):
    if not role_markers:
        return turn
    tag, body = split_turn(turn)
    role = "USER" if tag == user_speaker else "ASSISTANT"
    return f"[{role}] {tag}\n{body}"


def filter_example(
    context_turns,
    target_turn,
    user_speaker,
    tok,
    eot_id,
    eoturn_id,
    max_reply_tokens=60,
    max_total_tokens=256,
    role_markers=False
):
    target_tag, target_body = split_turn(target_turn)

    if target_tag == user_speaker:
        return False, "target_is_user"

    if not target_body.strip():
        return False, "empty_reply"

    if not any(
        split_turn(turn)[0] == user_speaker
        for turn in context_turns
    ):
        return False, "user_missing_from_context"

    if len(tok.encode(target_body).ids) > max_reply_tokens:
        return False, "reply_too_long"

    formatted_context = [
        format_turn(turn, user_speaker, role_markers)
        for turn in context_turns
    ]
    formatted_target = format_turn(target_turn, user_speaker, role_markers)

    ids, _ = build_example(
        formatted_context,
        formatted_target,
        tok,
        eot_id,
        eoturn_id,
    )

    if len(ids) > max_total_tokens:
        return False, "total_too_long"

    return True, "accepted"


def build_improv_examples(
    turns,
    tok,
    eot_id=0,
    eoturn_id=1,
    block_size=BLOCK,
    val_every=VAL_EVERY,
    min_episode=10,
    max_episode=15,
    min_context=1,
    max_context=3,
    max_reply_tokens=40,
    max_total_tokens=256,
    seed=42,
    role_markers=False
):
    rng = random.Random(seed)

    train = []
    val = []
    rejection_counts = Counter()      # per-target reasons, except "no_eligible_user", which counts whole episodes

    for block_index, block_start in enumerate(
        range(0, len(turns), block_size)
    ):
        block_end = min(block_start + block_size, len(turns))
        is_validation = (block_index + 1) % val_every == 0

        cursor = block_start
        episode_index = 0

        while cursor < block_end:
            episode_length = rng.randint(min_episode, max_episode)
            episode_end = min(cursor + episode_length, block_end)

            episode_turns = turns[cursor:episode_end]
            speakers = get_speakers(episode_turns)

            # Find USER characters who can actually appear in a
            # preceding context for at least one assistant target.
            eligible_users = []

            for candidate in speakers:
                for target_idx in range(cursor + 1, episode_end):
                    target_tag, _ = split_turn(turns[target_idx])

                    if target_tag == candidate:
                        continue

                    max_n = min(max_context, target_idx - cursor)

                    for n in range(min_context, max_n + 1):
                        context = turns[target_idx - n:target_idx]

                        if any(
                            split_turn(t)[0] == candidate
                            for t in context
                        ):
                            eligible_users.append(candidate)
                            break

                    if candidate in eligible_users:
                        break

            if eligible_users:
                # One persistent USER character for the whole episode.
                user_speaker = rng.choice(eligible_users)

                for target_idx in range(cursor + 1, episode_end):
                    target_turn = turns[target_idx]
                    target_tag, _ = split_turn(target_turn)

                    if target_tag == user_speaker:
                        rejection_counts["target_is_user"] += 1
                        continue

                    # Only consider contexts from this episode.
                    valid_context_lengths = []

                    max_n = min(max_context, target_idx - cursor)

                    for n in range(min_context, max_n + 1):
                        context = turns[target_idx - n:target_idx]

                        if any(
                            split_turn(t)[0] == user_speaker
                            for t in context
                        ):
                            valid_context_lengths.append(n)

                    if not valid_context_lengths:
                        rejection_counts["user_missing_from_context"] += 1
                        continue

                    n_context = rng.choice(valid_context_lengths)
                    context_indices = list(
                        range(target_idx - n_context, target_idx)
                    )
                    context_turns = [
                        turns[i] for i in context_indices
                    ]

                    accepted, reason = filter_example(
                        context_turns,
                        target_turn,
                        user_speaker,
                        tok,
                        eot_id,
                        eoturn_id,
                        max_reply_tokens,
                        max_total_tokens,
                    )

                    if not accepted:
                        rejection_counts[reason] += 1
                        continue

                    formatted_context = [
                        format_turn(t, user_speaker, role_markers)
                        for t in context_turns
                    ]
                    formatted_target = format_turn(
                        target_turn,
                        user_speaker,
                        role_markers
                    )

                    ids, prompt_len = build_example(
                        formatted_context,
                        formatted_target,
                        tok,
                        eot_id,
                        eoturn_id,
                    )

                    example = {
                        "ids": ids,
                        "prompt_len": prompt_len,
                        "total_len": len(ids),
                        "block_index": block_index,
                        "episode_index": episode_index,
                        "episode_start": cursor,
                        "episode_end": episode_end,
                        "is_final_episode": episode_end == block_end,
                        "context_indices": context_indices,
                        "target_index": target_idx,
                        "user_speaker": user_speaker,
                    }

                    if is_validation:
                        val.append(example)
                    else:
                        train.append(example)

            else:
                rejection_counts["no_eligible_user"] += 1

            cursor = episode_end
            episode_index += 1

    stats = {
        "train_count": len(train),
        "val_count": len(val),
        "rejections": dict(rejection_counts),
    }

    return train, val, stats