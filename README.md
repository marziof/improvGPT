# improvGPT (working title)

A small GPT-style language model built from scratch in PyTorch, taken through a post-training pipeline: pretraining, supervised finetuning, a "reply as the other character" improv stage, and (in progress) reinforcement learning with programmatic rewards. The goal is to implement and measure the following: tokenizer, RoPE, attention, KV cache, optimizers, evaluation, SFT, RL.

**Status: work in progress.** Pretraining, SFT, the improv stage and the KV cache are done and tested. The reward harness and the RL loop are next. The 12M-parameter model produces samples that are fluent in form but not totally coherent in content. The scope of the project is learning about the pipeline's components and their impact rather than perfect text outputs.


## Pipeline

```
TinyStories --> pretrain --> Shakespeare SFT --> improv SFT --> RL (programmatic reward, GRPO/PPO) --> quantifying results, summary
                 (done)         (done)             (done)          (next)
```

1. **Pretrain** on TinyStories (byte-level BPE, 4096 tokens).
2. **Shakespeare SFT** on Tiny Shakespeare: language and the native `SPEAKER:` play format.
3. **Improv SFT:** given 1 to 3 context turns, reply as the next character and stop (`<|endofturn|>`). The loss is masked to the reply only.
4. **RL** with direct, programmatic rewards: the reply should stop properly, pick up the user's words ("yes, and"), and avoid copying and non-words. KL penalty to the frozen SFT model.

## Model

| | |
|---|---|
| Parameters | 12.19M (6 layers, 6 heads, d_model 384, context 256, vocab 4096) |
| Blocks | pre-norm RMSNorm, causal attention with RoPE (half-split pairing), GELU MLP (4x), no biases |
| Embeddings | input and output embeddings tied |
| Training | AdamW (betas 0.9/0.95), warmup + cosine, gradient clipping, fp16 autocast with GradScaler on CUDA |
| Inference | KV cache (preallocated per-layer buffers, keys stored after RoPE), batched generation with per-sequence stop flags |

## Setup

```bash
# install PyTorch for your platform first, then:
pip install -e .            # add ".[dev]" for pytest, pandas, matplotlib, jupyter
pytest -q                   # unit tests (a small random model, no checkpoint needed)
```

Development runs on a laptop CPU with small configs (`configs/tiny.yaml`); long runs use a Kaggle T4 (`configs/base.yaml`). Optional logging with Weights & Biases (`--set wandb=true`).

## Reproducing the pipeline

```bash
# 1) pretraining data (a small dev run; the full run used 1M stories on Kaggle)
python scripts/prepare_data.py --streaming --num_stories 10000
python scripts/prepare_data.py --num_stories 1000000 --num_val_stories 5000 --out_dir data_full

# 2) pretrain
python -m gpt.train --config configs/base.yaml --set data_dir=data_full run_name=pretrain_v2

# 3) Shakespeare SFT (downloads Tiny Shakespeare; reuses the pretraining tokenizer)
python scripts/prep_sft.py
python -m gpt.train --config configs/sft.yaml --set data_dir=data_sft_blocks run_name=sft_blocks_lr1e-4 device=cpu

# 4) improv data and SFT (the embedding of <|endoftext|> initializes <|endofturn|>)
python scripts/prepare_improv.py
python -m gpt.train --config configs/improv.yaml --set 'copy_embedding=[1,0]' max_steps=200 warmup_steps=10 eval_interval=10 run_name=improv_warm

# sample from any run
python -m gpt.generate --run_dir runs/pretrain_v2 --prompt "Once upon a time"
```

Any config field can be overridden with `--set key=value`. Runs write to `runs/<run_name>/` (config, `metrics.jsonl`, `best.pt`, `last.pt`).

## Results so far

**Pretraining** (12.19M parameters, 1M stories, about 224M tokens, 13,500 steps, batch 64x256, Kaggle T4, fp16, roughly 90k tokens/s): validation loss **1.517** (perplexity about 4.6, about 0.55 bits per character). fp16 produced no overflows over the run.

**Shakespeare SFT** (validation split by blocks of turns, shared with the improv stage): validation loss 5.741 before finetuning, **3.151** after (1.83 bits per character at 2.48 characters per token). A higher learning rate (3e-4) did not help: its best was 3.160 and it overfit within 200 steps.

**Improv SFT** (1,630 training and 158 validation examples). Validation loss split into reply tokens and the stop token:

| Start | reply-token NLL | stop-token NLL |
|---|---|---|
| pretrained only, no finetuning | 5.999 | 16.6 |
| after Shakespeare SFT | 3.663 | 20.2 |
| improv SFT from Shakespeare SFT | 3.829 | 0.53 |
| + stop token initialized from `<|endoftext|>` | **3.709** | 0.52 |
| improv SFT directly from pretrained (ablation) | 4.272 | 0.38 |

- The Shakespeare stage is not redundant: skipping it costs about 0.44 nats on reply tokens (paired, same validation set).
- A freshly added special token that the model has never seen costs about 20 nats, and its gradient disturbs the reply predictions. Copying an existing token's embedding into it fixed most of the regression.
- Sampling (150 held-out contexts, 8 replies each): about 96% of replies end with a stop token within 60 tokens. Roughly 8% of words are non-words at temperature 1.0, against about 2% at temperature 0.8 with top-k 50. That decoding-only baseline is what RL has to beat.

**KV cache:** logits match full recomputation to about 1e-5 with 100% argmax agreement (batch sizes 1 and 3, cache reuse, guard checks, greedy and seeded-sampling equality of the generators). Cache size is 18.4 KB per token in fp32 (4.7 MB per full-context sequence, against 48.8 MB of weights). Decode throughput on a laptop CPU: 133 / 601 / 1,638 / 3,853 tokens/s at batch 1 / 4 / 16 / 64 (overhead-bound at small batch, compute-bound at large).

## Repository layout

```
src/gpt/        model, config, data loaders, optimizer/scheduler, training loop, generation (plain, cached, batched),
                improv data construction, shared split rule, reward (in progress), rl/ (in progress)
scripts/        data preparation (TinyStories, Shakespeare SFT, improv)
configs/        tiny, base (pretraining), sft, improv
tests/          KV cache, batched generation, improv data format and masking, splits, tokenizer hash
notebooks/      checkpoint checks, benchmarks, baselines and analysis
```

## Engineering notes

- **One split rule everywhere.** Turns are grouped in blocks of 50 and every 10th block is validation, for both Shakespeare SFT and the improv data, so no later stage trains on an earlier stage's validation text.
- **Tokenizer safety.** Every data folder and checkpoint records the SHA-256 of its tokenizer, and loading a checkpoint with a different tokenizer fails loudly (the vocabulary size alone cannot tell them apart).
- **Loss masking** is done by label (-100) at the data layer; the model code is unchanged.
- **Paired comparisons** on the same contexts and seeds, with standard errors, and a baseline for every claim.

## Data and licenses

No data or checkpoints are committed. Data is downloaded by the scripts.

- **TinyStories** (`roneneldan/TinyStories`): see the license on its dataset card.
- **Tiny Shakespeare:** the text of Shakespeare's plays (public domain), as compiled in the `karpathy/char-rnn` repository; see that repository for its own license.

## Roadmap

- [x] Tokenizer, data pipeline, model, training loop, checkpoint and resume
- [x] Pretraining on TinyStories (Kaggle)
- [x] KV cache and batched generation, tests, benchmarks
- [x] Shakespeare SFT with a validation split shared across stages
- [x] Improv SFT with reply-only loss and the stop-token fix
- [ ] Reward harness with per-term logging
- [ ] RL loop: GRPO first, then PPO, with a KL penalty to the frozen SFT model
- [ ] Comparisons: RL vs. base vs. sharper decoding vs. best-of-N; reward term ablations; KL sweep
- [ ] Optional: constraint-following stories with exact, code-checkable rewards; preference data and DPO; scaling and tokenizer experiments