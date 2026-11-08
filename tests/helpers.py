import torch

V, CTX = 100, 64


def rand_ids(B, T, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (B, T), generator=g)