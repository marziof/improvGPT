# tests/conftest.py
import pytest
import torch
from helpers import V, CTX
from gpt.config import Config
from gpt.model import GPT



@pytest.fixture(autouse=True)
def no_grad():
    with torch.no_grad():
        yield


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    cfg = Config(n_layer=2, n_head=2, d_model=64, context_len=CTX, batch_size=4, vocab_size=V)
    m = GPT(cfg)
    for p in m.parameters():
        if p.dim() >= 2:
            torch.nn.init.normal_(p, std=0.15)
    return m.eval()
