"""A miniature GPT you can train in ~a minute on the bundled corpus.

This module is *optional*: it needs PyTorch.  Everything else in the lab runs
without it.  Training produces ``artifacts/tiny_gpt.pt``, which the ``tiny``
backend loads so that students can run the exact same decoding algorithms on
a transformer they trained themselves.

Architecture: token + learned positional embeddings, N pre-norm blocks with
causal multi-head self-attention and an MLP, final LayerNorm, weight-tied
output head.  A small, honest implementation — no tricks.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

from .backend import DEFAULT_CHECKPOINT, DEFAULT_CORPUS, WordTokenizer

try:                                   # optional dependency
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:                    # pragma: no cover
    TORCH_AVAILABLE = False


@dataclass
class TinyGPTConfig:
    vocab_size: int = 256
    block_size: int = 64
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1


if TORCH_AVAILABLE:                    # pragma: no cover - requires torch

    class CausalSelfAttention(nn.Module):
        def __init__(self, cfg: TinyGPTConfig) -> None:
            super().__init__()
            self.n_head = cfg.n_head
            self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
            self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
            self.drop = nn.Dropout(cfg.dropout)

        def forward(self, x):
            B, T, C = x.shape
            q, k, v = self.qkv(x).split(C, dim=2)
            hs = C // self.n_head
            q = q.view(B, T, self.n_head, hs).transpose(1, 2)
            k = k.view(B, T, self.n_head, hs).transpose(1, 2)
            v = v.view(B, T, self.n_head, hs).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) / math.sqrt(hs)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            att = att.masked_fill(~mask, float("-inf"))
            att = self.drop(att.softmax(dim=-1))
            return self.drop(self.proj((att @ v).transpose(1, 2)
                                       .reshape(B, T, C)))

    class Block(nn.Module):
        def __init__(self, cfg: TinyGPTConfig) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(cfg.n_embd)
            self.attn = CausalSelfAttention(cfg)
            self.ln2 = nn.LayerNorm(cfg.n_embd)
            self.mlp = nn.Sequential(
                nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
                nn.GELU(),
                nn.Linear(4 * cfg.n_embd, cfg.n_embd),
                nn.Dropout(cfg.dropout))

        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))

    class TinyGPT(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            cfg = TinyGPTConfig(**kwargs)
            self.cfg = cfg
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
            self.drop = nn.Dropout(cfg.dropout)
            self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
            self.ln_f = nn.LayerNorm(cfg.n_embd)
            self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
            self.head.weight = self.tok_emb.weight      # weight tying
            self.apply(self._init)

        @staticmethod
        def _init(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

        def forward(self, idx, targets=None):
            B, T = idx.shape
            pos = torch.arange(T, device=idx.device)
            x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                       targets.view(-1))
            return logits, loss

        @torch.no_grad()
        def generate(self, idx, max_new_tokens: int, temperature: float = 1.0,
                     top_k: Optional[int] = None, top_p: Optional[float] = None):
            """Tiny convenience generator for smoke tests (the real lab uses
            lab.decoding)."""
            for _ in range(max_new_tokens):
                ctx = idx[:, -self.cfg.block_size:]
                logits, _ = self(ctx)
                logits = logits[:, -1, :] / max(temperature, 1e-6)
                if top_k:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = logits.softmax(dim=-1)
                idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
            return idx


def train(steps: int = 400, batch_size: int = 24, lr: float = 3e-3,
          block_size: int = 64, n_embd: int = 128, n_head: int = 4,
          n_layer: int = 2, seed: int = 123,
          corpus_path: Optional[Path] = None,
          out_path: Optional[Path] = None, quiet: bool = False) -> Path:
    """Train the miniature GPT on the bundled corpus and save a checkpoint."""
    if not TORCH_AVAILABLE:             # pragma: no cover - optional dep
        raise RuntimeError("'torch' is not installed. Install it with:\n"
                           "    pip install torch\n"
                           "or use the dependency-free backend (--backend ngram).")

    torch.manual_seed(seed)
    corpus = Path(corpus_path) if corpus_path else DEFAULT_CORPUS
    text = corpus.read_text(encoding="utf-8")
    tokenizer = WordTokenizer.fit([text])
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    cfg = TinyGPTConfig(vocab_size=tokenizer.vocab_size, block_size=block_size,
                        n_embd=n_embd, n_head=n_head, n_layer=n_layer)
    model = TinyGPT(**asdict(cfg))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    def batch():
        starts = torch.randint(0, len(ids) - block_size - 1, (batch_size,))
        x = torch.stack([ids[s:s + block_size] for s in starts])
        y = torch.stack([ids[s + 1:s + block_size + 1] for s in starts])
        return x, y

    started, losses = time.perf_counter(), []
    model.train()
    for step in range(1, steps + 1):
        x, y = batch()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
        if not quiet and (step % 50 == 0 or step == 1):
            print(f"  step {step:4d}/{steps}  loss {loss.item():.3f}  "
                  f"({time.perf_counter() - started:.1f}s)")

    out = Path(out_path) if out_path else DEFAULT_CHECKPOINT
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": asdict(cfg),
                "tokenizer": tokenizer.saved(),
                "meta": {"corpus": str(corpus), "steps": steps,
                         "final_loss": round(losses[-1], 4)}}, out)
    if not quiet:
        print(f"  checkpoint saved to {out} "
              f"({out.stat().st_size / 1e6:.1f} MB, vocab={cfg.vocab_size})")
    return out


if __name__ == "__main__":              # pragma: no cover
    train()
