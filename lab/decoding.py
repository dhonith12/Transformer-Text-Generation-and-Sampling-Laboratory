"""Decoding / sampling strategies implemented from scratch.

Every strategy consumes one row of logits and returns a next token:

``greedy``        always the arg-max token (deterministic)
``random``        uniform draw over the whole vocabulary (ignores the model)
``temperature``   softmax with temperature τ, then multinomial sampling
``topk``          temperature + keep only the k most likely tokens
``topp``          temperature + keep the smallest set with cumulative prob ≥ p
``topk_topp``     the industry default: top-k followed by top-p
``min_p``         keep tokens whose prob ≥ min_p × max-prob (modern default)

The functions are deliberately pure (lists of floats in, ints out) so they
can be unit-tested and reused with any backend.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

STRATEGIES: Tuple[str, ...] = (
    "greedy", "random", "temperature", "topk", "topp", "topk_topp", "min_p")

_DESCRIPTIONS = {
    "greedy": "arg-max decoding (no randomness)",
    "random": "uniform random tokens (model ignored)",
    "temperature": "softmax(T) + multinomial sampling",
    "topk": "temperature + top-k filtering",
    "topp": "temperature + nucleus (top-p) filtering",
    "topk_topp": "temperature + top-k + top-p filtering",
    "min_p": "temperature + min-p filtering",
}


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class DecodingConfig:
    strategy: str = "temperature"
    temperature: float = 1.0
    top_k: int = 40
    top_p: float = 0.95
    min_p: float = 0.05
    repetition_penalty: float = 1.0
    max_new_tokens: int = 40
    seed: int = 42
    stop: Tuple[str, ...] = ()

    # -- validation -------------------------------------------------------
    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}, "
                             f"got '{self.strategy}'")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0 (0 = greedy)")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if not 0.0 < self.min_p <= 1.0:
            raise ValueError("min_p must be in (0, 1]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

    # -- overrides --------------------------------------------------------
    def with_(self, **overrides) -> "DecodingConfig":
        return replace(self, **overrides)

    # -- labelling --------------------------------------------------------
    def label(self) -> str:
        if self.strategy == "greedy":
            base = "greedy"
        elif self.strategy == "random":
            base = "random (uniform)"
        elif self.strategy == "temperature":
            base = f"temperature={self.temperature:g}"
        elif self.strategy == "topk":
            base = f"top-k={self.top_k}"
        elif self.strategy == "topp":
            base = f"top-p={self.top_p:g}"
        elif self.strategy == "topk_topp":
            base = f"top-k={self.top_k} + top-p={self.top_p:g}"
        else:
            base = f"min-p={self.min_p:g}"
        if self.strategy != "greedy" and self.strategy != "random":
            base += f" (T={self.temperature:g})"
        if self.repetition_penalty != 1.0:
            base += f", rep.pen={self.repetition_penalty:g}"
        return base

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["stop"] = list(d["stop"])
        d["label"] = self.label()
        return d


# --------------------------------------------------------------------------
# Filters
# --------------------------------------------------------------------------
def apply_temperature(logits: Sequence[float],
                      temperature: float) -> List[float]:
    """Scale logits by 1/τ.  τ<1 sharpens, τ>1 flattens, τ=0 -> one-hot."""
    if temperature == 1.0:
        return list(logits)
    if temperature <= 0:
        best = max(range(len(logits)), key=lambda i: logits[i])
        return [0.0 if i == best else -math.inf for i in range(len(logits))]
    return [x / temperature for x in logits]


def filter_top_k(logits: Sequence[float], k: int) -> List[float]:
    """Keep only the k highest logits; set the rest to -inf."""
    if k <= 0 or k >= len(logits):
        return list(logits)
    keep = set(sorted(range(len(logits)), key=lambda i: -logits[i])[:k])
    return [x if i in keep else -math.inf for i, x in enumerate(logits)]


def filter_top_p(logits: Sequence[float], p: float) -> List[float]:
    """Nucleus filtering: keep the smallest prefix of tokens (sorted by
    probability) whose cumulative mass is >= p.  The most likely token is
    always kept."""
    if p >= 1.0:
        return list(logits)
    order = sorted(range(len(logits)), key=lambda i: -logits[i])
    finite = [i for i in order if logits[i] != -math.inf]
    if not finite:
        return list(logits)

    probs = []
    for i in finite:
        probs.append(math.exp(logits[i]) if logits[i] > -700 else 0.0)
    total = sum(probs) or 1.0
    probs = [pr / total for pr in probs]

    cumulative, cut = 0.0, len(finite)
    for pos, pr in enumerate(probs):
        cumulative += pr
        if cumulative >= p:
            cut = pos + 1
            break
    keep = set(finite[:max(cut, 1)])
    return [x if i in keep else -math.inf for i, x in enumerate(logits)]


def filter_min_p(logits: Sequence[float], min_p: float) -> List[float]:
    """min-p filtering: keep tokens whose probability is at least
    ``min_p`` times the probability of the most likely token."""
    if min_p <= 0.0:
        return list(logits)
    finite = [x for x in logits if x != -math.inf]
    if not finite:
        return list(logits)
    mx = max(finite)
    threshold = mx + math.log(min_p)
    return [x if x >= threshold else -math.inf for x in logits]


def apply_repetition_penalty(logits: Sequence[float],
                             seen_ids: Sequence[int],
                             penalty: float) -> List[float]:
    """CTRL-style penalty: discourage tokens that already appeared.

    Positive logits are divided by the penalty, negative logits multiplied by
    it (so both move towards zero).
    """
    if penalty == 1.0 or not seen_ids:
        return list(logits)
    out = list(logits)
    for tid in set(seen_ids):
        if 0 <= tid < len(out) and out[tid] != -math.inf:
            out[tid] /= penalty if out[tid] > 0 else (1.0 / penalty)
    return out


# --------------------------------------------------------------------------
# Per-step selection
# --------------------------------------------------------------------------
@dataclass
class StepInfo:
    token_id: int
    token: str
    logprob: float          # log-prob under the final sampling distribution
    entropy: float          # Shannon entropy (nats) of that distribution
    kept: int               # candidate tokens surviving the filters
    prob: float             # probability assigned to the chosen token

    def to_dict(self) -> dict:
        return {"token_id": self.token_id, "token": self.token,
                "logprob": self.logprob, "entropy": self.entropy,
                "kept": self.kept, "prob": self.prob}


def _normalize(logits: Sequence[float]) -> List[float]:
    finite = [x for x in logits if x != -math.inf]
    if not finite:
        return [1.0 / len(logits)] * len(logits)
    mx = max(finite)
    exps = [math.exp(x - mx) if x != -math.inf else 0.0 for x in logits]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def select_next_token(logits: Sequence[float], cfg: DecodingConfig,
                      seen_ids: Sequence[int] = (),
                      rng: Optional[random.Random] = None) -> StepInfo:
    """Apply the configured decoding strategy to *logits* and pick a token."""
    rng = rng or random.Random(cfg.seed)

    working = apply_repetition_penalty(logits, seen_ids,
                                       cfg.repetition_penalty)

    if cfg.strategy == "greedy" or cfg.temperature <= 0:
        token_id = max(range(len(working)), key=lambda i: working[i])
        kept = sum(1 for x in working if x != -math.inf)
        return StepInfo(token_id, "", 0.0, 0.0, kept, 1.0)

    if cfg.strategy == "random":
        token_id = rng.randrange(len(working))
        probs = [1.0 / len(working)] * len(working)
        return StepInfo(token_id, "", math.log(probs[token_id]),
                        math.log(len(working)), len(working),
                        probs[token_id])

    working = apply_temperature(working, cfg.temperature)
    if cfg.strategy in ("topk", "topk_topp"):
        working = filter_top_k(working, cfg.top_k)
    if cfg.strategy in ("topp", "topk_topp"):
        working = filter_top_p(working, cfg.top_p)
    if cfg.strategy == "min_p":
        working = filter_min_p(working, cfg.min_p)

    probs = _normalize(working)
    kept = sum(1 for x in working if x != -math.inf)

    # Inverse-CDF sampling is ~30x faster than random.choices on long vocabs.
    threshold = rng.random()
    cumulative, token_id = 0.0, len(probs) - 1
    for i, p in enumerate(probs):
        cumulative += p
        if cumulative >= threshold:
            token_id = i
            break

    prob = probs[token_id]
    logprob = math.log(prob) if prob > 0 else math.log(1e-12)
    return StepInfo(token_id, "", logprob, _entropy(probs), kept, prob)


# --------------------------------------------------------------------------
# Full generation
# --------------------------------------------------------------------------
@dataclass
class GenerationResult:
    prompt: str
    continuation: str
    full_text: str
    token_ids: List[int]
    steps: List[StepInfo]
    backend_name: str
    config: DecodingConfig
    elapsed_sec: float = 0.0

    # -- convenience ------------------------------------------------------
    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def mean_entropy(self) -> float:
        return sum(s.entropy for s in self.steps) / len(self.steps) if self.steps else 0.0

    def mean_kept(self) -> float:
        return sum(s.kept for s in self.steps) / len(self.steps) if self.steps else 0.0

    def mean_logprob(self) -> float:
        return sum(s.logprob for s in self.steps) / len(self.steps) if self.steps else 0.0

    def summary(self) -> Dict[str, float]:
        return {
            "strategy": self.config.label(),
            "tokens": self.num_tokens,
            "mean_entropy_nats": round(self.mean_entropy(), 3),
            "mean_candidates": round(self.mean_kept(), 1),
            "mean_logprob": round(self.mean_logprob(), 3),
            "seconds": round(self.elapsed_sec, 3),
        }

    def to_dict(self) -> dict:
        return {
            "prompt": self.prompt,
            "continuation": self.continuation,
            "full_text": self.full_text,
            "backend": self.backend_name,
            "config": self.config.to_dict(),
            "summary": self.summary(),
            "steps": [s.to_dict() for s in self.steps],
        }


def generate(backend, prompt: str,
             cfg: Optional[DecodingConfig] = None, **overrides) -> GenerationResult:
    """Generate *cfg.max_new_tokens* tokens after *prompt* with *backend*."""
    cfg = cfg.with_(**overrides) if (cfg and overrides) else (cfg or DecodingConfig())
    rng = random.Random(cfg.seed)
    started = time.perf_counter()

    ids = list(backend.encode(prompt))
    generated: List[int] = []
    steps: List[StepInfo] = []

    for _ in range(cfg.max_new_tokens):
        logits = backend.next_token_logits(ids)
        # Never sample masked tokens (e.g. <unk> of the n-gram backend).
        for bad in getattr(backend, "masked_ids", set()):
            if bad < len(logits):
                logits[bad] = -math.inf
        info = select_next_token(logits, cfg, ids, rng)
        info.token = backend.token_str(info.token_id)
        ids.append(info.token_id)
        generated.append(info.token_id)
        steps.append(info)

        if backend.eos_token_id is not None and info.token_id == backend.eos_token_id:
            break
        if cfg.stop:
            text_so_far = backend.decode(generated)
            if any(s and s in text_so_far for s in cfg.stop):
                break

    continuation = backend.decode(generated)
    return GenerationResult(
        prompt=prompt,
        continuation=continuation,
        full_text=prompt + (" " if prompt and not prompt[-1:].isspace() else "") + continuation,
        token_ids=generated,
        steps=steps,
        backend_name=backend.name,
        config=cfg,
        elapsed_sec=time.perf_counter() - started,
    )


def default_comparison_configs(base_seed: int = 42, tokens: int = 40):
    """The standard set of strategies the lab compares side by side."""
    return [
        DecodingConfig(strategy="greedy", max_new_tokens=tokens, seed=base_seed),
        DecodingConfig(strategy="random", max_new_tokens=tokens, seed=base_seed),
        DecodingConfig(strategy="temperature", temperature=0.6,
                       max_new_tokens=tokens, seed=base_seed),
        DecodingConfig(strategy="temperature", temperature=1.2,
                       max_new_tokens=tokens, seed=base_seed),
        DecodingConfig(strategy="topk", top_k=20, max_new_tokens=tokens, seed=base_seed),
        DecodingConfig(strategy="topk", top_k=5, max_new_tokens=tokens, seed=base_seed),
        DecodingConfig(strategy="topp", top_p=0.90, max_new_tokens=tokens, seed=base_seed),
        DecodingConfig(strategy="topp", top_p=0.99, max_new_tokens=tokens, seed=base_seed),
        DecodingConfig(strategy="topk_topp", top_k=50, top_p=0.95,
                       max_new_tokens=tokens, seed=base_seed),
        DecodingConfig(strategy="min_p", min_p=0.05, max_new_tokens=tokens, seed=base_seed),
    ]
