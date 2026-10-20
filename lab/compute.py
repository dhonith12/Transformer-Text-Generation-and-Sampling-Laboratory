"""Test-time compute experiments.

*Simple generation* spends one forward pass per token and returns the first
sample.  *Test-time compute* spends additional computation **after** the
prompt is seen, e.g.:

* best-of-N sampling + reranking — draw N candidates, score them, keep the best
* majority voting (self-consistency) — draw N samples, vote on the answer
* token budget — let the model think longer before it answers

These are the cheapest, most teachable versions of the idea; the point is to
measure the quality/compute trade-off on your own prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .backend import Backend
from .decoding import DecodingConfig, GenerationResult, generate
from .metrics import (ReferenceIndex, coherence_score, content_words,
                      factuality, relevance_score)
from .safety import load_knowledge, scan_text, total_flags
from .utils import clamp, format_table, mean


# --------------------------------------------------------------------------
# Candidate scoring (the "verifier" used by best-of-N)
# --------------------------------------------------------------------------
@dataclass
class CandidateScore:
    composite: float
    relevance: float
    coherence: float
    factuality: float
    safety: float
    text: str
    seed: int

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def score_candidate(result: GenerationResult, prompt: str, backend: Backend,
                    kb: dict, reference: Optional[ReferenceIndex] = None,
                    weights: Sequence[float] = (0.35, 0.25, 0.20, 0.20)
                    ) -> CandidateScore:
    """Transparent linear verifier — students can re-weight it live."""
    text = result.continuation
    rel = relevance_score(prompt, text)
    coh = coherence_score(backend, text, reference)["coherence"]
    fac = factuality(text, kb, prompt=prompt).score
    safe = clamp(1.0 - 0.5 * total_flags(scan_text(text, kb, prompt)))
    w_rel, w_coh, w_fac, w_safe = weights
    composite = clamp(w_rel * rel + w_coh * coh + w_fac * fac + w_safe * safe)
    return CandidateScore(round(composite, 3), round(rel, 3), round(coh, 3),
                          round(fac, 3), round(safe, 3), text,
                          result.config.seed)


def sample_candidates(backend: Backend, prompt: str, cfg: DecodingConfig,
                      n: int) -> List[GenerationResult]:
    """Draw *n* independent samples by varying the random seed."""
    return [generate(backend, prompt, cfg.with_(seed=cfg.seed + i))
            for i in range(n)]


# --------------------------------------------------------------------------
# Best-of-N
# --------------------------------------------------------------------------
@dataclass
class BestOfNResult:
    best: CandidateScore
    candidates: List[CandidateScore]
    n: int

    def table(self) -> str:
        rows = [[i + 1, c.composite, c.relevance, c.coherence, c.factuality,
                 c.safety, c.text]
                for i, c in enumerate(self.candidates)]
        return format_table(
            ["#", "score", "rel", "coh", "fact", "safe", "candidate"],
            rows, max_width=40)

    def to_dict(self) -> dict:
        return {"n": self.n, "best": self.best.to_dict(),
                "candidates": [c.to_dict() for c in self.candidates]}


def best_of_n(backend: Backend, prompt: str, cfg: DecodingConfig, n: int = 8,
              kb: Optional[dict] = None,
              reference: Optional[ReferenceIndex] = None) -> BestOfNResult:
    kb = kb or load_knowledge()
    results = sample_candidates(backend, prompt, cfg, n)
    scores = [score_candidate(r, prompt, backend, kb, reference)
              for r in results]
    best = max(scores, key=lambda c: c.composite)
    return BestOfNResult(best, scores, n)


# --------------------------------------------------------------------------
# Scaling curves
# --------------------------------------------------------------------------
@dataclass
class ScalingRow:
    n: int
    best_score: float
    mean_score: float
    worst_score: float
    gain_over_single: float

    def as_list(self) -> list:
        return [self.n, self.best_score, self.mean_score, self.worst_score,
                self.gain_over_single]


def quality_vs_n(backend: Backend, prompt: str, cfg: DecodingConfig,
                 ns: Sequence[int] = (1, 2, 4, 8, 16),
                 kb: Optional[dict] = None,
                 reference: Optional[ReferenceIndex] = None) -> List[ScalingRow]:
    """Best-of-N quality curve: how much does extra sampling compute buy?"""
    kb = kb or load_knowledge()
    max_n = max(ns)
    all_scores = [score_candidate(r, prompt, backend, kb, reference)
                  for r in sample_candidates(backend, prompt, cfg, max_n)]
    baseline = all_scores[0].composite
    rows = []
    for n in sorted(ns):
        pool = all_scores[:n]
        scores = [c.composite for c in pool]
        best = max(scores)
        rows.append(ScalingRow(n, round(best, 3), round(mean(scores), 3),
                               round(min(scores), 3),
                               round(best - baseline, 3)))
    return rows


@dataclass
class BudgetRow:
    max_new_tokens: int
    relevance: float
    coherence: float
    repetition: float
    distinct_2: float
    length: int

    def as_list(self) -> list:
        return [self.max_new_tokens, self.relevance, self.coherence,
                self.repetition, self.distinct_2, self.length]


def quality_vs_tokens(backend: Backend, prompt: str, cfg: DecodingConfig,
                      budgets: Sequence[int] = (8, 16, 32, 64, 128),
                      reference: Optional[ReferenceIndex] = None
                      ) -> List[BudgetRow]:
    """Does thinking/generating longer help, or does error accumulate?"""
    from .metrics import distinct_n, repetition_index, relevance_score

    rows: List[BudgetRow] = []
    for budget in budgets:
        result = generate(backend, prompt,
                          cfg.with_(max_new_tokens=budget, seed=cfg.seed))
        text = result.continuation
        rows.append(BudgetRow(
            budget,
            round(relevance_score(prompt, text), 3),
            round(coherence_score(backend, text, reference)["coherence"], 3),
            round(repetition_index(text, 3), 3),
            round(distinct_n(text, 2), 3),
            result.num_tokens))
    return rows


# --------------------------------------------------------------------------
# Self-consistency (majority vote)
# --------------------------------------------------------------------------
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def extract_answer(text: str) -> str:
    """Cheap answer extractor: last number, else last few content words."""
    numbers = NUMBER_RE.findall(text)
    if numbers:
        return numbers[-1]
    content = [w for w in content_words(text)]
    return " ".join(content[-3:]) if content else text.strip()[:30]


@dataclass
class VoteResult:
    answer: str
    agreement: float
    counts: Dict[str, int]
    answers: List[str]

    def table(self) -> str:
        rows = [[a, c] for a, c in
                sorted(self.counts.items(), key=lambda kv: -kv[1])]
        return format_table(["answer", "votes"], rows, max_width=40)

    def to_dict(self) -> dict:
        return {"answer": self.answer, "agreement": round(self.agreement, 3),
                "counts": self.counts, "answers": self.answers}


def majority_vote(texts: Sequence[str]) -> VoteResult:
    answers = [extract_answer(t) for t in texts]
    counts: Dict[str, int] = {}
    for a in answers:
        counts[a] = counts.get(a, 0) + 1
    answer, votes = max(counts.items(), key=lambda kv: kv[1])
    return VoteResult(answer, votes / len(answers) if answers else 0.0,
                      counts, answers)


def self_consistency(backend: Backend, prompt: str, cfg: DecodingConfig,
                     n: int = 9) -> VoteResult:
    results = sample_candidates(backend, prompt, cfg, n)
    return majority_vote([r.continuation for r in results])


# --------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------
def print_n_curve(rows: Sequence[ScalingRow]) -> None:
    print("\n=== Test-time compute I: best-of-N reranking ===")
    print(format_table(["N samples", "best score", "mean score", "worst score",
                        "gain vs N=1"],
                       [r.as_list() for r in rows]))


def print_budget_curve(rows: Sequence[BudgetRow]) -> None:
    print("\n=== Test-time compute II: token budget ===")
    print(format_table(["token budget", "relevance", "coherence", "repetition",
                        "distinct-2", "actual tokens"],
                       [r.as_list() for r in rows]))
