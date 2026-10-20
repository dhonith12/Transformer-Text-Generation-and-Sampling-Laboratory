"""Generation-quality metrics: diversity, repetition, coherence, randomness,
plus the rubric-style evaluation (helpfulness, relevance, factuality, safety).

Everything here is a *classroom heuristic*: transparent formulas that students
can reason about, not a production evaluator.  Every metric documents exactly
what it measures so that changing a sampling parameter has an interpretable
effect.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .utils import clamp, mean

WORD_RE = re.compile(r"[a-z0-9']+")
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

STOPWORDS = set("""a about after all also an and any are as at be because been
before being between both but by can could did do does doing down during each
few for from further had has have having he her here hers herself him himself
his how i if in into is it its itself just me more most my myself no nor not
now of off on once only or other our ours ourselves out over own same she so
some such than that the their theirs them themselves then there these they
this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves
is are was were be been being have has had do does did shall should will would
can could may might must""".split())

AI_TOPIC_KEYWORDS = set("""ai artificial intelligence model models data learning
machine neural network networks transformer transformers token tokens sampling
decode decoding generation prompt training parameters algorithm systems compute
future probability greedy top k p temperature""".split())


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
def words(text: str) -> List[str]:
    return WORD_RE.findall(text.lower())


def stems(text: str) -> set:
    """Very light stemming so 'generate'/'generating' count as one word."""
    out = set()
    for w in words(text):
        if w.endswith("'s"):
            w = w[:-2]
        for suffix in ("ing", "edly", "ed", "ies", "es", "s"):
            if len(w) > 4 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        out.add(w)
    return out


def content_words(text: str) -> set:
    return stems(text) - STOPWORDS


def ngrams(items: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(items[i:i + n]) for i in range(len(items) - n + 1)]


# --------------------------------------------------------------------------
# Diversity & repetition
# --------------------------------------------------------------------------
def distinct_n(text: str, n: int) -> float:
    """distinct-n = unique n-grams / total n-grams (1.0 = never repeats)."""
    grams = ngrams(words(text), n)
    return len(set(grams)) / len(grams) if grams else 0.0


def repetition_index(text: str, n: int) -> float:
    """1 - distinct-n: fraction of n-gram *slots* filled by repeats."""
    return 1.0 - distinct_n(text, n)


def longest_repeated_ngram(text: str, n_max: int = 6) -> Tuple[str, int]:
    """The most over-used n-gram: (phrase, occurrences)."""
    toks = words(text)
    best, best_count = "", 1
    for n in range(2, n_max + 1):
        counts = Counter(ngrams(toks, n))
        for gram, c in counts.items():
            if c > best_count or (c == best_count and len(gram) > len(best.split())):
                best, best_count = " ".join(gram), c
    return best, best_count


# --------------------------------------------------------------------------
# Randomness
# --------------------------------------------------------------------------
def word_entropy(text: str) -> float:
    """Shannon entropy (bits/word) of the generated text itself."""
    counts = Counter(words(text))
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def char_entropy(text: str) -> float:
    counts = Counter(text)
    total = sum(counts.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def randomness_from_trace(result, vocab_size: int) -> float:
    """Mean sampling entropy divided by ln(vocab): 0 = greedy, 1 = uniform."""
    if vocab_size <= 1:
        return 0.0
    return clamp(result.mean_entropy() / math.log(vocab_size))


# --------------------------------------------------------------------------
# Coherence
# --------------------------------------------------------------------------
class ReferenceIndex:
    """Bigram/unigram statistics of a reference corpus, used as a cheap
    'has the model stayed on planet Earth' coherence proxy."""

    def __init__(self, corpus_text: str) -> None:
        toks = words(corpus_text)
        self.unigrams = set(toks)
        self.bigrams = set(ngrams(toks, 2))

    def continuity(self, text: str) -> float:
        """Fraction of the text's word-bigrams that occur in the corpus."""
        grams = ngrams(words(text), 2)
        if not grams:
            return 0.0
        return sum(1 for g in grams if g in self.bigrams) / len(grams)

    def vocabulary_fit(self, text: str) -> float:
        toks = [w for w in words(text)]
        if not toks:
            return 0.0
        return sum(1 for w in toks if w in self.unigrams) / len(toks)


def fluency(backend, text: str) -> dict:
    """Self-perplexity of the text under *backend* (lower = smoother)."""
    return backend.score_text(text)


def coherence_score(backend, text: str, reference: Optional[ReferenceIndex]) -> dict:
    stats = fluency(backend, text)
    ppl = stats["perplexity"] or 1.0
    fluency_norm = clamp(1.0 - math.log(max(ppl, 1.0)) / math.log(1000.0))
    cont = reference.continuity(text) if reference else float("nan")
    combined = 0.5 * fluency_norm + 0.5 * cont if reference else fluency_norm
    return {"perplexity": round(ppl, 2), "avg_token_prob": round(stats["avg_prob"], 4),
            "continuity": round(cont, 3), "coherence": round(clamp(combined), 3)}


# --------------------------------------------------------------------------
# Relevance
# --------------------------------------------------------------------------
def prompt_overlap(prompt: str, text: str) -> float:
    """Recall of the prompt's content words inside the generation."""
    p = content_words(prompt)
    if not p:
        return 1.0
    return len(p & content_words(text)) / len(p)


def topic_fit(text: str, topic_words: Sequence[str] = ()) -> float:
    """Fraction of the topic's key concepts present in the text."""
    keys = set(topic_words) or AI_TOPIC_KEYWORDS
    toks = stems(text)
    return len(keys & toks) / len(keys) if keys else 0.0


def relevance_score(prompt: str, text: str) -> float:
    return clamp(0.5 * prompt_overlap(prompt, text) + 0.5 * topic_fit(text))


# --------------------------------------------------------------------------
# Factuality (against the bundled knowledge base)
# --------------------------------------------------------------------------
@dataclass
class FactualityReport:
    score: float
    verified: List[str] = field(default_factory=list)
    contradictions: List[str] = field(default_factory=list)
    unverifiable: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"score": self.score, "verified": self.verified,
                "contradictions": self.contradictions,
                "unverifiable": self.unverifiable}


def _matches_keywords(sentence: str, keywords: Sequence[str]) -> bool:
    s = sentence.lower()
    return all(re.search(r"\b" + re.escape(k.lower()) + r"\b", s)
               for k in keywords)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def factuality(text: str, kb: dict, prompt: str = "") -> FactualityReport:
    """Very small claim checker.

    * a sentence matching a known *fact*'s keywords  -> verified
    * a sentence matching a *false claim*'s keywords -> contradiction
    * an assertive sentence with numbers / capitalised nouns that is neither
      grounded in the knowledge base nor in the prompt -> unverifiable
    """
    facts = kb.get("facts", [])
    false_claims = kb.get("false_claims", [])
    prompt_stems = content_words(prompt)
    verified, contradictions, unverifiable = [], [], []

    for sentence in split_sentences(text):
        if _matches(sentence, [f["keywords"] for f in facts]):
            verified.append(sentence)
        elif _matches(sentence, [f["keywords"] for f in false_claims]):
            contradictions.append(sentence)
        else:
            assertive = bool(NUMBER_RE.search(sentence)) or bool(
                re.search(r"\b(according to|study|researchers?|survey|report)\b",
                          sentence.lower()))
            grounded = content_words(sentence) & prompt_stems
            if assertive and not grounded:
                unverifiable.append(sentence)

    total = len(verified) + len(unverifiable)
    score = (len(verified) / total) if total else 1.0
    # one explicit contradiction makes the whole passage untrustworthy
    score = clamp(score - 0.75 * len(contradictions))
    return FactualityReport(round(score, 3), verified, contradictions,
                            unverifiable)


def _matches(sentence: str, keyword_sets: Sequence[Sequence[str]]) -> bool:
    return any(_matches_keywords(sentence, ks) for ks in keyword_sets)


# --------------------------------------------------------------------------
# Rubric analysis
# --------------------------------------------------------------------------
@dataclass
class AnalysisReport:
    prompt: str
    strategy: str
    metrics: Dict[str, object]

    def rows(self) -> List[Tuple[str, str, str]]:
        notes = {
            "distinct_1": "unique words / words (diversity)",
            "distinct_2": "unique bigrams / bigrams (diversity)",
            "repetition": "1 - distinct-3 (how much the model loops)",
            "longest_repeat": "most over-used phrase",
            "randomness": "sampling entropy / ln(vocab): 0 greedy, 1 uniform",
            "text_entropy_bits": "Shannon entropy of generated words",
            "perplexity": "self-perplexity under the backend (lower=smoother)",
            "continuity": "bigrams that also occur in the reference corpus",
            "coherence": "blend of fluency and continuity (0..1)",
            "relevance": "prompt words + topic concepts present (0..1)",
            "factuality": "verified vs unverifiable claims vs knowledge base",
            "safety": "1 - detector flags in the output (0..1)",
            "helpfulness": "relevance+fluency+coherence+factuality+safety",
            "tokens": "number of generated tokens",
            "mean_entropy": "mean sampling entropy during generation (nats)",
            "mean_candidates": "mean tokens surviving the filters",
            "elapsed_sec": "wall-clock time of generation",
        }
        return [(k, v, notes.get(k, "")) for k, v in self.metrics.items()]

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "strategy": self.strategy,
                "metrics": self.metrics}


def analyze(result, backend, kb: Optional[dict] = None,
            reference: Optional[ReferenceIndex] = None,
            vocab_size: Optional[int] = None) -> AnalysisReport:
    """Full rubric-style analysis of one generation."""
    kb = kb or {"facts": [], "false_claims": []}
    text = result.continuation
    coh = coherence_score(backend, text, reference)
    safety_flags = 0  # filled in by lab.safety.scan_text when used
    from .safety import scan_text
    flags = scan_text(text, kb, prompt=result.prompt)
    safety_flags = sum(len(v) for v in flags.values())
    safety = clamp(1.0 - 0.5 * safety_flags)

    rel = relevance_score(result.prompt, text)
    fact = factuality(text, kb, prompt=result.prompt)
    flu = coh["coherence"]
    # Rubric weights: is the answer on-topic, smooth, true and harmless?
    helpful = clamp(0.30 * rel + 0.25 * flu + 0.20 * fact.score + 0.25 * safety)

    repeat_phrase, repeat_count = longest_repeated_ngram(text)
    metrics: Dict[str, object] = {
        "strategy": result.config.label(),
        "tokens": result.num_tokens,
        "distinct_1": round(distinct_n(text, 1), 3),
        "distinct_2": round(distinct_n(text, 2), 3),
        "repetition": round(repetition_index(text, 3), 3),
        "longest_repeat": f"{repeat_phrase!r} x{repeat_count}",
        "randomness": round(randomness_from_trace(
            result, vocab_size or backend.vocab_size), 3),
        "text_entropy_bits": round(word_entropy(text), 3),
        "mean_entropy": round(result.mean_entropy(), 3),
        "mean_candidates": round(result.mean_kept(), 1),
        "perplexity": coh["perplexity"],
        "continuity": coh["continuity"],
        "coherence": coh["coherence"],
        "relevance": round(rel, 3),
        "prompt_overlap": round(prompt_overlap(result.prompt, text), 3),
        "topic_fit": round(topic_fit(text), 3),
        "factuality": fact.score,
        "safety": round(safety, 3),
        "safety_flags": safety_flags,
        "helpfulness": round(helpful, 3),
        "elapsed_sec": round(result.elapsed_sec, 3),
    }
    return AnalysisReport(result.prompt, result.config.label(), metrics)


def aggregate_reports(reports: Sequence[AnalysisReport]) -> Dict[str, float]:
    """Mean of every numeric metric across several reports."""
    keys = [k for k, v in reports[0].metrics.items()
            if isinstance(v, (int, float))] if reports else []
    return {k: round(mean([float(r.metrics[k]) for r in reports]), 3)
            for k in keys}
