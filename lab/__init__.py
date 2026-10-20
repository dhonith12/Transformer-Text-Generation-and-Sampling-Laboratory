"""Transformer Text Generation and Sampling Laboratory.

A classroom lab for Unit IV / CO6 that lets students compare decoding
strategies (greedy, temperature, top-k, top-p), analyse the resulting text
(diversity, repetition, coherence, randomness, helpfulness, relevance,
factuality, safety), experiment with test-time compute, and run controlled
safety test cases (hallucination, misinformation, bias, unsafe requests,
prompt injection).

Submodules
----------
lab.backend    language-model backends (n-gram default, HuggingFace, tiny GPT)
lab.decoding   decoding / sampling algorithms implemented from scratch
lab.metrics    generation-quality metrics and the rubric-style analysis
lab.safety     controlled safety test cases, detectors and the guard wrapper
lab.compute    test-time compute experiments (best-of-n, budget sweeps)
lab.visualize  optional matplotlib charts
"""

__version__ = "1.0.0"

from .decoding import DecodingConfig, GenerationResult, generate, STRATEGIES  # noqa: F401
from .backend import resolve_backend  # noqa: F401
