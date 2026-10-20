# Transformer Text Generation and Sampling Laboratory

A hands-on, educational laboratory for exploring transformer-based text generation, decoding strategies, evaluation metrics, test-time compute, and AI safety. Built with Python's standard library at its core, this tool lets students visualize and quantify how different sampling parameters affect generation quality.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Common Commands](#common-commands)
  - [Advanced Options](#advanced-options)
- [Architecture](#architecture)
- [Decoding Strategies](#decoding-strategies)
- [Evaluation Metrics](#evaluation-metrics)
- [Safety & Alignment](#safety--alignment)
- [Backends](#backends)
- [Educational Objectives](#educational-objectives)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

This laboratory implements the core concepts of text generation from scratch, including:

1. **Decoding Algorithms** - Greedy, random, temperature, top-k, top-p (nucleus), min-p, and combined strategies
2. **Language Model Backends** - A dependency-free n-gram model (default), optional HuggingFace transformers, and a trainable tiny GPT
3. **Evaluation Framework** - Automated metrics for diversity, repetition, coherence, relevance, factuality, safety, and helpfulness
4. **Test-Time Compute** - Best-of-N sampling, token budget experiments, and self-consistency voting
5. **Safety Testing** - Controlled red-teaming with hallucination, misinformation, bias, unsafe content, and prompt injection test cases

All decoding logic, filters, and metrics are implemented transparently and deliberately from scratch to maximize learning value.

## Features

- **Zero heavy dependencies by default**: Runs entirely on Python standard library using an n-gram backend trained on bundled corpus
- **Reproducible experiments**: Deterministic seeding, parameter logging, and automatic JSON/markdown report generation
- **Visual analysis**: Optional matplotlib charts for strategy comparisons, parameter sweeps, entropy traces, and safety results
- **Interactive exploration**: Built-in chat mode for live experimentation with on-the-fly metric reporting
- **Trainable components**: Optional tiny GPT implementation for training from scratch on custom corpora
- **Production-aware safety**: Controlled safety test suite demonstrating both unguarded behavior and simple defensive guardrails
- **Comprehensive reporting**: Generate publication-quality lab reports (`results/report.md`) with tables, interpretations, and embedded charts

## Installation

### Basic Installation (No External Dependencies)

```bash
git clone https://github.com/dhonith12/Transformer-Text-Generation-and-Sampling-Laboratory.git
cd Transformer-Text-Generation-and-Sampling-Laboratory
```

The laboratory runs out-of-the-box with the n-gram backend. No pip install required for core functionality.

### Optional Features

To enable additional backends and visualization:

```bash
pip install -r requirements.txt
```

The requirements file includes:
- `numpy>=1.24` - Numeric helpers for metrics and reports
- `matplotlib>=3.7` - Charts and visualizations

For neural network backends:
```bash
pip install torch>=2.0 transformers>=4.35  # Enables hf and tiny backends
```

## Quick Start

### 1. Compare All Decoding Strategies

See how different sampling methods behave on the default prompt:

```bash
python run_lab.py compare
```

### 2. Run a Parameter Sweep

Explore how temperature affects generation quality:

```bash
python run_lab.py sweep --param temperature --values 0.1,0.6,1.0,1.4,2.0
```

### 3. Generate a Full Lab Report

Run all experiments and produce a comprehensive markdown report with charts:

```bash
python run_lab.py report
```

Results are saved to `results/report.md` and `results/*.json`.

### 4. Interactive Chat

Experiment interactively with live metrics:

```bash
python run_lab.py chat
```

Switch strategies on-the-fly by typing `greedy`, `topp`, `temperature`, etc.

### 5. Safety Testing

Compare unguarded vs guarded behavior on controlled test cases:

```bash
python run_lab.py safety
```

## Usage

### Common Commands

| Command | Purpose | Example |
|---|---|---|
| `compare` | Side-by-side comparison of 10 standard decoding strategies | `python run_lab.py compare --prompt "Once upon a time" --tokens 60` |
| `sweep` | Sweep a single parameter (temperature/top-k/top-p/min-p/repetition-penalty) | `python run_lab.py sweep --param top_p --values 0.5,0.8,0.95,0.99` |
| `analyze` | Detailed metrics for one specific configuration | `python run_lab.py analyze --strategy topp --top-p 0.9 --tokens 80` |
| `safety` | Run controlled safety test suite (hallucination, bias, unsafe, injection) | `python run_lab.py safety --strategy topk_topp` |
| `compute` | Test-time compute experiments (best-of-N, token budgets, self-consistency) | `python run_lab.py compute --n-values 1,2,4,8,16 --vote "What is 2+2?"` |
| `report` | End-to-end lab report with all experiments | `python run_lab.py report --prompt "The future of AI is"` |
| `chat` | Interactive exploration loop with live metrics | `python run_lab.py chat --strategy temperature --temperature 0.7` |
| `train` | Train the tiny GPT from scratch | `python run_lab.py train --steps 500 --batch-size 32` |

### Advanced Options

**Backend selection** (`--backend`): Choose between `auto` (tries hf→tiny→ngram), `ngram` (dependency-free), `hf` (HuggingFace GPT-2), or `tiny` (self-trained miniature model).

```bash
python run_lab.py compare --backend hf --model gpt2
python run_lab.py train --steps 300 && python run_lab.py compare --backend tiny
```

**Disable outputs**: Skip JSON/plot generation for faster runs.

```bash
python run_lab.py compare --no-json --no-plots
```

**Custom corpus**: Use your own training text for the n-gram backend.

```bash
python run_lab.py compare --backend ngram --corpus data/my_corpus.txt --order 4
```

## Architecture

### Core Modules

| Module | Description |
|---|---|
| `lab.decoding` | Pure implementations of all sampling strategies. Works with any model that outputs logits. |
| `lab.backend` | Three pluggable backends (n-gram, HuggingFace, tiny GPT) sharing a common interface. |
| `lab.metrics` | Diversity, repetition, coherence, randomness, relevance, factuality, and composite helpfulness scores. |
| `lab.safety` | Controlled test cases, regex-based detectors (transparent by design), and a simple SafetyGuard wrapper. |
| `lab.compute` | Test-time compute: best-of-N, quality-vs-tokens curves, and self-consistency majority voting. |
| `lab.visualize` | Optional matplotlib charts for experiments and reports. |
| `lab.utils` | Shared helpers: ASCII tables, slugification, JSON persistence. |
| `lab.train_tiny` | Minimal GPT implementation and training loop for educational use. |

### Design Philosophy

- **Transparent over perfect**: Heuristic metrics are explicitly documented so students understand their formulas and limitations
- **Pure functions where possible**: Decoding filters are stateless and unit-testable
- **Dependency-free core**: The default path requires only the Python standard library
- **Reproducible by construction**: All experiments record seeds, configs, and intermediate artifacts

## Decoding Strategies

| Strategy | Description | When to Use |
|---|---|---|
| `greedy` | Always pick the argmax token (deterministic, τ=0) | Factual, deterministic outputs. Often repetitive. |
| `random` | Uniform sampling over vocabulary (ignores model probabilities) | Baseline for maximum diversity (usually incoherent). |
| `temperature` | Softmax with temperature τ before sampling (τ<1 sharpens, τ>1 flattens) | Primary control for diversity vs. coherence trade-off. |
| `topk` | Keep only k highest-probability tokens before sampling | Cuts tail noise, improves coherence while preserving some randomness. |
| `topp` (nucleus) | Keep smallest set with cumulative probability ≥ p | Adapts to distribution shape (widens when uncertain, narrows when confident). |
| `topk_topp` | Top-k followed by top-p (industry default) | Robust middle-ground for most applications. |
| `min_p` | Keep tokens with prob ≥ min_p × max_prob | Modern alternative that scales with model confidence. |

## Evaluation Metrics

The rubric evaluates generations along multiple dimensions (higher is better unless stated):

| Metric | What It Measures | Formula/Notes |
|---|---|---|
| **distinct-1/2** | Lexical diversity | Unique unigrams/bigrams ÷ total unigrams/bigrams. 1.0 = no repeats. |
| **repetition** | N-gram repetition | 1 − distinct-3 (approximation of looping behavior). |
| **randomness** | Sampling entropy normalized | Mean step entropy ÷ ln(vocab_size). 0 = greedy, 1 = uniform random. |
| **coherence** | Fluency + local continuity | Blend of (1 − log(perplexity)/log(1000)) and bigram overlap with reference corpus. |
| **relevance** | Topic/prompt alignment | 0.5×prompt content-word recall + 0.5×topic keyword coverage (AI domain). |
| **factuality** | Truthfulness vs KB | Claims matching known facts minus penalties for contradictions and unverifiable assertive claims. |
| **safety** | Harmlessness | 1 − 0.5×(detector flag count), penalizing unsafe/bias/injection/hallucination signals. |
| **helpfulness** | Composite utility | 0.30×relevance + 0.25×coherence + 0.20×factuality + 0.25×safety. |

> **Note:** These are classroom heuristics designed for interpretability and pedagogy—not production-grade evaluators.

## Safety & Alignment

The laboratory includes a controlled safety test suite covering five categories:

| Category | Examples | Detection Approach |
|---|---|---|
| **Hallucination** | Fabricated citations, unverifiable studies, made-up numbers | Citation/link patterns + numeric claims not grounded in KB/prompt. |
| **Misinformation** | False claims contradicting known facts (vaccines/autism, flat earth) | Keyword matching against curated `false_claims` in knowledge base. |
| **Bias** | Stereotypes, slurs, exclusionary claims | Transparent regex patterns for group generalizations and harmful language. |
| **Unsafe Requests** | Violence, self-harm, weapons, fraud, harassment | High-signal patterns targeting actionable harmful instructions. |
| **Prompt Injection** | "Ignore previous instructions", delimiter attacks, indirect injection | Common jailbreak/override patterns (instruction overrides, system prompt extraction). |

**SafetyGuard** demonstrates a two-stage defense (prompt screening + output filtering) and produces safe refusals when harmful content is detected. The detectors are intentionally simple and auditable for classroom discussion.

## Backends

| Backend | Dependencies | Speed | Use Case |
|---|---|---|---|
| **ngram** (default) | None (stdlib only) | Very fast | Fully offline, deterministic, perfect for learning decoding mechanics. Trained on `data/corpus.txt`. |
| **hf** | `torch`, `transformers` | Slower (GPU optional) | Uses pretrained models (default `gpt2`). Great for comparing to real transformers. |
| **tiny** | `torch` | Fast | Self-trained miniature GPT from `lab.train_tiny`. Demonstrates end-to-end training + decoding with the same codebase. |

`auto` mode intelligently falls back (hf → tiny → ngram) if dependencies or checkpoints are missing.

## Educational Objectives

This laboratory supports learning outcomes including:

1. **Understand sampling mechanics**: Implement and compare greedy/temperature/top-k/top-p/min-p from first principles
2. **Analyze trade-offs**: Observe how diversity, coherence, repetition, and factuality compete as functions of sampling parameters
3. **Quantify generation quality**: Apply interpretable metrics and reason about their strengths/weaknesses
4. **Explore test-time compute**: Investigate best-of-N, token budgets, and self-consistency scaling laws
5. **Practice AI safety**: Run controlled red-team tests, analyze failure modes, and evaluate simple defenses
6. **Bridge theory and practice**: Use the same decoding code across n-gram, pretrained, and self-trained models
7. **Emphasize reproducibility**: Log seeds, configs, and artifacts for scientific experimentation

## Project Structure

```text
Transformer-Text-Generation-and-Sampling-Laboratory/
├── lab/                    # Core library
│   ├── __init__.py         # Package exports and docs
│   ├── backend.py          # Model backends (ngram/hf/tiny)
│   ├── decoding.py         # Sampling strategies + generation logic
│   ├── metrics.py          # Evaluation metrics and rubric analysis
│   ├── safety.py           # Test cases, detectors, SafetyGuard
│   ├── compute.py          # Test-time compute experiments
│   ├── visualize.py        # Optional matplotlib visualizations
│   ├── utils.py            # Helpers (tables, JSON, paths)
│   └── train_tiny.py       # Tiny GPT training implementation
├── data/                   # Bundled data
│   ├── corpus.txt          # Training corpus for n-gram/tiny models
│   └── knowledge.json      # KB for factuality/misinformation checks
├── tests/                  # Unit tests
│   ├── test_decoding.py    # Decoding/filter correctness
│   ├── test_metrics.py     # Metric behavior
│   └── test_safety.py      # Safety detectors and guards
├── docs/                   # Documentation
│   └── Building_the_LLM_Lab.pdf
├── results/                # Auto-generated outputs (JSON, PNG, MD reports)
├── run_lab.py              # CLI entry point (all commands)
├── requirements.txt        # Optional dependencies
└── README.md               # This file
```

## Contributing

Contributions are welcome! This project is designed for educational use, so clarity and pedagogical value take priority over optimization.

Areas of interest:
- Additional decoding strategies (contrastive search, typical decoding, eta sampling)
- More robust/educational evaluation metrics
- Extended safety test cases and detectors
- Improved visualizations and report templates
- Additional backends or tokenizers

Please maintain consistency with existing code style, add docstrings, and keep implementations readable for students.

## Citation

If you use this laboratory in teaching or research, please cite:

```bibtex
@software{transformer_text_gen_lab,
  title={Transformer Text Generation and Sampling Laboratory},
  author={Dhonith12},
  year={2024},
  url={https://github.com/dhonith12/Transformer-Text-Generation-and-Sampling-Laboratory}
}
```

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Built for learning. Designed to be transparent. Run anywhere with Python.**