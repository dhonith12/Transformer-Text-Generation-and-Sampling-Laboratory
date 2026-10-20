#!/usr/bin/env python
"""Transformer Text Generation and Sampling Laboratory — command line driver.

Examples
--------
    python run_lab.py compare                                   # default prompt, all strategies
    python run_lab.py compare --prompt "Once upon a time" --tokens 60
    python run_lab.py sweep --param top_p --values 0.5,0.8,0.9,0.95,0.99
    python run_lab.py analyze --strategy topp --top-p 0.9
    python run_lab.py safety            # unguarded vs guarded comparison
    python run_lab.py compute --n 8
    python run_lab.py report            # writes results/report.md (+ charts)
    python run_lab.py chat              # free exploration
    python run_lab.py train --steps 300 # optional: train the tiny transformer

Backends: --backend auto|ngram|hf|tiny  (default auto: hf -> tiny -> ngram)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from lab.backend import DEFAULT_CHECKPOINT, DEFAULT_CORPUS, resolve_backend
from lab.compute import (best_of_n, print_budget_curve, print_n_curve,
                         quality_vs_n, quality_vs_tokens, self_consistency)
from lab.decoding import STRATEGIES, DecodingConfig, generate
from lab.metrics import ReferenceIndex, aggregate_reports, analyze
from lab.safety import load_knowledge, run_safety_suite
from lab.utils import ensure_results_dir, format_table, save_json, slugify

DEFAULT_PROMPT = "The future of artificial intelligence is"


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------
def add_common_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("backend / environment")
    g.add_argument("--backend", default="auto",
                   choices=["auto", "ngram", "hf", "tiny"],
                   help="which language model to use (default: auto)")
    g.add_argument("--model", default="gpt2",
                   help="HuggingFace model name for --backend hf")
    g.add_argument("--corpus", type=Path, default=None,
                   help=f"corpus for the n-gram backend (default {DEFAULT_CORPUS})")
    g.add_argument("--order", type=int, default=3,
                   help="n-gram order for --backend ngram (default 3)")
    g.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT,
                   help="checkpoint for --backend tiny")


def add_decoding_args(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("decoding")
    g.add_argument("--strategy", default="temperature", choices=list(STRATEGIES))
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--min-p", type=float, default=0.05)
    g.add_argument("--repetition-penalty", type=float, default=1.0)
    g.add_argument("--tokens", type=int, default=40,
                   help="max new tokens (default 40)")
    g.add_argument("--seed", type=int, default=42)


def config_from_args(args: argparse.Namespace,
                     **overrides) -> DecodingConfig:
    values = dict(strategy=args.strategy, temperature=args.temperature,
                  top_k=args.top_k, top_p=args.top_p, min_p=args.min_p,
                  repetition_penalty=args.repetition_penalty,
                  max_new_tokens=args.tokens, seed=args.seed)
    values.update(overrides)
    return DecodingConfig(**values)


def make_backend(args: argparse.Namespace):
    return resolve_backend(args.backend, model=args.model,
                           corpus=args.corpus, order=args.order,
                           checkpoint=args.checkpoint)


def make_reference(corpus_path: Optional[Path]) -> ReferenceIndex:
    path = Path(corpus_path) if corpus_path else DEFAULT_CORPUS
    return ReferenceIndex(path.read_text(encoding="utf-8"))


def print_generations(results) -> None:
    rows = [[r.config.label(), r.num_tokens, r.continuation] for r in results]
    print(format_table(["strategy", "tok", "generation (continuation)"], rows,
                       max_width=70))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_compare(args: argparse.Namespace) -> int:
    backend = make_backend(args)
    kb = load_knowledge()
    reference = make_reference(args.corpus)
    prompt = args.prompt or DEFAULT_PROMPT

    print(f"backend   : {backend.describe()}")
    print(f"prompt    : {prompt!r}\n")

    configs = [DecodingConfig(strategy="greedy", max_new_tokens=args.tokens,
                              seed=args.seed),
               DecodingConfig(strategy="random", max_new_tokens=args.tokens,
                              seed=args.seed),
               DecodingConfig(strategy="temperature", temperature=0.6,
                              max_new_tokens=args.tokens, seed=args.seed),
               DecodingConfig(strategy="temperature", temperature=1.2,
                              max_new_tokens=args.tokens, seed=args.seed),
               DecodingConfig(strategy="topk", top_k=20,
                              max_new_tokens=args.tokens, seed=args.seed),
               DecodingConfig(strategy="topk", top_k=5,
                              max_new_tokens=args.tokens, seed=args.seed),
               DecodingConfig(strategy="topp", top_p=0.90,
                              max_new_tokens=args.tokens, seed=args.seed),
               DecodingConfig(strategy="topp", top_p=0.99,
                              max_new_tokens=args.tokens, seed=args.seed),
               DecodingConfig(strategy="topk_topp", top_k=50, top_p=0.95,
                              max_new_tokens=args.tokens, seed=args.seed),
               DecodingConfig(strategy="min_p", min_p=0.05,
                              max_new_tokens=args.tokens, seed=args.seed)]

    results, reports, rows = [], [], []
    for cfg in configs:
        result = generate(backend, prompt, cfg)
        report = analyze(result, backend, kb, reference)
        results.append(result)
        reports.append(report)
        row = {"label": cfg.label()}
        row.update({k: v for k, v in report.metrics.items()
                    if isinstance(v, (int, float))})
        rows.append(row)

    print_generations(results)

    metric_rows = [[r["label"], r.get("distinct_2"), r.get("repetition"),
                    r.get("randomness"), r.get("coherence"),
                    r.get("relevance"), r.get("factuality"),
                    r.get("safety"), r.get("helpfulness")]
                   for r in rows]
    print("\n=== Rubric metrics (higher is better unless stated) ===")
    print(format_table(["strategy", "diversity", "repetition", "randomness",
                        "coherence", "relevance", "factuality", "safety",
                        "helpfulness"], metric_rows, max_width=30))

    totals = aggregate_reports(reports)
    print("\nmacro-average over all strategies:",
          {k: totals[k] for k in
           ("distinct_2", "repetition", "randomness", "coherence",
            "relevance", "safety", "helpfulness") if k in totals})

    if not args.no_json:
        path = save_json(f"compare_{slugify(prompt)}",
                         {"backend": backend.describe(), "prompt": prompt,
                          "results": [r.to_dict() for r in results],
                          "reports": [r.to_dict() for r in reports]})
        print(f"saved -> {path}")

    if not args.no_plots:
        from lab.visualize import strategy_bars
        chart = strategy_bars(rows)
        if chart:
            print(f"chart  -> {chart}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    backend = make_backend(args)
    reference = make_reference(args.corpus)
    kb = load_knowledge()
    prompt = args.prompt or DEFAULT_PROMPT
    param = args.param

    values: List[float] = [float(v) for v in args.values.split(",")]
    rows = []
    print(f"backend: {backend.describe()}   prompt: {prompt!r}")
    for value in values:
        overrides = {"max_new_tokens": args.tokens, "seed": args.seed}
        if param in ("temperature", "top_p", "min_p"):
            overrides["strategy"] = {"temperature": "temperature",
                                     "top_p": "topp",
                                     "min_p": "min_p"}[param]
            overrides[param] = value
        elif param == "top_k":
            overrides["strategy"] = "topk"
            overrides["top_k"] = int(value)
        elif param == "repetition_penalty":
            overrides["strategy"] = "temperature"
            overrides["repetition_penalty"] = value
        cfg = DecodingConfig(**overrides)
        result = generate(backend, prompt, cfg)
        metrics = analyze(result, backend, kb, reference).metrics
        rows.append({"value": value, "label": cfg.label(),
                     "continuation": result.continuation, **metrics})
        print(f"  {param}={value:<6g} | d-2 {metrics['distinct_2']:.3f}  "
              f"rep {metrics['repetition']:.3f}  coh {metrics['coherence']:.3f}  "
              f"rel {metrics['relevance']:.3f} | {result.continuation[:60]}")

    print("\n" + format_table(
        [param, "distinct-1", "distinct-2", "repetition", "randomness",
         "coherence", "relevance", "helpfulness"],
        [[r["value"], r["distinct_1"], r["distinct_2"], r["repetition"],
          r["randomness"], r["coherence"], r["relevance"],
          r["helpfulness"]] for r in rows]))

    if not args.no_json:
        path = save_json(f"sweep_{slugify(param)}", {"param": param,
                                                     "rows": rows})
        print(f"saved -> {path}")
    if not args.no_plots:
        from lab.visualize import sweep_plot
        chart = sweep_plot(rows, param)
        if chart:
            print(f"chart -> {chart}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    backend = make_backend(args)
    reference = make_reference(args.corpus)
    kb = load_knowledge()
    prompt = args.prompt or DEFAULT_PROMPT
    cfg = config_from_args(args)

    result = generate(backend, prompt, cfg)
    report = analyze(result, backend, kb, reference)

    print(f"backend   : {backend.describe()}")
    print(f"decoding  : {cfg.label()}   seed={cfg.seed}")
    print(f"prompt    : {prompt!r}")
    print(f"generated : {result.continuation!r}\n")
    print(format_table(["metric", "value", "meaning"], report.rows(),
                       max_width=70))

    if not args.no_json:
        path = save_json(f"analyze_{slugify(prompt)}_{slugify(cfg.label())}",
                         {"result": result.to_dict(),
                          "report": report.to_dict()})
        print(f"\nsaved -> {path}")
    if not args.no_plots:
        from lab.visualize import entropy_trace
        chart = entropy_trace(result)
        if chart:
            print(f"chart -> {chart}")
    return 0


def cmd_safety(args: argparse.Namespace) -> int:
    backend = make_backend(args)
    kb = load_knowledge()
    cfg = config_from_args(args, strategy=args.strategy,
                           temperature=args.temperature)

    unguarded = run_safety_suite(backend, cfg.with_(max_new_tokens=args.tokens),
                                 kb, guarded=False)
    guarded = run_safety_suite(backend, cfg.with_(max_new_tokens=args.tokens),
                               kb, guarded=True)

    print("\n=== Summary: does the defence help? ===")
    print(format_table(["mode", "pass rate", "per-category pass rates"],
                       [["unguarded", f"{unguarded.pass_rate:.0%}",
                         str(unguarded.per_category())],
                        ["guarded", f"{guarded.pass_rate:.0%}",
                         str(guarded.per_category())]]))

    if not args.no_json:
        path = save_json("safety_suite",
                         {"backend": backend.describe(), "decoding": cfg.label(),
                          "unguarded": unguarded.to_dict(),
                          "guarded": guarded.to_dict()})
        print(f"saved -> {path}")
    if not args.no_plots:
        from lab.visualize import safety_bars
        chart = safety_bars([unguarded.to_dict(), guarded.to_dict()])
        if chart:
            print(f"chart -> {chart}")
    return 0


def cmd_compute(args: argparse.Namespace) -> int:
    backend = make_backend(args)
    reference = make_reference(args.corpus)
    kb = load_knowledge()
    prompt = args.prompt or "Explain top-p sampling in one sentence."
    cfg = config_from_args(args)

    print(f"backend : {backend.describe()}")
    print(f"prompt  : {prompt!r}")
    print(f"decoding: {cfg.label()}")

    # --- best-of-N -------------------------------------------------------
    n_rows = quality_vs_n(backend, prompt, cfg.with_(max_new_tokens=args.tokens),
                          ns=[int(n) for n in args.n_values.split(",")],
                          kb=kb, reference=reference)
    print_n_curve(n_rows)

    print("\nSingle sample (N=1):")
    single = best_of_n(backend, prompt, cfg.with_(max_new_tokens=args.tokens), n=1,
                       kb=kb, reference=reference)
    print(f"  score {single.best.composite} | {single.best.text}")

    best = best_of_n(backend, prompt, cfg.with_(max_new_tokens=args.tokens),
                     n=max(int(n) for n in args.n_values.split(",")),
                     kb=kb, reference=reference)
    print(f"\nBest of {best.n}:")
    print(f"  score {best.best.composite} | {best.best.text}")
    print("\nAll candidates:")
    print(best.table())

    # --- token budget ----------------------------------------------------
    budgets = [int(b) for b in args.budgets.split(",")]
    budget_rows = quality_vs_tokens(backend, prompt, cfg, budgets, reference)
    print_budget_curve(budget_rows)

    # --- self consistency ------------------------------------------------
    if args.vote:
        vote = self_consistency(backend, args.vote, cfg.with_(max_new_tokens=24),
                                n=args.vote_n)
        print(f"\n=== Self-consistency vote over {args.vote_n} samples ===")
        print(f"prompt: {args.vote!r}")
        print(vote.table())
        print(f"majority answer: {vote.answer!r} "
              f"(agreement {vote.agreement:.0%})")

    if not args.no_json:
        path = save_json("compute",
                         {"backend": backend.describe(), "prompt": prompt,
                          "decoding": cfg.label(),
                          "best_of_n": [r.as_list() for r in n_rows],
                          "token_budget": [r.as_list() for r in budget_rows],
                          "best_candidate": best.best.to_dict()})
        print(f"\nsaved -> {path}")
    if not args.no_plots:
        from lab.visualize import budget_plot
        chart = budget_plot([dict(zip(
            ("max_new_tokens", "relevance", "coherence", "repetition",
             "distinct_2", "length"), r.as_list())) for r in budget_rows])
        if chart:
            print(f"chart -> {chart}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Run every experiment and write a markdown lab report."""
    backend = make_backend(args)
    kb = load_knowledge()
    reference = make_reference(args.corpus)
    prompt = args.prompt or DEFAULT_PROMPT
    out_dir = ensure_results_dir()
    lines: List[str] = [
        "# Transformer Text Generation and Sampling Laboratory — Report",
        "",
        f"* backend: `{backend.describe()}`",
        f"* prompt: `{prompt}`",
        f"* decoding budget: {args.tokens} tokens, seed {args.seed}",
        "",
    ]
    charts: List[Path] = []

    # 1. comparison ------------------------------------------------------
    results, rows = [], []
    configs = [
        DecodingConfig(strategy="greedy", max_new_tokens=args.tokens, seed=args.seed),
        DecodingConfig(strategy="random", max_new_tokens=args.tokens, seed=args.seed),
        DecodingConfig(strategy="temperature", temperature=0.6,
                       max_new_tokens=args.tokens, seed=args.seed),
        DecodingConfig(strategy="temperature", temperature=1.2,
                       max_new_tokens=args.tokens, seed=args.seed),
        DecodingConfig(strategy="topk", top_k=5, max_new_tokens=args.tokens,
                       seed=args.seed),
        DecodingConfig(strategy="topp", top_p=0.90, max_new_tokens=args.tokens,
                       seed=args.seed),
        DecodingConfig(strategy="topk_topp", top_k=50, top_p=0.95,
                       max_new_tokens=args.tokens, seed=args.seed),
        DecodingConfig(strategy="min_p", min_p=0.05, max_new_tokens=args.tokens,
                       seed=args.seed)]
    print("1/4 comparing decoding strategies ...")
    reports = []
    for cfg in configs:
        result = generate(backend, prompt, cfg)
        report = analyze(result, backend, kb, reference)
        results.append(result)
        reports.append(report)
        row = {"label": cfg.label()}
        row.update({k: v for k, v in report.metrics.items()
                    if isinstance(v, (int, float))})
        rows.append(row)

    lines += ["## 1. Decoding strategies compared", ""]
    lines.append("| strategy | generation |")
    lines.append("|---|---|")
    for r in results:
        lines.append(f"| `{r.config.label()}` | {r.continuation} |")
    lines += ["", "| strategy | diversity (d-2) | repetition | randomness | "
              "coherence | relevance | factuality | safety | helpfulness |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| `{r['label']}` | {r.get('distinct_2')} | {r.get('repetition')} "
            f"| {r.get('randomness')} | {r.get('coherence')} "
            f"| {r.get('relevance')} | {r.get('factuality')} "
            f"| {r.get('safety')} | {r.get('helpfulness')} |")
    lines += ["", "Interpretation guide:", "",
              "* diversity goes **up** with temperature / top-p / min-p;",
              "* repetition goes **down** with the same knobs;",
              "* randomness (normalised sampling entropy) is 0 for greedy and "
              "1 for uniform random sampling;",
              "* coherence and relevance typically peak at a *moderate* "
              "temperature (0.6-1.0) and collapse for chaotic settings."]

    if not args.no_plots:
        from lab.visualize import entropy_trace, strategy_bars
        for chart in (strategy_bars(rows), entropy_trace(results[0]),
                      entropy_trace(results[-1])):
            if chart:
                charts.append(chart)

    # 2. sweep -----------------------------------------------------------
    print("2/4 sweeping temperature ...")
    sweep_rows = []
    for value in (0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0):
        result = generate(backend, prompt, DecodingConfig(
            strategy="temperature", temperature=value,
            max_new_tokens=args.tokens, seed=args.seed))
        metrics = analyze(result, backend, kb, reference).metrics
        sweep_rows.append({"value": value, **metrics,
                           "continuation": result.continuation})
    lines += ["", "## 2. Temperature sweep", "",
              "| T | distinct-2 | repetition | coherence | relevance | "
              "helpfulness | sample |", "|---|---|---|---|---|---|---|"]
    for r in sweep_rows:
        lines.append(f"| {r['value']} | {r['distinct_2']} | {r['repetition']} "
                     f"| {r['coherence']} | {r['relevance']} "
                     f"| {r['helpfulness']} | {r['continuation'][:70]} |")
    if not args.no_plots:
        from lab.visualize import sweep_plot
        chart = sweep_plot(sweep_rows, "temperature (T)")
        if chart:
            charts.append(chart)

    # 3. test-time compute ------------------------------------------------
    print("3/4 test-time compute experiments ...")
    compute_prompt = "Explain top-p sampling in one sentence."
    n_rows = quality_vs_n(backend, compute_prompt,
                          DecodingConfig(strategy="temperature",
                                         temperature=1.0,
                                         max_new_tokens=args.tokens,
                                         seed=args.seed),
                          ns=(1, 2, 4, 8, 16), kb=kb, reference=reference)
    budget_rows = quality_vs_tokens(
        backend, prompt,
        DecodingConfig(strategy="topk_topp", top_k=50, top_p=0.95,
                       seed=args.seed),
        budgets=(8, 16, 32, 64, 128), reference=reference)
    lines += ["", "## 3. Test-time compute", "",
              f"Best-of-N on the prompt `{compute_prompt}` "
              "(same decoding config, different seeds):", "",
              "| N | best score | mean score | worst score | gain vs N=1 |",
              "|---|---|---|---|---|"]
    for r in n_rows:
        lines.append(f"| {r.n} | {r.best_score} | {r.mean_score} "
                     f"| {r.worst_score} | {r.gain_over_single} |")
    lines += ["", f"Token budget sweep on `{prompt}`:", "",
              "| budget | relevance | coherence | repetition | distinct-2 |",
              "|---|---|---|---|---|"]
    for r in budget_rows:
        lines.append(f"| {r.max_new_tokens} | {r.relevance} | {r.coherence} "
                     f"| {r.repetition} | {r.distinct_2} |")
    lines += ["", "Reading: best-of-N gains flatten quickly, and generating "
              "for too long eventually *hurts* relevance and *raises* "
              "repetition — compute must be spent wisely."]

    # 4. safety ----------------------------------------------------------
    print("4/4 safety suite ...")
    cfg = DecodingConfig(strategy="topk_topp", top_k=50, top_p=0.95,
                         max_new_tokens=args.tokens, seed=args.seed)
    unguarded = run_safety_suite(backend, cfg, kb, guarded=False, verbose=False)
    guarded = run_safety_suite(backend, cfg, kb, guarded=True, verbose=False)
    lines += ["", "## 4. Safety: controlled test cases", "",
              f"Decoding: `{cfg.label()}`", "",
              "| id | category | unguarded | guarded | flags seen |",
              "|---|---|---|---|---|"]
    for u, g in zip(unguarded.outcomes, guarded.outcomes):
        lines.append(f"| {u.case.id} | {u.case.category} "
                     f"| {'PASS' if u.passed else 'FAIL'} "
                     f"| {'PASS' if g.passed else 'FAIL'} "
                     f"| {u.flag_summary()} |")
    lines += ["", f"* unguarded pass rate: **{unguarded.pass_rate:.0%}**",
              f"* guarded pass rate: **{guarded.pass_rate:.0%}**", "",
              "Note how the *guarded* system trades a little fluency for a "
              "large safety gain, and how some benign control cases "
              "(C1, C2) can be over-refused by naive keyword filters."]
    if not args.no_plots:
        from lab.visualize import safety_bars
        chart = safety_bars([unguarded.to_dict(), guarded.to_dict()])
        if chart:
            charts.append(chart)

    if charts:
        lines += ["", "## Charts", ""] + \
                 [f"![{c.stem}]({c.name})" for c in charts]

    lines += ["", "---",
              "*Generated by `python run_lab.py report`. All scores are "
              "classroom heuristics; see README.md for the exact formulas.*"]

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    save_json("report_data",
              {"backend": backend.describe(), "prompt": prompt,
               "comparison": rows, "sweep": sweep_rows,
               "best_of_n": [r.as_list() for r in n_rows],
               "token_budget": [r.as_list() for r in budget_rows],
               "safety": {"unguarded": unguarded.to_dict(),
                          "guarded": guarded.to_dict()}})
    print(f"\nreport -> {report_path}")
    for chart in charts:
        print(f"chart  -> {chart}")
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    backend = make_backend(args)
    kb = load_knowledge()
    reference = make_reference(args.corpus)
    cfg = DecodingConfig(strategy=args.strategy, temperature=args.temperature,
                         top_k=args.top_k, top_p=args.top_p, min_p=args.min_p,
                         max_new_tokens=args.tokens, seed=args.seed)
    print(f"backend : {backend.describe()}")
    print(f"decoding: {cfg.label()}   (type 'quit' to exit, "
          "'<strategy>' to switch, e.g. 'greedy' or 'topp')")
    while True:
        try:
            prompt = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.lower() in {"quit", "exit", "q"}:
            break
        if prompt.lower() in STRATEGIES:
            cfg = cfg.with_(strategy=prompt.lower())
            print(f"[decoding] switched to {cfg.label()}")
            continue
        result = generate(backend, prompt, cfg)
        print(f"model> {result.continuation}")
        metrics = analyze(result, backend, kb, reference).metrics
        print(f"[metrics] d-2={metrics['distinct_2']} rep={metrics['repetition']} "
              f"coh={metrics['coherence']} rel={metrics['relevance']} "
              f"safety={metrics['safety']} tokens={metrics['tokens']}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from lab.train_tiny import train
    train(steps=args.steps, batch_size=args.batch_size, lr=args.lr,
          block_size=args.block_size, n_embd=args.n_embd, n_head=args.n_head,
          n_layer=args.n_layer, seed=args.seed,
          corpus_path=args.corpus, out_path=args.checkpoint)
    print("try it now:  python run_lab.py compare --backend tiny")
    return 0


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_lab.py",
        description=__doc__.split("Examples")[0].strip(),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compare", help="compare decoding strategies side by side")
    add_common_args(p)
    add_decoding_args(p)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--no-json", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("sweep", help="sweep one sampling parameter")
    add_common_args(p)
    add_decoding_args(p)
    p.add_argument("--param", default="temperature",
                   choices=["temperature", "top_k", "top_p", "min_p",
                            "repetition_penalty"])
    p.add_argument("--values", default="0.2,0.6,1.0,1.4,1.8")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--no-json", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("analyze", help="detailed metrics for one generation")
    add_common_args(p)
    add_decoding_args(p)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--no-json", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("safety", help="run the controlled safety test cases")
    add_common_args(p)
    add_decoding_args(p)
    p.add_argument("--no-json", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_safety)

    p = sub.add_parser("compute", help="test-time compute experiments")
    add_common_args(p)
    add_decoding_args(p)
    p.add_argument("--prompt", default="Explain top-p sampling in one sentence.")
    p.add_argument("--n-values", default="1,2,4,8,16")
    p.add_argument("--budgets", default="8,16,32,64,128")
    p.add_argument("--vote", default=None,
                   help="prompt for the self-consistency majority vote")
    p.add_argument("--vote-n", type=int, default=9)
    p.add_argument("--no-json", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_compute)

    p = sub.add_parser("report", help="run everything, write results/report.md")
    add_common_args(p)
    add_decoding_args(p)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("chat", help="interactive exploration loop")
    add_common_args(p)
    add_decoding_args(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("train", help="train the optional tiny transformer")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--corpus", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.set_defaults(func=cmd_train)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n[interrupted]")
        return 130


if __name__ == "__main__":
    sys.exit(main())
