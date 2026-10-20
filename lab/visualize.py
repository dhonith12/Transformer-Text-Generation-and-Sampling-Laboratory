"""Optional matplotlib charts (the lab works fine without them).

Every function returns the saved path, or ``None`` when matplotlib is not
installed / no renderer is available.  Charts are written into ``results/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from .utils import ensure_results_dir, slugify


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:          # pragma: no cover - optional dependency
        return None


def _finish(fig, path: Path):
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=130)
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return path


def strategy_bars(rows: Sequence[dict], out_dir: Optional[Path] = None,
                  metric: str = "coherence") -> Optional[Path]:
    """Grouped bar chart: every strategy vs. the selected metric."""
    plt = _plt()
    if plt is None or not rows:
        return None
    out_dir = Path(out_dir) if out_dir else ensure_results_dir()
    labels = [r["label"] for r in rows]
    metrics = ["distinct_2", "repetition", "randomness", "coherence",
               "relevance", "safety"]
    present = [m for m in metrics if m in rows[0]]

    fig, ax = plt.subplots(figsize=(11, 4.6))
    width = 0.8 / len(present)
    xs = range(len(labels))
    for i, m in enumerate(present):
        ax.bar([x + i * width for x in xs], [r[m] for r in rows],
               width=width, label=m)
    ax.set_xticks([x + 0.8 / 2 for x in xs])
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (0-1)")
    ax.set_title("Decoding strategies compared")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, out_dir / "strategy_comparison.png")


def entropy_trace(result, out_dir: Optional[Path] = None) -> Optional[Path]:
    """Sampling entropy per step — shows how 'decisive' the model is."""
    plt = _plt()
    if plt is None or not result.steps:
        return None
    out_dir = Path(out_dir) if out_dir else ensure_results_dir()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot([s.entropy for s in result.steps], marker="o", ms=3)
    ax.set_xlabel("generation step")
    ax.set_ylabel("entropy (nats)")
    ax.set_title(f"Sampling entropy over time — {result.config.label()}")
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir / f"entropy_{slugify(result.config.label())}.png")


def sweep_plot(rows: Sequence[dict], param: str,
               out_dir: Optional[Path] = None) -> Optional[Path]:
    """Quality/diversity trade-off while sweeping one sampling parameter."""
    plt = _plt()
    if plt is None or not rows:
        return None
    out_dir = Path(out_dir) if out_dir else ensure_results_dir()
    xs = [r["value"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    for metric, style in (("distinct_2", "o-"), ("coherence", "s-"),
                          ("repetition", "^-"), ("relevance", "d-")):
        if metric in rows[0]:
            ax.plot(xs, [r[metric] for r in rows], style, label=metric, ms=4)
    ax.set_xlabel(param)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score (0-1)")
    ax.set_title(f"Sweeping {param}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir / f"sweep_{slugify(param)}.png")


def budget_plot(rows: Sequence[dict], out_dir: Optional[Path] = None
                ) -> Optional[Path]:
    """Test-time compute: quality vs. token budget."""
    plt = _plt()
    if plt is None or not rows:
        return None
    out_dir = Path(out_dir) if out_dir else ensure_results_dir()
    xs = [r["max_new_tokens"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 4))
    for metric in ("relevance", "coherence", "repetition", "distinct_2"):
        if metric in rows[0]:
            ax.plot(xs, [r[metric] for r in rows], "o-", label=metric, ms=4)
    ax.set_xlabel("max new tokens (compute budget)")
    ax.set_ylabel("score (0-1)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test-time compute: generating longer vs. quality")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _finish(fig, out_dir / "test_time_compute.png")


def safety_bars(reports: Sequence[dict], out_dir: Optional[Path] = None
                ) -> Optional[Path]:
    """Pass rates per safety category, guarded vs unguarded."""
    plt = _plt()
    if plt is None or not reports:
        return None
    out_dir = Path(out_dir) if out_dir else ensure_results_dir()
    categories = sorted({c for r in reports
                         for c in r.get("per_category", {})})
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(reports)
    xs = range(len(categories))
    for i, rep in enumerate(reports):
        values = [rep.get("per_category", {}).get(c, 0.0) for c in categories]
        ax.bar([x + i * width for x in xs], values, width=width,
               label=rep.get("mode", "?"))
    ax.set_xticks([x + 0.8 / 2 for x in xs])
    ax.set_xticklabels(categories, rotation=20)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("pass rate")
    ax.set_title("Safety suite: pass rate per category")
    ax.legend(fontsize=8)
    return _finish(fig, out_dir / "safety_pass_rates.png")
