"""Small shared helpers: tables, file naming, JSON persistence."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def slugify(text: str, limit: int = 40) -> str:
    """Turn an arbitrary string into a short filesystem-safe slug."""
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
    return text[:limit].rstrip("-") or "item"


def format_table(headers: Sequence[str], rows: Iterable[Sequence[Any]],
                 max_width: int = 46) -> str:
    """Render a simple left-aligned ASCII table.

    Cells longer than ``max_width`` are truncated with an ellipsis so the
    console stays readable when printing whole generated paragraphs.
    """
    def cell(value: Any) -> str:
        text = "" if value is None else str(value)
        text = text.replace("\n", " ")
        if len(text) > max_width:
            text = text[: max_width - 1] + "\u2026"
        return text

    body = [[cell(c) for c in row] for row in rows]
    head = [cell(h) for h in headers]
    widths = [len(h) for h in head]
    for row in body:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def line(cells: Sequence[str]) -> str:
        return " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells))

    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([line(head), sep] + [line(r) for r in body])


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def save_json(name: str, payload: Any) -> Path:
    """Persist *payload* under ``results/`` and return the file path."""
    ensure_results_dir()
    path = RESULTS_DIR / (slugify(name) + ".json")
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))
