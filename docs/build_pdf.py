#!/usr/bin/env python
"""Regenerate docs/Building_the_LLM_Lab.pdf from the plain-text source.

Uses only the Python standard library (no reportlab / fpdf / LaTeX), so the
guide can always be rebuilt on a bare interpreter:

    python docs/build_pdf.py

Input : docs/Building_the_LLM_Lab.txt
Output: docs/Building_the_LLM_Lab.pdf

Supported lightweight markup in the source file:
  * a line of only "=" characters under text      -> section heading
  * a line of only "-" characters under text      -> sub-section heading
  * a line starting with 4 spaces                 -> code / command block
  * "Title\n====="                                -> document title page
Everything else is treated as a normal paragraph.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "Building_the_LLM_Lab.txt"
OUT = HERE / "Building_the_LLM_Lab.pdf"

# A4 page geometry (points). 1 inch = 72 pt.
PAGE_W, PAGE_H = 595.28, 841.89
MARGIN_L, MARGIN_R, MARGIN_T, MARGIN_B = 64.0, 64.0, 72.0, 64.0
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

# Font metrics (Helvetica / Courier widths are close enough for wrapping).
AVG_CHAR_W = 0.50          # of font size, for the body font
MONO_CHAR_W = 0.60         # of font size, for Courier

# Styles: (font, size, leading, space_before, space_after)
BODY = ("F1", 10.5, 15.0, 0.0, 6.0)
TITLE = ("F2", 22.0, 28.0, 0.0, 14.0)
H2 = ("F2", 15.0, 20.0, 14.0, 6.0)
H3 = ("F2", 12.0, 17.0, 10.0, 4.0)
CODE = ("F3", 9.0, 13.0, 4.0, 6.0)
LIST = ("F1", 10.5, 15.0, 0.0, 3.0)


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------
_TRANSLATE = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "--", "\u2026": "...", "\u00a0": " ",
}


def _clean(text: str) -> str:
    """Normalise typographic characters the base-14 fonts render poorly."""
    for bad, good in _TRANSLATE.items():
        text = text.replace(bad, good)
    return text


def _esc(text: str) -> str:
    """Escape a string for a PDF literal string."""
    text = _clean(text)
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def wrap(text: str, style, indent: float = 0.0) -> list[str]:
    """Wrap ``text`` to the content width for the given style."""
    font, size, *_ = style
    char_w = (MONO_CHAR_W if font == "F3" else AVG_CHAR_W) * size
    width = max(10, int((CONTENT_W - indent) / char_w))
    return textwrap.wrap(text, width=width) or [""]


def parse_source(raw: str):
    """Turn the source text into a list of (style, text, indent) runs."""
    lines = raw.replace("\r\n", "\n").split("\n")
    runs = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        underline = lines[i + 1] if i + 1 < n else ""

        if stripped and set(underline.strip()) == {"="} and len(underline.strip()) >= 3:
            style = TITLE if i == 0 else H2
            runs.append((style, stripped, 0.0))
            i += 2
            continue
        if stripped and set(underline.strip()) == {"-"} and len(underline.strip()) >= 3:
            runs.append((H3, stripped, 0.0))
            i += 2
            continue
        if line.startswith("    ") or line.startswith("\t"):
            # collect a contiguous code block
            block = []
            while i < n and (lines[i].startswith("    ") or lines[i].startswith("\t")):
                block.append(lines[i].replace("\t", "    ")[4:])
                i += 1
            for bl in block:
                runs.append((CODE, bl, 0.0))
            continue
        m = re.match(r"^(\s*)[*\-]\s+(.*)$", line)
        if m:
            pad = min(len(m.group(1)), 8)
            runs.append((LIST, "- " + m.group(2), 10.0 + pad))
            i += 1
            continue
        if not stripped:
            runs.append((None, "", 0.0))
            i += 1
            continue
        runs.append((BODY, stripped, 0.0))
        i += 1
    return runs


# --------------------------------------------------------------------------
# Layout / PDF emission
# --------------------------------------------------------------------------
def build_pdf(runs) -> bytes:
    """Render runs into page content streams."""
    top_y = PAGE_H - MARGIN_T
    bottom_y = MARGIN_B

    pages: list[list[str]] = []
    stream: list[str] = []
    y = top_y

    def new_page():
        nonlocal stream, y
        if stream:
            pages.append(stream)
        stream = []
        y = top_y

    def put(style, text, indent):
        nonlocal y
        font, size, leading, before, after = style
        y -= before
        for segment in wrap(text, style, indent):
            if y - leading < bottom_y:
                new_page()
            stream.append(f"BT /{font} {size} Tf {MARGIN_L + indent:.2f} {y:.2f} Td "
                          f"({_esc(segment)}) Tj ET")
            y -= leading
        y -= after

    for style, text, indent in runs:
        if style is None:
            y -= BODY[2] * 0.5
            continue
        if style is TITLE and pages:
            new_page()
        put(style, text, indent)

    if stream:
        pages.append(stream)
    if not pages:
        pages = [[""]]

    # ---- assemble the document ------------------------------------------
    objects: list[bytes] = []

    def add(obj: bytes) -> int:
        objects.append(obj)
        return len(objects)

    # Reserve object numbers.
    font_bold = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
    font_norm = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    font_mono = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

    content_ids = []
    for page_stream in pages:
        data = "\n".join(page_stream).encode("latin-1", "replace")
        content_ids.append(add(b"<< /Length %d >>\nstream\n" % len(data) + data
                               + b"\nendstream"))

    # Resources object.
    resources = add(
        b"<< /Font << /F1 %d 0 R /F2 %d 0 R /F3 %d 0 R >> >>"
        % (font_norm, font_bold, font_mono)
    )

    font_bold, font_norm, font_mono = 1, 2, 3
    page_ids = []
    for content_id in content_ids:
        page_obj = (
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.2f %.2f] "
            b"/Resources %d 0 R /Contents %d 0 R >>"
            % (0, PAGE_W, PAGE_H, resources, content_id)
        )
        page_ids.append(add(page_obj))

    pages_id = len(objects) + 1
    kids = b" ".join(b"%d 0 R" % pid for pid in page_ids)
    add(b"<< /Type /Pages /Count %d /Kids [%s] >>" % (len(page_ids), kids))

    # Patch the parent reference now that pages_id is known.
    for idx, pid in enumerate(page_ids):
        objects[pid - 1] = objects[pid - 1].replace(b"Parent %d 0 R" % 0,
                                                    b"Parent %d 0 R" % pages_id)

    catalog_id = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages_id)

    # ---- write the file --------------------------------------------------
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for num, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % num + obj + b"\nendobj\n"
    xref_pos = len(out)
    out += b"xref\n0 %d\n" % (len(objects) + 1)
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += b"%010d 00000 n \n" % off
    out += (b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n"
            % (len(objects) + 1, catalog_id, xref_pos))
    return bytes(out)


def main() -> int:
    if not SRC.exists():
        raise SystemExit(f"source not found: {SRC}")
    runs = parse_source(SRC.read_text(encoding="utf-8"))
    pdf = build_pdf(runs)
    OUT.write_bytes(pdf)
    print(f"wrote {OUT} ({len(pdf)} bytes, {len(runs)} blocks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
