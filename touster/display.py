"""Notebook-native output — HTML via IPython.display, not a terminal library.

The primary surface is a Jupyter/Colab cell, not a terminal, so render real
HTML instead of ANSI/rich styling. Falls back to plain print when there's no
live IPython display (e.g. running under pytest or a bare script) so nothing
silently disappears outside a notebook.
"""
from __future__ import annotations

import html as _html

_BRAND = "#F59E0B"      # amber
_SUCCESS = "#22C55E"    # green
_WARNING = "#F97316"    # orange
_ERROR = "#EF4444"      # red
_DIM = "#78716C"        # stone


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


def _show(html_body: str, plain: str) -> None:
    if _in_notebook():
        from IPython.display import HTML, display
        display(HTML(html_body))
    else:
        print(plain)


def step(n: int, total: int, title: str) -> None:
    _show(
        f'<div style="border-bottom:2px solid {_BRAND};padding:4px 0;margin:8px 0;">'
        f'<b style="color:{_BRAND};">Step {n}/{total}</b> — {_html.escape(title)}</div>',
        f"== Step {n}/{total} — {title} ==",
    )


def success(msg: str) -> None:
    _show(f'<div style="color:{_SUCCESS};">&#10003; {_html.escape(msg)}</div>', f"[OK] {msg}")


def warning(msg: str) -> None:
    _show(f'<div style="color:{_WARNING};">&#9888; {_html.escape(msg)}</div>', f"[WARN] {msg}")


def error(msg: str) -> None:
    _show(f'<div style="color:{_ERROR};">&#10007; {_html.escape(msg)}</div>', f"[ERROR] {msg}")


def table(headers: list[str], rows: list[list], title: str = "") -> None:
    """Render a simple HTML table (or an aligned plain-text one outside a notebook)."""
    if _in_notebook():
        head = "".join(f"<th style='text-align:left;padding:4px 10px;'>{_html.escape(str(h))}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td style='padding:4px 10px;'>{_html.escape(str(c))}</td>" for c in row) + "</tr>"
            for row in rows
        )
        title_html = f'<b style="color:{_BRAND};">{_html.escape(title)}</b>' if title else ""
        html_body = (
            f'{title_html}<table style="border-collapse:collapse;">'
            f'<tr style="border-bottom:2px solid {_DIM};">{head}</tr>{body}</table>'
        )
        from IPython.display import HTML, display
        display(HTML(html_body))
        return

    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h)) for i, h in enumerate(headers)]
    lines = [title] if title else []
    lines.append("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    lines.append("-" * (sum(widths) + 2 * (len(widths) - 1)))
    for row in rows:
        lines.append("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
    print("\n".join(lines))
