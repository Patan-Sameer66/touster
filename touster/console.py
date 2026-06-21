from __future__ import annotations

import sys

from rich.console import Console
from rich.theme import Theme

# Touster brand palette — warm amber toast tones for screenshots
_THEME = Theme(
    {
        "touster.brand": "bold #F59E0B",        # amber-400
        "touster.accent": "bold #FBBF24",       # amber-300
        "touster.dim": "#78716C",               # stone-500
        "touster.success": "bold #22C55E",      # green-500
        "touster.warning": "bold #F97316",      # orange-500
        "touster.error": "bold #EF4444",        # red-500
        "touster.highlight": "bold #FFFFFF on #B45309",  # white on amber-700
        "touster.table.header": "bold #F59E0B",
        "touster.top": "bold #FBBF24 on #292524",  # brightened top-3 rows
        "touster.step": "bold #A78BFA",         # violet-400 — step labels
        "touster.code": "#86EFAC",              # green-300 — inline code
        "touster.model": "bold #38BDF8",        # sky-400 — model names
    }
)

console = Console(theme=_THEME)


def _safe_char(char: str, fallback: str) -> str:
    """Return char if the stdout encoding can represent it, else fallback."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        char.encode(enc)
        return char
    except (UnicodeEncodeError, LookupError):
        return fallback


_CHECK = _safe_char("✓", "OK")
_WARN_SYM = _safe_char("⚠", "!!")
_CROSS = _safe_char("✗", "XX")


def print_step(n: int, total: int, title: str) -> None:
    console.rule(
        f"[touster.step]Step {n}/{total}[/touster.step] [touster.brand]{title}[/touster.brand]",
        style="touster.dim",
    )


def print_success(msg: str) -> None:
    console.print(f"[touster.success]{_CHECK}[/touster.success] {msg}")


def print_warning(msg: str) -> None:
    console.print(f"[touster.warning]{_WARN_SYM}[/touster.warning]  {msg}")


def print_error(msg: str) -> None:
    console.print(f"[touster.error]{_CROSS}[/touster.error] {msg}")
