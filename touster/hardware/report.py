"""Rich terminal hardware report — the viral screenshot moment for Touster."""
from __future__ import annotations

from rich.columns import Columns
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from touster.config import HardwareConfig
from touster.console import console
from touster.hardware.catalog import ModelEntry, get_catalog, get_trainable
from touster.hardware.estimate import (
    estimate_tokens_per_second,
    estimate_vram_needed,
    model_fits,
)

_GB = 1_073_741_824  # 1 GiB in bytes
_MARGINAL_THRESHOLD = 0.10  # 10 % margin = "marginal"


def _platform_label(hw: HardwareConfig) -> str:
    """Return a human-readable platform string."""
    if hw.platform == "cuda":
        return "[bold green]CUDA / NVIDIA[/bold green]"
    if hw.platform == "mlx":
        return "[bold cyan]Apple Silicon (MLX)[/bold cyan]"
    return "[bold yellow]CPU only[/bold yellow]"


def _fits_symbol(vram_bytes: int, entry: ModelEntry) -> str:
    """Return a coloured fit symbol for the table."""
    needed_gb = estimate_vram_needed(entry.param_billions, entry.default_quant_bits)
    available_gb = vram_bytes / 1e9

    if available_gb <= 0:
        # CPU path: only models under 2 GB RAM footprint are shown
        if estimate_vram_needed(entry.param_billions, entry.default_quant_bits) < 2.0:
            return "[touster.success]OK[/touster.success]"
        return "[touster.error]XX[/touster.error]"

    ratio = available_gb / needed_gb if needed_gb > 0 else float("inf")
    if ratio >= (1 + _MARGINAL_THRESHOLD):
        return "[touster.success]✓[/touster.success]"
    if ratio >= (1 - _MARGINAL_THRESHOLD):
        return "[touster.warning]~[/touster.warning]"
    return "[touster.error]✗[/touster.error]"


def _build_specs_panel(hw: HardwareConfig) -> Panel:
    """Build the system specs Rich panel."""
    vram_budget_gb = hw.vram_bytes * 0.85 / 1e9
    vram_str = f"{hw.vram_bytes / _GB:.1f} GiB" if hw.vram_bytes > 0 else "None"
    gpu_str = f"{hw.gpu_name} ({vram_str})" if hw.gpu_name else "None"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="touster.dim", justify="right")
    grid.add_column()

    grid.add_row("Platform", _platform_label(hw))
    grid.add_row("GPU", f"[touster.model]{gpu_str}[/touster.model]")
    grid.add_row("System RAM", f"[bold]{hw.ram_bytes / _GB:.1f} GiB[/bold]")
    grid.add_row("CPU cores", f"[bold]{hw.cpu_count}[/bold]")
    if hw.vram_bytes > 0:
        grid.add_row(
            "VRAM budget",
            f"[touster.success]{vram_budget_gb:.1f} GB[/touster.success] available for model",
        )
    else:
        grid.add_row(
            "VRAM budget",
            "[touster.warning]CPU only — using RAM[/touster.warning]",
        )

    return Panel(
        grid,
        title="[touster.brand]🖥  System[/touster.brand]",
        border_style="touster.brand",
        expand=False,
    )


def _build_model_table(hw: HardwareConfig, entries: list[ModelEntry]) -> Table:
    """Build the ranked model Rich table."""
    table = Table(
        title="[touster.brand]📊  Model Ranking[/touster.brand]",
        border_style="touster.dim",
        header_style="touster.table.header",
        show_lines=False,
        expand=False,
    )

    table.add_column("#", justify="right", style="touster.dim", width=3)
    table.add_column("Model", style="touster.model", min_width=18)
    table.add_column("Params", justify="right", width=7)
    table.add_column("VRAM (GB)", justify="right", width=9)
    table.add_column("t/s est.", justify="right", width=9)
    table.add_column("Quality", justify="right", width=7)
    table.add_column("Fits", justify="center", width=5)

    for rank, entry in enumerate(entries, start=1):
        row_style = "touster.top" if rank <= 3 else ""
        tps = estimate_tokens_per_second(
            entry.param_billions,
            hw.gpu_bandwidth_gbps,
            entry.default_quant_bits,
        )
        vram_gb = estimate_vram_needed(entry.param_billions, entry.default_quant_bits)
        fits_sym = _fits_symbol(hw.vram_bytes, entry)

        tps_str = f"{tps:.0f}" if tps > 0 else "—"
        table.add_row(
            str(rank),
            entry.id,
            f"{entry.param_billions:.1f}B",
            f"{vram_gb:.1f}",
            tps_str,
            f"{entry.quality_score:.0f}",
            fits_sym,
            style=row_style,
        )

    return table


def print_hardware_report(
    hw: HardwareConfig,
    suggested_model: str | None = None,
) -> str:
    """Print the hardware specs panel and ranked model table; return chosen model id."""
    console.print()

    # Specs panel
    specs_panel = _build_specs_panel(hw)
    console.print(specs_panel)
    console.print()

    # Ranked model table
    catalog = get_catalog()
    trainable = get_trainable(hw, catalog)

    if not trainable:
        console.print(
            "[touster.warning]No models fit on this hardware. "
            "Defaulting to tiny-gpt2 for CPU validation.[/touster.warning]"
        )
        return "sshleifer/tiny-gpt2"

    table = _build_model_table(hw, trainable)
    console.print(table)
    console.print()

    # Determine suggestion
    top_model = trainable[0]
    default_choice: str

    if suggested_model:
        # Try to find the suggestion in trainable list
        match = next((e for e in trainable if suggested_model in (e.id, e.hf_id)), None)
        default_choice = match.id if match else top_model.id
    else:
        default_choice = top_model.id

    # Footer prompt
    console.print(
        f"[touster.dim]Top suggestion:[/touster.dim] "
        f"[touster.model]{default_choice}[/touster.model]"
    )
    console.print(
        "[touster.dim]Enter a model id from the table above, or press Enter to accept.[/touster.dim]"
    )

    import sys
    _interactive = sys.stdin is not None and hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
    if _interactive:
        try:
            chosen = Prompt.ask(
                "[touster.brand]Model[/touster.brand]",
                default=default_choice,
                console=console,
            )
        except (EOFError, KeyboardInterrupt):
            chosen = default_choice
    else:
        chosen = default_choice

    # Resolve to hf_id if user typed a short id
    resolved = next(
        (e.hf_id for e in trainable if chosen in (e.id, e.hf_id)),
        chosen,  # pass through unknown ids as-is
    )

    console.print(
        f"\n[touster.success]✓[/touster.success] "
        f"Selected: [touster.model]{resolved}[/touster.model]\n"
    )
    return resolved
