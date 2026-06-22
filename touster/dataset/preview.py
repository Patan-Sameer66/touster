from __future__ import annotations

from pathlib import Path

from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from touster.console import console
from touster.dataset.formats import apply_chat_template, count_tokens_dataset
from touster.dataset.schema import load_jsonl


def print_preview(
    dataset_path: Path,
    recipe,
    n_samples: int = 3,
) -> None:
    """
    Print a Rich Panel showing:
    - N sample rows formatted with the chat template
    - Token count stats (mean, max, p95)
    - Starting hyperparameters from recipe
    - Estimated dataset size + note about API cost
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must be >= 0, got {n_samples!r}")

    dataset_path = Path(dataset_path)
    ds = load_jsonl(dataset_path)

    if not ds.samples:
        console.print(
            "[touster.warning]Warning:[/touster.warning] Dataset is empty — no samples to preview."
        )
        return

    chat_template = "chatml"
    stats = count_tokens_dataset(ds, template=chat_template)

    # ------------------------------------------------------------------
    # Sample rows
    # ------------------------------------------------------------------
    samples_to_show = list(ds.samples[:n_samples])
    sample_texts: list[str] = []
    for i, sample in enumerate(samples_to_show):
        formatted = apply_chat_template(sample, template=chat_template)
        # Truncate long samples for display at a word boundary
        if len(formatted) > 500:
            truncated = formatted[:497]
            last_space = truncated.rfind(" ")
            formatted = (truncated[:last_space] if last_space > 400 else truncated) + "..."
        sample_texts.append(f"[bold]Sample {i + 1}:[/bold]\n[touster.code]{escape(formatted)}[/touster.code]")

    # ------------------------------------------------------------------
    # Token stats table
    # ------------------------------------------------------------------
    token_table = Table(
        title="Token Count Statistics",
        show_header=True,
        header_style="touster.table.header",
        box=None,
        padding=(0, 2),
    )
    token_table.add_column("Metric", style="touster.dim")
    token_table.add_column("Value", justify="right")
    token_table.add_row("Total tokens", str(stats["total"]))
    token_table.add_row("Mean tokens/sample", f"{stats['mean']:.1f}")
    token_table.add_row("Max tokens", str(stats["max"]))
    token_table.add_row("Min tokens", str(stats["min"]))
    token_table.add_row("p95 tokens", str(stats["p95"]))

    # ------------------------------------------------------------------
    # Hyperparameters table — use Text() to prevent markup injection from
    # recipe attribute values (e.g. model names containing [ ])
    # ------------------------------------------------------------------
    hparam_table = Table(
        title="Starting Hyperparameters",
        show_header=True,
        header_style="touster.table.header",
        box=None,
        padding=(0, 2),
    )
    hparam_table.add_column("Parameter", style="touster.dim")
    hparam_table.add_column("Value", justify="right")
    hparam_table.add_row("base_model", Text(str(recipe.base_model)))
    hparam_table.add_row("learning_rate", Text(str(recipe.learning_rate)))
    hparam_table.add_row("lora_rank", Text(str(recipe.lora_rank)))
    hparam_table.add_row("lora_alpha", Text(str(recipe.lora_alpha)))
    hparam_table.add_row("num_epochs", Text(str(recipe.num_epochs)))
    hparam_table.add_row("batch_size", Text(str(recipe.batch_size)))
    hparam_table.add_row("max_steps", Text(str(recipe.max_steps)))
    hparam_table.add_row("scheduler", Text(str(recipe.scheduler)))

    # ------------------------------------------------------------------
    # Render preview panel
    # ------------------------------------------------------------------
    console.rule("[touster.brand]Dataset Preview[/touster.brand]", style="touster.dim")

    for text in sample_texts:
        console.print(
            Panel(
                Text.from_markup(text),
                border_style="touster.dim",
                expand=False,
            )
        )

    console.print(token_table)
    console.print()
    console.print(hparam_table)
    console.print()

    file_size_kb = dataset_path.stat().st_size / 1024
    console.print(
        f"[touster.dim]Dataset:[/touster.dim] {len(ds)} samples  "
        f"[touster.dim]|[/touster.dim]  {file_size_kb:.1f} KB on disk  "
        f"[touster.dim]|[/touster.dim]  {stats['total']:,} total tokens"
    )
    console.print(
        "[touster.warning]Note:[/touster.warning] "
        "[touster.dim]API-generated datasets consume tokens. "
        "Estimated cost depends on your provider and model pricing.[/touster.dim]"
    )
    console.rule(style="touster.dim")
