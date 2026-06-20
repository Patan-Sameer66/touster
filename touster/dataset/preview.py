from __future__ import annotations

from pathlib import Path

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
    dataset_path = Path(dataset_path)
    ds = load_jsonl(dataset_path)

    chat_template = "chatml"
    stats = count_tokens_dataset(ds, template=chat_template)

    # ------------------------------------------------------------------
    # Sample rows
    # ------------------------------------------------------------------
    samples_to_show = list(ds.samples[:n_samples])
    sample_texts: list[str] = []
    for i, sample in enumerate(samples_to_show):
        formatted = apply_chat_template(sample, template=chat_template)
        # Truncate long samples for display
        if len(formatted) > 500:
            formatted = formatted[:497] + "..."
        sample_texts.append(f"[bold]Sample {i + 1}:[/bold]\n[touster.code]{formatted}[/touster.code]")

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
    # Hyperparameters table
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
    hparam_table.add_row("base_model", str(recipe.base_model))
    hparam_table.add_row("learning_rate", str(recipe.learning_rate))
    hparam_table.add_row("lora_rank", str(recipe.lora_rank))
    hparam_table.add_row("lora_alpha", str(recipe.lora_alpha))
    hparam_table.add_row("num_epochs", str(recipe.num_epochs))
    hparam_table.add_row("batch_size", str(recipe.batch_size))
    hparam_table.add_row("max_steps", str(recipe.max_steps))
    hparam_table.add_row("scheduler", str(recipe.scheduler))

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
