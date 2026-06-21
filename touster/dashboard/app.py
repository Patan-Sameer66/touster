from __future__ import annotations

from pathlib import Path

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Input, Label, Static


# ── helpers ───────────────────────────────────────────────────────────────────


def _load_run_summary(run_dir: Path) -> tuple[str, list[str]]:
    """Return (summary_text, experiment_lines).  Never raises."""
    try:
        from touster.state import load_experiments, load_state

        state = load_state(run_dir)
        if state is None:
            summary = "[dim]No run.json found[/dim]"
            return summary, []

        summary_lines = [
            f"[bold]Model:[/bold]   {state.base_model}",
            f"[bold]Dataset:[/bold] {state.dataset_path}",
            f"[bold]Phase:[/bold]   {state.phase}",
            f"[bold]Best bpb:[/bold] {state.best_bpb:.4f}" if state.best_bpb < float("inf") else "[bold]Best bpb:[/bold] n/a",
            f"[bold]Trials:[/bold]  {state.total_trials}",
        ]
        summary = "\n".join(summary_lines)

        experiments = load_experiments(run_dir)
        exp_lines: list[str] = []
        for rec in experiments:
            tag = "[green]kept[/green]" if rec.kept else "[dim]discarded[/dim]"
            exp_lines.append(
                f"Trial {rec.trial_id}: bpb={rec.eval_bpb:.4f} [{tag}]"
            )
        return summary, exp_lines
    except Exception as exc:  # noqa: BLE001
        return f"[red]Error loading run: {exc}[/red]", []


# ── Textual App ───────────────────────────────────────────────────────────────

_AMBER = "#F59E0B"
_AMBER_DIM = "#B45309"


class TuosterDashboard(App):
    """Interactive before/after comparison dashboard for Touster fine-tuning runs."""

    CSS = f"""
    Screen {{
        background: #1C1917;
    }}

    #header-label {{
        color: {_AMBER};
        text-style: bold;
        padding: 0 1;
    }}

    #run-summary {{
        width: 1fr;
        height: 1fr;
        border: solid {_AMBER_DIM};
        padding: 1 2;
        overflow-y: auto;
    }}

    #exp-log {{
        width: 1fr;
        height: 1fr;
        border: solid {_AMBER_DIM};
        padding: 1 2;
        overflow-y: auto;
    }}

    #summary-row {{
        height: 12;
    }}

    #prompt-row {{
        height: 5;
        border: solid {_AMBER_DIM};
        padding: 1 2;
        align: left middle;
    }}

    #prompt-label {{
        color: {_AMBER};
        text-style: bold;
        width: auto;
        padding-right: 1;
    }}

    #prompt-input {{
        width: 1fr;
    }}

    #send-btn {{
        width: auto;
        margin-left: 1;
        background: {_AMBER_DIM};
        color: #FFFFFF;
    }}

    #output-row {{
        height: 1fr;
    }}

    #base-pane {{
        width: 1fr;
        height: 1fr;
        border: solid blue;
        padding: 1 2;
        overflow-y: auto;
    }}

    #ft-pane {{
        width: 1fr;
        height: 1fr;
        border: solid green;
        padding: 1 2;
        overflow-y: auto;
    }}

    #base-title {{
        color: #60A5FA;
        text-style: bold;
        margin-bottom: 1;
    }}

    #ft-title {{
        color: #4ADE80;
        text-style: bold;
        margin-bottom: 1;
    }}
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("tab", "focus_next", "Focus"),
        Binding("enter", "submit_prompt", "Send", show=False),
    ]

    def __init__(
        self,
        base_model_id: str,
        adapter_path: Path | None,
        run_dir: Path,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self._base_model_id = base_model_id
        self._adapter_path = adapter_path
        self._run_dir = run_dir
        self._pair: object | None = None
        self._models_loaded = False
        self._loading = False

    # ── compose ───────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        summary_text, exp_lines = _load_run_summary(self._run_dir)

        with Horizontal(id="summary-row"):
            with Vertical(id="run-summary"):
                yield Label("[bold yellow]📊 Run Summary[/bold yellow]")
                yield Static(summary_text, id="summary-static")
            with Vertical(id="exp-log"):
                yield Label("[bold yellow]📝 Experiment Log[/bold yellow]")
                log_text = "\n".join(exp_lines) if exp_lines else "[dim]No experiments recorded yet[/dim]"
                yield Static(log_text, id="log-static")

        with Horizontal(id="prompt-row"):
            yield Label("💬 Prompt:", id="prompt-label")
            yield Input(placeholder="Enter a prompt…", id="prompt-input")
            yield Button("Send", id="send-btn", variant="primary")

        with Horizontal(id="output-row"):
            with Vertical(id="base-pane"):
                yield Label("🔵 Base model", id="base-title")
                yield Static("", id="base-output")
            with Vertical(id="ft-pane"):
                yield Label("🟢 Fine-tuned", id="ft-title")
                yield Static("", id="ft-output")

        yield Footer()

    # ── event handlers ────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "send-btn":
            self._submit()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._submit()

    def action_submit_prompt(self) -> None:
        self._submit()

    def _submit(self) -> None:
        if self._loading:
            return
        prompt_input = self.query_one("#prompt-input", Input)
        prompt = prompt_input.value.strip()
        if not prompt:
            return
        self._loading = True
        self.query_one("#base-output", Static).update("[dim]Generating…[/dim]")
        self.query_one("#ft-output", Static).update("[dim]Generating…[/dim]")
        self.run_worker(lambda: self._run_inference(prompt), thread=True)

    def _run_inference(self, prompt: str) -> None:
        """Worker (runs in thread): loads models on first call, then runs inference."""
        try:
            if not self._models_loaded:
                self.call_from_thread(self._set_status, "Loading models…")
                try:
                    from touster.dashboard.compare import ModelPair

                    pair = ModelPair(
                        base_model_id=self._base_model_id,
                        adapter_path=self._adapter_path,
                    )
                    pair.load()
                    self._pair = pair
                    self._models_loaded = True
                    status = (
                        "Models loaded"
                        if pair.has_adapter
                        else "Fine-tuned adapter not found — showing base model only"
                    )
                    self.call_from_thread(self._set_status, status)
                except Exception as exc:  # noqa: BLE001
                    self.call_from_thread(self._set_status, f"Model load failed: {exc}")
                    self._post_outputs(
                        f"[red]Model load failed: {exc}[/red]",
                        f"[red]Model load failed: {exc}[/red]",
                    )
                    return

            if self._pair is None:
                self._post_outputs("[red]Models not available[/red]", "[red]Models not available[/red]")
                return

            from touster.dashboard.compare import ModelPair

            pair: ModelPair = self._pair  # type: ignore[assignment]
            base_out = pair.generate_base(prompt)
            ft_out = pair.generate_finetuned(prompt)

            ft_display = (
                f"[dim](No adapter — showing base)[/dim]\n\n{ft_out}"
                if not pair.has_adapter
                else ft_out
            )
            self._post_outputs(base_out or "[dim](empty)[/dim]", ft_display or "[dim](empty)[/dim]")
        except Exception as exc:  # noqa: BLE001
            self._post_outputs(f"[red]Error: {exc}[/red]", f"[red]Error: {exc}[/red]")
        finally:
            self._loading = False

    def _post_outputs(self, base_text: str, ft_text: str) -> None:
        self.call_from_thread(self._update_outputs, base_text, ft_text)

    def _update_outputs(self, base_text: str, ft_text: str) -> None:
        self.query_one("#base-output", Static).update(base_text)
        self.query_one("#ft-output", Static).update(ft_text)

    def _set_status(self, msg: str) -> None:
        """Update footer / title to convey loading status."""
        self.sub_title = msg


# ── public entry-point ────────────────────────────────────────────────────────


def launch_dashboard(base_model_id: str, adapter_path: str, run_dir: Path) -> None:
    """Launch the Textual dashboard.  Blocks until user quits (q or Ctrl+C)."""
    app = TuosterDashboard(
        base_model_id=base_model_id,
        adapter_path=Path(adapter_path) if adapter_path else None,
        run_dir=run_dir,
    )
    app.run()
