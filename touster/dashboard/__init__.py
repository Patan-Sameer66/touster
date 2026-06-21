from __future__ import annotations

__all__ = ["launch_dashboard"]


def launch_dashboard(base_model_id: str, adapter_path: str, run_dir) -> None:
    from touster.dashboard.app import launch_dashboard as _launch
    _launch(base_model_id, adapter_path, run_dir)
