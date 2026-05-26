"""Path and output helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import resolve_path


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    for key in ["raw_root", "tables_root", "figures_root", "logs_root"]:
        resolve_path(config, key).mkdir(parents=True, exist_ok=True)


def require_path(path: str | Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved
