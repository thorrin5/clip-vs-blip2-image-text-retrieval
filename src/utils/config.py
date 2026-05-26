"""Pomocné funkcie na načítanie a interpretáciu experimentálnej konfigurácie."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path) -> Dict[str, Any]:
    """Načíta YAML konfiguráciu experimentu ako slovník."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read experiment_config.yaml") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config {config_path} did not contain a YAML mapping")
    return config


def project_root(config: Dict[str, Any]) -> Path:
    """Vráti koreň projektu definovaný v konfigurácii alebo aktuálny adresár."""
    root = config.get("paths", {}).get("project_root")
    if root:
        return Path(root)
    return Path.cwd()


def resolve_path(config: Dict[str, Any], key: str) -> Path:
    """Vyrieši konfiguračnú cestu relatívne ku koreňu projektu, ak treba."""
    paths = config.get("paths", {})
    if key not in paths:
        raise KeyError(f"Missing paths.{key} in config")
    value = Path(str(paths[key]))
    if value.is_absolute():
        return value
    return project_root(config) / value


def get_recall_k(config: Dict[str, Any]) -> list[int]:
    """Načíta zoznam hodnôt K pre metriky Recall@K."""
    values = config.get("evaluation", {}).get("recall_k", [1, 5, 10])
    return [int(value) for value in values]
