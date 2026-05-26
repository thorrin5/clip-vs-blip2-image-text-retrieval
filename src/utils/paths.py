"""Pomocné funkcie pre cesty a výstupné adresáre experimentov."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from .config import resolve_path


def timestamp() -> str:
    """Vráti časovú značku používanú pri názvoch výstupných súborov."""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def ensure_output_dirs(config: Dict[str, Any]) -> None:
    """
    Vytvorí základné výstupné adresáre definované v konfigurácii.

    Evaluácie aj tréningové skripty používajú spoločné kľúče pre surové JSON
    výsledky, tabuľky, obrázky a logy. Funkcia centralizuje ich vytváranie,
    aby jednotlivé skripty nemuseli riešiť rozdiely v lokálnych cestách.
    """
    for key in ["raw_root", "tables_root", "figures_root", "logs_root"]:
        resolve_path(config, key).mkdir(parents=True, exist_ok=True)


def require_path(path: str | Path, label: str) -> Path:
    """
    Overí existenciu povinnej cesty a vráti ju ako objekt `Path`.

    Používa sa pri vstupoch, ktoré musia byť pripravené pred spustením
    experimentu, napríklad anotácie datasetu alebo vopred vytvorené splity.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved
