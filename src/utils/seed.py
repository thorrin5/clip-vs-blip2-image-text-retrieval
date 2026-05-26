"""Pomocná funkcia na nastavenie reprodukovateľných seedov."""

from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    """
    Nastaví seed pre dostupné generátory náhodnosti.

    Reprodukovateľnosť je dôležitá najmä pri vytváraní splitov a pri
    fine-tuningu, kde náhodné miešanie dát alebo inicializácia optimalizácie
    môže ovplyvniť výsledné Recall@K hodnoty.
    """
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
