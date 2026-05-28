"""Spoločné dátové štruktúry pre modelové wrappery."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelLoadInfo:
    """
    Metadáta o načítanom modeli ukladané do výsledkov experimentov.

    Tieto informácie umožňujú spätne overiť, ktorá implementácia, checkpoint,
    zariadenie a numerická presnosť boli použité pri konkrétnom behu
    evaluácie alebo fine-tuningu.
    """

    implementation: str
    model_name: str
    device: str
    precision: str

    def to_dict(self) -> dict:
        """Prevedie metadáta na slovník vhodný na zápis do JSON výsledkov."""
        return {
            "implementation": self.implementation,
            "model_name": self.model_name,
            "device": self.device,
            "precision": self.precision,
        }
