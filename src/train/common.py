"""Spoločné pomocné funkcie pre fine-tuning na datasete SciCap.

Funkcie v tomto module používajú oba fine-tuning skripty. Centralizujú prevod
retrieval datasetu na trénovacie páry, nastavenie seedov, zápis checkpointov,
histórií a výstupných JSON súborov, aby CLIP aj BLIP-2 produkovali porovnateľné
artefakty.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import random
from typing import Any

from PIL import Image

from src.eval.common import load_dataset_from_config, write_json_result_to
from src.eval.retrieval_metrics import compact_metrics, compute_retrieval_metrics
from src.utils.config import get_recall_k, load_config, resolve_path
from src.utils.paths import ensure_output_dirs, timestamp


@dataclass
class TrainSample:
    image_path: str
    text: str


class PairDataset:
    """
    PyTorch dataset pre páry obrázok-text.

    Obrázok sa načíta až v `__getitem__`, čo je vhodné pre DataLoader workery a
    šetrí pamäť pri veľkých SciCap splitoch. Text sa môže voliteľne transformovať
    podľa potrieb konkrétneho modelu.
    """

    def __init__(self, samples: list[TrainSample], image_transform, text_transform=None) -> None:
        self.samples = samples
        self.image_transform = image_transform
        self.text_transform = text_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(Path(sample.image_path)).convert("RGB")
        image = self.image_transform(image)
        text = self.text_transform(sample.text) if self.text_transform else sample.text
        return image, text


def set_seed(seed: int) -> None:
    """Nastaví seed pre Python, NumPy a PyTorch kvôli reprodukovateľnosti behov."""
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


def dataset_to_samples(dataset) -> list[TrainSample]:
    """
    Prevedie retrieval dataset na zoznam trénovacích párov.

    SciCap má vo verejnej finálnej pipeline jeden cieľový text na obrázok.
    Funkcia preto vyberie prvý dostupný caption index a vytvorí z neho
    supervidovaný pár používaný pri kontrastívnom tréningu.
    """
    samples: list[TrainSample] = []
    for image_idx, image_path in enumerate(dataset.image_paths):
        caption_indices = dataset.image_to_captions[image_idx]
        if not caption_indices:
            continue
        samples.append(TrainSample(image_path=image_path, text=dataset.captions[caption_indices[0]]))
    return samples


def load_scicap_splits(config_path: str, max_train: int | None = None, max_eval: int | None = None):
    """Načíta SciCap train/val/test splity podľa konfiguračného súboru."""
    config = load_config(config_path)
    train = load_dataset_from_config(config, "scicap", "train", max_train)
    val = load_dataset_from_config(config, "scicap", "val", max_eval)
    test = load_dataset_from_config(config, "scicap", "test", max_eval)
    return config, train, val, test


def write_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    """Uloží checkpoint modelu alebo jeho trénovateľných častí."""
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(payload, path)
    return path


def append_training_csv(config: dict[str, Any], filename: str, row: dict[str, Any]) -> Path:
    """Pridá riadok s tréningovými metrikami do CSV tabuľky."""
    ensure_output_dirs(config)
    output_path = Path(config["paths"].get("tables_root", "results/tables")) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return output_path


def build_finetune_payload(
    *,
    implementation: str,
    model_info: dict[str, Any],
    dataset,
    protocol: str,
    metrics: dict[str, Any],
    zero_shot_metrics: dict[str, Any] | None,
    val_metrics: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    """
    Vytvorí jednotný JSON payload pre fine-tuned výsledok.

    Payload obsahuje informácie o modeli, datasete, validačných metrikách,
    testovacích metrikách a tréningovej konfigurácii. Tento formát umožňuje
    spätne vysledovať, z ktorého behu pochádza hodnota v tabuľke.
    """
    payload = {
        "status": "ok",
        "implementation": implementation,
        "model": model_info,
        "dataset": dataset.to_metadata(),
        "protocol": protocol,
        "metrics": metrics,
        "validation_metrics": val_metrics,
        "training": training,
    }
    if zero_shot_metrics is not None:
        payload["zero_shot_metrics"] = zero_shot_metrics
    return payload


def write_scicap_finetune_result(config: dict[str, Any], prefix: str, payload: dict[str, Any]) -> Path:
    """Zapíše výsledok fine-tuningu na SciCap do SciCap raw adresára."""
    raw_root = resolve_path(config, "scicap_raw_root")
    return write_json_result_to(raw_root, f"{prefix}_scicap_test_finetuned", payload)


def write_training_history(path: Path, history: list[dict[str, Any]]) -> None:
    """Uloží per-epoch históriu tréningu pre neskoršiu analýzu kriviek."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def scicap_checkpoint_name(model_key: str) -> str:
    """Vytvorí názov checkpointu s časovou značkou pre SciCap beh."""
    return f"{model_key}_scicap_finetuned_{timestamp()}.pt"


def compute_similarity_metrics(similarity, dataset, config: dict[str, Any], timings: dict[str, float] | None = None):
    """Vypočíta Recall@K metriky zo similarity matice a ground-truth mapovania."""
    return compute_retrieval_metrics(similarity, dataset.image_to_captions, get_recall_k(config), timings=timings or {})


def metric_row(model: str, implementation: str, dataset, metrics: dict[str, Any], raw_json: Path) -> dict[str, Any]:
    """Pripraví plochý CSV riadok s metrikami pre výsledkovú tabuľku."""
    return {
        "model": model,
        "implementation": implementation,
        "dataset": dataset.name,
        "split": dataset.split,
        **compact_metrics(metrics),
        "raw_json": str(raw_json),
    }
