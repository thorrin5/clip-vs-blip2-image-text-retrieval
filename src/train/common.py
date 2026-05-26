"""Shared helpers for canonical fine-tuning.

The SciCap fine-tuning scripts share dataset conversion, seed setup, checkpoint
writing, and JSON payload construction here. Keeping these details centralized
helps ensure CLIP and BLIP-2 write comparable result artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import random
from typing import Any

from PIL import Image

from src.eval.common import load_dataset_from_config, write_json_result, write_json_result_to
from src.eval.retrieval_metrics import compact_metrics, compute_retrieval_metrics
from src.utils.config import get_recall_k, load_config, resolve_path
from src.utils.paths import ensure_output_dirs, timestamp


@dataclass
class TrainSample:
    image_path: str
    text: str


class PairDataset:
    """Load paired image/text samples lazily for PyTorch DataLoader workers."""

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
    """Use the first caption per image as the supervised training pair."""
    samples: list[TrainSample] = []
    for image_idx, image_path in enumerate(dataset.image_paths):
        caption_indices = dataset.image_to_captions[image_idx]
        if not caption_indices:
            continue
        samples.append(TrainSample(image_path=image_path, text=dataset.captions[caption_indices[0]]))
    return samples


def load_ai2d_splits(config_path: str, max_train: int | None = None, max_eval: int | None = None):
    config = load_config(config_path)
    train = load_dataset_from_config(config, "ai2d", "train", max_train)
    val = load_dataset_from_config(config, "ai2d", "val", max_eval)
    test = load_dataset_from_config(config, "ai2d", "test", max_eval)
    return config, train, val, test


def load_scicap_splits(config_path: str, max_train: int | None = None, max_eval: int | None = None):
    config = load_config(config_path)
    train = load_dataset_from_config(config, "scicap", "train", max_train)
    val = load_dataset_from_config(config, "scicap", "val", max_eval)
    test = load_dataset_from_config(config, "scicap", "test", max_eval)
    return config, train, val, test


def write_checkpoint(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(payload, path)
    return path


def append_training_csv(config: dict[str, Any], filename: str, row: dict[str, Any]) -> Path:
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


def write_finetune_result(config: dict[str, Any], prefix: str, payload: dict[str, Any]) -> Path:
    return write_json_result(config, f"{prefix}_ai2d_test_finetuned", payload)


def write_scicap_finetune_result(config: dict[str, Any], prefix: str, payload: dict[str, Any]) -> Path:
    raw_root = resolve_path(config, "scicap_raw_root")
    return write_json_result_to(raw_root, f"{prefix}_scicap_test_finetuned", payload)


def write_training_history(path: Path, history: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def checkpoint_name(model_key: str) -> str:
    return f"{model_key}_ai2d_finetuned_{timestamp()}.pt"


def scicap_checkpoint_name(model_key: str) -> str:
    return f"{model_key}_scicap_finetuned_{timestamp()}.pt"


def compute_similarity_metrics(similarity, dataset, config: dict[str, Any], timings: dict[str, float] | None = None):
    return compute_retrieval_metrics(similarity, dataset.image_to_captions, get_recall_k(config), timings=timings or {})


def metric_row(model: str, implementation: str, dataset, metrics: dict[str, Any], raw_json: Path) -> dict[str, Any]:
    return {
        "model": model,
        "implementation": implementation,
        "dataset": dataset.name,
        "split": dataset.split,
        **compact_metrics(metrics),
        "raw_json": str(raw_json),
    }
