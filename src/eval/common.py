"""Shared evaluation helpers.

These helpers keep dataset dispatch and result writing consistent across CLIP,
BLIP-2, and fine-tuning scripts. The exported package keeps AI2D support because
some shared smoke tests and legacy scripts import it, even though the current
defense focus is Flickr30K plus SciCap.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict

from src.data.ai2d import load_ai2d
from src.data.flickr30k import load_flickr30k_karpathy
from src.data.scicap import load_scicap
from src.utils.config import resolve_path
from src.utils.paths import ensure_output_dirs, timestamp


def load_dataset_from_config(config: Dict[str, Any], dataset: str, split: str, max_images: int | None = None):
    """Resolve a configured dataset name to the corresponding RetrievalDataset."""
    if dataset == "flickr30k":
        return load_flickr30k_karpathy(
            resolve_path(config, "flickr30k_karpathy_json"),
            resolve_path(config, "flickr30k_images"),
            split=split,
            max_images=max_images,
        )
    if dataset == "ai2d":
        split_json = resolve_path(config, "ai2d_split_json")
        metadata = split_json if split_json.exists() else resolve_path(config, "ai2d_dataset_json")
        return load_ai2d(
            metadata,
            resolve_path(config, "ai2d_images"),
            split=split,
            max_images=max_images,
        )
    if dataset == "scicap":
        return load_scicap(
            resolve_path(config, "scicap_processed_dir"),
            split=split,
            max_images=max_images,
        )
    raise ValueError(f"Unknown dataset: {dataset}")


def write_json_result(config: Dict[str, Any], prefix: str, payload: Dict[str, Any]) -> Path:
    ensure_output_dirs(config)
    output_path = resolve_path(config, "raw_root") / f"{prefix}_{timestamp()}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def write_json_result_to(raw_root: Path, prefix: str, payload: Dict[str, Any]) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    output_path = raw_root / f"{prefix}_{timestamp()}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def append_metrics_csv_to(tables_root: Path, filename: str, row: Dict[str, Any]) -> Path:
    tables_root.mkdir(parents=True, exist_ok=True)
    output_path = tables_root / filename
    fieldnames = list(row.keys())
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return output_path


def append_metrics_csv(config: Dict[str, Any], filename: str, row: Dict[str, Any]) -> Path:
    ensure_output_dirs(config)
    output_path = resolve_path(config, "tables_root") / filename
    fieldnames = list(row.keys())
    write_header = not output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return output_path
