"""SciCap loader for scientific figure-caption retrieval experiments.

SciCap is represented as JSONL after preprocessing. Each row has exactly one
figure and one `caption_used`, which makes the retrieval task a clean one-to-one
matching problem for both image-to-text and text-to-image directions.
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import RetrievalDataset


def load_scicap(
    processed_dir: str | Path,
    split: str = "test",
    max_images: int | None = None,
) -> RetrievalDataset:
    """Load a SciCap processed split from JSONL.

    Each JSONL line is expected to have:
        id, image_path, caption_original, caption_used, source
    """
    processed_dir = Path(processed_dir)
    jsonl_path = processed_dir / f"{split}.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"SciCap processed split not found: {jsonl_path}\n"
            "Run: python -m src.prepare_scicap_split"
        )

    image_paths: list[str] = []
    captions: list[str] = []
    image_to_captions: dict[int, list[int]] = {}

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            image_path = Path(record["image_path"])
            if not image_path.exists():
                continue
            caption = record["caption_used"].strip()
            if not caption:
                continue
            image_idx = len(image_paths)
            caption_idx = len(captions)
            image_paths.append(str(image_path))
            captions.append(caption)
            image_to_captions[image_idx] = [caption_idx]

    if max_images is not None and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]
        keep_captions: list[str] = []
        new_mapping: dict[int, list[int]] = {}
        for new_idx in range(max_images):
            old_caps = image_to_captions[new_idx]
            new_mapping[new_idx] = list(range(len(keep_captions), len(keep_captions) + len(old_caps)))
            keep_captions.extend(captions[i] for i in old_caps)
        captions = keep_captions
        image_to_captions = new_mapping

    return RetrievalDataset(
        name="SciCap",
        split=split,
        image_paths=image_paths,
        captions=captions,
        image_to_captions=image_to_captions,
        metadata_file=str(jsonl_path),
        text_description="SciCap scientific figure captions (caption_used after preprocessing)",
    )
