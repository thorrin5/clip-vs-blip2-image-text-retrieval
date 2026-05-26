"""Common dataset structures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


@dataclass
class RetrievalDataset:
    name: str
    split: str
    image_paths: List[str]
    captions: List[str]
    image_to_captions: Dict[int, List[int]]
    metadata_file: str
    text_description: str

    @property
    def num_images(self) -> int:
        return len(self.image_paths)

    @property
    def num_captions(self) -> int:
        return len(self.captions)

    def to_metadata(self) -> dict:
        return {
            "name": self.name,
            "split": self.split,
            "num_images": self.num_images,
            "num_captions": self.num_captions,
            "metadata_file": self.metadata_file,
            "text_description": self.text_description,
        }


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def limit_dataset(dataset: RetrievalDataset, max_images: int | None) -> RetrievalDataset:
    if max_images is None or dataset.num_images <= max_images:
        return dataset

    keep_images = dataset.image_paths[:max_images]
    keep_caption_indices: list[int] = []
    new_mapping: dict[int, list[int]] = {}
    for new_img_idx, old_img_idx in enumerate(range(max_images)):
        old_caption_indices = dataset.image_to_captions[old_img_idx]
        new_mapping[new_img_idx] = list(range(len(keep_caption_indices), len(keep_caption_indices) + len(old_caption_indices)))
        keep_caption_indices.extend(old_caption_indices)

    captions = [dataset.captions[index] for index in keep_caption_indices]
    return RetrievalDataset(
        name=dataset.name,
        split=dataset.split,
        image_paths=keep_images,
        captions=captions,
        image_to_captions=new_mapping,
        metadata_file=dataset.metadata_file,
        text_description=dataset.text_description,
    )

