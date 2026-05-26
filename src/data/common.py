"""Spoločné dátové štruktúry pre image-text retrieval datasety."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
import json


@dataclass
class RetrievalDataset:
    """
    Jednotná reprezentácia datasetu pre retrieval evaluáciu.

    `image_paths` obsahuje cesty k obrázkom, `captions` obsahuje všetky textové
    kandidáty a `image_to_captions` určuje, ktoré caption indexy sú správne pre
    konkrétny obrázok. Táto štruktúra umožňuje použiť rovnaký výpočet Recall@K
    pre Flickr30K aj SciCap.
    """
    name: str
    split: str
    image_paths: List[str]
    captions: List[str]
    image_to_captions: Dict[int, List[int]]
    metadata_file: str
    text_description: str

    @property
    def num_images(self) -> int:
        """Vráti počet obrázkových dotazov v datasete."""
        return len(self.image_paths)

    @property
    def num_captions(self) -> int:
        """Vráti počet textových kandidátov v datasete."""
        return len(self.captions)

    def to_metadata(self) -> dict:
        """Pripraví stručné metadáta zapisované do raw JSON výsledkov."""
        return {
            "name": self.name,
            "split": self.split,
            "num_images": self.num_images,
            "num_captions": self.num_captions,
            "metadata_file": self.metadata_file,
            "text_description": self.text_description,
        }


def load_json(path: str | Path) -> dict:
    """Načíta JSON súbor s UTF-8 kódovaním."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def limit_dataset(dataset: RetrievalDataset, max_images: int | None) -> RetrievalDataset:
    """
    Obmedzí dataset na prvých `max_images` obrázkov.

    Používa sa pri smoke testoch a malých kontrolných behoch. Funkcia zároveň
    prečísluje caption indexy tak, aby mapovanie ostalo konzistentné.
    """
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
