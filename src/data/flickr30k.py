"""Flickr30K Karpathy split loader.

The thesis uses the standard Karpathy test split: 1,000 images with five
captions each. The loader preserves that one-to-many mapping so Recall@K counts
any of the five captions as correct for image-to-text retrieval.
"""

from __future__ import annotations

from pathlib import Path

from .common import RetrievalDataset, limit_dataset, load_json


def load_flickr30k_karpathy(
    dataset_json_path: str | Path,
    images_dir: str | Path,
    split: str = "test",
    max_images: int | None = None,
) -> RetrievalDataset:
    """Load one Flickr30K Karpathy split as a RetrievalDataset."""
    data = load_json(dataset_json_path)
    image_paths: list[str] = []
    captions: list[str] = []
    image_to_captions: dict[int, list[int]] = {}
    images_root = Path(images_dir)

    for image in data.get("images", []):
        if image.get("split") != split:
            continue
        image_path = images_root / image["filename"]
        if not image_path.exists():
            continue

        image_idx = len(image_paths)
        image_paths.append(str(image_path))
        raw_captions = [sentence.get("raw", "").strip() for sentence in image.get("sentences", [])]
        raw_captions = [caption for caption in raw_captions if caption]
        image_to_captions[image_idx] = list(range(len(captions), len(captions) + len(raw_captions)))
        captions.extend(raw_captions)

    dataset = RetrievalDataset(
        name="Flickr30K",
        split=f"karpathy_{split}",
        image_paths=image_paths,
        captions=captions,
        image_to_captions=image_to_captions,
        metadata_file=str(dataset_json_path),
        text_description="Karpathy sentence captions",
    )
    return limit_dataset(dataset, max_images)
