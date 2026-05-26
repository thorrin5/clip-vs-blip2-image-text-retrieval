"""AI2D loader for scientific diagram retrieval experiments."""

from __future__ import annotations

from pathlib import Path

from .common import RetrievalDataset, limit_dataset, load_json


def load_ai2d(
    dataset_json_path: str | Path,
    images_dir: str | Path,
    split: str = "test",
    max_images: int | None = None,
) -> RetrievalDataset:
    data = load_json(dataset_json_path)
    images_root = Path(images_dir)
    image_paths: list[str] = []
    captions: list[str] = []
    image_to_captions: dict[int, list[int]] = {}

    has_requested_split = any(image.get("split") == split for image in data.get("images", []))

    for image in data.get("images", []):
        if has_requested_split and image.get("split") != split:
            continue
        image_path = images_root / image["filename"]
        if not image_path.exists():
            continue

        raw_captions = [sentence.get("raw", "").strip() for sentence in image.get("sentences", [])]
        raw_captions = [caption for caption in raw_captions if caption]
        if not raw_captions:
            continue

        image_idx = len(image_paths)
        image_paths.append(str(image_path))
        image_to_captions[image_idx] = list(range(len(captions), len(captions) + len(raw_captions)))
        captions.extend(raw_captions)

    split_name = split if has_requested_split else "all_available"
    dataset = RetrievalDataset(
        name="AI2D",
        split=split_name,
        image_paths=image_paths,
        captions=captions,
        image_to_captions=image_to_captions,
        metadata_file=str(dataset_json_path),
        text_description="AI2D question/caption text from existing metadata",
    )
    return limit_dataset(dataset, max_images)

