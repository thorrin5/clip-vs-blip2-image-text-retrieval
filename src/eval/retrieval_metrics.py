"""
Purpose:
    Compute the shared image-text retrieval metrics for all thesis experiments.

Thesis context:
    This module is model-neutral and is used for CLIP, BLIP-2, Flickr30K,
    SciCap, zero-shot runs, and fine-tuned runs. Keeping metrics here ensures
    the model comparison changes only the score matrix, not the evaluation rule.

Inputs:
    - Image-by-text score matrix where larger scores are better.
    - Ground-truth image_to_captions mapping.
    - Recall@K values, usually 1, 5, and 10.
    - Optional timing metadata.

Outputs:
    - Image-to-text Recall@K and ranks.
    - Text-to-image Recall@K and ranks.
    - Mean Recall@K across both directions.
    - Median ranks and timing metadata.

Defense note:
    This file is the answer to how fairness is enforced in evaluation. CLIP and
    BLIP-2 produce different scores, but both scores are ranked and evaluated by
    exactly the same Recall@K implementation.
"""

from __future__ import annotations

from statistics import median
from typing import Any, Dict, Iterable, List


def _to_list_matrix(similarity: Any) -> list[list[float]]:
    if hasattr(similarity, "detach"):
        similarity = similarity.detach().cpu().tolist()
    elif hasattr(similarity, "tolist"):
        similarity = similarity.tolist()
    return [[float(value) for value in row] for row in similarity]


def _rank_desc(values: list[float]) -> list[int]:
    return sorted(range(len(values)), key=lambda index: values[index], reverse=True)


def _validate_inputs(matrix: list[list[float]], image_to_captions: Dict[int, List[int]]) -> tuple[int, int]:
    """Fail early when a score matrix cannot represent a valid retrieval task."""
    if not matrix:
        raise ValueError("Similarity matrix is empty")
    if not matrix[0]:
        raise ValueError("Similarity matrix has no text columns")

    num_images = len(matrix)
    num_texts = len(matrix[0])
    for row_idx, row in enumerate(matrix):
        if len(row) != num_texts:
            raise ValueError(
                f"Similarity matrix must be rectangular; row 0 has {num_texts} columns, "
                f"row {row_idx} has {len(row)}"
            )

    missing_images = [image_idx for image_idx in range(num_images) if image_idx not in image_to_captions]
    if missing_images:
        raise ValueError(f"Missing caption mapping for image indices: {missing_images}")

    for image_idx, caption_indices in image_to_captions.items():
        if image_idx < 0 or image_idx >= num_images:
            raise ValueError(f"Image index {image_idx} is outside matrix range 0..{num_images - 1}")
        if not caption_indices:
            raise ValueError(f"Image index {image_idx} has no caption indices")
        for caption_idx in caption_indices:
            if caption_idx < 0 or caption_idx >= num_texts:
                raise ValueError(f"Caption index {caption_idx} is outside matrix range 0..{num_texts - 1}")

    return num_images, num_texts


def compute_retrieval_metrics(
    similarity: Any,
    image_to_captions: Dict[int, List[int]],
    k_values: Iterable[int] = (1, 5, 10),
    timings: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """Compute bidirectional retrieval metrics from an image x text score matrix."""
    matrix = _to_list_matrix(similarity)
    image_to_captions = {
        int(image_idx): [int(caption_idx) for caption_idx in caption_indices]
        for image_idx, caption_indices in image_to_captions.items()
    }
    num_images, num_texts = _validate_inputs(matrix, image_to_captions)
    k_values = sorted({int(k) for k in k_values})
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain at least one positive integer")

    # Build the inverse mapping for text-to-image retrieval. A caption can only
    # be correct for one image in the evaluated splits.
    caption_to_image: dict[int, int] = {}
    for image_idx, caption_indices in image_to_captions.items():
        for caption_idx in caption_indices:
            if caption_idx in caption_to_image:
                raise ValueError(f"Caption index {caption_idx} is mapped to multiple images")
            caption_to_image[int(caption_idx)] = int(image_idx)
    missing_captions = [caption_idx for caption_idx in range(num_texts) if caption_idx not in caption_to_image]
    if missing_captions:
        raise ValueError(f"Missing image mapping for caption indices: {missing_captions}")

    i2t_ranks: list[int] = []
    for image_idx in range(num_images):
        ranked = _rank_desc(matrix[image_idx])
        rank_position = {candidate_idx: rank + 1 for rank, candidate_idx in enumerate(ranked)}
        correct = set(image_to_captions[int(image_idx)])
        best_rank = min(rank_position[caption_idx] for caption_idx in correct)
        i2t_ranks.append(best_rank)

    t2i_ranks: list[int] = []
    columns = [[matrix[image_idx][caption_idx] for image_idx in range(num_images)] for caption_idx in range(num_texts)]
    for caption_idx in range(num_texts):
        ranked = _rank_desc(columns[caption_idx])
        rank_position = {candidate_idx: rank + 1 for rank, candidate_idx in enumerate(ranked)}
        correct_image = caption_to_image[caption_idx]
        t2i_ranks.append(rank_position[correct_image])

    def recalls(ranks: list[int]) -> dict[str, float]:
        return {f"R@{k}": sum(1 for rank in ranks if rank <= k) / len(ranks) * 100.0 for k in k_values}

    i2t = recalls(i2t_ranks)
    t2i = recalls(t2i_ranks)
    mean_recall = {f"R@{k}": (i2t[f"R@{k}"] + t2i[f"R@{k}"]) / 2.0 for k in k_values}

    return {
        "image_to_text": {
            **i2t,
            "ranks": i2t_ranks,
            "median_rank": float(median(i2t_ranks)),
        },
        "text_to_image": {
            **t2i,
            "ranks": t2i_ranks,
            "median_rank": float(median(t2i_ranks)),
        },
        "mean": mean_recall,
        "timing": timings or {},
        "num_images": num_images,
        "num_texts": num_texts,
    }


def compact_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Return a flat metrics summary suitable for CSV tables."""
    output: dict[str, float] = {}
    for prefix, key in [("i2t", "image_to_text"), ("t2i", "text_to_image"), ("mean", "mean")]:
        for metric, value in metrics.get(key, {}).items():
            if metric.startswith("R@"):
                output[f"{prefix}_{metric.replace('@', '')}"] = float(value)
    return output
