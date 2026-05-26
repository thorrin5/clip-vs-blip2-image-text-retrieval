#!/usr/bin/env python3
"""
Purpose:
    Prepare deterministic SciCap figure-caption retrieval splits for the thesis
    experiments.

Thesis context:
    This script creates the SciCap train/validation/test JSONL files used by
    CLIP and BLIP-2 zero-shot evaluation and SciCap fine-tuning.

Inputs:
    - hidden-test.json from the local Hugging Face CrowdAILab/scicap snapshot.
    - img-hide_test.zip from the same snapshot.

Outputs:
    - data/scicap/images/*.png for selected scientific figures.
    - data/scicap_processed/train.jsonl, val.jsonl, and test.jsonl.

Defense note:
    This script is the reproducible bridge between the raw SciCap archive and
    the retrieval benchmark used in the experiments. It also stores
    caption_original and caption_used so caption preprocessing is auditable.

Additional implementation notes:

Reads annotation data from the HuggingFace cache (CrowdAILab/scicap),
extracts images from img-hide_test.zip, applies caption preprocessing,
and writes train/val/test JSONL splits to data/scicap_processed/.

Image source: hidden-test split from the SciCap HuggingFace snapshot.
We use these 47,639 figure-caption pairs to create our own train/val/test
splits for retrieval experiments — the challenge label ("hidden test") is
irrelevant here; for our purposes these are simply scientific figures with
ground-truth captions.

Split sizes (seed=42):
    train: 45,000
    val:   1,000
    test:  1,000

Inputs:
    - hidden-test.json from the local Hugging Face SciCap snapshot
    - img-hide_test.zip from the same snapshot

Outputs:
    - data/scicap/images/*.png for selected figures
    - data/scicap_processed/train.jsonl
    - data/scicap_processed/val.jsonl
    - data/scicap_processed/test.jsonl
"""

from __future__ import annotations

import json
import os
import re
import zipfile
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_SNAPSHOT = Path(os.environ.get("SCICAP_SNAPSHOT", PROJECT_ROOT / "data" / "scicap_source"))
IMAGES_DIR = PROJECT_ROOT / "data" / "scicap" / "images"
PROCESSED_DIR = PROJECT_ROOT / "data" / "scicap_processed"

SEED = 42
TRAIN_SIZE = 45000
VAL_SIZE = 1000
TEST_SIZE = 1000
TOTAL_NEEDED = TRAIN_SIZE + VAL_SIZE + TEST_SIZE

MIN_FIRST_SENTENCE_WORDS = 5
MIN_FIRST_SENTENCE_CHARS = 30

_FIG_PREFIX_RE = re.compile(
    r"^\s*(?:FIGURE|Figure|FIG|Fig\.?)\s*\d+[\.:)–-]?\s*",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s+")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Directory containing hidden-test.json and img-hide_test.zip.",
    )
    parser.add_argument(
        "--annotations-json",
        type=Path,
        default=None,
        help="Override path to hidden-test.json.",
    )
    parser.add_argument(
        "--images-zip",
        type=Path,
        default=None,
        help="Override path to img-hide_test.zip.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=IMAGES_DIR,
        help="Output directory for extracted SciCap images.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Output directory for train/val/test JSONL files.",
    )
    return parser


def remove_figure_prefix(text: str) -> str:
    return _FIG_PREFIX_RE.sub("", text).strip()


def normalize_whitespace(text: str) -> str:
    return _MULTI_SPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def make_caption_used(caption: str) -> str:
    """Create the shared comparison caption used by both CLIP and BLIP-2."""
    cleaned = remove_figure_prefix(caption)
    cleaned = normalize_whitespace(cleaned)
    sentences = split_sentences(cleaned)
    if not sentences:
        return cleaned

    first = sentences[0]
    words_in_first = len(first.split())
    too_short = words_in_first < MIN_FIRST_SENTENCE_WORDS or len(first) < MIN_FIRST_SENTENCE_CHARS

    if not too_short or len(sentences) == 1:
        return first

    combined = first + " " + sentences[1]
    return combined


def load_annotations(path: Path) -> tuple[dict[int, str], dict[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    id_to_filename: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}
    id_to_caption: dict[int, str] = {
        ann["image_id"]: ann["caption"] for ann in data["annotations"]
    }
    return id_to_filename, id_to_caption


def extract_images(zip_path: Path, img_ids: set[int], output_dir: Path) -> dict[int, str]:
    """Extract only selected figure images so the processed split is self-contained."""
    output_dir.mkdir(parents=True, exist_ok=True)
    id_to_path: dict[int, str] = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for name in names:
            if not name.endswith(".png"):
                continue
            basename = Path(name).name
            try:
                img_id = int(basename.split(".")[0])
            except ValueError:
                continue
            if img_id not in img_ids:
                continue
            dest = output_dir / basename
            if not dest.exists():
                data = zf.read(name)
                dest.write_bytes(data)
            id_to_path[img_id] = str(dest)
    return id_to_path


def build_records(
    id_to_filename: dict[int, str],
    id_to_caption: dict[int, str],
    id_to_path: dict[int, str],
) -> list[dict]:
    records = []
    for img_id, img_path in id_to_path.items():
        caption_orig = id_to_caption.get(img_id)
        if not caption_orig:
            continue
        caption_used = make_caption_used(caption_orig)
        records.append(
            {
                "id": str(img_id),
                "image_path": img_path,
                "caption_original": caption_orig,
                "caption_used": caption_used,
                "source": "scicap",
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def log_caption_stats(records: list[dict], label: str) -> None:
    orig_lens = [len(r["caption_original"].split()) for r in records]
    used_lens = [len(r["caption_used"].split()) for r in records]
    shortened = sum(1 for o, u in zip(orig_lens, used_lens) if u < o)
    avg_orig = sum(orig_lens) / len(orig_lens)
    avg_used = sum(used_lens) / len(used_lens)

    try:
        import clip as clip_module
        import torch
        clip_truncated = 0
        for rec in records:
            tokens = clip_module.tokenize([rec["caption_used"]], truncate=False)
            if int((tokens[0] != 0).sum()) > 77:
                clip_truncated += 1
    except Exception:
        clip_truncated = -1

    print(f"[{label}] n={len(records)}")
    print(f"  avg original length (words): {avg_orig:.1f}")
    print(f"  avg caption_used length (words): {avg_used:.1f}")
    print(f"  shortened captions: {shortened} ({100*shortened/len(records):.1f}%)")
    if clip_truncated >= 0:
        print(f"  captions still requiring CLIP truncation: {clip_truncated} ({100*clip_truncated/len(records):.1f}%)")
    else:
        print("  CLIP truncation count: not available (clip not in PATH)")


def main() -> None:
    import random
    args = build_parser().parse_args()
    random.seed(SEED)

    annotations_json = args.annotations_json or args.snapshot / "hidden-test.json"
    images_zip = args.images_zip or args.snapshot / "img-hide_test.zip"

    print("=" * 64)
    print("Preparing SciCap retrieval splits")
    print(f"  Annotations: {annotations_json}")
    print(f"  Images zip:  {images_zip}")
    print(f"  Output dir:  {args.processed_dir}")
    print("=" * 64)

    print("\n[1/4] Loading annotations...")
    id_to_filename, id_to_caption = load_annotations(annotations_json)
    paired_ids = [img_id for img_id in id_to_filename if img_id in id_to_caption]
    print(f"  Total paired figures: {len(paired_ids)}")

    if len(paired_ids) < TOTAL_NEEDED:
        raise RuntimeError(
            f"Only {len(paired_ids)} paired figures available, need {TOTAL_NEEDED}."
        )

    random.shuffle(paired_ids)
    selected_ids = paired_ids[:TOTAL_NEEDED]
    print(f"  Selected {TOTAL_NEEDED} figures for splits")

    print("\n[2/4] Extracting images from zip...")
    id_to_path = extract_images(images_zip, set(selected_ids), args.images_dir)
    print(f"  Extracted {len(id_to_path)} images to {args.images_dir}")

    missing = [img_id for img_id in selected_ids if img_id not in id_to_path]
    if missing:
        print(f"  WARNING: {len(missing)} images missing from zip, adjusting selection")
        selected_ids = [img_id for img_id in selected_ids if img_id in id_to_path]
        if len(selected_ids) < TOTAL_NEEDED:
            raise RuntimeError(f"Too few images after extraction: {len(selected_ids)}")
        selected_ids = selected_ids[:TOTAL_NEEDED]

    print("\n[3/4] Building and cleaning records...")
    all_records = build_records(id_to_filename, id_to_caption, id_to_path)
    id_set = set(selected_ids)
    all_records = [r for r in all_records if int(r["id"]) in id_set]
    id_order = {img_id: idx for idx, img_id in enumerate(selected_ids)}
    all_records.sort(key=lambda r: id_order[int(r["id"])])

    train_records = all_records[:TRAIN_SIZE]
    val_records = all_records[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    test_records = all_records[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    log_caption_stats(train_records + val_records + test_records, "all splits")

    print("\n[4/4] Writing JSONL splits...")
    write_jsonl(args.processed_dir / "train.jsonl", train_records)
    write_jsonl(args.processed_dir / "val.jsonl", val_records)
    write_jsonl(args.processed_dir / "test.jsonl", test_records)

    print(f"\nSplit summary (seed={SEED}):")
    print(f"  train: {len(train_records):>5} pairs -> {args.processed_dir/'train.jsonl'}")
    print(f"  val:   {len(val_records):>5} pairs -> {args.processed_dir/'val.jsonl'}")
    print(f"  test:  {len(test_records):>5} pairs -> {args.processed_dir/'test.jsonl'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
