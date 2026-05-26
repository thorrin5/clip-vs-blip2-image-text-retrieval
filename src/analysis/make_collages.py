"""
Purpose:
    Generate qualitative retrieval example collages for Flickr30K and SciCap.

Thesis context:
    This script supports the qualitative analysis part of the CLIP vs BLIP-2
    thesis. It loads the same model wrappers used by evaluation and visualizes
    selected image-to-text successes and failures.

Inputs:
    - Flickr30K and SciCap dataset paths from configs/experiment_config.yaml.
    - CLIP and BLIP-2 model wrappers.
    - Selected SciCap fine-tuned checkpoints.

Outputs:
    - PNG collages under results/figures/collages.

Defense note:
    The collages help explain what Recall@1 means in practice. They are not a
    replacement for quantitative metrics, but they make typical correct and
    incorrect retrievals visible during the defense.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from src.data.flickr30k import load_flickr30k_karpathy
from src.data.scicap import load_scicap
from src.models.openai_clip_wrapper import OfficialOpenAIClipWrapper
from src.models.hf_blip2_wrapper import HuggingFaceBlip2RetrievalWrapper
from src.utils.config import load_config, resolve_path

BASE = Path(__file__).resolve().parents[2]
CONFIG = BASE / "configs/experiment_config.yaml"

# Fine-tuned checkpoints are intentionally not included in the public repo.
# Set these environment variables when regenerating SciCap qualitative examples.
CLIP_SCICAP_CKPT = Path(
    os.environ.get("CLIP_SCICAP_CKPT", BASE / "checkpoints/scicap/clip_scicap_finetuned.pt")
)
BLIP2_SCICAP_CKPT = Path(
    os.environ.get("BLIP2_SCICAP_CKPT", BASE / "checkpoints/scicap/blip2_scicap_finetuned.pt")
)
OUT_DIR = BASE / "figures/qualitative_examples/generated"

# Target hit counts per collage — mirrors actual R@1 performance
# (n_hits, n_miss) summing to 3 panels
TARGET: dict[str, tuple[int, int]] = {
    "flickr30k_clip":   (2, 1),   # ~75% R@1
    "flickr30k_blip2":  (3, 0),   # ~97% R@1
    "scicap_clip":      (1, 2),   # ~34% R@1
    "scicap_blip2":     (1, 2),   # ~30% R@1
}

# ── Layout ──────────────────────────────────────────────────────────────────
IMG_W, IMG_H   = 420, 315
PANEL_PAD      = 22
TEXT_AREA_H    = 170
TITLE_H        = 65
PANEL_W        = IMG_W + 2 * PANEL_PAD
PANEL_H        = IMG_H + TEXT_AREA_H + PANEL_PAD
N_PANELS       = 3
COLLAGE_W      = N_PANELS * PANEL_W + (N_PANELS + 1) * PANEL_PAD
COLLAGE_H      = TITLE_H + PANEL_H + 2 * PANEL_PAD
BG_COLOR       = (248, 248, 248)
BORDER_HIT     = (34, 139, 34)
BORDER_MISS    = (178, 34, 34)
TEXT_COLOR     = (30,  30,  30)
GT_COLOR       = (0,  100,   0)
PRED_COLOR     = (0,    0,  160)
BORDER_W       = 4


# ── Font helpers ────────────────────────────────────────────────────────────
def _font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return ImageFont.load_default()


def _bold(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            pass
    return _font(size)


# ── Similarity computation ───────────────────────────────────────────────────
def compute_similarity(wrapper, dataset, batch_size: int = 16) -> torch.Tensor:
    """Compute the image-to-text score matrix used for example selection."""
    img_feats = wrapper.encode_images(dataset.image_paths, batch_size=batch_size)
    txt_feats = wrapper.encode_texts(dataset.captions, batch_size=256)
    if hasattr(wrapper, "compute_itc_similarity"):
        sim = wrapper.compute_itc_similarity(img_feats, txt_feats, image_batch_size=batch_size)
    else:
        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
        txt_feats = txt_feats / txt_feats.norm(dim=-1, keepdim=True)
        sim = img_feats @ txt_feats.t()
    return sim.cpu()


# ── Example selection ────────────────────────────────────────────────────────
def _spread(pool: list, n: int) -> list:
    """Pick n items evenly spread across pool for visual diversity."""
    if n == 0:
        return []
    if len(pool) <= n:
        return pool[:n]
    step = len(pool) / n
    return [pool[int(i * step + step / 2)] for i in range(n)]


def _word_overlap(a: str, b: str) -> float:
    """Jaccard word overlap between two strings. 0 = completely different."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _rank_misses_by_wrongness(misses: list[dict], captions: list[str],
                               image_to_captions: dict) -> list[dict]:
    """Sort misses so most clearly-wrong come first (lowest caption overlap)."""
    def score(entry: dict) -> float:
        gt_indices = image_to_captions[entry["img_idx"]]
        pred = captions[entry["top1_cap_idx"]]
        return min(_word_overlap(captions[i], pred) for i in gt_indices)
    return sorted(misses, key=score)


def _rank_hits_by_diversity(hits: list[dict], captions: list[str],
                             image_to_captions: dict) -> list[dict]:
    """Sort hits so those with a visually distinct retrieved caption come first.

    Prefer hits where the retrieved caption is worded differently from the
    first gt caption — this makes the collage panel more informative (two
    different sentences that both describe the image correctly).
    Hits where retrieved == gt_caps[0] (identical text) are pushed to the end.
    Hits with very high overlap (>0.45) are also deprioritised — they look like
    near-duplicates and make boring panels.
    """
    def score(entry: dict) -> float:
        gt_indices = image_to_captions[entry["img_idx"]]
        pred = captions[entry["top1_cap_idx"]]
        # Use min overlap across all gt captions so we don't penalise a hit
        # just because one of the 5 Flickr30K captions is worded differently.
        overlaps = [_word_overlap(captions[i], pred) for i in gt_indices]
        min_ov = min(overlaps)
        if pred in (captions[i] for i in gt_indices):
            return 1.0   # exact match with any gt caption — push to end
        if min_ov > 0.45:
            return 0.9   # near-duplicate wording — deprioritise
        return min_ov    # lower = more diverse = better
    return sorted(hits, key=score)


def select_examples(
    sim: torch.Tensor,
    image_to_captions: dict,
    n_hits: int,
    n_miss: int,
    captions: list[str] | None = None,
) -> list[dict]:
    """
    Classify all images as hit/miss.
    Hits: spread evenly for visual diversity.
    Misses: rank by caption dissimilarity (prefer clearly-wrong over near-miss),
            then spread evenly within that ranked pool.
    """
    hits, misses = [], []
    for img_idx in range(sim.shape[0]):
        top1 = int(sim[img_idx].argmax())
        gt   = set(image_to_captions[img_idx])
        entry = {"img_idx": img_idx, "top1_cap_idx": top1, "hit": top1 in gt}
        (hits if top1 in gt else misses).append(entry)

    # Sort misses so clearly-wrong ones come first
    if captions is not None and n_miss > 0:
        misses = _rank_misses_by_wrongness(misses, captions, image_to_captions)

    # Sort hits so diverse (non-identical) retrieved captions come first
    if captions is not None and n_hits > 0:
        hits = _rank_hits_by_diversity(hits, captions, image_to_captions)

    chosen_hits  = _spread(hits,   n_hits)
    chosen_miss  = _spread(misses, n_miss)

    # Fallback: if not enough hits/misses, draw from the other pool
    if len(chosen_hits) < n_hits:
        extra = _spread(misses, n_hits - len(chosen_hits))
        chosen_hits += extra
    if len(chosen_miss) < n_miss:
        extra = _spread(hits, n_miss - len(chosen_miss))
        chosen_miss += extra

    # Interleave: hit, miss, hit, miss … so panels alternate rather than cluster
    result: list[dict] = []
    hi, mi = iter(chosen_hits), iter(chosen_miss)
    for _ in range(n_hits + n_miss):
        try:
            result.append(next(hi) if len([x for x in result if x["hit"]]) < n_hits else next(mi))
        except StopIteration:
            break
    return result[: n_hits + n_miss]


# ── Collage drawing ──────────────────────────────────────────────────────────
def draw_collage(examples: list[dict], dataset, title: str, out_path: Path) -> None:
    """Render selected retrieval examples into a single PNG."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (COLLAGE_W, COLLAGE_H), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    draw.text((COLLAGE_W // 2, TITLE_H // 2), title,
              fill=TEXT_COLOR, font=_bold(20), anchor="mm")

    for col, ex in enumerate(examples):
        x0 = PANEL_PAD + col * (PANEL_W + PANEL_PAD)
        y0 = TITLE_H + PANEL_PAD

        # ── image thumbnail ──────────────────────────────────────────────
        img = Image.open(dataset.image_paths[ex["img_idx"]]).convert("RGB")
        img.thumbnail((IMG_W, IMG_H), Image.LANCZOS)
        thumb = Image.new("RGB", (IMG_W, IMG_H), (210, 210, 210))
        thumb.paste(img, ((IMG_W - img.width) // 2, (IMG_H - img.height) // 2))

        border_col = BORDER_HIT if ex["hit"] else BORDER_MISS
        framed = Image.new("RGB", (IMG_W + 2 * BORDER_W, IMG_H + 2 * BORDER_W), border_col)
        framed.paste(thumb, (BORDER_W, BORDER_W))
        canvas.paste(framed, (x0 + PANEL_PAD - BORDER_W, y0))

        # ── caption area ─────────────────────────────────────────────────
        ty = y0 + IMG_H + PANEL_PAD + 4
        tx = x0 + PANEL_PAD

        gt_caps   = [dataset.captions[i] for i in dataset.image_to_captions[ex["img_idx"]]]
        pred_text = dataset.captions[ex["top1_cap_idx"]]
        # For hits: show a gt caption different from the retrieved one (more informative)
        gt_text = next((c for c in gt_caps if c != pred_text), gt_caps[0])

        draw.text((tx, ty), "Ground truth:", fill=GT_COLOR, font=_bold(13))
        ty += 17
        for line in textwrap.wrap(gt_text, 47):
            draw.text((tx, ty), line, fill=GT_COLOR, font=_font(12))
            ty += 14

        ty += 7
        label = "✓ Retrieved (correct):" if ex["hit"] else "✗ Retrieved (wrong):"
        draw.text((tx, ty), label, fill=PRED_COLOR, font=_bold(13))
        ty += 17
        for line in textwrap.wrap(pred_text, 47):
            draw.text((tx, ty), line, fill=PRED_COLOR, font=_font(12))
            ty += 14

    canvas.save(out_path, dpi=(150, 150))
    print(f"  Saved → {out_path.name}")


# ── Per-dataset runners ──────────────────────────────────────────────────────
def flickr30k_collages(config: dict) -> None:
    dataset = load_flickr30k_karpathy(
        dataset_json_path=resolve_path(config, "flickr30k_karpathy_json"),
        images_dir=resolve_path(config, "flickr30k_images"),
        split="test",
    )
    jobs = [
        ("clip",  "CLIP",   (2, 1),
         lambda: OfficialOpenAIClipWrapper(config["models"]["clip"]["model_name"], device="cuda"),
         None),
        ("blip2", "BLIP-2", (3, 0),
         lambda: HuggingFaceBlip2RetrievalWrapper(config["models"]["blip2"]["model_name"], device="cuda", precision="fp32"),
         None),
    ]
    for key, label, (nh, nm), make_wrapper, ckpt in jobs:
        print(f"\n=== Flickr30K {label} ===")
        wrapper = make_wrapper()
        if ckpt:
            _load_checkpoint(wrapper, label, ckpt)
        wrapper.model.eval()
        sim = compute_similarity(wrapper, dataset)
        examples = select_examples(sim, dataset.image_to_captions, nh, nm, dataset.captions)
        hits_found = sum(e["hit"] for e in examples)
        print(f"  Examples: {hits_found} correct, {len(examples)-hits_found} wrong (target {nh}/{nm})")
        title = f"Flickr30K — {label} zero-shot  |  Image→Text Retrieval (R@1)"
        draw_collage(examples, dataset, title,
                     OUT_DIR / f"flickr30k_{key}_examples.png")
        del wrapper; torch.cuda.empty_cache()


def scicap_collages(config: dict) -> None:
    scicap_dir = resolve_path(config, "scicap_processed_dir")
    dataset = load_scicap(scicap_dir, split="test")
    jobs = [
        ("clip",  "CLIP",   (1, 2),
         lambda: OfficialOpenAIClipWrapper(config["models"]["clip"]["model_name"], device="cuda"),
         CLIP_SCICAP_CKPT),
        ("blip2", "BLIP-2", (1, 2),
         lambda: HuggingFaceBlip2RetrievalWrapper(config["models"]["blip2"]["model_name"], device="cuda", precision="fp32"),
         BLIP2_SCICAP_CKPT),
    ]
    for key, label, (nh, nm), make_wrapper, ckpt in jobs:
        print(f"\n=== SciCap {label} fine-tuned ===")
        wrapper = make_wrapper()
        _load_checkpoint(wrapper, label, ckpt)
        wrapper.model.eval()
        sim = compute_similarity(wrapper, dataset)
        examples = select_examples(sim, dataset.image_to_captions, nh, nm, dataset.captions)
        hits_found = sum(e["hit"] for e in examples)
        print(f"  Examples: {hits_found} correct, {len(examples)-hits_found} wrong (target {nh}/{nm})")
        title = f"SciCap — {label} fine-tuned (45k)  |  Image→Text Retrieval (R@1)"
        draw_collage(examples, dataset, title,
                     OUT_DIR / f"scicap_{key}_examples.png")
        del wrapper; torch.cuda.empty_cache()


def _load_checkpoint(wrapper, label: str, ckpt: Path) -> None:
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{label} checkpoint not found at {ckpt}. "
            "Set CLIP_SCICAP_CKPT or BLIP2_SCICAP_CKPT to regenerate SciCap collages."
        )
    print(f"  Loading checkpoint: {ckpt.name}")
    data = torch.load(ckpt, map_location="cuda")
    if label == "CLIP":
        wrapper.model.load_state_dict(data["model_state_dict"])
    else:
        for p in wrapper.model.parameters():
            p.requires_grad = False
        wrapper.model.load_state_dict(data["trainable_state_dict"], strict=False)


# ── Entry point ──────────────────────────────────────────────────────────────
def main() -> None:
    config = load_config(str(CONFIG))
    flickr30k_collages(config)
    scicap_collages(config)
    print(f"\nAll collages saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
