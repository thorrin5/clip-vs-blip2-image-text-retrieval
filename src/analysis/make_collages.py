"""
Generovanie kvalitatívnych retrieval koláží pre Flickr30K a SciCap.

Skript podporuje kvalitatívnu časť porovnania CLIP vs BLIP-2. Používa rovnaké
modelové wrappery ako evaluačné skripty a vizualizuje vybrané úspešné aj
neúspešné prípady image-to-text retrievalu.

Koláže nie sú náhradou kvantitatívnych metrík Recall@K. Slúžia ako vizuálna
pomôcka pri obhajobe, aby bolo vidieť, čo znamená správny alebo nesprávny
výsledok na prvej pozícii rebríčka.
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

# Fine-tuned checkpointy nie sú súčasťou verejného repozitára. Pri lokálnom
# opätovnom generovaní SciCap koláží treba nastaviť tieto premenné prostredia.
CLIP_SCICAP_CKPT = Path(
    os.environ.get("CLIP_SCICAP_CKPT", BASE / "checkpoints/scicap/clip_scicap_finetuned.pt")
)
BLIP2_SCICAP_CKPT = Path(
    os.environ.get("BLIP2_SCICAP_CKPT", BASE / "checkpoints/scicap/blip2_scicap_finetuned.pt")
)
OUT_DIR = BASE / "figures/qualitative_examples/generated"

# Cieľové počty zásahov a chýb v koláži približne kopírujú R@1 výkonnosť.
# Dvojica (n_hits, n_miss) sa vždy skladá do troch panelov.
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
    """Načíta dostupný bežný font pre kreslenie textu do koláže."""
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
    """Načíta dostupný tučný font pre nadpisy a popisky v koláži."""
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
    """
    Vypočíta maticu skóre použitú pri výbere príkladov do koláže.

    Pri CLIP ide o skalárny súčin normalizovaných embeddingov. Pri BLIP-2 sa
    použije jeho ITC podobnosť, ktorá berie maximum cez obrazové query tokeny.
    """
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
    """Vyberie n položiek rovnomerne z kandidátskeho zoznamu pre vizuálnu pestrosť."""
    if n == 0:
        return []
    if len(pool) <= n:
        return pool[:n]
    step = len(pool) / n
    return [pool[int(i * step + step / 2)] for i in range(n)]


def _word_overlap(a: str, b: str) -> float:
    """Vypočíta Jaccardovu podobnosť slov medzi dvoma textmi."""
    wa, wb = set(a.lower().split()), set(b.lower().split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _rank_misses_by_wrongness(misses: list[dict], captions: list[str],
                               image_to_captions: dict) -> list[dict]:
    """
    Zoradí chybné prípady tak, aby boli na začiatku najjasnejšie omyly.

    Nízke prekrytie slov medzi predikovaným a správnym popisom znamená, že
    chyba je pre čitateľa ľahšie interpretovateľná.
    """
    def score(entry: dict) -> float:
        gt_indices = image_to_captions[entry["img_idx"]]
        pred = captions[entry["top1_cap_idx"]]
        return min(_word_overlap(captions[i], pred) for i in gt_indices)
    return sorted(misses, key=score)


def _rank_hits_by_diversity(hits: list[dict], captions: list[str],
                             image_to_captions: dict) -> list[dict]:
    """
    Zoradí správne prípady tak, aby boli uprednostnené rozmanité formulácie.

    Pri Flickr30K má jeden obrázok viac správnych popisov. Ak model nájde
    správny, ale inak formulovaný popis, panel je informatívnejší než pri
    identickej vete alebo takmer duplikovanom texte.
    """
    def score(entry: dict) -> float:
        gt_indices = image_to_captions[entry["img_idx"]]
        pred = captions[entry["top1_cap_idx"]]
        # Použije sa minimum cez všetky správne popisy, aby sa nepenalizoval
        # zásah len preto, že jeden z Flickr30K popisov má inú formuláciu.
        overlaps = [_word_overlap(captions[i], pred) for i in gt_indices]
        min_ov = min(overlaps)
        if pred in (captions[i] for i in gt_indices):
            return 1.0   # presná zhoda s niektorým správnym popisom ide na koniec
        if min_ov > 0.45:
            return 0.9   # takmer rovnaká formulácia má nižšiu prioritu
        return min_ov    # nižšia hodnota znamená rozmanitejší panel
    return sorted(hits, key=score)


def select_examples(
    sim: torch.Tensor,
    image_to_captions: dict,
    n_hits: int,
    n_miss: int,
    captions: list[str] | None = None,
) -> list[dict]:
    """
    Vyberie príklady pre koláž podľa toho, či top-1 retrieval trafil správny text.

    Správne prípady sa vyberajú rovnomerne pre vizuálnu pestrosť. Chybné
    prípady sa najprv zoradia podľa odlišnosti textov, aby koláž obsahovala
    interpretovateľné omyly a nie iba hraničné near-miss situácie.
    """
    hits, misses = [], []
    for img_idx in range(sim.shape[0]):
        top1 = int(sim[img_idx].argmax())
        gt   = set(image_to_captions[img_idx])
        entry = {"img_idx": img_idx, "top1_cap_idx": top1, "hit": top1 in gt}
        (hits if top1 in gt else misses).append(entry)

    # Chybné príklady s najodlišnejším popisom sú pre kvalitatívnu analýzu jasnejšie.
    if captions is not None and n_miss > 0:
        misses = _rank_misses_by_wrongness(misses, captions, image_to_captions)

    # Správne príklady s neidentickou formuláciou lepšie ukazujú sémantickú zhodu.
    if captions is not None and n_hits > 0:
        hits = _rank_hits_by_diversity(hits, captions, image_to_captions)

    chosen_hits  = _spread(hits,   n_hits)
    chosen_miss  = _spread(misses, n_miss)

    # Ak model neposkytne dosť zásahov alebo chýb, doplnia sa z druhej skupiny.
    if len(chosen_hits) < n_hits:
        extra = _spread(misses, n_hits - len(chosen_hits))
        chosen_hits += extra
    if len(chosen_miss) < n_miss:
        extra = _spread(hits, n_miss - len(chosen_miss))
        chosen_miss += extra

    # Panely sa prekladajú, aby neboli všetky zásahy alebo chyby zoskupené spolu.
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
    """
    Vykreslí vybrané retrieval príklady do jedného PNG súboru.

    Každý panel obsahuje obrázok, referenčný popis a top-1 text nájdený modelom.
    Farba rámu odlišuje správne a nesprávne prípady.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (COLLAGE_W, COLLAGE_H), BG_COLOR)
    draw   = ImageDraw.Draw(canvas)

    draw.text((COLLAGE_W // 2, TITLE_H // 2), title,
              fill=TEXT_COLOR, font=_bold(20), anchor="mm")

    for col, ex in enumerate(examples):
        x0 = PANEL_PAD + col * (PANEL_W + PANEL_PAD)
        y0 = TITLE_H + PANEL_PAD

        # Miniatúra sa vloží do pevného priestoru, aby mali všetky panely rovnaký rozmer.
        img = Image.open(dataset.image_paths[ex["img_idx"]]).convert("RGB")
        img.thumbnail((IMG_W, IMG_H), Image.LANCZOS)
        thumb = Image.new("RGB", (IMG_W, IMG_H), (210, 210, 210))
        thumb.paste(img, ((IMG_W - img.width) // 2, (IMG_H - img.height) // 2))

        border_col = BORDER_HIT if ex["hit"] else BORDER_MISS
        framed = Image.new("RGB", (IMG_W + 2 * BORDER_W, IMG_H + 2 * BORDER_W), border_col)
        framed.paste(thumb, (BORDER_W, BORDER_W))
        canvas.paste(framed, (x0 + PANEL_PAD - BORDER_W, y0))

        # Textová časť ukazuje referenčný popis a najvyššie hodnotený retrieval výsledok.
        ty = y0 + IMG_H + PANEL_PAD + 4
        tx = x0 + PANEL_PAD

        gt_caps   = [dataset.captions[i] for i in dataset.image_to_captions[ex["img_idx"]]]
        pred_text = dataset.captions[ex["top1_cap_idx"]]
        # Pri zásahu sa preferuje iný správny popis než vyhľadaný text, ak existuje.
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
    """Vygeneruje zero-shot kvalitatívne koláže pre Flickr30K."""
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
    """Vygeneruje fine-tuned kvalitatívne koláže pre SciCap."""
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
    """Načíta lokálny fine-tuned checkpoint potrebný pre SciCap koláže."""
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
    """Načíta konfiguráciu a vytvorí všetky podporované koláže."""
    config = load_config(str(CONFIG))
    flickr30k_collages(config)
    scicap_collages(config)
    print(f"\nAll collages saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
