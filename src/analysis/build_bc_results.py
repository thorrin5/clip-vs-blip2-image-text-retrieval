"""
Purpose:
    Assemble the final bachelor-thesis result archive from already generated
    experiment artifacts.

Thesis context:
    This script is used after CLIP and BLIP-2 evaluation/fine-tuning runs. It
    packages selected Flickr30K and SciCap JSON files, histories, collages, and
    plots into the bc_results archive used for thesis tables and defense
    preparation.

Inputs:
    - Selected raw JSON result files under results/raw and results/scicap.
    - Selected training-history JSON files.
    - Optional qualitative collages and training plots.

Outputs:
    - bc_results/ directory with raw JSON copies, CSV summaries, README, and
      optional figures.
    - bc_results.zip archive.

Defense note:
    This is not an evaluation script. It is the traceability layer that makes
    the final reported numbers reproducible from stored raw outputs.

Additional implementation notes:

This script selects the raw JSON files, training histories, collages, and
training plots that back the current thesis tables. It is a result-packaging
utility, not an evaluation script: it copies existing artifacts and builds CSV
summaries from them.

Structure:
  bc_results/
  ├── README.md
  ├── results/
  │   ├── full_comparison.csv
  │   ├── flickr30k_results.csv
  │   ├── ai2d_results.csv
  │   ├── scicap_results.csv
  │   └── training_history/
  │       ├── clip_ai2d_epochs.json
  │       ├── blip2_ai2d_epochs.json
  │       ├── clip_scicap_45k_epochs.json
  │       └── blip2_scicap_45k_epochs.json
  ├── raw_json/
  │   ├── flickr30k_clip_zero_shot.json
  │   ├── flickr30k_blip2_zero_shot.json
  │   ├── ai2d_clip_zero_shot.json
  │   ├── ai2d_blip2_zero_shot.json
  │   ├── ai2d_clip_fine_tuned.json
  │   ├── ai2d_blip2_fine_tuned.json
  │   ├── scicap_clip_zero_shot.json
  │   ├── scicap_blip2_zero_shot.json
  │   ├── scicap_clip_fine_tuned_45k.json
  │   └── scicap_blip2_fine_tuned_45k.json
  └── collages/
      ├── flickr30k_clip_examples.png
      ├── flickr30k_blip2_examples.png
      ├── scicap_clip_examples.png
      └── scicap_blip2_examples.png
"""

from __future__ import annotations

import csv
import io
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results"
OUT_DIR = BASE / "bc_results_export"

# ---------------------------------------------------------------------------
# Canonical source files
# ---------------------------------------------------------------------------
RAW_FILES = {
    "flickr30k_clip_zero_shot.json":
        RESULTS / "raw_json/flickr30k_clip_zero_shot.json",
    "flickr30k_blip2_zero_shot.json":
        RESULTS / "raw_json/flickr30k_blip2_zero_shot.json",
    "ai2d_clip_zero_shot.json":
        RESULTS / "raw_json/ai2d_clip_zero_shot.json",
    "ai2d_blip2_zero_shot.json":
        RESULTS / "raw_json/ai2d_blip2_zero_shot.json",
    "ai2d_clip_fine_tuned.json":
        RESULTS / "raw_json/ai2d_clip_fine_tuned.json",
    "ai2d_blip2_fine_tuned.json":
        RESULTS / "raw_json/ai2d_blip2_fine_tuned.json",
    "scicap_clip_zero_shot.json":
        RESULTS / "raw_json/scicap_clip_zero_shot.json",
    "scicap_blip2_zero_shot.json":
        RESULTS / "raw_json/scicap_blip2_zero_shot.json",
    "scicap_clip_fine_tuned_45k.json":
        RESULTS / "raw_json/scicap_clip_fine_tuned_45k.json",
    "scicap_blip2_fine_tuned_45k.json":
        RESULTS / "raw_json/scicap_blip2_fine_tuned_45k.json",
}

HISTORY_FILES = {
    "clip_ai2d_epochs.json":
        RESULTS / "training_history/clip_ai2d_epochs.json",
    "blip2_ai2d_epochs.json":
        RESULTS / "training_history/blip2_ai2d_epochs.json",
    "clip_scicap_45k_epochs.json":
        RESULTS / "training_history/clip_scicap_45k_epochs.json",
    "blip2_scicap_45k_epochs.json":
        RESULTS / "training_history/blip2_scicap_45k_epochs.json",
}

COLLAGE_FILES = {
    "flickr30k_clip_examples.png":
        BASE / "figures/qualitative_examples/flickr30k_clip_examples.png",
    "flickr30k_blip2_examples.png":
        BASE / "figures/qualitative_examples/flickr30k_blip2_examples.png",
    "scicap_clip_examples.png":
        BASE / "figures/qualitative_examples/scicap_clip_examples.png",
    "scicap_blip2_examples.png":
        BASE / "figures/qualitative_examples/scicap_blip2_examples.png",
}

TRAINING_PLOT_FILES = {
    "ai2d_training_curves.png":
        BASE / "figures/plots/ai2d_training_curves.png",
    "scicap_training_curves.png":
        BASE / "figures/plots/scicap_training_curves.png",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(d: dict, *keys: str, default: Any = None) -> Any:
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _f(v: Any, dec: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{dec}f}"
    return str(v)


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CSV builders
# ---------------------------------------------------------------------------

RECALL_FIELDS = [
    "model", "dataset", "variant", "split",
    "i2t_R1", "i2t_R5", "i2t_R10",
    "t2i_R1", "t2i_R5", "t2i_R10",
    "mean_R1", "mean_R5", "mean_R10",
    "img_enc_s", "img_per_sec", "itm_rerank_s", "total_s",
    "train_samples", "lr", "effective_batch", "epochs_completed",
    "best_epoch", "best_val_mean_R1", "early_stopped",
]


def _row_from_json(label: str, dataset: str, variant: str, path: Path) -> dict:
    """Flatten one raw metric JSON into the common CSV schema."""
    d = _load(path)
    m = d.get("metrics", {})
    t = m.get("timing", {})
    tr = d.get("training", {})
    ds = d.get("dataset", {})

    img_enc = t.get("image_encoding_time", 0)
    txt_enc = t.get("text_encoding_time", 0)
    sim_t = t.get("similarity_time", 0)
    itm_t = t.get("itm_rerank_time") or (
        t.get("i2t_itm_rerank_time", 0) + t.get("t2i_itm_rerank_time", 0)
    )
    json_total = t.get("total_time", 0)
    true_total = max(json_total, img_enc + txt_enc + sim_t + itm_t)
    n_img = m.get("num_images") or ds.get("num_images") or 0

    return {
        "model": label,
        "dataset": dataset,
        "variant": variant,
        "split": ds.get("split", "test"),
        "i2t_R1":   _f(_get(m, "image_to_text", "R@1")),
        "i2t_R5":   _f(_get(m, "image_to_text", "R@5")),
        "i2t_R10":  _f(_get(m, "image_to_text", "R@10")),
        "t2i_R1":   _f(_get(m, "text_to_image", "R@1")),
        "t2i_R5":   _f(_get(m, "text_to_image", "R@5")),
        "t2i_R10":  _f(_get(m, "text_to_image", "R@10")),
        "mean_R1":  _f(_get(m, "mean", "R@1")),
        "mean_R5":  _f(_get(m, "mean", "R@5")),
        "mean_R10": _f(_get(m, "mean", "R@10")),
        "img_enc_s":    _f(img_enc, 1),
        "img_per_sec":  _f(n_img / img_enc if img_enc else None, 1),
        "itm_rerank_s": _f(itm_t, 0) if itm_t else "-",
        "total_s":      _f(true_total, 0),
        "train_samples":    tr.get("train_samples", "-"),
        "lr":               tr.get("lr", "-"),
        "effective_batch":  tr.get("effective_batch_size") or tr.get("batch_size") or "-",
        "epochs_completed": tr.get("epochs_completed", "-"),
        "best_epoch":       tr.get("best_epoch", "-"),
        "best_val_mean_R1": _f(tr.get("best_val_mean_R1")),
        "early_stopped":    _get(tr, "early_stopping", "stopped_early", default="-"),
    }


ENTRIES = [
    ("CLIP",   "Flickr30K", "zero-shot",  "flickr30k_clip_zero_shot.json"),
    ("BLIP-2", "Flickr30K", "zero-shot",  "flickr30k_blip2_zero_shot.json"),
    ("CLIP",   "AI2D",      "zero-shot",  "ai2d_clip_zero_shot.json"),
    ("BLIP-2", "AI2D",      "zero-shot",  "ai2d_blip2_zero_shot.json"),
    ("CLIP",   "AI2D",      "fine-tuned", "ai2d_clip_fine_tuned.json"),
    ("BLIP-2", "AI2D",      "fine-tuned", "ai2d_blip2_fine_tuned.json"),
    ("CLIP",   "SciCap",    "zero-shot",  "scicap_clip_zero_shot.json"),
    ("BLIP-2", "SciCap",    "zero-shot",  "scicap_blip2_zero_shot.json"),
    ("CLIP",   "SciCap",    "fine-tuned", "scicap_clip_fine_tuned_45k.json"),
    ("BLIP-2", "SciCap",    "fine-tuned", "scicap_blip2_fine_tuned_45k.json"),
]


def build_csvs(raw_dir: Path) -> dict[str, str]:
    """Return `{filename: csv_text}` for full and per-dataset result tables."""
    all_rows = []
    for label, dataset, variant, fname in ENTRIES:
        path = raw_dir / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        all_rows.append(_row_from_json(label, dataset, variant, path))

    def _to_csv(rows: list[dict]) -> str:
        buf = io.StringIO()
        if not rows:
            return ""
        w = csv.DictWriter(buf, fieldnames=RECALL_FIELDS)
        w.writeheader()
        w.writerows(rows)
        return buf.getvalue()

    csvs = {"full_comparison.csv": _to_csv(all_rows)}
    for ds in ["Flickr30K", "AI2D", "SciCap"]:
        rows = [r for r in all_rows if r["dataset"] == ds]
        csvs[f"{ds.lower()}_results.csv"] = _to_csv(rows)
    return csvs


# ---------------------------------------------------------------------------
# README builder
# ---------------------------------------------------------------------------

def build_readme(entries: list[dict]) -> str:
    def _l(label: str, dataset: str, variant: str, field: str) -> str:
        for e in entries:
            if e["model"] == label and e["dataset"] == dataset and e["variant"] == variant:
                v = e.get(field, "-")
                return str(v) if v not in (None, "", "nan") else "-"
        return "-"

    clip_ai2d_ep    = _l("CLIP",   "AI2D",    "fine-tuned", "epochs_completed")
    clip_ai2d_best  = _l("CLIP",   "AI2D",    "fine-tuned", "best_epoch")
    blip_ai2d_ep    = _l("BLIP-2", "AI2D",    "fine-tuned", "epochs_completed")
    blip_ai2d_best  = _l("BLIP-2", "AI2D",    "fine-tuned", "best_epoch")
    clip_sc_ep      = _l("CLIP",   "SciCap",  "fine-tuned", "epochs_completed")
    clip_sc_best    = _l("CLIP",   "SciCap",  "fine-tuned", "best_epoch")
    blip_sc_ep      = _l("BLIP-2", "SciCap",  "fine-tuned", "epochs_completed")
    blip_sc_best    = _l("BLIP-2", "SciCap",  "fine-tuned", "best_epoch")

    clip_ai2d_bval  = _l("CLIP",   "AI2D",    "fine-tuned", "best_val_mean_R1")
    blip_ai2d_bval  = _l("BLIP-2", "AI2D",    "fine-tuned", "best_val_mean_R1")
    clip_sc_bval    = _l("CLIP",   "SciCap",  "fine-tuned", "best_val_mean_R1")
    blip_sc_bval    = _l("BLIP-2", "SciCap",  "fine-tuned", "best_val_mean_R1")

    # recall metrics
    def r(label, ds, var, k): return _l(label, ds, var, k)

    return f"""# Comparative Analysis of CLIP vs BLIP-2 for Image-Text Retrieval
## Bachelor Thesis Experiment Results

This archive is the complete, self-contained record of all experiments conducted
for the bachelor thesis. It contains raw JSON metric payloads, CSV summary tables,
per-epoch training histories, and retrieval example collages.

---

## 1. Archive Structure

Every path below is relative to the `bc_results/` root folder.

```
bc_results/
│
├── README.md                              ← this file (all context for GPT/Codex)
│
├── results/
│   ├── full_comparison.csv               ← all 10 runs in one CSV (23 columns)
│   ├── flickr30k_results.csv             ← Flickr30K rows only (2 rows)
│   ├── ai2d_results.csv                  ← AI2D rows only (4 rows)
│   ├── scicap_results.csv                ← SciCap rows only (4 rows)
│   └── training_history/
│       ├── clip_ai2d_epochs.json         ← per-epoch val metrics, CLIP fine-tuned on AI2D
│       ├── blip2_ai2d_epochs.json        ← per-epoch val metrics, BLIP-2 fine-tuned on AI2D
│       ├── clip_scicap_45k_epochs.json   ← per-epoch val metrics, CLIP fine-tuned on SciCap 45k
│       └── blip2_scicap_45k_epochs.json  ← per-epoch val metrics, BLIP-2 fine-tuned on SciCap 45k
│
├── raw_json/
│   ├── flickr30k_clip_zero_shot.json     ← CLIP zero-shot on Flickr30K test set
│   ├── flickr30k_blip2_zero_shot.json    ← BLIP-2 zero-shot on Flickr30K test set
│   ├── ai2d_clip_zero_shot.json          ← CLIP zero-shot on AI2D test set
│   ├── ai2d_blip2_zero_shot.json         ← BLIP-2 zero-shot on AI2D test set
│   ├── ai2d_clip_fine_tuned.json         ← CLIP fine-tuned on AI2D, tested on AI2D
│   ├── ai2d_blip2_fine_tuned.json        ← BLIP-2 fine-tuned on AI2D, tested on AI2D
│   ├── scicap_clip_zero_shot.json        ← CLIP zero-shot on SciCap test set
│   ├── scicap_blip2_zero_shot.json       ← BLIP-2 zero-shot on SciCap test set
│   ├── scicap_clip_fine_tuned_45k.json   ← CLIP fine-tuned on SciCap 45k train, tested on SciCap
│   └── scicap_blip2_fine_tuned_45k.json  ← BLIP-2 fine-tuned on SciCap 45k train, tested on SciCap
│
├── collages/
│   ├── flickr30k_clip_examples.png       ← 3 retrieval examples, CLIP zero-shot, Flickr30K
│   ├── flickr30k_blip2_examples.png      ← 3 retrieval examples, BLIP-2 zero-shot, Flickr30K
│   ├── scicap_clip_examples.png          ← 3 retrieval examples, CLIP fine-tuned, SciCap
│   └── scicap_blip2_examples.png         ← 3 retrieval examples, BLIP-2 fine-tuned, SciCap
└── training_plots/
    ├── ai2d_training_curves.png          ← val Mean R@1 + train loss per epoch, AI2D
    └── scicap_training_curves.png        ← val Mean R@1 + train loss per epoch, SciCap
```

---

## 2. CSV Column Reference

All four CSV files share the same 23 columns:

| Column             | Type    | Description                                                       |
|--------------------|---------|-------------------------------------------------------------------|
| model              | string  | "CLIP" or "BLIP-2"                                               |
| dataset            | string  | "Flickr30K", "AI2D", or "SciCap"                                 |
| variant            | string  | "zero-shot" or "fine-tuned"                                      |
| split              | string  | always "test"                                                    |
| i2t_R1             | float   | Image-to-Text Recall@1 (%) — given image, find correct caption   |
| i2t_R5             | float   | Image-to-Text Recall@5 (%)                                       |
| i2t_R10            | float   | Image-to-Text Recall@10 (%)                                      |
| t2i_R1             | float   | Text-to-Image Recall@1 (%) — given caption, find correct image   |
| t2i_R5             | float   | Text-to-Image Recall@5 (%)                                       |
| t2i_R10            | float   | Text-to-Image Recall@10 (%)                                      |
| mean_R1            | float   | Mean of i2t_R1 and t2i_R1 — primary comparison metric           |
| mean_R5            | float   | Mean of i2t_R5 and t2i_R5                                        |
| mean_R10           | float   | Mean of i2t_R10 and t2i_R10                                      |
| img_enc_s          | float   | Seconds to encode all test images into embedding vectors          |
| img_per_sec        | float   | Image encoding throughput (images per second)                    |
| itm_rerank_s       | float   | Seconds for BLIP-2 ITM reranking (empty "-" for CLIP)            |
| total_s            | float   | Total inference time (img_enc + txt_enc + similarity + ITM)      |
| train_samples      | int     | Number of training pairs used for fine-tuning ("-" = zero-shot)  |
| lr                 | float   | Peak learning rate used during fine-tuning                       |
| effective_batch    | int     | Effective batch size (micro_batch × grad_accum_steps)            |
| epochs_completed   | int     | Total training epochs run before early stopping                  |
| best_epoch         | int     | Epoch at which best validation Mean R@1 was achieved             |
| best_val_mean_R1   | float   | Best validation Mean R@1 seen during training                    |
| early_stopped      | bool    | Whether training stopped early via patience mechanism            |

---

## 3. Raw JSON Schema

Each file in `raw_json/` follows this top-level structure:

```json
{{
  "dataset": {{
    "name": "...",
    "split": "test",
    "num_images": 1000,
    "num_captions": 1000
  }},
  "metrics": {{
    "num_images": 1000,
    "num_texts": 1000,
    "image_to_text": {{"R@1": 34.3, "R@5": 50.6, "R@10": 58.1}},
    "text_to_image": {{"R@1": 33.6, "R@5": 50.5, "R@10": 58.0}},
    "mean":          {{"R@1": 34.0, "R@5": 50.6, "R@10": 58.1}},
    "timing": {{
      "image_encoding_time": 9.6,
      "text_encoding_time":  0.8,
      "similarity_time":     0.02,
      "itm_rerank_time":     null,       // present only for BLIP-2
      "i2t_itm_rerank_time": null,       // alternative ITM key (BLIP-2 only)
      "t2i_itm_rerank_time": null,       // alternative ITM key (BLIP-2 only)
      "total_time":          10.4
    }}
  }},
  "training": {{                         // present only for fine-tuned runs
    "train_samples":    45000,
    "lr":               1e-06,
    "batch_size":       32,
    "effective_batch_size": 32,
    "epochs_completed": 12,
    "best_epoch":       7,
    "best_val_mean_R1": 34.5,
    "early_stopping":   {{"stopped_early": true, "patience": 6}}
  }}
}}
```

For BLIP-2 ITM timing: some runs store a single `itm_rerank_time`, others store
`i2t_itm_rerank_time` + `t2i_itm_rerank_time`. The CSV `itm_rerank_s` column
always uses their sum.

---

## 4. Training History JSON Schema

Each file in `results/training_history/` is an array of epoch records:

```json
[
  {{
    "epoch": 1,
    "train_loss": 2.34,
    "val_i2t_R1": 18.4,
    "val_t2i_R1": 17.9,
    "val_mean_R1": 18.2,
    "val_mean_R5": 32.1,
    "val_mean_R10": 40.5,
    "lr": 1e-06,
    "epoch_time_s": 312.4
  }},
  ...
]
```

These are saved at the end of each training epoch by the fine-tuning scripts.

---

## 5. Models

### CLIP ViT-L/14
- **Library:** Official OpenAI CLIP (`pip install git+https://github.com/openai/CLIP`)
- **Model ID:** `ViT-L/14`
- **Architecture:** Vision Transformer ViT-L/14 (307M params) + text transformer (123M)
- **Total parameters:** ~430M
- **Retrieval method:** Cosine similarity between normalized image and text embeddings
- **Inference pipeline:**
  1. Encode all images → matrix of shape [N_img, 768]
  2. Encode all texts → matrix of shape [N_txt, 768]
  3. Normalize both, compute dot-product similarity matrix [N_img, N_txt]
  4. R@K computed from row-wise and column-wise rankings
- **Fine-tuning:** Full model, contrastive loss (InfoNCE), in-batch negatives

### BLIP-2 (Salesforce/blip2-itm-vit-g)
- **Library:** HuggingFace Transformers (`transformers==4.45.2`)
- **Model ID:** `Salesforce/blip2-itm-vit-g`
- **Architecture:** ViT-G/14 (1.3B) + Q-Former (188M) + text encoder (125M)
- **Total parameters:** ~1.5B; trainable during fine-tuning: ~188M (Q-Former only)
- **Retrieval method:** Two-stage
  1. ITC (Image-Text Contrastive) — fast embedding similarity, scores all N×M pairs
  2. ITM (Image-Text Matching) — BLIP-2 cross-attention reranking of top-128 candidates per query
- **Fine-tuning:** Q-Former + query_tokens + projections + ITM head only; ViT-G frozen

---

## 6. Datasets

### Flickr30K
- **Domain:** Natural photographs (Flickr)
- **Test split:** 1,000 images, 5 captions each = 5,000 captions
- **Split source:** Karpathy splits (standard benchmark)
- **Evaluation:** i2t: retrieve 1 of 5 correct captions per image; t2i: retrieve 1 image per caption
- **Fine-tuning:** None (zero-shot only)

### AI2D (Allen Institute for AI Diagrams)
AI2D entries are retained as historical exploratory material only. The final
thesis comparison is based on Flickr30K and SciCap.

- **Domain:** Science diagram images with diagram-specific captions
- **Test split:** 310 images, 1 caption each
- **Train split:** 2,470 pairs
- **Evaluation:** 1:1 matching (one caption per image)
- **Fine-tuning:** Both CLIP and BLIP-2 fine-tuned on 2,470 training pairs

### SciCap
- **Domain:** Scientific figures from arXiv papers (physics, CS, biology, etc.)
- **Source:** HuggingFace dataset `CrowdAILab/scicap`, hidden-test split
- **Total available:** 47,639 figure-caption pairs
- **Splits used (seed=42):**
  - train: 45,000 pairs
  - val:   1,000 pairs
  - test:  1,000 pairs
- **Evaluation:** 1:1 matching (one caption per image)
- **Fine-tuning:** Both CLIP and BLIP-2 fine-tuned on 45,000 training pairs

#### SciCap Caption Preprocessing
Raw captions begin with a figure label (e.g. "Figure 3: ...") and are often
multiple sentences. Preprocessing steps:
1. Strip figure label prefix (regex: `^\\s*(?:FIGURE|Fig\\.?)\\s*\\d+[.:\\)]?\\s*`)
2. Normalize whitespace
3. Take first sentence only
4. Exception: if first sentence has fewer than 5 words or 30 characters, append second sentence
5. Result fed to CLIP tokenizer (max 77 tokens; long captions silently truncated)

Effect: avg caption length 37.1 words → 15.7 words; ~63% of captions shortened.

---

## 7. Fine-tuning Configuration

### CLIP Fine-tuning
| Parameter            | AI2D             | SciCap (45k)      |
|----------------------|-----------------|-------------------|
| Train samples        | 2,470            | 45,000             |
| Learning rate        | 1e-6             | 1e-6               |
| Batch size           | 32               | 32                 |
| Grad accum steps     | 1                | 1                  |
| Effective batch      | 32               | 32                 |
| In-batch negatives   | 31               | 31                 |
| Optimizer            | AdamW            | AdamW              |
| LR schedule          | none             | none               |
| Grad clipping        | none             | none               |
| Epochs completed     | {clip_ai2d_ep}   | {clip_sc_ep}       |
| Best epoch           | {clip_ai2d_best} | {clip_sc_best}     |
| Best val Mean R@1    | {clip_ai2d_bval} | {clip_sc_bval}     |
| Early stopping       | patience=6, min=8 | patience=6, min=8 |
| Trainable layers     | full model       | full model         |
| Loss                 | InfoNCE (contrastive) | InfoNCE       |

### BLIP-2 Fine-tuning
| Parameter            | AI2D             | SciCap (45k)      |
|----------------------|-----------------|-------------------|
| Train samples        | 2,470            | 45,000             |
| Learning rate        | 2e-5             | 1e-5               |
| Micro batch size     | 8                | 8                  |
| Grad accum steps     | 1                | 8                  |
| Effective batch      | 8                | 64                 |
| In-batch negatives   | 7                | 7 (see note below) |
| Optimizer            | AdamW            | AdamW (β1=0.9, β2=0.999) |
| LR schedule          | none             | warmup(200 steps) + cosine decay |
| Grad clipping        | none             | max_norm=1.0       |
| Epochs completed     | {blip_ai2d_ep}   | {blip_sc_ep}       |
| Best epoch           | {blip_ai2d_best} | {blip_sc_best}     |
| Best val Mean R@1    | {blip_ai2d_bval} | {blip_sc_bval}     |
| Early stopping       | patience=5, min=8 | patience=5, min=8 |
| Trainable layers     | Q-Former, query_tokens, projections, ITM head |
| Frozen layers        | ViT-G/14 image encoder                       |
| Loss                 | ITC + 0.2 × ITM                              |

> **ITC negatives bottleneck:** Gradient accumulation accumulates gradients across
> 8 micro-batches but does NOT share negatives between them. Each micro-batch of 8
> samples has only 7 in-batch negatives for the ITC contrastive loss. The original
> BLIP-2 paper uses batch=256 (255 negatives). This is the primary reason BLIP-2's
> fine-tuning gain is smaller than CLIP's on SciCap: the Q-Former never sees hard
> enough negatives to learn a strongly discriminative embedding.

---

## 8. Results

### 8.1 Recall@K — All Runs

#### Flickr30K (zero-shot, 1000 images × 5000 captions)
| Model  | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 | Mean R@1 |
|--------|---------|---------|----------|---------|---------|----------|----------|
| CLIP   | {r("CLIP","Flickr30K","zero-shot","i2t_R1")} | {r("CLIP","Flickr30K","zero-shot","i2t_R5")} | {r("CLIP","Flickr30K","zero-shot","i2t_R10")} | {r("CLIP","Flickr30K","zero-shot","t2i_R1")} | {r("CLIP","Flickr30K","zero-shot","t2i_R5")} | {r("CLIP","Flickr30K","zero-shot","t2i_R10")} | {r("CLIP","Flickr30K","zero-shot","mean_R1")} |
| BLIP-2 | {r("BLIP-2","Flickr30K","zero-shot","i2t_R1")} | {r("BLIP-2","Flickr30K","zero-shot","i2t_R5")} | {r("BLIP-2","Flickr30K","zero-shot","i2t_R10")} | {r("BLIP-2","Flickr30K","zero-shot","t2i_R1")} | {r("BLIP-2","Flickr30K","zero-shot","t2i_R5")} | {r("BLIP-2","Flickr30K","zero-shot","t2i_R10")} | {r("BLIP-2","Flickr30K","zero-shot","mean_R1")} |

#### AI2D (310 images × 310 captions)
| Model  | Variant    | i2t R@1 | t2i R@1 | Mean R@1 | Mean R@5 | Mean R@10 |
|--------|-----------|---------|---------|----------|----------|-----------|
| CLIP   | zero-shot  | {r("CLIP","AI2D","zero-shot","i2t_R1")} | {r("CLIP","AI2D","zero-shot","t2i_R1")} | {r("CLIP","AI2D","zero-shot","mean_R1")} | {r("CLIP","AI2D","zero-shot","mean_R5")} | {r("CLIP","AI2D","zero-shot","mean_R10")} |
| BLIP-2 | zero-shot  | {r("BLIP-2","AI2D","zero-shot","i2t_R1")} | {r("BLIP-2","AI2D","zero-shot","t2i_R1")} | {r("BLIP-2","AI2D","zero-shot","mean_R1")} | {r("BLIP-2","AI2D","zero-shot","mean_R5")} | {r("BLIP-2","AI2D","zero-shot","mean_R10")} |
| CLIP   | fine-tuned | {r("CLIP","AI2D","fine-tuned","i2t_R1")} | {r("CLIP","AI2D","fine-tuned","t2i_R1")} | {r("CLIP","AI2D","fine-tuned","mean_R1")} | {r("CLIP","AI2D","fine-tuned","mean_R5")} | {r("CLIP","AI2D","fine-tuned","mean_R10")} |
| BLIP-2 | fine-tuned | {r("BLIP-2","AI2D","fine-tuned","i2t_R1")} | {r("BLIP-2","AI2D","fine-tuned","t2i_R1")} | {r("BLIP-2","AI2D","fine-tuned","mean_R1")} | {r("BLIP-2","AI2D","fine-tuned","mean_R5")} | {r("BLIP-2","AI2D","fine-tuned","mean_R10")} |

#### SciCap (1000 images × 1000 captions; both models trained on 45k pairs)
| Model  | Variant    | i2t R@1 | t2i R@1 | Mean R@1 | Mean R@5 | Mean R@10 |
|--------|-----------|---------|---------|----------|----------|-----------|
| CLIP   | zero-shot  | {r("CLIP","SciCap","zero-shot","i2t_R1")} | {r("CLIP","SciCap","zero-shot","t2i_R1")} | {r("CLIP","SciCap","zero-shot","mean_R1")} | {r("CLIP","SciCap","zero-shot","mean_R5")} | {r("CLIP","SciCap","zero-shot","mean_R10")} |
| BLIP-2 | zero-shot  | {r("BLIP-2","SciCap","zero-shot","i2t_R1")} | {r("BLIP-2","SciCap","zero-shot","t2i_R1")} | {r("BLIP-2","SciCap","zero-shot","mean_R1")} | {r("BLIP-2","SciCap","zero-shot","mean_R5")} | {r("BLIP-2","SciCap","zero-shot","mean_R10")} |
| CLIP   | fine-tuned | {r("CLIP","SciCap","fine-tuned","i2t_R1")} | {r("CLIP","SciCap","fine-tuned","t2i_R1")} | {r("CLIP","SciCap","fine-tuned","mean_R1")} | {r("CLIP","SciCap","fine-tuned","mean_R5")} | {r("CLIP","SciCap","fine-tuned","mean_R10")} |
| BLIP-2 | fine-tuned | {r("BLIP-2","SciCap","fine-tuned","i2t_R1")} | {r("BLIP-2","SciCap","fine-tuned","t2i_R1")} | {r("BLIP-2","SciCap","fine-tuned","mean_R1")} | {r("BLIP-2","SciCap","fine-tuned","mean_R5")} | {r("BLIP-2","SciCap","fine-tuned","mean_R10")} |

### 8.2 Fine-tuning Gain (Mean R@1, percentage points)

| Model  | Dataset | Zero-shot | Fine-tuned | Gain (pp) |
|--------|---------|-----------|-----------|-----------|
| CLIP   | AI2D    | {r("CLIP","AI2D","zero-shot","mean_R1")} | {r("CLIP","AI2D","fine-tuned","mean_R1")} | — |
| BLIP-2 | AI2D    | {r("BLIP-2","AI2D","zero-shot","mean_R1")} | {r("BLIP-2","AI2D","fine-tuned","mean_R1")} | — |
| CLIP   | SciCap  | {r("CLIP","SciCap","zero-shot","mean_R1")} | {r("CLIP","SciCap","fine-tuned","mean_R1")} | — |
| BLIP-2 | SciCap  | {r("BLIP-2","SciCap","zero-shot","mean_R1")} | {r("BLIP-2","SciCap","fine-tuned","mean_R1")} | — |

(Compute gain = fine-tuned − zero-shot from the values above.)

---

## 9. Latency & Throughput

Measurements on NVIDIA H200, test-set evaluation.
Zero-shot runs use fp16; fine-tuned runs use fp32.

| Model  | Dataset   | Variant    | N img | ImgEnc (s) | img/sec | ITM rerank (s) | Total (s) |
|--------|-----------|-----------|-------|-----------|---------|---------------|----------|
| CLIP   | Flickr30K | zero-shot  | 1000  | {r("CLIP","Flickr30K","zero-shot","img_enc_s")} | {r("CLIP","Flickr30K","zero-shot","img_per_sec")} | — | {r("CLIP","Flickr30K","zero-shot","total_s")} |
| BLIP-2 | Flickr30K | zero-shot  | 1000  | {r("BLIP-2","Flickr30K","zero-shot","img_enc_s")} | {r("BLIP-2","Flickr30K","zero-shot","img_per_sec")} | {r("BLIP-2","Flickr30K","zero-shot","itm_rerank_s")} | {r("BLIP-2","Flickr30K","zero-shot","total_s")} |
| CLIP   | AI2D      | zero-shot  | 310   | {r("CLIP","AI2D","zero-shot","img_enc_s")} | {r("CLIP","AI2D","zero-shot","img_per_sec")} | — | {r("CLIP","AI2D","zero-shot","total_s")} |
| BLIP-2 | AI2D      | zero-shot  | 310   | {r("BLIP-2","AI2D","zero-shot","img_enc_s")} | {r("BLIP-2","AI2D","zero-shot","img_per_sec")} | {r("BLIP-2","AI2D","zero-shot","itm_rerank_s")} | {r("BLIP-2","AI2D","zero-shot","total_s")} |
| CLIP   | AI2D      | fine-tuned | 310   | {r("CLIP","AI2D","fine-tuned","img_enc_s")} | {r("CLIP","AI2D","fine-tuned","img_per_sec")} | — | {r("CLIP","AI2D","fine-tuned","total_s")} |
| BLIP-2 | AI2D      | fine-tuned | 310   | {r("BLIP-2","AI2D","fine-tuned","img_enc_s")} | {r("BLIP-2","AI2D","fine-tuned","img_per_sec")} | {r("BLIP-2","AI2D","fine-tuned","itm_rerank_s")} | {r("BLIP-2","AI2D","fine-tuned","total_s")} |
| CLIP   | SciCap    | zero-shot  | 1000  | {r("CLIP","SciCap","zero-shot","img_enc_s")} | {r("CLIP","SciCap","zero-shot","img_per_sec")} | — | {r("CLIP","SciCap","zero-shot","total_s")} |
| BLIP-2 | SciCap    | zero-shot  | 1000  | {r("BLIP-2","SciCap","zero-shot","img_enc_s")} | {r("BLIP-2","SciCap","zero-shot","img_per_sec")} | {r("BLIP-2","SciCap","zero-shot","itm_rerank_s")} | {r("BLIP-2","SciCap","zero-shot","total_s")} |
| CLIP   | SciCap    | fine-tuned | 1000  | {r("CLIP","SciCap","fine-tuned","img_enc_s")} | {r("CLIP","SciCap","fine-tuned","img_per_sec")} | — | {r("CLIP","SciCap","fine-tuned","total_s")} |
| BLIP-2 | SciCap    | fine-tuned | 1000  | {r("BLIP-2","SciCap","fine-tuned","img_enc_s")} | {r("BLIP-2","SciCap","fine-tuned","img_per_sec")} | {r("BLIP-2","SciCap","fine-tuned","itm_rerank_s")} | {r("BLIP-2","SciCap","fine-tuned","total_s")} |

**ITM reranking explanation:** BLIP-2 reranks the top-128 candidates returned by
ITC using a cross-attention model. For 1,000 images × 1,000 texts: 128,000 ITM
pairs for i2t + 128,000 for t2i = 256,000 total pairs per evaluation run. For
Flickr30K (1,000 images × 5,000 texts): up to 640,000 pairs. This is why
BLIP-2 total inference time is 100–500× slower than CLIP.

---

## 10. Collage Descriptions

Each collage PNG contains 3 side-by-side panels, 150 DPI, ~1,300 × 600 px.

| File                         | Dataset   | Model  | Variant    | Hit panels | Miss panels |
|------------------------------|-----------|--------|-----------|-----------|------------|
| flickr30k_clip_examples.png  | Flickr30K | CLIP   | zero-shot  | 2          | 1          |
| flickr30k_blip2_examples.png | Flickr30K | BLIP-2 | zero-shot  | 3          | 0          |
| scicap_clip_examples.png     | SciCap    | CLIP   | fine-tuned | 1          | 2          |
| scicap_blip2_examples.png    | SciCap    | BLIP-2 | fine-tuned | 1          | 2          |

Panel layout:
- **Green border + ✓:** model retrieved the correct caption (R@1 hit)
- **Red border + ✗:** model retrieved a wrong caption (R@1 miss)
- **Green text:** ground truth caption from the dataset
- **Blue text:** caption retrieved by the model (rank 1 prediction)

Images are selected to be visually diverse (spread evenly across hit/miss pools).
The hit/miss ratio per collage mirrors the actual model R@1 performance on each dataset.

---

## 11. Hardware & Environment

- **Cluster:** PERUN HPC (Supercomputing Center TUKE, Slovakia)
- **GPU:** NVIDIA H200 (80 GB VRAM)
- **CPU:** 4–8 cores per SLURM job
- **RAM:** 64–160 GB per SLURM job
- **Framework:** PyTorch 2.7.1+cu118
- **Transformers:** 4.45.2
- **CLIP:** Official OpenAI CLIP (ViT-L/14) via `openai/CLIP`
- **BLIP-2:** HuggingFace `Salesforce/blip2-itm-vit-g`
- **Precision:** fp16 for zero-shot inference; fp32 for fine-tuning and fine-tuned evaluation

### Approximate SLURM job durations
| Job                                | Approx. time |
|------------------------------------|-------------|
| CLIP zero-shot eval (any dataset)  | 5–15 min    |
| BLIP-2 zero-shot eval (Flickr30K)  | 90 min      |
| BLIP-2 zero-shot eval (SciCap)     | 45 min      |
| CLIP fine-tune AI2D                | 20 min      |
| BLIP-2 fine-tune AI2D              | 3–4 h       |
| CLIP fine-tune SciCap 45k          | 1.5 h       |
| BLIP-2 fine-tune SciCap 45k        | 38 h        |
| Generate collages (all 4)          | 30–90 min   |

---

## 12. Public Source Code Location

Active code lives in the public repository under `src/`, with SLURM launchers in `scripts/slurm/`.

Key source files:
```
clip-vs-blip2-image-text-retrieval/
├── src/
│   ├── models/
│   │   ├── openai_clip_wrapper.py          ← CLIP wrapper (encode_images, encode_texts)
│   │   └── hf_blip2_wrapper.py             ← BLIP-2 wrapper (ITC + ITM rerank)
│   ├── data/
│   │   ├── flickr30k.py                    ← Flickr30K Karpathy split loader
│   │   ├── ai2d.py                         ← AI2D dataset loader
│   │   └── scicap.py                       ← SciCap JSONL dataset loader
│   ├── train/
│   │   ├── finetune_clip_ai2d.py           ← CLIP fine-tuning (AI2D)
│   │   ├── finetune_clip_scicap.py         ← CLIP fine-tuning (SciCap)
│   │   ├── finetune_blip2_ai2d.py          ← BLIP-2 fine-tuning (AI2D)
│   │   └── finetune_blip2_scicap.py        ← BLIP-2 fine-tuning (SciCap, improved HP)
│   ├── eval/
│   │   ├── evaluate_clip.py                ← CLIP eval (outputs raw JSON)
│   │   └── evaluate_blip2.py               ← BLIP-2 eval (ITC + ITM, outputs raw JSON)
│   ├── analysis/
│   │   ├── build_bc_results.py             ← this script; assembles ZIP
│   │   ├── compare_all_results.py          ← terminal table comparison
│   │   └── make_collages.py                ← generates collage PNGs
│   └── prepare_scicap_split.py             ← creates 45k/1k/1k JSONL splits
├── configs/
│   └── experiment_config.yaml              ← all hyperparameters
├── scripts/slurm/
│   ├── run_finetune_clip_scicap.sbatch
│   ├── run_finetune_blip2_scicap.sbatch
│   ├── run_make_collages.sbatch
│   └── ...
└── results/
    ├── raw/                                ← AI2D + Flickr30K raw JSONs
    ├── scicap/raw_results/                 ← SciCap raw JSONs
    ├── scicap/checkpoints/                 ← saved model weights (.pt files)
    ├── scicap/logs/                        ← SciCap training history JSONs
    ├── logs/                               ← AI2D training history JSONs
    ├── figures/collages/                   ← generated PNG collages
    └── tables/                             ← CSV output from compare_all_results.py
```

To regenerate this ZIP from scratch:
```bash
python -m src.analysis.build_bc_results        # with collages
python -m src.analysis.build_bc_results --no-collages  # without collages
```
"""


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def assemble(include_collages: bool = True) -> Path:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()

    # 1. Copy raw JSON files
    raw_dir = OUT_DIR / "raw_json"
    raw_dir.mkdir()
    for dest_name, src_path in RAW_FILES.items():
        if src_path.exists():
            shutil.copy(src_path, raw_dir / dest_name)
            print(f"  copied {dest_name}")
        else:
            print(f"  [MISSING] {dest_name}")

    # 2. Build & write CSVs
    results_dir = OUT_DIR / "results"
    results_dir.mkdir()
    csvs = build_csvs(raw_dir)
    for fname, content in csvs.items():
        if content:
            (results_dir / fname).write_text(content)
            print(f"  wrote {fname}")

    # 3. Training history
    hist_dir = results_dir / "training_history"
    hist_dir.mkdir()
    for dest_name, src_path in HISTORY_FILES.items():
        if src_path.exists():
            shutil.copy(src_path, hist_dir / dest_name)
            print(f"  copied {dest_name}")
        else:
            print(f"  [MISSING] {dest_name}")

    # 4. Collages
    collage_dir = OUT_DIR / "collages"
    collage_dir.mkdir()
    if include_collages:
        for dest_name, src_path in COLLAGE_FILES.items():
            if src_path.exists():
                shutil.copy(src_path, collage_dir / dest_name)
                print(f"  copied {dest_name}")
            else:
                print(f"  [MISSING collage — run make_collages first] {dest_name}")
    else:
        print("  Collages skipped (run make_collages.py first)")

    # 4b. Training curve plots
    plots_dir = OUT_DIR / "training_plots"
    plots_dir.mkdir()
    for dest_name, src_path in TRAINING_PLOT_FILES.items():
        if src_path.exists():
            shutil.copy(src_path, plots_dir / dest_name)
            print(f"  copied {dest_name}")
        else:
            print(f"  [MISSING plot — run plot_training_curves first] {dest_name}")

    # 5. Build README
    all_rows = []
    for label, dataset, variant, fname in ENTRIES:
        path = raw_dir / fname
        if path.exists():
            all_rows.append(_row_from_json(label, dataset, variant, path))
    (OUT_DIR / "README.md").write_text(build_readme(all_rows))
    print("  wrote README.md")

    # 6. ZIP
    zip_path = BASE / "bc_results_export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(OUT_DIR.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(BASE))
    print(f"\n  ZIP created: {zip_path}")
    print(f"  Size: {zip_path.stat().st_size / 1024 / 1024:.1f} MB")
    return zip_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-collages", action="store_true",
                        help="Skip collages (use if make_collages has not run yet)")
    args = parser.parse_args()
    print("Assembling bc_results_export/ ...")
    assemble(include_collages=not args.no_collages)


if __name__ == "__main__":
    main()
