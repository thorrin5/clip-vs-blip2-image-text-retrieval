"""
Vytvorenie kompaktného exportu výsledkov pre verejný repozitár.

Skript nepočíta nové metriky. Pracuje iba s už existujúcimi raw JSON súbormi,
CSV tabuľkami, tréningovými históriami a obrázkami. Jeho úloha je zostaviť
samostatný adresár `bc_results_export/`, ktorý obsahuje finálne verejné
artefakty pre porovnanie modelov CLIP a BLIP-2 na Flickr30K a SciCap.
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
FIGURES = BASE / "figures"
OUT_DIR = BASE / "bc_results_export"

RAW_FILES = {
    "flickr30k_clip_zero_shot.json": RESULTS / "raw_json/flickr30k_clip_zero_shot.json",
    "flickr30k_blip2_zero_shot.json": RESULTS / "raw_json/flickr30k_blip2_zero_shot.json",
    "scicap_clip_zero_shot.json": RESULTS / "raw_json/scicap_clip_zero_shot.json",
    "scicap_blip2_zero_shot.json": RESULTS / "raw_json/scicap_blip2_zero_shot.json",
    "scicap_clip_fine_tuned_45k.json": RESULTS / "raw_json/scicap_clip_fine_tuned_45k.json",
    "scicap_blip2_fine_tuned_45k.json": RESULTS / "raw_json/scicap_blip2_fine_tuned_45k.json",
}

HISTORY_FILES = {
    "clip_scicap_45k_epochs.json": RESULTS / "training_history/clip_scicap_45k_epochs.json",
    "blip2_scicap_45k_epochs.json": RESULTS / "training_history/blip2_scicap_45k_epochs.json",
}

COLLAGE_FILES = {
    "flickr30k_clip_examples.png": FIGURES / "qualitative_examples/flickr30k_clip_examples.png",
    "flickr30k_blip2_examples.png": FIGURES / "qualitative_examples/flickr30k_blip2_examples.png",
    "scicap_clip_examples.png": FIGURES / "qualitative_examples/scicap_clip_examples.png",
    "scicap_blip2_examples.png": FIGURES / "qualitative_examples/scicap_blip2_examples.png",
}

TRAINING_PLOT_FILES = {
    "scicap_training_curves.png": FIGURES / "plots/scicap_training_curves.png",
}

RECALL_FIELDS = [
    "model",
    "dataset",
    "variant",
    "split",
    "i2t_R1",
    "i2t_R5",
    "i2t_R10",
    "t2i_R1",
    "t2i_R5",
    "t2i_R10",
    "mean_R1",
    "mean_R5",
    "mean_R10",
    "img_enc_s",
    "img_per_sec",
    "itm_rerank_s",
    "total_s",
    "train_samples",
    "lr",
    "effective_batch",
    "epochs_completed",
    "best_epoch",
    "best_val_mean_R1",
    "early_stopped",
]

ENTRIES = [
    ("CLIP", "Flickr30K", "zero-shot", "flickr30k_clip_zero_shot.json"),
    ("BLIP-2", "Flickr30K", "zero-shot", "flickr30k_blip2_zero_shot.json"),
    ("CLIP", "SciCap", "zero-shot", "scicap_clip_zero_shot.json"),
    ("BLIP-2", "SciCap", "zero-shot", "scicap_blip2_zero_shot.json"),
    ("CLIP", "SciCap", "fine-tuned", "scicap_clip_fine_tuned_45k.json"),
    ("BLIP-2", "SciCap", "fine-tuned", "scicap_blip2_fine_tuned_45k.json"),
]


def _get(mapping: dict, *keys: str, default: Any = None) -> Any:
    """Bezpečne získa vnorenú hodnotu zo slovníka výsledkov."""
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _format_value(value: Any, decimals: int = 2) -> str:
    """Prevedie číselnú alebo chýbajúcu hodnotu do formátu vhodného pre CSV."""
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def _load_json(path: Path) -> dict:
    """Načíta raw JSON výsledok jedného experimentálneho behu."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _row_from_json(label: str, dataset: str, variant: str, path: Path) -> dict[str, Any]:
    """
    Prevedie raw JSON metriky na jeden riadok verejnej výsledkovej tabuľky.

    Funkcia nemení hodnoty metrík. Iba ich normalizuje do rovnakých stĺpcov,
    aby bolo možné priamo porovnať CLIP a BLIP-2 v oboch retrieval smeroch.
    """
    payload = _load_json(path)
    metrics = payload.get("metrics", {})
    timing = metrics.get("timing", {})
    training = payload.get("training", {})
    dataset_meta = payload.get("dataset", {})

    image_time = timing.get("image_encoding_time", 0)
    text_time = timing.get("text_encoding_time", 0)
    similarity_time = timing.get("similarity_time", 0)
    itm_time = timing.get("itm_rerank_time") or (
        timing.get("i2t_itm_rerank_time", 0) + timing.get("t2i_itm_rerank_time", 0)
    )
    total_time = max(timing.get("total_time", 0), image_time + text_time + similarity_time + itm_time)
    num_images = metrics.get("num_images") or dataset_meta.get("num_images") or 0

    return {
        "model": label,
        "dataset": dataset,
        "variant": variant,
        "split": dataset_meta.get("split", "test"),
        "i2t_R1": _format_value(_get(metrics, "image_to_text", "R@1")),
        "i2t_R5": _format_value(_get(metrics, "image_to_text", "R@5")),
        "i2t_R10": _format_value(_get(metrics, "image_to_text", "R@10")),
        "t2i_R1": _format_value(_get(metrics, "text_to_image", "R@1")),
        "t2i_R5": _format_value(_get(metrics, "text_to_image", "R@5")),
        "t2i_R10": _format_value(_get(metrics, "text_to_image", "R@10")),
        "mean_R1": _format_value(_get(metrics, "mean", "R@1")),
        "mean_R5": _format_value(_get(metrics, "mean", "R@5")),
        "mean_R10": _format_value(_get(metrics, "mean", "R@10")),
        "img_enc_s": _format_value(image_time, 1),
        "img_per_sec": _format_value(num_images / image_time if image_time else None, 1),
        "itm_rerank_s": _format_value(itm_time, 0) if itm_time else "-",
        "total_s": _format_value(total_time, 0),
        "train_samples": training.get("train_samples", "-"),
        "lr": training.get("lr", "-"),
        "effective_batch": training.get("effective_batch_size") or training.get("batch_size") or "-",
        "epochs_completed": training.get("epochs_completed", "-"),
        "best_epoch": training.get("best_epoch", "-"),
        "best_val_mean_R1": _format_value(training.get("best_val_mean_R1")),
        "early_stopped": _get(training, "early_stopping", "stopped_early", default="-"),
    }


def _to_csv(rows: list[dict[str, Any]]) -> str:
    """Zapíše zoznam riadkov do CSV textu s jednotným poradím stĺpcov."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=RECALL_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def build_csvs(raw_dir: Path) -> dict[str, str]:
    """
    Vytvorí verejné CSV tabuľky z raw JSON výsledkov.

    Export obsahuje súhrnnú tabuľku a samostatné tabuľky pre Flickr30K a SciCap.
    Hodnoty sú preberané z raw JSON súborov bez ručných úprav.
    """
    all_rows = []
    for label, dataset, variant, filename in ENTRIES:
        path = raw_dir / filename
        if path.exists():
            all_rows.append(_row_from_json(label, dataset, variant, path))
        else:
            print(f"  [SKIP] {filename} not found")

    csvs = {"full_comparison.csv": _to_csv(all_rows)}
    for dataset in ["Flickr30K", "SciCap"]:
        rows = [row for row in all_rows if row["dataset"] == dataset]
        csvs[f"{dataset.lower()}_results.csv"] = _to_csv(rows)
    return csvs


def assemble(include_collages: bool = True) -> Path:
    """
    Zostaví adresár `bc_results_export/` a ZIP archív.

    Funkcia kopíruje iba vybrané verejné artefakty. Nevkladá datasety,
    checkpointy ani cache adresáre.
    """
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir()

    raw_dir = OUT_DIR / "raw_json"
    raw_dir.mkdir()
    for destination, source in RAW_FILES.items():
        if source.exists():
            shutil.copy(source, raw_dir / destination)
            print(f"  copied {destination}")
        else:
            print(f"  [MISSING] {destination}")

    results_dir = OUT_DIR / "results"
    results_dir.mkdir()
    for filename, content in build_csvs(raw_dir).items():
        (results_dir / filename).write_text(content, encoding="utf-8")
        print(f"  wrote {filename}")

    history_dir = results_dir / "training_history"
    history_dir.mkdir()
    for destination, source in HISTORY_FILES.items():
        if source.exists():
            shutil.copy(source, history_dir / destination)
            print(f"  copied {destination}")

    plot_dir = OUT_DIR / "training_plots"
    plot_dir.mkdir()
    for destination, source in TRAINING_PLOT_FILES.items():
        if source.exists():
            shutil.copy(source, plot_dir / destination)
            print(f"  copied {destination}")

    collage_dir = OUT_DIR / "collages"
    collage_dir.mkdir()
    if include_collages:
        for destination, source in COLLAGE_FILES.items():
            if source.exists():
                shutil.copy(source, collage_dir / destination)
                print(f"  copied {destination}")
            else:
                print(f"  [MISSING collage] {destination}")

    zip_path = BASE / "bc_results_export.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(OUT_DIR.rglob("*")):
            if file.is_file():
                archive.write(file, file.relative_to(BASE))
    print(f"\n  ZIP created: {zip_path}")
    return zip_path


def main() -> None:
    """CLI vstupný bod pre zostavenie exportu výsledkov."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-collages", action="store_true", help="Skip qualitative example collages")
    args = parser.parse_args()

    print("Assembling bc_results_export/ ...")
    assemble(include_collages=not args.no_collages)


if __name__ == "__main__":
    main()
