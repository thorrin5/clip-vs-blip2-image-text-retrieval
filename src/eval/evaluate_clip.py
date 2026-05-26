"""
Zero-shot evaluácia oficiálneho OpenAI CLIP ViT-L/14 pre image-text retrieval.

Skript je vstupným bodom pre CLIP experimenty na Flickr30K a SciCap. Načíta
zvolený dataset, zakóduje všetky obrázky aj texty, vypočíta maticu podobností
a z nej odvodí Recall@K metriky. Tento postup predstavuje škálovateľný
dual-encoder baseline, voči ktorému sa porovnáva presnosť a časová náročnosť
modelu BLIP-2.
"""

from __future__ import annotations

import argparse
import sys
import time

from src.eval.common import append_metrics_csv, append_metrics_csv_to, load_dataset_from_config, write_json_result, write_json_result_to
from src.eval.retrieval_metrics import compact_metrics, compute_retrieval_metrics
from src.models.openai_clip_wrapper import OfficialOpenAIClipWrapper
from src.utils.config import get_recall_k, load_config, resolve_path
from src.utils.paths import ensure_output_dirs


def build_parser() -> argparse.ArgumentParser:
    """Definuje CLI argumenty pre evaluáciu modelu CLIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--dataset", choices=["flickr30k", "scicap"], default="flickr30k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size-images", type=int, default=None)
    parser.add_argument("--batch-size-texts", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """
    Spustí CLIP evaluáciu a uloží raw JSON aj CSV zhrnutie.

    Postup zodpovedá dual-encoder retrieval pipeline: najprv sa zakódujú
    všetky obrázky, potom všetky texty a následne sa vypočíta matica podobnosti.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ensure_output_dirs(config)
    dataset = load_dataset_from_config(config, args.dataset, args.split, args.max_images)
    model_config = config["models"]["clip"]
    eval_config = config["evaluation"]
    batch_images = args.batch_size_images or int(eval_config["batch_size_images"])
    batch_texts = args.batch_size_texts or int(eval_config["batch_size_texts"])

    # Načítanie modelu je oddelené, aby chýbajúce závislosti alebo váhy
    # vytvorili auditovateľný blokovaný JSON namiesto nejasného zlyhania.
    try:
        model = OfficialOpenAIClipWrapper(
            model_config["model_name"],
            device="cuda",
            precision=model_config.get("precision", "fp16"),
        )
    except Exception as exc:
        payload = {
            "status": "blocked",
            "reason": str(exc),
            "model": model_config,
            "dataset": dataset.to_metadata(),
            "protocol": "embedding_similarity_exact",
            "implementation": "official_openai_clip",
        }
        if args.dataset == "scicap":
            output_path = write_json_result_to(resolve_path(config, "scicap_raw_root"), f"clip_{args.dataset}_{args.split}_blocked", payload)
        else:
            output_path = write_json_result(config, f"clip_{args.dataset}_{args.split}_blocked", payload)
        print(f"Blocked: {exc}")
        print(f"Blocker written to {output_path}")
        return 2

    timings: dict[str, float] = {}
    start = time.time()
    image_features = model.encode_images(dataset.image_paths, batch_size=batch_images)
    timings["image_encoding_time"] = time.time() - start

    start = time.time()
    text_features = model.encode_texts(dataset.captions, batch_size=batch_texts)
    timings["text_encoding_time"] = time.time() - start

    start = time.time()
    similarity = image_features @ text_features.t()
    timings["similarity_time"] = time.time() - start
    timings["total_time"] = sum(timings.values())
    timings["images_per_second"] = dataset.num_images / timings["image_encoding_time"] if timings["image_encoding_time"] else 0.0
    timings["texts_per_second"] = dataset.num_captions / timings["text_encoding_time"] if timings["text_encoding_time"] else 0.0

    # Všetky modely používajú rovnaký výpočet Recall@K, takže porovnanie je
    # ovplyvnené iba skóre modelu, nie odlišnou evaluačnou logikou.
    metrics = compute_retrieval_metrics(similarity, dataset.image_to_captions, get_recall_k(config), timings=timings)
    payload = {
        "status": "ok",
        "implementation": "official_openai_clip",
        "model": model.info.to_dict(),
        "dataset": dataset.to_metadata(),
        "protocol": "embedding_similarity_exact",
        "metrics": metrics,
    }
    if args.dataset == "scicap":
        raw_root = resolve_path(config, "scicap_raw_root")
        tables_root = resolve_path(config, "scicap_tables_root")
        output_path = write_json_result_to(raw_root, f"clip_{args.dataset}_{args.split}", payload)
        row = {
            "model": "CLIP",
            "implementation": "official_openai_clip",
            "dataset": dataset.name,
            "split": dataset.split,
            **compact_metrics(metrics),
            "raw_json": str(output_path),
        }
        append_metrics_csv_to(tables_root, "scicap_zero_shot_results.csv", row)
    else:
        output_path = write_json_result(config, f"clip_{args.dataset}_{args.split}", payload)
        row = {
            "model": "CLIP",
            "implementation": "official_openai_clip",
            "dataset": dataset.name,
            "split": dataset.split,
            **compact_metrics(metrics),
            "raw_json": str(output_path),
        }
        append_metrics_csv(config, "clip_metrics.csv", row)
    print(f"Results written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
