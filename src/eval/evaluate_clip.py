"""
Purpose:
    Evaluate official OpenAI CLIP ViT-L/14 for image-text retrieval.

Thesis context:
    This is the CLIP zero-shot evaluation entrypoint for Flickr30K and SciCap.
    It can also load AI2D because the shared loader keeps legacy compatibility,
    but the defense-focused path is Flickr30K plus SciCap.

Inputs:
    - Experiment YAML config.
    - Dataset name and split selected by --dataset and --split.
    - Optional max-images and batch-size arguments for small checks.

Outputs:
    - Raw JSON metrics with timing fields.
    - CSV summary row under the configured result tables directory.

Defense note:
    This script demonstrates the scalable CLIP retrieval workflow: encode all
    images, encode all captions, multiply normalized embeddings, and compute
    Recall@K. It is the baseline against which BLIP-2 accuracy and runtime are
    compared.
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--dataset", choices=["flickr30k", "ai2d", "scicap"], default="flickr30k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size-images", type=int, default=None)
    parser.add_argument("--batch-size-texts", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ensure_output_dirs(config)
    dataset = load_dataset_from_config(config, args.dataset, args.split, args.max_images)
    model_config = config["models"]["clip"]
    eval_config = config["evaluation"]
    batch_images = args.batch_size_images or int(eval_config["batch_size_images"])
    batch_texts = args.batch_size_texts or int(eval_config["batch_size_texts"])

    # Model loading is isolated so missing cluster dependencies produce a
    # documented "blocked" JSON instead of silently failing after a long job.
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

    # All models use the same metric implementation so the CLIP and BLIP-2
    # comparison differs only in scoring, not in evaluation logic.
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
