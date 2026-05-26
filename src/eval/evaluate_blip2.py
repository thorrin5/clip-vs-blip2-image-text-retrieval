"""
Zero-shot evaluácia Hugging Face BLIP-2 pre image-text retrieval.

Skript vyhodnocuje model Salesforce/blip2-itm-vit-g na Flickr30K a SciCap.
Najprv počíta ITC podobnosť pre všetky kandidátske dvojice. Voliteľne potom
pre najlepších top-k kandidátov spustí ITM preusporiadanie, ktoré je presnejšie,
ale výrazne náročnejšie na čas. Práve tento rozdiel tvorí jadro porovnania
presnosti a výpočtových nákladov oproti CLIP.
"""

from __future__ import annotations

import argparse
import sys
import time
from statistics import median

from src.eval.common import append_metrics_csv, append_metrics_csv_to, load_dataset_from_config, write_json_result, write_json_result_to
from src.eval.retrieval_metrics import compact_metrics, compute_retrieval_metrics
from src.models.hf_blip2_wrapper import HuggingFaceBlip2RetrievalWrapper
from src.utils.config import get_recall_k, load_config, resolve_path
from src.utils.paths import ensure_output_dirs


def build_parser() -> argparse.ArgumentParser:
    """Definuje CLI argumenty pre BLIP-2 ITC/ITM evaluáciu."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--dataset", choices=["flickr30k", "scicap"], default="flickr30k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--batch-size-images", type=int, default=None)
    parser.add_argument("--batch-size-texts", type=int, default=None)
    parser.add_argument("--pair-batch-size", type=int, default=None)
    parser.add_argument("--allow-large-exact", action="store_true")
    return parser


def _combine_directional_metrics(image_to_text: dict, text_to_image: dict, k_values: list[int], timings: dict) -> dict:
    """Spojí smerové metriky do rovnakého formátu, aký používa hlavný výpočet."""
    mean_recall = {
        f"R@{k}": (float(image_to_text[f"R@{k}"]) + float(text_to_image[f"R@{k}"])) / 2.0
        for k in k_values
    }
    return {
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
        "mean": mean_recall,
        "timing": timings,
        "num_images": len(image_to_text["ranks"]),
        "num_texts": len(text_to_image["ranks"]),
    }


def _rerank_matrix(model, base_similarity, dataset, top_k: int, direction: str, batch_size: int):
    """
    Vytvorí riedku maticu skóre po ITM preusporiadaní pre jeden smer retrievalu.

    Do ITM hlavy posiela iba top-k kandidátov podľa rýchlej ITC podobnosti.
    Ostatné pozície zostávajú na hodnote `-inf`, aby sa nemohli dostať na
    popredné miesta v rebríčku po preusporiadaní.
    """
    torch = model.torch
    num_images = dataset.num_images
    num_captions = dataset.num_captions
    reranked = torch.full_like(base_similarity, float("-inf"))
    image_pairs: list[str] = []
    text_pairs: list[str] = []
    positions: list[tuple[int, int]] = []

    # Len top-k ITC kandidáti sa posielajú cez drahú ITM hlavu. Ostatné skóre
    # ostanú -inf, takže nemôžu vyhrať v preusporiadanom rebríčku.
    if direction == "image_to_text":
        k = min(top_k, num_captions)
        candidates = torch.topk(base_similarity, k=k, dim=1).indices
        for image_idx in range(num_images):
            for caption_idx in candidates[image_idx].tolist():
                image_pairs.append(dataset.image_paths[image_idx])
                text_pairs.append(dataset.captions[caption_idx])
                positions.append((image_idx, caption_idx))
    elif direction == "text_to_image":
        k = min(top_k, num_images)
        candidates = torch.topk(base_similarity, k=k, dim=0).indices
        for caption_idx in range(num_captions):
            for image_idx in candidates[:, caption_idx].tolist():
                image_pairs.append(dataset.image_paths[image_idx])
                text_pairs.append(dataset.captions[caption_idx])
                positions.append((image_idx, caption_idx))
    else:
        raise ValueError(f"Unknown rerank direction: {direction}")

    scores = model.score_pairs(image_pairs, text_pairs, batch_size=batch_size)
    for (image_idx, caption_idx), score in zip(positions, scores):
        reranked[image_idx, caption_idx] = score
    return reranked, len(positions)


def _compute_topk_itm_metrics(model, base_similarity, dataset, top_k: int, k_values: list[int], batch_size: int) -> tuple[dict, dict]:
    """
    Spustí ITM preusporiadanie v oboch smeroch a vráti metriky aj časovanie.

    Image-to-text a text-to-image smer sa rátajú oddelene, pretože počet
    kandidátov a výsledné poradia sa líšia. Nakoniec sa metriky spoja do
    rovnakého formátu, aký používa základná Recall@K evaluácia.
    """
    timings: dict[str, float] = {"rerank_top_k": float(top_k)}

    start = time.time()
    i2t_matrix, i2t_pairs = _rerank_matrix(model, base_similarity, dataset, top_k, "image_to_text", batch_size)
    i2t_time = time.time() - start
    i2t_metrics = compute_retrieval_metrics(i2t_matrix, dataset.image_to_captions, k_values)

    start = time.time()
    t2i_matrix, t2i_pairs = _rerank_matrix(model, base_similarity, dataset, top_k, "text_to_image", batch_size)
    t2i_time = time.time() - start
    t2i_metrics = compute_retrieval_metrics(t2i_matrix, dataset.image_to_captions, k_values)

    timings.update(
        {
            "i2t_itm_rerank_time": i2t_time,
            "t2i_itm_rerank_time": t2i_time,
            "itm_rerank_time": i2t_time + t2i_time,
            "i2t_itm_pairs": i2t_pairs,
            "t2i_itm_pairs": t2i_pairs,
            "itm_pairs": i2t_pairs + t2i_pairs,
            "itm_pairs_per_second": (i2t_pairs + t2i_pairs) / (i2t_time + t2i_time) if (i2t_time + t2i_time) else 0.0,
        }
    )

    metrics = _combine_directional_metrics(
        {
            **{f"R@{k}": i2t_metrics["image_to_text"][f"R@{k}"] for k in k_values},
            "ranks": i2t_metrics["image_to_text"]["ranks"],
            "median_rank": float(median(i2t_metrics["image_to_text"]["ranks"])),
        },
        {
            **{f"R@{k}": t2i_metrics["text_to_image"][f"R@{k}"] for k in k_values},
            "ranks": t2i_metrics["text_to_image"]["ranks"],
            "median_rank": float(median(t2i_metrics["text_to_image"]["ranks"])),
        },
        k_values,
        timings,
    )
    return metrics, timings


def main(argv: list[str] | None = None) -> int:
    """
    Spustí BLIP-2 evaluáciu s voliteľným ITM re-rankingom.

    Najprv sa vypočíta ITC podobnosť pre všetky páry. Ak je zadané
    `--rerank-top-k`, najlepšie kandidáty sa následne prehodnotia drahšou ITM
    hlavou, čo je hlavný zdroj vyššej latencie BLIP-2 v práci.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    ensure_output_dirs(config)
    dataset = load_dataset_from_config(config, args.dataset, args.split, args.max_images)
    model_config = config["models"]["blip2"]
    eval_config = config["evaluation"]
    k_values = get_recall_k(config)
    batch_images = args.batch_size_images or int(eval_config["batch_size_images"])
    batch_texts = args.batch_size_texts or int(eval_config["batch_size_texts"])
    pair_batch_size = args.pair_batch_size or int(eval_config.get("blip2_pair_batch_size", 8))
    rerank_pair_limit = int(eval_config.get("blip2_itm_pair_limit", 800000))
    implementation = "huggingface_transformers"

    if args.rerank_top_k is not None and args.rerank_top_k <= 0:
        raise ValueError("--rerank-top-k must be positive")

    if args.rerank_top_k is not None:
        rerank_pairs = dataset.num_images * min(args.rerank_top_k, dataset.num_captions)
        rerank_pairs += dataset.num_captions * min(args.rerank_top_k, dataset.num_images)
        if rerank_pairs > rerank_pair_limit and not args.allow_large_exact:
            payload = {
                "status": "blocked",
                "reason": (
                    f"BLIP-2 ITM reranking would require {rerank_pairs} image-text pairs, above "
                    f"configured limit {rerank_pair_limit}. Use a smaller --rerank-top-k, a subset, "
                    "or --allow-large-exact after accepting the runtime."
                ),
                "implementation": implementation,
                "model": model_config,
                "dataset": dataset.to_metadata(),
                "protocol": "topk_itm_rerank_blocked",
                "requested_rerank_top_k": args.rerank_top_k,
                "rerank_pairs": rerank_pairs,
                "rerank_pair_limit": rerank_pair_limit,
            }
            if args.dataset == "scicap":
                output_path = write_json_result_to(resolve_path(config, "scicap_raw_root"), f"blip2_{args.dataset}_{args.split}_blocked", payload)
            else:
                output_path = write_json_result(config, f"blip2_{args.dataset}_{args.split}_blocked", payload)
            print(payload["reason"])
            print(f"Blocker written to {output_path}")
            return 2

    # Rovnako ako pri CLIP sa pri chýbajúcich závislostiach alebo váhach uloží
    # blokovaný výsledok, aby bol neúspešný beh spätne vysvetliteľný.
    try:
        model = HuggingFaceBlip2RetrievalWrapper(
            model_config["model_name"],
            device="cuda",
            precision=model_config.get("precision", "fp16"),
        )
    except Exception as exc:
        payload = {
            "status": "blocked",
            "reason": str(exc),
            "implementation": implementation,
            "model": model_config,
            "dataset": dataset.to_metadata(),
            "protocol": "model_load_blocked",
        }
        if args.dataset == "scicap":
            output_path = write_json_result_to(resolve_path(config, "scicap_raw_root"), f"blip2_{args.dataset}_{args.split}_blocked", payload)
        else:
            output_path = write_json_result(config, f"blip2_{args.dataset}_{args.split}_blocked", payload)
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
    similarity = model.compute_itc_similarity(image_features, text_features, image_batch_size=batch_images)
    timings["similarity_time"] = time.time() - start
    timings["total_time"] = sum(timings.values())
    timings["images_per_second"] = dataset.num_images / timings["image_encoding_time"] if timings["image_encoding_time"] else 0.0
    timings["texts_per_second"] = dataset.num_captions / timings["text_encoding_time"] if timings["text_encoding_time"] else 0.0

    itc_metrics = compute_retrieval_metrics(similarity, dataset.image_to_captions, k_values, timings=dict(timings))
    metrics = itc_metrics
    protocol = "huggingface_blip2_itc_similarity_exact"
    rerank_details: dict = {}

    if args.rerank_top_k is not None:
        # Toto je jadro kompromisu presnosť/čas: ITC vyberie kandidátov a ITM
        # prepočíta iba obmedzenú top-k množinu párov.
        rerank_metrics, rerank_timings = _compute_topk_itm_metrics(
            model,
            similarity,
            dataset,
            args.rerank_top_k,
            k_values,
            pair_batch_size,
        )
        timings.update(rerank_timings)
        timings["total_time"] = timings.get("total_time", 0.0) + rerank_timings.get("itm_rerank_time", 0.0)
        rerank_metrics["timing"] = timings
        metrics = rerank_metrics
        protocol = "huggingface_blip2_itc_topk_itm_rerank"
        rerank_details = {
            "rerank_top_k": args.rerank_top_k,
            "pair_batch_size": pair_batch_size,
            "itc_metrics": itc_metrics,
        }

    payload = {
        "status": "ok",
        "implementation": implementation,
        "model": model.info.to_dict(),
        "dataset": dataset.to_metadata(),
        "protocol": protocol,
        "itc_similarity_pairs": dataset.num_images * dataset.num_captions,
        "metrics": metrics,
        **rerank_details,
    }
    if args.dataset == "scicap":
        raw_root = resolve_path(config, "scicap_raw_root")
        tables_root = resolve_path(config, "scicap_tables_root")
        output_path = write_json_result_to(raw_root, f"blip2_{args.dataset}_{args.split}", payload)
        row = {
            "model": "BLIP-2",
            "implementation": "huggingface_transformers",
            "dataset": dataset.name,
            "split": dataset.split,
            **compact_metrics(metrics),
            "raw_json": str(output_path),
        }
        append_metrics_csv_to(tables_root, "scicap_zero_shot_results.csv", row)
    else:
        output_path = write_json_result(config, f"blip2_{args.dataset}_{args.split}", payload)
        row = {
            "model": "BLIP-2",
            "implementation": "huggingface_transformers",
            "dataset": dataset.name,
            "split": dataset.split,
            **compact_metrics(metrics),
            "raw_json": str(output_path),
        }
        append_metrics_csv(config, "blip2_metrics.csv", row)
    print(f"Results written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
