"""Dispatch canonical evaluations."""

from __future__ import annotations

import argparse
import sys

from src.eval import evaluate_blip2, evaluate_clip


MODEL_ORDER = ("clip", "blip2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--model", choices=["clip", "blip2", "all"], default="all")
    parser.add_argument("--dataset", choices=["flickr30k", "ai2d", "scicap"], default="flickr30k")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--pair-batch-size", type=int, default=None)
    parser.add_argument("--allow-large-exact", action="store_true")
    return parser


def selected_models(model: str) -> list[str]:
    if model == "all":
        return list(MODEL_ORDER)
    return [model]


def build_model_argvs(args: argparse.Namespace) -> dict[str, list[str]]:
    common_args = [
        "--config",
        args.config,
        "--dataset",
        args.dataset,
        "--split",
        args.split,
    ]
    if args.max_images is not None:
        common_args += ["--max-images", str(args.max_images)]

    model_argvs: dict[str, list[str]] = {}
    for model in selected_models(args.model):
        model_args = list(common_args)
        if model == "blip2":
            if args.rerank_top_k is not None:
                model_args += ["--rerank-top-k", str(args.rerank_top_k)]
            if args.pair_batch_size is not None:
                model_args += ["--pair-batch-size", str(args.pair_batch_size)]
            if args.allow_large_exact:
                model_args += ["--allow-large-exact"]
        model_argvs[model] = model_args
    return model_argvs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    model_argvs = build_model_argvs(args)
    codes: list[int] = []
    for model in selected_models(args.model):
        if model == "clip":
            codes.append(evaluate_clip.main(model_argvs[model]))
        elif model == "blip2":
            codes.append(evaluate_blip2.main(model_argvs[model]))
    return max(codes) if codes else 0


if __name__ == "__main__":
    sys.exit(main())
