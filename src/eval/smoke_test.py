"""Structural smoke tests for the canonical scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from src.eval.common import load_dataset_from_config
from src.eval.evaluate_all import build_model_argvs, build_parser as build_evaluate_all_parser
from src.eval.retrieval_metrics import compute_retrieval_metrics
from src.utils.config import load_config, resolve_path
from src.utils.paths import ensure_output_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--check-model-imports", action="store_true")
    parser.add_argument("--full-model-smoke", action="store_true")
    return parser


def check_model_imports() -> None:
    import importlib

    for module_name in ["clip", "transformers"]:
        importlib.import_module(module_name)
        print(f"import {module_name}: OK")
    from transformers import AutoProcessor, Blip2ForImageTextRetrieval

    assert AutoProcessor is not None
    assert Blip2ForImageTextRetrieval is not None
    print("import transformers.AutoProcessor: OK")
    print("import transformers.Blip2ForImageTextRetrieval: OK")


def check_config_shape(config: dict) -> None:
    required_path_keys = [
        "project_root",
        "data_root",
        "flickr30k_images",
        "flickr30k_karpathy_json",
        "ai2d_images",
        "ai2d_dataset_json",
        "raw_root",
        "tables_root",
        "figures_root",
        "logs_root",
    ]
    for key in required_path_keys:
        assert key in config.get("paths", {}), f"Missing paths.{key}"
    for model in ["clip", "blip2"]:
        assert model in config.get("models", {}), f"Missing models.{model}"
        assert "model_name" in config["models"][model], f"Missing models.{model}.model_name"
    assert "recall_k" in config.get("evaluation", {}), "Missing evaluation.recall_k"
    print("config shape: OK")


def check_slurm_script(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    required_snippets = [
        "#SBATCH --output=",
        "#SBATCH --error=",
        "PROJECT_ROOT=",
        "CONDA_INIT=",
        "hostname",
        "date",
        "nvidia-smi",
    ]
    for snippet in required_snippets:
        assert snippet in text, f"Missing {snippet!r} in {path}"


def run_full_model_smoke(config_path: str) -> None:
    commands = [
        [
            sys.executable,
            "-m",
            "src.eval.evaluate_all",
            "--config",
            config_path,
            "--model",
            "clip",
            "--dataset",
            "flickr30k",
            "--split",
            "test",
            "--max-images",
            "1",
        ],
        [
            sys.executable,
            "-m",
            "src.eval.evaluate_all",
            "--config",
            config_path,
            "--model",
            "blip2",
            "--dataset",
            "ai2d",
            "--split",
            "test",
            "--max-images",
            "1",
            "--allow-large-exact",
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True)
    print("full model smoke: OK")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    check_config_shape(config)
    ensure_output_dirs(config)

    for key in ["raw_root", "tables_root", "figures_root", "logs_root"]:
        path = resolve_path(config, key)
        assert path.exists(), f"Missing output dir {path}"
        print(f"output dir {key}: OK")

    flickr_ready = resolve_path(config, "flickr30k_images").exists() and resolve_path(config, "flickr30k_karpathy_json").exists()
    if flickr_ready:
        flickr = load_dataset_from_config(config, "flickr30k", "test", max_images=5)
        assert flickr.num_images == 5, f"Expected 5 Flickr images, got {flickr.num_images}"
        assert flickr.num_captions >= 25, f"Expected at least 25 Flickr captions, got {flickr.num_captions}"
        print(f"flickr30k loader: OK ({flickr.num_images} images, {flickr.num_captions} captions)")
    else:
        print("flickr30k loader: skipped (dataset files are not included in this repository)")

    ai2d_ready = resolve_path(config, "ai2d_images").exists() and (
        resolve_path(config, "ai2d_split_json").exists() or resolve_path(config, "ai2d_dataset_json").exists()
    )
    if ai2d_ready:
        ai2d = load_dataset_from_config(config, "ai2d", "test", max_images=5)
        assert ai2d.num_images == 5, f"Expected 5 AI2D images, got {ai2d.num_images}"
        print(f"ai2d loader: OK ({ai2d.num_images} images, {ai2d.num_captions} texts)")
    else:
        print("ai2d loader: skipped (dataset files are not included in this repository)")

    similarity = [
        [0.6, 0.2, 0.9, 0.8],
        [0.7, 0.3, 0.5, 0.4],
    ]
    mapping = {0: [0, 1], 1: [2, 3]}
    metrics = compute_retrieval_metrics(similarity, mapping, [1, 2], timings={"total_time": 0.01})
    assert metrics["image_to_text"]["R@1"] == 0.0
    assert metrics["image_to_text"]["R@2"] == 50.0
    assert metrics["image_to_text"]["ranks"] == [3, 2]
    assert metrics["image_to_text"]["median_rank"] == 2.5
    assert metrics["text_to_image"]["R@1"] == 0.0
    assert metrics["text_to_image"]["R@2"] == 100.0
    assert metrics["text_to_image"]["ranks"] == [2, 2, 2, 2]
    assert metrics["text_to_image"]["median_rank"] == 2.0
    assert metrics["mean"]["R@1"] == 0.0
    assert metrics["mean"]["R@2"] == 75.0
    assert metrics["timing"]["total_time"] == 0.01
    print("retrieval metrics: OK")

    evaluate_all_args = build_evaluate_all_parser().parse_args(
        [
            "--config",
            args.config,
            "--model",
            "all",
            "--dataset",
            "ai2d",
            "--split",
            "test",
            "--max-images",
            "3",
            "--rerank-top-k",
            "5",
            "--pair-batch-size",
            "2",
            "--allow-large-exact",
        ]
    )
    model_argvs = build_model_argvs(evaluate_all_args)
    assert list(model_argvs) == ["clip", "blip2"]
    assert model_argvs["clip"] == [
        "--config",
        args.config,
        "--dataset",
        "ai2d",
        "--split",
        "test",
        "--max-images",
        "3",
    ]
    assert model_argvs["blip2"] == [
        "--config",
        args.config,
        "--dataset",
        "ai2d",
        "--split",
        "test",
        "--max-images",
        "3",
        "--rerank-top-k",
        "5",
        "--pair-batch-size",
        "2",
        "--allow-large-exact",
    ]
    print("evaluate-all dispatch: OK")

    required_slurm = [
        "run_clip_flickr30k.sbatch",
        "run_blip2_flickr30k.sbatch",
        "run_clip_scicap.sbatch",
        "run_blip2_scicap.sbatch",
        "run_finetune_clip_scicap.sbatch",
        "run_finetune_blip2_scicap.sbatch",
        "run_make_collages.sbatch",
    ]
    slurm_dir = Path("scripts/slurm")
    for filename in required_slurm:
        path = slurm_dir / filename
        assert path.exists(), f"Missing SLURM script {path}"
        check_slurm_script(path)
    print("slurm scripts: OK")

    if args.check_model_imports:
        check_model_imports()
    else:
        print("model imports: skipped (use --check-model-imports inside the ML environment)")

    if args.full_model_smoke:
        run_full_model_smoke(args.config)
    else:
        print("full model smoke: skipped (use --full-model-smoke inside the ML environment)")

    print("smoke tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
