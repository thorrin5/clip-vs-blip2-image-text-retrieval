"""
Purpose:
    Fine-tune official OpenAI CLIP on SciCap retrieval pairs.

Thesis context:
    This script produces the domain-adapted CLIP SciCap result used in the
    thesis comparison. It trains on SciCap train, selects by validation Mean
    R@1, and evaluates the best checkpoint on SciCap test.

Inputs:
    - Experiment YAML config.
    - data/scicap_processed/{train,val,test}.jsonl.
    - SciCap images referenced by the JSONL rows.

Outputs:
    - Best CLIP checkpoint under results/scicap/checkpoints.
    - Per-epoch training history JSON.
    - Raw fine-tuned SciCap test metrics JSON.
    - CSV row in results/scicap/tables/scicap_finetuned_results.csv.

Defense note:
    This script shows how domain adaptation is tested for CLIP. The validation
    split prevents selecting on the test set, and the final metrics use the same
    Recall@K implementation as the zero-shot experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.openai_clip_wrapper import OfficialOpenAIClipWrapper
from src.train.common import (
    PairDataset,
    append_training_csv,
    build_finetune_payload,
    dataset_to_samples,
    load_scicap_splits,
    metric_row,
    scicap_checkpoint_name,
    set_seed,
    write_checkpoint,
    write_scicap_finetune_result,
    write_training_history,
)
from src.eval.retrieval_metrics import compact_metrics, compute_retrieval_metrics
from src.utils.config import get_recall_k, resolve_path
from src.utils.paths import ensure_output_dirs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment_config.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--skip-zero-shot", action="store_true")
    return parser


def contrastive_loss(image_features, text_features, logit_scale):
    """Symmetric CLIP InfoNCE loss with in-batch negatives."""
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logits = logit_scale.exp() * image_features @ text_features.t()
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def make_collate(clip_module):
    def collate(batch):
        images, texts = zip(*batch)
        image_tensor = torch.stack(list(images))
        text_tensor = clip_module.tokenize(list(texts), truncate=True)
        return image_tensor, text_tensor

    return collate


def evaluate(wrapper: OfficialOpenAIClipWrapper, dataset, config: dict, batch_images: int, batch_texts: int):
    """Run the same retrieval evaluation used by the zero-shot CLIP script."""
    start = time.time()
    image_features = wrapper.encode_images(dataset.image_paths, batch_size=batch_images)
    image_time = time.time() - start
    start = time.time()
    text_features = wrapper.encode_texts(dataset.captions, batch_size=batch_texts)
    text_time = time.time() - start
    start = time.time()
    similarity = image_features @ text_features.t()
    similarity_time = time.time() - start
    timings = {
        "image_encoding_time": image_time,
        "text_encoding_time": text_time,
        "similarity_time": similarity_time,
        "total_time": image_time + text_time + similarity_time,
    }
    k_values = get_recall_k(config)
    return compute_retrieval_metrics(similarity, dataset.image_to_captions, k_values, timings=timings)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, train_split, val_split, test_split = load_scicap_splits(args.config, args.max_train, args.max_eval)
    ensure_output_dirs(config)
    train_cfg = config.get("training", {}).get("clip_scicap", {})
    eval_cfg = config["evaluation"]
    set_seed(int(config["evaluation"].get("seed", 42)))

    epochs = args.epochs or int(train_cfg.get("epochs", 30))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 32))
    lr = args.lr or float(train_cfg.get("lr", 1e-6))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(train_cfg.get("weight_decay", 0.1))
    min_epochs = args.min_epochs if args.min_epochs is not None else int(train_cfg.get("min_epochs", 0))
    early_stopping_patience = (
        args.early_stopping_patience
        if args.early_stopping_patience is not None
        else train_cfg.get("early_stopping_patience")
    )
    if early_stopping_patience is not None:
        early_stopping_patience = int(early_stopping_patience)
    early_stopping_min_delta = (
        args.early_stopping_min_delta
        if args.early_stopping_min_delta is not None
        else float(train_cfg.get("early_stopping_min_delta", 0.0))
    )
    batch_images = int(eval_cfg.get("batch_size_images", 16))
    batch_texts = int(eval_cfg.get("batch_size_texts", 256))

    model_cfg = config["models"]["clip"]
    wrapper = OfficialOpenAIClipWrapper(model_cfg["model_name"], device="cuda", precision="fp32")
    wrapper.model.train()
    wrapper.model.float()

    train_samples = dataset_to_samples(train_split)
    train_dataset = PairDataset(train_samples, wrapper.preprocess)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=wrapper.device == "cuda",
        drop_last=True,
        collate_fn=make_collate(wrapper.clip),
    )

    optimizer = torch.optim.AdamW(wrapper.model.parameters(), lr=lr, weight_decay=weight_decay)
    # The optional zero-shot snapshot lets the raw JSON record the before/after
    # comparison without relying only on separate result files.
    zero_shot_metrics = None if args.skip_zero_shot else evaluate(wrapper, test_split, config, batch_images, batch_texts)

    best_val = -1.0
    best_epoch: int | None = None
    best_path: Path | None = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = ""
    history: list[dict] = []
    checkpoint_dir = resolve_path(config, "scicap_checkpoint_root")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        wrapper.model.train()
        total_loss = 0.0
        steps = 0
        start = time.time()
        for images, texts in loader:
            images = images.to(wrapper.device, non_blocking=True)
            texts = texts.to(wrapper.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            image_features = wrapper.model.encode_image(images)
            text_features = wrapper.model.encode_text(texts)
            loss = contrastive_loss(image_features, text_features, wrapper.model.logit_scale)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                wrapper.model.logit_scale.clamp_(0, 4.6052)
            total_loss += float(loss.detach().cpu())
            steps += 1

        wrapper.model.eval()
        val_metrics = evaluate(wrapper, val_split, config, batch_images, batch_texts)
        val_mean = float(val_metrics["mean"]["R@1"])
        # Mean R@1 is the validation criterion because it is the primary thesis
        # retrieval metric and is stricter than R@5/R@10.
        improved = best_epoch is None or val_mean > best_val + early_stopping_min_delta
        if improved:
            best_val = val_mean
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "val_mean_R1": val_mean,
            "is_best": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "seconds": time.time() - start,
        }
        history.append(row)
        print(row)
        if improved:
            best_path = checkpoint_dir / scicap_checkpoint_name("clip")
            write_checkpoint(
                best_path,
                {
                    "model_state_dict": wrapper.model.state_dict(),
                    "model_name": wrapper.model_name,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "training_config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "min_epochs": min_epochs,
                        "early_stopping_patience": early_stopping_patience,
                        "early_stopping_min_delta": early_stopping_min_delta,
                    },
                },
            )
        if (
            early_stopping_patience is not None
            and epoch >= min_epochs
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            stop_reason = (
                f"val_mean_R1 did not improve by more than {early_stopping_min_delta} "
                f"for {early_stopping_patience} epochs"
            )
            print(f"Early stopping at epoch {epoch}: {stop_reason}")
            break

    if best_path is not None:
        checkpoint = torch.load(best_path, map_location=wrapper.device)
        wrapper.model.load_state_dict(checkpoint["model_state_dict"])
    wrapper.model.eval()
    val_metrics = evaluate(wrapper, val_split, config, batch_images, batch_texts)
    test_metrics = evaluate(wrapper, test_split, config, batch_images, batch_texts)

    logs_root = resolve_path(config, "scicap_logs_root")
    logs_root.mkdir(parents=True, exist_ok=True)
    history_path = logs_root / f"clip_scicap_finetune_history_{int(time.time())}.json"
    write_training_history(history_path, history)

    payload = build_finetune_payload(
        implementation="official_openai_clip",
        model_info={**wrapper.info.to_dict(), "fine_tuned_on": "SciCap train"},
        dataset=test_split,
        protocol="official_openai_clip_scicap_finetuned_contrastive",
        metrics=test_metrics,
        zero_shot_metrics=zero_shot_metrics,
        val_metrics=val_metrics,
        training={
            "epochs": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "train_samples": len(train_samples),
            "checkpoint": str(best_path) if best_path else "",
            "history": str(history_path),
            "best_epoch": best_epoch,
            "best_val_mean_R1": best_val,
            "early_stopping": {
                "enabled": early_stopping_patience is not None,
                "patience": early_stopping_patience,
                "min_delta": early_stopping_min_delta,
                "min_epochs": min_epochs,
                "stopped_early": stopped_early,
                "stop_reason": stop_reason,
            },
        },
    )
    output_path = write_scicap_finetune_result(config, "clip", payload)

    tables_root = resolve_path(config, "scicap_tables_root")
    tables_root.mkdir(parents=True, exist_ok=True)
    row = {
        "model": "CLIP fine-tuned (SciCap)",
        "implementation": "official_openai_clip",
        "dataset": test_split.name,
        "split": test_split.split,
        **compact_metrics(test_metrics),
        "raw_json": str(output_path),
    }
    from src.eval.common import append_metrics_csv_to
    append_metrics_csv_to(tables_root, "scicap_finetuned_results.csv", row)
    print(f"Fine-tuned CLIP result written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
