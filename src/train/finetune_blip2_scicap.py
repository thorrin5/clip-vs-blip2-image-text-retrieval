"""
Purpose:
    Fine-tune Hugging Face BLIP-2 retrieval heads on SciCap pairs.

Thesis context:
    This script adapts Salesforce/blip2-itm-vit-g for SciCap while keeping the
    large vision backbone frozen. It is the BLIP-2 fine-tuning counterpart to
    the CLIP SciCap fine-tuning script.

Inputs:
    - Experiment YAML config.
    - data/scicap_processed/{train,val,test}.jsonl.
    - SciCap images referenced by the JSONL rows.

Outputs:
    - Trainable BLIP-2 checkpoint state under results/scicap/checkpoints.
    - Per-epoch training history JSON.
    - Raw fine-tuned SciCap test metrics JSON.
    - CSV row in results/scicap/tables/scicap_finetuned_results.csv.

Defense note:
    This script explains why BLIP-2 fine-tuning is computationally heavier than
    CLIP fine-tuning. Only retrieval-related parameters are trained, but
    validation and final testing can still include expensive ITM reranking.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.hf_blip2_wrapper import HuggingFaceBlip2RetrievalWrapper
from src.train.common import (
    PairDataset,
    dataset_to_samples,
    load_scicap_splits,
    scicap_checkpoint_name,
    set_seed,
    write_checkpoint,
    write_scicap_finetune_result,
    write_training_history,
)
from src.eval.common import append_metrics_csv_to
from src.eval.evaluate_blip2 import _compute_topk_itm_metrics
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
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--rerank-top-k", type=int, default=None)
    parser.add_argument("--pair-batch-size", type=int, default=None)
    parser.add_argument("--itm-loss-weight", type=float, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--skip-zero-shot", action="store_true")
    return parser


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    """Small local scheduler to avoid adding a dependency just for LR decay."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def set_trainable(model) -> None:
    """Freeze the large BLIP-2 backbone and train only retrieval-related parts."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    trainable_prefixes = (
        "query_tokens",
        "qformer",
        "vision_projection",
        "text_projection",
        "itm_head",
        "embeddings",
    )
    for name, parameter in model.named_parameters():
        if name.startswith(trainable_prefixes):
            parameter.requires_grad = True


def trainable_state_dict(model) -> dict:
    """Save only trainable weights so checkpoints stay smaller and clearer."""
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in {n for n, p in model.named_parameters() if p.requires_grad}}


def make_collate(processor):
    def collate(batch):
        images, texts = zip(*batch)
        inputs = processor(images=list(images), text=list(texts), return_tensors="pt", padding=True, truncation=True)
        return inputs

    return collate


def itc_loss(outputs):
    """Symmetric contrastive loss over the current micro-batch."""
    logits = outputs.logits_per_image.float()
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


def itm_loss(model, inputs):
    """Construct simple positive/negative ITM pairs inside the batch."""
    batch_size = inputs["input_ids"].shape[0]
    if batch_size < 2:
        return inputs["input_ids"].new_tensor(0.0, dtype=torch.float32)
    shifted = torch.roll(torch.arange(batch_size, device=inputs["input_ids"].device), shifts=1)
    pixel_values = torch.cat([inputs["pixel_values"], inputs["pixel_values"]], dim=0)
    input_ids = torch.cat([inputs["input_ids"], inputs["input_ids"][shifted]], dim=0)
    attention_mask = torch.cat([inputs["attention_mask"], inputs["attention_mask"][shifted]], dim=0)
    outputs = model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_image_text_matching_head=True,
    )
    labels = torch.cat(
        [
            torch.ones(batch_size, dtype=torch.long, device=input_ids.device),
            torch.zeros(batch_size, dtype=torch.long, device=input_ids.device),
        ]
    )
    return F.cross_entropy(outputs.logits_per_image.float(), labels)


def evaluate(wrapper: HuggingFaceBlip2RetrievalWrapper, dataset, config: dict, rerank_top_k: int | None, pair_batch_size: int):
    """Evaluate BLIP-2 with the same ITC plus optional ITM pipeline as zero-shot."""
    eval_cfg = config["evaluation"]
    batch_images = int(eval_cfg.get("batch_size_images", 16))
    batch_texts = int(eval_cfg.get("batch_size_texts", 256))
    start = time.time()
    image_features = wrapper.encode_images(dataset.image_paths, batch_size=batch_images)
    image_time = time.time() - start
    start = time.time()
    text_features = wrapper.encode_texts(dataset.captions, batch_size=batch_texts)
    text_time = time.time() - start
    start = time.time()
    similarity = wrapper.compute_itc_similarity(image_features, text_features, image_batch_size=batch_images)
    similarity_time = time.time() - start
    timings = {
        "image_encoding_time": image_time,
        "text_encoding_time": text_time,
        "similarity_time": similarity_time,
        "total_time": image_time + text_time + similarity_time,
    }
    k_values = [int(k) for k in config["evaluation"]["recall_k"]]
    metrics = compute_retrieval_metrics(similarity, dataset.image_to_captions, k_values, timings=timings)
    if rerank_top_k is not None:
        metrics, rerank_timings = _compute_topk_itm_metrics(wrapper, similarity, dataset, rerank_top_k, k_values, pair_batch_size)
        timings.update(rerank_timings)
        metrics["timing"] = timings
    return metrics


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, train_split, val_split, test_split = load_scicap_splits(args.config, args.max_train, args.max_eval)
    ensure_output_dirs(config)
    train_cfg = config.get("training", {}).get("blip2_scicap", {})
    set_seed(int(config["evaluation"].get("seed", 42)))

    epochs = args.epochs or int(train_cfg.get("epochs", 20))
    batch_size = args.batch_size or int(train_cfg.get("batch_size", 8))
    lr = args.lr or float(train_cfg.get("lr", 1e-5))
    weight_decay = args.weight_decay if args.weight_decay is not None else float(train_cfg.get("weight_decay", 0.05))
    itm_loss_weight_val = args.itm_loss_weight if args.itm_loss_weight is not None else float(train_cfg.get("itm_loss_weight", 0.2))
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
    warmup_steps = args.warmup_steps if args.warmup_steps is not None else int(train_cfg.get("warmup_steps", 200))
    grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else int(train_cfg.get("grad_accum_steps", 8))
    rerank_top_k = args.rerank_top_k if args.rerank_top_k is not None else int(config["evaluation"].get("blip2_rerank_top_k", 128))
    pair_batch_size = args.pair_batch_size or int(config["evaluation"].get("blip2_pair_batch_size", 32))

    model_cfg = config["models"]["blip2"]
    wrapper = HuggingFaceBlip2RetrievalWrapper(model_cfg["model_name"], device="cuda", precision="fp32")
    # Keeping the ViT-G vision encoder frozen makes the experiment feasible on
    # the cluster and matches the thesis explanation of trainable parameters.
    set_trainable(wrapper.model)
    wrapper.model.train()

    train_samples = dataset_to_samples(train_split)
    train_dataset = PairDataset(train_samples, lambda image: image)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=wrapper.device == "cuda",
        drop_last=True,
        collate_fn=make_collate(wrapper.processor),
    )

    optimizer = torch.optim.AdamW(
        (p for p in wrapper.model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
    )
    steps_per_epoch = max(1, len(loader) // grad_accum_steps)
    total_optimizer_steps = epochs * steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_optimizer_steps)
    print(f"LR schedule: warmup={warmup_steps} steps, total={total_optimizer_steps} optimizer steps, grad_accum={grad_accum_steps}")

    zero_shot_metrics = None if args.skip_zero_shot else evaluate(wrapper, test_split, config, rerank_top_k, pair_batch_size)

    best_val = -1.0
    best_epoch: int | None = None
    best_path: Path | None = None
    epochs_without_improvement = 0
    stopped_early = False
    stop_reason = ""
    history: list[dict] = []
    checkpoint_dir = resolve_path(config, "scicap_checkpoint_root")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        wrapper.model.train()
        total_loss = 0.0
        total_itc = 0.0
        total_itm = 0.0
        micro_steps = 0
        optimizer_steps = 0
        start = time.time()
        for inputs in loader:
            inputs = inputs.to(wrapper.device)
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=wrapper.dtype)
            outputs = wrapper.model(**inputs, use_image_text_matching_head=False)
            loss_itc = itc_loss(outputs)
            loss_itm = itm_loss(wrapper.model, inputs) if itm_loss_weight_val else loss_itc.new_tensor(0.0)
            loss = loss_itc + itm_loss_weight_val * loss_itm
            (loss / grad_accum_steps).backward()
            total_loss += float(loss.detach().cpu())
            total_itc += float(loss_itc.detach().cpu())
            total_itm += float(loss_itm.detach().cpu())
            micro_steps += 1
            if micro_steps % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in wrapper.model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        if micro_steps % grad_accum_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in wrapper.model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
        steps = micro_steps

        wrapper.model.eval()
        val_metrics = evaluate(wrapper, val_split, config, rerank_top_k, pair_batch_size)
        val_mean = float(val_metrics["mean"]["R@1"])
        # Validation Mean R@1 selects the checkpoint that best matches the main
        # thesis metric before the final test split is touched.
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
            "itc_loss": total_itc / max(steps, 1),
            "itm_loss": total_itm / max(steps, 1),
            "val_mean_R1": val_mean,
            "is_best": improved,
            "epochs_without_improvement": epochs_without_improvement,
            "seconds": time.time() - start,
        }
        history.append(row)
        print(row)
        if improved:
            best_path = checkpoint_dir / scicap_checkpoint_name("blip2")
            write_checkpoint(
                best_path,
                {
                    "trainable_state_dict": trainable_state_dict(wrapper.model),
                    "model_name": wrapper.model_name,
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "training_config": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "itm_loss_weight": itm_loss_weight_val,
                        "warmup_steps": warmup_steps,
                        "grad_accum_steps": grad_accum_steps,
                        "effective_batch_size": batch_size * grad_accum_steps,
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
        wrapper.model.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    wrapper.model.eval()
    val_metrics = evaluate(wrapper, val_split, config, rerank_top_k, pair_batch_size)
    test_metrics = evaluate(wrapper, test_split, config, rerank_top_k, pair_batch_size)

    logs_root = resolve_path(config, "scicap_logs_root")
    logs_root.mkdir(parents=True, exist_ok=True)
    history_path = logs_root / f"blip2_scicap_finetune_history_{int(time.time())}.json"
    write_training_history(history_path, history)

    payload = {
        "status": "ok",
        "implementation": "huggingface_transformers",
        "model": {**wrapper.info.to_dict(), "fine_tuned_on": "SciCap train"},
        "dataset": test_split.to_metadata(),
        "protocol": "huggingface_blip2_scicap_finetuned_itc_itm_topk_rerank",
        "metrics": test_metrics,
        "validation_metrics": val_metrics,
        "training": {
            "epochs": epochs,
            "epochs_completed": len(history),
            "batch_size": batch_size,
            "effective_batch_size": batch_size * grad_accum_steps,
            "grad_accum_steps": grad_accum_steps,
            "lr": lr,
            "warmup_steps": warmup_steps,
            "weight_decay": weight_decay,
            "itm_loss_weight": itm_loss_weight_val,
            "train_samples": len(train_samples),
            "checkpoint": str(best_path) if best_path else "",
            "history": str(history_path),
            "rerank_top_k": rerank_top_k,
            "pair_batch_size": pair_batch_size,
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
    }
    if zero_shot_metrics is not None:
        payload["zero_shot_metrics"] = zero_shot_metrics

    output_path = write_scicap_finetune_result(config, "blip2", payload)

    tables_root = resolve_path(config, "scicap_tables_root")
    tables_root.mkdir(parents=True, exist_ok=True)
    row = {
        "model": "BLIP-2 fine-tuned (SciCap)",
        "implementation": "huggingface_transformers",
        "dataset": test_split.name,
        "split": test_split.split,
        **compact_metrics(test_metrics),
        "raw_json": str(output_path),
    }
    append_metrics_csv_to(tables_root, "scicap_finetuned_results.csv", row)
    print(f"Fine-tuned BLIP-2 result written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
