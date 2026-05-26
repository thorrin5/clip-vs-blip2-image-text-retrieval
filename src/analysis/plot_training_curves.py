"""
Purpose:
    Generate training-curve PNG plots from saved epoch history JSON files.

Thesis context:
    This plotting utility supports the fine-tuning analysis for CLIP and BLIP-2.
    The SciCap curve is especially relevant to the final thesis comparison.

Inputs:
    - Selected training-history JSON files configured in the HISTORY mapping.

Outputs:
    - results/figures/training/ai2d_training_curves.png.
    - results/figures/training/scicap_training_curves.png.

Defense note:
    These plots show how validation Mean R@1 and training loss evolved during
    fine-tuning. They are useful for explaining early stopping and why a
    specific checkpoint was selected.

Additional implementation notes:

This script reads saved per-epoch histories and plots validation Mean R@1 plus
training loss. It is used for defense figures that explain early stopping and
fine-tuning behaviour without rerunning any training jobs.

Outputs (to results/figures/training/):
  ai2d_training_curves.png   — CLIP vs BLIP-2 on AI2D
  scicap_training_curves.png — CLIP vs BLIP-2 on SciCap
"""

from __future__ import annotations

import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = Path(__file__).resolve().parents[2]
RESULTS = BASE / "results"
OUT_DIR = BASE / "figures/plots"

# These are the public thesis histories included in this repository.
HISTORY = {
    "ai2d": {
        "CLIP":   RESULTS / "training_history/clip_ai2d_epochs.json",
        "BLIP-2": RESULTS / "training_history/blip2_ai2d_epochs.json",
    },
    "scicap": {
        "CLIP":   RESULTS / "training_history/clip_scicap_45k_epochs.json",
        "BLIP-2": RESULTS / "training_history/blip2_scicap_45k_epochs.json",
    },
}

COLORS = {"CLIP": "#2166ac", "BLIP-2": "#d6604d"}
MARKERS = {"CLIP": "o", "BLIP-2": "s"}


def fmt_percent(value: float) -> str:
    """Round percentages in the same way they are normally reported in tables."""
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def load(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def plot_dataset(dataset_key: str, title: str, out_path: Path) -> None:
    """Plot validation Mean R@1 and training loss for one dataset."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    histories = {}
    for model, path in HISTORY[dataset_key].items():
        if path.exists():
            histories[model] = load(path)
        else:
            print(f"  [MISSING] {path.name}")

    # ── Left: val_mean_R1 ───────────────────────────────────────────────────
    ax = axes[0]
    for model, data in histories.items():
        epochs = [d["epoch"] for d in data]
        vals   = [d["val_mean_R1"] for d in data]
        best_e = next((d["epoch"] for d in data if d.get("is_best")), None)
        # last is_best = True is the actual best
        best_e = max((d["epoch"] for d in data if d.get("is_best")), default=None)
        best_v = next((d["val_mean_R1"] for d in data if d["epoch"] == best_e), None)

        ax.plot(epochs, vals, color=COLORS[model], marker=MARKERS[model],
                markersize=5, linewidth=2, label=model)
        if best_e is not None:
            ax.axvline(best_e, color=COLORS[model], linestyle="--", linewidth=1, alpha=0.5)
            ax.scatter([best_e], [best_v], color=COLORS[model], s=80, zorder=5,
                       marker="*", label=f"{model} best (ep {best_e}: {fmt_percent(best_v)}%)")

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val Mean R@1 (%)", fontsize=11)
    ax.set_title("Validation Mean R@1", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # ── Right: train_loss ───────────────────────────────────────────────────
    ax = axes[1]
    for model, data in histories.items():
        epochs = [d["epoch"] for d in data]
        losses = [d["train_loss"] for d in data]
        ax.plot(epochs, losses, color=COLORS[model], marker=MARKERS[model],
                markersize=5, linewidth=2, label=model)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Train Loss", fontsize=11)
    ax.set_title("Training Loss", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {out_path.name}")


def main() -> None:
    plot_dataset(
        "ai2d",
        "AI2D Fine-tuning - CLIP vs BLIP-2 (2,470 train pairs)",
        OUT_DIR / "ai2d_training_curves.png",
    )
    plot_dataset(
        "scicap",
        "SciCap Fine-tuning - CLIP vs BLIP-2 (45,000 train pairs)",
        OUT_DIR / "scicap_training_curves.png",
    )
    print(f"\nAll plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
