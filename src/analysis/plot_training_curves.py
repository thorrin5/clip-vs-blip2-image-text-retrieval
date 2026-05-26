"""
Generovanie tréningových kriviek pre SciCap fine-tuning.

Skript číta uložené per-epoch histórie modelov CLIP a BLIP-2 a vytvára graf
validačnej metriky Mean R@1 spolu s tréningovou stratou. Graf slúži na
vysvetlenie správania fine-tuningu a výberu najlepšieho checkpointu bez toho,
aby bolo potrebné znovu spúšťať dlhé GPU tréningy.
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

# Verejná verzia obsahuje iba SciCap histórie použité vo finálnom porovnaní.
HISTORY = {
    "scicap": {
        "CLIP": RESULTS / "training_history/clip_scicap_45k_epochs.json",
        "BLIP-2": RESULTS / "training_history/blip2_scicap_45k_epochs.json",
    },
}

COLORS = {"CLIP": "#2166ac", "BLIP-2": "#d6604d"}
MARKERS = {"CLIP": "o", "BLIP-2": "s"}


def fmt_percent(value: float) -> str:
    """Zaokrúhli percentá rovnakým spôsobom, ako sa uvádzajú v tabuľkách."""
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def load(path: Path) -> list[dict]:
    """Načíta JSON históriu tréningu ako zoznam epochových záznamov."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def plot_dataset(dataset_key: str, title: str, out_path: Path) -> None:
    """
    Vykreslí validačné Mean R@1 a tréningovú stratu pre jeden dataset.

    Ľavý graf ukazuje kvalitu retrieval modelu na validačnej množine, pravý graf
    ukazuje optimalizačný priebeh. Zvislá čiara označuje epochu najlepšieho
    checkpointu podľa validačného Mean R@1.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)

    histories = {}
    for model, path in HISTORY[dataset_key].items():
        if path.exists():
            histories[model] = load(path)
        else:
            print(f"  [MISSING] {path.name}")

    # Validačná metrika Mean R@1 je hlavný signál pre výber checkpointu.
    ax = axes[0]
    for model, data in histories.items():
        epochs = [record["epoch"] for record in data]
        values = [record["val_mean_R1"] for record in data]
        best_epoch = max((record["epoch"] for record in data if record.get("is_best")), default=None)
        best_value = next((record["val_mean_R1"] for record in data if record["epoch"] == best_epoch), None)

        ax.plot(epochs, values, color=COLORS[model], marker=MARKERS[model], markersize=5, linewidth=2, label=model)
        if best_epoch is not None:
            ax.axvline(best_epoch, color=COLORS[model], linestyle="--", linewidth=1, alpha=0.5)
            ax.scatter(
                [best_epoch],
                [best_value],
                color=COLORS[model],
                s=80,
                zorder=5,
                marker="*",
                label=f"{model} best (ep {best_epoch}: {fmt_percent(best_value)}%)",
            )

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Val Mean R@1 (%)", fontsize=11)
    ax.set_title("Validation Mean R@1", fontsize=12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # Tréningová strata dopĺňa interpretáciu: ukazuje, či optimalizácia klesala.
    ax = axes[1]
    for model, data in histories.items():
        epochs = [record["epoch"] for record in data]
        losses = [record["train_loss"] for record in data]
        ax.plot(epochs, losses, color=COLORS[model], marker=MARKERS[model], markersize=5, linewidth=2, label=model)

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
    """Vygeneruje finálny SciCap graf použitý v dokumentácii repozitára."""
    plot_dataset(
        "scicap",
        "SciCap Fine-tuning - CLIP vs BLIP-2 (45,000 train pairs)",
        OUT_DIR / "scicap_training_curves.png",
    )
    print(f"\nAll plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
