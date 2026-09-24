"""Generate publication figures that are independent of model checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


COLORS = {
    "shared": "#4C78A8",
    "input": "#59A14F",
    "output": "#E15759",
    "neutral": "#6B7280",
    "light": "#F3F4F6",
}


def box(ax, xy, width, height, label, color, fontsize=10):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.3,
        edgecolor=color,
        facecolor="white",
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, label, ha="center", va="center", fontsize=fontsize)


def arrow(ax, start, end, color=COLORS["neutral"], style="-|>"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12, linewidth=1.2, color=color))


def architecture_figure(output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
    titles = ["Tied", "Untied", "Partial tying"]
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    ax = axes[0]
    box(ax, (0.08, 0.70), 0.32, 0.12, "token", COLORS["neutral"])
    box(ax, (0.60, 0.70), 0.32, 0.12, "hidden state", COLORS["neutral"])
    box(ax, (0.30, 0.42), 0.40, 0.14, "W_shared", COLORS["shared"], 11)
    box(ax, (0.30, 0.12), 0.40, 0.14, "W_shared.T", COLORS["shared"], 11)
    arrow(ax, (0.40, 0.70), (0.42, 0.56))
    arrow(ax, (0.60, 0.70), (0.58, 0.26))
    ax.text(0.50, 0.34, "same parameter", ha="center", va="center", fontsize=9, color=COLORS["shared"])

    ax = axes[1]
    box(ax, (0.06, 0.70), 0.28, 0.12, "token", COLORS["neutral"])
    box(ax, (0.66, 0.70), 0.28, 0.12, "hidden state", COLORS["neutral"])
    box(ax, (0.06, 0.42), 0.38, 0.14, "W_input", COLORS["input"], 11)
    box(ax, (0.56, 0.12), 0.38, 0.14, "W_output.T", COLORS["output"], 11)
    arrow(ax, (0.20, 0.70), (0.22, 0.56), COLORS["input"])
    arrow(ax, (0.80, 0.70), (0.78, 0.26), COLORS["output"])
    ax.text(0.50, 0.34, "independent full matrices", ha="center", va="center", fontsize=9)

    ax = axes[2]
    box(ax, (0.06, 0.72), 0.25, 0.10, "token", COLORS["neutral"])
    box(ax, (0.69, 0.72), 0.25, 0.10, "hidden state", COLORS["neutral"])
    box(ax, (0.34, 0.45), 0.32, 0.12, "W_shared", COLORS["shared"], 10)
    box(ax, (0.04, 0.20), 0.26, 0.10, "+ A_i B_i.T", COLORS["input"], 9)
    box(ax, (0.70, 0.20), 0.26, 0.10, "+ A_o B_o.T", COLORS["output"], 9)
    arrow(ax, (0.19, 0.72), (0.42, 0.57), COLORS["input"])
    arrow(ax, (0.81, 0.72), (0.58, 0.57), COLORS["output"])
    arrow(ax, (0.43, 0.45), (0.25, 0.30), COLORS["input"])
    arrow(ax, (0.57, 0.45), (0.75, 0.30), COLORS["output"])
    ax.text(0.50, 0.08, "shared base + small role-specific residuals", ha="center", va="center", fontsize=8.5)

    fig.suptitle("Three ways to parameterize input/output token embeddings", fontsize=15, y=1.08)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/primary/figures")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    architecture_figure(output_dir / "architecture.png")


if __name__ == "__main__":
    main()
