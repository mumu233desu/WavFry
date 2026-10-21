"""
Publication-grade Figure 2: Noise Robustness across Acoustic Degradation Tiers.
Reads data/experiments/baselines/vowel_robustness_metrics.json and renders
high-resolution comparison curves for IEEE/Elsevier CSL journal submission.
"""

from __future__ import annotations

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
METRICS_JSON = PROJECT_ROOT / "data/experiments/baselines/vowel_robustness_metrics.json"
OUTPUT_DIR = PROJECT_ROOT / "data/visualizations/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_fig2():
    if not METRICS_JSON.exists():
        raise FileNotFoundError(f"Metrics file not found: {METRICS_JSON}")

    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    tiers = ["Clean", "15dB", "5dB", "0dB", "Compound"]
    tier_labels = ["Clean", "Mild\n(15 dB)", "Moderate\n(5 dB)", "Severe\n(0 dB)", "Compound\n(Noise+RIR)"]

    models_config = {
        "WavFry (Ours)": {
            "color": "#D32F2F",  # Deep Crimson / Coral
            "marker": "o",
            "linestyle": "-",
            "linewidth": 2.8,
            "markersize": 8,
            "zorder": 5,
        },
        "DeepFry": {
            "color": "#1976D2",  # Cobalt Blue
            "marker": "s",
            "linestyle": "--",
            "linewidth": 1.8,
            "markersize": 6,
            "zorder": 4,
        },
        "Kane": {
            "color": "#388E3C",  # Forest Green
            "marker": "^",
            "linestyle": "-.",
            "linewidth": 1.8,
            "markersize": 6,
            "zorder": 3,
        },
        "Creapy": {
            "color": "#F57C00",  # Amber Orange
            "marker": "D",
            "linestyle": ":",
            "linewidth": 1.8,
            "markersize": 6,
            "zorder": 2,
        },
        "Praat": {
            "color": "#7B1FA2",  # Purple
            "marker": "v",
            "linestyle": "--",
            "linewidth": 1.6,
            "markersize": 6,
            "zorder": 1,
        },
    }

    # Set publication typography and style
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#E0E0E0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
    })

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

    x = np.arange(len(tiers))

    for model_name, style in models_config.items():
        f1_scores = []
        for tier in tiers:
            tier_data = metrics.get(tier, {})
            model_data = tier_data.get(model_name, {})
            f1 = model_data.get("F1-Score", np.nan)
            f1_scores.append(f1)

        ax.plot(
            x,
            f1_scores,
            label=model_name,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            zorder=style["zorder"],
            markeredgewidth=1.2,
            markeredgecolor="white",
        )

        # Highlight WavFry values with small annotations
        if model_name == "WavFry (Ours)":
            for xi, yi in zip(x, f1_scores):
                if not np.isnan(yi):
                    ax.annotate(
                        f"{yi:.3f}",
                        (xi, yi),
                        textcoords="offset points",
                        xytext=(0, 9),
                        ha="center",
                        fontsize=8.5,
                        fontweight="bold",
                        color=style["color"],
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(tier_labels, fontsize=11, fontweight="medium")
    ax.set_ylabel("F1-Score", fontsize=12, fontweight="bold")
    ax.set_xlabel("Acoustic Degradation Tier", fontsize=12, fontweight="bold", labelpad=8)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.6)

    ax.legend(
        loc="lower left",
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=10.5,
        ncol=2,
    )

    ax.set_title(
        "Figure 2: Model Robustness Across QUT Acoustic Degradation Tiers",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    plt.tight_layout()

    out_png = OUTPUT_DIR / "fig2_noise_robustness.png"
    out_pdf = OUTPUT_DIR / "fig2_noise_robustness.pdf"

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Successfully generated Figure 2:")
    print(f"  PNG: {out_png}")
    print(f"  PDF: {out_pdf}")


if __name__ == "__main__":
    plot_fig2()
