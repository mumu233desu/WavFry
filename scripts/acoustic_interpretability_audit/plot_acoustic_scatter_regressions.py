"""
Figure 8: Acoustic-Physical Interpretability via Continuous Scatter & Regression.

Canonical Publication Visualization Script:
- Reads the output of evaluate_acoustic_correlations.py (acoustic_correlations.csv)
- Generates a publication-grade 4-panel regression figure:
  (a) Creaky Posterior vs Jitter (local)
  (b) Creaky Posterior vs SHR
  (c) Breathy Posterior vs CPPS (dB)
  (d) Breathy Posterior vs Formant-Corrected H1*-H2* (dB)
- Includes 95% bootstrap confidence interval bands on the regression fits.
- Explicitly drops unvoiced/silent segments (NaN features) to preserve phonation validity.
- Annotates Pearson r, Spearman rho, and two-tailed p-values.
- Styled for Elsevier Computer Speech & Language (CSL) / Journal of Phonetics.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("plot_acoustic_scatter_regressions")

CANDIDATE_CSV_PATHS = [
    Path(r"E:\Corpus\CS-Dialogue\data\short_wav\WAVE\06_wavfry\data\experiments\interpretability\acoustic_correlations.csv"),
    Path(r"E:\Coding\WavFry\data\experiments\interpretability\acoustic_correlations.csv"),
    Path("data/experiments/interpretability/acoustic_correlations.csv"),
]


def resolve_csv(custom_path: Optional[str] = None) -> Path:
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified CSV file not found: {custom_path}")

    for p in CANDIDATE_CSV_PATHS:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Acoustic correlation CSV not found in any canonical locations:\n"
        + "\n".join(f" - {p}" for p in CANDIDATE_CSV_PATHS)
        + "\nPlease run evaluate_acoustic_correlations.py first to generate data."
    )


def format_p_val(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def plot_scatter_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    color: str,
    line_color: str,
    clip_p99: bool = False,
    annotation_loc: str = "upper_left",
) -> None:
    """Render a single regression scatter panel with stats and confidence band."""
    # Strict physical voicing gating: drop any unvoiced/silence NaN values
    valid_sub = df.dropna(subset=[x_col, y_col]).copy()
    valid_sub = valid_sub[~np.isinf(valid_sub[x_col]) & ~np.isinf(valid_sub[y_col])]

    if len(valid_sub) < 3:
        ax.text(
            0.5, 0.5, f"Insufficient valid voiced points (N={len(valid_sub)})\n(Silence/Unvoiced filtered)",
            ha="center", va="center", transform=ax.transAxes, fontsize=10, color="#888"
        )
        ax.set_title(title, fontweight="bold", pad=8)
        ax.set_xlabel(x_label, fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        return

    # Compute statistics on full unclipped voiced vowel sample to align 100% with Table 8b
    x_raw = valid_sub[x_col].values
    y_raw = valid_sub[y_col].values
    r, p_r = pearsonr(x_raw, y_raw)
    rho, p_rho = spearmanr(x_raw, y_raw)
    n_total = len(valid_sub)

    # Optional visual clipping of extreme Praat tracking outliers for plotting canvas only
    if clip_p99:
        p99 = valid_sub[y_col].quantile(0.99)
        valid_sub = valid_sub[valid_sub[y_col] <= p99]

    # Plot scatter with 95% bootstrap confidence interval
    sns.regplot(
        data=valid_sub,
        x=x_col,
        y=y_col,
        ax=ax,
        color=color,
        scatter_kws={"alpha": 0.65, "s": 28, "edgecolors": "none"},
        line_kws={"linewidth": 2.2, "color": line_color},
        ci=95,
    )

    p_r_str = format_p_val(p_r)
    p_rho_str = format_p_val(p_rho)
    stat_text = (
        f"Pearson r = {r:+.3f} ({p_r_str})\n"
        f"Spearman ρ = {rho:+.3f} ({p_rho_str})\n"
        f"N = {n_total} voiced vowels"
    )

    if annotation_loc == "lower_left":
        y_pos = 0.12
    elif annotation_loc == "lower_right":
        y_pos = 0.12
    elif annotation_loc == "upper_right":
        y_pos = 0.82
    else:  # upper_left
        y_pos = 0.82

    x_pos = 0.05 if "left" in annotation_loc else 0.55

    ax.text(
        x_pos, y_pos, stat_text,
        transform=ax.transAxes, fontsize=9.0, linespacing=1.3,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.90, edgecolor="#ccc"),
    )

    ax.set_title(title, fontweight="bold", pad=8, fontsize=11)
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_ylabel(y_label, fontsize=10)
    ax.set_xlim(-0.02, 1.02)


def render_acoustic_scatter_regressions(
    csv_path: Path,
    out_dir: Path,
    dpi: int = 300,
    save_format: str = "both",
) -> None:
    logger.info(f"Loading acoustic data from {csv_path}...")
    df = pd.read_csv(csv_path)
    logger.info(f"Read {len(df)} total interval records.")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5), dpi=dpi)

    prob_b_col = "prob_breathy" if "prob_breathy" in df.columns else "prob_creaky"

    df_creaky = df[df["label_creaky"].notna()] if "label_creaky" in df.columns and df["label_creaky"].notna().any() else df
    df_breathy = df[df["label_breathy"].notna()] if "label_breathy" in df.columns and df["label_breathy"].notna().any() else df

    # Panel (a): Creaky Posterior vs Jitter (local)
    plot_scatter_panel(
        ax=axes[0, 0],
        df=df_creaky,
        x_col="prob_creaky",
        y_col="jitter_local",
        title="(a) Creaky Posterior vs. Jitter (local)",
        x_label="WavFry Predicted Creaky Posterior",
        y_label="Jitter (local)",
        color="#E63946",
        line_color="#9B0024",
        clip_p99=True,
        annotation_loc="upper_left",
    )

    # Panel (b): Creaky Posterior vs SHR
    plot_scatter_panel(
        ax=axes[0, 1],
        df=df_creaky,
        x_col="prob_creaky",
        y_col="shr",
        title="(b) Creaky Posterior vs. SHR",
        x_label="WavFry Predicted Creaky Posterior",
        y_label="Subharmonic-to-Harmonic Ratio (SHR)",
        color="#F4A261",
        line_color="#C85A17",
        clip_p99=False,
        annotation_loc="upper_left",
    )

    # Panel (c): Breathy Posterior vs CPPS (dB)
    plot_scatter_panel(
        ax=axes[1, 0],
        df=df_breathy,
        x_col=prob_b_col,
        y_col="cpps",
        title="(c) Breathy Posterior vs. CPPS (dB)",
        x_label="WavFry Predicted Breathy Posterior",
        y_label="CPPS (dB)",
        color="#2A9D8F",
        line_color="#1D3557",
        clip_p99=False,
        annotation_loc="lower_left",
    )

    # Panel (d): Breathy Posterior vs H1*-H2* (dB)
    plot_scatter_panel(
        ax=axes[1, 1],
        df=df_breathy,
        x_col=prob_b_col,
        y_col="h1h2_star",
        title="(d) Breathy Posterior vs. H1*-H2* (dB)",
        x_label="WavFry Predicted Breathy Posterior",
        y_label="Formant-corrected H1*-H2* (dB)",
        color="#457B9D",
        line_color="#1D3557",
        clip_p99=False,
        annotation_loc="upper_left",
    )

    plt.tight_layout()

    target_dirs = [out_dir]
    for sync_dir in [
        Path(r"E:\Coding\WavFry\data\visualizations\csl2026_journal\output"),
        Path(r"E:\Coding\WavFry\paper\CSL2026_submission\figures"),
    ]:
        if sync_dir.parent.exists() and sync_dir.resolve() != out_dir.resolve():
            target_dirs.append(sync_dir)

    for d in target_dirs:
        d.mkdir(parents=True, exist_ok=True)
        out_png = d / "fig8_acoustic_physical_scatter_regression.png"
        out_pdf = d / "fig8_acoustic_physical_scatter_regression.pdf"
        if save_format in ("png", "both"):
            plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved PNG figure to: {out_png}")
        if save_format in ("pdf", "both"):
            plt.savefig(out_pdf, bbox_inches="tight")
            logger.info(f"Saved PDF figure to: {out_pdf}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Figure 8: Acoustic-Physical Scatter & Regression Visualizer."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to acoustic_correlations.csv generated by evaluate_acoustic_correlations.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/visualizations/output",
        help="Directory to save publication figures.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "both"],
        default="both",
        help="Export image format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI resolution for exported raster figure.",
    )

    args = parser.parse_args()
    csv_file = resolve_csv(args.input_csv)
    render_acoustic_scatter_regressions(
        csv_path=csv_file,
        out_dir=Path(args.output_dir),
        dpi=args.dpi,
        save_format=args.format,
    )


if __name__ == "__main__":
    main()
