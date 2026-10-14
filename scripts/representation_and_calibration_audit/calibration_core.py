"""
Core mathematical functions and visualization utilities for model confidence
calibration (ECE, MCE, Brier Score, and Reliability Diagrams).

References:
    - Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
    - Niculescu-Mizil & Caruana, "Predicting Good Probabilities With Supervised Learning", ICML 2005.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_1d_numpy(x: Union[torch.Tensor, np.ndarray, List[float]]) -> np.ndarray:
    """Ensure input is a 1D float64 numpy array."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().to(torch.float64).numpy()
    elif isinstance(x, (list, tuple)):
        x = np.array(x, dtype=np.float64)
    elif isinstance(x, np.ndarray):
        if x.dtype != np.float64:
            x = x.astype(np.float64)
    else:
        raise TypeError(f"Expected torch.Tensor, np.ndarray, or list, got {type(x)}")
    x = x.reshape(-1)
    return x


def compute_reliability_diagram_bins(
    probs: Union[torch.Tensor, np.ndarray, List[float]],
    labels: Union[torch.Tensor, np.ndarray, List[float]],
    num_bins: int = 10,
) -> Dict[str, Any]:
    """
    Partition predicted probabilities and labels into equal-width calibration bins.

    Args:
        probs: 1D array of predicted probabilities in [0, 1].
        labels: 1D array of true binary labels {0, 1}.
        num_bins: Number of equal-width bins (default 10).

    Returns:
        Dict containing bin edges, centers, accuracies, confidences, counts, and gaps.
    """
    p = _to_1d_numpy(probs)
    y = _to_1d_numpy(labels)

    if len(p) != len(y):
        raise ValueError(f"Length mismatch: probs has {len(p)}, labels has {len(y)}")

    # Filter out potential NaNs
    valid_mask = (~np.isnan(p)) & (~np.isnan(y))
    p = np.clip(p[valid_mask], 0.0, 1.0)
    y = y[valid_mask]

    n_samples = len(p)
    if n_samples == 0:
        return {
            "bin_edges": np.linspace(0.0, 1.0, num_bins + 1),
            "bin_centers": np.linspace(0.5 / num_bins, 1.0 - 0.5 / num_bins, num_bins),
            "accuracies": np.zeros(num_bins),
            "confidences": np.zeros(num_bins),
            "counts": np.zeros(num_bins, dtype=int),
            "gaps": np.zeros(num_bins),
            "ece": 0.0,
            "mce": 0.0,
            "brier_score": 0.0,
            "total_samples": 0,
        }

    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    accuracies = np.zeros(num_bins, dtype=np.float64)
    confidences = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)
    gaps = np.zeros(num_bins, dtype=np.float64)

    weighted_ece = 0.0
    max_ce = 0.0

    for i in range(num_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # For the last bin, include the upper edge 1.0
        if i == num_bins - 1:
            in_bin = (p >= bin_lower) & (p <= bin_upper)
        else:
            in_bin = (p >= bin_lower) & (p < bin_upper)

        count = int(np.sum(in_bin))
        counts[i] = count

        if count > 0:
            bin_acc = float(np.mean(y[in_bin]))
            bin_conf = float(np.mean(p[in_bin]))
            gap = abs(bin_acc - bin_conf)

            accuracies[i] = bin_acc
            confidences[i] = bin_conf
            gaps[i] = gap

            weighted_ece += (count / n_samples) * gap
            if gap > max_ce:
                max_ce = gap
        else:
            accuracies[i] = 0.0
            confidences[i] = bin_centers[i]
            gaps[i] = 0.0

    brier = float(np.mean((p - y) ** 2))

    return {
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "accuracies": accuracies,
        "confidences": confidences,
        "counts": counts,
        "gaps": gaps,
        "ece": float(weighted_ece),
        "mce": float(max_ce),
        "brier_score": brier,
        "total_samples": n_samples,
    }


def compute_ece_and_mce(
    probs: Union[torch.Tensor, np.ndarray, List[float]],
    labels: Union[torch.Tensor, np.ndarray, List[float]],
    num_bins: int = 10,
) -> Tuple[float, float, float]:
    """
    Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and Brier Score.

    Returns:
        (ece, mce, brier_score)
    """
    res = compute_reliability_diagram_bins(probs, labels, num_bins=num_bins)
    return res["ece"], res["mce"], res["brier_score"]


def plot_reliability_diagram(
    bin_data: Dict[str, Any],
    title: str = "Reliability Diagram",
    task_name: str = "Creaky Voice",
    output_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """
    Render publication-grade two-panel Reliability Diagram.

    Upper panel: Empirical accuracy vs mean confidence with calibration error gaps.
    Lower panel: Predicted probability distribution histogram.
    """
    bin_centers = bin_data["bin_centers"]
    accuracies = bin_data["accuracies"]
    confidences = bin_data["confidences"]
    counts = bin_data["counts"]
    ece = bin_data["ece"]
    mce = bin_data["mce"]
    brier = bin_data["brier_score"]
    total = bin_data["total_samples"]

    bin_width = 1.0 / len(bin_centers)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax_main, ax_hist) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(6.0, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.2], "hspace": 0.12},
    )

    # ── Upper Panel: Calibration Plot ──────────────────────────────────────────
    # Ideal calibration diagonal line
    ax_main.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.5, label="Perfect Calibration ($y = x$)")

    # Error Gap bars (difference between confidence and accuracy)
    for i in range(len(bin_centers)):
        if counts[i] > 0:
            c = confidences[i]
            a = accuracies[i]
            # Red gap for overconfident (conf > acc), blue gap for underconfident (conf < acc)
            gap_color = "#e74c3c" if c > a else "#3498db"
            ax_main.bar(
                bin_centers[i],
                abs(c - a),
                bottom=min(c, a),
                width=bin_width * 0.85,
                color=gap_color,
                alpha=0.35,
                edgecolor=gap_color,
                linewidth=1.0,
                label="Calibration Gap" if i == 0 else "",
            )

    # Main empirical accuracy bars
    ax_main.bar(
        bin_centers,
        accuracies,
        width=bin_width * 0.85,
        color="#2b5c8f",
        alpha=0.85,
        edgecolor="#1a365d",
        linewidth=1.2,
        label="Observed Accuracy",
    )

    ax_main.set_xlim(0.0, 1.0)
    ax_main.set_ylim(0.0, 1.05)
    ax_main.set_ylabel("Empirical Accuracy", fontsize=11, fontweight="bold")
    ax_main.set_title(f"{title}\nTask: {task_name.capitalize()}", fontsize=12, fontweight="bold", pad=10)

    # Metrics annotation box
    metrics_str = f"ECE = {ece * 100:.2f}%\nMCE = {mce * 100:.2f}%\nBrier = {brier:.4f}\nN = {total:,}"
    ax_main.text(
        0.05,
        0.95,
        metrics_str,
        transform=ax_main.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )
    ax_main.legend(loc="lower right", frameon=True, fontsize=9)

    # ── Lower Panel: Sample Count Histogram ────────────────────────────────────
    ax_hist.bar(
        bin_centers,
        counts,
        width=bin_width * 0.85,
        color="#7f8c8d",
        alpha=0.8,
        edgecolor="#4b5563",
        linewidth=1.0,
    )
    ax_hist.set_xlim(0.0, 1.0)
    ax_hist.set_xlabel("Predicted Confidence Score ($\\hat{p}$)", fontsize=11, fontweight="bold")
    ax_hist.set_ylabel("Count", fontsize=10, fontweight="bold")
    ax_hist.grid(True, linestyle=":", alpha=0.6)

    # Save outputs if path provided
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Vector PDF
        pdf_path = p.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
        # High-res PNG
        png_path = p.with_suffix(".png")
        fig.savefig(png_path, bbox_inches="tight", dpi=300, format="png")
        print(f"[INFO] Saved Reliability Diagram to {pdf_path} and {png_path}")

    return fig


def plot_domain_shift_comparison(
    domain_results: Dict[str, Dict[str, Any]],
    task_name: str,
    output_path: Union[str, Path],
) -> plt.Figure:
    """
    Render multi-condition reliability comparison curve across clean, noisy, and reverberant domains.

    Args:
        domain_results: Dict mapping domain name (e.g., 'Clean', 'Cafe (0dB)', 'Car (0dB)', 'Reverb')
                        to its compute_reliability_diagram_bins output.
        task_name: Name of the task ('Creaky' or 'Breathy').
        output_path: Destination path to save figures (.pdf and .png).
    """
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    # Perfect calibration reference
    ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1.8, label="Ideal Calibration ($y = x$)")

    colors = ["#2ecc71", "#e67e22", "#e74c3c", "#9b59b6", "#34495e", "#1abc9c"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, (domain_name, data) in enumerate(domain_results.items()):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        bin_centers = data["bin_centers"]
        accuracies = data["accuracies"]
        counts = data["counts"]
        ece = data["ece"]

        # Only plot points for bins with samples
        valid_pts = counts > 0
        ax.plot(
            bin_centers[valid_pts],
            accuracies[valid_pts],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=6,
            label=f"{domain_name} (ECE: {ece * 100:.2f}%)",
        )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Confidence Score ($\\hat{p}$)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Observed Positive Rate", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Domain Shift Confidence Calibration\nTask: {task_name.capitalize()}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    ax.grid(True, linestyle=":", alpha=0.6)

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = p.with_suffix(".pdf")
    png_path = p.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300, format="png")
    print(f"[INFO] Saved Domain Shift Calibration Plot to {pdf_path} and {png_path}")

    return fig


def format_paradigm_a_summary_table(
    domain_results: Dict[str, Dict[str, Any]],
    task_name: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Format evaluation results into Paradigm A Factorial Matrix (3 Noise Types x 3 SNR Levels).

    Returns:
        (markdown_table_string, structured_rows_list)
    """
    scenarios = [
        ("cafe", "Cafe (Non-stationary Babble)"),
        ("car", "Car (Low-freq Rumble)"),
        ("street", "Street (Dynamic Traffic)"),
    ]
    snr_levels = [15.0, 5.0, 0.0]
    snr_headers = ["15 dB (Mild)", "5 dB (Moderate)", "0 dB (Severe)"]

    table_lines = [
        f"### Paradigm A: Factorial Evaluation Matrix (Task: `{task_name.upper()}`)",
        "| Noise Scenario | 15 dB (Mild) | 5 dB (Moderate) | 0 dB (Severe) | Scenario Mean ECE |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    structured_rows: List[Dict[str, Any]] = []
    col_sums = [0.0, 0.0, 0.0]
    col_counts = [0, 0, 0]

    for scen_key, scen_label in scenarios:
        row_cells = []
        row_vals = []
        for col_idx, snr in enumerate(snr_levels):
            cond_key = f"{scen_key}_{int(snr)}db"
            if cond_key in domain_results:
                ece_pct = domain_results[cond_key]["ece"] * 100.0
                row_cells.append(f"{ece_pct:.2f}%")
                row_vals.append(ece_pct)
                col_sums[col_idx] += ece_pct
                col_counts[col_idx] += 1
                structured_rows.append({
                    "task": task_name,
                    "condition": cond_key,
                    "scenario": scen_key,
                    "snr_db": snr,
                    "ece_pct": round(ece_pct, 2),
                    "mce_pct": round(domain_results[cond_key]["mce"] * 100.0, 2),
                    "brier_score": round(domain_results[cond_key]["brier_score"], 4),
                    "n_samples": domain_results[cond_key]["total_samples"],
                })
            else:
                row_cells.append("—")

        mean_val = (sum(row_vals) / len(row_vals)) if row_vals else 0.0
        table_lines.append(
            f"| **{scen_label}** | {row_cells[0]} | {row_cells[1]} | {row_cells[2]} | **{mean_val:.2f}%** |"
        )

    # Average Across Scenarios row
    avg_cells = []
    total_vals = []
    for col_idx in range(len(snr_levels)):
        if col_counts[col_idx] > 0:
            avg_col = col_sums[col_idx] / col_counts[col_idx]
            avg_cells.append(f"**{avg_col:.2f}%**")
            total_vals.append(avg_col)
        else:
            avg_cells.append("—")
    overall_mean = (sum(total_vals) / len(total_vals)) if total_vals else 0.0
    table_lines.append(
        f"| **Average Across Scenarios** | {avg_cells[0]} | {avg_cells[1]} | {avg_cells[2]} | 🏆 **{overall_mean:.2f}%** |"
    )

    # Spatial and Reference conditions
    table_lines.extend([
        "",
        "#### Reference & Spatial Controls",
    ])
    for ref_key, ref_label in [
        ("clean", "Clean (In-Distribution Reference)"),
        ("reverb", "Reverb Only (Spatial RIR)"),
        ("reverb_cafe_5db", "Composite (Reverb + Cafe 5dB)"),
    ]:
        if ref_key in domain_results:
            d = domain_results[ref_key]
            table_lines.append(
                f"- **{ref_label}**: ECE = **{d['ece'] * 100:.2f}%** | MCE = {d['mce'] * 100:.2f}% | Brier = {d['brier_score']:.4f} (N={d['total_samples']:,})"
            )
            structured_rows.append({
                "task": task_name,
                "condition": ref_key,
                "scenario": "reference",
                "snr_db": None,
                "ece_pct": round(d["ece"] * 100.0, 2),
                "mce_pct": round(d["mce"] * 100.0, 2),
                "brier_score": round(d["brier_score"], 4),
                "n_samples": d["total_samples"],
            })

    table_lines.append("")
    return "\n".join(table_lines), structured_rows


def plot_snr_decay_curves(
    domain_results: Dict[str, Dict[str, Any]],
    task_name: str,
    output_path: Union[str, Path],
) -> Optional[plt.Figure]:
    """
    Render publication-grade SNR degradation curves across Cafe, Car, and Street.
    X-axis: SNR in descending order (Clean/Inf -> 15 dB -> 5 dB -> 0 dB).
    Y-axis: ECE (%).
    """
    scenarios = [
        ("cafe", "Cafe (Babble)", "#e67e22", "s"),
        ("car", "Car (Rumble)", "#2980b9", "o"),
        ("street", "Street (Traffic)", "#8e44ad", "^"),
    ]
    snr_levels = [15.0, 5.0, 0.0]

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    # Reference clean baseline if available
    if "clean" in domain_results:
        clean_ece = domain_results["clean"]["ece"] * 100.0
        ax.axhline(
            clean_ece,
            color="#27ae60",
            linestyle="--",
            linewidth=1.8,
            label=f"Clean Baseline ({clean_ece:.2f}%)",
        )

    # Plot lines for each scenario
    for scen_key, scen_label, color, marker in scenarios:
        x_pts = []
        y_pts = []
        for snr in snr_levels:
            cond_key = f"{scen_key}_{int(snr)}db"
            if cond_key in domain_results:
                x_pts.append(snr)
                y_pts.append(domain_results[cond_key]["ece"] * 100.0)

        if len(x_pts) > 0:
            ax.plot(
                x_pts,
                y_pts,
                color=color,
                marker=marker,
                linewidth=2.0,
                markersize=7,
                label=scen_label,
            )

    # Reverb markers
    if "reverb" in domain_results:
        ax.scatter(
            [0.0],
            [domain_results["reverb"]["ece"] * 100.0],
            color="#16a085",
            marker="*",
            s=120,
            zorder=5,
            label=f"Reverb Only ({domain_results['reverb']['ece'] * 100.0:.2f}%)",
        )
    if "reverb_cafe_5db" in domain_results:
        ax.scatter(
            [5.0],
            [domain_results["reverb_cafe_5db"]["ece"] * 100.0],
            color="#c0392b",
            marker="D",
            s=80,
            zorder=5,
            label=f"Reverb + Cafe 5dB ({domain_results['reverb_cafe_5db']['ece'] * 100.0:.2f}%)",
        )

    ax.set_xlim(16.0, -1.0)  # Invert X axis so noise increases to the right
    ax.set_xlabel("Signal-to-Noise Ratio (SNR dB) $\\rightarrow$ More Severe", fontsize=11, fontweight="bold")
    ax.set_ylabel("Expected Calibration Error (ECE %)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Acoustic Degradation Calibration Robustness\nTask: {task_name.capitalize()}",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    ax.legend(loc="upper left", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = p.with_suffix(".pdf")
    png_path = p.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300, format="png")
    print(f"[INFO] Saved SNR Decay Plot to {pdf_path} and {png_path}")

    return fig

