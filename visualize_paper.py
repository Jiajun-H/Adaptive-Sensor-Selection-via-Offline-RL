"""
Paper-oriented figures from saved JSON results.

This script does not load checkpoints. It reads experiment JSON files and
produces compact figures suitable for a manuscript draft:
  1) Baseline metric bars
  2) Pareto trade-off scatter/lines
  3) Ablation bars
  4) Predictor backbone mean/std bars when compare_predictors_summary.json exists
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHOD_COLORS = {
    "LSTM+IQL": "#1f77b4",
    "LSTM+CQL": "#9467bd",
    "LSTM+BC": "#2ca02c",
    "LSTM+SoftAttn": "#17becf",
    "LSTM+StaticTopM": "#ff7f0e",
    "LSTM+Random": "#7f7f7f",
    "Static-All": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper figures from saved JSON results.")
    parser.add_argument("--results", type=Path, default=Path("checkpoints/experiment_results_full.json"))
    parser.add_argument("--predictor-summary", type=Path, default=Path("checkpoints/compare_predictors/compare_predictors_summary.json"))
    parser.add_argument("--output", type=Path, default=Path("figures/paper"))
    parser.add_argument("--dpi", type=int, default=320)
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _baseline_block(results: Dict[str, Any]) -> Dict[str, Any]:
    return results.get("baseline_results") or results.get("results") or {}


def _metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    if isinstance(value, dict):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _method_label(name: str) -> str:
    return name.replace("LSTM+", "")


def _polish(ax, grid_axis="y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.25)
    ax.set_axisbelow(True)


def plot_baseline_bars(results: Dict[str, Any], output: Path, dpi: int) -> Path | None:
    baseline = _baseline_block(results)
    if not baseline:
        return None

    methods = list(baseline.keys())
    labels = [_method_label(m) for m in methods]
    f1 = np.array([_metric(baseline[m], "f1") for m in methods])
    auc = np.array([_metric(baseline[m], "auc") for m in methods])
    recall = np.array([_metric(baseline[m], "recall") for m in methods])
    colors = [METHOD_COLORS.get(m, "#607d8b") for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharey=True)
    for ax, values, title in zip(axes, [f1, auc, recall], ["F1", "AUC", "Recall"]):
        ax.bar(np.arange(len(methods)), values, color=colors, width=0.68)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(methods)))
        ax.set_xticklabels(labels, rotation=28, ha="right")
        ax.set_ylim(0, min(1.02, max(0.1, float(values.max()) + 0.12)))
        _polish(ax)
        for idx, val in enumerate(values):
            ax.text(idx, val + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    axes[0].set_ylabel("Score")
    fig.tight_layout()
    path = output / "paper_baseline_metrics.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_pareto(results: Dict[str, Any], output: Path, dpi: int) -> Path | None:
    pareto = results.get("pareto_results") or results.get("pareto") or {}
    if not pareto:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for method, points in pareto.items():
        m_vals = sorted(int(m) for m in points.keys())
        f1 = [float(points[str(m)].get("f1", points.get(m, {}).get("f1", 0.0))) for m in m_vals]
        auc = [float(points[str(m)].get("auc", points.get(m, {}).get("auc", 0.0))) for m in m_vals]
        lead = [float(points[str(m)].get("avg_lead", points.get(m, {}).get("avg_lead", 0.0))) for m in m_vals]
        color = METHOD_COLORS.get(f"LSTM+{method}", METHOD_COLORS.get(method, "#607d8b"))
        axes[0].plot(m_vals, f1, marker="o", lw=1.8, label=method, color=color)
        axes[1].scatter(f1, lead, s=np.maximum(35, 18 + 2.5 * np.asarray(m_vals)), label=method, color=color, alpha=0.9)
        for x, y, m in zip(f1, lead, m_vals):
            axes[1].annotate(str(m), (x, y), textcoords="offset points", xytext=(4, 3), fontsize=7)

    axes[0].set_xlabel("Active sensors M")
    axes[0].set_ylabel("F1")
    axes[0].set_title("Resource-performance curve")
    _polish(axes[0])
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].set_xlabel("F1")
    axes[1].set_ylabel("Average lead time (s)")
    axes[1].set_title("Accuracy-timeliness trade-off")
    _polish(axes[1], grid_axis="both")
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    path = output / "paper_pareto_tradeoff.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation(results: Dict[str, Any], output: Path, dpi: int) -> Path | None:
    ablation = results.get("ablation_results") or {}
    if not ablation:
        return None

    names = list(ablation.keys())
    f1 = np.array([_metric(ablation[n], "f1") for n in names])
    lead = np.array([_metric(ablation[n], "avg_lead_time") for n in names])

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    x = np.arange(len(names))
    axes[0].bar(x, f1, color="#1f77b4")
    axes[0].set_ylabel("F1")
    axes[0].set_title("Component ablation")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")
    _polish(axes[0])

    axes[1].bar(x, lead, color="#ff9f1c")
    axes[1].set_ylabel("Average lead time (s)")
    axes[1].set_title("Timeliness under ablation")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")
    _polish(axes[1])

    fig.tight_layout()
    path = output / "paper_ablation.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_predictor_summary(summary: Dict[str, Any], output: Path, dpi: int) -> Path | None:
    if not summary:
        return None
    names = sorted(summary.keys())
    f1_mean = np.array([float(summary[n].get("f1_mean", 0.0)) for n in names])
    f1_std = np.array([float(summary[n].get("f1_std", 0.0)) for n in names])
    auc_mean = np.array([float(summary[n].get("auc_mean", 0.0)) for n in names])
    auc_std = np.array([float(summary[n].get("auc_std", 0.0)) for n in names])
    tte_mean = np.array([float(summary[n].get("tte_mae_mean", 0.0)) for n in names])
    tte_std = np.array([float(summary[n].get("tte_mae_std", 0.0)) for n in names])

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
    x = np.arange(len(names))
    for ax, mean, std, title, ylabel in [
        (axes[0], f1_mean, f1_std, "Predictor F1", "F1"),
        (axes[1], auc_mean, auc_std, "Predictor AUC", "AUC"),
        (axes[2], tte_mean, tte_std, "TTE error", "MAE"),
    ]:
        ax.bar(x, mean, yerr=std, capsize=4, color="#4c78a8")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        _polish(ax)

    fig.tight_layout()
    path = output / "paper_predictor_backbones.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = load_json(args.results)
    predictor_summary = load_json(args.predictor_summary)

    paths = [
        plot_baseline_bars(results, args.output, args.dpi),
        plot_pareto(results, args.output, args.dpi),
        plot_ablation(results, args.output, args.dpi),
        plot_predictor_summary(predictor_summary, args.output, args.dpi),
    ]
    for path in paths:
        if path is not None:
            print(f"[paper-fig] saved {path}")


if __name__ == "__main__":
    main()
