"""Plot styling and figure builders for the Chapter 5 visualizer."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colors as mcolors

from visualize_ch5_data import (
    FAMILY_ORDER, FAMILY_DIRECTION, ExperimentData, family_series, group_indices, robust_z,
)

FAMILY_COLORS = {
    "temperature": "#D84A3A",
    "co": "#7E57C2",
    "transmittance": "#2F80ED",
    "heat_flux": "#F2994A",
    "smoke_layer": "#27AE60",
    "unknown": "#8A8F98",
}
METHOD_COLORS = {
    "LSTM+IQL": "#1F77B4",
    "LSTM+CQL": "#7B61FF",
    "LSTM+BC": "#2CA02C",
    "LSTM+SoftAttn": "#17BECF",
    "LSTM+StaticTopM": "#FF9F1C",
    "LSTM+Random": "#8C8C8C",
}
HIGHLIGHT_METHOD = "LSTM+IQL"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans", "sans-serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.dpi": 320,
    }
)


def short_label(text: str, max_len: int = 28) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "."

def method_label(name: str) -> str:
    return name.replace("LSTM+", "")

def method_color(name: str) -> str:
    if name in METHOD_COLORS:
        return METHOD_COLORS[name]
    prefixed = f"LSTM+{name}"
    return METHOD_COLORS.get(prefixed, "#6B7280")

def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )

def polish_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["left"].set_color("#2F3437")
    ax.spines["bottom"].set_color("#2F3437")
    ax.tick_params(colors="#2F3437", length=3.0, width=0.7)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#D7DEE8", lw=0.7, alpha=0.65)
        ax.set_axisbelow(True)

def add_bar_labels(ax: plt.Axes, bars, fmt: str = "{:.3f}", padding: float = 0.01) -> None:
    y0, y1 = ax.get_ylim()
    offset = (y1 - y0) * padding
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=7,
            color="#2F3437",
        )

def save_ch5_figure(fig: plt.Figure, path: Path, args: argparse.Namespace) -> None:
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight", facecolor="white")

def metric_value(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, dict):
                continue
            return float(value)
    lead = metrics.get("lead_time")
    if isinstance(lead, dict):
        for key in keys:
            if key in lead:
                return float(lead[key])
    miss = metrics.get("event_miss_rate")
    if isinstance(miss, dict):
        for key in keys:
            if key in miss:
                return float(miss[key])
    return default

def load_metrics(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def plot_signal_timeline(exp: ExperimentData, output_dir: Path, args: argparse.Namespace) -> Path:
    groups = group_indices(exp.sensor_types)
    start = max(0, exp.fire_onset - args.timeline_pre)
    end = min(len(exp.data), exp.fire_onset + args.timeline_post) if exp.fire_onset < len(exp.data) else len(exp.data)
    if end <= start:
        start, end = 0, len(exp.data)
    x = np.arange(start, end)
    x_rel = x - exp.fire_onset

    fig, axes = plt.subplots(2, 1, figsize=(11.6, 6.4), sharex=True, gridspec_kw={"height_ratios": [3.2, 1]})
    ax = axes[0]
    plotted = 0
    for family in FAMILY_ORDER:
        indices = groups.get(family, [])
        if not indices or family == "unknown":
            continue
        series = family_series(exp.data, indices, family)
        if len(series) == 0:
            continue
        directed = series * FAMILY_DIRECTION.get(family, 1.0)
        y = np.clip(robust_z(directed), -3.0, 5.0)
        ax.plot(x_rel, y[start:end], lw=2.0, color=FAMILY_COLORS[family], label=family.replace("_", " "))
        plotted += 1

    if exp.warning_onset < len(exp.data):
        ax.axvspan(exp.warning_onset - exp.fire_onset, 0, color="#F2C94C", alpha=0.22, label="30 s warning horizon")
        ax.text((exp.warning_onset - exp.fire_onset) / 2, 4.55, "warning horizon", ha="center", va="top", fontsize=8, color="#7A4F00")
    if exp.fire_onset < len(exp.data):
        ax.axvline(0, color="#B3261E", lw=1.6, ls="--", label="fire onset")
        ax.axvspan(0, end - exp.fire_onset, color="#EB5757", alpha=0.07)
    ax.axhline(0.0, color="#6B7280", lw=0.8, alpha=0.6)
    ax.set_ylabel("Directed robust signal (z)")
    ax.set_title("Early sensor response before the hazard decision point")
    polish_axes(ax, grid_axis="y")
    if plotted:
        ax.legend(ncol=3, fontsize=8, loc="upper left", handlelength=2.2)
    panel_label(ax, "a")

    axes[1].plot(x_rel, exp.modality_hits[start:end], color="#111827", lw=1.8)
    axes[1].axhline(args.min_modalities, color="#B3261E", lw=1.0, ls="--", label="hazard rule")
    if exp.warning_onset < len(exp.data):
        axes[1].axvspan(exp.warning_onset - exp.fire_onset, 0, color="#F2C94C", alpha=0.22)
    if exp.fire_onset < len(exp.data):
        axes[1].axvline(0, color="#B3261E", lw=1.4, ls="--")
    axes[1].set_ylabel("Active cues")
    axes[1].set_xlabel("Time relative to fire onset (s)")
    axes[1].set_ylim(-0.1, max(args.min_modalities + 1, int(np.max(exp.modality_hits[start:end]) + 1)))
    polish_axes(axes[1], grid_axis="y")
    axes[1].legend(fontsize=8, loc="upper left")
    panel_label(axes[1], "b")

    fig.tight_layout()
    path = output_dir / "ch5_signal_timeline.png"
    save_ch5_figure(fig, path, args)
    plt.close(fig)
    return path

def sensor_early_scores(exp: ExperimentData, delta: int) -> list[dict[str, Any]]:
    onset = exp.fire_onset
    if onset >= len(exp.data):
        onset = len(exp.data) // 2
    baseline_start = max(0, onset - 4 * delta)
    baseline_end = max(baseline_start + 1, onset - 2 * delta)
    warning_start = max(0, onset - delta)
    warning_end = max(warning_start + 1, onset)
    danger_end = min(len(exp.data), onset + 2 * delta)
    if danger_end <= onset:
        danger_end = min(len(exp.data), onset + 1)

    scores: list[dict[str, Any]] = []
    for idx in range(exp.data.shape[1]):
        sensor_type = exp.sensor_types[idx]
        direction = FAMILY_DIRECTION.get(sensor_type, 1.0)
        series = exp.data[:, idx].astype(np.float32) * direction
        base = series[baseline_start:baseline_end]
        warn = series[warning_start:warning_end]
        danger = series[onset:danger_end]
        valid = base[np.isfinite(base)]
        if len(valid) < 3:
            continue
        q25, q75 = np.quantile(valid, [0.25, 0.75])
        scale = float(q75 - q25)
        if scale < 1e-8:
            scale = float(np.std(valid))
        if scale < 1e-8:
            scale = 1.0
        warn_gain = (float(np.nanmedian(warn)) - float(np.nanmedian(base))) / scale
        danger_gain = (float(np.nanmedian(danger)) - float(np.nanmedian(base))) / scale
        score = max(0.0, 0.65 * warn_gain + 0.35 * danger_gain)
        scores.append(
            {
                "index": idx,
                "name": exp.names[idx],
                "family": sensor_type,
                "score": float(score),
                "warning_gain": float(warn_gain),
                "danger_gain": float(danger_gain),
            }
        )
    scores.sort(key=lambda item: item["score"], reverse=True)
    return scores

def plot_top_sensor_heatmap(exp: ExperimentData, output_dir: Path, args: argparse.Namespace) -> tuple[Path, list[dict[str, Any]]]:
    scores = sensor_early_scores(exp, args.delta)
    top = scores[: min(args.top_k, len(scores))]
    if not top:
        raise ValueError("Cannot compute top sensor scores.")
    indices = [int(item["index"]) for item in top]
    start = max(0, exp.fire_onset - args.heatmap_pre)
    end = min(len(exp.data), exp.fire_onset + args.heatmap_post) if exp.fire_onset < len(exp.data) else len(exp.data)
    if end <= start:
        start, end = 0, len(exp.data)

    rows = []
    labels = []
    for item in top:
        idx = int(item["index"])
        family = str(item["family"])
        direction = FAMILY_DIRECTION.get(family, 1.0)
        row = robust_z(exp.data[:, idx] * direction)
        rows.append(np.clip(row[start:end], -2.5, 5.0))
        labels.append(f"S{idx} {family.replace('_', ' ')}")
    mat = np.vstack(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, max(5.8, 0.25 * len(top))), gridspec_kw={"width_ratios": [2.5, 1.05]})
    ax = axes[0]
    norm = mcolors.TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=5.0)
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="RdBu_r", norm=norm)
    if exp.warning_onset < len(exp.data):
        ax.axvspan(exp.warning_onset - start, exp.fire_onset - start, color="#F2C94C", alpha=0.24)
    if exp.fire_onset < len(exp.data):
        ax.axvline(exp.fire_onset - start, color="#111827", lw=1.4, ls="--")
    ax.set_title("Sensor-level response ranked by pre-fire change")
    ax.set_xlabel("Time relative to fire onset (s)")
    ax.set_ylabel("Sensor")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    tick_pos = np.linspace(0, mat.shape[1] - 1, min(7, mat.shape[1])).astype(int)
    ax.set_xticks(tick_pos)
    ax.set_xticklabels([str(int(start + pos - exp.fire_onset)) for pos in tick_pos])
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("Directed robust signal (z)")
    panel_label(ax, "a")

    bar_colors = [FAMILY_COLORS.get(str(item["family"]), "#8A8F98") for item in top]
    y = np.arange(len(top))
    scores_for_plot = [float(item["score"]) for item in top]
    axes[1].barh(y, scores_for_plot, color=bar_colors, alpha=0.92)
    axes[1].invert_yaxis()
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(
        [f"S{int(item['index'])} {str(item['family']).replace('_', ' ')}" for item in top],
        fontsize=7,
    )
    axes[1].set_xlabel("Early signal score")
    axes[1].set_title("Score used for dynamic selection")
    polish_axes(axes[1], grid_axis="x")
    for yy, value in zip(y, scores_for_plot):
        axes[1].text(value + max(scores_for_plot) * 0.015, yy, f"{value:.1f}", va="center", fontsize=6.5, color="#2F3437")
    axes[1].set_xlim(0, max(scores_for_plot) * 1.18)
    panel_label(axes[1], "b")

    fig.tight_layout()
    path = output_dir / "ch5_top_sensor_heatmap.png"
    save_ch5_figure(fig, path, args)
    plt.close(fig)
    return path, top

def plot_sensor_family_profile(exp: ExperimentData, output_dir: Path, args: argparse.Namespace) -> Path:
    counts = {family: exp.sensor_types.count(family) for family in FAMILY_ORDER}
    scores = sensor_early_scores(exp, args.delta)
    family_scores: dict[str, list[float]] = {family: [] for family in FAMILY_ORDER}
    for item in scores:
        family_scores.setdefault(str(item["family"]), []).append(float(item["score"]))
    families = [family for family in FAMILY_ORDER if counts.get(family, 0) > 0]
    count_vals = [counts[family] for family in families]
    median_scores = [float(np.median(family_scores.get(family, [0.0])[:20])) if family_scores.get(family) else 0.0 for family in families]

    family_labels = [f.replace("_", " ") for f in families]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), gridspec_kw={"width_ratios": [1.05, 1.15]})
    bars0 = axes[0].barh(family_labels, count_vals, color=[FAMILY_COLORS.get(f, "#8A8F98") for f in families])
    axes[0].invert_yaxis()
    axes[0].set_title("Available sensing channels")
    axes[0].set_xlabel("Channels")
    polish_axes(axes[0], grid_axis="x")
    total_channels = max(sum(count_vals), 1)
    for bar, count in zip(bars0, count_vals):
        axes[0].text(count + total_channels * 0.012, bar.get_y() + bar.get_height() / 2, f"{count} ({count / total_channels:.0%})", va="center", fontsize=8)
    axes[0].set_xlim(0, max(count_vals) * 1.2)
    panel_label(axes[0], "a")

    bars1 = axes[1].barh(family_labels, median_scores, color=[FAMILY_COLORS.get(f, "#8A8F98") for f in families], alpha=0.92)
    axes[1].invert_yaxis()
    axes[1].set_title("Median early-change score")
    axes[1].set_xlabel("Score among candidate sensors")
    polish_axes(axes[1], grid_axis="x")
    max_score = max(max(median_scores), 1.0)
    for bar, score in zip(bars1, median_scores):
        axes[1].text(score + max_score * 0.02, bar.get_y() + bar.get_height() / 2, f"{score:.1f}", va="center", fontsize=8)
    axes[1].set_xlim(0, max_score * 1.25)
    panel_label(axes[1], "b")
    fig.tight_layout()
    path = output_dir / "ch5_sensor_family_profile.png"
    save_ch5_figure(fig, path, args)
    plt.close(fig)
    return path

def plot_metric_tradeoffs(metrics: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> Path | None:
    baseline = metrics.get("baseline_results", {})
    if not baseline:
        return None
    methods = list(baseline.keys())
    f1 = np.array([metric_value(baseline[m], "f1") for m in methods], dtype=float)
    auc = np.array([metric_value(baseline[m], "auc") for m in methods], dtype=float)
    lead = np.array([metric_value(baseline[m], "avg_lead_time", "avg_lead") for m in methods], dtype=float)
    switch = np.array([metric_value(baseline[m], "switching_freq") for m in methods], dtype=float)
    median_lead = np.array([metric_value(baseline[m], "median_lead_time", "median_lead") for m in methods], dtype=float)
    lead_ge_30 = np.array([metric_value(baseline[m], "lead_ge_30s_ratio") for m in methods], dtype=float)
    labels = [method_label(m) for m in methods]
    colors = [method_color(m) for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.9), gridspec_kw={"width_ratios": [1.3, 1.25, 1.15]})
    x = np.arange(len(methods))
    width = 0.38
    bars_f1 = axes[0].bar(x - width / 2, f1, width=width, label="F1", color=colors, alpha=0.92)
    bars_auc = axes[0].bar(x + width / 2, auc, width=width, label="AUC", color="#D0D7E2", edgecolor=colors, linewidth=1.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].set_ylim(max(0.84, float(np.min([f1.min(), auc.min()])) - 0.02), 1.01)
    axes[0].set_title("Warning accuracy")
    polish_axes(axes[0], grid_axis="y")
    axes[0].legend(fontsize=8)
    add_bar_labels(axes[0], bars_f1, "{:.3f}", padding=0.015)
    panel_label(axes[0], "a")

    switch_for_color = np.maximum(switch, 5e-4)
    scatter = axes[1].scatter(
        lead,
        f1,
        c=switch_for_color,
        s=110 + 620 * np.clip(lead_ge_30, 0, 1),
        cmap="viridis_r",
        norm=mcolors.LogNorm(vmin=5e-4, vmax=max(float(switch_for_color.max()), 5e-4)),
        edgecolor=["#B3261E" if m == HIGHLIGHT_METHOD else "white" for m in methods],
        linewidth=[1.6 if m == HIGHLIGHT_METHOD else 0.8 for m in methods],
    )
    offsets = {
        "IQL": (7, 7),
        "CQL": (7, -12),
        "BC": (7, 6),
        "SoftAttn": (7, -12),
        "StaticTopM": (-58, -3),
        "Random": (7, 7),
    }
    for label, xx, yy in zip(labels, lead, f1):
        axes[1].annotate(label, (xx, yy), textcoords="offset points", xytext=offsets.get(label, (6, 6)), fontsize=8)
    axes[1].set_xlabel("Average lead time (s)")
    axes[1].set_ylabel("F1")
    axes[1].set_title("Accuracy, timeliness and switching")
    axes[1].set_xlim(max(0, float(lead.min()) - 5), float(lead.max()) + 12)
    axes[1].set_ylim(float(f1.min()) - 0.02, min(1.0, float(f1.max()) + 0.012))
    polish_axes(axes[1], grid_axis="both")
    cbar = fig.colorbar(scatter, ax=axes[1], fraction=0.045, pad=0.02)
    cbar.set_label("Switching frequency")
    panel_label(axes[1], "b")

    x2 = np.arange(len(methods))
    bars_avg = axes[2].bar(x2 - width / 2, lead, width=width, color="#F2A65A", label="Average")
    bars_med = axes[2].bar(x2 + width / 2, median_lead, width=width, color="#8E6BBE", label="Median")
    axes[2].set_xticks(x2)
    axes[2].set_xticklabels(labels, rotation=25, ha="right")
    axes[2].set_ylabel("Lead time (s)")
    axes[2].set_title("Lead-time distribution summary")
    polish_axes(axes[2], grid_axis="y")
    axes[2].legend(fontsize=8)
    add_bar_labels(axes[2], bars_avg, "{:.1f}", padding=0.015)
    panel_label(axes[2], "c")

    fig.tight_layout()
    path = output_dir / "ch5_metric_tradeoffs.png"
    save_ch5_figure(fig, path, args)
    plt.close(fig)
    return path

def plot_ablation_effects(metrics: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> Path | None:
    ablation = metrics.get("ablation_results", {})
    if not ablation or "Full" not in ablation:
        return None
    configs = list(ablation.keys())
    labels = [config.replace("w/o ", "-") for config in configs]
    f1_vals = np.array([metric_value(ablation[c], "f1") for c in configs], dtype=float)
    auc_vals = np.array([metric_value(ablation[c], "auc") for c in configs], dtype=float)
    lead_vals = np.array([metric_value(ablation[c], "avg_lead_time", "avg_lead") for c in configs], dtype=float)
    switch_vals = np.array([metric_value(ablation[c], "switching_freq") for c in configs], dtype=float)
    colors = ["#1F77B4" if c == "Full" else "#AEB7C5" for c in configs]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.3))
    x = np.arange(len(configs))
    width = 0.38
    bars_f1 = axes[0].bar(x - width / 2, f1_vals, width=width, color=colors, label="F1")
    axes[0].bar(x + width / 2, auc_vals, width=width, color="#E1E6EF", edgecolor=colors, linewidth=1.0, label="AUC")
    axes[0].set_title("Accuracy after removing modules")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylim(max(0.88, float(min(f1_vals.min(), auc_vals.min())) - 0.02), 1.01)
    polish_axes(axes[0], grid_axis="y")
    axes[0].legend(fontsize=8)
    add_bar_labels(axes[0], bars_f1, "{:.3f}", padding=0.015)
    panel_label(axes[0], "a")

    bars_lead = axes[1].bar(x, lead_vals, color=colors)
    axes[1].set_title("Average lead time")
    axes[1].set_ylabel("Seconds")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    polish_axes(axes[1], grid_axis="y")
    add_bar_labels(axes[1], bars_lead, "{:.1f}", padding=0.015)
    panel_label(axes[1], "b")

    bars_switch = axes[2].bar(x, switch_vals, color=colors)
    axes[2].set_title("Sensor switching frequency")
    axes[2].set_ylabel("Switches per step")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=20, ha="right")
    polish_axes(axes[2], grid_axis="y")
    add_bar_labels(axes[2], bars_switch, "{:.3f}", padding=0.04)
    panel_label(axes[2], "c")

    fig.tight_layout()
    path = output_dir / "ch5_ablation_effects.png"
    save_ch5_figure(fig, path, args)
    plt.close(fig)
    return path

def plot_pareto_f1(metrics: dict[str, Any], output_dir: Path, args: argparse.Namespace) -> Path | None:
    pareto = metrics.get("pareto_results", {})
    if not pareto:
        return None
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    markers = ["o", "s", "^", "D", "v", "P", "X"]
    budget_values = sorted({int(k) for results in pareto.values() for k in results.keys()})
    for idx, (method, results) in enumerate(pareto.items()):
        xs = sorted(int(k) for k in results.keys())
        ys = [float(results[str(x)].get("f1", np.nan)) for x in xs]
        is_iql = method_label(method) == method_label(HIGHLIGHT_METHOD)
        color = method_color(method)
        ax.plot(
            xs,
            ys,
            marker=markers[idx % len(markers)],
            lw=2.7 if is_iql else 1.5,
            ms=7.0 if is_iql else 5.2,
            color=color,
            alpha=1.0 if is_iql else 0.86,
            label=method_label(method),
            zorder=3 if is_iql else 2,
        )
    ax.axhspan(0.97, 1.0, color="#1F77B4", alpha=0.07, lw=0)
    ax.axvline(20, color="#B3261E", lw=1.0, ls="--", alpha=0.85)
    ax.text(20.3, 0.887, "paper setting: M=20", rotation=90, va="bottom", ha="left", fontsize=8, color="#7A1E18")
    ax.set_xlabel("Active sensors M")
    ax.set_ylabel("F1")
    ax.set_title("Budget-performance Pareto curve")
    ax.set_ylim(0.88, 1.0)
    tick_values = list(range(5, max(budget_values) + 1, 5))
    ax.set_xlim(min(tick_values) - 1, max(tick_values) + 1)
    ax.set_xticks(tick_values)
    polish_axes(ax, grid_axis="both")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = output_dir / "ch5_pareto_f1.png"
    save_ch5_figure(fig, path, args)
    plt.close(fig)
    return path

def plot_experiment_profile(experiments: list[ExperimentData], output_dir: Path, args: argparse.Namespace) -> Path | None:
    if not experiments:
        return None
    lengths = np.array([len(exp.data) for exp in experiments], dtype=float)
    sensors = np.array([exp.data.shape[1] for exp in experiments], dtype=float)
    onset_ratio = np.array(
        [exp.fire_onset / max(len(exp.data), 1) if exp.fire_onset < len(exp.data) else np.nan for exp in experiments],
        dtype=float,
    )
    family_counts = {family: [] for family in FAMILY_ORDER}
    for exp in experiments:
        for family in FAMILY_ORDER:
            family_counts[family].append(exp.sensor_types.count(family))
    families = [family for family in FAMILY_ORDER if np.sum(family_counts[family]) > 0]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    axes[0].hist(lengths, bins=min(12, max(4, int(math.sqrt(len(lengths))))), color="#2F80ED", alpha=0.85)
    axes[0].set_title("Experiment lengths")
    axes[0].set_xlabel("Time steps")
    axes[0].set_ylabel("Files")
    polish_axes(axes[0], grid_axis="y")

    axes[1].scatter(lengths, sensors, c=np.nan_to_num(onset_ratio, nan=0.0), cmap="plasma", edgecolor="white", s=70)
    axes[1].set_title("Length vs available channels")
    axes[1].set_xlabel("Time steps")
    axes[1].set_ylabel("Valid sensors")
    polish_axes(axes[1], grid_axis="both")

    axes[2].bar(families, [float(np.median(family_counts[f])) for f in families], color=[FAMILY_COLORS.get(f, "#8A8F98") for f in families])
    axes[2].set_title("Median family count")
    axes[2].set_ylabel("Channels")
    axes[2].tick_params(axis="x", rotation=25)
    polish_axes(axes[2], grid_axis="y")

    fig.tight_layout()
    path = output_dir / "ch5_experiment_profile.png"
    save_ch5_figure(fig, path, args)
    plt.close(fig)
    return path
