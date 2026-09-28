"""Chapter 5 visualization command-line entry point.

Data loading and figure rendering live in visualize_ch5_data.py and
visualize_ch5_plots.py, respectively.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from visualize_ch5_data import FAMILY_ORDER, ExperimentData, load_excel_experiment
from visualize_ch5_plots import (
    load_metrics, plot_ablation_effects, plot_experiment_profile,
    plot_metric_tradeoffs, plot_pareto_f1, plot_sensor_family_profile,
    plot_signal_timeline, plot_top_sensor_heatmap,
)


def scan_experiment_dir(args: argparse.Namespace) -> list[ExperimentData]:
    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists() or not exp_dir.is_dir():
        return []
    files = sorted(p for p in exp_dir.glob("*.xlsx") if not p.name.startswith("~$"))
    if args.max_experiments > 0:
        files = files[: args.max_experiments]
    experiments: list[ExperimentData] = []
    for path in files:
        try:
            experiments.append(
                load_excel_experiment(
                    path,
                    broken_val=args.broken_val,
                    broken_ratio=args.broken_ratio,
                    auto_reverse=args.auto_reverse,
                    delta=args.delta,
                    fire_onset_override=None,
                    temp_threshold=args.temp_threshold,
                    trans_threshold=args.trans_threshold,
                    heat_threshold=args.heat_threshold,
                    co_threshold=args.co_threshold,
                    min_modalities=args.min_modalities,
                    min_consecutive=args.min_consecutive,
                )
            )
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
    return experiments

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Chapter 5 thesis visualizations without checkpoints.")
    parser.add_argument("--data", default="data.xlsx", help="Main raw Excel file used for signal visualizations.")
    parser.add_argument("--metrics", default="paper_metrics_ch5.json", help="Metric JSON for comparison figures.")
    parser.add_argument("--output", default=os.path.join("figures", "ch5"), help="Output directory.")
    parser.add_argument("--scan-experiments", action="store_true", help="Also scan converted_excel for dataset profile.")
    parser.add_argument("--experiment-dir", default="converted_excel", help="Directory of converted experiment Excel files.")
    parser.add_argument("--max-experiments", type=int, default=0, help="Limit scanned experiment files; 0 means all.")
    parser.add_argument("--fire-onset", type=int, default=None, help="Override main data fire onset if needed.")
    parser.add_argument("--delta", type=int, default=30, help="Warning horizon in seconds.")
    parser.add_argument("--window-size", type=int, default=60, help="Model window size for notes/summary.")
    parser.add_argument("--temp-threshold", type=float, default=200.0)
    parser.add_argument("--trans-threshold", type=float, default=0.3)
    parser.add_argument("--heat-threshold", type=float, default=2.5)
    parser.add_argument("--co-threshold", type=float, default=100.0)
    parser.add_argument("--min-modalities", type=int, default=2)
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--broken-val", type=float, default=-200.0)
    parser.add_argument("--broken-ratio", type=float, default=0.5)
    parser.add_argument("--no-auto-reverse", dest="auto_reverse", action="store_false")
    parser.set_defaults(auto_reverse=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--timeline-pre", type=int, default=160)
    parser.add_argument("--timeline-post", type=int, default=320)
    parser.add_argument("--heatmap-pre", type=int, default=150)
    parser.add_argument("--heatmap-post", type=int, default=220)
    parser.add_argument("--dpi", type=int, default=320)
    return parser

def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Main data file not found: {data_path}")

    metrics_path = Path(args.metrics) if args.metrics else None
    metrics = load_metrics(metrics_path)
    if metrics:
        print(f"[metrics] loaded {metrics_path}")
    else:
        print("[metrics] no metric JSON loaded; metric plots will be skipped")

    fire_onset_override = args.fire_onset
    if fire_onset_override is None and metrics:
        config = metrics.get("config", {})
        if isinstance(config, dict) and config.get("fire_onset") is not None:
            fire_onset_override = int(config["fire_onset"])

    exp = load_excel_experiment(
        data_path,
        broken_val=args.broken_val,
        broken_ratio=args.broken_ratio,
        auto_reverse=args.auto_reverse,
        delta=args.delta,
        fire_onset_override=fire_onset_override,
        temp_threshold=args.temp_threshold,
        trans_threshold=args.trans_threshold,
        heat_threshold=args.heat_threshold,
        co_threshold=args.co_threshold,
        min_modalities=args.min_modalities,
        min_consecutive=args.min_consecutive,
    )
    print(
        f"[data] {data_path} -> T={len(exp.data)}, N={exp.data.shape[1]}, "
        f"fire_onset={exp.fire_onset}, warning_onset={exp.warning_onset}"
    )

    figures: list[str] = []
    path = plot_signal_timeline(exp, output_dir, args)
    figures.append(str(path))
    path, top_sensors = plot_top_sensor_heatmap(exp, output_dir, args)
    figures.append(str(path))
    path = plot_sensor_family_profile(exp, output_dir, args)
    figures.append(str(path))

    for plotter in [plot_metric_tradeoffs, plot_ablation_effects, plot_pareto_f1]:
        maybe_path = plotter(metrics, output_dir, args)
        if maybe_path is not None:
            figures.append(str(maybe_path))

    scanned: list[ExperimentData] = []
    if args.scan_experiments:
        scanned = scan_experiment_dir(args)
        maybe_path = plot_experiment_profile(scanned, output_dir, args)
        if maybe_path is not None:
            figures.append(str(maybe_path))
        print(f"[scan] loaded {len(scanned)} experiment files")

    family_counts = {family: exp.sensor_types.count(family) for family in FAMILY_ORDER}
    summary = {
        "main_data": {
            "path": str(data_path),
            "time_steps": int(len(exp.data)),
            "n_sensors": int(exp.data.shape[1]),
            "fire_onset": int(exp.fire_onset),
            "warning_onset": int(exp.warning_onset),
            "window_size": int(args.window_size),
            "delta": int(args.delta),
            "family_counts": family_counts,
        },
        "metrics_source": str(metrics_path) if metrics else "",
        "top_early_sensors": top_sensors[:15],
        "scanned_experiments": {
            "enabled": bool(args.scan_experiments),
            "count": int(len(scanned)),
        },
        "figures": figures,
    }
    summary_path = output_dir / "ch5_visual_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[summary] {summary_path}")
    for fig_path in figures:
        print(f"[figure] {fig_path}")

if __name__ == "__main__":
    main()
