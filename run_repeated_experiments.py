"""
Repeated main experiments for publication-grade statistics.

Modes:
  random: repeat experiment-level random splits over multiple seeds.
  kfold : run experiment-level K-fold; each fold holds out a different test fold
          and uses the next fold as validation.

This script calls run_experiments() directly, so it reuses the exact baseline,
offline-RL, evaluation, and plotting logic from the main experiment driver while
writing each repeat to an isolated checkpoint/figure directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple


def parse_seed_list(text: str) -> List[int]:
    seeds: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        seeds.append(int(item))
    return seeds or [42]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated main experiments and aggregate statistics.")
    parser.add_argument("--mode", choices=["random", "kfold"], default="random")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds for --mode random.")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds for --mode kfold.")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_max_files", type=int, default=12)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None, help="Override predictor train batch size.")
    parser.add_argument("--data_num_workers", type=int, default=None, help="Override predictor DataLoader workers.")
    parser.add_argument("--iql_batch_size", type=int, default=None, help="Override IQL/BC batch size.")
    parser.add_argument("--cql_batch_size", type=int, default=None, help="Override CQL batch size.")
    parser.add_argument("--cql_max_negative_eval_batch", type=int, default=None, help="Max flattened random-action rows per CQL Q forward chunk.")
    parser.add_argument("--rl_num_workers", type=int, default=None, help="Override offline RL DataLoader workers when not preloaded.")
    parser.add_argument("--no_rl_preload", dest="rl_preload_to_device", action="store_false", help="Disable CUDA preloading for offline RL tensors.")
    parser.set_defaults(rl_preload_to_device=None)
    parser.add_argument("--offline_build_batch_steps", type=int, default=None, help="Decision steps grouped per offline-dataset predictor batch.")
    parser.add_argument("--offline_encode_batch_size", type=int, default=None, help="Chunk size for predictor encoding during offline dataset build.")
    parser.add_argument("--soft_attn_batch_size", type=int, default=None, help="Override SoftAttention train batch size.")
    parser.add_argument("--soft_attn_embed_batch_size", type=int, default=None, help="Override SoftAttention full-belief precompute batch size.")
    parser.add_argument("--no_soft_attn_preload", dest="soft_attn_preload_to_device", action="store_false", help="Disable CUDA preloading for SoftAttention tensors.")
    parser.set_defaults(soft_attn_preload_to_device=None)
    parser.add_argument("--policy_eval_batch_size", type=int, default=None, help="Batch size for stateless policy evaluation.")
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints/repeated"))
    parser.add_argument("--fig-root", type=Path, default=Path("figures/repeated"))
    return parser.parse_args()


def _metric(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


BASELINE_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "auc",
    "decision_threshold",
    "event_miss_rate",
    "avg_lead_time",
    "median_lead_time",
    "lead_ge_30s_ratio",
    "switching_freq",
]

PARETO_METRICS = [
    "f1",
    "auc",
    "avg_lead_time",
    "event_miss_rate",
    "switching_freq",
    "decision_threshold",
]


def compact_policy_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Drop arrays and keep only scalar metrics needed for paper tables."""
    lead = metrics.get("lead_time", {})
    if not isinstance(lead, dict):
        lead = {}
    miss = metrics.get("event_miss_rate", {})
    if not isinstance(miss, dict):
        miss = {}

    return {
        "accuracy": _metric(metrics, "accuracy"),
        "precision": _metric(metrics, "precision"),
        "recall": _metric(metrics, "recall"),
        "f1": _metric(metrics, "f1"),
        "auc": _metric(metrics, "auc"),
        "decision_threshold": _metric(metrics, "decision_threshold", _metric(metrics, "alarm_threshold", 0.5)),
        "val_threshold_score": _metric(metrics, "val_threshold_score"),
        "event_miss_rate": _metric(miss, "miss_rate"),
        "n_events": _metric(miss, "n_events"),
        "n_missed": _metric(miss, "n_missed"),
        "avg_lead_time": _metric(lead, "avg_lead"),
        "median_lead_time": _metric(lead, "median_lead"),
        "lead_ge_30s_ratio": _metric(lead, "lead_ge_30s_ratio"),
        "switching_freq": _metric(metrics, "switching_freq"),
    }


def compact_pareto_results(pareto_results: Dict[str, Dict[Any, Dict[str, Any]]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    compact: Dict[str, Dict[str, Dict[str, float]]] = {}
    for method, by_m in pareto_results.items():
        compact[method] = {}
        for m_value, metrics in by_m.items():
            compact[method][str(m_value)] = compact_policy_metrics(metrics)
    return compact


def aggregate_method_metrics(
    runs: List[Dict[str, Any]],
    section: str,
    metric_names: Iterable[str],
) -> Dict[str, Dict[str, float]]:
    by_method: Dict[str, Dict[str, List[float]]] = {}
    for run in runs:
        for method, metrics in run[section].items():
            bucket = by_method.setdefault(method, {})
            for key in metric_names:
                bucket.setdefault(key, []).append(_metric(metrics, key))

    summary: Dict[str, Dict[str, float]] = {}
    for method, metric_values in by_method.items():
        summary[method] = {}
        for key, values in metric_values.items():
            mean, std = _mean_std(values)
            summary[method][f"{key}_mean"] = mean
            summary[method][f"{key}_std"] = std
            summary[method][f"{key}_n"] = int(len(values))
    return summary


def aggregate_runs(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    return aggregate_method_metrics(runs, "baseline_results", BASELINE_METRICS)


def aggregate_pareto_runs(runs: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[Tuple[str, str], Dict[str, List[float]]] = {}
    for run in runs:
        for method, by_m in run.get("pareto_results", {}).items():
            for m_value, metrics in by_m.items():
                bucket = buckets.setdefault((method, str(m_value)), {})
                for key in PARETO_METRICS:
                    bucket.setdefault(key, []).append(_metric(metrics, key))

    summary: Dict[str, Dict[str, float]] = {}
    for (method, m_value), metric_values in buckets.items():
        row_key = f"{method}|M={m_value}"
        summary[row_key] = {"method": method, "m": float(m_value)}
        for key, values in metric_values.items():
            mean, std = _mean_std(values)
            summary[row_key][f"{key}_mean"] = mean
            summary[row_key][f"{key}_std"] = std
            summary[row_key][f"{key}_n"] = int(len(values))
    return summary


def save_summary_csv(path: Path, summary: Dict[str, Dict[str, float]], roots: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["method"]
        for root in roots:
            header.extend([f"{root}_mean", f"{root}_std", f"{root}_n"])
        writer.writerow(header)
        for method in sorted(summary.keys()):
            row = [method]
            for root in roots:
                row.extend([
                    summary[method].get(f"{root}_mean", 0.0),
                    summary[method].get(f"{root}_std", 0.0),
                    summary[method].get(f"{root}_n", 0),
                ])
            writer.writerow(row)


def save_pareto_summary_csv(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["method", "m"]
        for root in PARETO_METRICS:
            header.extend([f"{root}_mean", f"{root}_std", f"{root}_n"])
        writer.writerow(header)
        for row_key in sorted(summary.keys(), key=lambda k: (str(summary[k]["method"]), float(summary[k]["m"]))):
            row_data = summary[row_key]
            row = [row_data["method"], int(row_data["m"])]
            for root in PARETO_METRICS:
                row.extend([
                    row_data.get(f"{root}_mean", 0.0),
                    row_data.get(f"{root}_std", 0.0),
                    row_data.get(f"{root}_n", 0),
                ])
            writer.writerow(row)


def save_per_run_csv(path: Path, runs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "split_seed", "fold", "method", *BASELINE_METRICS])
        for run in runs:
            for method, metrics in sorted(run["baseline_results"].items()):
                writer.writerow([
                    run["run_id"],
                    run["split_seed"],
                    run["experiment_kfold_index"],
                    method,
                    *[metrics.get(metric, 0.0) for metric in BASELINE_METRICS],
                ])


def _fmt_mean_std(metrics: Dict[str, float], root: str, scale: float = 1.0, digits: int = 3) -> str:
    mean = metrics.get(f"{root}_mean", 0.0) * scale
    std = metrics.get(f"{root}_std", 0.0) * scale
    return f"{mean:.{digits}f} ± {std:.{digits}f}"


def save_baseline_markdown(path: Path, summary: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Method | F1 | AUC | Miss Rate (%) | Lead Time | Switch |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in sorted(summary.keys()):
        metrics = summary[method]
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    _fmt_mean_std(metrics, "f1"),
                    _fmt_mean_std(metrics, "auc"),
                    _fmt_mean_std(metrics, "event_miss_rate", scale=100.0, digits=1),
                    _fmt_mean_std(metrics, "avg_lead_time", digits=1),
                    _fmt_mean_std(metrics, "switching_freq"),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_one(args: argparse.Namespace, run_id: str, split_seed: int, split_mode: str, fold_index: int = 0) -> Dict[str, Any]:
    from run_experiments import run_experiments

    save_dir = args.output_root / run_id
    fig_dir = args.fig_root / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    run_args = SimpleNamespace(
        quick=args.quick,
        smoke=args.smoke,
        smoke_max_files=args.smoke_max_files,
        force_retrain=args.force_retrain,
        split_seed=split_seed,
        experiment_split_mode=split_mode,
        experiment_kfold_n_splits=args.folds,
        experiment_kfold_index=fold_index,
        save_dir=str(save_dir),
        fig_dir=str(fig_dir),
        batch_size=args.batch_size,
        data_num_workers=args.data_num_workers,
        iql_batch_size=args.iql_batch_size,
        cql_batch_size=args.cql_batch_size,
        cql_max_negative_eval_batch=args.cql_max_negative_eval_batch,
        rl_num_workers=args.rl_num_workers,
        rl_preload_to_device=args.rl_preload_to_device,
        offline_build_batch_steps=args.offline_build_batch_steps,
        offline_encode_batch_size=args.offline_encode_batch_size,
        soft_attn_batch_size=args.soft_attn_batch_size,
        soft_attn_embed_batch_size=args.soft_attn_embed_batch_size,
        soft_attn_preload_to_device=args.soft_attn_preload_to_device,
        policy_eval_batch_size=args.policy_eval_batch_size,
    )
    artifacts = run_experiments(run_args, return_artifacts=True)
    baseline_results = {
        method: compact_policy_metrics(metrics)
        for method, metrics in artifacts["baseline_results"].items()
    }
    pareto_results = compact_pareto_results(artifacts.get("pareto_results", {}))
    return {
        "run_id": run_id,
        "split_seed": split_seed,
        "experiment_split_mode": split_mode,
        "experiment_kfold_n_splits": args.folds,
        "experiment_kfold_index": fold_index,
        "save_dir": str(save_dir),
        "fig_dir": str(fig_dir),
        "baseline_results": baseline_results,
        "pareto_results": pareto_results,
        "elapsed": float(artifacts.get("elapsed", 0.0)),
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.fig_root.mkdir(parents=True, exist_ok=True)

    runs: List[Dict[str, Any]] = []
    if args.mode == "kfold":
        base_seed = parse_seed_list(args.seeds)[0]
        for fold in range(args.folds):
            run_id = f"kfold_seed{base_seed}_fold{fold}"
            runs.append(run_one(args, run_id, base_seed, "kfold", fold_index=fold))
    else:
        for seed in parse_seed_list(args.seeds):
            run_id = f"random_seed{seed}"
            runs.append(run_one(args, run_id, seed, "random", fold_index=0))

    summary = aggregate_runs(runs)
    pareto_summary = aggregate_pareto_runs(runs)
    payload = {
        "mode": args.mode,
        "runs": runs,
        "summary": summary,
        "pareto_summary": pareto_summary,
    }
    json_path = args.output_root / "repeated_experiment_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    save_summary_csv(args.output_root / "repeated_experiment_summary.csv", summary, BASELINE_METRICS)
    save_summary_csv(args.output_root / "baseline_summary.csv", summary, BASELINE_METRICS)
    save_baseline_markdown(args.output_root / "baseline_summary.md", summary)
    save_per_run_csv(args.output_root / "per_run_metrics.csv", runs)
    save_pareto_summary_csv(args.output_root / "pareto_summary.csv", pareto_summary)

    print(f"[Repeated] summary saved to {json_path}")
    for method, metrics in sorted(summary.items()):
        print(
            f"{method:<18} "
            f"F1={metrics.get('f1_mean', 0.0):.4f}±{metrics.get('f1_std', 0.0):.4f} "
            f"AUC={metrics.get('auc_mean', 0.0):.4f}±{metrics.get('auc_std', 0.0):.4f} "
            f"Lead={metrics.get('avg_lead_time_mean', 0.0):.1f}±{metrics.get('avg_lead_time_std', 0.0):.1f}"
        )


if __name__ == "__main__":
    main()
