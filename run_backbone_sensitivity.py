"""
Backbone sensitivity experiments for the paper.

This script runs the third-layer evidence:

1) predictor-only backbone comparison:
   LSTM / CNN / Transformer / TCN / PatchTST on risk prediction metrics.

2) end-to-end backbone sensitivity:
   Backbone + IQL, where the policy pipeline is fully retrained for each
   predictor backbone and evaluated with the same policy metrics used by the
   main experiment.

The policy side is intentionally fixed to IQL. This avoids an expensive and
low-value Cartesian product such as PatchTST+CQL, TCN+BC, etc.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("CONDA_DLL_SEARCH_MODIFICATION_ENABLE", "1")


def _configure_windows_dll_search_paths() -> None:
    """Make conda-installed PyTorch DLLs discoverable before importing torch."""
    if os.name != "nt":
        return
    prefix = Path(sys.exec_prefix)
    for dll_dir in [
        prefix / "Library" / "bin",
        prefix / "Lib" / "site-packages" / "torch" / "lib",
        prefix / "bin",
    ]:
        if not dll_dir.exists():
            continue
        try:
            os.add_dll_directory(str(dll_dir))
        except (AttributeError, FileNotFoundError, OSError):
            pass


_configure_windows_dll_search_paths()

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from artifact_utils import artifact_matches, build_artifact_signature
from build_offline_dataset import build_offline_dataset
from compare_predictors import (
    CompareConfig,
    build_model,
    compute_pos_weight,
    evaluate_model,
    load_or_build_dataloaders,
    set_seed,
    train_model,
)
from config import Config
from dataset import prepare_data_time_split
from iql_networks import PolicyNetwork
from model import MultiTaskLoss
from run_experiments import (
    _build_eval_scope,
    _configure_torch_runtime,
    _evaluate_policy_with_tuned_threshold,
    _set_global_seed,
    make_iql_policy_fn,
)
from train_iql import train_iql


def parse_seed_list(text: str) -> List[int]:
    seeds: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            seeds.append(int(item))
    return seeds or [42]


def parse_model_list(text: str) -> List[str]:
    models = []
    for item in str(text).split(","):
        item = item.strip().lower()
        if item:
            models.append(item)
    return models or ["lstm", "tcn", "transformer", "patchtst"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run predictor-only and end-to-end IQL backbone sensitivity experiments."
    )
    parser.add_argument("--mode", choices=["random", "kfold"], default="kfold")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds. K-fold uses the first seed.")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument(
        "--models",
        default="lstm,tcn,transformer,patchtst",
        help="Comma-separated backbone names: lstm,cnn,transformer,tcn,patchtst.",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke_max_files", type=int, default=12)
    parser.add_argument("--force_retrain", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("checkpoints/backbone_sensitivity"))
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--predictor_epochs", type=int, default=None)
    parser.add_argument("--predictor_patience", type=int, default=None)
    parser.add_argument("--iql_epochs", type=int, default=None)
    parser.add_argument("--iql_batch_size", type=int, default=None)
    parser.add_argument("--offline_build_batch_steps", type=int, default=None)
    parser.add_argument("--offline_encode_batch_size", type=int, default=None)
    parser.add_argument("--policy_eval_batch_size", type=int, default=None)
    parser.add_argument("--no_rl_preload", dest="rl_preload_to_device", action="store_false")
    parser.set_defaults(rl_preload_to_device=None)
    parser.add_argument("--transformer_d_model", type=int, default=None)
    parser.add_argument("--transformer_nhead", type=int, default=None)
    parser.add_argument("--transformer_ff_dim", type=int, default=None)
    parser.add_argument("--transformer_layers", type=int, default=None)
    parser.add_argument("--transformer_dropout", type=float, default=None)
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--patch_stride", type=int, default=None)
    parser.add_argument("--patch_d_model", type=int, default=None)
    parser.add_argument("--patch_layers", type=int, default=None)
    parser.add_argument("--tcn_kernel_size", type=int, default=None)
    return parser.parse_args()


def build_project_cfg(args: argparse.Namespace, seed: int, split_mode: str, fold_index: int, save_dir: Path) -> Config:
    cfg = Config()
    cfg.split_seed = int(seed)
    cfg.experiment_split_mode = split_mode
    cfg.experiment_kfold_n_splits = int(args.folds)
    cfg.experiment_kfold_index = int(fold_index)
    cfg.save_dir = str(save_dir)
    cfg.fig_dir = str(save_dir / "figures")
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.fig_dir, exist_ok=True)

    if args.batch_size is not None:
        cfg.batch_size = int(args.batch_size)
    if args.num_workers is not None:
        cfg.data_num_workers = max(0, int(args.num_workers))
    if args.iql_epochs is not None:
        cfg.iql_epochs = int(args.iql_epochs)
    if args.iql_batch_size is not None:
        cfg.iql_batch_size = int(args.iql_batch_size)
    if args.offline_build_batch_steps is not None:
        cfg.offline_build_batch_steps = int(args.offline_build_batch_steps)
    if args.offline_encode_batch_size is not None:
        cfg.offline_encode_batch_size = int(args.offline_encode_batch_size)
    if args.policy_eval_batch_size is not None:
        cfg.policy_eval_batch_size = int(args.policy_eval_batch_size)
    if args.rl_preload_to_device is not None:
        cfg.rl_preload_to_device = bool(args.rl_preload_to_device)

    if args.quick:
        cfg.epochs = min(cfg.epochs, 10)
        cfg.iql_epochs = min(cfg.iql_epochs, 20)
        cfg.iql_n_mask_samples = min(cfg.iql_n_mask_samples, 5)
        cfg.iql_step_interval = max(cfg.iql_step_interval, 10)
    if args.smoke:
        cfg.epochs = 2
        cfg.iql_epochs = 3
        cfg.iql_n_mask_samples = 1
        cfg.iql_step_interval = 30
        cfg.experiment_max_files = max(3, int(args.smoke_max_files))
    return cfg


def build_compare_cfg(project_cfg: Config, args: argparse.Namespace, model_name: str, seed: int, save_dir: Path) -> CompareConfig:
    cfg = CompareConfig()
    cfg.data_protocol = "project"
    cfg.data_path = project_cfg.data_path
    cfg.data_dir = project_cfg.data_dir
    cfg.experiment_glob = project_cfg.experiment_glob
    cfg.experiment_max_files = project_cfg.experiment_max_files
    cfg.experiment_split_mode = project_cfg.experiment_split_mode
    cfg.experiment_kfold_n_splits = project_cfg.experiment_kfold_n_splits
    cfg.experiment_kfold_index = project_cfg.experiment_kfold_index
    cfg.window_size = project_cfg.window_size
    cfg.delta = project_cfg.delta
    cfg.mask_budget = project_cfg.n_active
    cfg.batch_size = project_cfg.batch_size
    cfg.epochs = int(args.predictor_epochs if args.predictor_epochs is not None else project_cfg.epochs)
    cfg.patience = int(args.predictor_patience if args.predictor_patience is not None else project_cfg.patience)
    cfg.lr = project_cfg.lr
    cfg.weight_decay = project_cfg.weight_decay
    cfg.lambda_tte = project_cfg.lambda_tte
    cfg.huber_delta = project_cfg.huber_delta
    cfg.hidden_dim = project_cfg.hidden_dim
    cfg.num_layers = project_cfg.num_layers
    cfg.lstm_dropout = project_cfg.lstm_dropout
    cfg.head_dropout = project_cfg.head_dropout
    cfg.train_ratio = project_cfg.train_ratio
    cfg.val_ratio = project_cfg.val_ratio
    cfg.test_ratio = project_cfg.test_ratio
    cfg.temp_threshold = project_cfg.temp_threshold
    cfg.trans_threshold = project_cfg.trans_threshold
    cfg.heat_threshold = project_cfg.heat_threshold
    cfg.co_threshold = project_cfg.co_threshold
    cfg.hazard_min_modalities = project_cfg.hazard_min_modalities
    cfg.hazard_min_consecutive = project_cfg.hazard_min_consecutive
    cfg.auto_reverse_experiments = project_cfg.auto_reverse_experiments
    cfg.device = project_cfg.device
    cfg.seed = int(seed)
    cfg.seeds = (int(seed),)
    cfg.num_workers = project_cfg.data_num_workers
    cfg.model_names = (model_name,)
    cfg.min_pos_weight = project_cfg.min_pos_weight
    cfg.max_pos_weight = project_cfg.max_pos_weight
    cfg.save_dir = str(save_dir / "predictor")

    if args.transformer_d_model is not None:
        cfg.transformer_d_model = int(args.transformer_d_model)
    if args.transformer_nhead is not None:
        cfg.transformer_nhead = int(args.transformer_nhead)
    if args.transformer_ff_dim is not None:
        cfg.transformer_ff_dim = int(args.transformer_ff_dim)
    if args.transformer_layers is not None:
        cfg.transformer_layers = int(args.transformer_layers)
    if args.transformer_dropout is not None:
        cfg.transformer_dropout = float(args.transformer_dropout)
    if args.patch_len is not None:
        cfg.patch_len = int(args.patch_len)
    if args.patch_stride is not None:
        cfg.patch_stride = int(args.patch_stride)
    if args.patch_d_model is not None:
        cfg.patch_d_model = int(args.patch_d_model)
    if args.patch_layers is not None:
        cfg.patch_layers = int(args.patch_layers)
    if args.tcn_kernel_size is not None:
        cfg.tcn_kernel_size = int(args.tcn_kernel_size)
    return cfg


def compact_policy_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    lead = metrics.get("lead_time", {}) if isinstance(metrics.get("lead_time", {}), dict) else {}
    miss = metrics.get("event_miss_rate", {}) if isinstance(metrics.get("event_miss_rate", {}), dict) else {}
    return {
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "precision": float(metrics.get("precision", 0.0)),
        "recall": float(metrics.get("recall", 0.0)),
        "f1": float(metrics.get("f1", 0.0)),
        "auc": float(metrics.get("auc", 0.0)),
        "decision_threshold": float(metrics.get("decision_threshold", 0.5)),
        "event_miss_rate": float(miss.get("miss_rate", 0.0)),
        "avg_lead_time": float(lead.get("avg_lead", 0.0)),
        "switching_freq": float(metrics.get("switching_freq", 0.0)),
    }


def compact_predictor_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    keys = ["loss", "auc", "f1", "precision", "recall", "tte_huber", "tte_mse", "tte_mae"]
    return {key: float(metrics.get(key, 0.0)) for key in keys}


def aggregate(runs: List[Dict[str, Any]], section: str) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, Dict[str, List[float]]] = {}
    for run in runs:
        for model_name, metrics in run[section].items():
            model_bucket = buckets.setdefault(model_name, {})
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.floating, np.integer)) and np.isfinite(float(value)):
                    model_bucket.setdefault(key, []).append(float(value))

    summary: Dict[str, Dict[str, float]] = {}
    for model_name, metric_values in buckets.items():
        summary[model_name] = {}
        for key, values in metric_values.items():
            arr = np.asarray(values, dtype=np.float64)
            summary[model_name][f"{key}_mean"] = float(arr.mean()) if len(arr) else 0.0
            summary[model_name][f"{key}_std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            summary[model_name][f"{key}_n"] = int(len(arr))
    return summary


def json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def save_summary_csv(path: Path, summary: Dict[str, Dict[str, float]], metric_roots: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["model"]
        for metric in metric_roots:
            header.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])
        writer.writerow(header)
        for model_name in sorted(summary.keys()):
            row = [model_name]
            for metric in metric_roots:
                row.extend([
                    summary[model_name].get(f"{metric}_mean", 0.0),
                    summary[model_name].get(f"{metric}_std", 0.0),
                    summary[model_name].get(f"{metric}_n", 0),
                ])
            writer.writerow(row)


def train_or_load_predictor(
    model_name: str,
    compare_cfg: CompareConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.nn.Module, Dict[str, float], Dict[str, float], str]:
    train_loader, val_loader, test_loader, meta = load_or_build_dataloaders(compare_cfg, seed=compare_cfg.seed)
    pos_weight = compute_pos_weight(meta)
    predictor = build_model(model_name, n_sensors=int(meta["n_sensors"]), cfg=compare_cfg).to(device)

    ckpt_path = Path(compare_cfg.save_dir) / f"{model_name}_seed{int(compare_cfg.seed)}_best.pt"
    if ckpt_path.exists() and not args.force_retrain:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        predictor.load_state_dict(ckpt["model_state_dict"])
        best_val = compact_predictor_metrics(ckpt.get("val_metrics", {}))
        print(f"[Predictor] reuse {model_name}: {ckpt_path}")
    else:
        best_val_raw, _ = train_model(
            model=predictor,
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=compare_cfg,
            device=device,
            pos_weight=pos_weight,
        )
        best_val = compact_predictor_metrics(best_val_raw)

    criterion = MultiTaskLoss(
        lambda_tte=compare_cfg.lambda_tte,
        huber_delta=compare_cfg.huber_delta,
        pos_weight=pos_weight,
    ).to(device)
    test_metrics = evaluate_model(
        predictor,
        test_loader,
        criterion,
        device,
        threshold=compare_cfg.alarm_threshold,
        huber_delta=compare_cfg.huber_delta,
    )
    return predictor, best_val, compact_predictor_metrics(test_metrics), str(ckpt_path)


def run_one_backbone(args: argparse.Namespace, run_id: str, seed: int, split_mode: str, fold_index: int, model_name: str) -> Dict[str, Any]:
    model_dir = args.output_root / run_id / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    project_cfg = build_project_cfg(args, seed, split_mode, fold_index, model_dir)
    device = torch.device(project_cfg.device if torch.cuda.is_available() else "cpu")
    _configure_torch_runtime(project_cfg, device)
    _set_global_seed(seed)
    set_seed(seed)

    print("\n" + "=" * 72)
    print(f"[Backbone] run_id={run_id} model={model_name} device={device}")
    print("=" * 72)

    data_info = prepare_data_time_split(project_cfg)
    n_sensors = int(data_info["n_sensors"])
    project_cfg.n_sensors = n_sensors
    project_cfg.n_active = min(project_cfg.n_active, n_sensors)

    compare_cfg = build_compare_cfg(project_cfg, args, model_name, seed, model_dir)
    predictor, best_val_metrics, predictor_test_metrics, predictor_ckpt = train_or_load_predictor(
        model_name,
        compare_cfg,
        args,
        device,
    )
    predictor.eval()

    feature_dim = int(getattr(predictor, "feature_dim", project_cfg.hidden_dim))
    project_cfg.hidden_dim = feature_dim
    artifact_signature = build_artifact_signature(
        project_cfg,
        data_info,
        extra={
            "backbone": model_name,
            "predictor_feature_dim": feature_dim,
            "compare_config": asdict(compare_cfg),
        },
    )

    train_experiments = data_info.get("train_experiments", [])
    offline_path = Path(project_cfg.save_dir) / "offline_dataset.pt"
    if offline_path.exists() and not args.force_retrain and artifact_matches(str(offline_path), n_sensors, artifact_signature):
        print(f"[Offline] reuse {offline_path}")
    else:
        build_offline_dataset(
            project_cfg,
            data_norm=data_info["data_norm"],
            labels=data_info["labels"],
            ttes=data_info["ttes"],
            cls_labels=data_info["cls_labels"],
            n_sensors=n_sensors,
            train_range=data_info["train_range"],
            train_experiments=train_experiments,
            predictor=predictor,
            sensor_types=data_info.get("sensor_types", []),
            artifact_signature=artifact_signature,
        )

    iql_path = Path(project_cfg.save_dir) / "policy_best.pt"
    if iql_path.exists() and not args.force_retrain and artifact_matches(str(iql_path), n_sensors, artifact_signature):
        print(f"[IQL] reuse {iql_path}")
    else:
        train_iql(project_cfg, epochs=project_cfg.iql_epochs, artifact_signature=artifact_signature)

    ckpt = torch.load(iql_path, map_location=device, weights_only=False)
    pi_iql = PolicyNetwork(
        ckpt.get("hidden_dim", feature_dim),
        ckpt.get("n_sensors", n_sensors),
        project_cfg.iql_hidden_dim,
    ).to(device)
    pi_iql.load_state_dict(ckpt["pi_net"])
    pi_iql.eval()

    val_scope = _build_eval_scope(data_info, project_cfg, split="val")
    test_scope = _build_eval_scope(data_info, project_cfg, split="test")
    policy_fn = make_iql_policy_fn(
        pi_iql,
        feature_dim,
        device,
        belief_mode=str(getattr(project_cfg, "policy_belief_mode", "full")),
    )
    end_to_end_metrics = _evaluate_policy_with_tuned_threshold(
        predictor,
        policy_fn,
        val_scope,
        test_scope,
        project_cfg,
        device,
    )

    compact_e2e = compact_policy_metrics(end_to_end_metrics)
    print(
        f"[{model_name}+IQL] F1={compact_e2e['f1']:.4f} AUC={compact_e2e['auc']:.4f} "
        f"Miss={compact_e2e['event_miss_rate']:.3f} Lead={compact_e2e['avg_lead_time']:.1f} "
        f"Switch={compact_e2e['switching_freq']:.4f}"
    )

    return {
        "run_id": run_id,
        "seed": int(seed),
        "split_mode": split_mode,
        "fold_index": int(fold_index),
        "model": model_name,
        "save_dir": str(model_dir),
        "predictor_checkpoint": predictor_ckpt,
        "predictor_val": best_val_metrics,
        "predictor_only": {model_name: predictor_test_metrics},
        "end_to_end_iql": {model_name: compact_e2e},
    }


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    models = parse_model_list(args.models)

    runs: List[Dict[str, Any]] = []
    start = time.time()
    if args.mode == "kfold":
        seed = parse_seed_list(args.seeds)[0]
        for fold in range(args.folds):
            run_id = f"kfold_seed{seed}_fold{fold}"
            for model_name in models:
                runs.append(run_one_backbone(args, run_id, seed, "kfold", fold, model_name))
    else:
        for seed in parse_seed_list(args.seeds):
            run_id = f"random_seed{seed}"
            for model_name in models:
                runs.append(run_one_backbone(args, run_id, seed, "random", 0, model_name))

    predictor_summary = aggregate(runs, "predictor_only")
    e2e_summary = aggregate(runs, "end_to_end_iql")
    payload = {
        "elapsed_minutes": (time.time() - start) / 60.0,
        "args": json_safe(vars(args)),
        "runs": runs,
        "predictor_summary": predictor_summary,
        "end_to_end_iql_summary": e2e_summary,
    }

    json_path = args.output_root / "backbone_sensitivity_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    save_summary_csv(
        args.output_root / "predictor_only_summary.csv",
        predictor_summary,
        ["auc", "f1", "precision", "recall", "tte_mae", "tte_huber", "loss"],
    )
    save_summary_csv(
        args.output_root / "end_to_end_iql_summary.csv",
        e2e_summary,
        ["f1", "auc", "precision", "recall", "event_miss_rate", "avg_lead_time", "switching_freq"],
    )

    print(f"\n[BackboneSensitivity] saved: {json_path}")
    print("[BackboneSensitivity] End-to-end IQL summary:")
    for model_name, metrics in sorted(e2e_summary.items()):
        print(
            f"{model_name:<12} "
            f"F1={metrics.get('f1_mean', 0.0):.4f}±{metrics.get('f1_std', 0.0):.4f} "
            f"AUC={metrics.get('auc_mean', 0.0):.4f}±{metrics.get('auc_std', 0.0):.4f} "
            f"Lead={metrics.get('avg_lead_time_mean', 0.0):.1f}±{metrics.get('avg_lead_time_std', 0.0):.1f}"
        )


if __name__ == "__main__":
    main()
