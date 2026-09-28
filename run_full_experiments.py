"""
run_full_experiments.py — 完整实验脚本 (基线对比 + 消融实验)

统一入口，串联 run_experiments.py (基线对比) 和 run_ablation.py (消融实验)，
生成合并的结果文件和可视化图表。

输出:
  - results.txt           : 包含 Baseline Comparison + Ablation Studies 两个表格
  - figures/*.png         : Pareto 曲线, 时间线预警, 传感器热力图, 消融柱状图
  - experiment_results_full.json : JSON 格式完整结果

用法:
  python run_full_experiments.py                   # 完整实验
  python run_full_experiments.py --quick           # 快速测试模式
  python run_full_experiments.py --force_retrain   # 强制重新训练
  python run_full_experiments.py --skip_baselines  # 跳过基线对比 (仅消融)
  python run_full_experiments.py --skip_ablation   # 跳过消融实验 (仅基线)
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import argparse

import torch
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from config import Config
from dataset import prepare_data_time_split
from model import MaskAwareRiskPredictor
from run_experiments import run_experiments
from run_ablation import run_ablation
from visualize import (
    plot_pareto_curves, plot_timeline_warning,
    plot_sensor_heatmap, plot_ablation_bar_chart,
)


def _format_delta_line(label: str, full_f1: float, other_f1: float) -> str:
    diff = float(full_f1 - other_f1)
    if diff >= 0:
        return f"  {label}: F1 下降 {diff:.4f}\n"
    return f"  {label}: F1 上升 {abs(diff):.4f}\n"


def main():
    parser = argparse.ArgumentParser(description="Full Experiments: Baseline + Ablation")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (少量 epochs)")
    parser.add_argument("--smoke", action="store_true",
                        help="Ultra-light smoke mode (minimal files + minimal training).")
    parser.add_argument("--smoke_max_files", type=int, default=12,
                        help="Max experiment files in smoke mode (>=3).")
    parser.add_argument("--force_retrain", action="store_true",
                        help="强制重新训练所有模型")
    parser.add_argument("--split_seed", type=int, default=None,
                        help="Override experiment split seed.")
    parser.add_argument("--experiment_split_mode", type=str, default=None,
                        choices=["random", "kfold"],
                        help="Experiment-level split mode.")
    parser.add_argument("--experiment_kfold_n_splits", type=int, default=None,
                        help="Number of experiment-level K-fold splits.")
    parser.add_argument("--experiment_kfold_index", type=int, default=None,
                        help="Held-out fold index for experiment-level K-fold.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Override checkpoint/output directory.")
    parser.add_argument("--fig_dir", type=str, default=None,
                        help="Override figure output directory.")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override predictor train batch size.")
    parser.add_argument("--data_num_workers", type=int, default=None,
                        help="Override predictor DataLoader workers.")
    parser.add_argument("--iql_batch_size", type=int, default=None,
                        help="Override IQL/BC batch size.")
    parser.add_argument("--cql_batch_size", type=int, default=None,
                        help="Override CQL batch size.")
    parser.add_argument("--cql_max_negative_eval_batch", type=int, default=None,
                        help="Max flattened random-action rows per CQL Q forward chunk.")
    parser.add_argument("--rl_num_workers", type=int, default=None,
                        help="Override offline RL DataLoader workers when not preloaded.")
    parser.add_argument("--no_rl_preload", dest="rl_preload_to_device", action="store_false",
                        help="Disable CUDA preloading for offline RL tensors.")
    parser.set_defaults(rl_preload_to_device=None)
    parser.add_argument("--offline_build_batch_steps", type=int, default=None,
                        help="Decision steps grouped per offline-dataset predictor batch.")
    parser.add_argument("--offline_encode_batch_size", type=int, default=None,
                        help="Chunk size for predictor encoding during offline dataset build.")
    parser.add_argument("--soft_attn_batch_size", type=int, default=None,
                        help="Override SoftAttention train batch size.")
    parser.add_argument("--soft_attn_embed_batch_size", type=int, default=None,
                        help="Override SoftAttention full-belief precompute batch size.")
    parser.add_argument("--no_soft_attn_preload", dest="soft_attn_preload_to_device", action="store_false",
                        help="Disable CUDA preloading for SoftAttention tensors.")
    parser.set_defaults(soft_attn_preload_to_device=None)
    parser.add_argument("--policy_eval_batch_size", type=int, default=None,
                        help="Batch size for stateless policy evaluation.")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="跳过基线对比 (仅运行消融实验)")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="跳过消融实验 (仅运行基线对比)")
    args = parser.parse_args()

    print("=" * 80)
    print("完整实验流程: 基线对比 + 消融实验")
    print(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    cfg = Config()
    if args.split_seed is not None:
        cfg.split_seed = int(args.split_seed)
    if args.experiment_split_mode is not None:
        cfg.experiment_split_mode = str(args.experiment_split_mode)
    if args.experiment_kfold_n_splits is not None:
        cfg.experiment_kfold_n_splits = int(args.experiment_kfold_n_splits)
    if args.experiment_kfold_index is not None:
        cfg.experiment_kfold_index = int(args.experiment_kfold_index)
    if args.save_dir:
        cfg.save_dir = os.path.abspath(args.save_dir)
        os.makedirs(cfg.save_dir, exist_ok=True)
    if args.fig_dir:
        cfg.fig_dir = os.path.abspath(args.fig_dir)
        os.makedirs(cfg.fig_dir, exist_ok=True)
    for attr in [
        "batch_size",
        "data_num_workers",
        "iql_batch_size",
        "cql_batch_size",
        "cql_max_negative_eval_batch",
        "rl_num_workers",
        "offline_build_batch_steps",
        "offline_encode_batch_size",
        "soft_attn_batch_size",
        "soft_attn_embed_batch_size",
        "policy_eval_batch_size",
    ]:
        if getattr(args, attr, None) is not None:
            setattr(cfg, attr, int(getattr(args, attr)))
    for attr in ["rl_preload_to_device", "soft_attn_preload_to_device"]:
        if getattr(args, attr, None) is not None:
            setattr(cfg, attr, bool(getattr(args, attr)))
    total_start = time.time()

    baseline_results = {}
    pareto_results = {}
    method_probs = {}
    ablation_results = {}
    timeline_diagnostics = []
    predictor = None
    data_info = None
    device = None

    # ================================================================
    # Part 1: 基线对比实验
    # ================================================================
    if not args.skip_baselines:
        print("\n" + "#" * 60)
        print("Part 1: 基线对比实验 (Baseline Comparison)")
        print("#" * 60)

        artifacts = run_experiments(args, return_artifacts=True)

        baseline_results = artifacts["baseline_results"]
        pareto_results = artifacts["pareto_results"]
        method_probs = artifacts["method_probs"]
        timeline_diagnostics = artifacts.get("timeline_diagnostics", [])
        predictor = artifacts["predictor"]
        data_info = artifacts["data_info"]
        cfg = artifacts["cfg"]
        device = artifacts["device"]
    else:
        print("\n[跳过] 基线对比实验")
        # 需要单独加载必要信息供消融实验使用
        if getattr(args, "smoke", False):
            cfg.experiment_max_files = max(3, int(getattr(args, "smoke_max_files", 12)))
            cfg.save_dir = os.path.join(cfg.save_dir, "smoke")
            cfg.fig_dir = os.path.join(cfg.fig_dir, "smoke")
            os.makedirs(cfg.save_dir, exist_ok=True)
            os.makedirs(cfg.fig_dir, exist_ok=True)
            print(f"[mode] smoke (max_files={cfg.experiment_max_files})")

        device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        data_info = prepare_data_time_split(cfg)
        n_sensors = data_info["n_sensors"]
        cfg.n_sensors = n_sensors
        cfg.n_active = min(cfg.n_active, n_sensors)

        # 加载预训练 LSTM
        ckpt_path = os.path.join(cfg.save_dir, "best_model.pt")
        if os.path.exists(ckpt_path):
            predictor = MaskAwareRiskPredictor(
                n_sensors=n_sensors,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                lstm_dropout=cfg.lstm_dropout,
                head_dropout=cfg.head_dropout,
            ).to(device)
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            predictor.load_state_dict(ckpt["model_state_dict"])
        else:
            raise FileNotFoundError(f"请先训练 LSTM 预测器: {ckpt_path}")

    # ================================================================
    # Part 2: 消融实验
    # ================================================================
    if not args.skip_ablation:
        print("\n" + "#" * 60)
        print("Part 2: 消融实验 (Ablation Studies)")
        print("#" * 60)

        ablation_results = run_ablation(
            cfg, predictor, data_info, baseline_results, device, args
        )

        # 生成消融柱状图
        plot_ablation_bar_chart(ablation_results, cfg.fig_dir)
    else:
        print("\n[跳过] 消融实验")

    # ================================================================
    # Part 3: 生成合并结果文件
    # ================================================================
    print("\n" + "#" * 60)
    print("Part 3: 生成完整结果文件")
    print("#" * 60)

    total_elapsed = time.time() - total_start
    results_path = os.path.join(os.path.dirname(__file__), "results.txt")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Mask-Aware Risk Predictor + RL 策略 — 完整实验结果\n")
        f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {total_elapsed/60:.1f} 分钟\n")
        f.write("=" * 80 + "\n\n")

        # 实验配置
        f.write("实验配置:\n")
        f.write(f"  传感器数: {cfg.n_sensors}\n")
        f.write(f"  默认活跃传感器数 M: {cfg.n_active}\n")
        f.write(f"  窗口大小 W: {cfg.window_size}\n")
        f.write(f"  预警阈值 Δ: {cfg.delta}s\n")
        if data_info:
            f.write(f"  数据长度: {len(data_info['data_norm'])} 时间步\n")
            f.write(f"  火灾起始步: {data_info['fire_onset']}\n")
            f.write(f"  split_mode: {data_info.get('split_mode', 'unknown')}\n")
            if str(data_info.get("split_mode", "")).startswith("experiment_"):
                f.write(
                    f"  experiment_split_mode: {getattr(cfg, 'experiment_split_mode', 'random')}, "
                    f"split_seed={getattr(cfg, 'split_seed', 42)}, "
                    f"kfold={getattr(cfg, 'experiment_kfold_index', 0)}/"
                    f"{getattr(cfg, 'experiment_kfold_n_splits', 0)}\n"
                )
        f.write(f"  训练 epochs: LSTM={cfg.epochs}, IQL={cfg.iql_epochs}, "
                f"CQL={cfg.cql_epochs}, BC={cfg.bc_epochs}\n")
        f.write(
            f"  predictor_select_metric: {getattr(cfg, 'predictor_select_metric', 'f1_tuned')}, "
            f"balanced_train_sampling={bool(getattr(cfg, 'balanced_train_sampling', False))}, "
            f"pos_weight_clip=[{float(getattr(cfg, 'min_pos_weight', 0.1)):.1f}, "
            f"{float(getattr(cfg, 'max_pos_weight', 30.0)):.1f}]\n"
        )
        if getattr(cfg, "tune_alarm_threshold", False):
            f.write(
                f"  decision_threshold: val->{cfg.threshold_metric} "
                f"in [{cfg.threshold_search_min:.2f}, {cfg.threshold_search_max:.2f}] "
                f"x {cfg.threshold_search_steps}\n"
            )
        else:
            f.write(f"  decision_threshold: fixed={cfg.alarm_threshold:.3f}\n")
        f.write(f"  policy_belief_mode: {getattr(cfg, 'policy_belief_mode', 'full')}\n")
        f.write(f"  Pareto M values: {cfg.pareto_m_values}\n")
        f.write("\n")

        if len(timeline_diagnostics) > 0:
            f.write("Timeline Diagnostics:\n")
            f.write(f"{'Exp':<16} {'Len':>6} {'WarnOn':>8} {'FireOn':>8} {'WarnW':>8} {'DangW':>8}\n")
            f.write("-" * 80 + "\n")
            for diag in timeline_diagnostics:
                f.write(
                    f"{diag['exp_id']:<16} "
                    f"{diag['length']:>6} "
                    f"{diag['warning_onset']:>8} "
                    f"{diag['fire_onset']:>8} "
                    f"{diag['warning_width']:>8} "
                    f"{diag['danger_width']:>8}\n"
                )
            f.write("-" * 80 + "\n\n")

        # Section 1: Baseline Comparison
        if baseline_results:
            f.write("=" * 80 + "\n")
            f.write("Section 1: Baseline Comparison (基线对比)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Method':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
                    f"{'F1':>6} {'AUC':>6} {'Thr':>6} {'Miss%':>6} {'Lead':>6} {'Switch':>6}\n")
            f.write("-" * 80 + "\n")

            for method_name, metrics in baseline_results.items():
                em = metrics.get("event_miss_rate", {})
                miss_pct = em.get("miss_rate", 0) * 100
                f.write(f"{method_name:<20} "
                        f"{metrics['accuracy']:>6.3f} "
                        f"{metrics['precision']:>6.3f} "
                        f"{metrics['recall']:>6.3f} "
                        f"{metrics['f1']:>6.3f} "
                        f"{metrics['auc']:>6.3f} "
                        f"{metrics.get('decision_threshold', cfg.alarm_threshold):>6.3f} "
                        f"{miss_pct:>6.1f} "
                        f"{metrics['lead_time']['avg_lead']:>6.1f} "
                        f"{metrics['switching_freq']:>6.3f}\n")

            f.write("-" * 80 + "\n\n")

        # Section 2: Ablation Studies
        if ablation_results:
            f.write("=" * 80 + "\n")
            f.write("Section 2: Ablation Studies (消融实验)\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"{'Config':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
                    f"{'F1':>6} {'AUC':>6} {'Thr':>6} {'Miss%':>6} {'Lead':>6} {'Switch':>6}\n")
            f.write("-" * 80 + "\n")

            for config_name, metrics in ablation_results.items():
                em = metrics.get("event_miss_rate", {})
                miss_pct = em.get("miss_rate", 0) * 100
                f.write(f"{config_name:<20} "
                        f"{metrics['accuracy']:>6.3f} "
                        f"{metrics['precision']:>6.3f} "
                        f"{metrics['recall']:>6.3f} "
                        f"{metrics['f1']:>6.3f} "
                        f"{metrics['auc']:>6.3f} "
                        f"{metrics.get('decision_threshold', cfg.alarm_threshold):>6.3f} "
                        f"{miss_pct:>6.1f} "
                        f"{metrics['lead_time']['avg_lead']:>6.1f} "
                        f"{metrics['switching_freq']:>6.3f}\n")

            f.write("-" * 80 + "\n\n")

            # 消融分析
            f.write("消融分析:\n")
            if "Full" in ablation_results and "w/o Predictor" in ablation_results:
                f.write(_format_delta_line(
                    "移除 Mask-Aware Predictor",
                    ablation_results["Full"]["f1"],
                    ablation_results["w/o Predictor"]["f1"],
                ))
            if "Full" in ablation_results and "w/o Policy" in ablation_results:
                f.write(_format_delta_line(
                    "移除动态策略网络",
                    ablation_results["Full"]["f1"],
                    ablation_results["w/o Policy"]["f1"],
                ))
            if "Full" in ablation_results and "w/o RL" in ablation_results:
                f.write(_format_delta_line(
                    "移除强化学习 (BC替代)",
                    ablation_results["Full"]["f1"],
                    ablation_results["w/o RL"]["f1"],
                ))
            f.write("\n")

        # Section 3: Pareto Sweep
        if pareto_results:
            f.write("=" * 80 + "\n")
            f.write("Section 3: Pareto Sweep (M 值扫描)\n")
            f.write("=" * 80 + "\n\n")

            # F1 表格
            methods = list(pareto_results.keys())
            m_vals = cfg.pareto_m_values

            header = f"{'M':<5}"
            for method in methods:
                header += f" {method:>10}"
            f.write(header + " (F1)\n")
            f.write("-" * 80 + "\n")

            for m_val in m_vals:
                row = f"{m_val:<5}"
                for method in methods:
                    if m_val in pareto_results[method]:
                        f1_val = pareto_results[method][m_val]["f1"]
                        row += f" {f1_val:>10.4f}"
                    else:
                        row += f" {'N/A':>10}"
                f.write(row + "\n")
            f.write("-" * 80 + "\n\n")

        # Section 4: Lead Time 详情
        if baseline_results:
            f.write("=" * 80 + "\n")
            f.write("Section 4: Lead Time 详情\n")
            f.write("=" * 80 + "\n\n")

            for method_name, metrics in baseline_results.items():
                lt = metrics['lead_time']
                f.write(f"  {method_name}: avg={lt['avg_lead']:.1f}s, "
                        f"median={lt['median_lead']:.1f}s, "
                        f">=30s: {lt['lead_ge_30s_ratio']*100:.0f}%\n")
            f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("实验完成\n")

    print(f"[结果] 完整结果已保存至: {results_path}")

    # ================================================================
    # Part 4: 保存 JSON 结果
    # ================================================================
    json_path = os.path.join(cfg.save_dir, "experiment_results_full.json")

    results_json = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": total_elapsed / 60,
        "config": {
            "n_sensors": cfg.n_sensors,
            "n_active": cfg.n_active,
            "window_size": cfg.window_size,
            "delta": cfg.delta,
            "alarm_threshold": float(cfg.alarm_threshold),
            "tune_alarm_threshold": bool(getattr(cfg, "tune_alarm_threshold", False)),
            "threshold_metric": str(getattr(cfg, "threshold_metric", "f1")),
            "threshold_search_min": float(getattr(cfg, "threshold_search_min", cfg.alarm_threshold)),
            "threshold_search_max": float(getattr(cfg, "threshold_search_max", cfg.alarm_threshold)),
            "threshold_search_steps": int(getattr(cfg, "threshold_search_steps", 1)),
            "predictor_select_metric": str(getattr(cfg, "predictor_select_metric", "f1_tuned")),
            "policy_belief_mode": str(getattr(cfg, "policy_belief_mode", "full")),
            "experiment_split_mode": str(getattr(cfg, "experiment_split_mode", "random")),
            "split_seed": int(getattr(cfg, "split_seed", 42)),
            "experiment_kfold_n_splits": int(getattr(cfg, "experiment_kfold_n_splits", 0) or 0),
            "experiment_kfold_index": int(getattr(cfg, "experiment_kfold_index", 0) or 0),
            "balanced_train_sampling": bool(getattr(cfg, "balanced_train_sampling", False)),
            "min_pos_weight": float(getattr(cfg, "min_pos_weight", 0.1)),
            "max_pos_weight": float(getattr(cfg, "max_pos_weight", 30.0)),
            "pareto_m_values": cfg.pareto_m_values,
        },
        "baseline_results": {},
        "ablation_results": {},
        "pareto_results": {},
        "timeline_diagnostics": timeline_diagnostics,
    }

    # 转换基线结果
    for name, m in baseline_results.items():
        em = m.get("event_miss_rate", {})
        results_json["baseline_results"][name] = {
            "accuracy": float(m["accuracy"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "auc": float(m["auc"]),
            "decision_threshold": float(m.get("decision_threshold", cfg.alarm_threshold)),
            "threshold_metric": str(m.get("threshold_metric", getattr(cfg, "threshold_metric", "f1"))),
            "val_threshold_score": float(m.get("val_threshold_score", 0.0)),
            "event_miss_rate": float(em.get("miss_rate", 0)),
            "avg_lead_time": float(m["lead_time"]["avg_lead"]),
            "switching_freq": float(m["switching_freq"]),
        }

    # 转换消融结果
    for name, m in ablation_results.items():
        em = m.get("event_miss_rate", {})
        results_json["ablation_results"][name] = {
            "accuracy": float(m["accuracy"]),
            "precision": float(m["precision"]),
            "recall": float(m["recall"]),
            "f1": float(m["f1"]),
            "auc": float(m["auc"]),
            "decision_threshold": float(m.get("decision_threshold", cfg.alarm_threshold)),
            "threshold_metric": str(m.get("threshold_metric", getattr(cfg, "threshold_metric", "f1"))),
            "val_threshold_score": float(m.get("val_threshold_score", 0.0)),
            "event_miss_rate": float(em.get("miss_rate", 0)),
            "avg_lead_time": float(m["lead_time"]["avg_lead"]),
            "switching_freq": float(m["switching_freq"]),
        }

    # 转换 Pareto 结果
    for method_name, m_results in pareto_results.items():
        results_json["pareto_results"][method_name] = {
            str(m_val): {
                "f1": float(metrics["f1"]),
                "auc": float(metrics["auc"]),
                "decision_threshold": float(metrics.get("decision_threshold", cfg.alarm_threshold)),
                "val_threshold_score": float(metrics.get("val_threshold_score", 0.0)),
                "avg_lead": float(metrics["lead_time"]["avg_lead"]),
            }
            for m_val, metrics in m_results.items()
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"[结果] JSON 已保存至: {json_path}")

    # ================================================================
    # 完成
    # ================================================================
    print("\n" + "=" * 80)
    print(f"完整实验流程完成!")
    print(f"总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"结果文件: {results_path}")
    print(f"图表目录: {cfg.fig_dir}")
    print("=" * 80)

    return {
        "baseline_results": baseline_results,
        "ablation_results": ablation_results,
        "pareto_results": pareto_results,
    }


if __name__ == "__main__":
    main()
