"""
run_ablation.py — 消融实验 (Ablation Studies)

实现 4 种消融配置的训练与评估:

| #  | 配置名称       | 预测器                    | 策略              | 说明 |
|----|---------------|--------------------------|-------------------|------|
| 1  | Full          | LSTM (best_model.pt)     | IQL (policy_best) | 主方法 |
| 2  | w/o Predictor | LogisticRegression       | IQL (单独训练)     | 移除 mask-aware 预测器 |
| 3  | w/o Policy    | LSTM                     | StaticTopM (固定)  | 移除策略网络 |
| 4  | w/o RL        | LSTM                     | BC (policy_bc)     | 移除强化学习 |

用法:
  # 从 run_full_experiments.py 调用 (推荐)
  from run_ablation import run_ablation
  ablation_results = run_ablation(cfg, predictor, data_info, baseline_results, device, args)

  # 独立运行
  python run_ablation.py [--quick] [--force_retrain]
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import argparse

import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from artifact_utils import artifact_matches, build_artifact_signature
from config import Config
from dataset import prepare_data_time_split
from model import MaskAwareRiskPredictor, MultiTaskLoss
from evaluate import (
    evaluate_with_policy,
    evaluate_with_policy_multi_experiments,
    select_best_threshold,
)
from build_offline_dataset import build_offline_dataset
from train_iql import train_iql, train_bc
from iql_networks import PolicyNetwork
from baselines import (
    LogisticRegressionPredictor,
    train_logistic_predictor,
)
from visualize import plot_ablation_bar_chart


# ======================================================================
# 策略函数复用 (与 run_experiments.py 相同定义)
# ======================================================================

def _make_iql_policy_fn(pi_net, device, belief_mode="full"):
    """创建 IQL/BC 策略函数。"""
    state = {"mask": None}

    def policy_fn(predictor, obs_window, n_sensors, m, dev):
        if belief_mode == "previous_mask" and state["mask"] is not None:
            belief_mask = state["mask"]
        elif belief_mode == "previous_mask":
            belief_mask = np.zeros(n_sensors, dtype=np.float32)
            belief_mask[: min(m, n_sensors)] = 1.0
        else:
            belief_mask = np.ones(n_sensors, dtype=np.float32)
        obs_masked = obs_window * belief_mask[np.newaxis, :]
        mask_exp = np.tile(belief_mask, (obs_window.shape[0], 1))
        x = np.concatenate([obs_masked, mask_exp], axis=-1)
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            h = predictor.encode_belief(x_t)
            mask_out = pi_net.select_top_m(h, m=m)
        selected = mask_out.cpu().numpy().squeeze(0)
        state["mask"] = selected.copy()
        return selected
    policy_fn.reset = lambda: state.update({"mask": None})  # type: ignore[attr-defined]
    return policy_fn


def _make_static_policy_fn(data_norm, m):
    """创建静态 TopM 策略函数。"""
    var = data_norm.var(axis=0)
    top_idx = np.argsort(var)[::-1][:m]
    fixed_mask = np.zeros(data_norm.shape[1], dtype=np.float32)
    fixed_mask[top_idx] = 1.0
    def policy_fn(predictor, obs_window, n_sensors, m_, dev):
        return fixed_mask.copy()
    def batch_select(obs_windows, n_sensors, m_, dev):
        return np.repeat(fixed_mask[np.newaxis, :], len(obs_windows), axis=0)
    policy_fn.batch_select = batch_select  # type: ignore[attr-defined]
    return policy_fn


def _build_eval_scope(data_info, cfg, split="test"):
    split = split if split in {"val", "test"} else "test"
    split_mode = data_info.get("split_mode", "time_single")
    experiments = data_info.get(f"{split}_experiments", [])
    if split_mode.startswith("experiment_") and len(experiments) > 0:
        return {
            "mode": "multi",
            "experiments": experiments,
            "eval_data": None,
            "eval_cls": None,
            "eval_ttes": None,
            "eval_fire_onset": data_info.get("eval_fire_onset", 0),
        }

    split_range = data_info[f"{split}_range"]
    t0 = max(0, int(split_range[0]) - cfg.window_size + 1)
    t1 = int(split_range[1])
    return {
        "mode": "single",
        "experiments": [],
        "eval_data": data_info["data_norm"][t0:t1],
        "eval_cls": data_info["cls_labels"][t0:t1],
        "eval_ttes": data_info["ttes"][t0:t1],
        "eval_fire_onset": data_info.get("eval_fire_onset", 0),
    }


def _evaluate_policy_on_scope(predictor, policy_fn, eval_scope, cfg, device):
    if eval_scope["mode"] == "multi":
        return evaluate_with_policy_multi_experiments(
            predictor,
            policy_fn,
            eval_scope["experiments"],
            cfg,
            str(device),
        )
    return evaluate_with_policy(
        predictor,
        policy_fn,
        eval_scope["eval_data"],
        eval_scope["eval_cls"],
        eval_scope["eval_ttes"],
        eval_scope["eval_fire_onset"],
        cfg,
        str(device),
    )


def _build_threshold_grid(cfg):
    n_steps = max(1, int(getattr(cfg, "threshold_search_steps", 19)))
    thr_min = float(getattr(cfg, "threshold_search_min", 0.05))
    thr_max = float(getattr(cfg, "threshold_search_max", 0.95))
    default_threshold = float(getattr(cfg, "alarm_threshold", 0.5))
    if n_steps <= 1 or thr_max <= thr_min:
        return np.array([default_threshold], dtype=np.float32)
    return np.linspace(thr_min, thr_max, n_steps, dtype=np.float32)


def _evaluate_policy_with_tuned_threshold(
    predictor,
    policy_fn,
    val_scope,
    test_scope,
    cfg,
    device,
):
    default_threshold = float(getattr(cfg, "alarm_threshold", 0.5))
    selected_threshold = default_threshold
    tuned_info = None

    if getattr(cfg, "tune_alarm_threshold", False) and val_scope is not None:
        val_metrics = _evaluate_policy_on_scope(
            predictor,
            policy_fn,
            val_scope,
            cfg,
            device,
        )
        tuned_info = select_best_threshold(
            val_metrics.get("probs", np.array([], dtype=np.float32)),
            val_metrics.get("labels", np.array([], dtype=np.float32)),
            metric=getattr(cfg, "threshold_metric", "f1"),
            thresholds=_build_threshold_grid(cfg),
            default_threshold=default_threshold,
        )
        selected_threshold = float(tuned_info["threshold"])

    original_threshold = cfg.alarm_threshold
    cfg.alarm_threshold = selected_threshold
    try:
        test_metrics = _evaluate_policy_on_scope(
            predictor,
            policy_fn,
            test_scope,
            cfg,
            device,
        )
    finally:
        cfg.alarm_threshold = original_threshold

    test_metrics["decision_threshold"] = float(selected_threshold)
    test_metrics["threshold_metric"] = str(getattr(cfg, "threshold_metric", "f1"))
    if tuned_info is not None:
        test_metrics["val_threshold_score"] = float(tuned_info["score"])
        test_metrics["val_threshold_metrics"] = tuned_info["metrics"]
    return test_metrics


# ======================================================================
# 消融实验主函数
# ======================================================================

def run_ablation(
    cfg: Config,
    predictor: MaskAwareRiskPredictor,
    data_info: dict,
    baseline_results: dict,
    device: torch.device,
    args,
) -> dict:
    """
    执行消融实验。

    Args:
        cfg              : 配置对象
        predictor        : 已训练的 LSTM 风险预测器
        data_info        : prepare_data_time_split() 返回的字典
        baseline_results : run_experiments() 返回的基线结果字典
                           需包含 "LSTM+IQL", "LSTM+StaticTopM", "LSTM+BC" 等键
        device           : torch.device
        args             : 命令行参数 (quick, force_retrain)

    Returns:
        ablation_results : dict {config_name: metrics_dict, ...}
    """
    print("\n" + "=" * 60)
    print("消融实验 (Ablation Studies)")
    print("=" * 60)

    start_time = time.time()

    n_sensors = data_info["n_sensors"]
    data_norm = data_info["data_norm"]
    cls_labels = data_info["cls_labels"]
    ttes = data_info["ttes"]
    train_range = data_info["train_range"]
    train_loader = data_info["train_loader"]
    val_loader = data_info["val_loader"]
    train_experiments = data_info.get("train_experiments", [])
    val_scope = _build_eval_scope(data_info, cfg, split="val")
    eval_scope = _build_eval_scope(data_info, cfg, split="test")

    train_data_for_static = data_info.get("train_data_norm")
    if train_data_for_static is None or len(train_data_for_static) == 0:
        if len(train_experiments) > 0:
            train_data_for_static = np.concatenate(
                [exp["data_norm"] for exp in train_experiments], axis=0
            )
        else:
            train_data_for_static = data_norm[train_range[0]:train_range[1]]

    hidden_dim = cfg.hidden_dim
    M = cfg.n_active
    policy_belief_mode = str(getattr(cfg, "policy_belief_mode", "full"))
    artifact_signature = build_artifact_signature(cfg, data_info)

    ablation_results = {}

    # ================================================================
    # 配置 1: Full (完整方法) — 直接复用基线中的 LSTM+IQL 结果
    # ================================================================
    print("\n--- 消融 1/4: Full (完整方法) ---")
    if "LSTM+IQL" in baseline_results:
        ablation_results["Full"] = baseline_results["LSTM+IQL"]
        print(f"  [复用基线] F1={ablation_results['Full']['f1']:.4f}, "
              f"AUC={ablation_results['Full']['auc']:.4f}")
    else:
        print("  [警告] 基线结果中无 LSTM+IQL，需独立评估")
        iql_path = os.path.join(cfg.save_dir, "policy_best.pt")
        iql_ckpt = torch.load(iql_path, map_location=device, weights_only=False)
        pi_iql = PolicyNetwork(
            iql_ckpt.get("hidden_dim", hidden_dim),
            iql_ckpt.get("n_sensors", n_sensors),
            cfg.iql_hidden_dim
        ).to(device)
        pi_iql.load_state_dict(iql_ckpt["pi_net"])
        pi_iql.eval()
        policy_fn = _make_iql_policy_fn(pi_iql, device, belief_mode=policy_belief_mode)
        metrics = _evaluate_policy_with_tuned_threshold(
            predictor, policy_fn, val_scope, eval_scope, cfg, device
        )
        ablation_results["Full"] = metrics
        print(
            f"  F1={metrics['f1']:.4f}, AUC={metrics['auc']:.4f}, "
            f"Thr={metrics.get('decision_threshold', cfg.alarm_threshold):.3f}"
        )

    # ================================================================
    # 配置 2: w/o Predictor — 简单预测器 + IQL 策略
    # ================================================================
    print("\n--- 消融 2/4: w/o Predictor (LogisticRegression + IQL) ---")

    # Step 2a: 训练 LogisticRegressionPredictor
    logistic_path = os.path.join(cfg.save_dir, "logistic_predictor.pt")
    force = getattr(args, 'force_retrain', False)

    if os.path.exists(logistic_path) and not force and artifact_matches(logistic_path, n_sensors, artifact_signature):
        print(f"  [LogisticPredictor] 已存在，跳过训练")
        logistic_predictor = LogisticRegressionPredictor(n_sensors, hidden_dim).to(device)
        ckpt = torch.load(logistic_path, map_location=device, weights_only=False)
        logistic_predictor.load_state_dict(ckpt["model_state_dict"])
    else:
        if os.path.exists(logistic_path) and not force:
            print(f"  [LogisticPredictor] 旧 checkpoint 与当前传感器数不兼容，自动重训")
        logistic_predictor = train_logistic_predictor(
            cfg, train_loader, val_loader, n_sensors, device,
            artifact_signature=artifact_signature,
        )

    logistic_predictor.eval()

    # Step 2b: 用 LogisticPredictor 构建离线数据集
    logistic_ds_path = os.path.join(cfg.save_dir, "offline_dataset_logistic.pt")
    if os.path.exists(logistic_ds_path) and not force and artifact_matches(logistic_ds_path, n_sensors, artifact_signature):
        print(f"  [离线数据集-Logistic] 已存在，跳过构建")
    else:
        if os.path.exists(logistic_ds_path) and not force:
            print(f"  [离线数据集-Logistic] 与当前传感器数不兼容，自动重建")
        print(f"  [离线数据集-Logistic] 开始构建...")
        build_offline_dataset(
            cfg,
            data_norm=data_norm,
            labels=data_info["labels"],
            ttes=ttes,
            cls_labels=cls_labels,
            n_sensors=n_sensors,
            train_range=train_range,
            train_experiments=train_experiments,
            predictor=logistic_predictor.cpu(),  # 用 CPU 构建
            sensor_types=data_info.get("sensor_types", []),
            save_name="offline_dataset_logistic.pt",
            artifact_signature=artifact_signature,
        )
        logistic_predictor.to(device)

    # Step 2c: 用 logistic 离线数据集训练 IQL
    logistic_iql_path = os.path.join(cfg.save_dir, "policy_iql_logistic.pt")
    if os.path.exists(logistic_iql_path) and not force and artifact_matches(logistic_iql_path, n_sensors, artifact_signature):
        print(f"  [IQL-Logistic] 已存在，跳过训练")
    else:
        if os.path.exists(logistic_iql_path) and not force:
            print(f"  [IQL-Logistic] 旧 checkpoint 与当前传感器数不兼容，自动重训")
        print(f"  [IQL-Logistic] 开始训练...")
        iql_epochs = cfg.iql_epochs
        if getattr(args, 'quick', False):
            iql_epochs = 20
        train_iql(
            cfg,
            epochs=iql_epochs,
            dataset_path=logistic_ds_path,
            save_path=logistic_iql_path,
            artifact_signature=artifact_signature,
        )

    # Step 2d: 加载 LogisticPredictor + IQL-logistic 策略 → 评估
    logistic_predictor.to(device)
    logistic_predictor.eval()

    iql_log_ckpt = torch.load(logistic_iql_path, map_location=device, weights_only=False)
    pi_iql_log = PolicyNetwork(
        iql_log_ckpt.get("hidden_dim", hidden_dim),
        iql_log_ckpt.get("n_sensors", n_sensors),
        cfg.iql_hidden_dim
    ).to(device)
    pi_iql_log.load_state_dict(iql_log_ckpt["pi_net"])
    pi_iql_log.eval()

    policy_fn_log = _make_iql_policy_fn(pi_iql_log, device, belief_mode=policy_belief_mode)
    metrics_no_pred = _evaluate_policy_with_tuned_threshold(
        logistic_predictor, policy_fn_log, val_scope, eval_scope, cfg, device
    )
    ablation_results["w/o Predictor"] = metrics_no_pred
    print(
        f"  F1={metrics_no_pred['f1']:.4f}, AUC={metrics_no_pred['auc']:.4f}, "
        f"Thr={metrics_no_pred.get('decision_threshold', cfg.alarm_threshold):.3f}, "
        f"Lead={metrics_no_pred['lead_time']['avg_lead']:.1f}s"
    )

    # ================================================================
    # 配置 3: w/o Policy — LSTM + StaticTopM 固定选择
    # ================================================================
    print("\n--- 消融 3/4: w/o Policy (LSTM + StaticTopM) ---")
    if "LSTM+StaticTopM" in baseline_results:
        ablation_results["w/o Policy"] = baseline_results["LSTM+StaticTopM"]
        print(f"  [复用基线] F1={ablation_results['w/o Policy']['f1']:.4f}, "
              f"AUC={ablation_results['w/o Policy']['auc']:.4f}")
    else:
        print("  [独立评估]")
        static_policy_fn = _make_static_policy_fn(
            train_data_for_static, M
        )
        metrics_no_policy = _evaluate_policy_with_tuned_threshold(
            predictor, static_policy_fn, val_scope, eval_scope, cfg, device
        )
        ablation_results["w/o Policy"] = metrics_no_policy
        print(
            f"  F1={metrics_no_policy['f1']:.4f}, AUC={metrics_no_policy['auc']:.4f}, "
            f"Thr={metrics_no_policy.get('decision_threshold', cfg.alarm_threshold):.3f}"
        )

    # ================================================================
    # 配置 4: w/o RL — LSTM + BC 策略
    # ================================================================
    print("\n--- 消融 4/4: w/o RL (LSTM + BC) ---")
    if "LSTM+BC" in baseline_results:
        ablation_results["w/o RL"] = baseline_results["LSTM+BC"]
        print(f"  [复用基线] F1={ablation_results['w/o RL']['f1']:.4f}, "
              f"AUC={ablation_results['w/o RL']['auc']:.4f}")
    else:
        print("  [独立评估]")
        bc_path = os.path.join(cfg.save_dir, "policy_bc.pt")
        if not artifact_matches(bc_path, n_sensors, artifact_signature):
            raise RuntimeError(
                f"BC policy checkpoint not compatible with current data signature and n_sensors={n_sensors}: {bc_path}. "
                "Please run run_experiments.py to retrain BC first."
            )
        bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
        pi_bc = PolicyNetwork(
            bc_ckpt.get("hidden_dim", hidden_dim),
            bc_ckpt.get("n_sensors", n_sensors),
            cfg.iql_hidden_dim
        ).to(device)
        pi_bc.load_state_dict(bc_ckpt["pi_net"])
        pi_bc.eval()
        policy_fn_bc = _make_iql_policy_fn(pi_bc, device, belief_mode=policy_belief_mode)
        metrics_no_rl = _evaluate_policy_with_tuned_threshold(
            predictor, policy_fn_bc, val_scope, eval_scope, cfg, device
        )
        ablation_results["w/o RL"] = metrics_no_rl
        print(
            f"  F1={metrics_no_rl['f1']:.4f}, AUC={metrics_no_rl['auc']:.4f}, "
            f"Thr={metrics_no_rl.get('decision_threshold', cfg.alarm_threshold):.3f}"
        )

    # ================================================================
    # 消融实验汇总
    # ================================================================
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"消融实验完成! 耗时: {elapsed/60:.1f} 分钟")
    print(f"{'='*60}")

    print(f"\n{'消融配置':<20} {'F1':>8} {'AUC':>8} {'Recall':>8} {'Lead':>8} {'Switch':>8}")
    print(f"{'Setting':<20} {'F1':>8} {'AUC':>8} {'Thr':>8} {'Recall':>8} {'Lead':>8} {'Switch':>8}")
    print("-" * 82)
    for name, m in ablation_results.items():
        em = m.get("event_miss_rate", {})
        print(f"{name:<20} {m['f1']:>8.4f} {m['auc']:>8.4f} "
              f"{m.get('decision_threshold', cfg.alarm_threshold):>8.3f} "
              f"{m['recall']:>8.4f} {m['lead_time']['avg_lead']:>8.1f} "
              f"{m['switching_freq']:>8.4f}")

    return ablation_results


# ======================================================================
# 独立运行入口
# ======================================================================

def run_ablation_standalone(args):
    """独立运行消融实验（自动加载数据和预训练模型）。"""
    cfg = Config()

    if args.quick:
        cfg.epochs = 10
        cfg.iql_epochs = 20
        cfg.bc_epochs = 10

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[设备] {device}")

    # 加载数据
    data_info = prepare_data_time_split(cfg)
    n_sensors = data_info["n_sensors"]
    cfg.n_sensors = n_sensors
    cfg.n_active = min(cfg.n_active, n_sensors)

    # 加载预训练 LSTM 预测器
    ckpt_path = os.path.join(cfg.save_dir, "best_model.pt")
    assert os.path.exists(ckpt_path), f"请先训练 LSTM 预测器: {ckpt_path}"
    predictor = MaskAwareRiskPredictor(
        n_sensors=n_sensors,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lstm_dropout=cfg.lstm_dropout,
        head_dropout=cfg.head_dropout,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    predictor.load_state_dict(ckpt["model_state_dict"])

    # 尝试加载基线结果（如果存在）
    baseline_results = {}
    json_path = os.path.join(cfg.save_dir, "experiment_results.json")
    if os.path.exists(json_path):
        print("[消融] 尝试从基线结果复用...")
        # 基线结果是精简版 (无 probs/masks)，需要独立评估
        # 传空字典，让消融实验全部独立运行
        pass

    # 运行消融
    ablation_results = run_ablation(
        cfg, predictor, data_info, baseline_results, device, args
    )

    # 可视化
    plot_ablation_bar_chart(ablation_results, cfg.fig_dir)

    # 保存结果
    results_path = os.path.join(os.path.dirname(__file__), "ablation_results.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Ablation Studies 消融实验结果\n")
        f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        if getattr(cfg, "tune_alarm_threshold", False):
            f.write(
                f"Decision threshold: val->{cfg.threshold_metric} "
                f"in [{cfg.threshold_search_min:.2f}, {cfg.threshold_search_max:.2f}] "
                f"x {cfg.threshold_search_steps}\n\n"
            )
        else:
            f.write(f"Decision threshold: fixed={cfg.alarm_threshold:.3f}\n\n")

        f.write(f"{'Config':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
                f"{'F1':>6} {'AUC':>6} {'Thr':>6} {'Miss%':>6} {'Lead':>6} {'Switch':>6}\n")
        f.write("-" * 80 + "\n")

        for name, m in ablation_results.items():
            em = m.get("event_miss_rate", {})
            miss_pct = em.get("miss_rate", 0) * 100
            f.write(f"{name:<20} "
                    f"{m['accuracy']:>6.3f} "
                    f"{m['precision']:>6.3f} "
                    f"{m['recall']:>6.3f} "
                    f"{m['f1']:>6.3f} "
                    f"{m['auc']:>6.3f} "
                    f"{m.get('decision_threshold', cfg.alarm_threshold):>6.3f} "
                    f"{miss_pct:>6.1f} "
                    f"{m['lead_time']['avg_lead']:>6.1f} "
                    f"{m['switching_freq']:>6.3f}\n")

        f.write("-" * 80 + "\n")

    print(f"[结果] 消融实验结果已保存至: {results_path}")

    return ablation_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Studies")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式")
    parser.add_argument("--force_retrain", action="store_true",
                        help="强制重新训练")
    args = parser.parse_args()
    run_ablation_standalone(args)
