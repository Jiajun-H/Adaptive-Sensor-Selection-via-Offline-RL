"""
run_experiments.py — 基线对比实验主驱动 (Baseline Comparison)

流程:
  1. 时间切分数据 (70/15/15)
  2. 训练风险预测器 LSTM (100 epochs)
  3. 构建离线数据集 (3 种行为策略)
  4. 训练 IQL 策略 (200 epochs)
  5. 训练 CQL 策略 (200 epochs)
  6. 训练 BC 策略 (100 epochs)
  7. 训练 SoftAttention 门控 (100 epochs)
  8. 在全时间线上评估 7 种基线方法 (M=cfg.n_active)
  9. Pareto 扫描 M=[5,10,15,20,30]
  10. 可视化: Pareto曲线, 时间线预警, 传感器热力图
  11. 输出结果到 results.txt + JSON

用法:
  python run_experiments.py                  # 标准实验
  python run_experiments.py --quick          # 快速测试 (少量 epochs)
  python run_experiments.py --force_retrain  # 强制重新训练所有模型
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import time
import json
import argparse
import random

# 先导入 torch，再导入 numpy，避免 DLL 冲突
import torch
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from artifact_utils import artifact_matches, attach_artifact_signature, build_artifact_signature
from config import Config
from dataset import prepare_data_time_split
from model import MaskAwareRiskPredictor, MultiTaskLoss
from evaluate import (
    evaluate, print_metrics, evaluate_with_policy, evaluate_with_policy_multi_experiments,
    compute_lead_time, compute_switching_frequency,
    compute_event_miss_rate,
    collect_predictor_outputs,
    select_best_threshold,
)
from build_offline_dataset import (
    build_offline_dataset, RandomPolicy, StaticTopMPolicy
)
from train_iql import train_iql, train_bc, train_cql
from iql_networks import PolicyNetwork
from baselines import (
    LogisticRegressionPredictor, SoftAttentionGating,
    train_logistic_predictor, train_soft_attention,
)
from visualize import plot_pareto_curves, plot_timeline_warning, plot_sensor_heatmap


def _compute_pos_weight_from_loader(train_loader, cfg):
    n_pos, n_neg = 0, 0
    for i in range(len(train_loader.dataset)):
        _, label, _ = train_loader.dataset[i]
        if float(label) > 0.5:
            n_pos += 1
        else:
            n_neg += 1

    raw = n_neg / max(n_pos, 1)
    pos_weight = min(
        max(raw, float(getattr(cfg, "min_pos_weight", 0.1))),
        float(getattr(cfg, "max_pos_weight", 30.0)),
    )
    return int(n_pos), int(n_neg), float(pos_weight)


def _configure_torch_runtime(cfg, device):
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    if bool(getattr(cfg, "allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ======================================================================
# 训练风险预测器
# ======================================================================

def train_risk_predictor(cfg, train_loader, val_loader, n_sensors, device,
                         force_retrain=False, artifact_signature=None, preprocessor=None):
    """
    训练或加载风险预测器。
    """
    ckpt_path = os.path.join(cfg.save_dir, "best_model.pt")

    if os.path.exists(ckpt_path) and not force_retrain:
        if artifact_matches(ckpt_path, n_sensors, artifact_signature):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            ckpt_select_metric = str(ckpt.get("predictor_select_metric", "loss"))
            ckpt_balanced = bool(ckpt.get("balanced_train_sampling", False))
            ckpt_max_pos_weight = float(ckpt.get("max_pos_weight", 10.0))
            expected_select_metric = str(getattr(cfg, "predictor_select_metric", "f1_tuned"))
            expected_balanced = bool(getattr(cfg, "balanced_train_sampling", False))
            expected_max_pos_weight = float(getattr(cfg, "max_pos_weight", 30.0))

            recipe_matches = (
                ckpt_select_metric == expected_select_metric
                and ckpt_balanced == expected_balanced
                and abs(ckpt_max_pos_weight - expected_max_pos_weight) < 1e-8
            )
            if recipe_matches:
                print(f"[风险预测器] 已存在，跳过训练: {ckpt_path}")
                model = MaskAwareRiskPredictor(
                    n_sensors=n_sensors,
                    hidden_dim=cfg.hidden_dim,
                    num_layers=cfg.num_layers,
                    lstm_dropout=cfg.lstm_dropout,
                    head_dropout=cfg.head_dropout,
                ).to(device)
                model.load_state_dict(ckpt["model_state_dict"])
                return model
            print("[风险预测器] 检测到训练配方变更，自动重训 predictor")
        else:
            print(f"[风险预测器] 发现旧 checkpoint 与当前传感器数不兼容，自动重训: {ckpt_path}")

    print(f"\n{'='*60}")
    print(f"[风险预测器] 开始训练: {cfg.epochs} epochs")
    print(f"{'='*60}")

    model = MaskAwareRiskPredictor(
        n_sensors=n_sensors,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lstm_dropout=cfg.lstm_dropout,
        head_dropout=cfg.head_dropout,
    ).to(device)

    # 计算正样本权重
    n_pos, n_neg, pos_weight = _compute_pos_weight_from_loader(train_loader, cfg)
    print(f"[风险预测器] train windows pos/neg={n_pos}/{n_neg}, pos_weight={pos_weight:.2f}")

    criterion = MultiTaskLoss(
        lambda_tte=cfg.lambda_tte,
        huber_delta=cfg.huber_delta,
        pos_weight=pos_weight,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    best_val_loss = float("inf")
    best_selection_score = float("-inf")
    best_selection_threshold = float(getattr(cfg, "alarm_threshold", 0.5))
    best_epoch = 0
    selection_metric = str(getattr(cfg, "predictor_select_metric", "f1_tuned"))
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        # 训练
        model.train()
        train_loss = 0.0
        n_batches = 0
        for x, label, tte_true in train_loader:
            x = x.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            tte_true = tte_true.to(device, non_blocking=True)
            logit, tte_hat = model(x)
            loss, _, _ = criterion(logit, tte_hat, label, tte_true)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        # 验证
        val_metrics = evaluate(model, criterion, val_loader, device)
        val_outputs = collect_predictor_outputs(model, val_loader, device)
        tuned_info = select_best_threshold(
            val_outputs["probs"],
            val_outputs["labels"],
            metric=getattr(cfg, "threshold_metric", "f1"),
            thresholds=_build_threshold_grid(cfg),
            default_threshold=float(getattr(cfg, "alarm_threshold", 0.5)),
        )
        val_tuned_f1 = float(tuned_info["metrics"].get("f1", val_metrics.get("f1", 0.0)))
        val_tuned_thr = float(tuned_info["threshold"])
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{cfg.epochs} | "
                  f"Train={train_loss:.4f} Val={val_metrics['loss']:.4f} "
                  f"F1@0.5={val_metrics.get('f1', 0):.3f} "
                  f"F1*tuned={val_tuned_f1:.3f} Thr={val_tuned_thr:.3f}")

        # 早停
        if selection_metric == "loss":
            selection_score = -float(val_metrics["loss"])
        elif selection_metric == "auc":
            selection_score = float(val_metrics.get("auc", 0.0))
        elif selection_metric == "f1":
            selection_score = float(val_metrics.get("f1", 0.0))
        else:
            selection_score = val_tuned_f1

        improved = False
        if selection_score > best_selection_score + 1e-8:
            improved = True
        elif abs(selection_score - best_selection_score) <= 1e-8 and val_metrics["loss"] < best_val_loss - 1e-8:
            improved = True

        if improved:
            best_val_loss = float(val_metrics["loss"])
            best_selection_score = float(selection_score)
            best_selection_threshold = float(val_tuned_thr)
            best_epoch = epoch
            patience_counter = 0
            torch.save(attach_artifact_signature({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "val_tuned_threshold": best_selection_threshold,
                "val_tuned_f1": val_tuned_f1,
                "predictor_select_metric": selection_metric,
                "balanced_train_sampling": bool(getattr(cfg, "balanced_train_sampling", False)),
                "min_pos_weight": float(getattr(cfg, "min_pos_weight", 0.1)),
                "max_pos_weight": float(getattr(cfg, "max_pos_weight", 30.0)),
                "n_sensors": n_sensors,
                "preprocessor": preprocessor,
            }, artifact_signature), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  早停 @ epoch {epoch}")
                break

    # 重新加载最佳
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(
        f"[风险预测器] 训练完成，最佳 epoch={best_epoch} "
        f"{selection_metric}={best_selection_score:.4f} "
        f"val_loss={best_val_loss:.4f} "
        f"thr={best_selection_threshold:.3f}"
    )
    return model


# ======================================================================
# 策略函数定义 (用于 evaluate_with_policy)
# ======================================================================

def make_iql_policy_fn(pi_net, hidden_dim, device, belief_mode="full"):
    """创建 IQL 策略函数。"""
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


def make_bc_policy_fn(pi_net, hidden_dim, device, belief_mode="full"):
    """创建 BC 策略函数（与 IQL 相同，只是加载不同权重）。"""
    return make_iql_policy_fn(pi_net, hidden_dim, device, belief_mode=belief_mode)


def make_cql_policy_fn(pi_net, hidden_dim, device, belief_mode="full"):
    """创建 CQL 策略函数 (复用 IQL 的 policy_fn 结构)。"""
    return make_iql_policy_fn(pi_net, hidden_dim, device, belief_mode=belief_mode)


def make_soft_attn_policy_fn(gating_net, hidden_dim, device, belief_mode="full"):
    """创建 SoftAttention 策略函数。"""
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
            mask_out = gating_net.select_top_m(h, m=m)
        selected = mask_out.cpu().numpy().squeeze(0)
        state["mask"] = selected.copy()
        return selected
    policy_fn.reset = lambda: state.update({"mask": None})  # type: ignore[attr-defined]
    return policy_fn


def make_static_policy_fn(data_norm, m):
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


def make_random_policy_fn():
    """创建随机策略函数。"""
    def policy_fn(predictor, obs_window, n_sensors, m, dev):
        mask = np.zeros(n_sensors, dtype=np.float32)
        chosen = np.random.choice(n_sensors, m, replace=False)
        mask[chosen] = 1.0
        return mask
    def batch_select(obs_windows, n_sensors, m, dev):
        m = max(1, min(int(m), int(n_sensors)))
        n = len(obs_windows)
        masks = np.zeros((n, n_sensors), dtype=np.float32)
        scores = np.random.random((n, n_sensors))
        chosen = np.argpartition(scores, kth=m - 1, axis=1)[:, :m]
        rows = np.arange(n)[:, np.newaxis]
        masks[rows, chosen] = 1.0
        return masks
    policy_fn.batch_select = batch_select  # type: ignore[attr-defined]
    return policy_fn


def make_all_sensors_policy_fn():
    """创建全传感器基线策略函数。"""
    def policy_fn(predictor, obs_window, n_sensors, m, dev):
        return np.ones(n_sensors, dtype=np.float32)
    def batch_select(obs_windows, n_sensors, m, dev):
        return np.ones((len(obs_windows), n_sensors), dtype=np.float32)
    policy_fn.batch_select = batch_select  # type: ignore[attr-defined]
    return policy_fn


# ======================================================================
# 辅助: 用不同 M 值评估一个策略
# ======================================================================

def evaluate_with_m(predictor, policy_fn_factory, eval_data, eval_cls,
                    eval_ttes, fire_onset, cfg, device, m_value):
    """
    用指定的 M 值在全时间线上评估策略。
    policy_fn_factory: callable(m) → policy_fn
    """
    policy_fn = policy_fn_factory(m_value)
    original_m = cfg.n_active
    cfg.n_active = m_value
    metrics = evaluate_with_policy(
        predictor, policy_fn,
        eval_data, eval_cls, eval_ttes,
        fire_onset, cfg, str(device)
    )
    cfg.n_active = original_m
    return metrics


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

    # legacy single-file mode: use test segment only
    split_range = data_info[f"{split}_range"]
    t0 = max(0, int(split_range[0]) - cfg.window_size + 1)
    t1 = int(split_range[1])
    eval_data = data_info["data_norm"][t0:t1]
    eval_cls = data_info["cls_labels"][t0:t1]
    eval_ttes = data_info["ttes"][t0:t1]
    eval_fire_onset = data_info.get("eval_fire_onset", 0)
    return {
        "mode": "single",
        "experiments": [],
        "eval_data": eval_data,
        "eval_cls": eval_cls,
        "eval_ttes": eval_ttes,
        "eval_fire_onset": eval_fire_onset,
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


def _first_positive_idx(arr: np.ndarray) -> int:
    idx = np.where(np.asarray(arr) > 0.5)[0]
    return int(idx[0]) if len(idx) > 0 else int(len(arr))


def _clip_onset(onset: int, length: int) -> int:
    return max(0, min(int(onset), max(0, int(length))))


def _build_timeline_diagnostics(experiments):
    diagnostics = []
    for idx, exp in enumerate(experiments or []):
        cls_labels = np.asarray(exp.get("cls_labels", np.array([], dtype=np.float32)))
        exp_len = int(exp.get("length", len(cls_labels)))
        warning_onset = _clip_onset(_first_positive_idx(cls_labels), exp_len)
        fire_onset = _clip_onset(exp.get("fire_onset", exp_len), exp_len)
        diagnostics.append(
            {
                "exp_id": str(exp.get("exp_id", f"exp_{idx}")),
                "length": exp_len,
                "warning_onset": warning_onset,
                "fire_onset": fire_onset,
                "warning_width": max(0, exp_len - warning_onset),
                "danger_width": max(0, exp_len - fire_onset),
            }
        )
    return diagnostics


# ======================================================================
# 主实验流程
# ======================================================================

def run_experiments(args, return_artifacts=False):
    """执行基线对比实验流程。

    Args:
        args            : 命令行参数 (quick, force_retrain)
        return_artifacts: 若为 True，返回字典包含训练好的模型和数据对象，
                          供 run_ablation.py / run_full_experiments.py 复用。
    """
    cfg = Config()
    if hasattr(args, "split_seed") and args.split_seed is not None:
        cfg.split_seed = int(args.split_seed)
    if hasattr(args, "experiment_split_mode") and args.experiment_split_mode is not None:
        cfg.experiment_split_mode = str(args.experiment_split_mode)
    if hasattr(args, "experiment_kfold_n_splits") and args.experiment_kfold_n_splits is not None:
        cfg.experiment_kfold_n_splits = int(args.experiment_kfold_n_splits)
    if hasattr(args, "experiment_kfold_index") and args.experiment_kfold_index is not None:
        cfg.experiment_kfold_index = int(args.experiment_kfold_index)
    if hasattr(args, "save_dir") and args.save_dir:
        cfg.save_dir = os.path.abspath(str(args.save_dir))
        os.makedirs(cfg.save_dir, exist_ok=True)
    if hasattr(args, "fig_dir") and args.fig_dir:
        cfg.fig_dir = os.path.abspath(str(args.fig_dir))
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
        if hasattr(args, attr) and getattr(args, attr) is not None:
            setattr(cfg, attr, int(getattr(args, attr)))
    for attr in ["rl_preload_to_device", "soft_attn_preload_to_device"]:
        if hasattr(args, attr) and getattr(args, attr) is not None:
            setattr(cfg, attr, bool(getattr(args, attr)))

    # 快速模式：减少 epochs
    if getattr(args, "quick", False):
        cfg.epochs = 10
        cfg.iql_epochs = 20
        cfg.cql_epochs = 20
        cfg.bc_epochs = 10
        cfg.soft_attn_epochs = 10
        cfg.iql_n_mask_samples = 5
        cfg.pareto_m_values = [10, 20]  # 简化 Pareto 扫描
        print("[模式] 快速测试模式")

    # 烟雾测试模式：最小配置，只验证流程可跑通
    if getattr(args, "smoke", False):
        cfg.epochs = 2
        cfg.iql_epochs = 3
        cfg.cql_epochs = 3
        cfg.bc_epochs = 2
        cfg.soft_attn_epochs = 2
        cfg.iql_n_mask_samples = 1
        cfg.iql_step_interval = 30
        cfg.pareto_m_values = [10]
        cfg.experiment_max_files = max(3, int(getattr(args, "smoke_max_files", 12)))

        cfg.save_dir = os.path.join(cfg.save_dir, "smoke")
        cfg.fig_dir = os.path.join(cfg.fig_dir, "smoke")
        os.makedirs(cfg.save_dir, exist_ok=True)
        os.makedirs(cfg.fig_dir, exist_ok=True)

        print(
            f"[模式] smoke (max_files={cfg.experiment_max_files}, "
            f"epochs LSTM/IQL/CQL/BC/Soft={cfg.epochs}/{cfg.iql_epochs}/"
            f"{cfg.cql_epochs}/{cfg.bc_epochs}/{cfg.soft_attn_epochs})"
        )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    _configure_torch_runtime(cfg, device)
    _set_global_seed(int(getattr(cfg, "split_seed", 42)))
    print(f"[设备] {device}")
    print(
        f"[Runtime] batch_size={cfg.batch_size}, "
        f"iql_batch_size={cfg.iql_batch_size}, cql_batch_size={cfg.cql_batch_size}, "
        f"rl_preload={getattr(cfg, 'rl_preload_to_device', True)}, "
        f"amp={getattr(cfg, 'use_amp', False)}"
    )

    results = {}
    start_time = time.time()

    # ================================================================
    # Step 1: 加载时间切分数据
    # ================================================================
    print("\n" + "="*60)
    print("Step 1: 加载数据并划分 train/val/test")
    print("="*60)

    data_info = prepare_data_time_split(cfg)
    train_loader = data_info["train_loader"]
    val_loader = data_info["val_loader"]
    test_loader = data_info["test_loader"]
    n_sensors = data_info["n_sensors"]
    data_norm = data_info["data_norm"]
    cls_labels = data_info["cls_labels"]
    ttes = data_info["ttes"]
    fire_onset = data_info["fire_onset"]
    train_range = data_info["train_range"]
    test_range = data_info["test_range"]
    split_mode = data_info.get("split_mode", "time_single")
    train_experiments = data_info.get("train_experiments", [])
    val_experiments = data_info.get("val_experiments", [])
    test_experiments = data_info.get("test_experiments", [])
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

    cfg.n_sensors = n_sensors
    cfg.n_active = min(cfg.n_active, n_sensors)
    M = cfg.n_active

    print(f"  传感器数: {n_sensors}, 活跃数 M={M}")
    print(f"  split_mode: {split_mode}")
    preprocessing = data_info.get("preprocessing", {})
    if preprocessing:
        print(f"  preprocessing: {preprocessing.get('fit_scope', 'unknown')}")
    if getattr(cfg, "tune_alarm_threshold", False):
        print(
            f"  threshold tuning: val->{cfg.threshold_metric} "
            f"on [{cfg.threshold_search_min:.2f}, {cfg.threshold_search_max:.2f}] "
            f"x {cfg.threshold_search_steps}"
        )
    print(f"  policy_belief_mode: {getattr(cfg, 'policy_belief_mode', 'full')}")
    if split_mode.startswith("experiment_"):
        print(
            f"  experiments(train/val/test): "
            f"{len(train_experiments)}/{len(val_experiments)}/{len(test_experiments)}"
        )
    else:
        print(f"  测试范围: [{test_range[0]}, {test_range[1]})")

    # ================================================================
    # Step 2: 训练风险预测器
    # ================================================================
    print("\n" + "="*60)
    print("Step 2: 训练风险预测器")
    print("="*60)

    artifact_signature = build_artifact_signature(cfg, data_info)

    predictor = train_risk_predictor(
        cfg, train_loader, val_loader, n_sensors, device,
        force_retrain=args.force_retrain,
        artifact_signature=artifact_signature,
        preprocessor=data_info.get("preprocessor"),
    )

    # ================================================================
    # Step 3: 构建离线数据集
    # ================================================================
    print("\n" + "="*60)
    print("Step 3: 构建离线数据集")
    print("="*60)

    offline_path = os.path.join(cfg.save_dir, "offline_dataset.pt")
    reuse_offline = (
        os.path.exists(offline_path)
        and not args.force_retrain
        and artifact_matches(offline_path, n_sensors, artifact_signature)
    )
    if reuse_offline:
        print(f"[离线数据] 已存在，跳过构建: {offline_path}")
        # 读取策略统计
        _ds = torch.load(offline_path, map_location="cpu", weights_only=False)
        policy_stats = _ds.get("policy_stats", {})
        if policy_stats:
            for pname, ps in policy_stats.items():
                print(f"  [{pname}] {ps['n_transitions']} 条转移, "
                      f"avg_r={ps['avg_reward']:.4f}±{ps['std_reward']:.4f}")
        del _ds
    else:
        if os.path.exists(offline_path) and not args.force_retrain:
            print(f"[离线数据] 发现旧数据与当前传感器数不兼容，自动重建: {offline_path}")
        ds_result = build_offline_dataset(
            cfg,
            data_norm=data_norm,
            labels=data_info["labels"],
            ttes=ttes,
            cls_labels=cls_labels,
            n_sensors=n_sensors,
            train_range=train_range,
            train_experiments=train_experiments,
            sensor_types=data_info.get("sensor_types", []),
            artifact_signature=artifact_signature,
        )
        policy_stats = ds_result.get("policy_stats", {})

    # ================================================================
    # Step 4: 训练 IQL 策略
    # ================================================================
    print("\n" + "="*60)
    print("Step 4: 训练 IQL 策略")
    print("="*60)

    iql_path = os.path.join(cfg.save_dir, "policy_best.pt")
    if os.path.exists(iql_path) and not args.force_retrain and artifact_matches(iql_path, n_sensors, artifact_signature):
        print(f"[IQL] 已存在，跳过训练: {iql_path}")
    else:
        if os.path.exists(iql_path) and not args.force_retrain:
            print(f"[IQL] 发现旧 checkpoint 与当前传感器数不兼容，自动重训: {iql_path}")
        train_iql(cfg, epochs=cfg.iql_epochs, artifact_signature=artifact_signature)

    # ================================================================
    # Step 5: 训练 CQL 策略
    # ================================================================
    print("\n" + "="*60)
    print("Step 5: 训练 CQL 策略")
    print("="*60)

    cql_path = os.path.join(cfg.save_dir, "policy_cql.pt")
    if os.path.exists(cql_path) and not args.force_retrain and artifact_matches(cql_path, n_sensors, artifact_signature):
        print(f"[CQL] 已存在，跳过训练: {cql_path}")
    else:
        if os.path.exists(cql_path) and not args.force_retrain:
            print(f"[CQL] 发现旧 checkpoint 与当前传感器数不兼容，自动重训: {cql_path}")
        train_cql(cfg, epochs=cfg.cql_epochs, artifact_signature=artifact_signature)

    # ================================================================
    # Step 6: 训练 BC 策略
    # ================================================================
    print("\n" + "="*60)
    print("Step 6: 训练 BC 策略 (行为克隆)")
    print("="*60)

    bc_path = os.path.join(cfg.save_dir, "policy_bc.pt")
    if os.path.exists(bc_path) and not args.force_retrain and artifact_matches(bc_path, n_sensors, artifact_signature):
        print(f"[BC] 已存在，跳过训练: {bc_path}")
    else:
        if os.path.exists(bc_path) and not args.force_retrain:
            print(f"[BC] 发现旧 checkpoint 与当前传感器数不兼容，自动重训: {bc_path}")
        train_bc(cfg, epochs=cfg.bc_epochs, artifact_signature=artifact_signature)

    # ================================================================
    # Step 7: 训练 SoftAttention 门控
    # ================================================================
    print("\n" + "="*60)
    print("Step 7: 训练 SoftAttention 门控")
    print("="*60)

    soft_attn_path = os.path.join(cfg.save_dir, "soft_attention.pt")
    if os.path.exists(soft_attn_path) and not args.force_retrain and artifact_matches(soft_attn_path, n_sensors, artifact_signature):
        print(f"[SoftAttention] 已存在，跳过训练: {soft_attn_path}")
    else:
        if os.path.exists(soft_attn_path) and not args.force_retrain:
            print(f"[SoftAttention] 发现旧 checkpoint 与当前传感器数不兼容，自动重训: {soft_attn_path}")
        train_soft_attention(
            cfg, predictor, data_norm, cls_labels, ttes,
            train_range, n_sensors, device,
            train_experiments=train_experiments,
            artifact_signature=artifact_signature,
        )

    # ================================================================
    # Step 8: 全时间线评估 (7 种方法)
    # ================================================================
    print("\n" + "="*60)
    print("Step 8: 测试集评估 (7 种方法)")
    print("="*60)

    W = cfg.window_size
    if eval_scope["mode"] == "multi":
        eval_concat_labels = np.concatenate(
            [exp["cls_labels"][W - 1:] for exp in test_experiments if len(exp["cls_labels"]) >= W],
            axis=0,
        ) if len(test_experiments) > 0 else np.array([], dtype=np.float32)
        print(f"  评估范围: 测试实验拼接 ({len(test_experiments)} experiments)")
    else:
        eval_concat_labels = eval_scope["eval_cls"][W - 1:] if len(eval_scope["eval_cls"]) >= W else np.array([], dtype=np.float32)
        print(f"  评估范围: 测试时间段 [0, {len(eval_scope['eval_data'])})")

    n_neg = int((eval_concat_labels == 0).sum())
    n_pos = int((eval_concat_labels == 1).sum())
    print(f"  安全样本: {n_neg}, 危险样本: {n_pos}")

    hidden_dim = cfg.hidden_dim
    policy_belief_mode = str(getattr(cfg, "policy_belief_mode", "full"))

    # 加载 IQL Policy
    iql_ckpt = torch.load(iql_path, map_location=device, weights_only=False)
    pi_iql = PolicyNetwork(
        iql_ckpt.get("hidden_dim", hidden_dim),
        iql_ckpt.get("n_sensors", n_sensors),
        cfg.iql_hidden_dim
    ).to(device)
    pi_iql.load_state_dict(iql_ckpt["pi_net"])
    pi_iql.eval()

    # 加载 CQL Policy
    cql_ckpt = torch.load(cql_path, map_location=device, weights_only=False)
    pi_cql = PolicyNetwork(
        cql_ckpt.get("hidden_dim", hidden_dim),
        cql_ckpt.get("n_sensors", n_sensors),
        cfg.iql_hidden_dim
    ).to(device)
    pi_cql.load_state_dict(cql_ckpt["pi_net"])
    pi_cql.eval()

    # 加载 BC Policy
    bc_ckpt = torch.load(bc_path, map_location=device, weights_only=False)
    pi_bc = PolicyNetwork(
        bc_ckpt.get("hidden_dim", hidden_dim),
        bc_ckpt.get("n_sensors", n_sensors),
        cfg.iql_hidden_dim
    ).to(device)
    pi_bc.load_state_dict(bc_ckpt["pi_net"])
    pi_bc.eval()

    # 加载 SoftAttention
    soft_ckpt = torch.load(soft_attn_path, map_location=device, weights_only=False)
    gating_net = SoftAttentionGating(
        soft_ckpt.get("hidden_dim", hidden_dim),
        soft_ckpt.get("n_sensors", n_sensors),
        cfg.iql_hidden_dim
    ).to(device)
    gating_net.load_state_dict(soft_ckpt["gating_state_dict"])
    gating_net.eval()

    # 7 种方法配置
    method_configs = {
        "LSTM+IQL": make_iql_policy_fn(pi_iql, hidden_dim, device, belief_mode=policy_belief_mode),
        "LSTM+CQL": make_cql_policy_fn(pi_cql, hidden_dim, device, belief_mode=policy_belief_mode),
        "LSTM+BC": make_bc_policy_fn(pi_bc, hidden_dim, device, belief_mode=policy_belief_mode),
        "LSTM+SoftAttn": make_soft_attn_policy_fn(gating_net, hidden_dim, device, belief_mode=policy_belief_mode),
        "LSTM+StaticTopM": make_static_policy_fn(train_data_for_static, M),
        "LSTM+Random": make_random_policy_fn(),
        "Static-All": make_all_sensors_policy_fn(),
    }

    print(f"\n评估方法: {list(method_configs.keys())}\n")

    results = {}
    method_probs = {}  # 保存概率用于可视化

    for method_name, policy_fn in method_configs.items():
        print(f"\n--- 评估: {method_name} ---")
        metrics = _evaluate_policy_with_tuned_threshold(
            predictor,
            policy_fn,
            val_scope,
            eval_scope,
            cfg,
            device,
        )
        results[method_name] = metrics
        method_probs[method_name] = metrics.get("probs", np.array([]))

        em = metrics.get("event_miss_rate", {})
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Threshold: {metrics.get('decision_threshold', cfg.alarm_threshold):.3f}")
        print(f"  Event Miss Rate: {em.get('miss_rate', 0)*100:.1f}% "
              f"({em.get('n_missed', 0)}/{em.get('n_events', 0)} events)")
        print(f"  Lead Time: avg={metrics['lead_time']['avg_lead']:.2f}s")
        print(f"  Switch Freq (Jaccard): {metrics['switching_freq']:.4f}")

    # ================================================================
    # Step 9: Pareto 扫描 (M = [5, 10, 15, 20, 30])
    # ================================================================
    print("\n" + "="*60)
    print("Step 9: Pareto 扫描")
    print("="*60)

    pareto_methods = {
        "IQL": lambda m: make_iql_policy_fn(pi_iql, hidden_dim, device, belief_mode=policy_belief_mode),
        "CQL": lambda m: make_cql_policy_fn(pi_cql, hidden_dim, device, belief_mode=policy_belief_mode),
        "BC": lambda m: make_bc_policy_fn(pi_bc, hidden_dim, device, belief_mode=policy_belief_mode),
        "SoftAttn": lambda m: make_soft_attn_policy_fn(gating_net, hidden_dim, device, belief_mode=policy_belief_mode),
        "StaticTopM": lambda m: make_static_policy_fn(train_data_for_static, m),
        "Random": lambda m: make_random_policy_fn(),
    }

    pareto_results = {name: {} for name in pareto_methods}

    for m_val in cfg.pareto_m_values:
        print(f"\n--- M = {m_val} ---")
        for method_name, policy_factory in pareto_methods.items():
            policy_fn = policy_factory(m_val)
            original_m = cfg.n_active
            cfg.n_active = m_val
            metrics = _evaluate_policy_with_tuned_threshold(
                predictor,
                policy_fn,
                val_scope,
                eval_scope,
                cfg,
                device,
            )
            cfg.n_active = original_m
            pareto_results[method_name][m_val] = metrics
            print(f"  {method_name}: AUC={metrics['auc']:.4f} F1={metrics['f1']:.4f} "
                  f"Thr={metrics.get('decision_threshold', cfg.alarm_threshold):.3f} "
                  f"Lead={metrics['lead_time']['avg_lead']:.1f}s")

    # ================================================================
    # Step 10: 可视化
    # ================================================================
    print("\n" + "="*60)
    print("Step 10: 可视化")
    print("="*60)

    # 1. Pareto 曲线
    plot_pareto_curves(pareto_results, cfg.fig_dir)

    # 2. 时间线预警曲线
    timeline_labels = data_info["cls_labels"]
    timeline_segments = None
    timeline_diagnostics = []
    heatmap_fire_onset = data_info["fire_onset"]
    if split_mode.startswith("experiment_") and len(test_experiments) > 0:
        timeline_labels = np.concatenate([exp["cls_labels"] for exp in test_experiments], axis=0)
        timeline_fire_onset = 0
        timeline_diagnostics = _build_timeline_diagnostics(test_experiments)
        timeline_segments = [
            {
                "exp_id": diag["exp_id"],
                "length": diag["length"],
                "warning_onset": diag["warning_onset"],
                "fire_onset": diag["fire_onset"],
            }
            for diag in timeline_diagnostics
        ]
    else:
        timeline_labels = data_info["cls_labels"]
        timeline_fire_onset = data_info["fire_onset"]
        warning_onset = _first_positive_idx(timeline_labels)
        timeline_diagnostics = [
            {
                "exp_id": "eval_segment",
                "length": len(timeline_labels),
                "warning_onset": warning_onset,
                "fire_onset": _clip_onset(timeline_fire_onset, len(timeline_labels)),
                "warning_width": max(0, len(timeline_labels) - warning_onset),
                "danger_width": max(0, len(timeline_labels) - _clip_onset(timeline_fire_onset, len(timeline_labels))),
            }
        ]
    timeline_threshold = None
    threshold_values = [
        float(m.get("decision_threshold", cfg.alarm_threshold))
        for m in results.values()
        if isinstance(m, dict)
    ]
    if len(threshold_values) > 0 and max(threshold_values) - min(threshold_values) < 1e-6:
        timeline_threshold = threshold_values[0]
    if len(timeline_diagnostics) > 0:
        print("  Timeline diagnostics:")
        for diag in timeline_diagnostics:
            print(
                f"    [{diag['exp_id']}] len={diag['length']} "
                f"warning_onset={diag['warning_onset']} fire_onset={diag['fire_onset']} "
                f"warning_width={diag['warning_width']} danger_width={diag['danger_width']}"
            )
    if len(timeline_labels) > 0:
        plot_timeline_warning(
            method_probs,
            timeline_labels,
            timeline_fire_onset,
            cfg.window_size,
            timeline_threshold,
            cfg.fig_dir,
            experiment_infos=timeline_segments,
        )

    # 3. 传感器热力图 (IQL)
    iql_masks = results.get("LSTM+IQL", {}).get("masks", None)
    if iql_masks is not None and len(iql_masks) > 0:
        plot_sensor_heatmap(
            iql_masks, "LSTM+IQL",
            sensor_names=data_info.get("sensor_names", None),
            fire_onset=heatmap_fire_onset,
            window_size=1,
            save_dir=cfg.fig_dir
        )

    # ================================================================
    # Step 11: 输出结果
    # ================================================================
    print("\n" + "="*60)
    print("Step 11: 保存结果")
    print("="*60)

    elapsed = time.time() - start_time
    results_path = os.path.join(os.path.dirname(__file__), "results.txt")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Mask-Aware Risk Predictor + RL 策略实验结果 (Baseline Comparison)\n")
        f.write(f"日期: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总耗时: {elapsed/60:.1f} 分钟\n")
        f.write("=" * 80 + "\n\n")

        f.write("实验配置:\n")
        f.write(f"  传感器数: {n_sensors}\n")
        f.write(f"  默认活跃传感器数 M: {M}\n")
        f.write(f"  窗口大小 W: {cfg.window_size}\n")
        f.write(f"  预警阈值 Δ: {cfg.delta}s\n")
        if getattr(cfg, "tune_alarm_threshold", False):
            f.write(
                f"  decision_threshold: val->{cfg.threshold_metric} "
                f"in [{cfg.threshold_search_min:.2f}, {cfg.threshold_search_max:.2f}] "
                f"x {cfg.threshold_search_steps}\n"
            )
        else:
            f.write(f"  decision_threshold: fixed={cfg.alarm_threshold:.3f}\n")
        f.write(f"  policy_belief_mode: {getattr(cfg, 'policy_belief_mode', 'full')}\n")
        if preprocessing:
            f.write(
                f"  preprocessing: {preprocessing.get('imputation', 'unknown')}; "
                f"{preprocessing.get('normalization', 'unknown')}; "
                f"fit_scope={preprocessing.get('fit_scope', 'unknown')}\n"
            )
        f.write(f"  split_mode: {split_mode}\n")
        if split_mode.startswith("experiment_"):
            f.write(
                f"  experiment_split_mode: {getattr(cfg, 'experiment_split_mode', 'random')}, "
                f"split_seed={getattr(cfg, 'split_seed', 42)}, "
                f"kfold={getattr(cfg, 'experiment_kfold_index', 0)}/"
                f"{getattr(cfg, 'experiment_kfold_n_splits', 0)}\n"
            )
        if split_mode.startswith("experiment_"):
            f.write(
                f"  实验划分(train/val/test): "
                f"{len(train_experiments)}/{len(val_experiments)}/{len(test_experiments)}\n"
            )
            f.write(f"  评估范围: 测试实验拼接 (window labels={len(timeline_labels)})\n")
        else:
            f.write(
                f"  评估范围: 测试时间段 [0, {len(eval_scope['eval_data'])}) "
                f"(window labels={len(timeline_labels)})\n"
            )
        if split_mode.startswith("experiment_"):
            f.write("  timeline zones: segmented warning/danger per test experiment\n")
        else:
            f.write(f"  火灾起始步(评估域): {timeline_fire_onset}\n")
        f.write(f"  训练 epochs: LSTM={cfg.epochs}, IQL={cfg.iql_epochs}, "
                f"CQL={cfg.cql_epochs}, BC={cfg.bc_epochs}, SoftAttn={cfg.soft_attn_epochs}\n")
        f.write(
            f"  predictor_select_metric: {getattr(cfg, 'predictor_select_metric', 'f1_tuned')}, "
            f"balanced_train_sampling={bool(getattr(cfg, 'balanced_train_sampling', False))}, "
            f"pos_weight_clip=[{float(getattr(cfg, 'min_pos_weight', 0.1)):.1f}, "
            f"{float(getattr(cfg, 'max_pos_weight', 30.0)):.1f}]\n"
        )
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

        # 离线数据集统计
        if policy_stats:
            f.write("离线数据集构建 (行为策略统计):\n")
            for pname, ps in policy_stats.items():
                f.write(f"  {pname:<15} {ps['n_transitions']:>6} 条转移  "
                        f"avg_reward={ps['avg_reward']:.4f}\n")
            f.write("\n")

        # 基线对比结果表格
        f.write("Baseline Comparison (M={}):\n".format(M))
        f.write("-" * 80 + "\n")
        f.write(f"{'Method':<20} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
                f"{'F1':>6} {'AUC':>6} {'Thr':>6} {'Miss%':>6} {'Lead':>6} {'Switch':>6}\n")
        f.write("-" * 80 + "\n")

        for method_name, metrics in results.items():
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

        # Pareto 结果表格
        f.write("Pareto 扫描结果:\n")
        f.write("-" * 80 + "\n")
        header = f"{'M':<5}"
        for method in pareto_methods.keys():
            header += f" {method:>10}"
        f.write(header + " (F1)\n")
        f.write("-" * 80 + "\n")

        for m_val in cfg.pareto_m_values:
            row = f"{m_val:<5}"
            for method in pareto_methods.keys():
                f1_val = pareto_results[method][m_val]["f1"]
                row += f" {f1_val:>10.4f}"
            f.write(row + "\n")
        f.write("-" * 80 + "\n\n")

        # Lead Time 详情
        f.write("Lead Time 详情 (M=20):\n")
        for method_name, metrics in results.items():
            lt = metrics['lead_time']
            f.write(f"  {method_name}: avg={lt['avg_lead']:.1f}s, "
                    f"median={lt['median_lead']:.1f}s, >=30s: {lt['lead_ge_30s_ratio']*100:.0f}%\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("实验完成\n")

    print(f"[结果] 已保存至: {results_path}")

    # JSON 结果
    json_path = os.path.join(cfg.save_dir, "experiment_results.json")
    results_json = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": elapsed / 60,
        "config": {
            "n_sensors": n_sensors,
            "n_active": M,
            "window_size": cfg.window_size,
            "delta": cfg.delta,
            "alarm_threshold": float(cfg.alarm_threshold),
            "tune_alarm_threshold": bool(getattr(cfg, "tune_alarm_threshold", False)),
            "threshold_metric": str(getattr(cfg, "threshold_metric", "f1")),
            "threshold_search_min": float(getattr(cfg, "threshold_search_min", cfg.alarm_threshold)),
            "threshold_search_max": float(getattr(cfg, "threshold_search_max", cfg.alarm_threshold)),
            "threshold_search_steps": int(getattr(cfg, "threshold_search_steps", 1)),
            "predictor_select_metric": str(getattr(cfg, "predictor_select_metric", "f1_tuned")),
            "preprocessing": preprocessing,
            "policy_belief_mode": str(getattr(cfg, "policy_belief_mode", "full")),
            "balanced_train_sampling": bool(getattr(cfg, "balanced_train_sampling", False)),
            "min_pos_weight": float(getattr(cfg, "min_pos_weight", 0.1)),
            "max_pos_weight": float(getattr(cfg, "max_pos_weight", 30.0)),
            "pareto_m_values": cfg.pareto_m_values,
            "split_mode": split_mode,
            "experiment_split_mode": str(getattr(cfg, "experiment_split_mode", "random")),
            "split_seed": int(getattr(cfg, "split_seed", 42)),
            "experiment_kfold_n_splits": int(getattr(cfg, "experiment_kfold_n_splits", 0) or 0),
            "experiment_kfold_index": int(getattr(cfg, "experiment_kfold_index", 0) or 0),
            "n_train_experiments": len(train_experiments),
            "n_val_experiments": len(val_experiments),
            "n_test_experiments": len(test_experiments),
        },
        "results": {},
        "pareto": {},
        "timeline_diagnostics": timeline_diagnostics,
    }
    for name, m in results.items():
        em = m.get("event_miss_rate", {})
        results_json["results"][name] = {
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
    for method_name, m_results in pareto_results.items():
        results_json["pareto"][method_name] = {
            str(m_val): {
                "f1": float(metrics["f1"]),
                "auc": float(metrics["auc"]),
                "decision_threshold": float(metrics.get("decision_threshold", cfg.alarm_threshold)),
                "val_threshold_score": float(metrics.get("val_threshold_score", 0.0)),
                "avg_lead": float(metrics["lead_time"]["avg_lead"]),
            }
            for m_val, metrics in m_results.items()
        }

    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"[结果] JSON 已保存至: {json_path}")

    print(f"\n{'='*60}")
    print(f"基线对比实验完成! 总耗时: {elapsed/60:.1f} 分钟")
    print(f"图表已保存至: {cfg.fig_dir}")
    print(f"{'='*60}")

    # 返回结果
    if return_artifacts:
        return {
            "baseline_results": results,
            "pareto_results": pareto_results,
            "method_probs": method_probs,
            "data_info": data_info,
            "predictor": predictor,
            "cfg": cfg,
            "device": device,
            "n_sensors": n_sensors,
            "args": args,
            "elapsed": elapsed,
            "policy_stats": policy_stats,
            "timeline_diagnostics": timeline_diagnostics,
        }
    return results


# ======================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Experiments")
    parser.add_argument("--quick", action="store_true",
                        help="快速测试模式 (少量 epochs)")
    parser.add_argument("--smoke", action="store_true",
                        help="超轻量 smoke 模式（最小数据+最少训练）")
    parser.add_argument("--smoke_max_files", type=int, default=12,
                        help="smoke 模式最多使用的实验文件数 (>=3)")
    parser.add_argument("--force_retrain", action="store_true",
                        help="强制重新训练所有模型")
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
    args = parser.parse_args()

    run_experiments(args)
