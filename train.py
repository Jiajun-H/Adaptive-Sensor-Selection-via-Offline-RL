"""
train.py — Mask-Aware 风险预测器 完整训练流程

流程:
  1. 加载配置 → 准备数据 (dataset.py)
  2. 构建模型 & 损失函数 (model.py)
  3. 训练循环: 随机 Mask 增强 + 多任务损失 + 早停
  4. 评估 & 保存最优模型 (evaluate.py)

用法:
  python train.py                     # 使用默认配置
  python train.py --epochs 50         # 覆盖参数
  python train.py --device cpu        # CPU 训练
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

# 本地模块
sys.path.insert(0, os.path.dirname(__file__))
from config import Config
from dataset import prepare_data_time_split
from model import MaskAwareRiskPredictor, MultiTaskLoss
from evaluate import evaluate, print_metrics, detailed_report, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train Mask-Aware Risk Predictor")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lambda_tte", type=float, default=None)
    parser.add_argument("--n_active", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    return parser.parse_args()


def apply_args(cfg: Config, args):
    """将命令行参数覆盖到配置中。"""
    for key in ["epochs", "batch_size", "lr", "lambda_tte",
                "n_active", "window_size", "device", "hidden_dim"]:
        val = getattr(args, key, None)
        if val is not None:
            setattr(cfg, key, val)


# ======================================================================
# 训练主循环
# ======================================================================

def train_one_epoch(model, criterion, optimizer, dataloader, device, grad_clip=1.0):
    """一个 epoch 的训练，返回平均损失。"""
    model.train()
    total_loss, total_cls, total_tte = 0.0, 0.0, 0.0
    n_batches = 0

    for x, label, tte_true in dataloader:
        x = x.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        tte_true = tte_true.to(device, non_blocking=True)

        # 前向
        logit, tte_hat = model(x)
        loss, cls_l, tte_l = criterion(logit, tte_hat, label, tte_true)

        # 反向
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        total_cls += cls_l.item()
        total_tte += tte_l.item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "cls_loss": total_cls / max(n_batches, 1),
        "tte_loss": total_tte / max(n_batches, 1),
    }


def main():
    args = parse_args()
    cfg = Config()
    apply_args(cfg, args)

    # ---------- 设备 ----------
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，回退到 CPU")
        cfg.device = "cpu"
    device = torch.device(cfg.device)
    print(f"[Train] 设备: {device}")

    # ---------- 数据 ----------
    data_info = prepare_data_time_split(cfg)
    train_loader = data_info["train_loader"]
    val_loader = data_info["val_loader"]
    n_sensors = data_info["n_sensors"]
    preprocessor = data_info.get("preprocessor")
    preprocessing = data_info.get("preprocessing", {})
    cfg.n_sensors = n_sensors
    cfg.n_active = min(cfg.n_active, n_sensors)
    print(f"[Train] n_sensors={n_sensors}, n_active={cfg.n_active}")

    # ---------- 模型 ----------
    model = MaskAwareRiskPredictor(
        n_sensors=n_sensors,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lstm_dropout=cfg.lstm_dropout,
        head_dropout=cfg.head_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[Train] 模型参数量: {n_params:,}")

    # ---------- 损失 & 优化器 ----------
    # 计算正样本权重（处理类别不平衡）
    n_pos, n_neg = 0, 0
    for i in range(len(train_loader.dataset)):
        _, l, _ = train_loader.dataset[i]
        if l > 0.5:
            n_pos += 1
        else:
            n_neg += 1
    pos_weight = n_neg / max(n_pos, 1)
    pos_weight = min(
        max(pos_weight, float(getattr(cfg, "min_pos_weight", 0.1))),
        float(getattr(cfg, "max_pos_weight", 30.0)),
    )
    print(f"[Train] 正样本={n_pos}, 负样本={n_neg}, pos_weight={pos_weight:.2f}")

    criterion = MultiTaskLoss(
        lambda_tte=cfg.lambda_tte,
        huber_delta=cfg.huber_delta,
        pos_weight=pos_weight,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.lr * 0.01)

    # ---------- 训练记录 ----------
    history = {
        "train_loss": [], "val_loss": [],
        "val_f1": [], "val_auc": [], "val_tte_mae": [],
    }
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    best_model_path = os.path.join(cfg.save_dir, "best_model.pt")

    # ---------- 训练循环 ----------
    print(f"\n{'='*60}")
    print(f"开始训练: {cfg.epochs} epochs, batch_size={cfg.batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        # 训练
        train_metrics = train_one_epoch(
            model, criterion, optimizer, train_loader, device, cfg.grad_clip
        )

        # 验证
        val_metrics = evaluate(model, criterion, val_loader, device)

        # 学习率调度
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        # 记录
        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics.get("f1", 0))
        history["val_auc"].append(val_metrics.get("auc", 0))
        history["val_tte_mae"].append(val_metrics.get("tte_mae", 0))

        elapsed = time.time() - t0

        # 打印
        print(f"Epoch {epoch:3d}/{cfg.epochs} ({elapsed:.1f}s)  lr={lr_now:.2e}")
        print(f"  [Train] loss={train_metrics['loss']:.4f}  "
              f"cls={train_metrics['cls_loss']:.4f}  "
              f"tte={train_metrics['tte_loss']:.4f}")
        print_metrics(val_metrics, prefix="Val")

        # 早停检查
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            # 保存最优模型
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": cfg,
                "n_sensors": n_sensors,
                "preprocessor": preprocessor,
                "preprocessing": preprocessing,
            }, best_model_path)
            print(f"  ★ 最优模型已保存 (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"\n[早停] {cfg.patience} 轮无改善，停止训练。")
                break

        print()

    # ---------- 训练完成 ----------
    print(f"\n{'='*60}")
    print(f"训练完成! 最优 epoch={best_epoch}, val_loss={best_val_loss:.4f}")
    print(f"{'='*60}")

    # 加载最优模型进行最终评估
    ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print("\n===== 最终评估（最优模型）=====")
    final_metrics = evaluate(model, criterion, val_loader, device)
    print_metrics(final_metrics, prefix="Final")
    detailed_report(model, val_loader, device, threshold=cfg.alarm_threshold)

    # 绘制训练曲线
    plot_path = os.path.join(cfg.save_dir, "training_curves.png")
    plot_training_curves(history, save_path=plot_path)

    # 保存最终训练历史
    import json
    hist_path = os.path.join(cfg.save_dir, "history.json")
    # 将 numpy float32 转换为 Python 原生 float
    history_serializable = {
        k: [float(v) for v in vals] for k, vals in history.items()
    }
    with open(hist_path, "w") as f:
        json.dump(history_serializable, f, indent=2)
    print(f"[Save] 训练历史已保存至 {hist_path}")

    return model, final_metrics


if __name__ == "__main__":
    main()
