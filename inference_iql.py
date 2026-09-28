"""
inference_iql.py — 端到端推理 Demo

功能:
  1. 加载预训练的 RiskPredictor + IQL PolicyNetwork
  2. 在关键时间点（火灾前/中/后）对比三种传感器选择方式:
     - Policy 选择 (Top-M)
     - 随机选择 (Random-M)
     - 全传感器 (Full)
  3. 输出预测结果对比表
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import Config
from model import MaskAwareRiskPredictor
from iql_networks import PolicyNetwork
from dataset import load_and_clean, normalize, generate_labels


def load_models(cfg: Config):
    """加载预训练的 RiskPredictor 和 IQL Policy Network。"""
    device = torch.device("cpu")  # 推理在 CPU 即可

    # 1. 加载数据以获取 n_sensors
    data_raw, cats, names = load_and_clean(
        cfg.data_path,
        auto_reverse=getattr(cfg, "auto_reverse_experiments", True),
    )
    n_sensors = data_raw.shape[1]
    labels, ttes = generate_labels(
        data_raw, cats, names,
        temp_thresh=cfg.temp_threshold,
        trans_thresh=cfg.trans_threshold,
        heat_thresh=cfg.heat_threshold,
        co_thresh=cfg.co_threshold,
        min_modalities=cfg.hazard_min_modalities,
        min_consecutive=cfg.hazard_min_consecutive,
    )
    data_norm, _, _ = normalize(data_raw)

    # 基于 Δ 的二分类标签
    T = data_raw.shape[0]
    cls_labels = np.zeros(T, dtype=np.float32)
    for t in range(T):
        if ttes[t] <= cfg.delta:
            cls_labels[t] = 1.0

    # 2. 加载风险预测器
    predictor = MaskAwareRiskPredictor(
        n_sensors=n_sensors,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        lstm_dropout=cfg.lstm_dropout,
        head_dropout=cfg.head_dropout,
    )
    ckpt = os.path.join(cfg.save_dir, "best_model.pt")
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt_data:
        ckpt_data = ckpt_data["model_state_dict"]
    predictor.load_state_dict(ckpt_data)
    predictor.eval()
    print(f"[Inference] RiskPredictor 已加载: {ckpt}")

    # 3. 加载 IQL Policy
    policy_ckpt = os.path.join(cfg.save_dir, "policy_best.pt")
    policy_data = torch.load(policy_ckpt, map_location="cpu", weights_only=False)

    pi_net = PolicyNetwork(
        hidden_dim=cfg.hidden_dim,
        n_sensors=n_sensors,
        mlp_dim=cfg.iql_hidden_dim,
    )
    pi_net.load_state_dict(policy_data["pi_net"])
    pi_net.eval()
    print(f"[Inference] PolicyNetwork 已加载: {policy_ckpt}")
    print(f"  训练 epoch: {policy_data.get('epoch', '?')}, "
          f"π loss: {policy_data.get('pi_loss', '?'):.4f}")

    return predictor, pi_net, data_norm, cls_labels, ttes, n_sensors


def build_input(data_norm, t, W, mask, n_sensors):
    """构建模型输入 (1, W, 2N)"""
    obs = data_norm[t - W + 1: t + 1].copy()           # (W, N)
    obs_masked = obs * mask[np.newaxis, :]              # (W, N)
    mask_exp = np.tile(mask, (W, 1))                    # (W, N)
    x = np.concatenate([obs_masked, mask_exp], axis=-1) # (W, 2N)
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)


def run_inference(cfg: Config):
    """主推理函数。"""
    predictor, pi_net, data_norm, cls_labels, ttes, n_sensors = load_models(cfg)

    W = cfg.window_size
    M = cfg.n_active if cfg.n_active > 0 else 20
    T = data_norm.shape[0]

    # 找到火灾开始时间
    fire_onset = int(ttes[0]) if ttes[0] < T else T
    print(f"\n[Inference] 时间步总数: {T}, 火灾开始: step {fire_onset}")
    print(f"  窗口大小: {W}, 活跃传感器: {M}/{n_sensors}")

    # 关键时间点
    key_times = []
    # 火灾前
    for offset in [100, 50, 20, 10]:
        t = fire_onset - offset
        if t >= W:
            key_times.append((t, f"fire-{offset}"))
    # 火灾时
    if fire_onset >= W and fire_onset < T:
        key_times.append((fire_onset, "onset"))
    # 火灾后
    for offset in [10, 30, 50]:
        t = fire_onset + offset
        if t < T:
            key_times.append((t, f"fire+{offset}"))

    if not key_times:
        # 如果没有明确的火灾时间点，使用均匀采样
        step = max(1, (T - W) // 8)
        for i, t in enumerate(range(W, T, step)):
            key_times.append((t, f"t={t}"))
            if len(key_times) >= 8:
                break

    # ---- 对比三种方法 ----
    print(f"\n{'='*90}")
    print(f"  {'时间步':>8s} {'标签':>6s} {'真实TTE':>8s} | "
          f"{'Policy p̂':>10s} {'Random p̂':>10s} {'Full p̂':>10s} | "
          f"{'Policy TTE':>10s} {'Full TTE':>10s}")
    print(f"{'='*90}")

    np.random.seed(42)

    for t, label_str in key_times:
        true_label = cls_labels[t]
        true_tte = ttes[t]

        # --- (a) Policy 选择 ---
        # 先用全传感器获取信念状态，再让 Policy 选传感器
        full_mask = np.ones(n_sensors, dtype=np.float32)
        x_full = build_input(data_norm, t, W, full_mask, n_sensors)
        with torch.no_grad():
            h = predictor.encode_belief(x_full)       # (1, H)
            policy_mask = pi_net.select_top_m(h, m=M) # (1, N)
        policy_mask_np = policy_mask.squeeze(0).numpy()

        x_policy = build_input(data_norm, t, W, policy_mask_np, n_sensors)
        with torch.no_grad():
            p_policy, tte_policy = predictor.predict(x_policy)

        # --- (b) 随机选择 ---
        rand_mask = np.zeros(n_sensors, dtype=np.float32)
        chosen = np.random.choice(n_sensors, M, replace=False)
        rand_mask[chosen] = 1.0

        x_rand = build_input(data_norm, t, W, rand_mask, n_sensors)
        with torch.no_grad():
            p_rand, tte_rand = predictor.predict(x_rand)

        # --- (c) 全传感器 ---
        with torch.no_grad():
            p_full, tte_full = predictor.predict(x_full)

        print(
            f"  {t:>6d} ({label_str:>8s})  {true_label:>4.0f}  {true_tte:>7.1f} | "
            f" {p_policy.item():>9.4f}  {p_rand.item():>9.4f}  {p_full.item():>9.4f} | "
            f" {tte_policy.item():>9.2f}  {tte_full.item():>9.2f}"
        )

    print(f"{'='*90}")

    # ---- 传感器选择可视化 ----
    print(f"\n[Inference] Policy 选择的传感器 (最后一个时间点 t={key_times[-1][0]}):")
    selected = np.where(policy_mask_np > 0.5)[0]
    print(f"  选中传感器索引: {selected.tolist()}")
    print(f"  选中数量: {len(selected)}")

    print("\n✓ 推理 Demo 完成。")


# ======================================================================
if __name__ == "__main__":
    cfg = Config()
    run_inference(cfg)
