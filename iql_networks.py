"""
iql_networks.py — IQL 三网络定义 (Q / V / Policy)

Q 网络:  Q(h, a) → 标量             输入: concat(h, a)
V 网络:  V(h)    → 标量             输入: h
Policy:  π(h)    → logits (N,)      推理时 Top-M 选传感器
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# Expectile Loss
# ======================================================================

def expectile_loss(diff: torch.Tensor, tau: float = 0.8) -> torch.Tensor:
    """
    非对称平方损失 (expectile regression):
        L_τ(u) = |τ - 𝟙(u < 0)| · u²

    Args:
        diff : (B,)  Q(h,a) - V(h)
        tau  : expectile 参数, ∈ (0,1), 越大则 V 越逼近 Q 的上分位
    Returns:
        标量损失
    """
    weight = torch.where(diff < 0, 1 - tau, tau)     # (B,)
    return (weight * diff.pow(2)).mean()


# ======================================================================
# Q 网络
# ======================================================================

class QNetwork(nn.Module):
    """
    Q(h, a) — 输入 concat(h, a), 输出标量 Q 值。

    结构: Linear → LayerNorm → ReLU → Linear → ReLU → Linear(1)
    """

    def __init__(self, hidden_dim: int, n_sensors: int, mlp_dim: int = 256):
        super().__init__()
        input_dim = hidden_dim + n_sensors  # h ⊕ a
        self.net = nn.Sequential(
            nn.Linear(input_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : (B, hidden_dim)
            a : (B, n_sensors)  — 0/1 mask 向量
        Returns:
            q : (B,)
        """
        x = torch.cat([h, a], dim=-1)
        return self.net(x).squeeze(-1)


# ======================================================================
# V 网络
# ======================================================================

class VNetwork(nn.Module):
    """
    V(h) — 输入信念状态 h, 输出标量 V 值。

    结构: Linear → LayerNorm → ReLU → Linear → ReLU → Linear(1)
    """

    def __init__(self, hidden_dim: int, mlp_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : (B, hidden_dim)
        Returns:
            v : (B,)
        """
        return self.net(h).squeeze(-1)


# ======================================================================
# 策略网络
# ======================================================================

class PolicyNetwork(nn.Module):
    """
    π(h) — 输入信念状态 h, 输出每个传感器的 logits。
    训练时: 配合优势加权 BCE 损失学习。
    推理时: 取 Top-M 个 logit 最大的传感器。

    结构: Linear → LayerNorm → ReLU → Linear → ReLU → Linear(n_sensors)
    """

    def __init__(self, hidden_dim: int, n_sensors: int, mlp_dim: int = 256):
        super().__init__()
        self.n_sensors = n_sensors
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, n_sensors),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h : (B, hidden_dim)
        Returns:
            logits : (B, n_sensors)
        """
        return self.net(h)

    @torch.no_grad()
    def select_top_m(self, h: torch.Tensor, m: int) -> torch.Tensor:
        """
        推理接口: 给定信念状态 h, 输出 Top-M mask 向量。

        Args:
            h : (B, hidden_dim)
            m : 要选择的传感器数量
        Returns:
            mask : (B, n_sensors)  — 0/1 向量, 恰好 m 个 1
        """
        self.eval()
        logits = self.forward(h)                          # (B, n_sensors)
        _, topk_idx = logits.topk(m, dim=-1)              # (B, m)
        mask = torch.zeros_like(logits)
        mask.scatter_(1, topk_idx, 1.0)
        return mask


# ======================================================================
# 快速 sanity check
# ======================================================================
if __name__ == "__main__":
    H, N, B = 128, 179, 8

    q_net = QNetwork(H, N)
    v_net = VNetwork(H)
    pi_net = PolicyNetwork(H, N)

    h = torch.randn(B, H)
    a = torch.zeros(B, N)
    # 随机选 20 个传感器
    for i in range(B):
        idx = torch.randperm(N)[:20]
        a[i, idx] = 1.0

    q_val = q_net(h, a)
    v_val = v_net(h)
    logits = pi_net(h)
    mask = pi_net.select_top_m(h, m=20)

    print(f"Q shape: {q_val.shape}, val: {q_val[:4]}")
    print(f"V shape: {v_val.shape}, val: {v_val[:4]}")
    print(f"π logits shape: {logits.shape}")
    print(f"mask shape: {mask.shape}, sum per row: {mask.sum(dim=1)}")

    # expectile loss
    diff = q_val.detach() - v_val
    e_loss = expectile_loss(diff, tau=0.8)
    print(f"Expectile loss: {e_loss.item():.4f}")

    print(f"Q params: {sum(p.numel() for p in q_net.parameters()):,}")
    print(f"V params: {sum(p.numel() for p in v_net.parameters()):,}")
    print(f"π params: {sum(p.numel() for p in pi_net.parameters()):,}")
    print("✓ IQL networks sanity check passed.")
